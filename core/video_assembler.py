"""Video assembler module - stitches scenes into final video with audio."""

import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from models.project import Project, ProjectStatus
from models.scene import Scene

logger = logging.getLogger(__name__)


class VideoAssembler:
    """Assembles individual scene videos into a final music video."""

    def __init__(
        self,
        ffmpeg_path: str = "ffmpeg",
        ffprobe_path: str = "ffprobe",
    ):
        """
        Initialize the assembler.

        Args:
            ffmpeg_path: Path to ffmpeg executable
            ffprobe_path: Path to ffprobe executable
        """
        self.ffmpeg_path = ffmpeg_path
        self.ffprobe_path = ffprobe_path

    def assemble(
        self,
        project: Project,
        output_path: Path,
        include_audio: bool = True,
        transition_duration: float = 0.5,
        output_resolution: str = "1920x1080",
    ) -> Path:
        """
        Assemble scenes into final video.

        Args:
            project: Project with generated scenes
            output_path: Where to save the final video
            include_audio: Whether to include the original audio
            transition_duration: Duration of crossfade transitions (0 for hard cuts)
            output_resolution: Output resolution (WxH)

        Returns:
            Path to the assembled video
        """
        logger.info(f"Assembling {len(project.scenes)} scenes into final video")
        project.update_status(ProjectStatus.RENDERING)

        # Get scenes with videos, sorted by number
        scenes_with_video = [
            s for s in sorted(project.scenes, key=lambda x: x.scene_number)
            if s.has_video and s.video_path
        ]

        if not scenes_with_video:
            raise ValueError("No scenes with video to assemble")

        try:
            # Create temporary directory for intermediate files
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)

                # Step 1: Trim each scene to exact duration
                trimmed_scenes = self._trim_scenes(scenes_with_video, temp_path)

                # Step 2: Concatenate scenes
                if transition_duration > 0:
                    concat_path = self._concat_with_transitions(
                        trimmed_scenes, temp_path, transition_duration
                    )
                else:
                    concat_path = self._concat_simple(trimmed_scenes, temp_path)

                # Step 3: Add audio if requested
                if include_audio and project.audio_path:
                    final_path = self._add_audio(
                        concat_path, project.audio_path, output_path
                    )
                else:
                    # Just copy to output
                    final_path = output_path
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    subprocess.run(
                        [self.ffmpeg_path, "-i", str(concat_path), "-c", "copy", str(output_path), "-y"],
                        check=True,
                        capture_output=True,
                    )

            project.final_video_path = final_path
            project.update_status(ProjectStatus.COMPLETE)
            logger.info(f"Assembly complete: {final_path}")

            return final_path

        except Exception as e:
            logger.error(f"Assembly failed: {e}")
            project.update_status(ProjectStatus.FAILED, str(e))
            raise

    def _trim_scenes(
        self,
        scenes: list[Scene],
        temp_dir: Path,
    ) -> list[Path]:
        """Trim each scene video to its exact duration."""
        trimmed_paths = []

        for scene in scenes:
            duration = scene.duration
            input_path = scene.video_path
            output_path = temp_dir / f"trimmed_{scene.scene_number:03d}.mp4"

            cmd = [
                self.ffmpeg_path,
                "-i", str(input_path),
                "-t", str(duration),
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "18",
                "-an",  # Remove audio (we'll add the song later)
                "-y",
                str(output_path),
            ]

            logger.debug(f"Trimming scene {scene.scene_number} to {duration}s")
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                logger.warning(f"Trim warning for scene {scene.scene_number}: {result.stderr}")

            trimmed_paths.append(output_path)

        return trimmed_paths

    def _concat_simple(
        self,
        video_paths: list[Path],
        temp_dir: Path,
    ) -> Path:
        """Concatenate videos with hard cuts (no transitions)."""
        # Create concat file
        concat_file = temp_dir / "concat.txt"
        with open(concat_file, "w") as f:
            for path in video_paths:
                f.write(f"file '{path}'\n")

        output_path = temp_dir / "concat_output.mp4"

        cmd = [
            self.ffmpeg_path,
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_file),
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "18",
            "-y",
            str(output_path),
        ]

        logger.debug("Concatenating scenes (simple)")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"Concat failed: {result.stderr}")

        return output_path

    def _concat_with_transitions(
        self,
        video_paths: list[Path],
        temp_dir: Path,
        transition_duration: float,
    ) -> Path:
        """Concatenate videos with crossfade transitions."""
        if len(video_paths) == 1:
            return video_paths[0]

        output_path = temp_dir / "concat_output.mp4"

        # Build complex filter for xfade transitions
        # This is complex but gives smooth crossfades
        filter_parts = []
        input_labels = []

        # Input labels
        for i in range(len(video_paths)):
            input_labels.append(f"[{i}:v]")

        # Build xfade chain
        # First pair
        current_offset = self._get_video_duration(video_paths[0]) - transition_duration
        filter_parts.append(
            f"[0:v][1:v]xfade=transition=fade:duration={transition_duration}:offset={current_offset}[v1]"
        )

        # Subsequent pairs
        for i in range(2, len(video_paths)):
            prev_label = f"v{i-1}"
            new_label = f"v{i}"
            current_offset += self._get_video_duration(video_paths[i-1]) - transition_duration
            filter_parts.append(
                f"[{prev_label}][{i}:v]xfade=transition=fade:duration={transition_duration}:offset={current_offset}[{new_label}]"
            )

        final_label = f"v{len(video_paths)-1}" if len(video_paths) > 2 else "v1"
        filter_complex = ";".join(filter_parts)

        # Build command
        cmd = [self.ffmpeg_path]
        for path in video_paths:
            cmd.extend(["-i", str(path)])

        cmd.extend([
            "-filter_complex", filter_complex,
            "-map", f"[{final_label}]",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "18",
            "-y",
            str(output_path),
        ])

        logger.debug("Concatenating scenes with transitions")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            logger.warning(f"Transition concat failed, falling back to simple: {result.stderr}")
            return self._concat_simple(video_paths, temp_dir)

        return output_path

    def _add_audio(
        self,
        video_path: Path,
        audio_path: Path,
        output_path: Path,
    ) -> Path:
        """Add audio track to video."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Get video duration
        video_duration = self._get_video_duration(video_path)

        cmd = [
            self.ffmpeg_path,
            "-i", str(video_path),
            "-i", str(audio_path),
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            "-t", str(video_duration),  # Trim audio to video length
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-shortest",
            "-y",
            str(output_path),
        ]

        logger.debug("Adding audio track")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"Add audio failed: {result.stderr}")

        return output_path

    def _get_video_duration(self, video_path: Path) -> float:
        """Get duration of a video file in seconds."""
        cmd = [
            self.ffprobe_path,
            "-v", "quiet",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            logger.warning(f"Could not get duration for {video_path}")
            return 5.0  # Default fallback

        try:
            return float(result.stdout.strip())
        except ValueError:
            return 5.0

    def create_preview(
        self,
        project: Project,
        output_path: Path,
        max_duration: float = 30.0,
    ) -> Path:
        """
        Create a short preview video from the first few scenes.

        Args:
            project: Project with scenes
            output_path: Where to save the preview
            max_duration: Maximum preview duration

        Returns:
            Path to preview video
        """
        scenes_with_video = [
            s for s in sorted(project.scenes, key=lambda x: x.scene_number)
            if s.has_video and s.video_path
        ]

        # Select scenes for preview
        preview_scenes = []
        total_duration = 0.0

        for scene in scenes_with_video:
            if total_duration >= max_duration:
                break
            preview_scenes.append(scene)
            total_duration += scene.duration

        if not preview_scenes:
            raise ValueError("No scenes available for preview")

        # Create a temporary project with just preview scenes
        preview_project = Project(
            id=f"{project.id}_preview",
            title=f"{project.title} (Preview)",
            audio_path=project.audio_path,
            scenes=preview_scenes,
        )

        return self.assemble(
            preview_project,
            output_path,
            include_audio=True,
            transition_duration=0.3,
        )

    def extract_thumbnail(
        self,
        video_path: Path,
        output_path: Path,
        timestamp: float = 1.0,
    ) -> Path:
        """Extract a thumbnail from a video."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            self.ffmpeg_path,
            "-i", str(video_path),
            "-ss", str(timestamp),
            "-vframes", "1",
            "-y",
            str(output_path),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"Thumbnail extraction failed: {result.stderr}")

        return output_path


def check_ffmpeg_available() -> bool:
    """Check if ffmpeg is available on the system."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True,
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def get_ffmpeg_version() -> Optional[str]:
    """Get the installed ffmpeg version."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            # First line contains version
            first_line = result.stdout.split("\n")[0]
            return first_line
        return None
    except FileNotFoundError:
        return None
