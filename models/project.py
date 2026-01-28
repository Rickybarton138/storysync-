"""Project data model."""

import json
from enum import Enum
from pathlib import Path
from typing import Optional
from datetime import datetime
import uuid

from pydantic import BaseModel, Field

from .director_brief import DirectorBrief
from .scene import Scene, SceneStatus


class ProjectStatus(str, Enum):
    """Status of a project in the pipeline."""

    DRAFT = "draft"  # Brief created, not processed
    ANALYSING = "analysing"  # Audio/story analysis in progress
    SCENES_READY = "scenes_ready"  # Scenes generated, awaiting video generation
    GENERATING = "generating"  # Video generation in progress
    REVIEW = "review"  # All scenes generated, awaiting user approval
    RENDERING = "rendering"  # Final video being assembled
    COMPLETE = "complete"  # Final video ready
    FAILED = "failed"  # Pipeline failed


class TimedWord(BaseModel):
    """A word with its timestamp."""

    word: str
    start: float
    end: float


class SongSection(BaseModel):
    """A section of the song (verse, chorus, etc.)."""

    section_type: str
    start_time: float
    end_time: float
    lyrics_preview: str = ""


class Project(BaseModel):
    """A complete music video project."""

    # Identity
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: Optional[str] = Field(default=None)
    title: str = Field(default="Untitled Project")

    # Status
    status: ProjectStatus = Field(default=ProjectStatus.DRAFT)
    error_message: Optional[str] = Field(default=None)

    # Audio
    audio_path: Optional[Path] = Field(default=None)
    audio_url: Optional[str] = Field(default=None)
    audio_duration: Optional[float] = Field(default=None)
    audio_tempo: Optional[float] = Field(default=None)

    # Lyrics
    lyrics_raw: str = Field(default="")
    lyrics_timed: list[TimedWord] = Field(default_factory=list)

    # Structure
    sections: list[SongSection] = Field(default_factory=list)

    # Director's brief
    director_brief: Optional[DirectorBrief] = Field(default=None)

    # Scenes
    scenes: list[Scene] = Field(default_factory=list)

    # Story analysis output
    narrative_summary: str = Field(default="")
    emotional_arc: str = Field(default="")

    # Final output
    final_video_path: Optional[Path] = Field(default=None)
    final_video_url: Optional[str] = Field(default=None)

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    @property
    def scene_count(self) -> int:
        return len(self.scenes)

    @property
    def scenes_pending(self) -> int:
        return sum(1 for s in self.scenes if s.status == SceneStatus.PENDING)

    @property
    def scenes_complete(self) -> int:
        return sum(1 for s in self.scenes if s.status in (SceneStatus.COMPLETE, SceneStatus.APPROVED))

    @property
    def scenes_failed(self) -> int:
        return sum(1 for s in self.scenes if s.status == SceneStatus.FAILED)

    @property
    def all_scenes_ready(self) -> bool:
        """Check if all scenes have been successfully generated."""
        return all(s.is_generated for s in self.scenes)

    @property
    def all_scenes_approved(self) -> bool:
        """Check if all scenes have been approved."""
        return all(s.status == SceneStatus.APPROVED for s in self.scenes)

    @property
    def progress_percent(self) -> float:
        """Overall generation progress as percentage."""
        if not self.scenes:
            return 0.0
        return (self.scenes_complete / len(self.scenes)) * 100

    def get_timed_lyrics_text(self) -> str:
        """Get lyrics formatted with timestamps for LLM."""
        if not self.lyrics_timed:
            return self.lyrics_raw

        lines = []
        current_line = []
        current_line_start = None

        for word in self.lyrics_timed:
            if current_line_start is None:
                current_line_start = word.start

            current_line.append(word.word)

            # Check for line breaks (pause > 0.5s or punctuation)
            if word.word.endswith((".", "!", "?", ",")) or (
                len(current_line) > 8
            ):
                line_text = " ".join(current_line)
                lines.append(f"[{current_line_start:.1f}s] {line_text}")
                current_line = []
                current_line_start = None

        # Don't forget remaining words
        if current_line:
            line_text = " ".join(current_line)
            lines.append(f"[{current_line_start:.1f}s] {line_text}")

        return "\n".join(lines)

    def get_sections_text(self) -> str:
        """Get sections formatted for LLM."""
        if not self.sections:
            return "No sections detected — please infer from lyrics structure."

        lines = []
        for section in self.sections:
            lines.append(
                f"- {section.section_type.upper()}: "
                f"{section.start_time:.1f}s - {section.end_time:.1f}s"
            )
        return "\n".join(lines)

    def get_scene_by_number(self, scene_number: int) -> Optional[Scene]:
        """Get a scene by its number."""
        for scene in self.scenes:
            if scene.scene_number == scene_number:
                return scene
        return None

    def add_scene(self, scene: Scene) -> None:
        """Add a scene to the project."""
        scene.project_id = self.id
        self.scenes.append(scene)
        self.scenes.sort(key=lambda s: s.scene_number)
        self.updated_at = datetime.utcnow()

    def update_status(self, status: ProjectStatus, error: Optional[str] = None) -> None:
        """Update project status."""
        self.status = status
        self.error_message = error
        self.updated_at = datetime.utcnow()

    def to_summary(self) -> str:
        """Generate a human-readable summary."""
        lines = [
            f"Project: {self.title}",
            f"ID: {self.id}",
            f"Status: {self.status.value}",
            f"Duration: {self.audio_duration:.1f}s" if self.audio_duration else "Duration: Unknown",
            f"Scenes: {self.scenes_complete}/{self.scene_count} complete",
            f"Progress: {self.progress_percent:.0f}%",
        ]
        if self.error_message:
            lines.append(f"Error: {self.error_message}")
        return "\n".join(lines)

    def save(self, path: Path) -> None:
        """Save project to JSON file."""
        # Convert Path objects to strings for JSON serialization
        data = self.model_dump(mode="json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

    @classmethod
    def load(cls, path: Path) -> "Project":
        """Load project from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Convert string paths back to Path objects
        if data.get("audio_path"):
            data["audio_path"] = Path(data["audio_path"])
        if data.get("final_video_path"):
            data["final_video_path"] = Path(data["final_video_path"])

        # Handle nested scenes
        for scene_data in data.get("scenes", []):
            if scene_data.get("video_path"):
                scene_data["video_path"] = Path(scene_data["video_path"])
            if scene_data.get("thumbnail_path"):
                scene_data["thumbnail_path"] = Path(scene_data["thumbnail_path"])

        return cls.model_validate(data)

    @classmethod
    def create(
        cls,
        title: str,
        audio_path: Path,
        director_brief: DirectorBrief,
    ) -> "Project":
        """Create a new project with required inputs."""
        return cls(
            title=title,
            audio_path=audio_path,
            director_brief=director_brief,
            status=ProjectStatus.DRAFT,
        )
