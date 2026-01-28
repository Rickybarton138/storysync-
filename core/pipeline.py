"""Pipeline orchestrator - coordinates the full video generation process."""

import asyncio
import logging
from pathlib import Path
from typing import Optional, Callable
from dataclasses import dataclass

from config.settings import settings
from models.project import Project, ProjectStatus
from models.director_brief import DirectorBrief
from models.scene import SceneStatus
from core.audio_processor import AudioProcessor
from core.story_analyzer import StoryAnalyzer
from core.prompt_generator import PromptGenerator
from core.video_generator import (
    VideoGeneratorFactory,
    generate_scenes_parallel,
    generate_scenes_sequential,
)
from core.video_assembler import VideoAssembler

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the pipeline."""

    video_model: str = "kling"
    parallel_generation: bool = True
    max_parallel: int = 3
    transition_duration: float = 0.5
    output_resolution: str = "1920x1080"
    skip_video_generation: bool = False
    use_quick_prompts: bool = False  # Use non-LLM prompt generation


@dataclass
class PipelineProgress:
    """Progress information for callbacks."""

    stage: str
    stage_progress: float  # 0.0 to 1.0
    overall_progress: float  # 0.0 to 1.0
    message: str


ProgressCallback = Callable[[PipelineProgress], None]


class Pipeline:
    """
    Orchestrates the full music video generation process.
    
    Usage:
        pipeline = Pipeline()
        project = await pipeline.run(
            audio_path=Path("song.mp3"),
            brief_path=Path("brief.json"),
            output_path=Path("output.mp4"),
        )
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ):
        """
        Initialize the pipeline.
        
        Args:
            config: Pipeline configuration
            progress_callback: Optional callback for progress updates
        """
        self.config = config or PipelineConfig()
        self.progress_callback = progress_callback

        # Initialize components
        self.audio_processor = AudioProcessor()
        self.story_analyzer = StoryAnalyzer()
        self.prompt_generator = PromptGenerator()
        self.video_generator = VideoGeneratorFactory.create(self.config.video_model)
        self.video_assembler = VideoAssembler()

    def _report_progress(
        self,
        stage: str,
        stage_progress: float,
        overall_progress: float,
        message: str,
    ):
        """Report progress to callback if set."""
        if self.progress_callback:
            self.progress_callback(PipelineProgress(
                stage=stage,
                stage_progress=stage_progress,
                overall_progress=overall_progress,
                message=message,
            ))

    async def run(
        self,
        audio_path: Path,
        brief_path: Path,
        output_path: Path,
        project_save_path: Optional[Path] = None,
    ) -> Project:
        """
        Run the full pipeline.
        
        Args:
            audio_path: Path to the audio file
            brief_path: Path to the director's brief JSON
            output_path: Path for the final video output
            project_save_path: Optional path to save/load project state
            
        Returns:
            Completed Project
        """
        # Load or create project
        if project_save_path and project_save_path.exists():
            logger.info(f"Loading existing project from {project_save_path}")
            project = Project.load(project_save_path)
            # Update paths in case they changed
            project.audio_path = audio_path
        else:
            logger.info("Creating new project")
            brief = DirectorBrief.from_json_file(brief_path)
            project = Project.create(
                title=brief.title,
                audio_path=audio_path,
                director_brief=brief,
            )

        try:
            # Stage 1: Audio Processing (20% of total)
            if not project.lyrics_raw:
                self._report_progress("audio", 0.0, 0.0, "Processing audio...")
                project = self.audio_processor.process_audio(project)
                self._report_progress("audio", 1.0, 0.2, "Audio processed")
                logger.info(f"Audio processed: {project.audio_duration:.1f}s")

            # Save checkpoint
            if project_save_path:
                project.save(project_save_path)

            # Stage 2: Story Analysis (20% of total)
            if not project.scenes:
                self._report_progress("story", 0.0, 0.2, "Analyzing story...")
                project = self.story_analyzer.analyze(project)
                self._report_progress("story", 1.0, 0.4, "Story analyzed")
                logger.info(f"Generated {len(project.scenes)} scenes")

            # Save checkpoint
            if project_save_path:
                project.save(project_save_path)

            # Stage 3: Prompt Generation (10% of total)
            scenes_need_prompts = [s for s in project.scenes if not s.positive_prompt]
            if scenes_need_prompts:
                self._report_progress("prompts", 0.0, 0.4, "Generating prompts...")
                
                for i, scene in enumerate(scenes_need_prompts):
                    self.prompt_generator.generate_prompt(scene, project.director_brief)
                    progress = (i + 1) / len(scenes_need_prompts)
                    self._report_progress("prompts", progress, 0.4 + progress * 0.1, f"Prompt {i+1}/{len(scenes_need_prompts)}")
                
                logger.info(f"Generated {len(scenes_need_prompts)} prompts")

            # Save checkpoint
            if project_save_path:
                project.save(project_save_path)

            # Stage 4: Video Generation (50% of total)
            if not self.config.skip_video_generation:
                scenes_to_generate = [s for s in project.scenes if s.needs_generation]
                
                if scenes_to_generate:
                    self._report_progress("video", 0.0, 0.5, "Generating videos...")
                    project.update_status(ProjectStatus.GENERATING)
                    
                    output_dir = output_path.parent / "scenes"
                    
                    if self.config.parallel_generation:
                        await generate_scenes_parallel(
                            scenes_to_generate,
                            self.video_generator,
                            output_dir,
                            self.config.max_parallel,
                        )
                    else:
                        def video_progress(current, total, status):
                            progress = current / total
                            self._report_progress(
                                "video",
                                progress,
                                0.5 + progress * 0.4,
                                f"Scene {current}/{total}: {status}",
                            )

                        await generate_scenes_sequential(
                            scenes_to_generate,
                            self.video_generator,
                            output_dir,
                            video_progress,
                        )

                    complete = sum(1 for s in project.scenes if s.is_generated)
                    logger.info(f"Video generation complete: {complete}/{len(project.scenes)}")

                # Save checkpoint
                if project_save_path:
                    project.save(project_save_path)

                # Stage 5: Assembly (10% of total)
                if project.all_scenes_ready:
                    self._report_progress("assembly", 0.0, 0.9, "Assembling video...")
                    
                    self.video_assembler.assemble(
                        project,
                        output_path,
                        include_audio=True,
                        transition_duration=self.config.transition_duration,
                        output_resolution=self.config.output_resolution,
                    )
                    
                    self._report_progress("assembly", 1.0, 1.0, "Complete!")
                    logger.info(f"Final video: {output_path}")
                else:
                    failed = [s for s in project.scenes if s.status == SceneStatus.FAILED]
                    logger.warning(f"{len(failed)} scenes failed to generate")
                    project.update_status(ProjectStatus.REVIEW)

            else:
                logger.info("Skipping video generation (config.skip_video_generation=True)")
                project.update_status(ProjectStatus.SCENES_READY)

            # Final save
            if project_save_path:
                project.save(project_save_path)

            return project

        except Exception as e:
            logger.exception("Pipeline failed")
            project.update_status(ProjectStatus.FAILED, str(e))
            if project_save_path:
                project.save(project_save_path)
            raise

    async def regenerate_failed_scenes(
        self,
        project: Project,
        output_dir: Path,
    ) -> Project:
        """
        Retry generation for failed scenes.
        
        Args:
            project: Project with some failed scenes
            output_dir: Directory for scene videos
            
        Returns:
            Updated project
        """
        failed_scenes = [s for s in project.scenes if s.status == SceneStatus.FAILED]
        
        if not failed_scenes:
            logger.info("No failed scenes to regenerate")
            return project

        logger.info(f"Regenerating {len(failed_scenes)} failed scenes")

        # Reset scenes for regeneration
        for scene in failed_scenes:
            scene.reset_for_regeneration()

        # Generate
        if self.config.parallel_generation:
            await generate_scenes_parallel(
                failed_scenes,
                self.video_generator,
                output_dir,
                self.config.max_parallel,
            )
        else:
            await generate_scenes_sequential(
                failed_scenes,
                self.video_generator,
                output_dir,
            )

        return project


def run_pipeline_sync(
    audio_path: Path,
    brief_path: Path,
    output_path: Path,
    config: Optional[PipelineConfig] = None,
    project_save_path: Optional[Path] = None,
) -> Project:
    """
    Synchronous wrapper for running the pipeline.
    
    Useful for CLI and simple scripts.
    """
    pipeline = Pipeline(config=config)
    return asyncio.run(pipeline.run(
        audio_path=audio_path,
        brief_path=brief_path,
        output_path=output_path,
        project_save_path=project_save_path,
    ))
