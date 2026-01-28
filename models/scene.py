"""Scene data model."""

from enum import Enum
from pathlib import Path
from typing import Optional
from datetime import datetime

from pydantic import BaseModel, Field


class SceneStatus(str, Enum):
    """Status of a scene in the generation pipeline."""

    PENDING = "pending"
    GENERATING = "generating"
    COMPLETE = "complete"
    FAILED = "failed"
    APPROVED = "approved"


class Scene(BaseModel):
    """A single scene in the music video."""

    # Identity
    scene_number: int = Field(description="Scene number (1-indexed)")
    project_id: Optional[str] = Field(default=None, description="Parent project ID")

    # Timing
    start_time: float = Field(description="Start time in seconds")
    end_time: float = Field(description="End time in seconds")

    # Song context
    section_type: str = Field(
        default="verse",
        description="Type of section: intro, verse, pre_chorus, chorus, bridge, outro, instrumental",
    )
    lyrics_segment: str = Field(
        default="",
        description="Lyrics that play during this scene",
    )

    # Story content (from LLM analysis)
    narrative_beat: str = Field(
        default="",
        description="What's happening in the story at this moment",
    )
    visual_description: str = Field(
        default="",
        description="Detailed visual description of the scene",
    )
    camera_direction: str = Field(
        default="",
        description="Camera movement and framing",
    )
    characters_present: list[str] = Field(
        default_factory=list,
        description="Names of characters in this scene",
    )
    emotional_tone: str = Field(
        default="",
        description="Emotional feel of the scene",
    )
    lighting: str = Field(
        default="",
        description="Lighting description",
    )
    key_visual_elements: list[str] = Field(
        default_factory=list,
        description="Important visual elements to include",
    )

    # Generated prompts
    positive_prompt: str = Field(
        default="",
        description="The prompt sent to video generation",
    )
    negative_prompt: str = Field(
        default="",
        description="Negative prompt for video generation",
    )
    suggested_camera_motion: str = Field(
        default="static",
        description="Suggested camera motion type",
    )

    # Generation status
    status: SceneStatus = Field(
        default=SceneStatus.PENDING,
        description="Current generation status",
    )
    generation_attempts: int = Field(
        default=0,
        description="Number of generation attempts",
    )
    video_path: Optional[Path] = Field(
        default=None,
        description="Path to generated video file",
    )
    thumbnail_path: Optional[Path] = Field(
        default=None,
        description="Path to thumbnail image",
    )
    video_url: Optional[str] = Field(
        default=None,
        description="URL of generated video (for remote storage)",
    )
    error_message: Optional[str] = Field(
        default=None,
        description="Error message if generation failed",
    )

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    generated_at: Optional[datetime] = Field(default=None)

    @property
    def duration(self) -> float:
        """Duration of the scene in seconds."""
        return self.end_time - self.start_time

    @property
    def is_generated(self) -> bool:
        """Check if video has been successfully generated."""
        return self.status in (SceneStatus.COMPLETE, SceneStatus.APPROVED)

    @property
    def needs_generation(self) -> bool:
        """Check if scene needs video generation."""
        return self.status in (SceneStatus.PENDING, SceneStatus.FAILED)

    @property
    def has_video(self) -> bool:
        """Check if scene has a video file available."""
        if self.video_path and self.video_path.exists():
            return True
        if self.video_url:
            return True
        return False

    def mark_generating(self) -> None:
        """Mark scene as currently generating."""
        self.status = SceneStatus.GENERATING
        self.generation_attempts += 1
        self.updated_at = datetime.utcnow()
        self.error_message = None

    def mark_complete(self, video_path: Path, thumbnail_path: Optional[Path] = None) -> None:
        """Mark scene as successfully generated."""
        self.status = SceneStatus.COMPLETE
        self.video_path = video_path
        self.thumbnail_path = thumbnail_path
        self.generated_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
        self.error_message = None

    def mark_failed(self, error: str) -> None:
        """Mark scene as failed."""
        self.status = SceneStatus.FAILED
        self.error_message = error
        self.updated_at = datetime.utcnow()

    def approve(self) -> None:
        """Mark scene as approved by user."""
        if self.status == SceneStatus.COMPLETE:
            self.status = SceneStatus.APPROVED
            self.updated_at = datetime.utcnow()

    def reset_for_regeneration(self) -> None:
        """Reset scene for regeneration."""
        self.status = SceneStatus.PENDING
        self.video_path = None
        self.thumbnail_path = None
        self.video_url = None
        self.error_message = None
        self.updated_at = datetime.utcnow()

    def to_summary(self) -> str:
        """Generate a human-readable summary."""
        status_icon = {
            SceneStatus.PENDING: "⏳",
            SceneStatus.GENERATING: "🔄",
            SceneStatus.COMPLETE: "✅",
            SceneStatus.FAILED: "❌",
            SceneStatus.APPROVED: "✓✓",
        }
        return (
            f"{status_icon.get(self.status, '?')} Scene {self.scene_number} "
            f"[{self.start_time:.1f}s - {self.end_time:.1f}s] "
            f"({self.section_type}) - {self.narrative_beat[:50]}..."
        )

    def to_generation_request(self) -> dict:
        """Convert to a dict suitable for video generation API."""
        return {
            "prompt": self.positive_prompt,
            "negative_prompt": self.negative_prompt,
            "duration": min(self.duration, 10.0),  # Most APIs cap at 10s
            "camera_motion": self.suggested_camera_motion,
        }
