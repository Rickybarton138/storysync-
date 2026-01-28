"""Director's Brief data models."""

from pathlib import Path
from typing import Optional
import json

from pydantic import BaseModel, Field


class Character(BaseModel):
    """A character in the music video."""

    name: str = Field(description="Character's name for reference")
    description: str = Field(description="Visual description of the character")
    role: str = Field(
        default="supporting",
        description="Role in the story: protagonist, love_interest, antagonist, supporting",
    )
    reference_image_path: Optional[Path] = Field(
        default=None,
        description="Path to reference image for character consistency",
    )

    def to_prompt_description(self) -> str:
        """Convert to a description suitable for video prompts."""
        return f"{self.name}: {self.description}"


class Setting(BaseModel):
    """The setting/world of the music video."""

    location: str = Field(description="Primary location description")
    time_period: str = Field(default="contemporary", description="Era or time period")
    environment: str = Field(
        default="",
        description="Additional environment details (weather, atmosphere, etc.)",
    )

    def to_prompt_description(self) -> str:
        """Convert to a description suitable for video prompts."""
        parts = [self.location]
        if self.time_period and self.time_period != "contemporary":
            parts.append(f"({self.time_period})")
        if self.environment:
            parts.append(f"— {self.environment}")
        return " ".join(parts)


class Tone(BaseModel):
    """Visual tone and style for the music video."""

    mood: str = Field(description="Emotional mood (e.g., 'melancholic, hopeful')")
    colour_palette: str = Field(
        default="natural",
        description="Colour grading description",
    )
    visual_style: str = Field(
        default="cinematic",
        description="Visual style (e.g., 'cinematic, handheld, documentary')",
    )

    def to_prompt_description(self) -> str:
        """Convert to style keywords for video prompts."""
        parts = []
        if self.visual_style:
            parts.append(self.visual_style)
        if self.colour_palette and self.colour_palette != "natural":
            parts.append(f"{self.colour_palette} color grade")
        if self.mood:
            parts.append(f"{self.mood} mood")
        return ", ".join(parts)


class KeyMoment(BaseModel):
    """A specific moment the director wants to see in the video."""

    section: str = Field(description="Song section (intro, verse_1, chorus, etc.)")
    description: str = Field(description="What should happen visually at this moment")


class DirectorBrief(BaseModel):
    """Complete director's brief for a music video project."""

    title: str = Field(description="Song/project title")
    setting: Setting = Field(description="World and location settings")
    characters: list[Character] = Field(
        default_factory=list,
        description="Characters appearing in the video",
    )
    tone: Tone = Field(description="Visual tone and style")
    key_moments: list[KeyMoment] = Field(
        default_factory=list,
        description="Specific moments the director wants to include",
    )
    reference_images: list[Path] = Field(
        default_factory=list,
        description="Mood board / reference images",
    )
    style_references: list[str] = Field(
        default_factory=list,
        description="Reference films or videos for style",
    )
    additional_notes: str = Field(
        default="",
        description="Any additional direction or notes",
    )

    @classmethod
    def from_json_file(cls, path: Path) -> "DirectorBrief":
        """Load a director's brief from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.model_validate(data)

    @classmethod
    def from_simple_input(
        cls,
        title: str,
        location: str,
        time_period: str = "contemporary",
        mood: str = "cinematic",
        visual_style: str = "cinematic, 35mm film aesthetic",
        characters: list[dict] | None = None,
        key_moments: list[dict] | None = None,
    ) -> "DirectorBrief":
        """Create a brief from simple inputs (for quick CLI usage)."""
        return cls(
            title=title,
            setting=Setting(
                location=location,
                time_period=time_period,
            ),
            characters=[Character(**c) for c in (characters or [])],
            tone=Tone(
                mood=mood,
                visual_style=visual_style,
            ),
            key_moments=[KeyMoment(**km) for km in (key_moments or [])],
        )

    def get_character_by_name(self, name: str) -> Optional[Character]:
        """Find a character by name."""
        for char in self.characters:
            if char.name.lower() == name.lower():
                return char
        return None

    def get_key_moment_for_section(self, section: str) -> Optional[KeyMoment]:
        """Get the key moment specified for a section, if any."""
        section_lower = section.lower()
        for km in self.key_moments:
            if km.section.lower() == section_lower:
                return km
        return None

    def to_prompt_context(self) -> str:
        """Generate context string for LLM prompts."""
        lines = [
            f"## Setting\n{self.setting.to_prompt_description()}",
            f"\n## Visual Style\n{self.tone.to_prompt_description()}",
        ]

        if self.characters:
            char_lines = [c.to_prompt_description() for c in self.characters]
            lines.append(f"\n## Characters\n" + "\n".join(f"- {c}" for c in char_lines))

        if self.style_references:
            lines.append(
                f"\n## Style References\n"
                + "\n".join(f"- {ref}" for ref in self.style_references)
            )

        return "\n".join(lines)

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return self.model_dump_json(indent=indent)

    def save(self, path: Path) -> None:
        """Save to a JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_json())
