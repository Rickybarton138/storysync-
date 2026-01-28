"""Story analyzer module - interprets lyrics and creates scene breakdowns."""

import json
import logging
from typing import Optional

from anthropic import Anthropic

from config.settings import settings
from config.prompts import (
    STORY_ANALYSIS_SYSTEM_PROMPT,
    STORY_ANALYSIS_USER_TEMPLATE,
)
from models.project import Project, ProjectStatus
from models.scene import Scene, SceneStatus
from models.director_brief import DirectorBrief

logger = logging.getLogger(__name__)


class StoryAnalyzer:
    """Analyzes lyrics and director's brief to create scene breakdowns."""

    def __init__(self, anthropic_api_key: Optional[str] = None):
        """Initialize with API key."""
        self.client = Anthropic(
            api_key=anthropic_api_key or settings.anthropic_api_key
        )

    def analyze(self, project: Project) -> Project:
        """
        Analyze a project and generate scene breakdown.

        Args:
            project: Project with lyrics, sections, and director's brief

        Returns:
            Updated project with scenes
        """
        if not project.director_brief:
            raise ValueError("Project has no director's brief")

        if not project.lyrics_raw and not project.lyrics_timed:
            raise ValueError("Project has no lyrics")

        logger.info(f"Analyzing story for project: {project.title}")
        project.update_status(ProjectStatus.ANALYSING)

        # Build the prompt
        user_prompt = self._build_analysis_prompt(project)

        # Call Claude
        response = self.client.messages.create(
            model=settings.claude_model,
            max_tokens=settings.max_tokens_story_analysis,
            system=STORY_ANALYSIS_SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": user_prompt}
            ],
        )

        # Parse response
        response_text = response.content[0].text
        scene_data = self._parse_scene_response(response_text)

        # Store narrative summary
        project.narrative_summary = scene_data.get("narrative_summary", "")
        project.emotional_arc = scene_data.get("emotional_arc", "")

        # Create Scene objects
        project.scenes = []
        for i, scene_dict in enumerate(scene_data.get("scenes", []), 1):
            scene = Scene(
                scene_number=i,
                project_id=project.id,
                start_time=scene_dict.get("start_time", 0),
                end_time=scene_dict.get("end_time", 0),
                section_type=scene_dict.get("section_type", "verse"),
                lyrics_segment=scene_dict.get("lyrics_segment", ""),
                narrative_beat=scene_dict.get("narrative_beat", ""),
                visual_description=scene_dict.get("visual_description", ""),
                camera_direction=scene_dict.get("camera_direction", ""),
                characters_present=scene_dict.get("characters_present", []),
                emotional_tone=scene_dict.get("emotional_tone", ""),
                lighting=scene_dict.get("lighting", ""),
                key_visual_elements=scene_dict.get("key_visual_elements", []),
                status=SceneStatus.PENDING,
            )
            project.scenes.append(scene)

        # Validate scene coverage
        self._validate_scene_coverage(project)

        project.update_status(ProjectStatus.SCENES_READY)
        logger.info(f"Story analysis complete: {len(project.scenes)} scenes generated")

        return project

    def _build_analysis_prompt(self, project: Project) -> str:
        """Build the user prompt for story analysis."""
        brief = project.director_brief

        # Format setting
        setting_text = brief.setting.to_prompt_description()

        # Format characters
        if brief.characters:
            characters_text = "\n".join(
                f"- **{c.name}** ({c.role}): {c.description}"
                for c in brief.characters
            )
        else:
            characters_text = "No specific characters defined — use the lyrics to infer."

        # Format tone
        tone_text = brief.tone.to_prompt_description()

        # Format key moments
        if brief.key_moments:
            key_moments_text = "\n".join(
                f"- **{km.section}**: {km.description}"
                for km in brief.key_moments
            )
        else:
            key_moments_text = "No specific key moments defined — use your judgement."

        # Additional notes
        additional_notes = ""
        if brief.additional_notes:
            additional_notes = f"\n### Additional Notes\n{brief.additional_notes}"

        if brief.style_references:
            additional_notes += "\n\n### Style References\n"
            additional_notes += "\n".join(f"- {ref}" for ref in brief.style_references)

        return STORY_ANALYSIS_USER_TEMPLATE.format(
            title=project.title,
            duration=project.audio_duration or 180,
            tempo=project.audio_tempo or 120,
            timed_lyrics=project.get_timed_lyrics_text(),
            section_markers=project.get_sections_text(),
            setting=setting_text,
            characters=characters_text,
            tone=tone_text,
            key_moments=key_moments_text,
            additional_notes=additional_notes,
        )

    def _parse_scene_response(self, response_text: str) -> dict:
        """Parse the JSON response from Claude."""
        logger.debug(f"Raw response length: {len(response_text)} chars")
        logger.debug(f"Response preview: {response_text[:500]}...")

        # Extract JSON from response
        json_text = self._extract_json(response_text)

        if not json_text or not json_text.strip():
            logger.error("No JSON extracted from response")
            logger.error(f"Full response: {response_text}")
            raise ValueError("No JSON found in story analysis response")

        try:
            return json.loads(json_text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse scene response: {e}")
            logger.error(f"Extracted JSON text: {json_text[:1000]}...")
            raise ValueError(f"Failed to parse story analysis response: {e}")

    def _extract_json(self, text: str) -> str:
        """Extract JSON from text that might contain markdown code blocks."""
        if not text:
            return ""

        # Try to find JSON in code blocks first
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            if end > start:
                extracted = text[start:end].strip()
                logger.debug(f"Extracted JSON from ```json block: {len(extracted)} chars")
                return extracted
            else:
                # No closing ```, try to extract from { to end
                logger.debug("No closing ``` found, trying to extract JSON object")
                json_start = text.find("{", start)
                if json_start >= 0:
                    return self._extract_json_object(text[json_start:])

        if "```" in text:
            start = text.find("```") + 3
            # Skip any language identifier on the same line
            newline = text.find("\n", start)
            if newline > start:
                start = newline + 1
            end = text.find("```", start)
            if end > start:
                extracted = text[start:end].strip()
                logger.debug(f"Extracted JSON from ``` block: {len(extracted)} chars")
                return extracted

        # Try to find raw JSON (starts with {)
        start = text.find("{")
        if start >= 0:
            return self._extract_json_object(text[start:])

        logger.warning("Could not extract JSON from response")
        return ""

    def _extract_json_object(self, text: str) -> str:
        """Extract a complete JSON object by matching braces."""
        if not text or not text.startswith("{"):
            # Find the first {
            start = text.find("{")
            if start < 0:
                return ""
            text = text[start:]

        depth = 0
        in_string = False
        escape_next = False

        for i, char in enumerate(text):
            if escape_next:
                escape_next = False
                continue

            if char == '\\' and in_string:
                escape_next = True
                continue

            if char == '"' and not escape_next:
                in_string = not in_string
                continue

            if in_string:
                continue

            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    extracted = text[:i + 1]
                    logger.debug(f"Extracted JSON object: {len(extracted)} chars")
                    return extracted

        # If we get here, braces weren't balanced - return what we have
        logger.warning("Unbalanced braces in JSON, attempting partial extraction")
        return text

    def _validate_scene_coverage(self, project: Project) -> None:
        """
        Validate that scenes cover the full song duration.

        Logs warnings for gaps or overlaps but doesn't fail.
        """
        if not project.scenes:
            logger.warning("No scenes generated!")
            return

        duration = project.audio_duration or 0

        # Check first scene starts near 0
        first_scene = project.scenes[0]
        if first_scene.start_time > 1.0:
            logger.warning(
                f"First scene starts at {first_scene.start_time}s, "
                f"not at beginning of song"
            )

        # Check for gaps between scenes
        for i in range(len(project.scenes) - 1):
            current = project.scenes[i]
            next_scene = project.scenes[i + 1]

            gap = next_scene.start_time - current.end_time
            if gap > 0.5:
                logger.warning(
                    f"Gap of {gap:.1f}s between scene {current.scene_number} "
                    f"and scene {next_scene.scene_number}"
                )
            elif gap < -0.5:
                logger.warning(
                    f"Overlap of {-gap:.1f}s between scene {current.scene_number} "
                    f"and scene {next_scene.scene_number}"
                )

        # Check last scene ends near duration
        last_scene = project.scenes[-1]
        if duration and (duration - last_scene.end_time) > 2.0:
            logger.warning(
                f"Last scene ends at {last_scene.end_time}s, "
                f"but song is {duration}s"
            )

    def regenerate_scene(
        self,
        project: Project,
        scene_number: int,
        additional_direction: Optional[str] = None,
    ) -> Scene:
        """
        Regenerate a specific scene with optional additional direction.

        Args:
            project: The project
            scene_number: Which scene to regenerate
            additional_direction: Extra direction for this specific scene

        Returns:
            The regenerated Scene
        """
        scene = project.get_scene_by_number(scene_number)
        if not scene:
            raise ValueError(f"Scene {scene_number} not found")

        logger.info(f"Regenerating scene {scene_number}")

        # Build a focused prompt for just this scene
        prompt = self._build_single_scene_prompt(project, scene, additional_direction)

        response = self.client.messages.create(
            model=settings.claude_model,
            max_tokens=1024,
            system=STORY_ANALYSIS_SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": prompt}
            ],
        )

        response_text = response.content[0].text
        scene_data = self._parse_scene_response(response_text)

        # Update the scene
        if scene_data.get("scenes"):
            new_scene_data = scene_data["scenes"][0]
            scene.narrative_beat = new_scene_data.get("narrative_beat", scene.narrative_beat)
            scene.visual_description = new_scene_data.get("visual_description", scene.visual_description)
            scene.camera_direction = new_scene_data.get("camera_direction", scene.camera_direction)
            scene.characters_present = new_scene_data.get("characters_present", scene.characters_present)
            scene.emotional_tone = new_scene_data.get("emotional_tone", scene.emotional_tone)
            scene.lighting = new_scene_data.get("lighting", scene.lighting)
            scene.key_visual_elements = new_scene_data.get("key_visual_elements", scene.key_visual_elements)

            # Reset generation status
            scene.reset_for_regeneration()

        return scene

    def _build_single_scene_prompt(
        self,
        project: Project,
        scene: Scene,
        additional_direction: Optional[str],
    ) -> str:
        """Build prompt for regenerating a single scene."""
        brief = project.director_brief

        prompt_parts = [
            f"## Regenerate Scene {scene.scene_number}",
            f"\nThis scene covers {scene.start_time:.1f}s - {scene.end_time:.1f}s",
            f"Section type: {scene.section_type}",
            f"\nLyrics for this section:\n{scene.lyrics_segment}",
            f"\n## Context",
            f"Setting: {brief.setting.to_prompt_description()}",
            f"Style: {brief.tone.to_prompt_description()}",
        ]

        if scene.characters_present:
            prompt_parts.append(f"Characters in scene: {', '.join(scene.characters_present)}")

        if additional_direction:
            prompt_parts.append(f"\n## Additional Direction\n{additional_direction}")

        prompt_parts.append(
            "\n\nGenerate a new visual interpretation for this scene. "
            "Return JSON with a single scene in the 'scenes' array."
        )

        return "\n".join(prompt_parts)
