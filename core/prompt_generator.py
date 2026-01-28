"""Prompt generator module - converts scene descriptions to video API prompts."""

import json
import logging
from typing import Optional

from anthropic import Anthropic

from config.settings import settings
from config.prompts import (
    PROMPT_GENERATOR_SYSTEM_PROMPT,
    VIDEO_PROMPT_USER_TEMPLATE,
)
from models.project import Project
from models.scene import Scene
from models.director_brief import DirectorBrief

logger = logging.getLogger(__name__)


class PromptGenerator:
    """Generates optimised video prompts from scene descriptions."""

    def __init__(self, anthropic_api_key: Optional[str] = None):
        """Initialize with API key."""
        self.client = Anthropic(
            api_key=anthropic_api_key or settings.anthropic_api_key
        )

    def generate_prompt(
        self,
        scene: Scene,
        brief: DirectorBrief,
    ) -> Scene:
        """
        Generate video prompt for a single scene.

        Args:
            scene: Scene with visual description
            brief: Director's brief for context

        Returns:
            Scene with populated prompts
        """
        logger.debug(f"Generating prompt for scene {scene.scene_number}")

        user_prompt = self._build_user_prompt(scene, brief)

        response = self.client.messages.create(
            model=settings.claude_model,
            max_tokens=settings.max_tokens_prompt_generation,
            system=PROMPT_GENERATOR_SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": user_prompt}
            ],
        )

        response_text = response.content[0].text
        prompt_data = self._parse_prompt_response(response_text)

        # Update scene with generated prompts
        scene.positive_prompt = prompt_data.get("positive_prompt", "")
        scene.negative_prompt = prompt_data.get("negative_prompt", "")
        scene.suggested_camera_motion = prompt_data.get("camera_motion", "static")

        return scene

    def generate_prompts_batch(
        self,
        project: Project,
        scenes: Optional[list[Scene]] = None,
    ) -> list[Scene]:
        """
        Generate prompts for multiple scenes.

        Args:
            project: Project with director's brief
            scenes: Specific scenes to process (defaults to all pending)

        Returns:
            List of scenes with generated prompts
        """
        if not project.director_brief:
            raise ValueError("Project has no director's brief")

        scenes_to_process = scenes or [
            s for s in project.scenes if not s.positive_prompt
        ]

        logger.info(f"Generating prompts for {len(scenes_to_process)} scenes")

        for scene in scenes_to_process:
            self.generate_prompt(scene, project.director_brief)

        return scenes_to_process

    def _build_user_prompt(self, scene: Scene, brief: DirectorBrief) -> str:
        """Build the user prompt for video prompt generation."""

        # Build style context from brief
        style_context = brief.to_prompt_context()

        # Build character descriptions
        if scene.characters_present and brief.characters:
            char_descs = []
            for char_name in scene.characters_present:
                char = brief.get_character_by_name(char_name)
                if char:
                    char_descs.append(f"- {char.to_prompt_description()}")
            character_descriptions = "\n".join(char_descs) if char_descs else "None specified"
        else:
            character_descriptions = "None specified"

        return VIDEO_PROMPT_USER_TEMPLATE.format(
            scene_number=scene.scene_number,
            start_time=scene.start_time,
            end_time=scene.end_time,
            section_type=scene.section_type,
            lyrics_segment=scene.lyrics_segment or "(instrumental)",
            narrative_beat=scene.narrative_beat,
            visual_description=scene.visual_description,
            camera_direction=scene.camera_direction,
            characters_present=", ".join(scene.characters_present) or "None",
            emotional_tone=scene.emotional_tone,
            lighting=scene.lighting,
            key_visual_elements=", ".join(scene.key_visual_elements) or "None specified",
            style_context=style_context,
            character_descriptions=character_descriptions,
        )

    def _parse_prompt_response(self, response_text: str) -> dict:
        """Parse the JSON response containing prompts."""
        json_text = self._extract_json(response_text)

        try:
            return json.loads(json_text)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse prompt JSON, using raw text: {e}")
            # Fall back to using the response as the prompt directly
            return {
                "positive_prompt": response_text.strip(),
                "negative_prompt": "blurry, low quality, distorted, text, watermark",
                "camera_motion": "static",
            }

    def _extract_json(self, text: str) -> str:
        """Extract JSON from text that might contain markdown code blocks."""
        # Try to find JSON in code blocks first
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            if end > start:
                return text[start:end].strip()

        if "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            if end > start:
                return text[start:end].strip()

        # Try to find raw JSON (starts with {)
        start = text.find("{")
        if start >= 0:
            depth = 0
            for i, char in enumerate(text[start:], start):
                if char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        return text[start:i + 1]

        return text

    def enhance_prompt(
        self,
        scene: Scene,
        brief: DirectorBrief,
        feedback: str,
    ) -> Scene:
        """
        Enhance an existing prompt based on feedback.

        Args:
            scene: Scene with existing prompt
            brief: Director's brief
            feedback: What to improve

        Returns:
            Scene with enhanced prompt
        """
        enhance_prompt = f"""## Current Prompt
{scene.positive_prompt}

## Feedback / Improvement Request
{feedback}

## Context
{brief.to_prompt_context()}

---

Generate an improved prompt that addresses the feedback while maintaining the scene's core visual intent. Return JSON format."""

        response = self.client.messages.create(
            model=settings.claude_model,
            max_tokens=settings.max_tokens_prompt_generation,
            system=PROMPT_GENERATOR_SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": enhance_prompt}
            ],
        )

        response_text = response.content[0].text
        prompt_data = self._parse_prompt_response(response_text)

        scene.positive_prompt = prompt_data.get("positive_prompt", scene.positive_prompt)
        scene.negative_prompt = prompt_data.get("negative_prompt", scene.negative_prompt)
        scene.suggested_camera_motion = prompt_data.get("camera_motion", scene.suggested_camera_motion)

        return scene


class QuickPromptGenerator:
    """
    A simpler prompt generator that doesn't use LLM calls.
    
    Useful for testing or when you want direct control over prompts.
    """

    def __init__(self):
        self.base_negative = (
            "blurry, low quality, distorted, deformed, "
            "text, watermark, logo, signature, "
            "cartoon, anime, 3D render"
        )

    def generate_prompt(
        self,
        scene: Scene,
        brief: DirectorBrief,
    ) -> Scene:
        """Generate a simple prompt by concatenating scene elements."""

        # Build positive prompt from scene elements
        parts = []

        # Start with visual description
        if scene.visual_description:
            parts.append(scene.visual_description)

        # Add character details
        for char_name in scene.characters_present:
            char = brief.get_character_by_name(char_name)
            if char:
                parts.append(char.description)

        # Add camera direction
        if scene.camera_direction:
            parts.append(scene.camera_direction)

        # Add lighting
        if scene.lighting:
            parts.append(scene.lighting)

        # Add style from brief
        parts.append(brief.tone.to_prompt_description())

        # Add key visual elements
        if scene.key_visual_elements:
            parts.append(", ".join(scene.key_visual_elements))

        scene.positive_prompt = ". ".join(parts)

        # Build negative prompt
        negative_parts = [self.base_negative]

        # Add period-specific negatives
        time_period = brief.setting.time_period.lower()
        if any(era in time_period for era in ["1940", "1930", "1920", "victorian", "medieval"]):
            negative_parts.append("modern clothing, smartphones, cars, contemporary")

        scene.negative_prompt = ", ".join(negative_parts)

        # Infer camera motion
        camera_lower = scene.camera_direction.lower()
        if "push in" in camera_lower or "dolly in" in camera_lower:
            scene.suggested_camera_motion = "push_in"
        elif "pull" in camera_lower or "dolly out" in camera_lower:
            scene.suggested_camera_motion = "pull_back"
        elif "pan left" in camera_lower:
            scene.suggested_camera_motion = "pan_left"
        elif "pan right" in camera_lower:
            scene.suggested_camera_motion = "pan_right"
        elif "tilt" in camera_lower:
            scene.suggested_camera_motion = "tilt_up" if "up" in camera_lower else "tilt_down"
        elif "tracking" in camera_lower or "follow" in camera_lower:
            scene.suggested_camera_motion = "tracking"
        elif "handheld" in camera_lower:
            scene.suggested_camera_motion = "handheld"
        else:
            scene.suggested_camera_motion = "static"

        return scene
