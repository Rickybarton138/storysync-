"""StorySync Configuration Module"""

from .settings import settings
from .prompts import STORY_ANALYSIS_SYSTEM_PROMPT, PROMPT_GENERATOR_SYSTEM_PROMPT

__all__ = [
    "settings",
    "STORY_ANALYSIS_SYSTEM_PROMPT",
    "PROMPT_GENERATOR_SYSTEM_PROMPT",
]
