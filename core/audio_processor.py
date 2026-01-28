"""Audio processing module - transcription and analysis."""

import json
import logging
from pathlib import Path
from typing import Optional

from openai import OpenAI
from anthropic import Anthropic

from config.settings import settings
from config.prompts import SECTION_DETECTION_SYSTEM_PROMPT, SECTION_DETECTION_USER_TEMPLATE
from models.project import Project, TimedWord, SongSection

logger = logging.getLogger(__name__)


class AudioProcessor:
    """Handles audio transcription and analysis."""

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
    ):
        """Initialize with API keys."""
        self.openai_client = OpenAI(
            api_key=openai_api_key or settings.openai_api_key
        )
        self.anthropic_client = Anthropic(
            api_key=anthropic_api_key or settings.anthropic_api_key
        )

    def transcribe(self, audio_path: Path) -> dict:
        """
        Transcribe audio file using Whisper.

        Returns dict with:
            - text: Full transcription
            - words: List of {word, start, end}
            - duration: Audio duration in seconds
        """
        logger.info(f"Transcribing audio: {audio_path}")

        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        with open(audio_path, "rb") as audio_file:
            response = self.openai_client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="verbose_json",
                timestamp_granularities=["word", "segment"],
            )

        # Extract data from response
        result = {
            "text": response.text,
            "words": [],
            "segments": [],
            "duration": 0.0,
        }

        # Process words
        if hasattr(response, "words") and response.words:
            for word_data in response.words:
                result["words"].append({
                    "word": word_data.word,
                    "start": word_data.start,
                    "end": word_data.end,
                })
                # Track duration from last word
                if word_data.end > result["duration"]:
                    result["duration"] = word_data.end

        # Process segments
        if hasattr(response, "segments") and response.segments:
            for segment in response.segments:
                result["segments"].append({
                    "text": segment.text,
                    "start": segment.start,
                    "end": segment.end,
                })
                if segment.end > result["duration"]:
                    result["duration"] = segment.end

        logger.info(f"Transcription complete: {len(result['words'])} words, {result['duration']:.1f}s")
        return result

    def detect_sections(
        self,
        lyrics_timed: list[TimedWord],
        duration: float,
    ) -> list[SongSection]:
        """
        Detect song sections (verse, chorus, etc.) using LLM analysis.

        Args:
            lyrics_timed: List of timed words
            duration: Total song duration in seconds

        Returns:
            List of SongSection objects
        """
        logger.info("Detecting song sections...")

        # Format timed lyrics for the prompt
        timed_lyrics_text = self._format_timed_lyrics(lyrics_timed)

        # Call Claude to analyse structure
        response = self.anthropic_client.messages.create(
            model=settings.claude_model,
            max_tokens=2048,
            system=SECTION_DETECTION_SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": SECTION_DETECTION_USER_TEMPLATE.format(
                        duration=duration,
                        timed_lyrics=timed_lyrics_text,
                    ),
                }
            ],
        )

        # Parse response
        response_text = response.content[0].text

        # Extract JSON from response (handle markdown code blocks)
        json_text = self._extract_json(response_text)
        data = json.loads(json_text)

        # Convert to SongSection objects
        sections = []
        for section_data in data.get("sections", []):
            sections.append(SongSection(
                section_type=section_data["type"],
                start_time=section_data["start_time"],
                end_time=section_data["end_time"],
                lyrics_preview=section_data.get("lyrics_preview", ""),
            ))

        logger.info(f"Detected {len(sections)} sections")
        return sections

    def estimate_tempo(self, audio_path: Path) -> float:
        """
        Estimate tempo (BPM) from audio file.

        This is a simplified version - for production, you'd use librosa
        or a dedicated beat detection service.
        """
        # For MVP, return a reasonable default
        # TODO: Implement proper tempo detection with librosa
        logger.info("Tempo detection not yet implemented - using default")
        return 120.0

    def get_audio_duration(self, audio_path: Path) -> float:
        """Get duration of audio file in seconds."""
        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_file(audio_path)
            return len(audio) / 1000.0  # pydub uses milliseconds
        except Exception as e:
            logger.warning(f"Could not get duration with pydub: {e}")
            # Fall back to ffprobe if available
            return self._get_duration_ffprobe(audio_path)

    def _get_duration_ffprobe(self, audio_path: Path) -> float:
        """Get duration using ffprobe."""
        import subprocess

        try:
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v", "quiet",
                    "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1",
                    str(audio_path),
                ],
                capture_output=True,
                text=True,
            )
            return float(result.stdout.strip())
        except Exception as e:
            logger.error(f"Could not get duration: {e}")
            raise

    def _format_timed_lyrics(self, lyrics_timed: list[TimedWord]) -> str:
        """Format timed lyrics for LLM consumption."""
        lines = []
        current_line = []
        current_line_start = None

        for word in lyrics_timed:
            if current_line_start is None:
                current_line_start = word.start

            current_line.append(word.word)

            # Create new line after punctuation or every ~10 words
            is_line_end = (
                word.word.rstrip().endswith((".", "!", "?", ","))
                or len(current_line) >= 10
            )

            if is_line_end:
                line_text = " ".join(current_line)
                lines.append(f"[{current_line_start:.1f}s - {word.end:.1f}s] {line_text}")
                current_line = []
                current_line_start = None

        # Handle remaining words
        if current_line:
            line_text = " ".join(current_line)
            end_time = lyrics_timed[-1].end if lyrics_timed else 0
            lines.append(f"[{current_line_start:.1f}s - {end_time:.1f}s] {line_text}")

        return "\n".join(lines)

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
            # Find matching closing brace
            depth = 0
            for i, char in enumerate(text[start:], start):
                if char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        return text[start:i + 1]

        return text

    def process_audio(self, project: Project) -> Project:
        """
        Process audio for a project: transcribe and detect sections.

        Args:
            project: Project with audio_path set

        Returns:
            Updated project with lyrics and sections
        """
        if not project.audio_path:
            raise ValueError("Project has no audio_path set")

        # Get duration
        project.audio_duration = self.get_audio_duration(project.audio_path)

        # Transcribe
        transcription = self.transcribe(project.audio_path)
        project.lyrics_raw = transcription["text"]
        project.lyrics_timed = [
            TimedWord(**word) for word in transcription["words"]
        ]

        # If transcription gave us a longer duration, use that
        if transcription["duration"] > project.audio_duration:
            project.audio_duration = transcription["duration"]

        # Estimate tempo
        project.audio_tempo = self.estimate_tempo(project.audio_path)

        # Detect sections
        project.sections = self.detect_sections(
            project.lyrics_timed,
            project.audio_duration,
        )

        logger.info(
            f"Audio processing complete: "
            f"{project.audio_duration:.1f}s, "
            f"{len(project.lyrics_timed)} words, "
            f"{len(project.sections)} sections"
        )

        return project

    def process_audio_with_lyrics(self, project: Project, lyrics: str) -> Project:
        """
        Process audio with manually provided lyrics (skips Whisper transcription).

        Args:
            project: Project with audio_path set
            lyrics: Raw lyrics text with optional section markers like [Verse 1], [Chorus]

        Returns:
            Updated project with lyrics and sections
        """
        import re

        if not project.audio_path:
            raise ValueError("Project has no audio_path set")

        # Get duration
        project.audio_duration = self.get_audio_duration(project.audio_path)
        logger.info(f"Audio duration: {project.audio_duration:.1f}s")

        # Store raw lyrics
        project.lyrics_raw = lyrics

        # Parse sections from lyrics markers
        sections = self._parse_sections_from_lyrics(lyrics, project.audio_duration)
        project.sections = sections

        # Create timed words by distributing across duration
        project.lyrics_timed = self._create_timed_words(lyrics, project.audio_duration)

        # Estimate tempo
        project.audio_tempo = self.estimate_tempo(project.audio_path)

        logger.info(
            f"Audio processing complete (manual lyrics): "
            f"{project.audio_duration:.1f}s, "
            f"{len(project.lyrics_timed)} words, "
            f"{len(project.sections)} sections"
        )

        return project

    def _parse_sections_from_lyrics(self, lyrics: str, duration: float) -> list[SongSection]:
        """
        Parse section markers from lyrics text.

        Looks for patterns like [Verse 1], [Chorus], [Bridge], etc.
        """
        import re

        # Find all section markers
        section_pattern = r'\[([^\]]+)\]'
        matches = list(re.finditer(section_pattern, lyrics))

        if not matches:
            # No section markers - create a single section
            return [SongSection(
                section_type="song",
                start_time=0.0,
                end_time=duration,
                lyrics_preview=lyrics[:100].strip(),
            )]

        sections = []
        for i, match in enumerate(matches):
            section_type = match.group(1).lower().replace(" ", "_")

            # Map common variations to standard types
            type_map = {
                "verse_1": "verse",
                "verse_2": "verse",
                "verse_3": "verse",
                "chorus_1": "chorus",
                "chorus_2": "chorus",
                "pre-chorus": "pre_chorus",
                "pre_chorus": "pre_chorus",
            }
            section_type = type_map.get(section_type, section_type)

            # Calculate timing - distribute sections evenly
            start_time = (i / len(matches)) * duration
            end_time = ((i + 1) / len(matches)) * duration

            # Get lyrics for this section (up to next marker or end)
            section_start = match.end()
            section_end = matches[i + 1].start() if i + 1 < len(matches) else len(lyrics)
            section_lyrics = lyrics[section_start:section_end].strip()

            # Clean up lyrics preview
            preview = re.sub(r'\[.*?\]', '', section_lyrics)
            preview = ' '.join(preview.split())[:100]

            sections.append(SongSection(
                section_type=section_type,
                start_time=start_time,
                end_time=end_time,
                lyrics_preview=preview,
            ))

        return sections

    def _create_timed_words(self, lyrics: str, duration: float) -> list[TimedWord]:
        """
        Create approximate timed words by distributing evenly across duration.
        """
        import re

        # Remove section markers and clean up
        clean_lyrics = re.sub(r'\[.*?\]', '', lyrics)
        words = clean_lyrics.split()

        if not words:
            return []

        # Calculate time per word (with small gaps)
        total_words = len(words)
        time_per_word = duration / total_words

        timed_words = []
        for i, word in enumerate(words):
            start = i * time_per_word
            end = start + (time_per_word * 0.8)  # Leave small gap

            timed_words.append(TimedWord(
                word=word,
                start=start,
                end=end,
            ))

        return timed_words
