"""Tests for data models."""

import json
import tempfile
from pathlib import Path

import pytest

from models.director_brief import DirectorBrief, Character, Setting, Tone, KeyMoment
from models.scene import Scene, SceneStatus
from models.project import Project, ProjectStatus, TimedWord


class TestDirectorBrief:
    """Tests for DirectorBrief model."""

    def test_create_simple_brief(self):
        """Test creating a brief from simple inputs."""
        brief = DirectorBrief.from_simple_input(
            title="Test Song",
            location="A city street",
            time_period="modern day",
            mood="energetic",
        )

        assert brief.title == "Test Song"
        assert brief.setting.location == "A city street"
        assert brief.tone.mood == "energetic"

    def test_create_full_brief(self):
        """Test creating a complete brief."""
        brief = DirectorBrief(
            title="Full Test",
            setting=Setting(
                location="Paris",
                time_period="1920s",
                environment="Rainy cobblestone streets",
            ),
            characters=[
                Character(
                    name="Alice",
                    description="Young woman in a red dress",
                    role="protagonist",
                ),
            ],
            tone=Tone(
                mood="romantic",
                colour_palette="warm sepia tones",
                visual_style="cinematic noir",
            ),
            key_moments=[
                KeyMoment(section="chorus", description="Dancing in the rain"),
            ],
        )

        assert len(brief.characters) == 1
        assert brief.characters[0].name == "Alice"
        assert len(brief.key_moments) == 1

    def test_brief_to_prompt_context(self):
        """Test generating prompt context from brief."""
        brief = DirectorBrief.from_simple_input(
            title="Test",
            location="Beach at sunset",
            mood="peaceful",
        )

        context = brief.to_prompt_context()
        assert "Beach at sunset" in context
        assert "peaceful" in context

    def test_save_and_load_brief(self):
        """Test saving and loading brief from JSON."""
        brief = DirectorBrief.from_simple_input(
            title="Save Test",
            location="Mountains",
            characters=[
                {"name": "Hero", "description": "Brave warrior", "role": "protagonist"}
            ],
        )

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            temp_path = Path(f.name)

        try:
            brief.save(temp_path)
            loaded = DirectorBrief.from_json_file(temp_path)

            assert loaded.title == brief.title
            assert loaded.setting.location == brief.setting.location
            assert len(loaded.characters) == 1
            assert loaded.characters[0].name == "Hero"
        finally:
            temp_path.unlink()


class TestScene:
    """Tests for Scene model."""

    def test_create_scene(self):
        """Test creating a scene."""
        scene = Scene(
            scene_number=1,
            start_time=0.0,
            end_time=10.0,
            section_type="intro",
            narrative_beat="Opening shot",
        )

        assert scene.scene_number == 1
        assert scene.duration == 10.0
        assert scene.status == SceneStatus.PENDING

    def test_scene_status_transitions(self):
        """Test scene status changes."""
        scene = Scene(scene_number=1, start_time=0, end_time=5)

        assert scene.needs_generation
        assert not scene.is_generated

        scene.mark_generating()
        assert scene.status == SceneStatus.GENERATING
        assert scene.generation_attempts == 1

        scene.mark_complete(Path("/tmp/video.mp4"))
        assert scene.status == SceneStatus.COMPLETE
        assert scene.is_generated

        scene.approve()
        assert scene.status == SceneStatus.APPROVED

    def test_scene_failure(self):
        """Test marking scene as failed."""
        scene = Scene(scene_number=1, start_time=0, end_time=5)

        scene.mark_generating()
        scene.mark_failed("API error")

        assert scene.status == SceneStatus.FAILED
        assert scene.error_message == "API error"
        assert scene.needs_generation

    def test_scene_regeneration(self):
        """Test resetting scene for regeneration."""
        scene = Scene(
            scene_number=1,
            start_time=0,
            end_time=5,
            video_path=Path("/tmp/old.mp4"),
        )
        scene.mark_complete(Path("/tmp/old.mp4"))

        scene.reset_for_regeneration()

        assert scene.status == SceneStatus.PENDING
        assert scene.video_path is None
        assert scene.needs_generation


class TestProject:
    """Tests for Project model."""

    def test_create_project(self):
        """Test creating a project."""
        brief = DirectorBrief.from_simple_input(
            title="Test Project",
            location="Studio",
        )

        project = Project.create(
            title="Test Project",
            audio_path=Path("/tmp/audio.mp3"),
            director_brief=brief,
        )

        assert project.title == "Test Project"
        assert project.status == ProjectStatus.DRAFT
        assert len(project.scenes) == 0

    def test_project_progress(self):
        """Test project progress calculation."""
        project = Project(title="Test")

        # Add some scenes
        for i in range(5):
            scene = Scene(scene_number=i + 1, start_time=i * 10, end_time=(i + 1) * 10)
            if i < 3:  # Mark first 3 as complete
                scene.mark_complete(Path(f"/tmp/{i}.mp4"))
            project.add_scene(scene)

        assert project.scene_count == 5
        assert project.scenes_complete == 3
        assert project.progress_percent == 60.0
        assert not project.all_scenes_ready

    def test_timed_lyrics_formatting(self):
        """Test timed lyrics formatting for LLM."""
        project = Project(title="Test")
        project.lyrics_timed = [
            TimedWord(word="Hello", start=0.0, end=0.5),
            TimedWord(word="world,", start=0.5, end=1.0),
            TimedWord(word="how", start=1.0, end=1.3),
            TimedWord(word="are", start=1.3, end=1.5),
            TimedWord(word="you?", start=1.5, end=2.0),
        ]

        text = project.get_timed_lyrics_text()
        assert "[0.0s]" in text
        assert "Hello" in text

    def test_save_and_load_project(self):
        """Test saving and loading project."""
        brief = DirectorBrief.from_simple_input(title="Save Test", location="Test")
        project = Project.create(
            title="Save Test",
            audio_path=Path("/tmp/audio.mp3"),
            director_brief=brief,
        )

        scene = Scene(scene_number=1, start_time=0, end_time=10, narrative_beat="Test scene")
        project.add_scene(scene)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            temp_path = Path(f.name)

        try:
            project.save(temp_path)
            loaded = Project.load(temp_path)

            assert loaded.title == project.title
            assert len(loaded.scenes) == 1
            assert loaded.scenes[0].narrative_beat == "Test scene"
        finally:
            temp_path.unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
