"""StorySync Command Line Interface."""

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint

from config.settings import settings
from models.project import Project, ProjectStatus
from models.director_brief import DirectorBrief
from models.scene import SceneStatus
from core.audio_processor import AudioProcessor
from core.story_analyzer import StoryAnalyzer
from core.prompt_generator import PromptGenerator, QuickPromptGenerator
from core.video_generator import VideoGeneratorFactory, generate_scenes_sequential
from core.video_assembler import VideoAssembler, check_ffmpeg_available

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

console = Console()


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
def cli(verbose: bool):
    """StorySync - Turn your songs into cinematic music videos."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)


@cli.command()
@click.option("--audio", "-a", required=True, type=click.Path(exists=True), help="Path to audio file")
@click.option("--brief", "-b", required=True, type=click.Path(exists=True), help="Path to director's brief JSON")
@click.option("--output", "-o", required=True, type=click.Path(), help="Output video path")
@click.option("--video-model", "-m", type=click.Choice(["kling", "minimax", "mock"]), default=None, help="Video generation model")
@click.option("--skip-generation", is_flag=True, help="Skip video generation (analysis only)")
@click.option("--project-file", "-p", type=click.Path(), default=None, help="Save/load project state to file")
def generate(
    audio: str,
    brief: str,
    output: str,
    video_model: Optional[str],
    skip_generation: bool,
    project_file: Optional[str],
):
    """Generate a complete music video from audio and director's brief."""
    
    audio_path = Path(audio)
    brief_path = Path(brief)
    output_path = Path(output)
    project_path = Path(project_file) if project_file else None

    # Validate inputs
    if not audio_path.exists():
        console.print(f"[red]Audio file not found: {audio_path}[/red]")
        sys.exit(1)

    if not brief_path.exists():
        console.print(f"[red]Brief file not found: {brief_path}[/red]")
        sys.exit(1)

    # Check ffmpeg
    if not skip_generation and not check_ffmpeg_available():
        console.print("[red]FFmpeg not found. Please install FFmpeg to assemble videos.[/red]")
        sys.exit(1)

    console.print(Panel.fit(
        "[bold blue]StorySync[/bold blue]\n"
        "Turning your song into a cinematic music video",
        border_style="blue",
    ))

    # Load director's brief
    console.print("\n[bold]Loading director's brief...[/bold]")
    try:
        director_brief = DirectorBrief.from_json_file(brief_path)
        console.print(f"  Title: {director_brief.title}")
        console.print(f"  Setting: {director_brief.setting.location}")
        console.print(f"  Characters: {len(director_brief.characters)}")
        console.print(f"  Key moments: {len(director_brief.key_moments)}")
    except Exception as e:
        console.print(f"[red]Failed to load brief: {e}[/red]")
        sys.exit(1)

    # Create or load project
    if project_path and project_path.exists():
        console.print(f"\n[bold]Loading existing project from {project_path}...[/bold]")
        project = Project.load(project_path)
    else:
        project = Project.create(
            title=director_brief.title,
            audio_path=audio_path,
            director_brief=director_brief,
        )

    # Run the pipeline
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            
            # Stage 1: Audio Processing
            if not project.lyrics_raw:
                task = progress.add_task("[cyan]Processing audio...", total=None)
                audio_processor = AudioProcessor()
                project = audio_processor.process_audio(project)
                progress.update(task, completed=True)
                console.print(f"  ✓ Transcribed {len(project.lyrics_timed)} words")
                console.print(f"  ✓ Detected {len(project.sections)} sections")
                console.print(f"  ✓ Duration: {project.audio_duration:.1f}s")

            # Stage 2: Story Analysis
            if not project.scenes:
                task = progress.add_task("[cyan]Analyzing story...", total=None)
                analyzer = StoryAnalyzer()
                project = analyzer.analyze(project)
                progress.update(task, completed=True)
                console.print(f"  ✓ Generated {len(project.scenes)} scenes")
                console.print(f"  ✓ Narrative: {project.narrative_summary[:100]}...")

            # Stage 3: Generate Video Prompts
            scenes_need_prompts = [s for s in project.scenes if not s.positive_prompt]
            if scenes_need_prompts:
                task = progress.add_task("[cyan]Generating prompts...", total=len(scenes_need_prompts))
                prompt_gen = PromptGenerator()
                for scene in scenes_need_prompts:
                    prompt_gen.generate_prompt(scene, project.director_brief)
                    progress.advance(task)
                console.print(f"  ✓ Generated {len(scenes_need_prompts)} video prompts")

            # Save project state
            if project_path:
                project.save(project_path)
                console.print(f"  ✓ Saved project to {project_path}")

            # Stage 4: Generate Videos
            if not skip_generation:
                model = video_model or settings.default_video_model
                console.print(f"\n[bold]Generating videos with {model}...[/bold]")
                
                generator = VideoGeneratorFactory.create(model)
                scenes_to_generate = [s for s in project.scenes if s.needs_generation]
                
                if scenes_to_generate:
                    task = progress.add_task(
                        f"[cyan]Generating {len(scenes_to_generate)} scenes...",
                        total=len(scenes_to_generate)
                    )

                    def progress_callback(current, total, status):
                        progress.update(task, completed=current)

                    output_dir = output_path.parent / "scenes"
                    asyncio.run(generate_scenes_sequential(
                        scenes_to_generate,
                        generator,
                        output_dir,
                        progress_callback,
                    ))

                    # Report results
                    complete = sum(1 for s in project.scenes if s.is_generated)
                    failed = sum(1 for s in project.scenes if s.status == SceneStatus.FAILED)
                    console.print(f"  ✓ Generated: {complete}/{len(project.scenes)}")
                    if failed:
                        console.print(f"  ⚠ Failed: {failed}")

                # Save project state again
                if project_path:
                    project.save(project_path)

                # Stage 5: Assemble Final Video
                if project.all_scenes_ready:
                    task = progress.add_task("[cyan]Assembling final video...", total=None)
                    assembler = VideoAssembler()
                    assembler.assemble(project, output_path)
                    progress.update(task, completed=True)
                    console.print(f"\n[bold green]✓ Video saved to: {output_path}[/bold green]")
                else:
                    console.print("\n[yellow]⚠ Not all scenes generated. Run again to retry failed scenes.[/yellow]")

            else:
                console.print("\n[yellow]Skipped video generation (--skip-generation)[/yellow]")

    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        logger.exception("Pipeline failed")
        sys.exit(1)

    # Show final summary
    _show_project_summary(project)


@cli.command()
@click.option("--audio", "-a", required=True, type=click.Path(exists=True), help="Path to audio file")
@click.option("--brief", "-b", required=True, type=click.Path(exists=True), help="Path to director's brief JSON")
@click.option("--output", "-o", required=True, type=click.Path(), help="Output JSON path for scene breakdown")
@click.option("--lyrics", "-l", type=click.Path(exists=True), default=None, help="Path to lyrics file (skips Whisper transcription)")
@click.option("--lyrics-text", type=str, default=None, help="Lyrics as text string (skips Whisper transcription)")
def analyze(audio: str, brief: str, output: str, lyrics: Optional[str], lyrics_text: Optional[str]):
    """Analyze audio and generate scene breakdown (no video generation)."""

    audio_path = Path(audio)
    brief_path = Path(brief)
    output_path = Path(output)

    console.print("[bold]Analyzing song...[/bold]")

    # Load brief
    director_brief = DirectorBrief.from_json_file(brief_path)

    # Create project
    project = Project.create(
        title=director_brief.title,
        audio_path=audio_path,
        director_brief=director_brief,
    )

    # Process audio
    console.print("  Processing audio...")
    audio_processor = AudioProcessor()

    # Check if manual lyrics provided
    manual_lyrics = None
    if lyrics_text:
        manual_lyrics = lyrics_text
    elif lyrics:
        lyrics_path = Path(lyrics)
        manual_lyrics = lyrics_path.read_text(encoding="utf-8")

    if manual_lyrics:
        console.print("  Using manual lyrics (skipping Whisper)")
        project = audio_processor.process_audio_with_lyrics(project, manual_lyrics)
        console.print(f"  ✓ Processed {len(project.lyrics_timed)} words from manual lyrics")
    else:
        project = audio_processor.process_audio(project)
        console.print(f"  ✓ Transcribed {len(project.lyrics_timed)} words")

    # Analyze story
    console.print("  Analyzing story...")
    analyzer = StoryAnalyzer()
    project = analyzer.analyze(project)
    console.print(f"  ✓ Generated {len(project.scenes)} scenes")

    # Generate prompts
    console.print("  Generating prompts...")
    prompt_gen = PromptGenerator()
    prompt_gen.generate_prompts_batch(project)
    console.print(f"  ✓ Generated prompts for all scenes")

    # Save project
    project.save(output_path)
    console.print(f"\n[bold green]✓ Saved to: {output_path}[/bold green]")

    _show_project_summary(project)


@cli.command()
@click.option("--project", "-p", required=True, type=click.Path(exists=True), help="Project JSON file")
@click.option("--scene", "-s", required=True, type=int, help="Scene number to regenerate")
@click.option("--direction", "-d", default=None, help="Additional direction for the scene")
def regenerate(project: str, scene: int, direction: Optional[str]):
    """Regenerate a specific scene."""
    
    project_path = Path(project)
    proj = Project.load(project_path)

    scene_obj = proj.get_scene_by_number(scene)
    if not scene_obj:
        console.print(f"[red]Scene {scene} not found[/red]")
        sys.exit(1)

    console.print(f"[bold]Regenerating scene {scene}...[/bold]")

    # Regenerate story analysis for this scene
    analyzer = StoryAnalyzer()
    scene_obj = analyzer.regenerate_scene(proj, scene, direction)

    # Regenerate prompt
    prompt_gen = PromptGenerator()
    prompt_gen.generate_prompt(scene_obj, proj.director_brief)

    # Save
    proj.save(project_path)
    console.print(f"[green]✓ Scene {scene} regenerated[/green]")
    console.print(f"  Narrative: {scene_obj.narrative_beat}")
    console.print(f"  Visual: {scene_obj.visual_description[:100]}...")


@cli.command()
@click.option("--project", "-p", required=True, type=click.Path(exists=True), help="Project JSON file")
@click.option("--audio", "-a", required=True, type=click.Path(exists=True), help="Original audio file")
@click.option("--output", "-o", required=True, type=click.Path(), help="Output video path")
def assemble(project: str, audio: str, output: str):
    """Assemble final video from generated scenes."""
    
    project_path = Path(project)
    audio_path = Path(audio)
    output_path = Path(output)

    proj = Project.load(project_path)
    proj.audio_path = audio_path

    if not proj.all_scenes_ready:
        console.print("[yellow]Warning: Not all scenes are ready[/yellow]")
        ready = sum(1 for s in proj.scenes if s.is_generated)
        console.print(f"  Ready: {ready}/{len(proj.scenes)}")

    console.print("[bold]Assembling video...[/bold]")
    assembler = VideoAssembler()
    assembler.assemble(proj, output_path)

    console.print(f"\n[bold green]✓ Video saved to: {output_path}[/bold green]")


@cli.command()
@click.option("--project", "-p", required=True, type=click.Path(exists=True), help="Project JSON file")
def status(project: str):
    """Show project status and scene breakdown."""
    
    project_path = Path(project)
    proj = Project.load(project_path)

    _show_project_summary(proj)


@cli.command()
def check():
    """Check system requirements and API configuration."""
    
    console.print("[bold]System Check[/bold]\n")

    # Check FFmpeg
    if check_ffmpeg_available():
        console.print("  ✓ FFmpeg installed")
    else:
        console.print("  ✗ FFmpeg not found")

    # Check API keys
    console.print("\n[bold]API Keys[/bold]")
    
    if settings.openai_api_key:
        console.print("  ✓ OpenAI API key configured")
    else:
        console.print("  ✗ OpenAI API key not set (needed for Whisper)")

    if settings.anthropic_api_key:
        console.print("  ✓ Anthropic API key configured")
    else:
        console.print("  ✗ Anthropic API key not set (needed for story analysis)")

    console.print("\n[bold]Video Generation APIs[/bold]")
    
    if settings.kling_api_key:
        console.print("  ✓ Kling API key configured")
    else:
        console.print("  ○ Kling API key not set")

    if settings.minimax_api_key:
        console.print("  ✓ Minimax API key configured")
    else:
        console.print("  ○ Minimax API key not set")

    if settings.runway_api_key:
        console.print("  ✓ Runway API key configured")
    else:
        console.print("  ○ Runway API key not set")

    console.print("\n[bold]fal.ai (Recommended)[/bold]")

    if settings.fal_api_key:
        console.print("  ✓ fal.ai API key configured")
        console.print("    Unlocks: Kling, MiniMax, Veo3, and more")
    else:
        console.print("  ○ fal.ai API key not set")
        console.print("    Get yours at: https://fal.ai/dashboard/keys")

    available = VideoGeneratorFactory.get_available_models()
    console.print(f"\n  Available models: {', '.join(available)}")


def _show_project_summary(project: Project):
    """Display project summary table."""
    
    console.print("\n")
    
    # Project info
    table = Table(title=f"Project: {project.title}")
    table.add_column("Property", style="cyan")
    table.add_column("Value")
    
    table.add_row("Status", project.status.value)
    table.add_row("Duration", f"{project.audio_duration:.1f}s" if project.audio_duration else "Unknown")
    table.add_row("Scenes", str(len(project.scenes)))
    table.add_row("Progress", f"{project.progress_percent:.0f}%")
    
    console.print(table)

    # Scene breakdown
    if project.scenes:
        console.print("\n")
        scene_table = Table(title="Scenes")
        scene_table.add_column("#", style="dim")
        scene_table.add_column("Time")
        scene_table.add_column("Type")
        scene_table.add_column("Status")
        scene_table.add_column("Narrative", max_width=50)

        status_colors = {
            SceneStatus.PENDING: "yellow",
            SceneStatus.GENERATING: "blue",
            SceneStatus.COMPLETE: "green",
            SceneStatus.APPROVED: "green",
            SceneStatus.FAILED: "red",
        }

        for scene in project.scenes:
            color = status_colors.get(scene.status, "white")
            scene_table.add_row(
                str(scene.scene_number),
                f"{scene.start_time:.1f}s - {scene.end_time:.1f}s",
                scene.section_type,
                f"[{color}]{scene.status.value}[/{color}]",
                scene.narrative_beat[:50] + "..." if len(scene.narrative_beat) > 50 else scene.narrative_beat,
            )

        console.print(scene_table)


if __name__ == "__main__":
    cli()
