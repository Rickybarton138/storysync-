# StorySync

**Turn your songs into cinematic music videos that tell your story.**

StorySync is an AI-powered music video generator that interprets your lyrics through a director's brief to create cohesive visual narratives — not just abstract visualisers or stock footage montages.

## Features

- **Lyric Analysis**: Automatic transcription with word-level timestamps
- **Story Interpretation**: AI understands the narrative in your lyrics
- **Director's Brief**: Provide context (setting, characters, tone) to guide visuals
- **Scene-by-Scene Generation**: Review and regenerate individual scenes
- **Cinematic Output**: Professional quality video synced to your music

## Quick Start (CLI)

```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys

# Generate a music video
python -m cli.main generate \
    --audio "path/to/song.mp3" \
    --brief "path/to/brief.json" \
    --output "output.mp4"
```

## Director's Brief Format

Create a JSON file describing your vision:

```json
{
  "title": "Song Title",
  "setting": {
    "location": "1940s rural Ireland",
    "time_period": "World War II era",
    "environment": "Rolling green hills, stone cottages"
  },
  "characters": [
    {
      "name": "Thomas",
      "description": "Young soldier, early 20s, dark hair",
      "role": "protagonist"
    }
  ],
  "tone": {
    "mood": "melancholic, hopeful undertones",
    "colour_palette": "muted greens, warm amber",
    "visual_style": "cinematic, 35mm film aesthetic"
  },
  "key_moments": [
    {
      "section": "chorus",
      "description": "Reunion at the train station"
    }
  ]
}
```

## Project Structure

```
storysync/
├── config/          # Configuration and prompts
├── core/            # Core processing modules
├── models/          # Data models
├── api/             # FastAPI backend (Phase 3)
├── workers/         # Background job processing
├── cli/             # Command-line interface
└── tests/           # Test suite
```

## Requirements

- Python 3.10+
- FFmpeg installed on system
- API keys for:
  - OpenAI (Whisper)
  - Anthropic (Claude)
  - Video generation (Kling/Minimax/Runway)

## License

Proprietary - All rights reserved

## Author

Ricky
