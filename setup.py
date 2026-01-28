"""StorySync package setup."""

from setuptools import setup, find_packages

setup(
    name="storysync",
    version="0.1.0",
    description="Turn your songs into cinematic music videos",
    author="Ricky",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "python-dotenv>=1.0.0",
        "pydantic>=2.0.0",
        "pydantic-settings>=2.0.0",
        "openai>=1.0.0",
        "anthropic>=0.18.0",
        "httpx>=0.25.0",
        "click>=8.1.0",
        "rich>=13.0.0",
        "ffmpeg-python>=0.2.0",
    ],
    extras_require={
        "dev": [
            "pytest>=8.0.0",
            "pytest-asyncio>=0.23.0",
            "black>=24.0.0",
            "ruff>=0.2.0",
            "mypy>=1.8.0",
        ],
        "audio": [
            "pydub>=0.25.1",
            "librosa>=0.10.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "storysync=cli.main:cli",
        ],
    },
)
