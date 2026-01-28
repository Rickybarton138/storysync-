"""Video generator module - handles video generation API calls."""

import asyncio
import logging
import os
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Callable, Literal
import httpx

from config.settings import settings
from models.scene import Scene, SceneStatus

logger = logging.getLogger(__name__)

# Fal.ai model endpoint mappings
FAL_MODEL_ENDPOINTS = {
    "kling": "fal-ai/kling-video/v2.1/master/text-to-video",
    "kling-standard": "fal-ai/kling-video/v1/standard/text-to-video",
    "kling-pro": "fal-ai/kling-video/v1/pro/text-to-video",
    "minimax": "fal-ai/minimax/video-01",
    "veo3": "fal-ai/veo3",
}


class VideoGenerator(ABC):
    """Abstract base class for video generators."""

    @abstractmethod
    async def generate(
        self,
        scene: Scene,
        output_dir: Path,
    ) -> Scene:
        """
        Generate video for a scene.

        Args:
            scene: Scene with prompts populated
            output_dir: Directory to save the video

        Returns:
            Scene with video_path updated
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Return the name of this generator."""
        pass


class KlingGenerator(VideoGenerator):
    """Video generator using Kling API."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.kling_api_key
        self.base_url = "https://api.klingai.com/v1"
        self.poll_interval = 5  # seconds
        self.max_wait_time = 300  # 5 minutes

    def get_name(self) -> str:
        return "kling"

    async def generate(
        self,
        scene: Scene,
        output_dir: Path,
    ) -> Scene:
        """Generate video using Kling API."""
        scene.mark_generating()

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                # Submit generation request
                task_id = await self._submit_task(client, scene)

                if not task_id:
                    raise Exception("Failed to get task ID from Kling API")

                # Poll for completion
                video_url = await self._poll_for_result(client, task_id)

                if not video_url:
                    raise Exception("Video generation timed out or failed")

                # Download video
                video_path = await self._download_video(client, video_url, output_dir, scene)

                scene.mark_complete(video_path)

        except Exception as e:
            logger.error(f"Kling generation failed for scene {scene.scene_number}: {e}")
            scene.mark_failed(str(e))

        return scene

    async def _submit_task(self, client: httpx.AsyncClient, scene: Scene) -> Optional[str]:
        """Submit video generation task to Kling."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "prompt": scene.positive_prompt,
            "negative_prompt": scene.negative_prompt,
            "duration": min(scene.duration, 10.0),  # Kling caps at 10s
            "aspect_ratio": "16:9",
        }

        # Add camera motion if supported
        if scene.suggested_camera_motion and scene.suggested_camera_motion != "static":
            payload["camera_motion"] = scene.suggested_camera_motion

        response = await client.post(
            f"{self.base_url}/videos/text-to-video",
            headers=headers,
            json=payload,
        )

        if response.status_code != 200:
            logger.error(f"Kling API error: {response.status_code} - {response.text}")
            return None

        data = response.json()
        return data.get("task_id") or data.get("id")

    async def _poll_for_result(
        self,
        client: httpx.AsyncClient,
        task_id: str,
    ) -> Optional[str]:
        """Poll Kling API until video is ready."""
        headers = {"Authorization": f"Bearer {self.api_key}"}

        start_time = time.time()
        while time.time() - start_time < self.max_wait_time:
            response = await client.get(
                f"{self.base_url}/videos/{task_id}",
                headers=headers,
            )

            if response.status_code != 200:
                logger.warning(f"Poll request failed: {response.status_code}")
                await asyncio.sleep(self.poll_interval)
                continue

            data = response.json()
            status = data.get("status", "").lower()

            if status == "completed" or status == "success":
                return data.get("video_url") or data.get("output", {}).get("video_url")

            if status in ("failed", "error"):
                logger.error(f"Kling task failed: {data}")
                return None

            logger.debug(f"Task {task_id} status: {status}")
            await asyncio.sleep(self.poll_interval)

        logger.error(f"Polling timed out for task {task_id}")
        return None

    async def _download_video(
        self,
        client: httpx.AsyncClient,
        video_url: str,
        output_dir: Path,
        scene: Scene,
    ) -> Path:
        """Download video from URL."""
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"scene_{scene.scene_number:03d}.mp4"

        response = await client.get(video_url)
        response.raise_for_status()

        with open(output_path, "wb") as f:
            f.write(response.content)

        logger.info(f"Downloaded scene {scene.scene_number} to {output_path}")
        return output_path


class MinimaxGenerator(VideoGenerator):
    """Video generator using Minimax API."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.minimax_api_key
        self.base_url = "https://api.minimax.chat/v1"
        self.poll_interval = 5
        self.max_wait_time = 300

    def get_name(self) -> str:
        return "minimax"

    async def generate(
        self,
        scene: Scene,
        output_dir: Path,
    ) -> Scene:
        """Generate video using Minimax API."""
        scene.mark_generating()

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                # Submit generation request
                task_id = await self._submit_task(client, scene)

                if not task_id:
                    raise Exception("Failed to get task ID from Minimax API")

                # Poll for completion
                video_url = await self._poll_for_result(client, task_id)

                if not video_url:
                    raise Exception("Video generation timed out or failed")

                # Download video
                video_path = await self._download_video(client, video_url, output_dir, scene)

                scene.mark_complete(video_path)

        except Exception as e:
            logger.error(f"Minimax generation failed for scene {scene.scene_number}: {e}")
            scene.mark_failed(str(e))

        return scene

    async def _submit_task(self, client: httpx.AsyncClient, scene: Scene) -> Optional[str]:
        """Submit video generation task to Minimax."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": "video-01",
            "prompt": scene.positive_prompt,
        }

        response = await client.post(
            f"{self.base_url}/video/generation",
            headers=headers,
            json=payload,
        )

        if response.status_code != 200:
            logger.error(f"Minimax API error: {response.status_code} - {response.text}")
            return None

        data = response.json()
        return data.get("task_id")

    async def _poll_for_result(
        self,
        client: httpx.AsyncClient,
        task_id: str,
    ) -> Optional[str]:
        """Poll Minimax API until video is ready."""
        headers = {"Authorization": f"Bearer {self.api_key}"}

        start_time = time.time()
        while time.time() - start_time < self.max_wait_time:
            response = await client.get(
                f"{self.base_url}/video/generation/{task_id}",
                headers=headers,
            )

            if response.status_code != 200:
                await asyncio.sleep(self.poll_interval)
                continue

            data = response.json()
            status = data.get("status", "").lower()

            if status == "success":
                return data.get("file_id")  # Minimax returns file_id

            if status == "failed":
                return None

            await asyncio.sleep(self.poll_interval)

        return None

    async def _download_video(
        self,
        client: httpx.AsyncClient,
        file_id: str,
        output_dir: Path,
        scene: Scene,
    ) -> Path:
        """Download video from Minimax."""
        headers = {"Authorization": f"Bearer {self.api_key}"}

        # Get download URL
        response = await client.get(
            f"{self.base_url}/files/{file_id}",
            headers=headers,
        )
        data = response.json()
        video_url = data.get("download_url")

        # Download
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"scene_{scene.scene_number:03d}.mp4"

        response = await client.get(video_url)
        response.raise_for_status()

        with open(output_path, "wb") as f:
            f.write(response.content)

        return output_path


class FalGenerator(VideoGenerator):
    """
    Video generator using fal.ai API.

    Supports multiple video models through fal.ai's unified API:
    - Kling (v1 standard/pro, v2.1 master)
    - MiniMax (Hailuo AI)
    - Veo3 (Google)

    Uses the fal-client Python SDK for API communication.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Literal["kling", "kling-standard", "kling-pro", "minimax", "veo3"] = "kling",
    ):
        self.api_key = api_key or settings.fal_api_key
        self.model = model
        self.endpoint = FAL_MODEL_ENDPOINTS.get(model, FAL_MODEL_ENDPOINTS["kling"])
        self._fal_client = None

    def get_name(self) -> str:
        return f"fal-{self.model}"

    def _get_client(self):
        """Lazily import and configure fal_client."""
        if self._fal_client is None:
            try:
                import fal_client
            except ImportError:
                raise ImportError(
                    "fal-client is required for fal.ai video generation. "
                    "Install it with: pip install fal-client"
                )

            # Set the API key in environment for fal_client
            if self.api_key:
                os.environ["FAL_KEY"] = self.api_key

            self._fal_client = fal_client
        return self._fal_client

    def _build_request_args(self, scene: Scene) -> dict:
        """Build request arguments based on the model type."""
        # Base arguments common to all models
        args = {
            "prompt": scene.positive_prompt,
        }

        # Model-specific parameters
        if self.model.startswith("kling"):
            # Kling supports negative prompts, duration, aspect ratio, cfg_scale
            if scene.negative_prompt:
                args["negative_prompt"] = scene.negative_prompt

            # Duration: Kling supports 5 or 10 seconds
            duration = min(scene.duration, 10.0)
            args["duration"] = "10" if duration > 5 else "5"

            args["aspect_ratio"] = "16:9"
            args["cfg_scale"] = 0.5

        elif self.model == "minimax":
            # MiniMax has simpler parameters
            # Camera movements can be embedded in prompt with [Pan left], [Zoom in], etc.
            if scene.suggested_camera_motion and scene.suggested_camera_motion != "static":
                motion_map = {
                    "pan_left": "[Pan left]",
                    "pan_right": "[Pan right]",
                    "zoom_in": "[Zoom in]",
                    "zoom_out": "[Zoom out]",
                    "tilt_up": "[Tilt up]",
                    "tilt_down": "[Tilt down]",
                    "dolly_in": "[Push in]",
                    "dolly_out": "[Pull out]",
                }
                motion_bracket = motion_map.get(scene.suggested_camera_motion, "")
                if motion_bracket:
                    args["prompt"] = f"{motion_bracket} {args['prompt']}"

            args["prompt_optimizer"] = True

        elif self.model == "veo3":
            # Veo3 parameters
            args["aspect_ratio"] = "16:9"
            if scene.negative_prompt:
                args["negative_prompt"] = scene.negative_prompt

        return args

    async def generate(
        self,
        scene: Scene,
        output_dir: Path,
    ) -> Scene:
        """Generate video using fal.ai API."""
        scene.mark_generating()

        try:
            fal_client = self._get_client()

            # Build request arguments
            request_args = self._build_request_args(scene)

            logger.info(
                f"Submitting scene {scene.scene_number} to fal.ai ({self.endpoint})"
            )
            logger.debug(f"Request args: {request_args}")

            # Use subscribe for automatic polling and result retrieval
            # This handles the queue submission and polling internally
            result = await asyncio.to_thread(
                fal_client.subscribe,
                self.endpoint,
                arguments=request_args,
            )

            # Extract video URL from response
            video_url = None
            if isinstance(result, dict):
                if "video" in result:
                    video_url = result["video"].get("url")
                elif "video_url" in result:
                    video_url = result["video_url"]
                elif "output" in result:
                    video_url = result["output"].get("video", {}).get("url")

            if not video_url:
                raise Exception(f"No video URL in fal.ai response: {result}")

            # Download the video
            video_path = await self._download_video(video_url, output_dir, scene)

            scene.mark_complete(video_path)
            logger.info(f"Scene {scene.scene_number} generated successfully via fal.ai")

        except Exception as e:
            logger.error(f"fal.ai generation failed for scene {scene.scene_number}: {e}")
            scene.mark_failed(str(e))

        return scene

    async def _download_video(
        self,
        video_url: str,
        output_dir: Path,
        scene: Scene,
    ) -> Path:
        """Download video from fal.ai storage."""
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"scene_{scene.scene_number:03d}.mp4"

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.get(video_url)
            response.raise_for_status()

            with open(output_path, "wb") as f:
                f.write(response.content)

        logger.info(f"Downloaded scene {scene.scene_number} to {output_path}")
        return output_path


class FalAsyncGenerator(VideoGenerator):
    """
    Async-native fal.ai generator using the queue API directly.

    Provides more control over the generation process with explicit
    queue submission, status polling, and result retrieval.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Literal["kling", "kling-standard", "kling-pro", "minimax", "veo3"] = "kling",
        poll_interval: float = 3.0,
        max_wait_time: float = 600.0,
    ):
        self.api_key = api_key or settings.fal_api_key
        self.model = model
        self.endpoint = FAL_MODEL_ENDPOINTS.get(model, FAL_MODEL_ENDPOINTS["kling"])
        self.poll_interval = poll_interval
        self.max_wait_time = max_wait_time
        self._fal_client = None

    def get_name(self) -> str:
        return f"fal-async-{self.model}"

    def _get_client(self):
        """Lazily import and configure fal_client."""
        if self._fal_client is None:
            try:
                import fal_client
            except ImportError:
                raise ImportError(
                    "fal-client is required for fal.ai video generation. "
                    "Install it with: pip install fal-client"
                )

            if self.api_key:
                os.environ["FAL_KEY"] = self.api_key

            self._fal_client = fal_client
        return self._fal_client

    def _build_request_args(self, scene: Scene) -> dict:
        """Build request arguments (same as FalGenerator)."""
        args = {"prompt": scene.positive_prompt}

        if self.model.startswith("kling"):
            if scene.negative_prompt:
                args["negative_prompt"] = scene.negative_prompt
            duration = min(scene.duration, 10.0)
            args["duration"] = "10" if duration > 5 else "5"
            args["aspect_ratio"] = "16:9"
            args["cfg_scale"] = 0.5
        elif self.model == "minimax":
            args["prompt_optimizer"] = True
        elif self.model == "veo3":
            args["aspect_ratio"] = "16:9"

        return args

    async def generate(
        self,
        scene: Scene,
        output_dir: Path,
    ) -> Scene:
        """Generate video using fal.ai async queue API."""
        scene.mark_generating()

        try:
            fal_client = self._get_client()
            request_args = self._build_request_args(scene)

            logger.info(f"Submitting scene {scene.scene_number} to fal.ai queue")

            # Submit to queue asynchronously
            response = await fal_client.submit_async(
                self.endpoint,
                arguments=request_args,
            )

            # Poll for completion using event iteration
            result = None
            start_time = time.time()

            async for event in response.iter_events(with_logs=True):
                if time.time() - start_time > self.max_wait_time:
                    raise TimeoutError(
                        f"Video generation timed out after {self.max_wait_time}s"
                    )

                if hasattr(fal_client, 'Queued') and isinstance(event, fal_client.Queued):
                    logger.debug(f"Scene {scene.scene_number} queued at position {event.position}")
                elif hasattr(fal_client, 'InProgress') and isinstance(event, fal_client.InProgress):
                    for log in getattr(event, 'logs', []):
                        logger.debug(f"fal.ai: {log.get('message', log)}")
                elif hasattr(fal_client, 'Completed') and isinstance(event, fal_client.Completed):
                    logger.info(f"Scene {scene.scene_number} completed on fal.ai")

            # Get the final result
            result = await response.get()

            # Extract video URL
            video_url = None
            if isinstance(result, dict):
                if "video" in result:
                    video_url = result["video"].get("url")
                elif "video_url" in result:
                    video_url = result["video_url"]

            if not video_url:
                raise Exception(f"No video URL in response: {result}")

            # Download video
            video_path = await self._download_video(video_url, output_dir, scene)
            scene.mark_complete(video_path)

        except Exception as e:
            logger.error(f"fal.ai async generation failed for scene {scene.scene_number}: {e}")
            scene.mark_failed(str(e))

        return scene

    async def _download_video(
        self,
        video_url: str,
        output_dir: Path,
        scene: Scene,
    ) -> Path:
        """Download video from URL."""
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"scene_{scene.scene_number:03d}.mp4"

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.get(video_url)
            response.raise_for_status()

            with open(output_path, "wb") as f:
                f.write(response.content)

        return output_path


class MockGenerator(VideoGenerator):
    """Mock video generator for testing without API calls."""

    def __init__(self, delay: float = 2.0):
        self.delay = delay

    def get_name(self) -> str:
        return "mock"

    async def generate(
        self,
        scene: Scene,
        output_dir: Path,
    ) -> Scene:
        """Simulate video generation."""
        scene.mark_generating()

        logger.info(f"[MOCK] Generating scene {scene.scene_number}...")
        await asyncio.sleep(self.delay)

        # Create a placeholder file
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"scene_{scene.scene_number:03d}_mock.mp4"

        # Write a small placeholder (in real use, this would be actual video)
        with open(output_path, "wb") as f:
            f.write(b"MOCK VIDEO FILE")

        scene.mark_complete(output_path)
        logger.info(f"[MOCK] Scene {scene.scene_number} complete")

        return scene


class VideoGeneratorFactory:
    """Factory for creating video generators."""

    @staticmethod
    def create(
        model: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> VideoGenerator:
        """
        Create a video generator for the specified model.

        Args:
            model: Model name. Options:
                - Direct API: kling, minimax, runway
                - fal.ai: fal-kling, fal-minimax, fal-veo3
                - Testing: mock
            api_key: Optional API key override

        Returns:
            VideoGenerator instance
        """
        model = model or settings.default_video_model

        # Direct API generators
        if model == "kling":
            return KlingGenerator(api_key)
        elif model == "minimax":
            return MinimaxGenerator(api_key)

        # fal.ai generators (recommended)
        elif model == "fal-kling":
            return FalGenerator(api_key, model="kling")
        elif model == "fal-kling-standard":
            return FalGenerator(api_key, model="kling-standard")
        elif model == "fal-kling-pro":
            return FalGenerator(api_key, model="kling-pro")
        elif model == "fal-minimax":
            return FalGenerator(api_key, model="minimax")
        elif model == "fal-veo3":
            return FalGenerator(api_key, model="veo3")

        # Async fal.ai generators (for more control)
        elif model == "fal-async-kling":
            return FalAsyncGenerator(api_key, model="kling")
        elif model == "fal-async-minimax":
            return FalAsyncGenerator(api_key, model="minimax")

        # Testing
        elif model == "mock":
            return MockGenerator()
        else:
            raise ValueError(
                f"Unknown video model: {model}. "
                f"Available: kling, minimax, fal-kling, fal-minimax, fal-veo3, mock"
            )

    @staticmethod
    def get_available_models() -> list[str]:
        """Get list of models with configured API keys."""
        available = []

        # Direct API models
        if settings.kling_api_key:
            available.append("kling")
        if settings.minimax_api_key:
            available.append("minimax")
        if settings.runway_api_key:
            available.append("runway")

        # fal.ai models (single key unlocks multiple models)
        if settings.fal_api_key:
            available.extend([
                "fal-kling",
                "fal-kling-standard",
                "fal-kling-pro",
                "fal-minimax",
                "fal-veo3",
            ])

        available.append("mock")  # Always available
        return available


async def generate_scenes_parallel(
    scenes: list[Scene],
    generator: VideoGenerator,
    output_dir: Path,
    max_parallel: int = 3,
) -> list[Scene]:
    """
    Generate videos for multiple scenes in parallel.

    Args:
        scenes: List of scenes to generate
        generator: Video generator to use
        output_dir: Directory for output files
        max_parallel: Maximum concurrent generations

    Returns:
        List of scenes with updated status
    """
    semaphore = asyncio.Semaphore(max_parallel)

    async def generate_with_semaphore(scene: Scene) -> Scene:
        async with semaphore:
            return await generator.generate(scene, output_dir)

    # Generate all scenes concurrently (limited by semaphore)
    tasks = [generate_with_semaphore(scene) for scene in scenes]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Handle any exceptions
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error(f"Scene {scenes[i].scene_number} failed: {result}")
            scenes[i].mark_failed(str(result))

    return scenes


async def generate_scenes_sequential(
    scenes: list[Scene],
    generator: VideoGenerator,
    output_dir: Path,
    progress_callback: Optional[Callable] = None,
) -> list[Scene]:
    """
    Generate videos for scenes one at a time.

    Args:
        scenes: List of scenes to generate
        generator: Video generator to use
        output_dir: Directory for output files
        progress_callback: Optional callback(scene_number, total, status)

    Returns:
        List of scenes with updated status
    """
    total = len(scenes)

    for i, scene in enumerate(scenes, 1):
        if progress_callback:
            progress_callback(i, total, "generating")

        await generator.generate(scene, output_dir)

        if progress_callback:
            status = "complete" if scene.is_generated else "failed"
            progress_callback(i, total, status)

    return scenes
