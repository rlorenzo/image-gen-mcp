"""Real image generation tests using actual API calls.

These tests make real API calls to test the complete functionality.
They require valid API keys and will incur costs.

Run with: pytest tests/integration/test_real_image_generation.py -v -s
"""

import asyncio
import json
from pathlib import Path

import pytest

from image_gen_mcp.config.settings import Settings
from image_gen_mcp.storage.manager import ImageStorageManager
from image_gen_mcp.tools.image_generation import ImageGenerationTool
from image_gen_mcp.types.enums import (
    ImageQuality,
    ImageSize,
    ImageStyle,
)
from image_gen_mcp.utils.cache import CacheManager


@pytest.mark.integration
@pytest.mark.slow
class TestRealImageGeneration:
    """Integration tests for real image generation."""

    @pytest.fixture
    async def real_tool(self, tmp_path):
        """Create a real image generation tool with actual settings."""
        settings = Settings()

        # Use project storage directory to persist generated images
        from image_gen_mcp.config.settings import StorageSettings
        from pathlib import Path

        project_storage = Path(__file__).parent.parent.parent / "storage"
        storage_settings = StorageSettings(
            base_path=str(project_storage),
            retention_days=30,
            max_size_gb=10.0,
            cleanup_interval_hours=24,
        )

        storage_manager = ImageStorageManager(settings=storage_settings)
        cache_manager = CacheManager(settings=settings.cache)

        tool = ImageGenerationTool(
            storage_manager=storage_manager,
            cache_manager=cache_manager,
            settings=settings,
        )

        yield tool

        # Cleanup
        if hasattr(storage_manager, 'close'):
            await storage_manager.close()
        if hasattr(cache_manager, 'close'):
            await cache_manager.close()

    async def test_list_available_models(self, real_tool):
        """Test listing available models."""
        print("\n" + "=" * 80)
        print("TEST: List Available Models")
        print("=" * 80)

        models_info = real_tool.get_supported_models()
        print(json.dumps(models_info, indent=2))

        assert isinstance(models_info, dict)
        assert "providers" in models_info or len(models_info) > 0

        available_providers = real_tool.get_available_providers()
        print(f"\nAvailable Providers: {available_providers}")
        assert isinstance(available_providers, list)

    async def test_basic_vivid_generation(self, real_tool):
        """Test basic image generation with vivid style."""
        print("\n" + "=" * 80)
        print("TEST: Basic Vivid Style Generation")
        print("=" * 80)

        prompt = (
            "A serene Japanese zen garden with cherry blossoms, "
            "koi pond, and traditional stone lanterns at sunset, "
            "digital art style"
        )
        print(f"Prompt: {prompt}")

        result = await real_tool.generate(
            prompt=prompt,
            style=ImageStyle.VIVID,
            quality=ImageQuality.AUTO,
            size=ImageSize.LANDSCAPE,
        )

        print(f"\nImage ID: {result['image_id']}")
        print(f"Task ID: {result['task_id']}")
        print(f"Image URL: {result['image_url']}")
        print(f"Resource URI: {result['resource_uri']}")
        print(f"\nMetadata:")
        print(json.dumps(result['metadata'], indent=2))

        # Assertions
        assert result['image_id']
        assert result['task_id']
        assert result['image_url']
        assert result['metadata']['style'] == 'vivid'
        assert result['metadata']['prompt'] == prompt

    async def test_natural_style_generation(self, real_tool):
        """Test natural style generation."""
        print("\n" + "=" * 80)
        print("TEST: Natural Style Generation")
        print("=" * 80)

        prompt = (
            "A cozy coffee shop interior with wooden furniture, "
            "warm lighting, plants, and people working on laptops"
        )
        print(f"Prompt: {prompt}")

        result = await real_tool.generate(
            prompt=prompt,
            style=ImageStyle.NATURAL,
            quality=ImageQuality.AUTO,
            size=ImageSize.LANDSCAPE,
        )

        print(f"\nImage ID: {result['image_id']}")
        print(f"Image URL: {result['image_url']}")
        print(f"Style: {result['metadata']['style']}")

        assert result['image_id']
        assert result['metadata']['style'] == 'natural'

    async def test_portrait_orientation(self, real_tool):
        """Test portrait orientation image."""
        print("\n" + "=" * 80)
        print("TEST: Portrait Orientation")
        print("=" * 80)

        prompt = (
            "A majestic snow leopard standing on a rocky mountain peak, "
            "dramatic clouds in background, wildlife photography style"
        )
        print(f"Prompt: {prompt}")

        result = await real_tool.generate(
            prompt=prompt,
            style=ImageStyle.VIVID,
            quality=ImageQuality.AUTO,
            size=ImageSize.PORTRAIT,
        )

        print(f"\nImage ID: {result['image_id']}")
        print(f"Size: {result['metadata']['size']}")

        assert result['image_id']
        assert result['metadata']['size'] == '1024x1536'

    async def test_square_size(self, real_tool):
        """Test square size image."""
        print("\n" + "=" * 80)
        print("TEST: Square Size")
        print("=" * 80)

        prompt = (
            "A futuristic cyberpunk city street at night with neon signs, "
            "flying cars, and rain-soaked pavement"
        )
        print(f"Prompt: {prompt}")

        result = await real_tool.generate(
            prompt=prompt,
            style=ImageStyle.VIVID,
            quality=ImageQuality.AUTO,
            size=ImageSize.SQUARE,
        )

        print(f"\nImage ID: {result['image_id']}")
        print(f"Size: {result['metadata']['size']}")

        assert result['image_id']
        assert result['metadata']['size'] == '1024x1024'

    async def test_high_quality(self, real_tool):
        """Test high quality setting."""
        print("\n" + "=" * 80)
        print("TEST: High Quality")
        print("=" * 80)

        prompt = (
            "An incredibly detailed macro photograph of a monarch butterfly "
            "on a purple flower, with water droplets and bokeh background"
        )
        print(f"Prompt: {prompt}")

        result = await real_tool.generate(
            prompt=prompt,
            style=ImageStyle.VIVID,
            quality=ImageQuality.HIGH,
            size=ImageSize.LANDSCAPE,
        )

        print(f"\nImage ID: {result['image_id']}")
        print(f"Quality: {result['metadata']['quality']}")
        print(f"Cost: ${result['metadata'].get('cost_estimate', 'N/A')}")

        assert result['image_id']
        assert result['metadata']['quality'] == 'high'


@pytest.mark.integration
@pytest.mark.slow
class TestCreativeUseCases:
    """Test various creative use case scenarios."""

    @pytest.fixture
    async def real_tool(self, tmp_path):
        """Create a real image generation tool with actual settings."""
        settings = Settings()

        # Use project storage directory to persist generated images
        from image_gen_mcp.config.settings import StorageSettings
        from pathlib import Path

        project_storage = Path(__file__).parent.parent.parent / "storage"
        storage_settings = StorageSettings(
            base_path=str(project_storage),
            retention_days=30,
            max_size_gb=10.0,
            cleanup_interval_hours=24,
        )

        storage_manager = ImageStorageManager(settings=storage_settings)
        cache_manager = CacheManager(settings=settings.cache)

        tool = ImageGenerationTool(
            storage_manager=storage_manager,
            cache_manager=cache_manager,
            settings=settings,
        )

        yield tool

        # Cleanup
        if hasattr(storage_manager, 'close'):
            await storage_manager.close()
        if hasattr(cache_manager, 'close'):
            await cache_manager.close()

    async def test_product_photography(self, real_tool):
        """Test product photography style."""
        print("\n" + "=" * 80)
        print("TEST: Product Photography")
        print("=" * 80)

        prompt = (
            "Professional product photography of a luxury smartwatch "
            "on a marble surface with dramatic studio lighting and reflections"
        )
        print(f"Prompt: {prompt}")

        result = await real_tool.generate(
            prompt=prompt,
            style=ImageStyle.VIVID,
            quality=ImageQuality.AUTO,
            size=ImageSize.LANDSCAPE,
        )

        print(f"\nImage ID: {result['image_id']}")
        print(f"URL: {result['image_url']}")

        assert result['image_id']
        assert result['metadata']['style'] == 'vivid'

    async def test_social_media_graphic(self, real_tool):
        """Test social media graphic generation."""
        print("\n" + "=" * 80)
        print("TEST: Social Media Graphic")
        print("=" * 80)

        prompt = (
            "Modern minimalist Instagram post background with pastel gradient, "
            "geometric shapes, and space for text overlay"
        )
        print(f"Prompt: {prompt}")

        result = await real_tool.generate(
            prompt=prompt,
            style=ImageStyle.NATURAL,
            quality=ImageQuality.AUTO,
            size=ImageSize.SQUARE,
        )

        print(f"\nImage ID: {result['image_id']}")
        print(f"URL: {result['image_url']}")

        assert result['image_id']
        assert result['metadata']['size'] == '1024x1024'

    async def test_concept_art(self, real_tool):
        """Test fantasy concept art."""
        print("\n" + "=" * 80)
        print("TEST: Fantasy Concept Art")
        print("=" * 80)

        prompt = (
            "Fantasy concept art of a floating island with waterfalls, "
            "ancient ruins, and bioluminescent plants, magical atmosphere"
        )
        print(f"Prompt: {prompt}")

        result = await real_tool.generate(
            prompt=prompt,
            style=ImageStyle.VIVID,
            quality=ImageQuality.AUTO,
            size=ImageSize.LANDSCAPE,
        )

        print(f"\nImage ID: {result['image_id']}")
        print(f"URL: {result['image_url']}")

        assert result['image_id']

    async def test_batch_generation(self, real_tool):
        """Test generating multiple images in sequence."""
        print("\n" + "=" * 80)
        print("TEST: Batch Generation")
        print("=" * 80)

        prompts = [
            "A peaceful mountain lake at sunrise with mist",
            "A modern workspace with plants and natural light",
            "Abstract geometric pattern in blue and gold",
        ]

        results = []
        for i, prompt in enumerate(prompts, 1):
            print(f"\n{i}. Generating: {prompt}")

            result = await real_tool.generate(
                prompt=prompt,
                style=ImageStyle.VIVID,
                quality=ImageQuality.AUTO,
                size=ImageSize.LANDSCAPE,
            )

            print(f"   Image ID: {result['image_id']}")
            results.append(result)

        assert len(results) == len(prompts)
        assert all(r['image_id'] for r in results)

        # Summary
        total_cost = sum(
            r['metadata'].get('cost_estimate', 0) for r in results
        )
        print(f"\nTotal Cost: ${total_cost:.4f}")


if __name__ == "__main__":
    """Allow running tests directly with python."""
    pytest.main([__file__, "-v", "-s"])
