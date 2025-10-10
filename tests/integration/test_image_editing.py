"""Image editing functionality tests.

These tests verify the image editing capabilities including:
- Simple edits with text instructions
- Masked editing for targeted changes
- Quality and format options

Run with: pytest tests/integration/test_image_editing.py -v -s
"""

import base64
import json
from pathlib import Path

import pytest

from image_gen_mcp.config.settings import Settings, StorageSettings
from image_gen_mcp.storage.manager import ImageStorageManager
from image_gen_mcp.tools.image_editing import ImageEditingTool
from image_gen_mcp.utils.cache import CacheManager
from image_gen_mcp.utils.openai_client import OpenAIClientManager


@pytest.mark.integration
@pytest.mark.slow
class TestImageEditing:
    """Test image editing with OpenAI's image edit API."""

    @pytest.fixture
    async def editing_setup(self, tmp_path):
        """Create image editing tool and find existing images."""
        settings = Settings()

        # Use project storage directory
        project_storage = Path(__file__).parent.parent.parent / "storage"
        storage_settings = StorageSettings(
            base_path=str(project_storage),
            retention_days=30,
            max_size_gb=10.0,
            cleanup_interval_hours=24,
        )

        storage_manager = ImageStorageManager(settings=storage_settings)
        cache_manager = CacheManager(settings=settings.cache)

        # Create OpenAI client for editing
        openai_client = OpenAIClientManager(settings.providers.openai)
        # Initialize the client by accessing the property
        _ = openai_client.client

        tool = ImageEditingTool(
            storage_manager=storage_manager,
            cache_manager=cache_manager,
            settings=settings,
            openai_client=openai_client,
        )

        # Find existing images from today's generation
        today_dir = project_storage / "images" / "2025" / "10" / "10"
        image_files = list(today_dir.glob("*.png")) if today_dir.exists() else []

        yield tool, storage_manager, image_files

        # Cleanup
        if hasattr(storage_manager, 'close'):
            await storage_manager.close()
        if hasattr(cache_manager, 'close'):
            await cache_manager.close()

    def _load_image_as_base64(self, image_path: Path) -> str:
        """Load image and convert to base64."""
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        return base64.b64encode(image_bytes).decode()

    def _load_metadata(self, image_path: Path) -> dict:
        """Load image metadata."""
        json_path = image_path.with_suffix('.json')
        if json_path.exists():
            with open(json_path, 'r') as f:
                return json.load(f)
        return {}

    async def test_add_lighting_effect(self, editing_setup):
        """Test editing: Add lighting effect to an image."""
        print("\n" + "=" * 80)
        print("TEST: Add Golden Sunlight Effect")
        print("=" * 80)

        tool, storage_manager, image_files = editing_setup

        if not image_files:
            pytest.skip("No existing images found to edit")

        # Use the Japanese zen garden image (first one)
        source_image = image_files[0]
        source_meta = self._load_metadata(source_image)

        print(f"\nSource Image: {source_image.name}")
        print(f"Original Prompt: {source_meta.get('prompt', 'N/A')[:80]}...")

        # Load image
        image_b64 = self._load_image_as_base64(source_image)
        image_data = f"data:image/png;base64,{image_b64}"

        # Edit: Add golden sunlight
        edit_prompt = "Add warm golden sunlight streaming through the scene with beautiful light rays"

        print(f"\nEdit Instruction: {edit_prompt}")
        print("Processing edit...")

        result = await tool.edit(
            image_data=image_data,
            prompt=edit_prompt,
            size="1024x1024",
            quality="auto",
        )

        print(f"\n✅ Edit Complete!")
        print(f"Edited Image ID: {result['image_id']}")
        print(f"Image URL: {result['image_url']}")
        print(f"Cost: ${result['metadata']['cost_estimate']:.4f}")

        assert result['image_id']
        assert result['operation'] == 'edit'
        assert result['metadata']['has_mask'] is False

    async def test_change_time_of_day(self, editing_setup):
        """Test editing: Change time of day."""
        print("\n" + "=" * 80)
        print("TEST: Change Time of Day to Night")
        print("=" * 80)

        tool, storage_manager, image_files = editing_setup

        if len(image_files) < 2:
            pytest.skip("Need at least 2 images")

        # Use the coffee shop image
        source_image = image_files[1]
        source_meta = self._load_metadata(source_image)

        print(f"\nSource Image: {source_image.name}")
        print(f"Original Prompt: {source_meta.get('prompt', 'N/A')[:80]}...")

        # Load image
        image_b64 = self._load_image_as_base64(source_image)
        image_data = f"data:image/png;base64,{image_b64}"

        # Edit: Change to night scene
        edit_prompt = "Transform to a cozy night scene with warm interior lighting, dark windows showing nighttime outside"

        print(f"\nEdit Instruction: {edit_prompt}")
        print("Processing edit...")

        result = await tool.edit(
            image_data=image_data,
            prompt=edit_prompt,
            size="1024x1024",
            quality="auto",
        )

        print(f"\n✅ Edit Complete!")
        print(f"Edited Image ID: {result['image_id']}")
        print(f"Image URL: {result['image_url']}")

        assert result['image_id']
        assert result['operation'] == 'edit'

    async def test_add_weather_effect(self, editing_setup):
        """Test editing: Add weather effects."""
        print("\n" + "=" * 80)
        print("TEST: Add Snow and Winter Atmosphere")
        print("=" * 80)

        tool, storage_manager, image_files = editing_setup

        if len(image_files) < 3:
            pytest.skip("Need at least 3 images")

        # Use the snow leopard image (portrait)
        source_image = image_files[2]
        source_meta = self._load_metadata(source_image)

        print(f"\nSource Image: {source_image.name}")
        print(f"Original Prompt: {source_meta.get('prompt', 'N/A')[:80]}...")

        # Load image
        image_b64 = self._load_image_as_base64(source_image)
        image_data = f"data:image/png;base64,{image_b64}"

        # Edit: Add falling snow
        edit_prompt = "Add gentle falling snow and frost effects to create a winter atmosphere"

        print(f"\nEdit Instruction: {edit_prompt}")
        print("Processing edit...")

        result = await tool.edit(
            image_data=image_data,
            prompt=edit_prompt,
            size="1024x1024",
            quality="high",
        )

        print(f"\n✅ Edit Complete!")
        print(f"Edited Image ID: {result['image_id']}")
        print(f"Image URL: {result['image_url']}")
        print(f"Quality: {result['metadata']['quality']}")

        assert result['image_id']
        assert result['metadata']['quality'] == 'high'

    async def test_enhance_cyberpunk_atmosphere(self, editing_setup):
        """Test editing: Enhance cyberpunk visual effects."""
        print("\n" + "=" * 80)
        print("TEST: Enhance Cyberpunk Neon Effects")
        print("=" * 80)

        tool, storage_manager, image_files = editing_setup

        if len(image_files) < 4:
            pytest.skip("Need at least 4 images")

        # Use the cyberpunk city image
        source_image = image_files[3]
        source_meta = self._load_metadata(source_image)

        print(f"\nSource Image: {source_image.name}")
        print(f"Original Prompt: {source_meta.get('prompt', 'N/A')[:80]}...")

        # Load image
        image_b64 = self._load_image_as_base64(source_image)
        image_data = f"data:image/png;base64,{image_b64}"

        # Edit: Enhance neon
        edit_prompt = "Intensify the neon lights and add more holographic advertisements with vibrant purple and cyan colors"

        print(f"\nEdit Instruction: {edit_prompt}")
        print("Processing edit...")

        result = await tool.edit(
            image_data=image_data,
            prompt=edit_prompt,
            size="1024x1024",
            quality="high",
        )

        print(f"\n✅ Edit Complete!")
        print(f"Edited Image ID: {result['image_id']}")
        print(f"Image URL: {result['image_url']}")

        assert result['image_id']

    async def test_add_magical_elements(self, editing_setup):
        """Test editing: Add fantasy elements."""
        print("\n" + "=" * 80)
        print("TEST: Add Magical Sparkles and Glow")
        print("=" * 80)

        tool, storage_manager, image_files = editing_setup

        if len(image_files) < 5:
            pytest.skip("Need at least 5 images")

        # Use the butterfly macro image
        source_image = image_files[4]
        source_meta = self._load_metadata(source_image)

        print(f"\nSource Image: {source_image.name}")
        print(f"Original Prompt: {source_meta.get('prompt', 'N/A')[:80]}...")

        # Load image
        image_b64 = self._load_image_as_base64(source_image)
        image_data = f"data:image/png;base64,{image_b64}"

        # Edit: Add magical effects
        edit_prompt = "Add magical sparkles and ethereal glow around the butterfly wings"

        print(f"\nEdit Instruction: {edit_prompt}")
        print("Processing edit...")

        result = await tool.edit(
            image_data=image_data,
            prompt=edit_prompt,
            size="1024x1024",
            quality="high",
        )

        print(f"\n✅ Edit Complete!")
        print(f"Edited Image ID: {result['image_id']}")
        print(f"Image URL: {result['image_url']}")

        assert result['image_id']


@pytest.mark.integration
@pytest.mark.slow
class TestImageEditingBatch:
    """Test batch editing operations."""

    @pytest.fixture
    async def editing_setup(self, tmp_path):
        """Create image editing tool."""
        settings = Settings()

        project_storage = Path(__file__).parent.parent.parent / "storage"
        storage_settings = StorageSettings(
            base_path=str(project_storage),
            retention_days=30,
            max_size_gb=10.0,
            cleanup_interval_hours=24,
        )

        storage_manager = ImageStorageManager(settings=storage_settings)
        cache_manager = CacheManager(settings=settings.cache)
        openai_client = OpenAIClientManager(settings.providers.openai)
        # Initialize the client
        _ = openai_client.client

        tool = ImageEditingTool(
            storage_manager=storage_manager,
            cache_manager=cache_manager,
            settings=settings,
            openai_client=openai_client,
        )

        today_dir = project_storage / "images" / "2025" / "10" / "10"
        image_files = list(today_dir.glob("*.png")) if today_dir.exists() else []

        yield tool, storage_manager, image_files

        if hasattr(storage_manager, 'close'):
            await storage_manager.close()
        if hasattr(cache_manager, 'close'):
            await cache_manager.close()

    async def test_multiple_edits_summary(self, editing_setup):
        """Test and summarize multiple editing operations."""
        print("\n" + "=" * 80)
        print("TEST: Multiple Image Edits Summary")
        print("=" * 80)

        tool, storage_manager, image_files = editing_setup

        if len(image_files) < 3:
            pytest.skip("Need at least 3 images for batch editing")

        # This test just verifies the setup is working
        # Actual batch edits are done in individual tests above

        print(f"\n📊 Available images for editing: {len(image_files)}")
        print(f"✅ Editing tool initialized successfully")
        print(f"✅ Storage manager ready")
        print(f"✅ Ready for batch editing operations")

        assert len(image_files) > 0
        assert tool is not None


if __name__ == "__main__":
    """Allow running tests directly with python."""
    pytest.main([__file__, "-v", "-s"])
