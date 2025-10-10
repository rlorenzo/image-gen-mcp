"""Advanced feature tests: image editing, prompt templates, etc.

These tests verify advanced functionality beyond basic image generation.
They require valid API keys and will incur costs.

Run with: pytest tests/integration/test_advanced_features.py -v -s
"""

import asyncio
import base64
import json
from pathlib import Path

import pytest

from image_gen_mcp.config.settings import Settings
from image_gen_mcp.prompts.template_manager import template_manager
from image_gen_mcp.storage.manager import ImageStorageManager
from image_gen_mcp.tools.image_editing import ImageEditingTool
from image_gen_mcp.tools.image_generation import ImageGenerationTool
from image_gen_mcp.types.enums import ImageQuality, ImageSize, ImageStyle
from image_gen_mcp.utils.cache import CacheManager
from image_gen_mcp.utils.openai_client import OpenAIClientManager


@pytest.mark.integration
@pytest.mark.slow
class TestPromptTemplates:
    """Test prompt template functionality."""

    def test_list_all_templates(self):
        """Test listing all available templates."""
        print("\n" + "=" * 80)
        print("TEST: List All Prompt Templates")
        print("=" * 80)

        templates = template_manager.list_templates()
        print(f"\nTotal templates: {len(templates)}")

        for template in templates:
            print(f"\n- {template['id']}")
            print(f"  Title: {template['title']}")
            print(f"  Category: {template['category']}")
            print(f"  Parameters: {template['parameter_count']}")

        assert len(templates) > 0
        assert all('id' in t for t in templates)

    def test_list_templates_by_category(self):
        """Test listing templates organized by category."""
        print("\n" + "=" * 80)
        print("TEST: Templates by Category")
        print("=" * 80)

        categories = template_manager.list_templates_by_category()

        for cat_id, cat_data in categories.items():
            print(f"\n📁 {cat_data['category']['name']}")
            print(f"   {cat_data['category']['description']}")
            print(f"   Templates: {len(cat_data['templates'])}")

            for template in cat_data['templates']:
                print(f"   - {template['id']}: {template['title']}")

        assert len(categories) > 0

    def test_get_template_details(self):
        """Test getting detailed information about a template."""
        print("\n" + "=" * 80)
        print("TEST: Template Details")
        print("=" * 80)

        template_id = "creative_image"
        details = template_manager.get_template_details(template_id)

        print(f"\nTemplate: {details['title']}")
        print(f"Description: {details['description']}")
        print(f"\nParameters:")

        for param_name, param in details['parameters'].items():
            req = "REQUIRED" if param['required'] else "optional"
            print(f"  - {param_name} ({param['type']}, {req})")
            print(f"    {param['description']}")

        print(f"\nMetadata:")
        print(f"  Recommended Size: {details['metadata']['recommended_size']}")
        print(f"  Quality: {details['metadata']['quality']}")
        print(f"  Style: {details['metadata']['style']}")

        assert details is not None
        assert 'parameters' in details

    def test_render_template(self):
        """Test rendering a template with parameters."""
        print("\n" + "=" * 80)
        print("TEST: Render Template")
        print("=" * 80)

        template_id = "creative_image"
        params = {
            "subject": "a majestic dragon",
            "style": "fantasy digital art",
            "mood": "epic and mysterious",
            "color_palette": "deep blues and purples with gold accents"
        }

        rendered, metadata = template_manager.render_template(template_id, **params)

        print(f"\nRendered Prompt:")
        print(f"{rendered}")
        print(f"\nRecommended Settings:")
        print(f"  Size: {metadata['recommended_size']}")
        print(f"  Quality: {metadata['quality']}")
        print(f"  Style: {metadata['style']}")

        assert rendered is not None
        assert len(rendered) > 0
        assert "dragon" in rendered.lower()


@pytest.mark.integration
@pytest.mark.slow
class TestPromptTemplateGeneration:
    """Test generating images using prompt templates."""

    @pytest.fixture
    async def real_tool(self, tmp_path):
        """Create a real image generation tool."""
        settings = Settings()

        from image_gen_mcp.config.settings import StorageSettings

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

        if hasattr(storage_manager, 'close'):
            await storage_manager.close()
        if hasattr(cache_manager, 'close'):
            await cache_manager.close()

    async def test_creative_image_template(self, real_tool):
        """Test generating image using creative_image template."""
        print("\n" + "=" * 80)
        print("TEST: Creative Image Template")
        print("=" * 80)

        # Render template
        rendered, metadata = template_manager.render_template(
            "creative_image",
            subject="a futuristic space station orbiting a ringed planet",
            style="sci-fi concept art",
            mood="awe-inspiring and grand",
            color_palette="cosmic blues, silvers, and starlight"
        )

        print(f"Prompt: {rendered}")

        # Generate image
        result = await real_tool.generate(
            prompt=rendered,
            style=ImageStyle.VIVID,
            quality=ImageQuality.AUTO,
            size=ImageSize.LANDSCAPE,
        )

        print(f"\nImage ID: {result['image_id']}")
        print(f"Image URL: {result['image_url']}")

        assert result['image_id']
        assert 'space station' in rendered.lower()

    async def test_product_photography_template(self, real_tool):
        """Test product photography template."""
        print("\n" + "=" * 80)
        print("TEST: Product Photography Template")
        print("=" * 80)

        rendered, metadata = template_manager.render_template(
            "product_photography",
            product="sleek wireless earbuds in matte black",
            background="white studio background",
            lighting="soft natural lighting from left",
            angle="45-degree angle view"
        )

        print(f"Prompt: {rendered}")

        result = await real_tool.generate(
            prompt=rendered,
            style=ImageStyle.NATURAL,
            quality=ImageQuality.HIGH,
            size=ImageSize.SQUARE,
        )

        print(f"\nImage ID: {result['image_id']}")
        print(f"Quality: {result['metadata']['quality']}")

        assert result['image_id']

    async def test_drawing_reference_template(self, real_tool):
        """Test drawing reference template."""
        print("\n" + "=" * 80)
        print("TEST: Gesture Drawing Template")
        print("=" * 80)

        rendered, metadata = template_manager.render_template(
            "gesture_drawing",
            subject="human figure in dynamic action pose",
            pose_type="action",
            emphasis="movement and flow",
            time_limit="quick"
        )

        print(f"Prompt: {rendered}")

        result = await real_tool.generate(
            prompt=rendered,
            style=ImageStyle.NATURAL,
            quality=ImageQuality.HIGH,
            size=ImageSize.LANDSCAPE,
        )

        print(f"\nImage ID: {result['image_id']}")
        print(f"Image URL: {result['image_url']}")

        assert result['image_id']


@pytest.mark.integration
@pytest.mark.slow
class TestImageEditing:
    """Test image editing functionality."""

    @pytest.fixture
    async def editing_tool(self, tmp_path):
        """Create image editing tool with OpenAI client."""
        settings = Settings()

        from image_gen_mcp.config.settings import StorageSettings

        project_storage = Path(__file__).parent.parent.parent / "storage"
        storage_settings = StorageSettings(
            base_path=str(project_storage),
            retention_days=30,
            max_size_gb=10.0,
            cleanup_interval_hours=24,
        )

        storage_manager = ImageStorageManager(settings=storage_settings)
        cache_manager = CacheManager(settings=settings.cache)

        # Create OpenAI client
        openai_client = OpenAIClientManager(settings.providers.openai)

        tool = ImageEditingTool(
            storage_manager=storage_manager,
            cache_manager=cache_manager,
            settings=settings,
            openai_client=openai_client,
        )

        yield tool, storage_manager

        if hasattr(storage_manager, 'close'):
            await storage_manager.close()
        if hasattr(cache_manager, 'close'):
            await cache_manager.close()

    async def test_simple_edit(self, editing_tool):
        """Test simple image editing."""
        print("\n" + "=" * 80)
        print("TEST: Simple Image Edit")
        print("=" * 80)

        tool, storage_manager = editing_tool

        # First, generate a base image to edit
        print("Generating base image...")

        # Find an existing image to edit instead
        image_files = list(Path("storage/images").rglob("*.png"))
        if not image_files:
            pytest.skip("No existing images found to edit")

        # Use the first available image
        image_path = image_files[0]
        print(f"Using existing image: {image_path}")

        # Read and encode the image
        with open(image_path, "rb") as f:
            image_bytes = f.read()

        image_b64 = base64.b64encode(image_bytes).decode()
        image_data = f"data:image/png;base64,{image_b64}"

        # Edit the image
        print("\nEditing image...")
        edit_prompt = "Add golden sunlight streaming through the scene"

        result = await tool.edit(
            image_data=image_data,
            prompt=edit_prompt,
            size="1024x1024",
            quality="auto",
        )

        print(f"\nEdited Image ID: {result['image_id']}")
        print(f"Image URL: {result['image_url']}")
        print(f"Edit Prompt: {edit_prompt}")
        print(f"Cost: ${result['metadata']['cost_estimate']:.4f}")

        assert result['image_id']
        assert result['operation'] == 'edit'


if __name__ == "__main__":
    """Allow running tests directly with python."""
    pytest.main([__file__, "-v", "-s"])
