"""Unit tests for OpenAI provider, especially gpt-image-2 capabilities."""

import pytest

from image_gen_mcp.providers.base import ProviderConfig
from image_gen_mcp.providers.openai import OpenAIProvider


@pytest.fixture
def provider():
    return OpenAIProvider(ProviderConfig(api_key="sk-test", enabled=True))


class TestGptImage2Registration:
    def test_gpt_image_2_registered(self):
        assert "gpt-image-2" in OpenAIProvider.SUPPORTED_MODELS

    def test_gpt_image_2_capabilities(self):
        cap = OpenAIProvider.SUPPORTED_MODELS["gpt-image-2"]
        assert cap.supports_custom_sizes is True
        assert cap.supports_background is True
        assert cap.supports_compression is True
        assert cap.size_constraints is not None
        assert cap.size_constraints["multiple_of"] == 16
        assert cap.size_constraints["max_edge"] == 3840
        assert cap.size_constraints["max_aspect_ratio"] == 3.0
        assert "3840x2160" in cap.supported_sizes

    def test_gpt_image_1_still_has_fixed_sizes(self):
        """Regression: legacy gpt-image-1 / 1.5 remain preset-only."""
        for mid in ("gpt-image-1", "gpt-image-1.5"):
            cap = OpenAIProvider.SUPPORTED_MODELS[mid]
            assert cap.supports_custom_sizes is False
            assert cap.size_constraints is None


class TestValidateCustomSize:
    constraints = {
        "multiple_of": 16,
        "max_edge": 3840,
        "max_aspect_ratio": 3.0,
        "min_pixels": 655_360,
        "max_pixels": 8_294_400,
    }

    @pytest.mark.parametrize(
        "size",
        ["1024x1024", "3840x2160", "2048x1152", "1600x896", "1536x1024"],
    )
    def test_valid_sizes(self, size):
        assert OpenAIProvider._validate_custom_size(size, self.constraints) is True

    @pytest.mark.parametrize(
        "size,reason",
        [
            ("1000x1000", "not multiple of 16"),
            ("4096x2160", "edge exceeds 3840"),
            ("3840x1024", "aspect ratio 3.75 exceeds 3.0 limit"),
            ("512x512", "too few pixels"),
            ("garbage", "malformed"),
            ("16x16", "too few pixels"),
            ("", "empty string"),
            ("1024", "missing dimension"),
        ],
    )
    def test_invalid_sizes(self, size, reason):
        valid = OpenAIProvider._validate_custom_size(size, self.constraints)
        assert valid is False, reason


class TestEstimateCostGptImage2:
    def test_cost_uses_30_per_1m(self, provider):
        result = provider.estimate_cost("gpt-image-2", "a red circle", image_count=1)
        # 1750 tokens/image * $30/1M = $0.0525 per image, plus tiny text cost
        image_cost = result["breakdown"]["image_output_cost"]
        assert image_cost == pytest.approx(0.0525, abs=1e-4)
        assert result["estimated_cost_usd"] > 0
        assert result["provider"] == "openai"
        assert result["model"] == "gpt-image-2"

    def test_cost_cheaper_than_gpt_image_1_5(self, provider):
        prompt = "a blue square on a white background"
        v2 = provider.estimate_cost("gpt-image-2", prompt, 1)["estimated_cost_usd"]
        v1_5 = provider.estimate_cost("gpt-image-1.5", prompt, 1)["estimated_cost_usd"]
        assert v2 < v1_5


class TestResolveSize:
    def test_preset_size_passthrough(self, provider):
        cap = OpenAIProvider.SUPPORTED_MODELS["gpt-image-2"]
        assert provider._resolve_size("1024x1024", cap, "gpt-image-2") == "1024x1024"

    def test_custom_size_allowed_for_v2(self, provider):
        cap = OpenAIProvider.SUPPORTED_MODELS["gpt-image-2"]
        assert provider._resolve_size("2048x1152", cap, "gpt-image-2") == "2048x1152"

    def test_invalid_custom_size_falls_back(self, provider):
        cap = OpenAIProvider.SUPPORTED_MODELS["gpt-image-2"]
        assert provider._resolve_size("9999x9999", cap, "gpt-image-2") == "auto"

    def test_custom_size_rejected_for_v1(self, provider):
        """gpt-image-1 has no custom-size support; non-preset falls back."""
        cap = OpenAIProvider.SUPPORTED_MODELS["gpt-image-1"]
        assert provider._resolve_size("2048x1152", cap, "gpt-image-1") == "auto"


class TestValidateModelParamsSize:
    """validate_model_params must resolve the size so metadata and cache
    keys reflect what is actually sent to the API."""

    def test_invalid_custom_size_normalized(self, provider):
        params = provider.validate_model_params(
            "gpt-image-2", {"size": "9999x9999"}
        )
        assert params["size"] == "auto"

    def test_valid_custom_size_preserved(self, provider):
        params = provider.validate_model_params(
            "gpt-image-2", {"size": "2048x1152"}
        )
        assert params["size"] == "2048x1152"

    def test_preset_size_preserved(self, provider):
        params = provider.validate_model_params(
            "gpt-image-2", {"size": "1024x1024"}
        )
        assert params["size"] == "1024x1024"

    def test_v1_non_preset_falls_back(self, provider):
        """Legacy gpt-image-1 without custom-size support is normalized
        to a supported value."""
        params = provider.validate_model_params(
            "gpt-image-1", {"size": "2048x1152"}
        )
        assert params["size"] in OpenAIProvider.SUPPORTED_MODELS[
            "gpt-image-1"
        ].supported_sizes

    def test_enum_size_unwrapped(self, provider):
        """ImageSize enum values pass through to the underlying WxH string.

        str(ImageSize.LANDSCAPE) returns 'ImageSize.LANDSCAPE' (not the value)
        because ImageSize is a str/Enum hybrid, not StrEnum. The provider
        must unwrap to .value so the API sees '1536x1024', not a dotted name.
        """
        from image_gen_mcp.types.enums import ImageSize

        params = provider.validate_model_params(
            "gpt-image-2", {"size": ImageSize.LANDSCAPE}
        )
        assert params["size"] == "1536x1024"

    def test_whitespace_and_case_normalized(self, provider):
        """Raw callers passing non-canonical strings get a normalized value."""
        params = provider.validate_model_params(
            "gpt-image-2", {"size": "  1600X896  "}
        )
        assert params["size"] == "1600x896"

    def test_whitespace_preset_preserved_on_fixed_size_model(self, provider):
        """For models without custom-size support, the preset match must
        survive whitespace/case variants — otherwise super()'s exact match
        would downgrade "  1024X1024  " to the default before we normalize."""
        params = provider.validate_model_params(
            "gpt-image-1", {"size": "  1024X1024  "}
        )
        assert params["size"] == "1024x1024"
