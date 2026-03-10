"""Gemini provider for image generation using the google-genai SDK."""

import json
import logging
import os
from typing import Any

from google import genai
from google.genai import types
from google.oauth2 import service_account

from .base import (
    ImageResponse,
    LLMProvider,
    ModelCapability,
    ProviderConfig,
    ProviderError,
)

logger = logging.getLogger(__name__)


class GeminiProvider(LLMProvider):
    """Gemini provider for image generation using Imagen models
    via the google-genai SDK."""

    # Shared capabilities for Imagen 4 family models
    _IMAGEN_4_CAPABILITY = dict(
        supported_sizes=["1024x1024", "1536x1024", "1024x1536"],
        supported_qualities=["auto", "high", "medium", "low"],
        supported_formats=["png", "jpeg", "webp"],
        max_images_per_request=1,
        supports_style=False,
        supports_background=False,
        supports_compression=True,
        custom_parameters={
            "aspect_ratio": [
                "1:1",
                "3:4",
                "4:3",
                "9:16",
                "16:9",
            ],
        },
    )

    # Supported Imagen models and their capabilities
    SUPPORTED_MODELS = {
        "imagen-4": ModelCapability(
            model_id="imagen-4.0-generate-001",
            **_IMAGEN_4_CAPABILITY,
        ),
        "imagen-4-ultra": ModelCapability(
            model_id="imagen-4.0-ultra-generate-001",
            **_IMAGEN_4_CAPABILITY,
        ),
        "imagen-4-fast": ModelCapability(
            model_id="imagen-4.0-fast-generate-001",
            **_IMAGEN_4_CAPABILITY,
        ),
        "imagen-3": ModelCapability(
            model_id="imagen-3.0-generate-002",
            **_IMAGEN_4_CAPABILITY,
        ),
    }

    # Size to aspect ratio mapping
    SIZE_TO_ASPECT_RATIO = {
        "1024x1024": "1:1",
        "1536x1024": "4:3",
        "1024x1536": "3:4",
        "auto": "1:1",
    }

    def __init__(self, config: ProviderConfig):
        super().__init__(config)

        # For Vertex AI (Imagen models), api_key is a path to a
        # service account JSON file.
        self.credentials_path = config.api_key

        # --- Security validations (path traversal, size, etc.) ---
        resolved_path = os.path.abspath(self.credentials_path)

        allowed_dirs = [
            os.path.abspath(os.getcwd()),
            os.path.expanduser("~/.config/gcloud"),
            os.path.expanduser("~/.google"),
        ]

        path_is_allowed = any(
            resolved_path.startswith(
                os.path.abspath(d) + os.sep
            )
            for d in allowed_dirs
        )
        if not path_is_allowed:
            raise ValueError(
                f"Credentials file path is not in allowed "
                f"directories: {resolved_path}. "
                f"Allowed directories: {allowed_dirs}"
            )

        if not os.path.exists(resolved_path):
            raise ValueError(
                f"Service account file not found: "
                f"{resolved_path}. Please ensure the service "
                "account JSON file exists and the path is "
                "correct."
            )

        if not os.path.isfile(resolved_path):
            raise ValueError(
                f"Credentials path is not a file: "
                f"{resolved_path}"
            )

        # Load service account credentials
        self.credentials = (
            service_account.Credentials.from_service_account_file(
                resolved_path,
                scopes=[
                    "https://www.googleapis.com/auth/cloud-platform"
                ],
            )
        )

        # Extract project_id from the credentials file
        try:
            max_file_size = 1024 * 1024  # 1 MB
            file_size = os.path.getsize(resolved_path)
            if file_size > max_file_size:
                raise ValueError(
                    f"Service account file '{resolved_path}' is "
                    f"too large ({file_size} bytes). Maximum "
                    f"allowed size is {max_file_size} bytes."
                )

            with open(resolved_path, encoding="utf-8") as f:
                try:
                    cred_data = json.load(f)
                except json.JSONDecodeError as e:
                    raise ValueError(
                        f"Invalid JSON in service account file "
                        f"'{resolved_path}': {e}."
                    ) from e

                self.project_id = cred_data.get("project_id")
                if not self.project_id:
                    keys = (
                        list(cred_data.keys())
                        if isinstance(cred_data, dict)
                        else "N/A"
                    )
                    raise ValueError(
                        f"'project_id' not found in "
                        f"'{resolved_path}'. "
                        f"Available fields: {keys}."
                    )

        except PermissionError as e:
            raise ValueError(
                f"Permission denied reading "
                f"'{resolved_path}': {e}."
            ) from e
        except OSError as e:
            raise ValueError(
                f"Unable to read service account file "
                f"'{resolved_path}': {e}."
            ) from e

        # Initialize the google-genai SDK client.
        # Image generation can take 30-60s; pass an explicit
        # httpx client so the timeout is actually respected.
        import httpx

        self._httpx_client = httpx.AsyncClient(
            timeout=httpx.Timeout(config.timeout),
        )
        self.client = genai.Client(
            vertexai=True,
            project=self.project_id,
            location="us-central1",
            credentials=self.credentials,
            http_options=types.HttpOptions(
                api_version="v1",
                httpx_async_client=self._httpx_client,
            ),
        )

    async def close(self) -> None:
        """Close the underlying httpx client to avoid resource leaks."""
        try:
            if not self._httpx_client.is_closed:
                await self._httpx_client.aclose()
        except Exception as e:
            self._logger.debug(
                f"Error closing httpx client: {e}"
            )

    def get_supported_models(self) -> set[str]:
        """Return set of supported Gemini model IDs."""
        return set(self.SUPPORTED_MODELS.keys())

    def get_model_capabilities(
        self, model_id: str
    ) -> ModelCapability | None:
        """Get capabilities for a specific Gemini model."""
        return self.SUPPORTED_MODELS.get(model_id)

    def _convert_size_to_aspect_ratio(self, size: str) -> str:
        """Convert OpenAI size format to Gemini aspect ratio."""
        return self.SIZE_TO_ASPECT_RATIO.get(size, "1:1")

    # Map quality names to output_compression_quality (0-100).
    # Only meaningful for lossy formats (JPEG, WebP); PNG ignores it.
    _QUALITY_TO_COMPRESSION = {
        "high": 90,
        "medium": 75,
        "low": 50,
    }

    async def generate_image(
        self,
        model: str,
        prompt: str,
        quality: str = "auto",
        size: str = "1536x1024",
        style: str = "vivid",
        output_format: str = "png",
        compression: int = 100,
        **kwargs,
    ) -> ImageResponse:
        """Generate image using the google-genai SDK."""

        if model not in self.SUPPORTED_MODELS:
            raise ProviderError(
                f"Model '{model}' is not supported by "
                "Gemini provider",
                provider_name=self.name,
                error_code="UNSUPPORTED_MODEL",
            )

        actual_model_id = self.SUPPORTED_MODELS[model].model_id

        aspect_ratio = (
            self._convert_size_to_aspect_ratio(size)
            if size != "auto"
            else "1:1"
        )

        mime_by_format = {
            "png": "image/png",
            "jpeg": "image/jpeg",
            "webp": "image/webp",
        }

        # Determine compression quality for the SDK.
        # Explicit compression param takes priority; fall back to
        # the quality name mapping for lossy formats.
        if compression < 100:
            compression_quality = compression
        elif quality in self._QUALITY_TO_COMPRESSION:
            compression_quality = self._QUALITY_TO_COMPRESSION[
                quality
            ]
        else:
            compression_quality = None  # let the API decide

        config_kwargs: dict[str, Any] = {
            "number_of_images": 1,
            "aspect_ratio": aspect_ratio,
            "output_mime_type": mime_by_format.get(
                output_format, "image/png"
            ),
        }
        if compression_quality is not None:
            config_kwargs[
                "output_compression_quality"
            ] = compression_quality

        config = types.GenerateImagesConfig(**config_kwargs)

        try:
            self._logger.info(
                f"Generating image with Gemini model {model}"
            )

            response = (
                await self.client.aio.models.generate_images(
                    model=actual_model_id,
                    prompt=prompt,
                    config=config,
                )
            )

            if not response.generated_images:
                raise ProviderError(
                    "No images generated in Imagen response",
                    provider_name=self.name,
                    error_code="INVALID_RESPONSE",
                )

            image = response.generated_images[0].image
            image_bytes = image.image_bytes

            if not image_bytes:
                raise ProviderError(
                    "Empty image data in Imagen response",
                    provider_name=self.name,
                    error_code="INVALID_RESPONSE",
                )

            metadata = {
                "model": model,
                "prompt": prompt,
                "size": size,
                "quality": quality,
                "provider": self.name,
                "created_at": None,
            }

            return ImageResponse(
                image_data=image_bytes,
                metadata=metadata,
                provider_response={},
            )

        except ProviderError:
            raise
        except Exception as e:
            msg = str(e) or type(e).__name__
            self._logger.error(
                f"Error generating image with Gemini: {msg}"
            )
            raise ProviderError(
                f"Gemini image generation failed: {msg}",
                provider_name=self.name,
                error_code="GENERATION_FAILED",
            )

    async def edit_image(
        self,
        model: str,
        image_data: str | bytes,
        prompt: str,
        mask_data: str | bytes | None = None,
        quality: str = "auto",
        size: str = "1536x1024",
        output_format: str = "png",
        compression: int = 100,
        background: str = "auto",
        n: int = 1,
        **kwargs,
    ) -> ImageResponse:
        """Edit image using Gemini's Images API.

        Not yet implemented. The google-genai SDK supports
        client.aio.models.edit_image() for future use.
        """
        raise ProviderError(
            "Image editing is not yet supported by "
            "Gemini provider",
            provider_name=self.name,
            error_code="FEATURE_NOT_SUPPORTED",
        )

    async def check_health(self) -> dict[str, Any]:
        """Verify credentials and list available Imagen models.

        Uses the v1beta1 publisher models list endpoint via httpx
        (the SDK's models.list() has known reliability issues
        with Vertex AI publisher models).
        """
        try:
            import httpx
            from google.auth.transport.requests import Request

            self.credentials.refresh(Request())
            url = (
                "https://us-central1-aiplatform.googleapis.com"
                "/v1beta1/publishers/google/models"
            )
            headers = {
                "Authorization": (
                    f"Bearer {self.credentials.token}"
                ),
            }

            async with httpx.AsyncClient(
                timeout=10
            ) as client:
                resp = await client.get(
                    url, headers=headers
                )
                if resp.status_code != 200:
                    body = resp.json()
                    msg = body.get("error", {}).get(
                        "message",
                        f"HTTP {resp.status_code}",
                    )
                    return {
                        "status": "unhealthy",
                        "error": msg,
                    }
                data = resp.json()

            api_models = {
                m["name"].split("/")[-1]
                for m in data.get("publisherModels", [])
            }

            available = []
            missing = []
            for alias, cap in self.SUPPORTED_MODELS.items():
                if cap.model_id in api_models:
                    available.append(alias)
                else:
                    missing.append(alias)

            result: dict[str, Any] = {
                "models_available": available,
            }
            if missing:
                result["status"] = "unhealthy"
                result["error"] = (
                    "Models no longer available: "
                    + ", ".join(missing)
                )
            else:
                result["status"] = "healthy"
            return result

        except Exception as e:
            self._logger.warning(
                f"Gemini health check failed: {e}"
            )
            return {"status": "unhealthy", "error": str(e)}

    def validate_model_params(
        self, model: str, params: dict[str, Any]
    ) -> dict[str, Any]:
        """Validate and normalize parameters for Gemini models."""

        params = super().validate_model_params(model, params)

        capabilities = self.get_model_capabilities(model)
        if not capabilities:
            raise ProviderError(
                f"No capabilities found for model '{model}'",
                provider_name=self.name,
                error_code="MISSING_CAPABILITIES",
            )

        # Convert size to aspect ratio and validate
        if "size" in params:
            aspect_ratio = self._convert_size_to_aspect_ratio(
                params["size"]
            )
            if aspect_ratio not in capabilities.custom_parameters.get(
                "aspect_ratio", []
            ):
                params["size"] = "1024x1024"
                self._logger.warning(
                    "Aspect ratio not supported, "
                    "using 1:1 (1024x1024)"
                )

        # Imagen 4 Ultra only supports 1 image at a time
        if model == "imagen-4-ultra" and params.get("n", 1) > 1:
            params["n"] = 1
            self._logger.warning(
                "Imagen 4 Ultra only supports generating "
                "1 image at a time"
            )

        # Remove unsupported parameters
        for param in ["style", "background", "moderation"]:
            if param in params:
                self._logger.debug(
                    f"Removing unsupported parameter "
                    f"'{param}' for Gemini"
                )
                del params[param]

        return params

    def estimate_cost(
        self, model: str, prompt: str, image_count: int = 1
    ) -> dict[str, Any]:
        """Estimate cost for Gemini image generation."""

        pricing = {
            "imagen-4": {"cost_per_image": 0.04},
            "imagen-4-ultra": {"cost_per_image": 0.06},
            "imagen-4-fast": {"cost_per_image": 0.02},
            "imagen-3": {"cost_per_image": 0.02},
        }

        if model not in pricing:
            return super().estimate_cost(
                model, prompt, image_count
            )

        model_pricing = pricing[model]
        total_cost = model_pricing["cost_per_image"] * image_count

        return {
            "provider": self.name,
            "model": model,
            "estimated_cost_usd": round(total_cost, 4),
            "currency": "USD",
            "breakdown": {
                "per_image": model_pricing["cost_per_image"],
                "total_images": image_count,
                "base_cost": total_cost,
            },
        }
