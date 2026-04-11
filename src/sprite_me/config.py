"""Configuration for sprite-me."""

from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_prefix": "SPRITE_ME_"}

    # RunPod
    runpod_api_key: str = ""
    runpod_endpoint_id: str = ""
    runpod_base_url: str = "https://api.runpod.ai/v2"

    # Generation defaults
    default_model: str = "flux1-dev"
    default_lora: str = "Flux-2D-Game-Assets-LoRA"
    default_lora_strength: float = 0.85
    default_width: int = 512
    default_height: int = 512
    default_steps: int = 30
    default_guidance: float = 3.5
    default_smart_crop_mode: str = "tightest"
    lora_trigger_word: str = "GRPZA"

    # Animation defaults
    default_animation_frames: int = 6
    default_edge_margin: int = 6
    auto_enhance_prompt: bool = True

    # Storage
    assets_dir: Path = Path("./assets")
    manifest_path: Path = Path("./sprite-me-assets.json")

    # Server
    api_host: str = "0.0.0.0"
    api_port: int = 8420
    mcp_transport: str = "stdio"  # "stdio" or "sse"

    @property
    def runpod_endpoint_url(self) -> str:
        return f"{self.runpod_base_url}/{self.runpod_endpoint_id}"


settings = Settings()
