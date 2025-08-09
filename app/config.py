"""
Configuration management using Pydantic Settings
"""

from pydantic_settings import BaseSettings
from typing import List
import os


class Settings(BaseSettings):
    # Server Configuration
    uvicorn_host: str = "0.0.0.0"
    uvicorn_port: int = 8000
    
    # Mode Configuration
    use_stub: bool = True
    
    # CORS Configuration
    allowed_origins: str = "https://innovate-labs.in,http://localhost:3000"
    
    # Model Paths
    yolo_esb_weights: str = "./models/esb_yolov8_best.pt"
    yolo_disease_weights: str = "./models/disease_yolov8s_seg.pt"
    tabnet_esb_path: str = "./models/tabnet_esb.zip"
    tabnet_disease_path: str = "./models/tabnet_disease.zip"
    
    # Inference Thresholds
    fusion_yolo_threshold: float = 0.60
    fusion_tabnet_threshold: float = 0.70
    
    # Static Files
    ref_image_base_url: str = "/static/reference"
    
    class Config:
        env_file = ".env"
        case_sensitive = False
    
    @property
    def allowed_origins_list(self) -> List[str]:
        """Convert comma-separated origins to list"""
        return [origin.strip() for origin in self.allowed_origins.split(",")]
    
    def model_exists(self, path: str) -> bool:
        """Check if model file exists"""
        return os.path.exists(path) and os.path.isfile(path)


# Global settings instance
settings = Settings()
