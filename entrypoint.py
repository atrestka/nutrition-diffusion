#Modal Entrypoint to Stable Diffusion

import modal
from pathlib import Path

app = modal.App("diffusion-nutrition-training")

MINUTES = 60
HOURS = 60 * MINUTES

# Add mount for local diffusion directory
diffusion_mount = modal.Mount.from_local_dir(
    "~/Desktop/diffusion",
    remote_path="/root/diffusion"
)

#Potentially Change
output_volume = modal.Volume.from_name("diffusion-V0-outputs", create_if_missing=True)
cache_volume = modal.Volume.from_name("diffusion-V0-cache", create_if_missing=True)
OUTPUT_DIR = Path("/outputs")
CACHE_DIR = Path("/cache")

# Create image with dependencies
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .pip_install(
        "torch==2.1.2", 
        "mosaicml",
        "composer",
        "accelerate", 
        "diffusers",
        "transformers",
        "wandb",
        "mosaicml-streaming"
    )
)

@app.cls(
    image=image,
    gpu=modal.gpu.H100(count=8),
    volumes={OUTPUT_DIR: output_volume, CACHE_DIR: cache_volume},
    mounts=[diffusion_mount],
    timeout=24 * HOURS,
    secrets=[
        modal.Secret.from_name("aws-nutrition"),
        modal.Secret.from_name("wandb-Nutrition-Diffusion")
    ]
)
class DiffusionTrainer:

    @modal.enter()
    def setup(self):
        import sys
        sys.path.append("/root/diffusion")
        # Install the mounted package in editable mode
        import subprocess
        subprocess.run(["pip", "install", "-e", "/root/diffusion"], check=True)

    @modal.method()
    def train(self, config_name: str = "SD-2-base-256.yaml"):
        import os
        os.chdir("/root/diffusion")
        
        cmd = [
            "composer",
            "run.py",
            "--config-path=yamls/hydra-yamls",
            f"--config-name={config_name}"
        ]
        
        import subprocess
        subprocess.run(cmd, check=True)
        
        # Commit changes to volumes
        output_volume.commit()

@app.local_entrypoint()
def main(config_name: str = "SD-2-base-256.yaml"):
    trainer = DiffusionTrainer()
    trainer.train.remote(config_name)