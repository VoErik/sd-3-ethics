import os
import torch
import pandas as pd
from dataclasses import dataclass
from typing import Any
from dotenv import load_dotenv
from huggingface_hub import login
from diffusers import StableDiffusion3Pipeline, SD3Transformer2DModel, BitsAndBytesConfig
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# need to get access to the models from HF first:
# https://huggingface.co/stabilityai
MODEL_DICT = {
    "medium-3.5": "stabilityai/stable-diffusion-3.5-medium",
    "large-3.5": "stabilityai/stable-diffusion-3.5-large",
    "turbo-3.5": "stabilityai/stable-diffusion-3.5-large-turbo"
}

@dataclass
class SDConfig:
    model_id: str = "turbo-3.5"
    csv_path: str = "./assets/prompts.csv"
    output_base_dir: str = "./images"
    num_images_per_prompt: int = 3
    inference_steps: int = 50
    enable_model_cpu_offload: bool = True
    height: int = 512
    width: int = 512
    guidance_scale: float = 7.0
    device: str = "cuda"
    use_quantized: bool = False
    csv_delimiter: str = ";"

def load_pipeline(cfg: SDConfig):
    device = "cuda" if cfg.device == "cuda" and torch.cuda.is_available() else "cpu"
    model_id = MODEL_DICT[cfg.model_id]

    logger.info("Loading model: %s", model_id)

    if cfg.use_quantized:
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        model_nf4 = SD3Transformer2DModel.from_pretrained(
            model_id,
            subfolder="transformer",
            quantization_config=nf4_config,
            torch_dtype=torch.bfloat16
        )
        pipe = StableDiffusion3Pipeline.from_pretrained(
            model_id,
            transformer=model_nf4,
            torch_dtype=torch.bfloat16
        )
    else:
        pipe = StableDiffusion3Pipeline.from_pretrained(
            pretrained_model_name_or_path=model_id,
            torch_dtype=torch.float16
        )

    if cfg.enable_model_cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)

    logger.info(f"Model loaded and moved to {device} %s", "with CPU offloading" if cfg.enable_model_cpu_offload else "")
    return pipe

def generate_images(cfg: SDConfig, pipe: StableDiffusion3Pipeline):
    df = pd.read_csv(cfg.csv_path, delimiter=cfg.csv_delimiter)
    str_cols = ['G', 'E', 'P']
    df[str_cols] = df[str_cols].astype('string')
    df[str_cols] = df[str_cols].fillna('') 
    df["Negative Prompt"] = df.get("Negative Prompt", "")

    logger.info("Generating images for %d prompts...", len(df))

    for idx, row in df.iterrows():
        print(row)
        parts = [str(row[col]).lower() for col in ['G', 'E', 'P'] if str(row[col]).strip() != '']
        combination = '-'.join(parts) 
        prompt = str(row["Prompt"])
        negative_prompt = str(row["Negative Prompt"])
        output_dir = os.path.join(cfg.output_base_dir, cfg.model_id, f"{idx}-{combination}")
        os.makedirs(output_dir, exist_ok=True)

        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=cfg.inference_steps,
            height=cfg.height,
            width=cfg.width,
            guidance_scale=cfg.guidance_scale,
            num_images_per_prompt=cfg.num_images_per_prompt
        )

        for i, img in enumerate(result.images):
            img_path = os.path.join(output_dir, f"{combination}-{idx}-{i}.png")
            img.save(img_path)


        with open(os.path.join(output_dir, f"{combination}-{idx}.txt"), "w") as f:
            f.write(f"combination: {combination}\n")
            f.write(f"prompt: {prompt}\n")
            f.write(f"negative_prompt: {negative_prompt}\n")

    logger.info("Image generation finished.")

def run_pipeline(cfg: SDConfig | Any):
    pipe = load_pipeline(cfg)
    generate_images(cfg, pipe)

if __name__ == "__main__":
    load_dotenv()
    login(token=os.getenv("HF_KEY"))

    cfg = SDConfig(
        csv_path="./assets/prompts.csv",
        num_images_per_prompt=10
    )
    run_pipeline(cfg)
