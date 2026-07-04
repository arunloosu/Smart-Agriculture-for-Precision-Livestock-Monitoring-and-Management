"""
Generate synthetic lumpy-skin-disease images from the fine-tuned LoRA adapter,
watermarked to prevent misuse as real diagnostic photos (Ethics Note in README).

Usage:
    python generate_synthetic.py --lora ./outputs/lora-lumpy --n 200 --out ./outputs/synthetic_images
"""
import argparse
import os

import torch
from diffusers import StableDiffusionPipeline
from peft import PeftModel
from PIL import Image, ImageDraw

BASE_MODEL = "runwayml/stable-diffusion-v1-5"
CONDITIONS = ["overcast lighting", "bright sunlight", "muddy pasture", "close-up on lesion",
              "side profile", "rainy weather", "dusty barn"]


def watermark(img: Image.Image) -> Image.Image:
    draw = ImageDraw.Draw(img)
    draw.text((10, img.height - 20), "SYNTHETIC - NOT A REAL DIAGNOSTIC IMAGE", fill=(255, 0, 0))
    return img


def main(args):
    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pipe = StableDiffusionPipeline.from_pretrained(BASE_MODEL, torch_dtype=torch.float16 if device == "cuda" else torch.float32)
    pipe.unet = PeftModel.from_pretrained(pipe.unet, args.lora)
    pipe = pipe.to(device)

    for i in range(args.n):
        condition = CONDITIONS[i % len(CONDITIONS)]
        prompt = f"a photograph of a cow with lumpy skin disease lesions, {condition}"
        image = pipe(prompt, num_inference_steps=30).images[0]
        image = watermark(image)
        image.save(os.path.join(args.out, f"synthetic_{i:04d}.png"))

        # Strip/tag EXIF so downstream tooling can programmatically detect synthetic origin
        exif = image.getexif()
        exif[0x9286] = "SYNTHETIC_GENAI_AUGMENTATION"  # UserComment tag
        image.save(os.path.join(args.out, f"synthetic_{i:04d}.png"), exif=exif)

    print(f"Generated {args.n} watermarked synthetic images to {args.out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora", required=True)
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    main(args)
