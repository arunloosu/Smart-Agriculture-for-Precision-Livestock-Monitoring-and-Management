"""
LoRA fine-tuning of Stable Diffusion on real lumpy-skin-disease images for
higher-fidelity synthetic augmentation (GPU recommended).

Dataset: real "lumpy" class images from level_2_dl.

Usage:
    python train_diffusion_lora.py --data ../level_2_dl/data/images/lumpy --steps 800
"""
import argparse
import os

from diffusers import StableDiffusionPipeline, DDPMScheduler, UNet2DConditionModel
from peft import LoraConfig, get_peft_model
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer

BASE_MODEL = "runwayml/stable-diffusion-v1-5"
PROMPT_TEMPLATE = "a photograph of a cow with lumpy skin disease lesions, {condition}"
CONDITIONS = ["overcast lighting", "bright sunlight", "muddy pasture", "close-up on lesion",
              "side profile", "rainy weather", "dusty barn"]


class LumpyImageDataset(Dataset):
    def __init__(self, folder, tokenizer, size=512):
        self.paths = [os.path.join(folder, f) for f in os.listdir(folder)
                      if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        self.tokenizer = tokenizer
        self.tf = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        pixel_values = self.tf(img)
        prompt = PROMPT_TEMPLATE.format(condition=CONDITIONS[idx % len(CONDITIONS)])
        input_ids = self.tokenizer(prompt, padding="max_length", truncation=True,
                                    max_length=self.tokenizer.model_max_length,
                                    return_tensors="pt").input_ids[0]
        return {"pixel_values": pixel_values, "input_ids": input_ids}


def main(args):
    os.makedirs("outputs/lora-lumpy", exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = CLIPTokenizer.from_pretrained(BASE_MODEL, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(BASE_MODEL, subfolder="text_encoder").to(device)
    unet = UNet2DConditionModel.from_pretrained(BASE_MODEL, subfolder="unet").to(device)
    noise_scheduler = DDPMScheduler.from_pretrained(BASE_MODEL, subfolder="scheduler")

    lora_config = LoraConfig(r=8, lora_alpha=32, target_modules=["to_q", "to_v"], lora_dropout=0.05)
    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()

    pipe = StableDiffusionPipeline.from_pretrained(BASE_MODEL)
    vae = pipe.vae.to(device)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    dataset = LumpyImageDataset(args.data, tokenizer)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    optimizer = torch.optim.AdamW(unet.parameters(), lr=1e-4)

    unet.train()
    step = 0
    while step < args.steps:
        for batch in loader:
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)

            latents = vae.encode(pixel_values).latent_dist.sample() * 0.18215
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=device)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            encoder_hidden_states = text_encoder(input_ids)[0]
            noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

            loss = torch.nn.functional.mse_loss(noise_pred, noise)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            step += 1
            if step % 50 == 0:
                print(f"step {step}/{args.steps} — loss={loss.item():.4f}")
            if step >= args.steps:
                break

    unet.save_pretrained("outputs/lora-lumpy")
    print("Saved LoRA adapter to outputs/lora-lumpy")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--steps", type=int, default=800)
    args = parser.parse_args()
    main(args)
