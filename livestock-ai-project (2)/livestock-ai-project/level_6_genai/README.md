# Level Gen AI — Generative AI
## Synthetic Cattle-Disease Image Augmentation

### Real Seed Dataset
Reuses the **Lumpy Skin Disease Images** dataset from Level DL (real photos of
healthy vs. LSD-infected cattle). Rare-disease classes and unusual lighting/
pose conditions are under-represented in the real data — Gen AI is used to
synthetically expand exactly those tail cases, per the Phase-3 spec ("augment
training datasets, especially for rare diseases... stress-test monitoring
models under diverse environmental conditions").

### Approach
1. **Fine-tuned diffusion model (DreamBooth-style / LoRA on Stable Diffusion)**
   trained on the real "lumpy" class images to generate additional synthetic
   lesion examples, plus prompt-driven variation for lighting/weather/breed.
2. **Classical augmentation GAN (DCGAN)** included as a lighter-weight
   alternative for teams without GPU access to a diffusion pipeline.
3. **Stress-testing:** synthetic images with programmatically varied
   backgrounds/lighting are run through the Level-DL CNN to measure
   robustness degradation — a real evaluation of "diverse environmental
   conditions," not just visual novelty.

### Run
```bash
pip install -r requirements.txt
# A) Diffusion-based augmentation (needs GPU + HF token for base SD model)
python train_diffusion_lora.py --data ../level_2_dl/data/images/lumpy --steps 800
python generate_synthetic.py --lora ./outputs/lora-lumpy --n 200 --out ./outputs/synthetic_images

# B) Lightweight DCGAN alternative (CPU-feasible, lower fidelity)
python train_dcgan.py --data ../level_2_dl/data/images/lumpy --epochs 100

# C) Stress-test the Level-DL CNN against synthetic + perturbed images
python stress_test.py --cnn ../level_2_dl/outputs/cnn_lumpy_detector.h5 \
    --synthetic ./outputs/synthetic_images
```

### Evaluation
- **FID (Fréchet Inception Distance)** between real and synthetic lumpy images.
- **Downstream utility:** CNN accuracy when trained with vs. without the
  synthetic augmentation, on a held-out real test set (the metric that
  actually matters — realism scores alone can be misleading).
- **Robustness delta:** CNN accuracy on brightness/contrast/blur-perturbed
  real images before vs. after augmentation-assisted retraining.

### Deliverables
- `outputs/lora-lumpy/` — fine-tuned diffusion adapter
- `outputs/synthetic_images/` — generated samples
- `outputs/fid_score.json`
- `outputs/downstream_utility.json` — accuracy with/without synthetic augmentation
- `outputs/robustness_report.json`

### Ethics Note
Synthetic disease imagery is for **training/augmentation only** — outputs are
watermarked (`_synthetic` suffix + EXIF tag) and must never be presented to a
farmer/vet as a real diagnostic photo. This is enforced in `generate_synthetic.py`.
