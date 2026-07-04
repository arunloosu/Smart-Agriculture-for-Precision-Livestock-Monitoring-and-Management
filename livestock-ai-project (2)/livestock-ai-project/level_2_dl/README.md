# Level DL — Deep Learning
## (A) CNN Health Assessment from Images + (B) LSTM Behavior Sequences

### Real Datasets
**A. Lumpy Skin Disease Images** (healthy vs. LSD-infected cattle photos)
- Kaggle: `warcoder/lumpy-skin-images-dataset` (~8,000 images, used in published
  ViT/CNN research achieving 95–98% accuracy)
- Alternative real source: Mendeley Data, Kumar & Shastri, *"Lumpy Skin Images
  Dataset"*, DOI `10.17632/w36hpf86j2.1`

```bash
kaggle datasets download -d warcoder/lumpy-skin-images-dataset -p ./data/images --unzip
```
Expected layout after unzip: `./data/images/{healthy,lumpy}/*.jpg` (rename
folders to match if the archive uses different class names — check `data/images/`).

**B. Accelerometer sequences** — reuses the raw (unwindowed) CSV from
`level_1_ml/data/` for an RNN that learns temporal behavior patterns instead
of hand-crafted window statistics.

### Models
- **CNN (transfer learning):** MobileNetV2 backbone (ImageNet weights) +
  fine-tuned classification head → healthy / lumpy. Chosen because published
  cattle-disease research shows MobileNetV2 is the most deployment-friendly
  (runs on-device/edge, matching the LLD's low-latency alerting requirement).
- **RNN (LSTM):** raw tri-axial sequences → behavior label, to capture
  temporal dependencies the ML-level statistical features miss (e.g.
  transition patterns that precede lameness).

### Run
```bash
pip install -r requirements.txt
python train_cnn.py --data ./data/images
python train_lstm.py --data ../level_1_ml/data/cattle_behavior.csv
```

### Deliverables
- `outputs/cnn_lumpy_detector.h5`, `outputs/training_curves_cnn.png`
- `outputs/lstm_behavior_model.h5`, `outputs/training_curves_lstm.png`
- `outputs/dl_vs_ml_comparison.json` — DL metrics vs. the Level-ML baseline
