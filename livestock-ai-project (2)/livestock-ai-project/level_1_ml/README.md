# Level ML — Baseline Machine Learning
## Cattle Behavior Classification from Accelerometer Data

### Problem
Classify a cow's behavior (grazing / lying / standing / walking / ruminating)
from tri-axial collar-accelerometer readings — the core signal used in real
precision-livestock systems to flag illness (e.g., a sudden drop in grazing
time, excessive lying) before it's visible to a farmer.

### Real Dataset
**Beef Cattle Behavior Dataset** — Kaggle: `lucyfirst/beef-cattle-behavior-data-set`
(tri-axial accelerometer readings labeled with observed behavior).

Alternative / supplementary real dataset (same task family, also real,
peer-reviewed, and freely downloadable):
**ActBeCalf** — accelerometer-based calf behavior dataset, published via
Mendeley/ScienceDirect (30 calves, 27.4 hours of labeled accelerometer data,
DOI in the paper: "ActBeCalf: Accelerometer-based multivariate time-series
dataset for calf behavior classification", ScienceDirect 2025).

```bash
# Option A: Kaggle CLI
pip install kaggle
kaggle datasets download -d lucyfirst/beef-cattle-behavior-data-set -p ./data --unzip

# Option B: Kaggle CLI (calves)
kaggle datasets download -d <ActBeCalf mirror if available on Kaggle> -p ./data --unzip
```
Place the resulting CSV(s) in `./data/`. The script auto-detects columns named
like `acc_x, acc_y, acc_z, behavior` (adjust `COLUMN_MAP` in `train.py` if the
raw column names differ — real-world exports vary).

### Pipeline
1. **Preprocessing** — drop nulls, z-score normalize the 3 acceleration axes,
   window the signal (default 5s @ sampling rate in file) into fixed-length segments.
2. **Feature engineering** — per-window statistical features: mean, std, min,
   max, RMS, zero-crossing rate, correlation between axes (matches the
   feature set used in the published cattle-behavior literature).
3. **Models** — Logistic Regression, Random Forest, and SVM (RBF), compared
   via stratified 5-fold CV.
4. **Evaluation** — accuracy, precision/recall/F1 (macro, since behaviors are
   imbalanced — lying/grazing dominate), ROC-AUC (one-vs-rest), confusion matrix.

### Run
```bash
pip install -r requirements.txt
python train.py --data ./data/cattle_behavior.csv
```

### Deliverables produced by `train.py`
- `outputs/model_rf.pkl`, `outputs/model_svm.pkl` — trained models
- `outputs/metrics.json` — accuracy/precision/recall/F1/ROC-AUC per model
- `outputs/confusion_matrix.png`
- `outputs/feature_importance.png` (Random Forest)
