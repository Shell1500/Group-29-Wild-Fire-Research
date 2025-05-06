# Group-29-Wild-Fire-Research
Deep Learning for Wildfire Burned Area Segmentation and Spread Prediction

## 🔍 1. Dataset Exploration & Preparation (`29_Dataset.ipynb`)

- **Datasets covered**  
  - *Turkish Wild Fires* (Sentinel-2)  
  - *EO4WildFires* (HuggingFace)  
  - *California Burned Areas* (HuggingFace)

- **Key steps**  
  1. Download / load GeoTIFF & TIFF files via `rasterio`.  
  2. Convert multispectral images → NumPy arrays.  
  3. Generate train/val splits.  
  4. Visualize sample images & masks, pixel distributions.

- **Primary libraries**  
  `rasterio`, `geopandas`, `numpy`, `matplotlib`, HuggingFace `datasets`.

---

## 🤖 2. Baseline Segmentation (`29_baseline_model.ipynb`)

- **Framework**: PyTorch  
- **Model**: ResNet-UNet (encoder-decoder)  
- **Loss**: BCE + Dice  
- **Metrics**: Accuracy & IoU  
- **Flow**:  
  1. Dataset & DataLoader setup  
  2. Model instantiation (`torch.nn`)  
  3. Training loop with `tqdm` progress bars  
  4. Validation & test metrics reporting  
  5. Plotting loss, accuracy curves

---

## 🚀 3. Improvement Round 1 (`29_improvement_1.ipynb`)

- **Framework**: PyTorch  
- **Enhancements**:  
  - Lightweight U-Net variant (~1M params)  
  - Data augmentations: flips, rotations, color jitter (`albumentations`)  
  - Early stopping & LR scheduling  
- **Outcome**: Higher IoU vs. baseline with far fewer parameters  
- **Extras**: side-by-side mask visualizations  

---

## 📈 4. Progression & Final Segmentation (`29_improvement_2.ipynb`)

- **Framework**: TensorFlow & Keras  
- **Dataset**: NDWS Western (Next Day Wildfire Spread) in TFRecord  
- **Pipeline**:  
  1. Download via Kaggle API  
  2. `tf.data` pipeline: parsing, normalization, caching, batching  
  3. Custom `DiceCoefficient` metric  
  4. U-Net model built in Keras (Conv2D, Conv2DTranspose, BatchNorm, etc.)  
  5. Training with `ModelCheckpoint` & `ReduceLROnPlateau` callbacks  
  6. Test-set evaluation & sample predictions  
  7. Exporting the best model

---
