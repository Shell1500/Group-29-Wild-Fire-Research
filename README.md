# Wildfire Burnt‐Area Segmentation and Progression Analysis

This repository contains four Jupyter notebooks that walk through:

1. **Exploring & loading the Turkish Wild Fires dataset**  
2. **Baseline burnt‐area segmentation model**  
3. **Improvement 1: Lightweight UNet (PyTorch)**  
4. **Improvement 2: UNet on TFRecord wildfire progression data (TensorFlow)**  

---

## File Structure

- `29_Dataset.ipynb`  
  - **Purpose:** Download, inspect and visualize the Turkish Wild Fires imagery and binary‐mask dataset.  
  - **Highlights:**  
    - Uses `rasterio` to open georeferenced images  
    - Plots samples with `matplotlib`  
    - Loads via HuggingFace Datasets API:  
      ```python
      from datasets import load_dataset
      ds = load_dataset("...")
      ```  

- `29_baseline_model.ipynb`  
  - **Purpose:** Build and train a **ResNet-UNet** segmentation baseline on Turkish Wild Fires.  
  - **Highlights:**  
    - Colab drive mount to fetch data  
    - Custom `Dataset` & `DataLoader` in PyTorch  
      ```python
      class WildfireDataset(Dataset):
          def __init__(...): ...
          def __len__(...): ...
          def __getitem__(...): ...
      ```  
    - ResNet-18 encoder + UNet decoder via `torchvision.models`  
    - Training loop with `tqdm`, `torch.optim.Adam`, weight decay  
    - Visualizations: loss curves, sample predictions  

- `29_improvement_1.ipynb`  
  - **Purpose:** Replace heavy ResNet-UNet with a **LightweightUNet** in PyTorch.  
  - **Highlights:**  
    - Same Turkish Wild Fires data pipeline as baseline  
    - New `LightweightUNet` class (1.08 M params vs 14.4 M)  
    - All layers trained from scratch, base channels tunable  
      ```python
      class LightweightUNet(nn.Module):
          def __init__(self, base_c=16): ...
      ```  
    - Training & evaluation analogous to baseline, with side‐by‐side metric comparison  

- `29_improvement_2.ipynb`  
  - **Purpose:** Train a UNet model on **wildfire progression** TFRecord dataset (NDWS Western).  
  - **Highlights:**  
    - TFRecord parsing & pipeline with `tf.data`  
      ```python
      def _parse_function(example_proto): ...
      def get_dataset(tfrecord_paths): ...
      ```  
    - Custom Dice coefficient metric in Keras  
    - UNet built in TensorFlow/Keras, compiled with `BinaryCrossentropy` + `Adam`  
    - Callbacks: `EarlyStopping`, `ReduceLROnPlateau`, `ModelCheckpoint`  
    - Visualize sample predictions and save the final model  

---

## Requirements

Tested on Google Colab with A100 GPU ENV. Key dependencies:

- Data & plotting  
  - `numpy`  
  - `matplotlib`  
  - `rasterio`  
- Dataset loading  
  - `datasets`
- Baseline & Improvement 1 (PyTorch)  
  - `torch`, `torchvision`  
  - `scikit-learn`  
  - `tqdm`  
- Improvement 2 (TensorFlow)  
  - `tensorflow>=2.x` 
