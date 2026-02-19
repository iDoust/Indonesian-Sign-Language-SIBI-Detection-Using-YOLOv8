# 🤟 SIBI Alphabet Detection — YOLOv8

Deteksi huruf alfabet **Bahasa Isyarat Indonesia (SIBI)** secara real-time menggunakan **YOLOv8**. Model mengenali **26 huruf (A-Z)** dari gesture tangan.

![Training Results](Result/results.png)

## 📊 Performa Model

| Metrik | Nilai |
|---|---|
| **Precision** | 99.66% |
| **Recall** | 99.32% |
| **mAP50** | 99.26% |
| **mAP50-95** | 88.00% |

> Trained with YOLOv8n, 20 epochs, Adam optimizer, 640×640 image size.

<details>
<summary>📉 Confusion Matrix</summary>

![Confusion Matrix](Result/confusion_matrix.png)

Huruf dengan akurasi lebih rendah: C (49), W (55), G (57), P (60).

</details>

## 📁 Struktur File

```
## 📁 Project Structure
📁 SIBI Detection Using YOLOv8/
├── 📁 Notebooks/           # Original Jupyter Notebooks
│   ├── 1. YOLOv8-Training.ipynb
│   └── 2. YOLOv8-Predict.ipynb
├── 📁 Scripts/             # Standalone Python Scripts
│   ├── 1. train.py
│   └── 2. predict.py
├── 📁 Model/               # Model weights (.pt, .onnx)
│   ├── best.pt                  # Model terbaik (PyTorch)
│   ├── best.onnx                # Model terbaik (ONNX / deployment)
│   └── last.pt                  # Model iterasi terakhir
├── 📁 Result/              # Training results & metrics
│   ├── results.csv              # Training metrics per epoch
│   ├── results.png              # Training curves
│   └── confusion_matrix.png
├── .env                    # Roboflow API Key
├── requirements.txt
└── README.md

## 🚀 Usage Guide

### 1. Installation
Install the required dependencies:
```bash
pip install -r requirements.txt
```

### 2. Training the Model
Run the training script from the project root:
```bash
python Scripts/1. train.py --epochs 50 --model yolov8n.pt
```

### 3. Running Predictions
Standard predictions on images, folders, or webcam:
```bash
# Predicted from image
python Scripts/2. predict.py --source image.jpg

# Predicted from folder
python Scripts/2. predict.py --source data/test/images/

# Real-time webcam detection
python Scripts/2. predict.py --webcam
```
# Simpan hasil ke folder output/
python Scripts/2. predict.py --source image.jpg --save

# Custom confidence threshold
python Scripts/2. predict.py --source image.jpg --conf 0.7

# Gunakan model PyTorch (default: ONNX)
python "2. predict.py" --source image.jpg --model Model/best.pt
```

## 🏋️ Training

```bash
# Training default (20 epochs, YOLOv8n)
python "1. train.py"

# Custom training
python "1. train.py" --epochs 100 --model yolov8s.pt --imgsz 640

# Lihat semua opsi
python "1. train.py" --help
```

## 🛠 Teknologi

- **Python** 3.x
- **YOLOv8** (Ultralytics) — Object Detection
- **PyTorch** — Deep Learning Framework
- **OpenCV** — Computer Vision
- **ONNX** — Cross-platform Model Format
- **Roboflow** — Dataset Management

## 📊 Arsitektur Model

```
Input (640×640) → CSPDarknet (Backbone) → PANet/FPN (Neck) → Detection Head
                                                                ├── Bounding Box
                                                                ├── Confidence Score
                                                                └── Class (A-Z)
```

## 📚 Dataset

Dataset gesture tangan bersumber dari Roboflow:
🔗 [USIBI Dataset — Roboflow](https://universe.roboflow.com/usibi-image-translate/usibi-jueew)

- **26 kelas**: Huruf alfabet A-Z
- **Format**: YOLOv8 (bounding box annotations)
- **Augmentasi**: Default Ultralytics (mosaic, flipping, scaling)

## 📝 Lisensi

MIT License
