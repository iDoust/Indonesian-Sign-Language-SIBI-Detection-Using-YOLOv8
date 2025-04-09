# SIBI-Indonesian-Sign-Language-Detection
# Deteksi Bahasa Isyarat Indonesia (SIBI) Menggunakan YOLOv8

Proyek ini menggunakan YOLOv8 (You Only Look Once versi 8) untuk melakukan deteksi objek pada dataset khusus. Model ini dilatih menggunakan Ultralytics YOLOv8 dan diuji dengan berbagai data untuk menghasilkan model deteksi objek yang akurat.

Dataset gesture tangan yang digunakan dalam proyek ini bersumber dari Roboflow dan telah dikonfigurasi untuk mendukung pendeteksian keypoint gesture tangan: 🔗 [USIBI Dataset – Roboflow](https://universe.roboflow.com/usibi-image-translate/usibi-jueew)  

## 🔧 Fitur Utama  
- Sistem merekam gesture tangan dalam bentuk gambar untuk tiap label bahasa isyarat.
- Data diberi label menggunakan format standar YOLO dan juga dilengkapi dengan keypoint detection untuk mengenali posisi jari dan tangan secara spesifik.
- Menggunakan algoritma YOLOv8 untuk mendeteksi posisi dan jenis isyarat secara cepat dan efisien.  
- Model diuji pada data uji terpisah untuk mengevaluasi akurasi dan performa dalam mengenali gestur tangan.
- Model akhir diekspor ke format .pt dan .onnx sehingga dapat di-deploy ke berbagai platform (web, mobile, embedded).  

## 📁 Struktur File  
- YOLOv8.ipynb  # Notebook pelatihan model.  
- YOLOv8-Testing.ipynb  # Notebook pengujian model.  
- best.pt  # Model terbaik dalam format PyTorch.  
- best.onnx  # Model terbaik dalam format ONNX (untuk deployment).  
- last.pt  # Model terakhir dari proses training.
- README.md

## 🚀 Teknologi yang Digunakan   
- Python 3.x  
- Ultralytics YOLOv8  
- PyTorch  
- OpenCV  
- ONNX (untuk ekspor model)  
- Pandas, NumPy  

## 📊 Arsitektur Model  
Model YOLOv8 menggunakan:  
- Backbone: CSPDarknet yang digunakan untuk ekstraksi fitur dari input gambar.  
- Neck: PANet atau FPN yang menggabungkan fitur dari berbagai tingkatan.  
- Head: Modul deteksi yang menghasilkan prediksi bounding box, confidence score, dan keypoints.
Model ini juga mendukung ekspor ke berbagai format seperti **.pt**, **.onnx**, dan **OpenVINO** untuk deployment lintas platform.

## 🎯 Tujuan Proyek  
- Menerapkan YOLOv8 untuk deteksi objek real-time.
- Mengekspor model ke format lintas platform.
- Mengevaluasi performa model dalam konteks aplikasi dunia nyata.

## 📝 Catatan  
- Dataset gesture tangan yang digunakan diambil dari Roboflow: 🔗 [USIBI Dataset – Roboflow](https://universe.roboflow.com/usibi-image-translate/usibi-jueew)
- Model YOLOv8 yang digunakan mendukung keypoint detection, sehingga cocok untuk mendeteksi titik-titik penting pada tangan (jari, telapak, dsb).  
- File yang digunakan untuk model:  
  - best.pt – Model terbaik dari hasil pelatihan  
  - last.pt – Model dari iterasi terakhir pelatihan  
  - best.onnx – Versi ONNX untuk deployment ke platform lain
- Hasil evaluasi model disimpan dalam file results.csv yang memuat informasi deteksi dan confidence score.  
- Untuk pelabelan data (annotasi bounding box dan keypoints), disarankan menggunakan tools seperti Roboflow Annotate atau CVAT.  
- YOLOv8 dilatih menggunakan konfigurasi yang disesuaikan agar optimal untuk dataset SIBI (jumlah kelas gesture, ukuran input gambar, augmentasi, dll).
