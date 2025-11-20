# Vision Transformer Comparison: Swin Transformer vs DeiT

Proyek ini membandingkan dua arsitektur Vision Transformer yang berbeda: **Swin Transformer Tiny** dan **DeiT Small Distilled** pada dataset **STL-10** untuk tugas klasifikasi gambar.

## Deskripsi Proyek

Eksperimen ini bertujuan untuk:
- Mengimplementasikan dan fine-tune dua model Vision Transformer
- Membandingkan performa berdasarkan akurasi, efisiensi, dan kecepatan
- Memberikan rekomendasi pemilihan model berdasarkan use case
- Menganalisis trade-off antara kompleksitas model dan performa

## Hasil Utama

| Metrik | Swin-Tiny | DeiT-Small | 
|--------|-----------|------------|
| Test Accuracy | **97.36%** | 97.01% | 
| Inference Time/img | 5.62 ms | **1.92ms** | 
| Throughput | 178 img/s | **520 img/s** | 
| Training Time | 8.2 min | **5 min** | 
| Parameters | 27.5M | **21.6M** | 

## Dataset

**STL-10** (Self-Taught Learning)
- 10 kelas: airplane, bird, car, cat, deer, dog, horse, monkey, ship, truck
- Custom split:
  - Training: 4,000 gambar
  - Validation: 1,000 gambar
  - Test: 8,000 gambar
- Image size: 224×224 (resized dari 96×96)

## Arsitektur Model

### Swin Transformer Tiny
- Model: `swin_tiny_patch4_window7_224`
- Patch size: 4×4
- Embed dimension: 96
- Depths: [2, 2, 6, 2]
- Heads: [3, 6, 12, 24]
- Hierarchical architecture dengan shifted window attention

### DeiT Small Distilled
- Model: `deit_small_distilled_patch16_224`
- Patch size: 16×16
- Embed dimension: 384
- Depth: 12 layers
- Heads: 6
- Global attention dengan knowledge distillation

## Setup dan Instalasi

### Requirements

```bash
torch>=2.0.0
torchvision>=0.15.0
timm>=0.9.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
tqdm>=4.65.0
```

### Instalasi

```bash
# Clone repository
git clone https://github.com/bintangfikrif/VisionTransformer-Comparison.git
cd VisionTransformer-Comparison

# Install dependencies
pip install -r requirements.txt
```

## Cara Menjalankan

### Training dari Scratch

```python
# Buka notebook di Google Colab atau Jupyter
jupyter notebook notebooks/vision_transformer_comparison_complete.ipynb
```

### Menggunakan Pre-trained Model

```python
import torch
import timm

# Load Swin Transformer Tiny
model_swin = timm.create_model('swin_tiny_patch4_window7_224', 
                                pretrained=True, 
                                num_classes=10)
model_swin.load_state_dict(torch.load('models/swin_tiny_best.pth'))

# Load DeiT Small
model_deit = timm.create_model('deit_small_distilled_patch16_224', 
                                pretrained=True, 
                                num_classes=10)
model_deit.load_state_dict(torch.load('models/deit_small_best.pth'))
```

## Hasil Eksperimen

### Performa Per Kelas (Confusion Metrices)

| Kelas | Swin-Tiny | DeiT-Small |
|-------|-----------|------------|
| Airplane | 0.99 | 0.99 |
| Bird | 0.98 | 0.97 |
| Car | 0.97 | 0.98 |
| Cat | 0.96 | 0.95 |
| Deer | 0.97 | 0.96 |
| Dog | 0.94 | 0.94 |
| Horse | 0.98 | 0.97 |
| Monkey | 0.98 | 0.98 |
| Ship | 0.99 | 0.99 |
| Truck | 0.96 | 0.97 |

### Learning Curves

Training dan validation curves menunjukkan:
- Kedua model konvergen dengan baik tanpa overfitting signifikan
- Swin Tiny mencapai best validation accuracy di epoch 12
- DeiT Small konvergen lebih cepat di epoch 9

## Rekomendasi

### Untuk Akurasi Maksimal
**Pilih: Swin Transformer Tiny**
- Akurasi tertinggi (97.36%)
- Cocok untuk: medical imaging, quality control

### Untuk Efisiensi Komputasi
**Pilih: DeiT Small**
- Model lebih kecil
- Training lebih cepat (3 menit lebih cepat)
- Cocok untuk: mobile apps, edge devices


## Metodologi

### Training Configuration
- **Optimizer**: AdamW (lr=0.0001, weight_decay=0.01)
- **Scheduler**: CosineAnnealingLR
- **Batch size**: 16
- **Epochs**: 15

### Data Augmentation
- RandomResizedCrop
- Random horizontal flip
- Random rotation
- Color jitter (brightness, contrast, saturation, hue)
- ImageNet normalization

### Hardware
- Platform: Google Colab
- GPU: NVIDIA Tesla T4
- CUDA: 12.x

## Author

**Bintang Fikri Fauzan**  
NIM: 122140008  
Institut Teknologi Sumatera  
Teknik Informatika

**Note**: Proyek ini merupakan tugas eksplorasi untuk mata kuliah Deep Learning, Semester Ganjil 2025/2026, Institut Teknologi Sumatera.