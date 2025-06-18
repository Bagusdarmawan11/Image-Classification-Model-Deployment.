# Klasifikasi Gambar Sayuran dengan CNN

## Deskripsi Proyek

Proyek ini berfokus pada klasifikasi gambar berbagai jenis sayuran menggunakan pendekatan Deep Learning berbasis **Convolutional Neural Network (CNN)**. Model dikembangkan dengan framework **TensorFlow** dan **Keras**, serta dikonversi ke berbagai format agar dapat digunakan di perangkat mobile, web, dan backend.

## Tujuan

- Membangun model klasifikasi gambar sayuran yang akurat.
- Mengintegrasikan model dalam berbagai platform melalui konversi ke:
  - TensorFlow Lite (TFLite)
  - TensorFlow.js (TFJS)
  - SavedModel

## Dataset

Dataset diambil dari Kaggle: [Vegetable Image Dataset](https://www.kaggle.com/datasets/misrakahmed/vegetable-image-dataset). Dataset mencakup 15 kelas sayuran dan terdiri dari folder `train`, `test`, dan `validation`.

Contoh kelas:
- Bean
- Bitter_Gourd
- Bottle_Gourd
- Brinjal
- Broccoli
- Cabbage
- Capsicum
- Carrot
- Cauliflower
- Cucumber
- Papaya
- Potato
- Pumpkin
- Radish
- Tomato
- Turnip

Semua gambar digabungkan ke dalam satu folder (`Dataset Sayuran`) untuk proses pelatihan.

## Arsitektur Model

Model CNN yang dibangun terdiri dari:
- Beberapa lapisan Conv2D dan MaxPooling2D
- Flatten layer
- Dense layer dengan aktivasi softmax

Fungsi aktivasi yang digunakan: ReLU dan Softmax.

Augmentasi data dilakukan menggunakan `ImageDataGenerator`.

## Evaluasi Model

Model dilatih menggunakan early stopping dan checkpoint terbaik berdasarkan akurasi validasi.

Hasil evaluasi:
- **Akurasi Training**: ~97.87%
- **Akurasi Validasi**: ~97.19%

Disertai visualisasi akurasi dan loss pada proses training.

## Konversi Model

Model akhir dikonversi ke format:
- **TFLite**:
```python
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open("model.tflite", "wb") as f:
    f.write(tflite_model)
```

- **TFJS**:
```python
import tensorflowjs as tfjs
tfjs.converters.save_keras_model(model, "tfjs_model")
```

- **SavedModel**:
```python
model.export("saved_model")
```

## 📁 Struktur Direktori

```
Klasifikasi-Gambar-Sayuran/
├── notebook.ipynb                      # Notebook pelatihan dan export model
├── README.md                           # Dokumentasi proyek
├── requirements.txt                    # Dependency Python
├── saved_model/                        # Format TensorFlow SavedModel
│   ├── fingerprint.pb
│   ├── saved_model.pb
│   └── variables/
│       ├── variables.data-00000-of-00001
│       └── variables.index
├── tfjs_model/                         # Model untuk TensorFlow.js (website)
│   ├── model.json
│   └── group1-shard1of1.bin
├── tflite/                             # Model untuk perangkat mobile (TFLite)
│   └── model.tflite
```

## Cara Menjalankan

### Google Colab (Direkomendasikan)
1. Upload `notebook.ipynb` dan `kaggle.json` ke Google Collab.
2. Jalankan semua sel untuk pelatihan dan konversi model.

### Lokal
1. Pastikan menggunakan Python 3.x.
2. Instal dependensi:
```bash
pip install -r requirements.txt
```
3. Jalankan `notebook.ipynb` melalui Jupyter Notebook.

## Lisensi dan Referensi

- Dataset: [Kaggle - Vegetable Image Dataset](https://www.kaggle.com/datasets/misrakahmed/vegetable-image-dataset)
- Framework: [TensorFlow](https://www.tensorflow.org/), [TensorFlow.js](https://www.tensorflow.org/js)
