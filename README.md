# 🍅 Tomato Disease Classification API

API untuk mengklasifikasi penyakit pada daun tomat menggunakan deep learning dengan TensorFlow.

## 🚀 Features

- **11 Klasifikasi Penyakit**: Deteksi berbagai jenis penyakit pada daun tomat
- **Validasi Gambar**: Memastikan input adalah gambar daun tomat yang valid
- **Model Download Otomatis**: Model didownload otomatis dari Google Drive saat deployment
- **CORS Support**: Dapat diakses dari frontend web
- **Detailed Response**: Informasi lengkap tentang penyakit, gejala, penyebab, dan pengobatan

## 📋 Klasifikasi Penyakit

1. **Bercak Bakteri** - Bacterial Spot
2. **Bercak Daun Septoria** - Septoria Leaf Spot  
3. **Bercak Target** - Target Spot
4. **Bercak Daun Awal** - Early Blight
5. **Busuk Daun Lanjut** - Late Blight
6. **Embun Tepung** - Powdery Mildew
7. **Jamur Daun** - Leaf Mold
8. **Sehat** - Healthy
9. **Tungau Dua Bercak** - Two Spotted Spider Mite
10. **Virus Keriting Daun Kuning** - Tomato Yellow Leaf Curl Virus
11. **Virus Mosaik Tomat** - Tomato Mosaic Virus

## 🛠️ Tech Stack

- **Backend**: Flask + TensorFlow
- **Image Processing**: OpenCV + PIL
- **Deployment**: Railway
- **Model**: Custom CNN trained on tomato leaf dataset

## 📡 API Endpoints

### 1. Home
```
GET /
```
Informasi dasar tentang API

### 2. Health Check
```
GET /health
```
Status API dan model

### 3. Predict Disease
```
POST /predict
```
**Request**: Form-data dengan key `image` (file gambar)

**Response**:
```json
{
  "success": true,
  "data": {
    "classification": {
      "class": "Bercak_bakteri",
      "class_name": "Bercak Bakteri",
      "confidence": 0.95,
      "confidence_percentage": 95.0,
      "is_healthy": false
    },
    "disease_info": {
      "name": "Bercak Bakteri",
      "symptoms": "Bercak coklat kecil dengan tepi kuning...",
      "causes": "Bakteri Xanthomonas campestris",
      "prevention": "Gunakan benih bebas penyakit...",
      "treatment": "Gunakan bakterisida yang mengandung tembaga...",
      "severity": "sedang"
    }
  }
}
```

### 4. Get All Diseases Info
```
GET /diseases
```
Informasi lengkap semua penyakit

### 5. Test Classes
```
GET /test-classes
```
Debug endpoint untuk melihat urutan kelas

## 🔧 Usage Example

### Using cURL
```bash
curl -X POST https://your-railway-app.railway.app/predict \
  -F "image=@path/to/tomato_leaf.jpg"
```

### Using Python requests
```python
import requests

url = "https://your-railway-app.railway.app/predict"
files = {"image": open("tomato_leaf.jpg", "rb")}
response = requests.post(url, files=files)
print(response.json())
```

### Using JavaScript/Fetch
```javascript
const formData = new FormData();
formData.append('image', imageFile);

fetch('https://your-railway-app.railway.app/predict', {
  method: 'POST',
  body: formData
})
.then(response => response.json())
.then(data => console.log(data));
```

## 🎯 Image Requirements

- **Format**: JPG, JPEG, PNG, GIF
- **Size**: Max 16MB
- **Content**: Daun tomat yang jelas
- **Background**: Kontras dengan daun
- **Quality**: Cukup detail untuk analisis

## 🔍 Validation Features

API melakukan validasi berlapis:

1. **Color Analysis**: Deteksi dominasi warna hijau
2. **Edge Detection**: Analisis struktur daun
3. **Aspect Ratio**: Validasi proporsi gambar
4. **Brightness/Contrast**: Kualitas pencahayaan
5. **Model Confidence**: Threshold confidence minimum

## 🚀 Deployment

Deploy otomatis ke Railway dengan:
1. Push ke GitHub repository
2. Connect Railway ke repository
3. Model akan didownload otomatis saat startup
4. API ready dalam beberapa menit

## 📊 Model Information

- **Architecture**: Custom CNN
- **Input Size**: 224x224x3
- **Preprocessing**: ResNet50 preprocessing
- **Classes**: 11 disease categories
- **Model Size**: ~202MB

## 🔒 Error Handling

API menangani berbagai error:
- Invalid file format
- Non-leaf images
- Low quality images
- Model prediction errors
- Network timeouts

## 📝 Response Codes

- **200**: Success
- **400**: Bad Request (invalid image, validation failed)
- **500**: Internal Server Error (model error)

## 🤝 Contributing

1. Fork repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Create Pull Request

## 📄 License

MIT License - see LICENSE file for details