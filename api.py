from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from PIL import Image
import io
import base64
import random

app = Flask(__name__)

# CORS Configuration
CORS(app, resources={
    r"/*": {
        "origins": ["*"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Class names for simulation
class_names = [
    'Bercak_bakteri',
    'Bercak_daun_Septoria', 
    'Bercak_Target',
    'Bercak_daun_awal',
    'Busuk_daun_lanjut',
    'Embun_tepung',
    'Jamur_daun',
    'Sehat',
    'Tungau_dua_bercak',
    'Virus_keriting_daun_kuning',
    'Virus_mosaik_tomat',
]

def get_disease_info(disease_name):
    """Get disease information"""
    info = {
        'Bercak_bakteri': {
            'name': 'Bercak Bakteri',
            'symptoms': 'Bercak coklat kecil dengan tepi kuning pada daun, buah, dan batang',
            'causes': 'Bakteri Xanthomonas campestris',
            'prevention': 'Gunakan benih bebas penyakit, hindari penyiraman dari atas, rotasi tanaman',
            'treatment': 'Gunakan bakterisida yang mengandung tembaga, praktikkan rotasi tanaman',
            'severity': 'sedang'
        },
        'Bercak_daun_Septoria': {
            'name': 'Bercak Daun Septoria',
            'symptoms': 'Bercak bulat kecil dengan pusat abu-abu dan tepi coklat pada daun',
            'causes': 'Jamur Septoria lycopersici',
            'prevention': 'Hindari penyiraman dari atas, mulsa tanah, rotasi tanaman',
            'treatment': 'Hapus daun yang terinfeksi dan gunakan fungisida yang mengandung tembaga',
            'severity': 'sedang'
        },
        'Bercak_Target': {
            'name': 'Bercak Target',
            'symptoms': 'Lesi coklat dengan pola cincin target pada daun dan buah',
            'causes': 'Jamur Corynespora cassiicola',
            'prevention': 'Jaga sirkulasi udara, hindari penanaman terlalu rapat',
            'treatment': 'Gunakan fungisida dan hindari penanaman rapat',
            'severity': 'sedang'
        },
        'Bercak_daun_awal': {
            'name': 'Bercak Daun Awal',
            'symptoms': 'Lesi coklat dengan cincin konsentris pada daun, dimulai dari daun bawah',
            'causes': 'Jamur Alternaria solani',
            'prevention': 'Jaga drainase yang baik, hindari stres pada tanaman, mulsa tanah',
            'treatment': 'Gunakan fungisida yang mengandung chlorothalonil, buang daun yang terinfeksi',
            'severity': 'sedang'
        },
        'Busuk_daun_lanjut': {
            'name': 'Busuk Daun Lanjut',
            'symptoms': 'Bercak berair yang menjadi coklat pada daun dan batang, bulu putih di bawah daun',
            'causes': 'Oomycete Phytophthora infestans',
            'prevention': 'Hindari kelembaban tinggi, sirkulasi udara yang baik, tanam varietas tahan',
            'treatment': 'Gunakan fungisida sistemik seperti metalaxyl, hancurkan tanaman yang terinfeksi',
            'severity': 'tinggi'
        },
        'Embun_tepung': {
            'name': 'Embun Tepung',
            'symptoms': 'Lapisan putih seperti tepung pada permukaan daun',
            'causes': 'Jamur Leveillula atau Oidium',
            'prevention': 'Jaga sirkulasi udara, hindari kelembaban',
            'treatment': 'Gunakan fungisida sulfur atau potassium bicarbonate',
            'severity': 'sedang'
        },
        'Jamur_daun': {
            'name': 'Jamur Daun',
            'symptoms': 'Bercak kuning pada permukaan atas daun, lapisan fuzzy hijau-abu di bawah daun',
            'causes': 'Jamur Passalora fulva',
            'prevention': 'Tingkatkan sirkulasi udara, kurangi kelembaban, jaga jarak tanam',
            'treatment': 'Tingkatkan sirkulasi udara dan gunakan fungisida yang sesuai',
            'severity': 'sedang'
        },
        'Sehat': {
            'name': 'Tanaman Sehat',
            'symptoms': 'Daun hijau segar tanpa bercak',
            'causes': 'Tidak ada penyakit',
            'prevention': 'Pertahankan kondisi optimal',
            'treatment': 'Tanaman sehat, lanjutkan perawatan optimal',
            'severity': 'tidak ada'
        },
        'Tungau_dua_bercak': {
            'name': 'Tungau Dua Bercak',
            'symptoms': 'Daun menguning, bintik putih kecil, jaring laba-laba halus',
            'causes': 'Tungau Tetranychus urticae',
            'prevention': 'Jaga kelembaban udara, hindari stres kekeringan',
            'treatment': 'Gunakan mitisida atau sabun insektisida',
            'severity': 'sedang'
        },
        'Virus_keriting_daun_kuning': {
            'name': 'Virus Keriting Daun Kuning',
            'symptoms': 'Daun menguning, menggulung ke atas, pertumbuhan terhambat',
            'causes': 'Virus TYLCV oleh kutu kebul',
            'prevention': 'Kendalikan kutu kebul, gunakan mulsa reflektif',
            'treatment': 'Tanam varietas tahan, kendalikan kutu kebul',
            'severity': 'tinggi'
        },
        'Virus_mosaik_tomat': {
            'name': 'Virus Mosaik Tomat',
            'symptoms': 'Pola mosaik hijau terang dan gelap pada daun, daun keriting',
            'causes': 'Virus TMV yang menular',
            'prevention': 'Benih bebas virus, sterilisasi alat',
            'treatment': 'Hancurkan tanaman terinfeksi, sterilisasi alat',
            'severity': 'tinggi'
        }
    }
    
    return info.get(disease_name, {
        'name': disease_name,
        'symptoms': 'Informasi tidak tersedia',
        'causes': 'Tidak diketahui',
        'prevention': 'Konsultasikan dengan ahli pertanian',
        'treatment': 'Konsultasikan dengan ahli setempat',
        'severity': 'unknown'
    })

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def is_healthy_plant(class_name):
    """Determine if the predicted class represents a healthy plant"""
    healthy_classes = ['Sehat', 'healthy', 'Tanaman_Sehat']
    return class_name in healthy_classes

def simulate_prediction():
    """Simulate ML prediction with realistic results"""
    # Simulate more realistic distribution
    # Healthy plants should be less common in disease detection
    weights = [0.08, 0.12, 0.10, 0.15, 0.08, 0.10, 0.12, 0.05, 0.08, 0.06, 0.06]
    predicted_class = random.choices(class_names, weights=weights)[0]
    
    # Generate realistic confidence (higher for easier cases)
    if predicted_class == 'Sehat':
        confidence = random.uniform(0.85, 0.98)
    else:
        confidence = random.uniform(0.72, 0.95)
    
    return predicted_class, confidence

@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        response = jsonify({})
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add('Access-Control-Allow-Headers', "*")
        response.headers.add('Access-Control-Allow-Methods', "*")
        return response

@app.route('/', methods=['GET'])
def home():
    """Home endpoint"""
    return jsonify({
        'success': True,
        'message': 'Tomato Disease Classification API',
        'version': '1.0.0 (Demo Mode)',
        'status': 'API is working with simulated AI predictions',
        'note': 'This demo uses simulated ML predictions. Real model will be integrated soon.',
        'endpoints': {
            'health': '/health',
            'predict': '/predict (POST)',
            'diseases': '/diseases',
            'test_classes': '/test-classes'
        },
        'supported_formats': ['JPG', 'JPEG', 'PNG', 'GIF'],
        'max_file_size': '16MB'
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Check API status"""
    return jsonify({
        'success': True,
        'message': 'API is running',
        'status': 'healthy',
        'mode': 'demo_simulation',
        'ml_model': 'simulated (real model coming soon)',
        'api_version': '1.0.0'
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Simulate disease prediction"""
    print("🔍 Predict endpoint called")
    print(f"📁 Files in request: {list(request.files.keys())}")
    
    if 'image' not in request.files:
        print("❌ No 'image' key in request.files")
        return jsonify({'success': False, 'error': 'No image provided'}), 400

    file = request.files['image']
    print(f"📷 File received: {file.filename}")
    
    if file.filename == '':
        print("❌ Empty filename")
        return jsonify({'success': False, 'error': 'No image selected'}), 400

    if not allowed_file(file.filename):
        print(f"❌ Invalid file type: {file.filename}")
        return jsonify({'success': False, 'error': 'Invalid file type. Supported: JPG, JPEG, PNG, GIF'}), 400

    try:
        print("🔄 Processing image...")
        image_bytes = file.read()
        print(f"📊 Image bytes length: {len(image_bytes)}")
        
        # Basic image processing
        image = Image.open(io.BytesIO(image_bytes))
        print(f"🖼️ Original image - Mode: {image.mode}, Size: {image.size}")
        
        # Simulate ML prediction
        predicted_class, confidence = simulate_prediction()
        confidence_percentage = round(confidence * 100, 2)
        
        print(f"🤖 Simulated prediction: {predicted_class} ({confidence_percentage}%)")
        
        # Get disease information
        disease_info = get_disease_info(predicted_class)
        is_plant_healthy = is_healthy_plant(predicted_class)
        
        # Convert image to base64 for response
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        
        # Generate mock top predictions
        top_predictions = []
        remaining_classes = [c for c in class_names if c != predicted_class]
        random.shuffle(remaining_classes)
        
        # Add the main prediction
        top_predictions.append({
            'class': predicted_class,
            'confidence': confidence,
            'percentage': confidence_percentage
        })
        
        # Add 2 more random predictions with lower confidence
        for i, cls in enumerate(remaining_classes[:2]):
            mock_conf = random.uniform(0.02, min(0.15, confidence - 0.1))
            top_predictions.append({
                'class': cls,
                'confidence': mock_conf,
                'percentage': round(mock_conf * 100, 2)
            })
        
        print("✅ Simulation successful")
        return jsonify({
            'success': True,
            'mode': 'simulation',
            'note': 'This is a simulated AI prediction for demonstration purposes.',
            'data': {
                'classification': {
                    'class': predicted_class,
                    'class_name': disease_info['name'],
                    'confidence': confidence,
                    'confidence_percentage': confidence_percentage,
                    'is_healthy': is_plant_healthy,
                    'predicted_index': class_names.index(predicted_class)
                },
                'disease_info': disease_info,
                'image_info': {
                    'size': image.size,
                    'mode': image.mode,
                    'format': image.format
                },
                'debug_info': {
                    'top_predictions': top_predictions,
                    'total_classes': len(class_names),
                    'simulation_mode': True
                },
                'image_base64': image_base64
            }
        })

    except Exception as e:
        print(f"❌ Processing error: {str(e)}")
        return jsonify({'success': False, 'error': f'Image processing failed: {str(e)}'}), 500

@app.route('/test-classes', methods=['GET'])
def test_classes():
    """Return class information"""
    return jsonify({
        'success': True,
        'data': {
            'class_names': class_names,
            'num_classes': len(class_names),
            'mode': 'simulation'
        }
    })

@app.route('/diseases', methods=['GET'])
def get_diseases_info():
    """Return list of all known diseases and their descriptions"""
    try:
        data = []
        for class_name in class_names:
            data.append({
                'class': class_name,
                'info': get_disease_info(class_name)
            })
        return jsonify({'success': True, 'data': data})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    print("🚀 Starting Tomato Disease Classification API (Demo Mode)...")
    print("📝 Mode: AI Simulation (Real ML model will be integrated soon)")
    print("🎯 Features:")
    print("- ✅ Image upload and validation")
    print("- ✅ Disease classification simulation")
    print("- ✅ Detailed disease information")
    print("- ✅ Complete API structure")
    print("- 🔄 ML model integration (coming soon)")
    print(f"🌐 Endpoints available:")
    print("- GET  / (API info)")
    print("- GET  /health (health check)")
    print("- POST /predict (simulated prediction)")
    print("- GET  /diseases (disease info)")
    print("- GET  /test-classes (class info)")
    
    port = int(os.environ.get('PORT', 5000))
    print(f"🌐 Server starting on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
