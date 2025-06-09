from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import random
# Import PIL only when needed to avoid startup issues
import io
import base64
import requests

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

# Create upload folder safely
try:
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
except Exception as e:
    print(f"Warning: Could not create upload folder: {e}")

print("✅ Flask app initialized successfully")

# Model configuration
MODEL_FILE_ID = "1dIi88dezOiW1AtQCb6oSP_mXGWKmb_hX"
MODEL_PATH = "model.h5"

# Multiple model sources (in order of preference)
MODEL_URLS = {
    "drive_direct": f"https://drive.google.com/uc?export=download&id={MODEL_FILE_ID}&confirm=t",
    "drive_alt": f"https://docs.google.com/uc?export=download&id={MODEL_FILE_ID}",
}

# Global model variable
model = None

def download_model():
    """Download model from Google Drive if not exists"""
    if os.path.exists(MODEL_PATH):
        print(f"✅ Model already exists at {MODEL_PATH}")
        return True
    
    print("📥 Downloading model from Google Drive...")
    try:
        # Try multiple download methods
        methods = [
            ("gdown with fuzzy", lambda: gdown.download(f"https://drive.google.com/uc?id={MODEL_FILE_ID}", MODEL_PATH, quiet=False, fuzzy=True)),
            ("direct requests", lambda: download_with_requests()),
            ("gdown direct", lambda: gdown.download(f"https://drive.google.com/uc?id={MODEL_FILE_ID}", MODEL_PATH, quiet=False))
        ]
        
        for method_name, method_func in methods:
            try:
                print(f"📥 Trying {method_name}...")
                method_func()
                
                # Verify file was downloaded and has reasonable size
                if os.path.exists(MODEL_PATH):
                    file_size = os.path.getsize(MODEL_PATH)
                    if file_size > 1024 * 1024:  # At least 1MB
                        print(f"✅ Model downloaded successfully using {method_name}! Size: {file_size / (1024*1024):.1f} MB")
                        return True
                    else:
                        print(f"⚠️ Downloaded file too small ({file_size} bytes), trying next method...")
                        os.remove(MODEL_PATH)
                else:
                    print(f"⚠️ File not found after {method_name}, trying next method...")
                    
            except Exception as e:
                print(f"⚠️ {method_name} failed: {e}")
                if os.path.exists(MODEL_PATH):
                    os.remove(MODEL_PATH)
                continue
        
        print("❌ All download methods failed")
        return False
        
    except Exception as e:
        print(f"❌ Error in download_model: {e}")
        return False

def download_model():
    """Download model from multiple sources if not exists"""
    if os.path.exists(MODEL_PATH):
        file_size = os.path.getsize(MODEL_PATH)
        print(f"✅ Model already exists at {MODEL_PATH} ({file_size / (1024*1024):.1f} MB)")
        return True
    
    print("📥 Downloading model from available sources...")
    
    # Try multiple download methods in order of reliability
    download_methods = [
        ("GitHub Release", download_from_github),
        ("Google Drive (gdown)", download_with_gdown),
        ("Google Drive (requests)", download_with_requests),
    ]
    
    for method_name, method_func in download_methods:
        try:
            print(f"📥 Trying {method_name}...")
            if method_func():
                print(f"✅ Successfully downloaded model using {method_name}!")
                return True
            else:
                print(f"⚠️ {method_name} failed, trying next method...")
        except Exception as e:
            print(f"⚠️ {method_name} error: {e}")
            if os.path.exists(MODEL_PATH):
                os.remove(MODEL_PATH)
    
    print("❌ All download methods failed")
    return False

def download_from_url(url, source_name):
    """Generic download function for any URL"""
    import requests
    
    try:
        print(f"📥 Downloading from {source_name}: {url}")
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()
        
        # Check content type
        content_type = response.headers.get('content-type', '')
        if 'text/html' in content_type.lower():
            print(f"⚠️ Received HTML instead of binary file from {source_name}")
            return False
        
        total_size = int(response.headers.get('content-length', 0))
        print(f"📊 {source_name} file size: {total_size / (1024*1024):.1f} MB")
        
        with open(MODEL_PATH, 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0 and downloaded % (1024*1024*10) == 0:  # Print every 10MB
                        percent = (downloaded / total_size) * 100
                        print(f"📥 Downloaded: {percent:.1f}%")
        
        # Verify download
        if os.path.exists(MODEL_PATH):
            file_size = os.path.getsize(MODEL_PATH)
            if file_size > 1024 * 1024:  # At least 1MB
                print(f"✅ {source_name} download completed! Size: {file_size / (1024*1024):.1f} MB")
                return True
            else:
                print(f"⚠️ {source_name} file too small ({file_size} bytes)")
                os.remove(MODEL_PATH)
        
        return False
        
    except Exception as e:
        print(f"❌ {source_name} download failed: {e}")
        return False

def download_model():
    """Download model from multiple sources if not exists"""
    if os.path.exists(MODEL_PATH):
        file_size = os.path.getsize(MODEL_PATH)
        print(f"✅ Model already exists at {MODEL_PATH} ({file_size / (1024*1024):.1f} MB)")
        return True
    
    print("📥 Downloading model from available sources...")
    
    # Try each source in order
    for source_name, url in MODEL_URLS.items():
        try:
            print(f"📥 Trying {source_name}...")
            if download_from_url(url, source_name):
                return True
        except Exception as e:
            print(f"⚠️ {source_name} error: {e}")
            if os.path.exists(MODEL_PATH):
                os.remove(MODEL_PATH)
    
    # Enhanced fallback with different gdown approaches
    print("📥 Trying enhanced gdown approaches...")
    gdown_methods = [
        ("gdown_fuzzy", lambda: gdown_download_fuzzy()),
        ("gdown_no_check", lambda: gdown_download_no_check()),
        ("manual_session", lambda: manual_drive_session()),
    ]
    
    for method_name, method_func in gdown_methods:
        try:
            print(f"📥 Trying {method_name}...")
            if method_func():
                return True
        except Exception as e:
            print(f"⚠️ {method_name} failed: {e}")
            if os.path.exists(MODEL_PATH):
                os.remove(MODEL_PATH)
    
    print("❌ All download methods failed")
    return False

def gdown_download_fuzzy():
    """Try gdown with fuzzy matching"""
    try:
        import gdown
        result = gdown.download(f"https://drive.google.com/uc?id={MODEL_FILE_ID}", 
                              MODEL_PATH, quiet=False, fuzzy=True)
        return verify_download()
    except Exception as e:
        print(f"gdown_fuzzy error: {e}")
        return False

def gdown_download_no_check():
    """Try gdown without verification"""
    try:
        import gdown
        result = gdown.download(f"https://drive.google.com/uc?id={MODEL_FILE_ID}", 
                              MODEL_PATH, quiet=False, verify=False)
        return verify_download()
    except Exception as e:
        print(f"gdown_no_check error: {e}")
        return False

def manual_drive_session():
    """Manual Google Drive session handling"""
    import requests
    
    try:
        session = requests.Session()
        
        # Step 1: Get the file page
        url1 = f"https://drive.google.com/file/d/{MODEL_FILE_ID}/view"
        response1 = session.get(url1)
        
        # Step 2: Try direct download
        url2 = f"https://drive.google.com/uc?export=download&id={MODEL_FILE_ID}"
        response2 = session.get(url2, stream=True)
        
        # Step 3: Handle confirmation if needed
        if 'download_warning' in response2.cookies:
            confirm_token = response2.cookies['download_warning']
            url3 = f"https://drive.google.com/uc?export=download&id={MODEL_FILE_ID}&confirm={confirm_token}"
            response2 = session.get(url3, stream=True)
        
        # Save file
        with open(MODEL_PATH, 'wb') as f:
            for chunk in response2.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        return verify_download()
        
    except Exception as e:
        print(f"manual_session error: {e}")
        return False

def verify_download():
    """Verify downloaded file"""
    if os.path.exists(MODEL_PATH):
        file_size = os.path.getsize(MODEL_PATH)
        if file_size > 1024 * 1024:  # At least 1MB
            print(f"✅ Download verified! Size: {file_size / (1024*1024):.1f} MB")
            return True
        else:
            print(f"⚠️ File too small ({file_size} bytes)")
            os.remove(MODEL_PATH)
    return False

def download_with_requests():
    """Alternative download method using requests with session"""
    import requests
    
    try:
        session = requests.Session()
        
        # First request to get the file
        print(f"📥 Requesting file from Google Drive...")
        response = session.get(f"https://drive.google.com/uc?export=download&id={MODEL_FILE_ID}", stream=True)
        
        # Check for virus scan warning and get download link
        token = None
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                token = value
                break
        
        if token:
            print("📥 Handling Google Drive warning, getting actual download link...")
            params = {'id': MODEL_FILE_ID, 'confirm': token}
            response = session.get("https://drive.google.com/uc?export=download", params=params, stream=True)
        
        response.raise_for_status()
        
        # Check content type
        content_type = response.headers.get('content-type', '')
        if 'text/html' in content_type.lower():
            print("❌ Received HTML page instead of file - check if file is public")
            return False
        
        # Download the file
        total_size = int(response.headers.get('content-length', 0))
        print(f"📊 Starting download... Expected size: {total_size / (1024*1024):.1f} MB")
        
        with open(MODEL_PATH, 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0 and downloaded % (1024*1024*10) == 0:  # Print every 10MB
                        percent = (downloaded / total_size) * 100
                        print(f"📥 Downloaded: {percent:.1f}%")
        
        # Verify download
        if os.path.exists(MODEL_PATH):
            file_size = os.path.getsize(MODEL_PATH)
            print(f"✅ Download completed! Final size: {file_size / (1024*1024):.1f} MB")
            
            if file_size > 1024 * 1024:  # At least 1MB
                return True
            else:
                print("❌ Downloaded file too small, might be an error page")
                os.remove(MODEL_PATH)
                return False
        else:
            print("❌ File not found after download")
            return False
            
    except Exception as e:
        print(f"❌ Download with requests failed: {e}")
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)
        return False

def load_model():
    """Load the TensorFlow model"""
    global model
    
    if model is not None:
        print("✅ Model already loaded")
        return True
        
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model file not found at {MODEL_PATH}")
        return False
        
    try:
        # Try to import TensorFlow
        print("📦 Importing TensorFlow...")
        import tensorflow as tf
        import numpy as np
        
        # Check file size before loading
        file_size = os.path.getsize(MODEL_PATH)
        print(f"📂 Loading model from {MODEL_PATH} ({file_size / (1024*1024):.1f} MB)...")
        
        # Load model with error handling
        model = tf.keras.models.load_model(MODEL_PATH)
        print("✅ Model loaded successfully!")
        
        # Print model details
        print(f"📊 Model input shape: {model.input_shape}")
        print(f"📊 Model output shape: {model.output_shape}")
        print(f"📊 Number of classes: {model.output_shape[-1]}")
        
        # Validate model output matches our class names
        model_classes = model.output_shape[-1]
        expected_classes = len(class_names)
        if model_classes != expected_classes:
            print(f"⚠️ Model has {model_classes} classes but we expect {expected_classes}")
            print("📝 This might cause prediction issues")
        
        return True
        
    except ImportError as e:
        print(f"⚠️ TensorFlow not available: {e}")
        print("🔄 Will use simulation mode")
        return False
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("🔄 Will use simulation mode")
        
        # If model file is corrupted, delete it
        if "cannot read" in str(e).lower() or "invalid" in str(e).lower():
            print("🗑️ Removing potentially corrupted model file...")
            try:
                os.remove(MODEL_PATH)
                print("✅ Corrupted model file removed")
            except:
                pass
        
        return False

def preprocess_image(image, target_size=(224, 224)):
    """Preprocess image for model prediction"""
    try:
        import tensorflow as tf
        from tensorflow.keras.preprocessing.image import img_to_array
        from tensorflow.keras.applications.resnet50 import preprocess_input
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        image = image.resize(target_size)
        img_array = img_to_array(image)
        img_array = np.expand_dims(img_array, axis=0)
        
        # Use the same preprocessing as during training
        img_array = preprocess_input(img_array)
        
        return img_array
    except Exception as e:
        print(f"❌ Preprocessing error: {e}")
        return None

# Initialize model on startup
print("🚀 Initializing model...")
if download_model():
    load_model()
else:
    print("⚠️ Model download failed, will use simulation mode")

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

def generate_mock_predictions(predicted_class, confidence):
    """Generate mock top predictions for simulation mode"""
    top_predictions = []
    remaining_classes = [c for c in class_names if c != predicted_class]
    random.shuffle(remaining_classes)
    
    # Add the main prediction
    top_predictions.append({
        'class': predicted_class,
        'confidence': confidence,
        'percentage': round(confidence * 100, 2)
    })
    
    # Add 2 more random predictions with lower confidence
    for i, cls in enumerate(remaining_classes[:2]):
        mock_conf = random.uniform(0.02, min(0.15, confidence - 0.1))
        top_predictions.append({
            'class': cls,
            'confidence': mock_conf,
            'percentage': round(mock_conf * 100, 2)
        })
    
    return top_predictions

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
    try:
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
    except Exception as e:
        print(f"❌ Home route error: {e}")
        return jsonify({'success': False, 'error': 'Internal server error'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Check API status"""
    try:
        model_status = {
            'loaded': model is not None,
            'file_exists': os.path.exists(MODEL_PATH),
            'mode': 'real_model' if model is not None else 'simulation'
        }
        
        if model is not None:
            model_status.update({
                'input_shape': str(model.input_shape),
                'output_shape': str(model.output_shape),
                'num_classes': len(class_names)
            })
        
        return jsonify({
            'success': True,
            'message': 'API is running',
            'status': 'healthy',
            'model_status': model_status,
            'api_version': '2.0.0',
            'server_info': {
                'port': os.environ.get('PORT', 'unknown'),
                'host': '0.0.0.0'
            }
        })
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return jsonify({'success': False, 'error': 'Health check failed'}), 500

@app.route('/predict', methods=['POST'])
def predict():
    """Simulate disease prediction"""
    try:
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

        print("🔄 Processing image...")
        image_bytes = file.read()
        print(f"📊 Image bytes length: {len(image_bytes)}")
        
        # Import PIL only when needed
        try:
            from PIL import Image
            # Basic image processing
            image = Image.open(io.BytesIO(image_bytes))
            print(f"🖼️ Original image - Mode: {image.mode}, Size: {image.size}")
            
            # Convert image to base64 for response
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')
            
            image_info = {
                'size': image.size,
                'mode': image.mode,
                'format': image.format
            }
        except Exception as pil_error:
            print(f"⚠️ PIL processing failed: {pil_error}")
            # Fallback without image processing
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')
            image_info = {
                'size': 'unknown',
                'mode': 'unknown', 
                'format': 'unknown'
            }
        
        # Try real ML prediction first, fallback to simulation
        if model is not None:
            try:
                print("🤖 Using real ML model for prediction...")
                import numpy as np
                
                # Preprocess image for model
                img_array = preprocess_image(image)
                if img_array is not None:
                    # Make prediction with real model
                    prediction = model.predict(img_array, verbose=0)
                    predicted_index = np.argmax(prediction)
                    predicted_class = class_names[predicted_index]
                    confidence = float(np.max(prediction))
                    
                    print(f"🎯 Real ML prediction: {predicted_class} ({confidence*100:.2f}%)")
                    
                    # Get top 3 predictions
                    top_indices = np.argsort(prediction[0])[::-1][:3]
                    top_predictions = []
                    for idx in top_indices:
                        top_predictions.append({
                            'class': class_names[idx],
                            'confidence': float(prediction[0][idx]),
                            'percentage': round(float(prediction[0][idx]) * 100, 2)
                        })
                    
                    ml_mode = "real_model"
                else:
                    raise Exception("Image preprocessing failed")
                    
            except Exception as ml_error:
                print(f"⚠️ ML prediction failed: {ml_error}, falling back to simulation")
                predicted_class, confidence = simulate_prediction()
                top_predictions = generate_mock_predictions(predicted_class, confidence)
                ml_mode = "simulation_fallback"
        else:
            print("🎭 Using simulation mode...")
            predicted_class, confidence = simulate_prediction()
            top_predictions = generate_mock_predictions(predicted_class, confidence)
            ml_mode = "simulation"
        
        confidence_percentage = round(confidence * 100, 2)
        
        print(f"🤖 Simulated prediction: {predicted_class} ({confidence_percentage}%)")
        
        # Get disease information
        disease_info = get_disease_info(predicted_class)
        is_plant_healthy = is_healthy_plant(predicted_class)
        
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
        
        print(f"✅ Prediction successful using {ml_mode}")
        return jsonify({
            'success': True,
            'mode': ml_mode,
            'note': 'Real ML model' if ml_mode == 'real_model' else 'Simulated AI prediction for demonstration purposes.',
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
                'image_info': image_info,
                'debug_info': {
                    'top_predictions': top_predictions,
                    'total_classes': len(class_names),
                    'ml_mode': ml_mode,
                    'model_loaded': model is not None
                },
                'image_base64': image_base64
            }
        })

    except Exception as e:
        print(f"❌ Processing error: {str(e)}")
        import traceback
        traceback.print_exc()
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
    
    port = int(os.environ.get('PORT', 8080))  # Default to 8080 for Railway
    print(f"🌐 Server starting on port {port}")
    print(f"🔗 Will be available at: http://0.0.0.0:{port}")
    try:
        app.run(host='0.0.0.0', port=port, debug=False)
    except Exception as e:
        print(f"❌ Failed to start Flask app: {e}")
        raise

# Add this outside of main block for gunicorn
print("🎯 Flask application module loaded successfully")
print("📊 Available routes:")
for rule in app.url_map.iter_rules():
    print(f"   {rule.rule} -> {rule.endpoint}")
