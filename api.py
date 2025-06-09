from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf
import os
from PIL import Image
import io
import base64
import cv2

app = Flask(__name__)

# CORS Configuration - Allow Railway domain
CORS(app, resources={
    r"/*": {
        "origins": ["*"],  # Allow all origins for Railway
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

# Create uploads directory if it doesn't exist
try:
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
except:
    pass

# Load model with better error handling
model = None
model_error = None

def load_model_safely():
    """Safely load the model with detailed error reporting"""
    global model, model_error
    
    model_paths = ['model.h5', './model.h5', '/app/model.h5']
    
    for model_path in model_paths:
        try:
            print(f"🔍 Trying to load model from: {model_path}")
            
            # Check if file exists
            if os.path.exists(model_path):
                file_size = os.path.getsize(model_path)
                print(f"📁 File found! Size: {file_size} bytes")
                
                # Try to load
                model = tf.keras.models.load_model(model_path)
                print("✅ Model loaded successfully!")
                print(f"📊 Model input shape: {model.input_shape}")
                print(f"📊 Model output shape: {model.output_shape}")
                print(f"📊 Number of classes: {model.output_shape[-1]}")
                return True
            else:
                print(f"❌ File not found: {model_path}")
                
        except Exception as e:
            print(f"❌ Error loading model from {model_path}: {e}")
            model_error = str(e)
            continue
    
    # If we reach here, model loading failed
    print("❌ Failed to load model from all attempted paths")
    
    # List all files in current directory for debugging
    try:
        print("📁 Files in current directory:")
        for file in os.listdir('.'):
            if os.path.isfile(file):
                size = os.path.getsize(file)
                print(f"  - {file} ({size} bytes)")
    except:
        print("❌ Could not list files in current directory")
    
    return False

# Try to load model
load_model_safely()

# Class names - ensure order matches training
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

def validate_tomato_leaf_image(image):
    """
    Validate if image is a tomato leaf using computer vision
    Returns: (is_valid, reason, confidence)
    """
    try:
        # Convert PIL to OpenCV format
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # 1. Color Analysis - Check green color dominance
        hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
        
        # Define green color range in HSV
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        
        # Create mask for green colors
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        green_ratio = np.sum(green_mask > 0) / (green_mask.shape[0] * green_mask.shape[1])
        
        print(f"🟢 Green color ratio: {green_ratio:.3f}")
        
        # 2. Edge Detection - Check leaf structure
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_ratio = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        
        print(f"📏 Edge ratio: {edge_ratio:.3f}")
        
        # 3. Aspect Ratio - Leaves shouldn't be too extreme
        height, width = image.size[1], image.size[0]
        aspect_ratio = max(width, height) / min(width, height)
        
        print(f"📐 Aspect ratio: {aspect_ratio:.2f}")
        
        # 4. Brightness and Contrast Analysis
        gray_array = np.array(gray)
        brightness = np.mean(gray_array)
        contrast = np.std(gray_array)
        
        print(f"💡 Brightness: {brightness:.2f}, Contrast: {contrast:.2f}")
        
        # Validation Rules - More permissive
        reasons = []
        
        # Rule 1: Must have sufficient green color (at least 8%)
        if green_ratio < 0.08:
            reasons.append(f"Insufficient green color ({green_ratio*100:.1f}%)")
        
        # Rule 2: Must have reasonable edge structure
        if edge_ratio < 0.008:
            reasons.append("Image structure too simple")
        elif edge_ratio > 0.5:
            reasons.append("Image structure too complex")
        
        # Rule 3: Aspect ratio shouldn't be too extreme
        if aspect_ratio > 12:
            reasons.append(f"Aspect ratio too extreme ({aspect_ratio:.1f}:1)")
        
        # Rule 4: Brightness should be reasonable
        if brightness < 15:
            reasons.append("Image too dark")
        elif brightness > 235:
            reasons.append("Image too bright")
        
        # Rule 5: Should have reasonable contrast
        if contrast < 10:
            reasons.append("Image contrast too low")
        
        # Calculate confidence
        confidence = 0
        confidence += min(green_ratio * 3, 0.4)
        confidence += min(edge_ratio * 3, 0.3)
        confidence += max(0, 0.2 - (aspect_ratio - 1) * 0.015)
        confidence += min((brightness - 20) / 150 * 0.1, 0.1)
        
        is_valid = len(reasons) == 0 and confidence > 0.15
        
        return is_valid, reasons, confidence
        
    except Exception as e:
        print(f"❌ Validation error: {e}")
        return True, [], 0.5  # Be permissive if validation fails

def validate_with_model_confidence(prediction, confidence_threshold=0.3):
    """
    Additional validation based on model confidence
    """
    max_confidence = np.max(prediction)
    
    if max_confidence < confidence_threshold:
        # Check if prediction is evenly distributed (sign of uncertainty)
        sorted_probs = np.sort(prediction[0])[::-1]
        top_diff = sorted_probs[0] - sorted_probs[1]
        
        if top_diff < 0.12:
            return False, f"Model uncertain about prediction (confidence: {max_confidence*100:.1f}%)"
    
    return True, None

def preprocess_image(image, target_size=(224, 224)):
    """Preprocess image for model prediction"""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    image = image.resize(target_size)
    img_array = img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Normalize to [0, 1] range (adjust based on your training)
    img_array = img_array / 255.0
    
    return img_array

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def is_healthy_plant(class_name):
    """Determine if the predicted class represents a healthy plant"""
    healthy_classes = ['Sehat', 'healthy', 'Tanaman_Sehat']
    return class_name in healthy_classes

def get_disease_info(disease_name):
    """Get detailed disease information"""
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
        'symptoms': 'Information not available',
        'causes': 'Unknown',
        'prevention': 'Consult agricultural expert',
        'treatment': 'Consult local expert',
        'severity': 'unknown'
    })

# Add OPTIONS handler for preflight requests
@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        response = jsonify({})
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add('Access-Control-Allow-Headers', "*")
        response.headers.add('Access-Control-Allow-Methods', "*")
        return response

@app.route('/')
def home():
    """Root endpoint - welcome message"""
    return jsonify({
        'success': True,
        'message': 'Tomato Disease Classification API is running',
        'model_info': {
            'input_shape': str(model.input_shape) if model else None,
            'num_classes': len(class_names),
            'output_shape': str(model.output_shape) if model else None
        },
        'model_loaded': model is not None,
        'model_error': model_error if model is None else None,
        'status': 'healthy' if model else 'model_not_loaded',
        'endpoints': {
            'health': '/health',
            'predict': '/predict (POST)',
            'diseases': '/diseases',
            'test_classes': '/test-classes'
        },
        'api_version': '1.0',
        'description': 'Enhanced Tomato Disease Classification API with Image Validation'
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Check API and model status"""
    return jsonify({
        'success': True,
        'message': 'API is running',
        'model_loaded': model is not None,
        'model_error': model_error if model is None else None,
        'status': 'healthy' if model else 'model_not_loaded',
        'model_info': {
            'input_shape': str(model.input_shape) if model else None,
            'output_shape': str(model.output_shape) if model else None,
            'num_classes': len(class_names)
        },
        'tf_version': tf.__version__
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Classify disease from uploaded image with validation"""
    print("🔍 Predict endpoint called")
    print(f"📁 Files in request: {list(request.files.keys())}")
    
    if model is None:
        print("❌ Model not loaded")
        return jsonify({
            'success': False, 
            'error': 'Model not loaded',
            'model_error': model_error
        }), 500

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
        return jsonify({'success': False, 'error': 'Invalid file type'}), 400

    try:
        print("🔄 Processing image...")
        image_bytes = file.read()
        print(f"📊 Image bytes length: {len(image_bytes)}")
        
        # Open and validate image
        image = Image.open(io.BytesIO(image_bytes))
        print(f"🖼️ Original image - Mode: {image.mode}, Size: {image.size}")
        
        # STEP 1: Pre-validation - Check if image looks like a tomato leaf
        print("🔍 Validating if image is a tomato leaf...")
        is_valid_leaf, validation_reasons, leaf_confidence = validate_tomato_leaf_image(image)
        
        if not is_valid_leaf:
            print(f"❌ Image validation failed: {validation_reasons}")
            return jsonify({
                'success': False, 
                'error': 'Uploaded image does not appear to be a tomato leaf',
                'details': {
                    'reasons': validation_reasons,
                    'confidence': leaf_confidence,
                    'suggestion': 'Please upload a clear image of a tomato leaf with good contrast'
                }
            }), 400
        
        print(f"✅ Image validation passed with confidence: {leaf_confidence:.3f}")
        
        # STEP 2: Preprocess image for model
        img_array = preprocess_image(image)
        print(f"📊 Preprocessed array shape: {img_array.shape}")
        print(f"📊 Array min/max: {img_array.min():.3f}/{img_array.max():.3f}")

        # STEP 3: Make prediction
        print("🤖 Making prediction...")
        prediction = model.predict(img_array, verbose=0)
        print(f"📊 Raw prediction shape: {prediction.shape}")
        print(f"📊 Raw prediction: {prediction[0]}")
        
        # STEP 4: Post-validation - Check model confidence
        model_valid, model_reason = validate_with_model_confidence(prediction, confidence_threshold=0.25)
        
        if not model_valid:
            print(f"❌ Model validation failed: {model_reason}")
            return jsonify({
                'success': False,
                'error': 'Model cannot confidently identify the image as a tomato leaf',
                'details': {
                    'reason': model_reason,
                    'suggestion': 'Ensure the image is a clear, high-quality tomato leaf'
                }
            }), 400
        
        # STEP 5: Extract results
        predicted_index = np.argmax(prediction)
        predicted_class = class_names[predicted_index]
        confidence = float(np.max(prediction))
        confidence_percentage = round(confidence * 100, 2)

        print(f"📊 Predicted index: {predicted_index}")
        print(f"📊 Predicted class: {predicted_class}")
        print(f"📊 Confidence: {confidence_percentage}%")
        
        # Get top 3 predictions for debugging
        top_indices = np.argsort(prediction[0])[::-1][:3]
        print("📊 Top 3 predictions:")
        for i, idx in enumerate(top_indices):
            print(f"   {i+1}. {class_names[idx]}: {prediction[0][idx]*100:.2f}%")

        # Determine if plant is healthy
        is_plant_healthy = is_healthy_plant(predicted_class)
        print(f"📊 Is healthy: {is_plant_healthy}")

        # Get disease information
        disease_info = get_disease_info(predicted_class)
        
        # Convert image to base64 for response
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        
        print("✅ Prediction successful")
        return jsonify({
            'success': True,
            'data': {
                'classification': {
                    'class': predicted_class,
                    'class_name': disease_info['name'],
                    'confidence': confidence,
                    'confidence_percentage': confidence_percentage,
                    'is_healthy': is_plant_healthy,
                    'predicted_index': int(predicted_index)
                },
                'disease_info': disease_info,
                'validation_info': {
                    'leaf_confidence': leaf_confidence,
                    'passed_pre_validation': True,
                    'passed_model_validation': True
                },
                'debug_info': {
                    'top_predictions': [
                        {
                            'class': class_names[idx],
                            'confidence': float(prediction[0][idx]),
                            'percentage': round(float(prediction[0][idx]) * 100, 2)
                        }
                        for idx in top_indices
                    ],
                    'model_input_shape': str(model.input_shape),
                    'preprocessing_applied': 'normalize_0_1'
                },
                'image_base64': image_base64
            }
        })

    except Exception as e:
        print(f"❌ Prediction error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/test-classes', methods=['GET'])
def test_classes():
    """Endpoint for testing class names order"""
    return jsonify({
        'success': True,
        'data': {
            'class_names': class_names,
            'num_classes': len(class_names),
            'model_output_shape': str(model.output_shape) if model else None
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
    print("🚀 Starting Enhanced Tomato Disease Classification API...")
    print(f"📦 Model loaded: {'Yes' if model is not None else 'No'}")
    if model:
        print(f"📊 Model input shape: {model.input_shape}")
        print(f"📊 Model output classes: {len(class_names)}")
    else:
        print(f"❌ Model error: {model_error}")
    
    print("🌐 Endpoints:")
    print("- GET  /")
    print("- GET  /health")
    print("- POST /predict (with image validation)")
    print("- GET  /diseases")
    print("- GET  /test-classes")
    
    # Use PORT from environment (Railway sets this)
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('ENVIRONMENT', 'production') != 'production'
    
    print(f"🌐 Server starting on http://0.0.0.0:{port}")
    app.run(host='0.0.0.0', port=port, debug=debug_mode)
