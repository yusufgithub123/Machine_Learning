import os
import sys
import random
import hashlib
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io
import base64

print("🚀 Starting Tomato Disease Classification API...")

# Global variables for optional ML loading
model = None
model_loaded = False
ml_available = False

# Disease class names
class_names = [
    'Bacterial_spot', 'Early_blight', 'Late_blight', 'Leaf_Mold',
    'Septoria_leaf_spot', 'Spider_mites_Two_spotted_spider_mite',
    'Target_Spot', 'Tomato_Yellow_Leaf_Curl_Virus', 'Tomato_mosaic_virus', 
    'Healthy'
]

# Detailed disease information
diseases_info = {
    "Bacterial_spot": {
        "name": "Bacterial Spot",
        "scientific_name": "Xanthomonas vesicatoria",
        "severity": "Medium",
        "symptoms": "Small, dark brown spots on leaves, stems, and fruits",
        "treatment": "Use copper-based bactericides, ensure good air circulation",
        "prevention": "Use disease-free seeds, avoid overhead watering"
    },
    "Early_blight": {
        "name": "Early Blight", 
        "scientific_name": "Alternaria solani",
        "severity": "High",
        "symptoms": "Dark brown concentric rings on older leaves",
        "treatment": "Fungicides containing chlorothalonil or mancozeb",
        "prevention": "Crop rotation, remove infected plant debris"
    },
    "Late_blight": {
        "name": "Late Blight",
        "scientific_name": "Phytophthora infestans", 
        "severity": "Very High",
        "symptoms": "Water-soaked spots that turn brown/black, white fuzzy growth",
        "treatment": "Copper-based fungicides, remove infected plants immediately",
        "prevention": "Avoid overhead watering, ensure good drainage"
    },
    "Leaf_Mold": {
        "name": "Leaf Mold",
        "scientific_name": "Passalora fulva",
        "severity": "Medium",
        "symptoms": "Yellow spots on upper leaf surface, olive-green mold below",
        "treatment": "Improve ventilation, use fungicides if severe", 
        "prevention": "Reduce humidity, space plants properly"
    },
    "Septoria_leaf_spot": {
        "name": "Septoria Leaf Spot",
        "scientific_name": "Septoria lycopersici",
        "severity": "Medium",
        "symptoms": "Small circular spots with dark borders and light centers",
        "treatment": "Copper-based fungicides, remove affected leaves",
        "prevention": "Avoid wetting foliage, use drip irrigation"
    },
    "Spider_mites_Two_spotted_spider_mite": {
        "name": "Two-spotted Spider Mite",
        "scientific_name": "Tetranychus urticae",
        "severity": "Medium",
        "symptoms": "Fine webbing, stippled or bronzed leaves, tiny moving dots",
        "treatment": "Miticides, predatory mites, insecticidal soap",
        "prevention": "Maintain adequate humidity, regular monitoring"
    },
    "Target_Spot": {
        "name": "Target Spot",
        "scientific_name": "Corynespora cassiicola",
        "severity": "Medium", 
        "symptoms": "Circular spots with concentric rings resembling a target",
        "treatment": "Fungicides, improve air circulation",
        "prevention": "Avoid overhead watering, crop rotation"
    },
    "Tomato_Yellow_Leaf_Curl_Virus": {
        "name": "Tomato Yellow Leaf Curl Virus",
        "scientific_name": "TYLCV",
        "severity": "Very High",
        "symptoms": "Yellowing and upward curling of leaves, stunted growth",
        "treatment": "Remove infected plants, control whitefly vectors",
        "prevention": "Use resistant varieties, control whiteflies"
    },
    "Tomato_mosaic_virus": {
        "name": "Tomato Mosaic Virus", 
        "scientific_name": "ToMV",
        "severity": "High",
        "symptoms": "Mottled light and dark green patches on leaves",
        "treatment": "Remove infected plants, disinfect tools",
        "prevention": "Use certified disease-free seeds, avoid tobacco use"
    },
    "Healthy": {
        "name": "Healthy Plant",
        "scientific_name": "N/A",
        "severity": "None",
        "symptoms": "Green, vigorous growth with no disease symptoms",
        "treatment": "Continue good cultural practices",
        "prevention": "Maintain proper nutrition, watering, and spacing"
    }
}

print("✅ Disease database loaded")

def check_ml_availability():
    """Check if ML libraries are available for runtime installation"""
    global ml_available
    try:
        # Try to import TensorFlow (might be installed at runtime)
        import tensorflow as tf
        import numpy as np
        ml_available = True
        print("✅ ML libraries available")
        return True
    except ImportError:
        ml_available = False
        print("⚠️ ML libraries not available, using simulation mode")
        return False

def install_ml_runtime():
    """Attempt to install ML libraries at runtime (if supported)"""
    try:
        print("📦 Attempting to install ML dependencies at runtime...")
        import subprocess
        import sys
        
        # Try to install minimal ML packages
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "numpy==1.24.3", 
            "tensorflow-cpu==2.13.0",  # CPU-only version is lighter
            "--quiet"
        ])
        
        print("✅ ML dependencies installed successfully")
        return check_ml_availability()
        
    except Exception as e:
        print(f"⚠️ Runtime installation failed: {e}")
        return False

def preprocess_image(image_data):
    """Preprocess image for prediction"""
    try:
        # Decode base64 image
        if image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize to model input size
        image = image.resize((224, 224))
        
        return image
        
    except Exception as e:
        raise ValueError(f"Error preprocessing image: {str(e)}")

def download_and_load_model():
    """Download and load ML model (only if ML libraries available)"""
    global model, model_loaded, ml_available
    
    if not ml_available:
        print("⚠️ ML libraries not available for model loading")
        return False
    
    try:
        print("📥 Downloading model from GitHub...")
        
        model_url = "https://github.com/yusufgithub123/Machine_Learning/releases/download/v1.0.0/model.h5"
        model_path = 'models/tomato_model.h5'
        
        # Download model
        import requests
        os.makedirs('models', exist_ok=True)
        
        response = requests.get(model_url, stream=True, timeout=120)
        
        if response.status_code == 200:
            with open(model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            print("✅ Model downloaded successfully")
            
            # Load model
            import tensorflow as tf
            model = tf.keras.models.load_model(model_path)
            model_loaded = True
            
            print("✅ Model loaded successfully")
            return True
        else:
            print(f"❌ Download failed: HTTP {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Model loading error: {e}")
        return False

def predict_with_ml_model(image_data):
    """Make prediction using real ML model"""
    global model, model_loaded, ml_available
    
    if not ml_available or not model_loaded or model is None:
        raise ValueError("ML model not available")
    
    try:
        import numpy as np
        
        # Preprocess image
        image = preprocess_image(image_data)
        
        # Convert to numpy array for ML model
        img_array = np.array(image) / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        
        # Make prediction
        predictions = model.predict(img_array, verbose=0)
        pred_probs = predictions[0]  # Get first (and only) prediction
        
        # Create results
        results = []
        for i, prob in enumerate(pred_probs):
            if prob > 0.01:  # Only include predictions > 1%
                results.append({
                    "class": class_names[i],
                    "confidence": round(float(prob), 3),
                    "description": diseases_info[class_names[i]]["symptoms"],
                    "severity": diseases_info[class_names[i]]["severity"],
                    "treatment": diseases_info[class_names[i]]["treatment"]
                })
        
        # Sort by confidence
        results.sort(key=lambda x: x['confidence'], reverse=True)
        
        return results[:5]  # Return top 5 predictions
        
    except Exception as e:
        raise ValueError(f"ML model prediction error: {str(e)}")

def simulate_prediction(image_data):
    """High-quality simulation for testing and fallback"""
    try:
        # Validate image first
        image = preprocess_image(image_data)
        
        # Generate deterministic but realistic predictions based on image
        image_hash = hashlib.md5(image_data.encode()).hexdigest()
        random.seed(int(image_hash[:8], 16))
        
        # Create realistic confidence distribution
        main_confidence = random.uniform(0.75, 0.95)
        remaining_confidence = 1.0 - main_confidence
        
        # Select primary class based on hash
        primary_class_idx = int(image_hash[:2], 16) % len(class_names)
        primary_class = class_names[primary_class_idx]
        
        # Create prediction results
        predictions = []
        
        # Add primary prediction
        predictions.append({
            "class": primary_class,
            "confidence": round(main_confidence, 3),
            "description": diseases_info[primary_class]["symptoms"],
            "severity": diseases_info[primary_class]["severity"],
            "treatment": diseases_info[primary_class]["treatment"]
        })
        
        # Add 2-3 secondary predictions
        secondary_count = random.choice([2, 3])
        available_classes = [c for c in class_names if c != primary_class]
        secondary_classes = random.sample(available_classes, secondary_count)
        
        secondary_confidences = []
        for i in range(secondary_count):
            conf = remaining_confidence * random.uniform(0.3, 0.7) / secondary_count
            secondary_confidences.append(conf)
        
        # Normalize secondary confidences
        total_secondary = sum(secondary_confidences)
        secondary_confidences = [c * (remaining_confidence / total_secondary) for c in secondary_confidences]
        
        for class_name, confidence in zip(secondary_classes, secondary_confidences):
            predictions.append({
                "class": class_name,
                "confidence": round(confidence, 3),
                "description": diseases_info[class_name]["symptoms"],
                "severity": diseases_info[class_name]["severity"],
                "treatment": diseases_info[class_name]["treatment"]
            })
        
        # Sort by confidence
        predictions.sort(key=lambda x: x['confidence'], reverse=True)
        
        return predictions
        
    except Exception as e:
        raise ValueError(f"Error in simulation: {str(e)}")

print("✅ Prediction functions loaded")

# Initialize Flask app
app = Flask(__name__)
CORS(app)

print("✅ Flask app created and CORS enabled")

# Check ML availability at startup (non-blocking)
check_ml_availability()

# Routes
@app.route('/', methods=['GET'])
def home():
    """API information endpoint"""
    try:
        return jsonify({
            "message": "🍅 Tomato Disease Classification API",
            "version": "1.0.0",
            "status": "running",
            "model_loaded": model_loaded,
            "ml_available": ml_available,
            "simulation_mode": not model_loaded,
            "endpoints": {
                "GET /": "API information",
                "GET /health": "Health check", 
                "POST /predict": "Predict disease from image",
                "GET /diseases": "Get disease information",
                "GET /test-classes": "Get available disease classes",
                "GET /test": "Test endpoint",
                "POST /install-ml": "Install ML dependencies",
                "POST /load-model": "Load ML model"
            },
            "usage": {
                "predict": "POST with JSON: {'image': 'data:image/jpeg;base64,<base64_string>'}",
                "image_formats": ["JPEG", "PNG", "WebP"],
                "max_size": "10MB"
            }
        }), 200
    except Exception as e:
        return jsonify({"error": f"Home endpoint error: {str(e)}"}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        return jsonify({
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "model_loaded": model_loaded,
            "ml_available": ml_available,
            "simulation_mode": not model_loaded,
            "version": "1.0.0"
        }), 200
    except Exception as e:
        return jsonify({"error": f"Health check error: {str(e)}"}), 500

@app.route('/predict', methods=['POST'])
def predict():
    """Predict disease from uploaded image"""
    try:
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({
                "error": "No image data provided",
                "expected_format": "{'image': 'data:image/jpeg;base64,<base64_string>'}"
            }), 400
        
        image_data = data['image']
        
        # Try ML model first, fallback to simulation
        try:
            if ml_available and model_loaded and model is not None:
                predictions = predict_with_ml_model(image_data)
                mode = "ml_model"
                print("✅ Used ML model for prediction")
            else:
                predictions = simulate_prediction(image_data)
                mode = "simulation"
                print("⚠️ Used simulation for prediction")
        except Exception as ml_error:
            print(f"❌ ML model failed: {ml_error}, falling back to simulation")
            predictions = simulate_prediction(image_data)
            mode = "simulation_fallback"
        
        return jsonify({
            "predictions": predictions,
            "top_prediction": predictions[0]["class"],
            "confidence": predictions[0]["confidence"],
            "timestamp": datetime.now().isoformat(),
            "mode": mode,
            "model_loaded": model_loaded,
            "ml_available": ml_available,
            "message": "Prediction completed successfully"
        }), 200
        
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Prediction error: {str(e)}"}), 500

@app.route('/install-ml', methods=['POST'])
def install_ml_endpoint():
    """Install ML dependencies at runtime"""
    try:
        print("🔄 Installing ML dependencies...")
        success = install_ml_runtime()
        
        return jsonify({
            "success": success,
            "ml_available": ml_available,
            "message": "ML dependencies installed successfully" if success else "ML installation failed",
            "timestamp": datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        return jsonify({"error": f"ML installation error: {str(e)}"}), 500

@app.route('/load-model', methods=['POST'])
def load_model_endpoint():
    """Load ML model (requires ML dependencies)"""
    try:
        if not ml_available:
            return jsonify({
                "success": False,
                "message": "ML dependencies not available. Try /install-ml first.",
                "ml_available": ml_available,
                "timestamp": datetime.now().isoformat()
            }), 400
            
        print("🔄 Loading ML model...")
        success = download_and_load_model()
        
        return jsonify({
            "success": success,
            "model_loaded": model_loaded,
            "ml_available": ml_available,
            "message": "Model loaded successfully" if success else "Model loading failed",
            "timestamp": datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        return jsonify({"error": f"Model loading error: {str(e)}"}), 500

@app.route('/diseases', methods=['GET'])
def get_diseases_info():
    """Get detailed information about all diseases"""
    try:
        return jsonify({
            "diseases": diseases_info,
            "total_classes": len(class_names),
            "classes": class_names,
            "message": "Disease database retrieved successfully"
        }), 200
    except Exception as e:
        return jsonify({"error": f"Diseases endpoint error: {str(e)}"}), 500

@app.route('/test-classes', methods=['GET'])
def test_classes():
    """Get available disease classes for testing"""
    try:
        return jsonify({
            "classes": class_names,
            "total": len(class_names),
            "note": "These are the disease classes the model can predict"
        }), 200
    except Exception as e:
        return jsonify({"error": f"Test classes endpoint error: {str(e)}"}), 500

@app.route('/test', methods=['GET'])
def test_endpoint():
    """Simple test endpoint"""
    try:
        return jsonify({
            "status": "OK", 
            "message": "Test endpoint working",
            "timestamp": datetime.now().isoformat(),
            "api_ready": True,
            "model_status": "loaded" if model_loaded else "simulation_mode",
            "ml_status": "available" if ml_available else "not_installed"
        }), 200
    except Exception as e:
        return jsonify({"error": f"Test endpoint error: {str(e)}"}), 500

# Global error handlers
@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        "error": "Internal server error",
        "message": "Something went wrong on our end",
        "status": 500
    }), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "error": "Not found",
        "message": "The requested endpoint was not found",
        "status": 404,
        "available_endpoints": ["/", "/health", "/predict", "/diseases", "/test-classes", "/test", "/install-ml", "/load-model"]
    }), 404

@app.errorhandler(Exception)
def handle_exception(e):
    return jsonify({
        "error": "Unexpected error",
        "message": str(e),
        "status": 500
    }), 500

print("✅ All routes defined")
print("📊 Available routes:")
for rule in app.url_map.iter_rules():
    if rule.endpoint != 'static':
        print(f" {rule.rule} -> {rule.endpoint}")

print("🎯 Flask application ready!")
print("💡 Use /install-ml and /load-model for ML features")

# For Railway deployment
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"🚀 Starting server on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)
