import os
import sys
import random
import hashlib
import numpy as np
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io
import base64

print("🚀 Starting Tomato Disease Classification API...")

# Global variables
model = None
model_loaded = False

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

def download_model_async():
    """Download model in background after Flask starts"""
    global model, model_loaded
    
    print("📥 Starting model download in background...")
    
    # Model sources dengan prioritas
    model_sources = [
        {
            "name": "github_release",
            "url": "https://github.com/yusufgithub123/Machine_Learning/releases/download/v1.0.0/model.h5"
        },
        {
            "name": "direct_url",
            "url": os.getenv('MODEL_DIRECT_URL')
        },
        {
            "name": "backup",
            "url": os.getenv('BACKUP_MODEL_URL')
        }
    ]
    
    for source in model_sources:
        if not source["url"]:
            print(f"⚠️ Skipping {source['name']} - URL not configured")
            continue
            
        try:
            print(f"📥 Trying {source['name']}: {source['url']}")
            
            import requests
            response = requests.get(source["url"], timeout=60)
            
            if response.status_code == 200:
                content_type = response.headers.get('content-type', '')
                if 'application/octet-stream' in content_type or 'binary' in content_type or source["url"].endswith('.h5'):
                    os.makedirs('models', exist_ok=True)
                    model_path = 'models/tomato_model.h5'
                    
                    with open(model_path, 'wb') as f:
                        f.write(response.content)
                    
                    print(f"✅ Model downloaded successfully from {source['name']}")
                    
                    # Try to load the model
                    if load_model():
                        return True
                else:
                    print(f"⚠️ Received {content_type} instead of binary file")
                    continue
                    
        except Exception as e:
            print(f"❌ Failed to download from {source['name']}: {e}")
            continue
    
    print("⚠️ All model download attempts failed, using simulation mode")
    return False

def load_model():
    """Load the ML model"""
    global model, model_loaded
    
    model_path = 'models/tomato_model.h5'
    
    if not os.path.exists(model_path):
        print("⚠️ Model file not found")
        return False
    
    try:
        # Try TensorFlow first
        try:
            import tensorflow as tf
            model = tf.keras.models.load_model(model_path)
            model_loaded = True
            print("✅ Model loaded successfully with TensorFlow")
            return True
        except ImportError:
            print("⚠️ TensorFlow not available, trying alternative...")
            
        # Try alternative loading method
        try:
            import keras
            model = keras.models.load_model(model_path)
            model_loaded = True
            print("✅ Model loaded successfully with Keras")
            return True
        except ImportError:
            print("⚠️ Keras not available")
            
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        
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
        
        # Convert to numpy array for ML model
        img_array = np.array(image) / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        
        return img_array
        
    except Exception as e:
        raise ValueError(f"Error preprocessing image: {str(e)}")

def predict_with_ml_model(image_data):
    """Make prediction using real ML model"""
    global model, model_loaded
    
    try:
        if not model_loaded or model is None:
            raise ValueError("Model not loaded")
            
        # Preprocess image
        img_array = preprocess_image(image_data)
        
        # Make prediction
        predictions = model.predict(img_array)
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
    """Fallback simulation when ML model unavailable"""
    try:
        # Validate image first (but don't convert to numpy)
        if image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Generate deterministic but realistic predictions
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
            "simulation_mode": not model_loaded,
            "endpoints": {
                "GET /": "API information",
                "GET /health": "Health check", 
                "POST /predict": "Predict disease from image",
                "GET /diseases": "Get disease information",
                "GET /test-classes": "Get available disease classes",
                "GET /test": "Test endpoint",
                "POST /load-model": "Trigger model download"
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
            "simulation_mode": not model_loaded,
            "version": "1.0.0",
            "uptime": "Running"
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
            if model_loaded and model is not None:
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
            "message": "Prediction completed successfully"
        }), 200
        
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Prediction error: {str(e)}"}), 500

@app.route('/load-model', methods=['POST'])
def load_model_endpoint():
    """Manually trigger model download and loading"""
    try:
        print("🔄 Manual model loading triggered...")
        success = download_model_async()
        
        return jsonify({
            "success": success,
            "model_loaded": model_loaded,
            "message": "Model loaded successfully" if success else "Model loading failed, using simulation",
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
            "model_status": "loaded" if model_loaded else "simulation_mode"
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
        "available_endpoints": ["/", "/health", "/predict", "/diseases", "/test-classes", "/test", "/load-model"]
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

# Try to load model in background (non-blocking)
import threading
def background_model_load():
    """Load model in background after Flask starts"""
    import time
    time.sleep(5)  # Wait for Flask to fully start
    download_model_async()

# Start background model loading
model_thread = threading.Thread(target=background_model_load, daemon=True)
model_thread.start()

# For Railway deployment
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print(f"🚀 Starting server on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)
