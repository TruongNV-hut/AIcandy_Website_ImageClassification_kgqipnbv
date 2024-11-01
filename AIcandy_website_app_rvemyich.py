"""

@author:  AIcandy 
@website: aicandy.vn

"""

from flask import Flask, request, jsonify, render_template
from utils import load_model, process_image, get_prediction
from pyngrok import ngrok

app = Flask(__name__)

# Load the model globally
model = load_model('model.pth')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    try:
        # Get the file
        file = request.files['file']
        image_bytes = file.read()
        
        # Process the image
        image_tensor = process_image(image_bytes)
        
        # Get prediction
        predicted_class = get_prediction(model, image_tensor)
        
        # Return only the class name
        return jsonify({'class': predicted_class})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Start ngrok
    public_url = ngrok.connect(7979)
    print(f' * Public URL: {public_url}')
    
    # Run Flask app
    app.run(threaded=True, host="0.0.0.0", port=7979)