# app.py
import os
from dotenv import load_dotenv
import google.generativeai as genai
from flask import Flask, render_template, request, jsonify, send_from_directory
from text_speech import generate_audio_from_text  # Import the function from text_to_speech.py

# Load environment variables from .env file
load_dotenv()

# Replace with your actual Gemini API key
GOOGLE_API_KEY = os.getenv('AI_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)

# Choose the Gemini model you want to use
model = genai.GenerativeModel('gemini-1.5-flash')

# Initialize a chat session
chat = model.start_chat(history=[])

# Create Flask app
app = Flask(__name__)

# Folder to store audio files
app.config['AUDIO_FOLDER'] = 'audio'

# Ensure the audio folder exists
os.makedirs(app.config['AUDIO_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/send_message', methods=['POST'])
def send_message():
    user_message = request.json.get('message')
    
    # Send the message to the model and get the response
    response = chat.send_message(user_message)
    bot_response = response.text

    # Generate speech from the bot's response text
    audio_filename = generate_audio_from_text(bot_response)

    # Return the response text and audio file name (URL)
    audio_url = f"/audio/{audio_filename}"
    return jsonify({'response': bot_response, 'audio_url': audio_url})

@app.route('/audio/<filename>')
def serve_audio(filename):
    """
    Serve the generated audio files
    """
    return send_from_directory(app.config['AUDIO_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
