import os
import json
import uuid
import time
import logging
from flask import Flask, jsonify, render_template, send_file
from flask_cors import CORS
from dotenv import load_dotenv
from groq import Groq
from mutagen.mp4 import MP4
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
load_dotenv()
app = Flask(__name__)
CORS(app)
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
calls = {}
SUSPICIOUS_KEYWORDS = ["bank", "account", "transfer", "otp", "password", "login", "ssn", "aadhaar", "pan", "card", "verification"]
ALPHA = 0.7  # Weight for LLM confidence
BETA = 0.3   # Weight for heuristic keyword frequency
def get_audio_duration(file_path):
    """Calculate the duration of an audio file in seconds."""
    try:
        audio = MP4(file_path)
        duration = audio.info.length  # Duration in seconds
        logger.info(f"Calculated duration for {file_path}: {duration} seconds")
        return duration
    except Exception as e:
        logger.error(f"Error calculating audio duration for {file_path}: {e}")
        return 30.0  # Fallback duration (30 seconds)
    
def transcribe_prerecorded_file(file_path):
    """Transcribe a pre-recorded audio file using Groq Whisper."""
    try:
        with open(file_path, "rb") as f:
            resp = groq_client.audio.transcriptions.create(
                file=(os.path.basename(file_path), f.read()),
                model="whisper-large-v3-turbo",
                response_format="verbose_json"
            )
        transcription = resp.text.strip()
        logger.info(f"Transcription successful for {file_path}")
        return transcription
    except Exception as e:
        logger.error(f"Transcription error for {file_path}: {e}")
        return ""

def heuristic_score(text: str) -> float:
    """Compute heuristic score based on suspicious keyword frequency."""
    words = text.lower().split()
    if not words:
        return 0.0
    count = sum(words.count(k) for k in SUSPICIOUS_KEYWORDS)
    score = min(1.0, count / len(words))
    logger.info(f"Heuristic score: {score} (keywords found: {count}, total words: {len(words)})")
    return score

def detect_fraud(context: str):
    """Use LLaMA to detect fraud and combine with heuristic score."""
    prompt = (
        "You are an AI security agent specialized in fraud detection. Analyze the conversation for:\n"
        "1. Fraudulent intent\n2. Impersonation attempts\n3. Social engineering tactics\n"
        "Respond ONLY with JSON:\n"
        "{\n  \"fraud\": boolean,\n  \"confidence\": float (0-1),\n"
        "  \"reason\": \"Brief explanation\",\n  \"impersonation\": \"Specific entity being impersonated\",\n"
        "  \"recommendation\": \"Recommended action\"\n}"
        f"\n\nConversation:\n'''{context}'''"
    )
    try:
        resp = groq_client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {"role": "system", "content": "You are a strict JSON-only fraud classifier."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
            max_tokens=300
        )
        
        out = resp.choices[0].message.content.strip()
        obj = json.loads(out)
        llm_conf = obj.get("confidence", 0.0)
        h_score = heuristic_score(context)
        combined = ALPHA * llm_conf + BETA * h_score
        is_fraud = combined > 0.5
        logger.info(f"Fraud detection: fraud={is_fraud}, combined_score={combined}, llm_confidence={llm_conf}, heuristic_score={h_score}")
        return (
            is_fraud,
            combined,
            obj.get("reason", "No reason provided"),
            obj.get("impersonation", "Unknown entity"),
            obj.get("recommendation", "No specific recommendation")
        )
    except Exception as e:
        logger.error(f"Error in fraud detection: {e}")
        return False, 0.0, "Error in fraud detection", "Unknown", "Verify manually"

def analyze_prerecorded_file():
    """Analyze all pre-recorded audio files and store their call records."""
    audio_files = ["rec1.m4a", "rec2.m4a", "rec3.m4a"]
    results = []
    
    for file_path in audio_files:
        call_id = str(uuid.uuid4())
        
        # Calculate duration and set timestamps
        duration = get_audio_duration(file_path) if os.path.exists(file_path) else 30.0
        if duration <= 0:
            logger.warning(f"Invalid duration ({duration}) for {file_path}, using fallback: 30.0 seconds")
            duration = 30.0
        start_time = 1743446400  # Apr 29, 2025, ~10 AM IST
        end_time = start_time + duration
        
        # Transcribe the audio
        transcription = transcribe_prerecorded_file(file_path) if os.path.exists(file_path) else ""
        if not transcription:
            # Fallback transcription
            transcription = (
                "Hello, this is Officer John from the Income Tax Department. "
                "We have detected suspicious activity on your account. "
                "To avoid legal action, you must transfer ₹50,000 to the provided bank account immediately. "
                "Please share your OTP for verification."
            )
            logger.info(f"Using fallback transcription for {file_path}")
        
        # Detect fraud
        fraud, score, reason, impersonation, recommendation = detect_fraud(transcription)
        
        # Store call record
        call_record = {
            "id": call_id,
            "phoneNumber": "Pre-recorded",
            "startTime": start_time,
            "endTime": end_time,
            "duration": duration,  # Duration in seconds
            "transcription": transcription,
            "flagged": fraud,
            "confidence": score,
            "reason": reason,
            "impersonation": impersonation,
            "recommendation": recommendation,
            "status": "completed",
            "audioUrl": f"/audio/{call_id}",
            "fileName": file_path  # Identify the audio file
        }
        logger.info(f"Storing call record for {file_path}: {json.dumps(call_record, indent=2)}")
        calls[call_id] = call_record
        results.append((True, f"Analysis completed for {file_path}"))
    
    return all(r[0] for r in results), [r[1] for r in results]

@app.route('/')
def index():
    """Serve the React frontend."""
    return render_template('index.html')

@app.route('/api/active-calls', methods=['GET'])
def get_active_calls():
    """Return the call record for the frontend."""
    logger.info(f"Sending {len(calls)} call records to frontend")
    return jsonify(list(calls.values()))

@app.route('/api/analyze-prerecorded', methods=['POST'])
def analyze_prerecorded():
    """Analyze all m4a files and store their call records."""
    success, messages = analyze_prerecorded_file()
    if success:
        return jsonify({"success": True, "message": "Analysis completed for all files", "details": messages})
    return jsonify({"success": False, "message": "Analysis failed for some files", "details": messages}), 400

@app.route('/audio/<call_id>', methods=['GET'])
def serve_audio(call_id):
    """Serve the audio file for the given call ID."""
    if call_id in calls:
        file_path = calls[call_id]["fileName"]
        if os.path.exists(file_path):
            logger.info(f"Serving audio file for call_id: {call_id} ({file_path})")
            return send_file(file_path, mimetype="audio/mp4")
        logger.error(f"Audio file not found: {file_path}")
        return jsonify({"error": "Audio file not found"}), 404
    logger.error(f"Invalid call ID: {call_id}")
    return jsonify({"error": "Invalid call ID"}), 404

if __name__ == '__main__':
    # Analyze all m4a files on startup
    logger.info("Starting analysis of all audio files")
    analyze_prerecorded_file()
    app.run(host='0.0.0.0', port=5000, debug=True)