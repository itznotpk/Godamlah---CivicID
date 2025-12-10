import os
import cv2
import numpy as np
import psycopg2
import base64
import time
from flask import Flask, render_template, request, jsonify, redirect, url_for
from deepface import DeepFace

app = Flask(__name__)

# --- CONFIGURATION ---
# Replace with your actual URI
DB_URI = "postgres://avnadmin:AVNS_wSQi-mmxoN8CP5T-0d5@pg-1de2613d-godamlah-facial-recognition.h.aivencloud.com:16358/defaultdb?sslmode=require"
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
MODEL_NAME = "Facenet512"
PASSING_THRESHOLD_DISTANCE = 20.0 

# --- AI WARMUP ---
print("⏳ Warming up DeepFace AI... (This runs once)")
try:
    DeepFace.represent(img_path=np.zeros((100,100,3), np.uint8), model_name=MODEL_NAME, enforce_detection=False)
    print("✅ AI Ready!")
except:
    pass

def get_db_connection():
    return psycopg2.connect(DB_URI)

def generate_embedding(img_input):
    embedding_obj = DeepFace.represent(
        img_path = img_input, 
        model_name = MODEL_NAME, 
        enforce_detection = False
    )
    return embedding_obj[0]["embedding"]

# --- ROUTES ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_ic', methods=['POST'])
def upload_ic():
    if 'ic_image' not in request.files:
        return jsonify({"status": "error", "message": "No file part"}), 400
    file = request.files['ic_image']
    if file.filename == '':
        return jsonify({"status": "error", "message": "No selected file"}), 400

    filepath = os.path.join(UPLOAD_FOLDER, "user_ic.jpg")
    file.save(filepath)

    try:
        embedding = generate_embedding(filepath)
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Reset DB for single-user session
        cur.execute("DELETE FROM pictures;") 
        cur.execute("INSERT INTO pictures (picture, embedding) VALUES (%s, %s)", 
                    ("user_ic.jpg", embedding))
        conn.commit()
        conn.close()
        print("✅ New IC Registered!")
        
        # --- FIX IS HERE: Return JSON, NOT redirect ---
        return jsonify({"status": "success", "redirect": url_for('verify_page')})
        
    except Exception as e:
        print(e)
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/verify_page')
def verify_page():
    return render_template('verify.html')

@app.route('/success')
def success_page():
    return render_template('success.html')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    try:
        data = request.json['image']
        image_data = data.split(',')[1]
        decoded_image = base64.b64decode(image_data)
        np_arr = np.frombuffer(decoded_image, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = haar_cascade.detectMultiScale(gray_img, 1.05, minNeighbors=2, minSize=(100,100))

        if len(faces) == 0:
            return jsonify({"status": "error", "message": "No face detected"})

        largest_face = max(faces, key=lambda f: f[2] * f[3])
        x, y, w, h = largest_face
        face_crop = frame[y:y+h, x:x+w]
        rgb_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        
        embedding = generate_embedding(rgb_face)

        conn = get_db_connection()
        cur = conn.cursor()
        string_rep = "["+ ",".join(str(x) for x in embedding) +"]"
        
        cur.execute("""
            SELECT picture, (embedding <-> %s) as distance 
            FROM pictures 
            ORDER BY embedding <-> %s ASC 
            LIMIT 1;
        """, (string_rep, string_rep))
        row = cur.fetchone()
        conn.close()

        if row:
            distance = row[1]
            max_score_dist = PASSING_THRESHOLD_DISTANCE * 2
            raw_score = ((max_score_dist - distance) / max_score_dist) * 100
            score = round(max(0, min(100, raw_score)))

            print(f"DEBUG: Distance: {distance:.2f} | Score: {score}%")

            if distance < PASSING_THRESHOLD_DISTANCE:
                return jsonify({
                    "status": "success", 
                    "score": score, 
                    "message": "Identity Verified",
                    "redirect": url_for('success_page')
                })
            else:
                return jsonify({"status": "fail", "score": score, "message": "Face mismatch"})
        else:
            return jsonify({"status": "error", "message": "No ID record found"})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"status": "error", "message": str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)