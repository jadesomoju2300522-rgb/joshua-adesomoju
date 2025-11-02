import os, io, base64, sqlite3, datetime
from flask import Flask, render_template, request, jsonify, send_from_directory
import numpy as np
from PIL import Image

try:
    # Primary (robust) detector
    from fer import FER
    _FER_AVAILABLE = True
except Exception:
    _FER_AVAILABLE = False

# Optional: OpenCV for basic face detection fallback (not strictly required by fer)
try:
    import cv2
    _CV2_AVAILABLE = True
except Exception:
    _CV2_AVAILABLE = False

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(APP_DIR, "app.db")
UPLOAD_DIR = os.path.join(APP_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(__name__)

# -------------------- DB helpers --------------------
def init_db():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            mode TEXT NOT NULL,       -- 'online' (camera) or 'offline' (upload)
            image_path TEXT NOT NULL,
            emotion TEXT NOT NULL,
            score REAL,
            created_at TEXT NOT NULL
        )
    """)
    con.commit()
    con.close()

def insert_record(name, mode, image_path, emotion, score):
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute(
        "INSERT INTO records (name, mode, image_path, emotion, score, created_at) VALUES (?,?,?,?,?,?)",
        (name, mode, image_path, emotion, float(score) if score is not None else None, datetime.datetime.utcnow().isoformat())
    )
    con.commit()
    con.close()

def fetch_records(limit=50):
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("SELECT id, name, mode, image_path, emotion, score, created_at FROM records ORDER BY id DESC LIMIT ?", (limit,))
    rows = cur.fetchall()
    con.close()
    keys = ["id","name","mode","image_path","emotion","score","created_at"]
    return [dict(zip(keys, r)) for r in rows]

init_db()

# -------------------- Emotion detector --------------------
class EmotionEngine:
    def __init__(self):
        self.using = None
        self.detector = None
        if _FER_AVAILABLE:
            try:
                # Use FER; it bundles a pre-trained emotion model.
                self.detector = FER()
                self.using = "fer"
            except Exception:
                self.detector = None
                self.using = None

    def predict(self, rgb_image_np):
        """
        rgb_image_np: H x W x 3 (RGB, uint8)
        Returns (label, score). If no face / fail -> ("neutral", 0.0)
        """
        if self.using == "fer" and self.detector is not None:
            results = self.detector.detect_emotions(rgb_image_np)  # expects RGB
            if results:
                best = max(results, key=lambda r: max(r["emotions"].values()))
                emotions = best["emotions"]
                label = max(emotions, key=emotions.get)
                score = emotions[label]
                return label, float(score)
            return "neutral", 0.0
        # Fallback heuristic if fer not available
        return "neutral", 0.0

engine = EmotionEngine()

# -------------------- Utils --------------------
def decode_base64_image(data_url):
    # data_url: "data:image/png;base64,...."
    if "," in data_url:
        _, b64 = data_url.split(",", 1)
    else:
        b64 = data_url
    raw = base64.b64decode(b64)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    return img

def read_file_image(file_storage):
    img = Image.open(file_storage.stream).convert("RGB")
    return img

def save_image(pil_img, name_hint="capture"):
    ts = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    safe = "".join(c for c in name_hint if c.isalnum() or c in ("_", "-"))[:30]
    filename = f"{ts}_{safe or 'capture'}.jpg"
    path = os.path.join(UPLOAD_DIR, filename)
    pil_img.save(path, quality=92)
    return "uploads/" + filename, path

# -------------------- Routes --------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.post("/predict_camera")
def predict_camera():
    data = request.get_json(force=True)
    name = (data.get("name") or "Anonymous").strip() or "Anonymous"
    frame_b64 = data.get("image_base64")
    if not frame_b64:
        return jsonify({"ok": False, "error": "No image"}), 400
    pil = decode_base64_image(frame_b64)
    rgb = np.array(pil)
    label, score = engine.predict(rgb)
    rel_path, _ = save_image(pil, name_hint=name.replace(" ", "_"))
    insert_record(name=name, mode="online", image_path=rel_path, emotion=label, score=score)
    return jsonify({"ok": True, "emotion": label, "score": score, "image_path": rel_path})

@app.post("/predict_upload")
def predict_upload():
    name = (request.form.get("name") or "Anonymous").strip() or "Anonymous"
    f = request.files.get("image")
    if not f:
        return jsonify({"ok": False, "error": "No file"}), 400
    pil = read_file_image(f)
    rgb = np.array(pil)
    label, score = engine.predict(rgb)
    rel_path, _ = save_image(pil, name_hint=name.replace(" ", "_"))
    insert_record(name=name, mode="offline", image_path=rel_path, emotion=label, score=score)
    return jsonify({"ok": True, "emotion": label, "score": score, "image_path": rel_path})

@app.get("/records")
def get_records():
    try:
        limit = int(request.args.get("limit", "50"))
    except Exception:
        limit = 50
    return jsonify({"ok": True, "records": fetch_records(limit=limit)})

@app.get("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_DIR, filename)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)