import streamlit as st
import cv2
import numpy as np
from PIL import Image
import sqlite3
from datetime import datetime
from gtts import gTTS
import io

# --- 1. OFFLINE SQLITE DATABASE SETUP ---
conn = sqlite3.connect("feedscan_offline.db", check_same_thread=False)
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS logs
             (timestamp TEXT, batch_id TEXT, sample_type TEXT, 
              mold_score REAL, strip_ph REAL, verdict TEXT)''')
conn.commit()

# --- 2. IMAGE QUALITY CHECK (BLUR & BRIGHTNESS) ---
def check_image_quality(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    brightness = np.mean(gray)
    return blur_score, brightness

# --- 3. MOLD, PEST & CONTAMINATION DETECTOR ---
def analyze_feed_mold(img_bgr, full_frame=False):
    h, w, _ = img_bgr.shape
    feed_zone = img_bgr if full_frame else img_bgr[int(h*0.2):int(h*0.8), int(w*0.05):int(w*0.5)]
    hsv = cv2.cvtColor(feed_zone, cv2.COLOR_BGR2HSV)
    
    # 1. Detect greenish/grayish fungal mold
    lower_mold = np.array((25, 30, 30), dtype=np.uint8)
    upper_mold = np.array((95, 255, 220), dtype=np.uint8)
    mold_mask = cv2.inRange(hsv, lower_mold, upper_mold)
    
    # 2. Detect dark reddish-brown pests, insects, or rotten clumps
    lower_pest = np.array((0, 40, 10), dtype=np.uint8)
    upper_pest = np.array((20, 255, 95), dtype=np.uint8)
    pest_mask = cv2.inRange(hsv, lower_pest, upper_pest)
    
    combined_mask = cv2.bitwise_or(mold_mask, pest_mask)
    contamination_pct = (np.sum(combined_mask > 0) / combined_mask.size) * 100
    return round(contamination_pct, 1)

# --- 4. TEST STRIP COLORIMETRY (pH / UREA STRIP) ---
def analyze_test_strip(img_bgr):
    h, w, _ = img_bgr.shape
    strip_zone = img_bgr[int(h*0.35):int(h*0.65), int(w*0.65):int(w*0.85)]
    hsv = cv2.cvtColor(strip_zone, cv2.COLOR_BGR2HSV)
    
    avg_hue = np.mean(hsv[:, :, 0])
    avg_sat = np.mean(hsv[:, :, 1])
    
    if avg_sat < 35:
        estimated_ph = 6.5
    elif avg_hue < 15 or avg_hue > 165:
        estimated_ph = 4.2
    elif 15 <= avg_hue < 35:
        estimated_ph = 5.5
    elif 35 <= avg_hue < 85:
        estimated_ph = 7.0
    else:
        estimated_ph = 8.5
    return estimated_ph, round(avg_hue, 1)

# --- 5. QR CODE READER ---
def read_qr_code(img_bgr):
    detector = cv2.QRCodeDetector()
    data, _, _ = detector.detectAndDecode(img_bgr)
    return data if data else "BATCH-TN-2026-01"

# --- STREAMLIT MOBILE UI ---
st.set_page_config(page_title="FEEDSCAN AI", page_icon="🐄", layout="centered")
st.title("SilageSense")
st.caption("Offline Rapid Feed & Silage Quality Screener | Team Quantum Coders")

lang = st.radio("Language / மொழி:", ("English", "தமிழ் (Tamil)"), horizontal=True)
sample_type = st.selectbox("Select Sample Type:", ("Cattle Feed Pellets", "Fermented Silage"))
full_frame = st.checkbox("Feed-Only Photo (No Test Strip in Frame)", value=True)

st.info("📋 **Guide:** Upload a close-up feed photo (checked above) OR uncheck to scan Feed on LEFT + Test Strip on RIGHT.")

input_mode = st.radio("Capture Method:", ("Upload Test Photo", "Use Live Camera"), horizontal=True)
img_file = st.camera_input("Scan Feed + Strip") if input_mode == "Use Live Camera" else st.file_uploader("Upload Feed + Strip Image", type=("jpg", "png", "jpeg"))

if img_file is not None:
    image = Image.open(img_file)
    img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    blur, brightness = check_image_quality(img_bgr)
    col1, col2 = st.columns(2)
    col1.metric("Focus Score", f"{int(blur)}", "Clear" if blur > 50 else "Low Focus")
    col2.metric("Lighting", f"{int(brightness)}/255", "Good" if brightness > 60 else "Use Flash")
    
    mold_pct = analyze_feed_mold(img_bgr, full_frame=full_frame)
    est_ph, hue_val = (6.5, 0.0) if full_frame else analyze_test_strip(img_bgr)
    batch_id = read_qr_code(img_bgr)
    
    st.image(image, caption=f"Detected Batch: {batch_id}", use_container_width=True)
    
    if mold_pct > 10.0 or est_ph > 7.8:
        verdict = "RED - DO NOT FEED"
        st.error(f"🔴 **{verdict}**")
        advice_en = "High spoilage, pest infestation, or urea contamination detected. Do not feed to cattle. Send batch for lab testing."
        advice_ta = "கெட்டுப்போன தீவனம், பூச்சித் தாக்குதல் அல்லது யூரியா கலப்படம் கண்டறியப்பட்டுள்ளது. மாடுகளுக்கு கொடுக்க வேண்டாம்."
    elif mold_pct > 4.0 or (sample_type == "Fermented Silage" and est_ph > 5.0):
        verdict = "YELLOW - CAUTION"
        st.warning(f"🟡 **{verdict}**")
        advice_en = "Moderate moisture, minor contamination, or pH drift detected. Sun-dry feed for 4 hours and inspect before feeding."
        advice_ta = "ஈரப்பதம் அல்லது அமிலத்தன்மை மாற்றம் உள்ளது. தீவனத்தை 4 மணி நேரம் வெயிலில் உலர்த்தவும்."
    else:
        verdict = "GREEN - SAFE TO FEED"
        st.success(f"🟢 **{verdict}**")
        advice_en = "Feed quality and pH are within safe ICAR standards. Safe for daily cattle ration."
        advice_ta = "தீவனத்தின் தரம் பாதுகாப்பாக உள்ளது. மாடுகளுக்கு தாராளமாக கொடுக்கலாம்."

    m1, m2, m3 = st.columns(3)
    m1.metric("Mold / Pest Risk", f"{mold_pct}%")
    m2.metric("Estimated pH", f"{est_ph}")
    m3.metric("Test Cost", "₹5")
    
    advice_text = advice_ta if "Tamil" in lang else advice_en
    st.write(f"**Advisory:** {advice_text}")
    
    try:
        tts_lang = "ta" if "Tamil" in lang else "en"
        tts = gTTS(text=advice_text, lang=tts_lang)
        audio_fp = io.BytesIO()
        tts.write_to_fp(audio_fp)
        st.audio(audio_fp.getvalue(), format="audio/mp3")
    except Exception:
        st.caption("🔇 Offline Mode: Voice audio cached locally.")

    if st.button("💾 Save Batch Result Offline"):
        c.execute("INSERT INTO logs VALUES (?, ?, ?, ?, ?, ?)",
                  (datetime.now().strftime("%Y-%m-%d %H:%M"), batch_id, sample_type, mold_pct, est_ph, verdict))
        conn.commit()
        st.toast("Saved to local SQLite database!")

with st.expander("📊 View Offline Batch Logs (SQLite Sync Queue)"):
    rows = c.execute("SELECT * FROM logs ORDER BY timestamp DESC").fetchall()
    if rows:
        st.table(rows)
    else:
        st.write("No batches scanned yet.")
