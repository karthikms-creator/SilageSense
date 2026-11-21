import io
import sqlite3
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# -------------------- APP CONFIG --------------------
st.set_page_config(
    page_title="SilageSense | FEEDSCAN AI",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

DB_PATH = Path("feedscan_offline.db")


# -------------------- DATABASE --------------------
def get_connection():
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def setup_database():
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS scans (
                scan_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                batch_id TEXT,
                feed_name TEXT,
                image_quality TEXT,
                blur_score REAL,
                brightness REAL,
                visual_flag TEXT,
                ph_value REAL,
                ph_status TEXT,
                notes TEXT
            )
            """
        )
        conn.commit()


def save_scan(record):
    with get_connection() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO scans
            (scan_id, timestamp, batch_id, feed_name, image_quality,
             blur_score, brightness, visual_flag, ph_value, ph_status, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record["scan_id"], record["timestamp"], record["batch_id"],
                record["feed_name"], record["image_quality"], record["blur_score"],
                record["brightness"], record["visual_flag"], record["ph_value"],
                record["ph_status"], record["notes"],
            ),
        )
        conn.commit()


def load_scans():
    with get_connection() as conn:
        return pd.read_sql_query(
            "SELECT * FROM scans ORDER BY timestamp DESC", conn
        )


setup_database()


# -------------------- IMAGE ANALYSIS --------------------
def pil_to_bgr(image):
    rgb = np.array(image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def assess_image_quality(image):
    bgr = pil_to_bgr(image)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(np.mean(gray))

    problems = []
    if blur_score < 35:
        problems.append("Image may be blurry")
    if brightness < 45:
        problems.append("Image may be too dark")
    elif brightness > 220:
        problems.append("Image may be overexposed")

    status = "Good" if not problems else "Retake recommended"
    return status, blur_score, brightness, problems


def screen_feed_image(image):
    """
    Basic colour-based visual screening.
    This is NOT a validated mold, toxin, pesticide, or feed-safety detector.
    """
    bgr = pil_to_bgr(image)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # Broad colour masks used only to identify unusual visible regions.
    # They cannot reliably distinguish mold from normal feed ingredients.
    green_low = np.array([30, 35, 25])
    green_high = np.array([95, 255, 255])
    very_dark_low = np.array([0, 0, 0])
    very_dark_high = np.array([180, 255, 55])

    green_mask = cv2.inRange(hsv, green_low, green_high)
    dark_mask = cv2.inRange(hsv, very_dark_low, very_dark_high)

    total = max(1, image.width * image.height)
    green_pct = 100.0 * cv2.countNonZero(green_mask) / total
    dark_pct = 100.0 * cv2.countNonZero(dark_mask) / total

    # Conservative wording: visual flag only, never a diagnosis.
    if dark_pct > 18:
        flag = "Dark regions noticed — inspect manually"
    elif green_pct > 35:
        flag = "Strong green colour detected — verify feed appearance"
    else:
        flag = "No strong colour flag from this basic screen"

    return flag, green_pct, dark_pct


def crop_center(image, box_fraction=0.35):
    """Create a central crop for a simple pH-strip colour sample."""
    w, h = image.size
    crop_w = max(1, int(w * box_fraction))
    crop_h = max(1, int(h * box_fraction))
    left = (w - crop_w) // 2
    top = (h - crop_h) // 2
    return image.crop((left, top, left + crop_w, top + crop_h))


def sample_strip_colour(image):
    """Return median RGB colour from the central crop."""
    crop = crop_center(image)
    arr = np.asarray(crop.convert("RGB"), dtype=np.uint8)
    median = np.median(arr.reshape(-1, 3), axis=0).astype(int)
    return tuple(int(v) for v in median), crop


# -------------------- SIDEBAR --------------------
st.sidebar.markdown("## 🌾 SilageSense")
st.sidebar.caption("FEEDSCAN AI · Preliminary feed screening")
page = st.sidebar.radio(
    "Navigate",
    ["New Scan", "Scan History", "About & Limitations"],
)
st.sidebar.divider()
st.sidebar.caption("Local database · Saves on this computer")


# -------------------- HEADER --------------------
st.markdown(
    """
    <div style="padding: 8px 0 18px 0;">
      <div style="font-size: 13px; letter-spacing: 2px; color: #6b8f71; font-weight: 700;">
        AGRICULTURE • LIVESTOCK • DIGITAL SCREENING
      </div>
      <h1 style="margin: 4px 0 0 0;">SilageSense <span style="color:#6b8f71;">/ FEEDSCAN AI</span></h1>
      <p style="color: #8a918b; margin-top: 4px;">
        A practical visual screening and batch-recording workspace for silage and feed samples.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# -------------------- NEW SCAN --------------------
if page == "New Scan":
    st.subheader("New sample assessment")
    st.caption("Enter sample details, upload images, review screening indicators, and save the record.")

    details_col, feed_col = st.columns([0.9, 1.1], gap="large")

    with details_col:
        st.markdown("### 01 · Sample details")
        batch_id = st.text_input("Batch ID", placeholder="e.g. SILAGE-2026-001")
        feed_name = st.selectbox(
            "Feed / sample type",
            ["Silage", "Green fodder", "Dry fodder", "Mixed feed", "Other"],
        )
        notes = st.text_area("Notes (optional)", placeholder="Location, storage condition, observations…", height=100)

        st.markdown("### 02 · Feed image")
        feed_file = st.file_uploader(
            "Upload a clear feed sample photo",
            type=["jpg", "jpeg", "png"],
            key="feed_image",
            help="Use even lighting, avoid filters, and include the sample clearly in the frame.",
        )

    with feed_col:
        st.markdown("### 03 · pH strip image")
        ph_file = st.file_uploader(
            "Upload a photo of the normal pH strip",
            type=["jpg", "jpeg", "png"],
            key="ph_image",
            help="Place the strip beside its colour chart if possible. Avoid shadows and glare.",
        )
        st.info(
            "The pH strip reader is not calibrated yet. This version samples the strip colour "
            "and records it; it does not invent a pH value."
        )

    feed_image = None
    feed_results = None
    if feed_file is not None:
        try:
            feed_image = Image.open(feed_file).convert("RGB")
            st.markdown("#### Feed photo preview")
            st.image(feed_image, use_container_width=True)
            quality, blur, brightness, problems = assess_image_quality(feed_image)
            visual_flag, green_pct, dark_pct = screen_feed_image(feed_image)
            feed_results = (quality, blur, brightness, problems, visual_flag, green_pct, dark_pct)

            q1, q2, q3 = st.columns(3)
            q1.metric("Image quality", quality)
            q2.metric("Sharpness score", f"{blur:.1f}")
            q3.metric("Average brightness", f"{brightness:.1f}")
            if problems:
                st.warning(" · ".join(problems) + ". Consider taking another photo.")
            st.markdown("**Visual screening note**")
            st.write(visual_flag)
            st.caption(
                f"Colour coverage indicators only: green-like pixels {green_pct:.1f}% · "
                f"dark pixels {dark_pct:.1f}%. These are not mold measurements."
            )
        except Exception as exc:
            st.error(f"Could not read the feed image: {exc}")

    strip_rgb = None
    strip_crop = None
    if ph_file is not None:
        try:
            strip_image = Image.open(ph_file).convert("RGB")
            strip_rgb, strip_crop = sample_strip_colour(strip_image)
            st.markdown("#### pH strip preview")
            preview_col1, preview_col2 = st.columns(2)
            with preview_col1:
                st.image(strip_image, caption="Uploaded strip image", use_container_width=True)
            with preview_col2:
                st.image(strip_crop, caption="Central colour sample", use_container_width=True)
            st.write(f"**Sampled median RGB:** `{strip_rgb}`")
            st.caption(
                "Lighting, camera white balance, strip brand, reaction time, and the printed "
                "colour chart affect the reading. No numeric pH is assigned in this version."
            )
        except Exception as exc:
            st.error(f"Could not read the pH strip image: {exc}")

    st.divider()
    can_save = feed_image is not None
    if st.button("💾 Save scan record", type="primary", disabled=not can_save, use_container_width=True):
        now = datetime.now()
        scan_id = "FS-" + now.strftime("%Y%m%d-%H%M%S-%f")[:19]
        if feed_results:
            quality, blur, brightness, problems, visual_flag, green_pct, dark_pct = feed_results
        else:
            quality, blur, brightness, visual_flag = "Not analysed", None, None, "No feed image"
        ph_status = f"Colour sampled RGB {strip_rgb}" if strip_rgb else "No pH strip image"
        record = {
            "scan_id": scan_id,
            "timestamp": now.isoformat(timespec="seconds"),
            "batch_id": batch_id.strip() or "Not provided",
            "feed_name": feed_name,
            "image_quality": quality,
            "blur_score": blur,
            "brightness": brightness,
            "visual_flag": visual_flag,
            "ph_value": None,
            "ph_status": ph_status,
            "notes": notes.strip(),
        }
        save_scan(record)
        st.success(f"Scan saved successfully. Record ID: {scan_id}")
        st.caption("This record is stored locally in feedscan_offline.db.")


# -------------------- HISTORY --------------------
elif page == "Scan History":
    st.subheader("Scan history")
    st.caption("Previously saved sample records from this computer.")
    history = load_scans()

    if history.empty:
        st.info("No scans saved yet. Go to New Scan to create your first record.")
    else:
        m1, m2, m3 = st.columns(3)
        m1.metric("Total records", len(history))
        m2.metric("Feed types", history["feed_name"].nunique())
        m3.metric("Latest record", history.iloc[0]["timestamp"][:10])

        search = st.text_input("Search by batch ID or feed type")
        shown = history.copy()
        if search.strip():
            mask = (
                shown["batch_id"].astype(str).str.contains(search, case=False, na=False)
                | shown["feed_name"].astype(str).str.contains(search, case=False, na=False)
            )
            shown = shown[mask]

        st.dataframe(shown, use_container_width=True, hide_index=True)
        csv_data = shown.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download records as CSV",
            data=csv_data,
            file_name="silagesense_scan_history.csv",
            mime="text/csv",
        )


# -------------------- ABOUT --------------------
else:
    st.subheader("About this prototype")
    st.markdown(
        """
        **SilageSense – FEEDSCAN AI** is a prototype for recording feed-sample observations
        and performing basic image-based screening.

        ### What this version can do
        - Check whether a feed photo may be blurry or poorly lit.
        - Calculate broad colour-coverage indicators from the uploaded image.
        - Sample the central colour of a photographed pH strip.
        - Store sample details and screening notes in a local SQLite database.
        - Display and export scan history.

        ### Important limitations
        - Colour masks do **not** identify mold species, mycotoxins, insects, urea adulteration,
          pathogens, or nutritional quality.
        - A photograph alone cannot establish whether silage is safe for livestock.
        - pH is **not calculated** until the exact strip's printed colour chart and range are
          used to calibrate the reader under controlled lighting.
        - Treat all outputs as preliminary observations. Suspected spoilage or unsafe feed
          should be checked by a qualified veterinarian, animal nutritionist, or laboratory.

        ### Local data
        Scan records are stored in `feedscan_offline.db` in the project folder. Keep a backup
        if the records are important.
        """
    )

st.sidebar.divider()
st.sidebar.caption("Prototype build · Use as a screening aid, not a safety certification.")

