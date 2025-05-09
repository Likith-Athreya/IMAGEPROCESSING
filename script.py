import streamlit as st
import cv2
import numpy as np
from PIL import Image
from io import BytesIO

def enhance_edges(image):
    """Enhance edges using Canny edge detection and sharpening"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # Sharpen the image
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(image, -1, kernel)

    # Combine with edges
    return cv2.addWeighted(sharpened, 0.7, edges_colored, 0.3, 0)

def detect_flag_bounds(flag):
    """Detect the bounding box of the white flag area"""
    gray = cv2.cvtColor(flag, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)  # detect white

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("Couldn't detect white flag region.")

    # Assume largest white area is the flag
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    return x, y, w, h

def fit_pattern_to_flag(pattern, flag, zoom_out_factor=0.9):
    try:
        # Convert RGBA to RGB if needed
        if pattern.shape[2] == 4:
            pattern = cv2.cvtColor(pattern, cv2.COLOR_RGBA2RGB)
        if flag.shape[2] == 4:
            flag = cv2.cvtColor(flag, cv2.COLOR_RGBA2RGB)

        # Detect white flag region
        x, y, w, h = detect_flag_bounds(flag)

        # Resize pattern to fit inside white region
        pat_h, pat_w = pattern.shape[:2]
        scale_w = w / pat_w
        scale_h = h / pat_h
        scale = min(scale_w, scale_h) * zoom_out_factor

        new_w, new_h = int(pat_w * scale), int(pat_h * scale)
        pattern_resized = cv2.resize(pattern, (new_w, new_h), interpolation=cv2.INTER_AREA)
        pattern_resized = enhance_edges(pattern_resized)

        # Create blank canvas
        pattern_canvas = np.full_like(flag, 255)
        x_offset = x + (w - new_w) // 2
        y_offset = y + (h - new_h) // 2
        pattern_canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = pattern_resized

        # Blend with flag using fold mask
        gray_flag = cv2.cvtColor(flag, cv2.COLOR_BGR2GRAY)
        fold_mask = 1 - cv2.GaussianBlur(cv2.equalizeHist(gray_flag).astype(np.float32) / 255.0, (41, 41), 0)
        fold_mask_3c = np.repeat(fold_mask[:, :, np.newaxis], 3, axis=2)

        blended = flag.astype(np.float32) * (1 - fold_mask_3c) + pattern_canvas.astype(np.float32) * fold_mask_3c
        output = np.clip(blended, 0, 255).astype(np.uint8)

        return output

    except Exception as e:
        print(f"❌ Error: {e}")
        return None

# ---------- Streamlit UI ----------
st.title("Pattern-to-Flag Fitting Tool")

uploaded_pattern = st.file_uploader("Upload Pattern Image", type=["jpg", "jpeg", "png"])
uploaded_flag = st.file_uploader("Upload Flag Image (white flag)", type=["jpg", "jpeg", "png"])

zoom_out_factor = st.slider("Zoom Out Pattern", 0.5, 1.0, 0.9)

if uploaded_pattern and uploaded_flag:
    pattern = Image.open(uploaded_pattern)
    flag = Image.open(uploaded_flag)

    pattern_np = np.array(pattern)
    flag_np = np.array(flag)

    st.image(pattern_np, caption="Pattern Image", use_column_width=True)
    st.image(flag_np, caption="White Flag Image", use_column_width=True)

    output = fit_pattern_to_flag(pattern_np, flag_np, zoom_out_factor)

    if output is not None:
        st.image(output, caption="Pattern Fitted to Flag", use_column_width=True)

        output_pil = Image.fromarray(output)
        buf = BytesIO()
        output_pil.save(buf, format="PNG")
        st.download_button(
            label="Download Result",
            data=buf.getvalue(),
            file_name="pattern_on_flag.png",
            mime="image/png"
        )
    else:
        st.error("Failed to fit pattern to flag.")
