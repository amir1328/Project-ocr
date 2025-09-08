import io
import os
import json
from tempfile import NamedTemporaryFile
from PIL import Image
import streamlit as st

# Ensure 'src' is on sys.path when running via `streamlit run`
import sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from ocr_pipeline.pipeline import OcrPipeline, OcrPipelineConfig

st.set_page_config(page_title="Enhanced OCR for Historic Records", layout="wide")

st.title("Enhanced OCR for Regional and Historic Records")

with st.sidebar:
    st.header("Settings")
    language = st.text_input("Tesseract languages", value="eng")
    psm = st.number_input("PSM", value=3, min_value=0, max_value=13, step=1)
    oem = st.number_input("OEM", value=3, min_value=0, max_value=3, step=1)
    binarize = st.checkbox("Binarize", value=True)
    denoise = st.checkbox("Denoise", value=True)
    deskew = st.checkbox("Deskew", value=True)
    dewarp = st.checkbox("Dewarp (stub)", value=False)

uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "tif", "tiff"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Original", width='stretch')

    cfg = OcrPipelineConfig(
        language_hints=language,
        psm=int(psm),
        oem=int(oem),
        binarize=binarize,
        denoise=denoise,
        deskew=deskew,
        dewarp=dewarp,
    )
    pipe = OcrPipeline(cfg)

    # Save to a temporary file and pass path to pipeline
    with NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        image.save(tmp.name)
        tmp_path = tmp.name

    with st.spinner("Running OCR..."):
        result = pipe.process_image(tmp_path)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Detected Text")
        edited_text = st.text_area("Text", value=result.get("text", ""), height=300)
        if st.download_button("Download results.json", data=json.dumps(result, ensure_ascii=False, indent=2), file_name="results.json"):
            pass
    with col2:
        st.subheader("Regions")
        st.write(f"Detected regions: {len(result.get('regions', []))}")
        st.json(result)

    # Clean up temp file
    try:
        os.remove(tmp_path)
    except Exception:
        pass
