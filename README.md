## Enhanced OCR Pipeline for Regional and Historic Records

An end-to-end Python pipeline for OCR on challenging regional and historical documents. Includes preprocessing (deskew, denoise, binarize, dewarp), layout analysis, OCR with language hints, postprocessing (lexicons, spelling, diacritics/numerals normalization), and evaluation (CER/WER).

### Quickstart

1) Create a virtual environment and install deps:
```bash
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2) Install Tesseract OCR engine:
- Windows: Download installer from `https://github.com/tesseract-ocr/tesseract` (add install path to PATH, e.g. `C:\\Program Files\\Tesseract-OCR`)
- Add language packs as needed (e.g., `eng`, `deu`, `fra`, `ara`, etc.)

3) Run the CLI on an image or folder:
```bash
python -m src.ocr_pipeline.cli --input path/to/image_or_dir --lang eng --out outputs
```

4) Launch the Streamlit UI:
```bash
streamlit run src/ocr_pipeline/app.py
```

### Project Structure
```
src/ocr_pipeline/
  preprocessing.py  # deskew, binarize, denoise, dewarp
  layout.py         # text block detection and region crops
  ocr.py            # pytesseract integration per region
  postprocess.py    # lexicons, spellcheck, normalization
  evaluation.py     # CER/WER and confusion report
  pipeline.py       # orchestrates full pipeline
  cli.py            # command-line interface
  app.py            # streamlit app
```

### Notes
- Lexicon samples live under `data/lexicons/`. Provide your own domain dictionaries for best results.
- For right-to-left scripts, bidi and Arabic shaping are supported.
