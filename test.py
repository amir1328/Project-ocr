from PIL import Image, ImageDraw
from src.ocr_pipeline.pipeline import OcrPipeline, OcrPipelineConfig

# create sample image
img = Image.new("RGB", (800, 200), "white")
d = ImageDraw.Draw(img)
d.text((50, 80), "Hello 1895 from OCR!", fill="black")
img.save("inputs/hello.png")

# run OCR
pipe = OcrPipeline(OcrPipelineConfig(language_hints="eng", psm=6))
res = pipe.process_image("inputs/hello.png")
print(res["text"])