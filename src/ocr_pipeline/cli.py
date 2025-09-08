import os
import json
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from .pipeline import OcrPipeline, OcrPipelineConfig

console = Console()


def _gather_inputs(path: str):
    p = Path(path)
    if p.is_dir():
        for f in sorted(p.iterdir()):
            if f.is_file():
                yield str(f)
    elif p.is_file():
        yield str(p)


@click.command()
@click.option("--input", "input_path", required=True, type=str, help="Image file or directory")
@click.option("--lang", "language", default="eng", show_default=True, help="Tesseract language hints (e.g., eng+deu)")
@click.option("--psm", default=3, show_default=True, type=int, help="Tesseract PSM mode")
@click.option("--oem", default=3, show_default=True, type=int, help="Tesseract OEM mode")
@click.option("--out", "out_dir", default="outputs", show_default=True, type=str, help="Output directory")
@click.option("--min-area", default=500, show_default=True, type=int, help="Minimum region area")
@click.option("--lexicon", multiple=True, type=str, help="Lexicon files (can pass multiple)")
@click.option("--no-binarize", is_flag=True, help="Disable binarization")
@click.option("--no-denoise", is_flag=True, help="Disable denoising")
@click.option("--no-deskew", is_flag=True, help="Disable deskew")
@click.option("--dewarp", is_flag=True, help="Enable dewarp (stub)")
def main(input_path: str, language: str, psm: int, oem: int, out_dir: str, min_area: int, lexicon: tuple,
         no_binarize: bool, no_denoise: bool, no_deskew: bool, dewarp: bool):
    os.makedirs(out_dir, exist_ok=True)
    cfg = OcrPipelineConfig(
        language_hints=language,
        psm=psm,
        oem=oem,
        min_region_area=min_area,
        binarize=not no_binarize,
        denoise=not no_denoise,
        deskew=not no_deskew,
        dewarp=dewarp,
        lexicon_paths=list(lexicon) if lexicon else None,
    )
    pipe = OcrPipeline(cfg)

    table = Table(title="OCR Results")
    table.add_column("Image")
    table.add_column("Text (first 120 chars)")

    all_results = []
    for path in _gather_inputs(input_path):
        console.log(f"Processing {path}...")
        result = pipe.process_image(path)
        all_results.append(result)
        short = (result["text"] or "").replace("\n", " ")[:120]
        table.add_row(os.path.basename(path), short)

    console.print(table)

    out_json = os.path.join(out_dir, "results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    console.log(f"Saved JSON to {out_json}")


if __name__ == "__main__":
    main()
