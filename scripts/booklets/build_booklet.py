"""Assemble standalone booklet figures into captioned booklet PDFs."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import yaml
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parents[2]
CAPTIONS_PATH = Path(__file__).resolve().parent / "captions.yaml"


def load_captions(path: Path = CAPTIONS_PATH) -> dict[str, str]:
    with open(path) as fh:
        data = yaml.safe_load(fh)
    return dict(data)


def assemble_booklet(fig_dir: Path, out_pdf: Path, captions: dict[str, str]) -> int:
    """Render each fig_*.pdf's PNG sibling (or rasterised PDF) as a booklet page
    with a caption band. Returns the page count."""
    pdfs = sorted(fig_dir.glob("fig_*.pdf"))
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(out_pdf) as pp:
        for pdf in pdfs:
            stem = pdf.stem
            png = pdf.with_suffix(".png")
            page = plt.figure(figsize=(8.27, 11.69))  # A4 portrait
            ax_img = page.add_axes([0.06, 0.18, 0.88, 0.76])
            ax_img.axis("off")
            if png.exists():
                ax_img.imshow(mpimg.imread(png))
            cap = captions.get(stem, "")
            page.text(0.06, 0.10, cap, ha="left", va="top", wrap=True,
                      fontsize=10, family="serif")
            page.text(0.94, 0.04, stem, ha="right", va="bottom",
                      fontsize=7, color="#7f8c8d", family="serif")
            pp.savefig(page)
            plt.close(page)
    return len(pdfs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default=str(ROOT / "outputs" / "booklets"))
    args = ap.parse_args()
    out_root = Path(args.out_root)
    caps = load_captions()
    for area, name in [("models", "models_booklet.pdf"),
                       ("methodology", "methodology_booklet.pdf")]:
        fig_dir = out_root / area
        if not fig_dir.exists():
            print(f"skip {area}: {fig_dir} missing")
            continue
        n = assemble_booklet(fig_dir, out_root / name, caps)
        print(f"assembled {area}: {n} pages -> {out_root / name}")


if __name__ == "__main__":
    main()
