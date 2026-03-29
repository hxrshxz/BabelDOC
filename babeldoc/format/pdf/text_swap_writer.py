"""PyMuPDF-based text swap writer.

Bypasses BabelDOC's typesetting + content-stream backend entirely.
Opens the *original* PDF with PyMuPDF, redacts original text inside each
translated paragraph box, and inserts the translated text using
``insert_htmlbox`` (which handles CJK, mixed scripts, and auto-scaling).
"""

import logging
import re
from pathlib import Path

import pymupdf

from babeldoc.format.pdf.document_il import il_version_1
from babeldoc.format.pdf.translation_config import TranslateResult
from babeldoc.format.pdf.translation_config import TranslationConfig
from babeldoc.format.pdf.translation_config import WatermarkOutputMode

logger = logging.getLogger(__name__)


def _extract_rgb_color(pci: str) -> tuple[float, float, float] | None:
    """Try to extract an RGB fill colour from a passthrough instruction string.

    Looks for patterns like ``R G B rg`` (non-stroking colour).  Returns
    ``None`` when no match is found.
    """
    # Match the last "R G B rg" pattern (non-stroking fill colour)
    m = re.search(
        r"([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+rg",
        pci,
        re.IGNORECASE,
    )
    if m:
        try:
            return (float(m.group(1)), float(m.group(2)), float(m.group(3)))
        except (ValueError, IndexError):
            pass
    return None


def _rgb_to_css(rgb: tuple[float, float, float]) -> str:
    """Convert an ``(r, g, b)`` triple (0-1 range) to a CSS colour string."""
    r = max(0, min(255, int(rgb[0] * 255)))
    g = max(0, min(255, int(rgb[1] * 255)))
    b = max(0, min(255, int(rgb[2] * 255)))
    return f"rgb({r},{g},{b})"


def _il_box_to_rect(
    box: il_version_1.Box,
    page_height: float,
    cropbox_origin_x: float = 0.0,
    cropbox_origin_y: float = 0.0,
) -> pymupdf.Rect:
    """Convert a BabelDOC IL ``Box`` to a PyMuPDF ``Rect``.

    BabelDOC uses PDF coordinates (origin bottom-left, y up).
    PyMuPDF uses screen coordinates (origin top-left, y down).

    If the IL boxes are in *MediaBox* space but the page has a non-zero
    CropBox origin, subtract that origin so we land in CropBox space
    (which is what PyMuPDF's page methods operate on).
    """
    x0 = box.x - cropbox_origin_x
    x1 = box.x2 - cropbox_origin_x
    y0 = page_height - (box.y2 - cropbox_origin_y)  # PDF y2 → screen top
    y1 = page_height - (box.y - cropbox_origin_y)  # PDF y  → screen bottom
    return pymupdf.Rect(x0, y0, x1, y1)


def text_swap_write(
    docs: il_version_1.Document,
    original_pdf_path: str | Path,
    translation_config: TranslationConfig,
) -> TranslateResult:
    """Write translated text into the original PDF using PyMuPDF overlay.

    Instead of rebuilding the PDF via BabelDOC's content-stream backend,
    this function:

    1. Opens the original (cleaned) PDF with PyMuPDF.
    2. For every translated paragraph, adds a redaction annotation over
       the original text area — then applies redactions while preserving
       vector art and raster images.
    3. Inserts the translated text with ``insert_htmlbox``, which handles
       CJK / mixed-script rendering and automatic font-size scaling.
    4. Saves the result as the mono PDF (and optionally a dual PDF).

    Returns a :class:`TranslateResult` that mirrors the shape produced by
    the normal ``PDFCreater.write()`` path.
    """
    original_pdf_path = str(original_pdf_path)
    pdf = pymupdf.open(original_pdf_path)

    basename = Path(translation_config.input_file).stem
    debug_suffix = ".debug" if translation_config.debug else ""
    if translation_config.watermark_output_mode != WatermarkOutputMode.Watermarked:
        debug_suffix += ".no_watermark"

    mono_out_path = translation_config.get_output_file_path(
        f"{basename}{debug_suffix}.{translation_config.lang_out}.mono.pdf",
    )

    for il_page in docs.page:
        page_num = il_page.page_number
        if page_num is None or page_num >= len(pdf):
            logger.warning(
                "Skipping IL page %s (out of range for %d-page PDF)",
                page_num,
                len(pdf),
            )
            continue

        page = pdf[page_num]
        page_height = page.rect.height

        # CropBox origin offset — BabelDOC boxes may be in MediaBox space
        cropbox = page.cropbox
        cropbox_origin_x = cropbox.x0 if cropbox else 0.0
        cropbox_origin_y = cropbox.y0 if cropbox else 0.0

        # --- Pass 1: collect redaction rects + paragraph data ----------------
        para_data: list[
            tuple[pymupdf.Rect, str, float, str]
        ] = []  # (rect, text, font_size, css_color)

        for paragraph in il_page.pdf_paragraph:
            if not paragraph.box:
                continue
            if not paragraph.unicode or not paragraph.unicode.strip():
                continue
            if not paragraph.pdf_paragraph_composition:
                continue

            rect = _il_box_to_rect(
                paragraph.box,
                page_height,
                cropbox_origin_x,
                cropbox_origin_y,
            )

            # Skip degenerate rectangles
            if rect.is_empty or rect.is_infinite:
                continue
            if rect.width < 2 or rect.height < 2:
                continue

            # --- font size ---------------------------------------------------
            font_size = 12.0
            if paragraph.pdf_style and paragraph.pdf_style.font_size:
                font_size = paragraph.pdf_style.font_size

            # --- text colour -------------------------------------------------
            css_color = "black"
            if paragraph.pdf_style and paragraph.pdf_style.graphic_state:
                pci = (
                    paragraph.pdf_style.graphic_state.passthrough_per_char_instruction
                    or ""
                )
                rgb = _extract_rgb_color(pci)
                if rgb is not None:
                    # Check for common white text (on dark backgrounds)
                    if all(c > 0.95 for c in rgb):
                        css_color = "white"
                    elif any(c < 0.05 for c in rgb) and all(c < 0.05 for c in rgb):
                        css_color = "black"
                    else:
                        css_color = _rgb_to_css(rgb)

            # Add redaction (remove original text)
            page.add_redact_annot(rect)

            para_data.append((rect, paragraph.unicode, font_size, css_color))

        # Apply all redactions at once —
        # preserve vector graphics and raster images.
        if para_data:
            page.apply_redactions(
                images=pymupdf.PDF_REDACT_IMAGE_NONE,
                graphics=pymupdf.PDF_REDACT_LINE_ART_NONE,
            )

        # --- Pass 2: insert translated text ----------------------------------
        for rect, translated_text, font_size, css_color in para_data:
            css = (
                f"* {{ font-size: {font_size:.1f}px; "
                f"color: {css_color}; "
                f"font-family: sans-serif; }}"
            )

            excess = page.insert_htmlbox(
                rect,
                translated_text,
                css=css,
                scale_low=0.2,  # allow scaling down to 20%
                rotate=0,
            )

            if excess > 0:
                logger.debug(
                    "Text overflow at (%.0f,%.0f)-(%.0f,%.0f): %.0f excess",
                    rect.x0,
                    rect.y0,
                    rect.x1,
                    rect.y1,
                    excess,
                )

    # --- Handle only_include_translated_page ---------------------------------
    if translation_config.only_include_translated_page:
        translated_page_nums = {
            p.page_number
            for p in docs.page
            if translation_config.should_translate_page((p.page_number or 0) + 1)
        }
        all_pages = set(range(len(pdf)))
        pages_to_remove = sorted(all_pages - translated_page_nums)
        if pages_to_remove:
            pdf.delete_pages(pages_to_remove)

    # --- Subset embedded fonts to reduce file size ---------------------------
    try:
        pdf.subset_fonts()
    except Exception:
        logger.debug("subset_fonts failed (non-fatal)", exc_info=True)

    # --- Save mono PDF -------------------------------------------------------
    if not translation_config.no_mono:
        pdf.save(str(mono_out_path), garbage=1, deflate=True, clean=False)
        logger.info("text_swap_write: saved mono PDF → %s", mono_out_path)
    else:
        mono_out_path = None

    # --- Dual PDF (simple: original page then translated page) ---------------
    dual_out_path = None
    if not translation_config.no_dual:
        dual_out_path = translation_config.get_output_file_path(
            f"{basename}{debug_suffix}.{translation_config.lang_out}.dual.pdf",
        )
        try:
            original_pdf = pymupdf.open(original_pdf_path)
            translated_pdf = pymupdf.open(str(mono_out_path)) if mono_out_path else pdf

            if translation_config.use_alternating_pages_dual:
                # Alternating pages: orig-page-0, trans-page-0, orig-page-1, ...
                dual = pymupdf.open()
                page_count = min(len(original_pdf), len(translated_pdf))
                for i in range(page_count):
                    dual.insert_pdf(original_pdf, from_page=i, to_page=i)
                    dual.insert_pdf(translated_pdf, from_page=i, to_page=i)
            else:
                # Side-by-side: for each page, create a double-width page
                dual = pymupdf.open()
                page_count = min(len(original_pdf), len(translated_pdf))
                for i in range(page_count):
                    orig_page = original_pdf[i]
                    trans_page = translated_pdf[i]
                    w = orig_page.rect.width
                    h = orig_page.rect.height

                    if translation_config.dual_translate_first:
                        # Translated on left, original on right
                        new_page = dual.new_page(width=w * 2, height=h)
                        new_page.show_pdf_page(
                            pymupdf.Rect(0, 0, w, h),
                            translated_pdf,
                            i,
                        )
                        new_page.show_pdf_page(
                            pymupdf.Rect(w, 0, w * 2, h),
                            original_pdf,
                            i,
                        )
                    else:
                        # Original on left, translated on right
                        new_page = dual.new_page(width=w * 2, height=h)
                        new_page.show_pdf_page(
                            pymupdf.Rect(0, 0, w, h),
                            original_pdf,
                            i,
                        )
                        new_page.show_pdf_page(
                            pymupdf.Rect(w, 0, w * 2, h),
                            translated_pdf,
                            i,
                        )

            dual.save(str(dual_out_path), garbage=1, deflate=True, clean=False)
            dual.close()
            original_pdf.close()
            if mono_out_path:
                translated_pdf.close()
            logger.info("text_swap_write: saved dual PDF → %s", dual_out_path)
        except Exception:
            logger.exception("text_swap_write: failed to create dual PDF")
            dual_out_path = None

    pdf.close()

    # --- Glossary output (reuse shared context) ------------------------------
    auto_extracted_glossary_path = None
    if (
        translation_config.save_auto_extracted_glossary
        and translation_config.shared_context_cross_split_part.auto_extracted_glossary
    ):
        auto_extracted_glossary_path = translation_config.get_output_file_path(
            f"{basename}{debug_suffix}.{translation_config.lang_out}.glossary.csv"
        )
        with auto_extracted_glossary_path.open("w", encoding="utf-8-sig") as f:
            f.write(
                translation_config.shared_context_cross_split_part.auto_extracted_glossary.to_csv()
            )

    return TranslateResult(
        mono_pdf_path=mono_out_path,
        dual_pdf_path=dual_out_path,
        auto_extracted_glossary_path=auto_extracted_glossary_path,
    )
