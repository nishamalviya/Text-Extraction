from __future__ import annotations

import io
import os
import threading
from typing import Dict, List, Optional, Tuple
import re
os.environ.setdefault("FLAGS_use_mkldnn", "0")
os.environ.setdefault("FLAGS_enable_pir_api", "0")
DEFAULT_PDF_ZOOM = float(os.getenv("OCR_PDF_ZOOM", "1.6"))
DEFAULT_MIN_SCORE = float(os.getenv("OCR_MIN_SCORE", "0.6"))
DEFAULT_PREPROCESS = os.getenv("OCR_PREPROCESS", "true").lower() in {"1", "true", "yes"}
DEFAULT_USE_STRUCTURE = os.getenv("OCR_USE_STRUCTURE", "false").lower() in {"1", "true", "yes"}
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from paddleocr import PaddleOCR
from PIL import Image
import fitz
import numpy as np
import httpx

app = FastAPI(title="Document OCR API", version="1.0.0")
_ocr_lock = threading.Lock()
_ocr_cache: Dict[str, PaddleOCR] = {}
def get_ocr(lang: str) -> PaddleOCR:
    lang = (lang or "en").lower().strip()
    if lang in {"tel", "te"}:
        lang = "te"
    if lang not in {"en", "hi", "ta", "te"}:
        raise HTTPException(status_code=400, detail="Unsupported lang. Use 'en','hi','ta','te' or 'multi'.")

    with _ocr_lock:
        if lang not in _ocr_cache:
            try:
                _ocr_cache[lang] = PaddleOCR(
                    use_textline_orientation=True,
                    lang=lang,
                )
            except TypeError:
                _ocr_cache[lang] = PaddleOCR(
                    use_angle_cls=True,
                    lang=lang,
                )
        return _ocr_cache[lang]


def _run_ocr(ocr: PaddleOCR, img_arr):
    try:
        return ocr.ocr(img_arr, cls=True)
    except TypeError:
        return ocr.predict(img_arr)


def _flatten_ocr_result(result) -> List[Dict]:
    lines: List[Dict] = []
    if not result:
        return lines
    if (
        isinstance(result, list)
        and len(result) == 1
        and isinstance(result[0], list)
        and result[0]
        and isinstance(result[0][0], (list, tuple))
        and len(result[0][0]) == 2
    ):
        result = result[0]

    for item in result:
        if isinstance(item, dict):
            text = item.get("rec_text") or item.get("text")
            box = _normalize_box(item.get("bbox") or item.get("points") or item.get("box"))
            score = item.get("rec_score") or item.get("score")
            if text:
                lines.append({"text": str(text), "bbox": box, "score": score})
            continue
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            box, rec = item[0], item[1]
            box = _normalize_box(box)
            text = None
            score = None
            if isinstance(rec, (list, tuple)) and rec:
                text = rec[0]
                if len(rec) > 1:
                    score = rec[1]
            elif isinstance(rec, str):
                text = rec
            if text:
                lines.append({"text": str(text), "bbox": box, "score": score})
    return lines


def _ocr_image_array(ocr: PaddleOCR, img: Image.Image) -> List[str]:
    img = img.convert("RGB")
    img_arr = np.array(img)
    result = _run_ocr(ocr, img_arr)
    lines = _flatten_ocr_result(result)
    return [l["text"] for l in lines if l.get("text")]


def _ocr_lines_with_boxes(ocr: PaddleOCR, img: Image.Image) -> List[Dict]:
    img = img.convert("RGB")
    img_arr = np.array(img)
    result = _run_ocr(ocr, img_arr)
    lines = _flatten_ocr_result(result)
    return [l for l in lines if l.get("bbox")]


def _box_key(box: List[Tuple[float, float]], quant: int = 6) -> Tuple[int, int, int, int]:
    xs = [p[0] for p in box]
    ys = [p[1] for p in box]
    left, right = int(min(xs) // quant), int(max(xs) // quant)
    top, bottom = int(min(ys) // quant), int(max(ys) // quant)
    return (left, top, right, bottom)


def _ocr_lines_with_boxes_multi(ocrs: List[PaddleOCR], img: Image.Image) -> List[Dict]:
    merged: Dict[Tuple[int, int, int, int], Dict] = {}
    for ocr in ocrs:
        lines = _ocr_lines_with_boxes(ocr, img)
        for l in lines:
            box = l.get("bbox")
            if not box:
                continue
            key = _box_key(box)
            score = l.get("score")
            if key not in merged:
                merged[key] = l
            else:
                prev = merged[key]
                prev_score = prev.get("score")
                if score is not None and (prev_score is None or score > prev_score):
                    merged[key] = l
    return list(merged.values())


def _parse_langs(lang: str) -> List[str]:
    if not lang:
        return ["en"]
    lang = lang.strip().lower()
    if lang in {"auto", "multi"}:
        return ["en", "hi", "ta", "te"]
    if "," in lang:
        langs = []
        for part in lang.split(","):
            p = part.strip().lower()
            if p in {"tel", "te"}:
                p = "te"
            if p:
                langs.append(p)
        return langs or ["en"]
    if lang in {"tel", "te"}:
        lang = "te"
    return [lang]


def _detect_script(text: str) -> str:
    for ch in text:
        code = ord(ch)
        if 0x0C00 <= code <= 0x0C7F:
            return "tel"
        if 0x0B80 <= code <= 0x0BFF:
            return "ta"
        if 0x0900 <= code <= 0x097F:
            return "hi"
    return "en"


def _apply_preprocess(img: Image.Image) -> Image.Image:
    from PIL import ImageOps, ImageFilter

    gray = ImageOps.grayscale(img)
    gray = ImageOps.autocontrast(gray)
    gray = gray.filter(ImageFilter.UnsharpMask(radius=1.5, percent=150, threshold=3))
    return gray.convert("RGB")


def _local_process_image(
    img: Image.Image,
    ocrs: List[PaddleOCR],
    lang: str,
    use_structure: bool,
    min_score: float,
    preprocess: bool,
) -> Dict:
    if preprocess:
        img = _apply_preprocess(img)

    if use_structure:
        structured = _structured_from_ppstructure(img, lang)
    else:
        structured = {}

    if len(ocrs) > 1:
        lines_with_boxes = _ocr_lines_with_boxes_multi(ocrs, img)
    else:
        lines_with_boxes = _ocr_lines_with_boxes(ocrs[0], img)

    if min_score > 0:
        lines_with_boxes = [
            l for l in lines_with_boxes if (l.get("score") is None or l["score"] >= min_score)
        ]

    lines = [l["text"] for l in lines_with_boxes]
    tables, paragraphs, key_values = _infer_tables(lines_with_boxes)

    if structured.get("tables"):
        tables = structured.get("tables")
    if not paragraphs:
        paragraphs = _group_paragraphs(lines_with_boxes)
    if not key_values:
        key_values = _extract_key_values(lines_with_boxes)

    languages = sorted({_detect_script(t) for t in lines if t})

    return {
        "text": structured.get("text") or "\n".join(lines),
        "elements": structured.get("elements", []),
        "tables": tables,
        "lines": lines,
        "paragraphs": paragraphs,
        "key_values": key_values,
        "languages": languages,
    }


def _tables_from_markdown(md: str) -> List[Dict]:
    if not md:
        return []
    lines = [ln.strip() for ln in md.splitlines() if ln.strip()]
    tables: List[Dict] = []
    current: List[List[str]] = []
    for ln in lines:
        if ln.startswith("|") and ln.endswith("|"):
            parts = [p.strip() for p in ln.strip("|").split("|")]
            if all(set(p) <= set("-:") for p in parts):
                continue
            current.append(parts)
        else:
            if current:
                tables.append({"title": "markdown_table", "rows": current})
                current = []
    if current:
        tables.append({"title": "markdown_table", "rows": current})
    return tables


def _clean_markdown(md: str) -> str:
    if not md:
        return ""
    import html as _html
    md = re.sub(r"\[tbl-\d+\.html\]\([^)]+\)", "", md)
    md = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", md)
    md = "\n".join([ln.rstrip() for ln in md.splitlines()])
    return _html.unescape(md).strip()


def _extract_header_kv_from_text(lines: List[str]) -> List[Dict[str, str]]:
    kv = []
    seen = set()
    for ln in lines:
        s = ln.strip()
        if not s:
            continue
        if ":" in s:
            k, v = s.split(":", 1)
            k = k.strip()
            v = v.strip()
            if k and v:
                key = (k, v)
                if key not in seen:
                    kv.append({k: v})
                    seen.add(key)
                continue
        for tag in ["IFSC", "A/c", "Account", "Bank"]:
            if tag.lower() in s.lower():
                key = ("info", s)
                if key not in seen:
                    kv.append({"info": s})
                    seen.add(key)
                break
    return kv


def _extract_header_text(lines: List[str]) -> List[str]:
    keywords = [
        "examiner",
        "officer",
        "director",
        "ifsc",
        "bank",
        "a/c",
        "account",
        "cfms",
    ]
    out = []
    seen = set()
    for ln in lines:
        s = ln.strip()
        if not s:
            continue
        low = s.lower()
        if any(k in low for k in keywords):
            if s not in seen:
                out.append(s)
                seen.add(s)
    return out


def _tables_from_html(html: str) -> List[Dict]:
    if not html:
        return []
    try:
        from bs4 import BeautifulSoup
    except Exception:
        return []
    soup = BeautifulSoup(html, "html.parser")
    out = []
    def _is_numeric_cell(s: str) -> bool:
        s = (s or "").strip()
        if not s:
            return False
        s = s.replace(",", "").replace("/", "").replace("-", "")
        return s.isdigit()

    def _clean_total_row(row: List[str]) -> List[str]:
        label = None
        nums = []
        for c in row:
            c_norm = (c or "").strip()
            if not c_norm:
                continue
            if label is None and ("total" in c_norm.lower() or "net amount" in c_norm.lower() or "deduction" in c_norm.lower()):
                label = c_norm
                continue
            if _is_numeric_cell(c_norm):
                nums.append(c_norm)
        if label or nums:
            return [label or "", *nums]
        return row

    def _normalize_table_rows(rows: List[List[str]]) -> List[List[str]]:
        if not rows:
            return rows
        cleaned = []
        for r in rows:
            rr = [c.strip() for c in r]
            cleaned.append(rr)
        cleaned = [r for r in cleaned if any(c for c in r)]
        if not cleaned:
            return cleaned
        max_cols = max(len(r) for r in cleaned)
        norm = []
        for r in cleaned:
            r = r + [""] * (max_cols - len(r))
            norm.append(r)

        keep_cols = []
        for ci in range(max_cols):
            if any(row[ci].strip() for row in norm):
                keep_cols.append(ci)
        norm = [[row[ci] for ci in keep_cols] for row in norm] if keep_cols else norm
        return norm

    def _find_index_row(rows: List[List[str]]) -> int:
        for i, r in enumerate(rows):
            nums = [c.strip() for c in r if c.strip().isdigit()]
            if len(nums) >= 4:
                return i
        return -1

    def _is_header_like(row: List[str]) -> bool:
        cells = [c.strip() for c in row if c.strip()]
        if not cells:
            return False
        avg_len = sum(len(c) for c in cells) / len(cells)
        has_colon = any(":" in c for c in cells)
        numeric_cells = sum(1 for c in cells if c.replace("/", "").replace("-", "").isdigit())
        return avg_len <= 24 and numeric_cells == 0 and not has_colon and len(cells) >= 2

    def _header_block(rows: List[List[str]], idx_row: int) -> List[List[str]]:
        if idx_row <= 0:
            return rows[:1]
        header_rows = []
        i = idx_row - 1
        while i >= 0 and _is_header_like(rows[i]):
            header_rows.insert(0, rows[i])
            i -= 1
        return header_rows or rows[:idx_row]

    def _build_columns(rows: List[List[str]], idx_row: int) -> List[str]:
        if not rows:
            return []
        header_rows = _header_block(rows, idx_row) if idx_row >= 0 else rows[:1]
        col_count = max(len(r) for r in rows)
        cols = []
        for ci in range(col_count):
            parts = []
            for hr in header_rows:
                if ci < len(hr):
                    v = hr[ci].strip()
                    if v and (not parts or parts[-1] != v):
                        parts.append(v)
            if parts:
                label = " / ".join(parts).strip()
            else:
                label = ""
            cols.append(label if label else f"col_{ci+1}")
        header_text = " ".join([" ".join(hr).lower() for hr in header_rows])
        if "remarks" in header_text and not any(c.lower() == "remarks" for c in cols if c):
            for i in range(len(cols) - 1, -1, -1):
                if cols[i].startswith("col_") or not cols[i].strip():
                    cols[i] = "Remarks"
                    break
        return cols

    def _build_kv_rows_from_columns(columns: List[str], rows: List[List[str]], start_idx: int) -> List[Dict[str, str]]:
        kv_rows = []
        seen = set()
        for r in rows[start_idx:]:
            non_empty = [c for c in r if c.strip()]
            if not non_empty:
                continue
            density = len(non_empty) / max(1, len(r))
            if density < 0.25:
                continue
            kv = {}
            for i, col in enumerate(columns):
                if i < len(r):
                    val = r[i].strip()
                    if val:
                        kv[col] = val
            if kv:
                key = tuple(sorted(kv.items()))
                if key not in seen:
                    kv_rows.append(kv)
                    seen.add(key)
        return kv_rows

    def _pairwise_kv(rows: List[List[str]], allow_blank: bool = False) -> List[Dict[str, str]]:
        kv_rows = []
        seen = set()
        for r in rows:
            non_empty = [c for c in r if c.strip()]
            if len(non_empty) < 1:
                continue
            pairs = []
            i = 0
            while i + 1 < len(r):
                k = r[i].strip()
                v = r[i + 1].strip()
                if k and (allow_blank or v):
                    pairs.append((k, v))
                    i += 2
                else:
                    i += 1
            if not pairs:
                continue
            kv = {}
            for k, v in pairs:
                if k not in kv:
                    kv[k] = v
            key = tuple(sorted(kv.items()))
            if key not in seen:
                kv_rows.append(kv)
                seen.add(key)
        return kv_rows

    def _trim_empty_columns(columns: List[str], rows: List[List[str]]) -> (List[str], List[List[str]]):
        if not rows:
            return columns, rows
        max_cols = max(len(r) for r in rows)
        cols = columns + [f"col_{i+1}" for i in range(len(columns), max_cols)]
        norm = [r + [""] * (max_cols - len(r)) for r in rows]
        keep = []
        for ci in range(max_cols):
            if cols[ci].strip() or any(row[ci].strip() for row in norm):
                keep.append(ci)
        new_cols = [cols[i] for i in keep]
        new_rows = [[row[i] for i in keep] for row in norm]
        return new_cols, new_rows

    def _drop_sparse_columns(columns: List[str], rows: List[List[str]]) -> (List[str], List[List[str]]):
        if not rows:
            return columns, rows
        max_cols = max(len(r) for r in rows)
        cols = columns + [f"col_{i+1}" for i in range(len(columns), max_cols)]
        norm = [r + [""] * (max_cols - len(r)) for r in rows]
        keep = []
        for ci in range(max_cols):
            label = cols[ci].strip()
            has_data = any(row[ci].strip() for row in norm)
            if label.startswith("col_") and not has_data:
                continue
            if not label and not has_data:
                continue
            keep.append(ci)
        if not keep:
            return columns, rows
        new_cols = [cols[i] for i in keep]
        new_rows = [[row[i] for i in keep] for row in norm]
        return new_cols, new_rows

    def _trim_row_trailing_blanks(rows: List[List[str]]) -> List[List[str]]:
        trimmed = []
        for r in rows:
            rr = r[:]
            while rr and not rr[-1].strip():
                rr.pop()
            trimmed.append(rr)
        return trimmed

    def _adjust_narrative_column(columns: List[str], rows: List[List[str]]) -> List[str]:
        if not columns or not rows:
            return columns
        narrative_tokens = ["remarks", "purpose", "narration", "ఉద్దేశం"]
        col_stats = []
        for ci, col in enumerate(columns):
            vals = [row[ci] if ci < len(row) else "" for row in rows]
            non_empty = [v for v in vals if v.strip()]
            avg_len = sum(len(v) for v in non_empty) / len(non_empty) if non_empty else 0.0
            col_stats.append((ci, col, len(non_empty), avg_len))

        narrative_cols = [s for s in col_stats if any(t in s[1].lower() for t in narrative_tokens)]
        placeholder_cols = [s for s in col_stats if s[1].startswith("col_")]

        for nci, nlabel, ncount, navg in narrative_cols:
            best = None
            for pci, plabel, pcount, pavg in placeholder_cols:
                if pcount > ncount and pavg >= navg and pavg > 12:
                    if best is None or pavg > best[3]:
                        best = (pci, plabel, pcount, pavg)
            if best:
                columns[nci], columns[best[0]] = columns[best[0]], columns[nci]
        return columns

    def _drop_blank_cells(columns: List[str], rows: List[List[str]]) -> (List[str], List[List[str]]):
        if not rows:
            return columns, rows
        new_rows = []
        for r in rows:
            new_rows.append([c for c in r if c.strip()])
        new_cols = [c for c in columns if c.strip()]
        return new_cols, new_rows

    for t in soup.find_all("table"):
        rows = []
        for tr in t.find_all("tr"):
            cells = []
            for cell in tr.find_all(["th", "td"]):
                import html as _html
                cells.append(_html.unescape(cell.get_text(" ", strip=True)))
            if cells:
                if any(("total" in (c or "").lower() or "net amount" in (c or "").lower() or "deduction" in (c or "").lower()) for c in cells):
                    rows.append(_clean_total_row(cells))
                else:
                    rows.append(cells)
        rows = _normalize_table_rows(rows)
        if rows:
            idx_row = _find_index_row(rows)
            if idx_row >= 0:
                columns = _build_columns(rows, idx_row)
                data_rows = rows[idx_row + 1 :]
                columns, data_rows = _trim_empty_columns(columns, data_rows)
                columns, data_rows = _drop_sparse_columns(columns, data_rows)
                data_rows = _trim_row_trailing_blanks(data_rows)
                columns = _adjust_narrative_column(columns, data_rows)
                grid = {"columns": columns, "rows": data_rows}
                header_rows = _header_block(rows, idx_row)
                header_start = len(rows) - len(rows[idx_row:]) - len(header_rows)
                meta_rows = rows[:max(0, idx_row - len(header_rows))]
                table_kv = []
                for mr in meta_rows:
                    non_empty = [c for c in mr if c.strip()]
                    if not non_empty:
                        continue
                    has_colon = any(":" in c for c in non_empty)
                    numeric_cells = sum(1 for c in non_empty if c.replace("/", "").replace("-", "").isdigit())
                    if has_colon or (len(non_empty) % 2 == 0 and len(non_empty) <= 6 and numeric_cells < len(non_empty)):
                        table_kv.extend(_pairwise_kv([mr], allow_blank=False))
                entry = {"title": "html_table"}
                if table_kv:
                    entry["table_kv"] = table_kv
                entry["table_grid"] = grid
                out.append(entry)
            else:
                table_kv = _pairwise_kv(rows, allow_blank=False)
                if table_kv:
                    out.append({"title": "html_table", "table_kv": table_kv})
                else:
                    columns = _build_columns(rows, 1)
                    data_rows = rows[1:]
                    columns, data_rows = _trim_empty_columns(columns, data_rows)
                    columns, data_rows = _drop_blank_cells(columns, data_rows)
                    grid = {"columns": columns, "rows": data_rows}
                    out.append({"title": "html_table", "table_grid": grid})
    return out




def _normalize_box(box) -> List[Tuple[float, float]]:
    if not box:
        return []
    if isinstance(box, dict):
        box = box.get("points") or box.get("bbox") or box.get("box") or []
    if len(box) == 1 and isinstance(box[0], (list, tuple)) and len(box[0]) == 4:
        box = box[0]
    pts: List[Tuple[float, float]] = []
    for p in box:
        if isinstance(p, (list, tuple)) and len(p) >= 2:
            try:
                pts.append((float(p[0]), float(p[1])))
            except Exception:
                continue
    return pts


def _line_stats(line: Dict) -> Tuple[float, float, float]:
    box = _normalize_box(line.get("bbox") or line.get("box") or line.get("points"))
    ys = [p[1] for p in box] if box else [0.0]
    xs = [p[0] for p in box] if box else [0.0]
    top = float(min(ys))
    bottom = float(max(ys))
    left = float(min(xs))
    return top, bottom, left


def _group_paragraphs(lines: List[Dict]) -> List[str]:
    if not lines:
        return []
    lines_sorted = sorted(lines, key=_line_stats)
    heights = [(_line_stats(l)[1] - _line_stats(l)[0]) for l in lines_sorted]
    median_h = sorted(heights)[len(heights) // 2] if heights else 10.0
    gap_thresh = max(6.0, median_h * 0.8)

    paragraphs: List[List[str]] = []
    current: List[str] = []
    prev_bottom = None
    for line in lines_sorted:
        top, bottom, _ = _line_stats(line)
        if prev_bottom is not None and (top - prev_bottom) > gap_thresh:
            if current:
                paragraphs.append(current)
            current = []
        current.append(line["text"])
        prev_bottom = bottom
    if current:
        paragraphs.append(current)

    return [" ".join(p).strip() for p in paragraphs if p]


def _cluster_by_centers(values: List[float], tol: float) -> List[float]:
    if not values:
        return []
    centers = sorted(values)
    clusters: List[List[float]] = [[centers[0]]]
    for v in centers[1:]:
        if abs(v - clusters[-1][-1]) <= tol:
            clusters[-1].append(v)
        else:
            clusters.append([v])
    return [sum(c) / len(c) for c in clusters]


def _rows_from_lines(lines: List[Dict]) -> List[List[Dict]]:
    if not lines:
        return []
    items = []
    heights = []
    for l in lines:
        box = _normalize_box(l.get("bbox"))
        if len(box) < 4:
            continue
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        left, right = float(min(xs)), float(max(xs))
        top, bottom = float(min(ys)), float(max(ys))
        cx = (left + right) / 2.0
        cy = (top + bottom) / 2.0
        h = bottom - top
        heights.append(h)
        items.append(
            {
                "text": l.get("text", ""),
                "left": left,
                "right": right,
                "top": top,
                "bottom": bottom,
                "cx": cx,
                "cy": cy,
                "h": h,
            }
        )

    if not items:
        return []

    median_h = sorted(heights)[len(heights) // 2] if heights else 12.0
    row_tol = max(6.0, median_h * 0.7)
    items.sort(key=lambda x: x["cy"])
    rows: List[List[Dict]] = []
    for it in items:
        if not rows:
            rows.append([it])
            continue
        if abs(it["cy"] - rows[-1][0]["cy"]) <= row_tol:
            rows[-1].append(it)
        else:
            rows.append([it])

    for r in rows:
        r.sort(key=lambda x: x["left"])
    return rows


def _build_entries(lines: List[Dict]) -> List[Dict]:
    entries = []
    for l in lines:
        box = _normalize_box(l.get("bbox"))
        if len(box) < 4:
            continue
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        left, right = float(min(xs)), float(max(xs))
        top, bottom = float(min(ys)), float(max(ys))
        cx = (left + right) / 2.0
        cy = (top + bottom) / 2.0
        entries.append(
            {
                "text": l.get("text", ""),
                "left": left,
                "right": right,
                "top": top,
                "bottom": bottom,
                "cx": cx,
                "cy": cy,
                "h": bottom - top,
            }
        )
    return entries


def _normalize_text(s: str) -> str:
    return "".join(ch.lower() for ch in s if ch.isalnum() or ch in ".-")


_NUM_RE = re.compile(r"^\d[\d,]*$")


def _split_alnum_token(text: str) -> List[str]:
    parts = []
    buf = ""
    last_is_digit = None
    for ch in text:
        is_digit = ch.isdigit() or ch == ","
        if last_is_digit is None:
            buf = ch
        elif is_digit != last_is_digit:
            parts.append(buf)
            buf = ch
        else:
            buf += ch
        last_is_digit = is_digit
    if buf:
        parts.append(buf)
    return [p for p in parts if p.strip()]


def _extract_table_by_header(entries: List[Dict], header_keys: List[str], stop_keys: List[str], title: str) -> Optional[Dict]:
    if not entries:
        return None
    heights = [e["h"] for e in entries]
    median_h = sorted(heights)[len(heights) // 2] if heights else 12.0
    row_tol = max(6.0, median_h * 0.7)

    entries_sorted = sorted(entries, key=lambda x: x["cy"])
    rows: List[List[Dict]] = []
    for e in entries_sorted:
        if not rows:
            rows.append([e])
            continue
        if abs(e["cy"] - rows[-1][0]["cy"]) <= row_tol:
            rows[-1].append(e)
        else:
            rows.append([e])

    header_row = None
    header_row_score = 0
    for r in rows:
        row_text = " ".join([e["text"] for e in r])
        score = 0
        for key in header_keys:
            if _normalize_text(key) in _normalize_text(row_text):
                score += 1
        if score > header_row_score:
            header_row_score = score
            header_row = r
    if not header_row or header_row_score < min(3, len(header_keys)):
        return None

    header_row.sort(key=lambda x: x["cx"])
    col_centers = [h["cx"] for h in header_row]
    headers = [h["text"] for h in header_row]
    header_y = sorted([h["cy"] for h in header_row])[len(header_row) // 2]

    stop_y = None
    for e in entries:
        t = _normalize_text(e["text"])
        if e["cy"] > header_y and any(_normalize_text(k) in t for k in stop_keys):
            stop_y = e["top"]
            break

    rows = []
    for e in entries:
        if e["cy"] <= header_y + row_tol:
            continue
        if stop_y is not None and e["cy"] >= stop_y:
            continue
        rows.append(e)

    if not rows:
        return None

    rows.sort(key=lambda x: x["cy"])
    grouped: List[List[Dict]] = []
    for e in rows:
        if not grouped:
            grouped.append([e])
            continue
        if abs(e["cy"] - grouped[-1][0]["cy"]) <= row_tol:
            grouped[-1].append(e)
        else:
            grouped.append([e])

    table_rows: List[List[str]] = []
    for g in grouped:
        g.sort(key=lambda x: x["cx"])
        row_cells = [""] * len(col_centers)
        for e in g:
            idx = min(range(len(col_centers)), key=lambda i: abs(e["cx"] - col_centers[i]))
            row_cells[idx] = (row_cells[idx] + " " + e["text"]).strip() if row_cells[idx] else e["text"]
        if any(row_cells):
            table_rows.append(row_cells)

    if len(table_rows) < 2:
        return None

    if title == "earnings_deductions" and len(table_rows[0]) >= 4:
        for r in table_rows:
            split_cells = []
            for c in r:
                if not c:
                    split_cells.append([])
                else:
                    toks = []
                    for t in c.split():
                        toks.extend(_split_alnum_token(t))
                    split_cells.append(toks)

            rebuilt = ["", "", "", ""]
            if split_cells[0]:
                rebuilt[0] = " ".join([t for t in split_cells[0] if not _NUM_RE.match(t)])
                nums = [t for t in split_cells[0] if _NUM_RE.match(t)]
                if nums:
                    rebuilt[1] = " ".join(nums)
            if len(split_cells) > 2 and split_cells[2]:
                rebuilt[2] = " ".join([t for t in split_cells[2] if not _NUM_RE.match(t)])
                nums = [t for t in split_cells[2] if _NUM_RE.match(t)]
                if nums:
                    rebuilt[3] = " ".join(nums)
            if len(split_cells) > 1 and split_cells[1]:
                nums = [t for t in split_cells[1] if _NUM_RE.match(t)]
                if nums:
                    rebuilt[1] = " ".join(nums)
            if len(split_cells) > 3 and split_cells[3]:
                nums = [t for t in split_cells[3] if _NUM_RE.match(t)]
                if nums:
                    rebuilt[3] = " ".join(nums)
            r[:] = rebuilt

    return {
        "title": title,
        "columns": col_centers,
        "headers": headers,
        "rows": table_rows,
    }


def _rows_to_cells(rows: List[List[Dict]]) -> List[List[Dict]]:
    cell_rows: List[List[Dict]] = []
    for r in rows:
        if not r:
            continue
        gaps = []
        for i in range(len(r) - 1):
            gaps.append(r[i + 1]["left"] - r[i]["right"])
        median_gap = sorted(gaps)[len(gaps) // 2] if gaps else 0.0
        gap_thresh = max(18.0, median_gap * 1.5)

        cells: List[Dict] = []
        cur = {"text": r[0]["text"], "left": r[0]["left"], "right": r[0]["right"]}
        for i in range(1, len(r)):
            gap = r[i]["left"] - r[i - 1]["right"]
            if gap > gap_thresh:
                cells.append(cur)
                cur = {"text": r[i]["text"], "left": r[i]["left"], "right": r[i]["right"]}
            else:
                cur["text"] = (cur["text"] + " " + r[i]["text"]).strip()
                cur["right"] = r[i]["right"]
        cells.append(cur)
        cell_rows.append(cells)
    return cell_rows


def _align_cells_to_columns(cell_rows: List[List[Dict]]) -> Tuple[List[float], List[List[str]]]:
    if not cell_rows:
        return [], []
    centers = []
    widths = []
    for r in cell_rows:
        for c in r:
            cx = (c["left"] + c["right"]) / 2.0
            centers.append(cx)
            widths.append(c["right"] - c["left"])
    median_w = sorted(widths)[len(widths) // 2] if widths else 50.0
    col_tol = max(25.0, median_w * 0.7)
    col_centers = _cluster_by_centers(centers, col_tol)
    col_centers.sort()

    aligned_rows: List[List[str]] = []
    for r in cell_rows:
        row_cells = [""] * len(col_centers)
        for c in r:
            cx = (c["left"] + c["right"]) / 2.0
            idx = min(range(len(col_centers)), key=lambda i: abs(cx - col_centers[i]))
            row_cells[idx] = c["text"] if not row_cells[idx] else (row_cells[idx] + " " + c["text"]).strip()
        aligned_rows.append(row_cells)
    return col_centers, aligned_rows


def _detect_tables_from_aligned_rows(aligned_rows: List[List[str]]) -> List[Dict]:
    if not aligned_rows:
        return []
    col_counts = [sum(1 for c in r if c) for r in aligned_rows]
    tables: List[Dict] = []
    start = None
    for i, cnt in enumerate(col_counts):
        is_table_row = cnt >= 3
        if is_table_row and start is None:
            start = i
        if not is_table_row and start is not None:
            if i - start >= 2:
                tables.append({"start_row": start, "end_row": i - 1})
            start = None
    if start is not None and len(aligned_rows) - start >= 2:
        tables.append({"start_row": start, "end_row": len(aligned_rows) - 1})
    return tables


def _infer_tables(lines: List[Dict], entries_override: Optional[List[Dict]] = None) -> Tuple[List[Dict], List[str], List[Dict]]:
    entries = entries_override if entries_override is not None else _build_entries(lines)
    tables: List[Dict] = []

    tables = []

    rows = _rows_from_lines(lines)
    cell_rows = _rows_to_cells(rows)
    col_centers, aligned_rows = _align_cells_to_columns(cell_rows)
    tables_meta = _detect_tables_from_aligned_rows(aligned_rows)

    table_rows_idx = set()
    for t in tables_meta:
        start, end = t["start_row"], t["end_row"]
        table_rows_idx.update(range(start, end + 1))
        if not tables:
            tables.append(
                {
                    "columns": col_centers,
                    "rows": aligned_rows[start : end + 1],
                }
            )

    paragraphs: List[str] = []
    key_values: List[Dict] = []
    for idx, row in enumerate(aligned_rows):
        if idx in table_rows_idx:
            continue
        text = " ".join([c for c in row if c]).strip()
        if not text:
            continue
        if ":" in text:
            key, value = text.split(":", 1)
            key = key.strip()
            value = value.strip()
            if key and value:
                key_values.append({"key": key, "value": value})
                continue
        paragraphs.append(text)

    return tables, paragraphs, key_values


def _extract_key_values(lines: List[Dict]) -> List[Dict]:
    kv: List[Dict] = []
    if not lines:
        return kv

    entries = _build_entries(lines)
    heights = [e["h"] for e in entries]
    median_h = sorted(heights)[len(heights) // 2] if heights else 12.0
    row_tol = max(6.0, median_h * 0.7)

    entries.sort(key=lambda x: x["cy"])
    rows: List[List[Dict]] = []
    for e in entries:
        if not rows:
            rows.append([e])
            continue
        if abs(e["cy"] - rows[-1][0]["cy"]) <= row_tol:
            rows[-1].append(e)
        else:
            rows.append([e])

    for r in rows:
        r.sort(key=lambda x: x["left"])
        texts = [e["text"] for e in r]
        row_text = " ".join(texts).strip()
        if ":" in row_text:
            key, value = row_text.split(":", 1)
            key = key.strip()
            value = value.strip()
            if key and value:
                kv.append({"key": key, "value": value})
                continue
      
        key_idx = None
        for i, e in enumerate(r):
            if ":" in e["text"]:
                key_idx = i
                break
        if key_idx is not None:
            key_text = r[key_idx]["text"].split(":", 1)[0].strip()
            value_parts = [e["text"] for e in r[key_idx + 1 :]]
            value_text = " ".join(value_parts).strip()
            if key_text and value_text:
                kv.append({"key": key_text, "value": value_text})
                continue

        if len(r) >= 2:
            left = r[0]
            right = r[-1]
            if right["left"] - left["right"] > max(40.0, median_h * 2):
                key_text = left["text"].strip()
                value_text = " ".join([e["text"] for e in r[1:]]).strip()
                if key_text and value_text:
                    kv.append({"key": key_text, "value": value_text})

    return kv


def _structured_from_ppstructure(img: Image.Image, lang: str) -> Dict:
    try:
        from paddleocr import PPStructure
    except Exception:
        return {}

    img_arr = np.array(img.convert("RGB"))
    plang = (lang or "en").lower()
    if plang in {"multi", "auto"}:
        plang = "en"
    if plang == "hi":
        plang = "devanagari"
    if plang not in {"en", "ta", "te", "devanagari"}:
        plang = "en"
    try:
        structure = PPStructure(lang=plang, table=True, ocr=True, layout=True)
    except TypeError:
        try:
            structure = PPStructure(lang=plang)
        except TypeError:
            structure = PPStructure()

    res = structure(img_arr) or []
    elements: List[Dict] = []
    flat_text: List[str] = []
    tables: List[Dict] = []

    for el in res:
        if not isinstance(el, dict):
            continue
        etype = el.get("type") or "unknown"
        bbox = el.get("bbox")
        out = {"type": etype, "bbox": bbox}
        if etype == "table":
            html = None
            if isinstance(el.get("res"), dict):
                html = el["res"].get("html")
            if not html:
                html = el.get("html")
            out["table_html"] = html
            tables.append(out)
        else:
            text = None
            r = el.get("res")
            if isinstance(r, list):
                parts = []
                for ritem in r:
                    if isinstance(ritem, dict) and ritem.get("text"):
                        parts.append(str(ritem["text"]))
                if parts:
                    text = "\n".join(parts)
            elif isinstance(r, dict):
                text = r.get("text") or r.get("html")
            elif isinstance(r, str):
                text = r
            if text:
                out["text"] = text
                flat_text.append(text)
        elements.append(out)

    return {"elements": elements, "tables": tables, "text": "\n".join(flat_text).strip()}


def _pdf_to_images(data: bytes, zoom: Optional[float] = None) -> List[Image.Image]:
    doc = fitz.open(stream=data, filetype="pdf")
    images: List[Image.Image] = []
    z = zoom if zoom is not None else DEFAULT_PDF_ZOOM
    mat = fitz.Matrix(z, z)
    for page in doc:
        pix = page.get_pixmap(matrix=mat)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        images.append(img)
    return images


def _doctr_available() -> bool:
    try:
        import doctr
        return True
    except Exception:
        return False


def _mistral_key() -> str:
    key = "MgLlbypqBJPSjVxrpWwRHqxSSjCcKkYO"
    if not key:
        raise HTTPException(status_code=500, detail="Error processing")
    return key


async def _mistral_upload_file(data: bytes, filename: str) -> str:
    key = _mistral_key()
    headers = {"Authorization": f"Bearer {key}"}
    files = {"file": (filename or "document.pdf", data)}
    data_form = {"purpose": "ocr"}
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post("https://api.mistral.ai/v1/files", data=data_form, files=files, headers=headers)
        if resp.status_code >= 400:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)
        body = resp.json()
        file_id = body.get("id")
        if not file_id:
            raise HTTPException(status_code=500, detail="Mistral file upload failed: missing file id.")
        return file_id


async def _mistral_ocr(data: bytes, filename: str) -> Dict:
    key = _mistral_key()
    file_id = await _mistral_upload_file(data, filename)
    model = os.getenv("MISTRAL_OCR_MODEL", "mistral-ocr-latest")
    payload = {
        "model": model,
        "document": {"type": "file", "file_id": file_id},
        "table_format": os.getenv("MISTRAL_TABLE_FORMAT", "html"),
        "extract_header": True,
        "extract_footer": True,
    }
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post("https://api.mistral.ai/v1/ocr", json=payload, headers=headers)
        if resp.status_code >= 400:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)
        return resp.json()


def _doctr_lines_from_pdf(data: bytes) -> List[List[Dict]]:
    from doctr.io import DocumentFile
    from doctr.models import ocr_predictor

    doc = DocumentFile.from_pdf(data)
    predictor = ocr_predictor(pretrained=True)
    result = predictor(doc)

    pages_lines: List[List[Dict]] = []
    for page in result.pages:
        h, w = page.dimensions
        lines: List[Dict] = []
        for block in page.blocks:
            for line in block.lines:
                (xmin, ymin, xmax, ymax) = line.geometry
                box = [
                    (xmin * w, ymin * h),
                    (xmax * w, ymin * h),
                    (xmax * w, ymax * h),
                    (xmin * w, ymax * h),
                ]
                text = line.render()
                if text:
                    lines.append({"text": text, "bbox": box})
        pages_lines.append(lines)
    return pages_lines


def _doctr_lines_from_image(img: Image.Image) -> List[Dict]:
    from doctr.io import DocumentFile
    from doctr.models import ocr_predictor
    img = img.convert("RGB")
    doc = DocumentFile.from_images([np.array(img)])
    predictor = ocr_predictor(pretrained=True)
    result = predictor(doc)
    page = result.pages[0]
    h, w = page.dimensions
    lines: List[Dict] = []
    for block in page.blocks:
        for line in block.lines:
            (xmin, ymin, xmax, ymax) = line.geometry
            box = [
                (xmin * w, ymin * h),
                (xmax * w, ymin * h),
                (xmax * w, ymax * h),
                (xmin * w, ymax * h),
            ]
            text = line.render()
            if text:
                lines.append({"text": text, "bbox": box})
    return lines


def _doctr_entries_from_pdf(data: bytes) -> List[List[Dict]]:
    from doctr.io import DocumentFile
    from doctr.models import ocr_predictor

    doc = DocumentFile.from_pdf(data)
    predictor = ocr_predictor(pretrained=True)
    result = predictor(doc)

    pages_entries: List[List[Dict]] = []
    for page in result.pages:
        h, w = page.dimensions
        entries: List[Dict] = []
        for block in page.blocks:
            for line in block.lines:
                for word in line.words:
                    (xmin, ymin, xmax, ymax) = word.geometry
                    box = [
                        (xmin * w, ymin * h),
                        (xmax * w, ymin * h),
                        (xmax * w, ymax * h),
                        (xmin * w, ymax * h),
                    ]
                    if word.value:
                        entries.append({"text": word.value, "bbox": box})
        pages_entries.append(entries)
    return pages_entries


def _doctr_entries_from_image(img: Image.Image) -> List[Dict]:
    from doctr.io import DocumentFile
    from doctr.models import ocr_predictor

    img = img.convert("RGB")
    doc = DocumentFile.from_images([np.array(img)])
    predictor = ocr_predictor(pretrained=True)
    result = predictor(doc)

    page = result.pages[0]
    h, w = page.dimensions
    entries: List[Dict] = []
    for block in page.blocks:
        for line in block.lines:
            for word in line.words:
                (xmin, ymin, xmax, ymax) = word.geometry
                box = [
                    (xmin * w, ymin * h),
                    (xmax * w, ymin * h),
                    (xmax * w, ymax * h),
                    (xmin * w, ymax * h),
                ]
                if word.value:
                    entries.append({"text": word.value, "bbox": box})
    return entries


def _docx_text(data: bytes) -> str:
    try:
        import docx
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"python-docx not available: {exc}")

    doc = docx.Document(io.BytesIO(data))
    parts = [p.text for p in doc.paragraphs if p.text]
    return "\n".join(parts)


async def _ocr_file(
    file: UploadFile = File(...),
    lang: str = Form("multi"),
    mode: str = Form("structured"),
    backend: str = Form("paddle"),
    min_score: float = Form(DEFAULT_MIN_SCORE),
    preprocess: bool = Form(DEFAULT_PREPROCESS),
):
    filename = (file.filename or "").lower()
    data = await file.read()

    if not data:
        raise HTTPException(status_code=400, detail="Empty file.")

    if backend == "mistral":
        result = await _mistral_ocr(data, file.filename or "")
        pages_out = []
        all_text = []
        pages = result.get("pages") or result.get("data") or []
        for i, p in enumerate(pages, start=1):
            md = p.get("markdown") or p.get("text") or p.get("content") or ""
            md = _clean_markdown(md)
            lines = [ln for ln in md.splitlines() if ln.strip()]
            titles = [ln.lstrip("#").strip() for ln in lines if ln.strip().startswith("#")]
            header_kv = _extract_header_kv_from_text(lines)
            header_text = _extract_header_text(lines)
            tables = []
            raw_tables = p.get("tables") or []
            for rt in raw_tables:
                if isinstance(rt, dict) and rt.get("format") == "html":
                    tables.extend(_tables_from_html(rt.get("content", "")))
            if not tables:
                tables = _tables_from_markdown(md)
            if not header_kv and tables:
                derived = []
                for t in tables:
                    for kv in t.get("table_kv", [])[:3]:
                        derived.append(kv)
                    grid = t.get("table_grid", {})
                    for row in grid.get("rows", []):
                        line = " ".join([c for c in row if c.strip()])
                        if any(k in line.lower() for k in ["ifsc", "bank", "a/c", "account", "cfms"]):
                            derived.append({"info": line})
                header_kv = derived
            if not header_text and header_kv:
                header_text = []
                seen = set()
                for kv in header_kv:
                    for v in kv.values():
                        if v and v not in seen:
                            header_text.append(v)
                            seen.add(v)
            languages = sorted({ _detect_script(t) for t in lines if t })
            pages_out.append(
                {
                    "page": i,
                    "tables": tables,
                    "paragraphs": [ln for ln in lines],
                    "key_values": [],
                    "languages": languages,
                }
            )
            all_text.append(md)
        if not pages_out:
            text = result.get("text") or result.get("markdown") or ""
            all_text.append(text)
        return JSONResponse(
            {
                "filename": file.filename,
                "lang": lang,
                "mode": mode,
                "pages": pages_out,
            }
        )

    if filename.endswith(".pdf"):
        langs = _parse_langs(lang)
        ocrs = [get_ocr(l) for l in langs] if backend == "paddle" else []
        images = _pdf_to_images(data) if backend == "paddle" else []
        pages = []
        all_lines: List[str] = []
        if backend == "doctr":
            if not _doctr_available():
                raise HTTPException(status_code=500, detail="DocTR backend not installed.")
            doctr_pages = _doctr_lines_from_pdf(data)
            doctr_entries_pages = _doctr_entries_from_pdf(data)
            doctr_images = _pdf_to_images(data)
            for i, lines_with_boxes in enumerate(doctr_pages, start=1):
                entries = doctr_entries_pages[i - 1] if i - 1 < len(doctr_entries_pages) else None
                lines = [l["text"] for l in lines_with_boxes]
                tables, paragraphs, key_values = _infer_tables(lines_with_boxes, entries_override=entries)
                languages = sorted({ _detect_script(t) for t in lines if t })
                pages.append(
                    {
                        "page": i,
                        "text": "\n".join(lines),
                        "elements": [],
                        "tables": tables,
                        "lines": lines,
                        "paragraphs": paragraphs,
                        "key_values": key_values,
                        "languages": languages,
                    }
                )
                all_lines.extend(lines)
            return JSONResponse(
                {
                    "filename": file.filename,
                    "lang": lang,
                    "mode": mode,
                    "backend": backend,
                    "text": "\n".join(all_lines),
                    "pages": pages,
                }
            )

        for i, img in enumerate(images, start=1):
            page = _local_process_image(
                img=img,
                ocrs=ocrs,
                lang=lang,
                use_structure=(mode == "structured" and DEFAULT_USE_STRUCTURE),
                min_score=min_score,
                preprocess=preprocess,
            )
            page["page"] = i
            pages.append(page)
            all_lines.extend(page["lines"])
        return JSONResponse(
            {
                "filename": file.filename,
                "lang": lang,
                "mode": mode,
                "text": "\n".join(all_lines),
                "pages": pages,
            }
        )

    if filename.endswith(".docx"):
        text = _docx_text(data)
        return JSONResponse({"filename": file.filename, "lang": "docx", "text": text, "pages": []})

    if filename.endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp")):
        langs = _parse_langs(lang)
        ocrs = [get_ocr(l) for l in langs] if backend == "paddle" else []
        img = Image.open(io.BytesIO(data))
        if backend == "doctr":
            if not _doctr_available():
                raise HTTPException(status_code=500, detail="DocTR backend not installed.")
            lines_with_boxes = _doctr_lines_from_image(img)
            entries = _doctr_entries_from_image(img)
            lines = [l["text"] for l in lines_with_boxes]
            tables, paragraphs, key_values = _infer_tables(lines_with_boxes, entries_override=entries)
            languages = sorted({ _detect_script(t) for t in lines if t })
            return JSONResponse(
                {
                    "filename": file.filename,
                    "lang": lang,
                    "mode": mode,
                    "text": "\n".join(lines),
                    "pages": [
                        {
                            "page": 1,
                            "text": "\n".join(lines),
                            "elements": [],
                            "tables": tables,
                            "lines": lines,
                            "paragraphs": paragraphs,
                            "key_values": key_values,
                            "languages": languages,
                        }
                    ],
                }
            )
        page = _local_process_image(
            img=img,
            ocrs=ocrs,
            lang=lang,
            use_structure=(mode == "structured" and DEFAULT_USE_STRUCTURE),
            min_score=min_score,
            preprocess=preprocess,
        )
        return JSONResponse(
            {
                "filename": file.filename,
                "lang": lang,
                "mode": mode,
                "text": page["text"],
                "pages": [
                    {
                        "page": 1,
                        **page,
                    }
                ],
            }
        )

    raise HTTPException(status_code=400, detail="Unsupported file type. Use PDF, DOCX, or image files.")


@app.post("/extract-text", include_in_schema=False)
async def extract_text(file: UploadFile = File(...)):
    return await _ocr_file(
        file=file,
        lang="multi",
        mode="structured",
        backend="paddle",
        min_score=DEFAULT_MIN_SCORE,
        preprocess=DEFAULT_PREPROCESS,
    )


@app.post("/text-extraction-ocr", summary="Text Extraction OCR")
async def text_extraction_ocr(file: UploadFile = File(...)):
    return await _ocr_file(
        file=file,
        lang="multi",
        mode="structured",
        backend="mistral",
        min_score=0.0,
        preprocess=False,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
