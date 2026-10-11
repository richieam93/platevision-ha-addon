"""Presentation-only selection of the most useful capture from the newest pass.

History, searching, exports and saved images are never altered by this module.
The optional clarity value is a lightweight photographic heuristic, not a
measure of OCR correctness; the minimum OCR confidence remains mandatory.
"""

from datetime import datetime, timezone
from difflib import SequenceMatcher
import math
import re


def normalized_plate(text):
    return re.sub(r'[^A-Z0-9]', '', str(text or '').upper())


def confidence_value(value):
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return -1.0
    return number if math.isfinite(number) and 0 <= number <= 1 else -1.0


def parse_time(value):
    try:
        value = str(value or '').replace('Z', '+00:00')
        result = datetime.fromisoformat(value)
        if result.tzinfo:
            result = result.astimezone(timezone.utc).replace(tzinfo=None)
        return result
    except (TypeError, ValueError, OverflowError):
        return None


def plate_image_sharpness(image):
    """Return a bounded 0..1 edge-clarity score for a plate crop.

    Work at a fixed max width to avoid making large crops automatically look
    sharper. The Laplacian variance is a *heuristic*, not a trained classifier;
    keep it out of the OCR confidence, history filtering and plate decisions.
    """
    try:
        import cv2
        if image is None or image.size == 0 or len(image.shape) not in (2, 3):
            return None
        gray = (cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                if len(image.shape) == 3 and image.shape[2] == 3 else
                cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
                if len(image.shape) == 3 and image.shape[2] == 4 else image)
        height, width = gray.shape[:2]
        if width < 12 or height < 8:
            return None
        if width > 220:
            gray = cv2.resize(gray, (220, max(8, round(height * 220 / width))),
                              interpolation=cv2.INTER_AREA)
        variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if not math.isfinite(variance):
            return None
        return round(max(0.0, min(1.0, variance / (variance + 300.0))), 4)
    except Exception:
        # An optional quality heuristic must never make OCR processing fail.
        return None


def _same_capture_group(anchor, row):
    """Never combine two different video jobs or manual uploads by accident."""
    if str(row.get('source') or '') != str(anchor.get('source') or ''):
        return False
    source = str(anchor.get('source') or '').lower()
    if source == 'video_upload':
        anchor_job = str(anchor.get('video_job_id') or '')
        row_job = str(row.get('video_job_id') or '')
        if anchor_job or row_job:
            return bool(anchor_job and anchor_job == row_job)
        anchor_file = str(anchor.get('filename') or '')
        row_file = str(row.get('filename') or '')
        return (anchor_file == row_file) if (anchor_file or row_file) else True
    if source == 'image_upload':
        anchor_upload = str(anchor.get('upload_id') or '')
        row_upload = str(row.get('upload_id') or '')
        if anchor_upload or row_upload:
            return bool(anchor_upload and anchor_upload == row_upload)
        return str(anchor.get('filename') or '') == str(row.get('filename') or '')
    return True


def best_image_candidate(recent_rows, *, minimum=0.8, window_seconds=180,
                         prefer_sharpness=True, sharpness_weight=0.25):
    """Select one pictured capture from the newest matching vehicle pass.

    Minimum OCR confidence is always enforced. The clarity metric affects only
    ranking among otherwise eligible captures. Legacy events without clarity
    data retain pure confidence ordering, avoiding misleading mixed scores.
    The caller supplies newest-first metadata, not decoded image blobs.
    """
    if not recent_rows:
        return None
    try:
        raw_minimum = float(minimum)
        raw_weight = float(sharpness_weight)
        if not all(map(math.isfinite, (raw_minimum, raw_weight))):
            return None
        minimum = max(0.8, min(1.0, raw_minimum))
        window_seconds = max(10, min(900, int(window_seconds)))
        weight = max(0.0, min(0.4, raw_weight))
    except (TypeError, ValueError, OverflowError):
        return None
    anchor = recent_rows[0]
    anchor_time = parse_time(anchor.get('timestamp_local') or anchor.get('timestamp'))
    anchor_plate = normalized_plate(anchor.get('plate_normalized') or anchor.get('plate_text'))
    matches = []
    for row in recent_rows:
        confidence = confidence_value(row.get('confidence'))
        if confidence < minimum or not row.get('images_available', bool(row.get('vehicle_image') or row.get('plate_image'))):
            continue
        if not _same_capture_group(anchor, row):
            continue
        observed = parse_time(row.get('timestamp_local') or row.get('timestamp'))
        if anchor_time and observed:
            delta = (anchor_time - observed).total_seconds()
            if delta < 0 or delta > window_seconds:
                continue
        elif row is not anchor:
            continue
        plate = normalized_plate(row.get('plate_normalized') or row.get('plate_text'))
        similarity = SequenceMatcher(None, anchor_plate, plate).ratio() if anchor_plate and plate else 0
        if similarity < 0.78:
            continue
        clarity = confidence_value(row.get('plate_sharpness'))
        matches.append((row, confidence, similarity, clarity))
    if not matches:
        return None
    use_clarity = bool(prefer_sharpness and weight > 0 and
                       all(score >= 0 for _, _, _, score in matches))
    chosen = max(matches, key=lambda item: (
        item[1] * (1 - weight) + item[3] * weight if use_clarity else item[1],
        item[1], item[2]))
    return chosen[0]
