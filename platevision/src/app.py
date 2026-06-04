"""
PlateVision - License Plate Detection System
Flask-based Web Application with RTSP Support
Version 0.8.23 FastPlateOCR Vehicle Intelligence
"""

from flask import Flask, render_template, request, jsonify, Response, send_from_directory
from flask_socketio import SocketIO, emit
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
import threading
import time
import json
import os
import uuid
from datetime import datetime, timedelta
import base64
from PIL import Image
import io
import queue
import logging
import re
import csv
from difflib import SequenceMatcher
from collections import Counter, defaultdict
from pathlib import Path

# ============================================================
# KONFIGURATION & INITIALISIERUNG
# ============================================================

app = Flask(__name__, 
            static_folder='static', 
            template_folder='templates')
app.config['SECRET_KEY'] = 'platevision_secret_2024'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max upload

socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Logging konfigurieren
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================
# VERZEICHNISSE ERSTELLEN
# ============================================================

DIRECTORIES = [
    'uploads',
    'uploads/images',
    'uploads/videos',
    'uploads/processed',
    'uploads/people_tests',
    'uploads/models',
    'static',
    'static/css',
    'static/js',
    'static/images',
    'data',
    'data/plates_detected',
    'data/vehicles_detected',
    'data/history',
    'data/people',
    'data/people/images',
    'data/people/images/crops',
    'data/people/images/full_frames',
    'data/people/images/annotated',
    'data/models',
    'models',
    'templates'
]

for directory in DIRECTORIES:
    Path(directory).mkdir(parents=True, exist_ok=True)

# ============================================================
# VIDEO PROCESSING JOBS
# ============================================================

video_processing_jobs = {}


# ============================================================
# KENNZEICHEN-HILFSFUNKTIONEN
# ============================================================

class PlateUtils:
    """Normalisierung, Validierung, Kandidatenbildung und Ähnlichkeit von Kennzeichen."""

    OCR_CONFUSIONS = str.maketrans({
        'Ä': 'A', 'Ö': 'O', 'Ü': 'U', 'À': 'A', 'Á': 'A', 'Â': 'A',
        'È': 'E', 'É': 'E', 'Ê': 'E', 'Ì': 'I', 'Í': 'I', 'Î': 'I',
        'Ò': 'O', 'Ó': 'O', 'Ô': 'O', 'Ù': 'U', 'Ú': 'U', 'Û': 'U',
        ' ': '', '-': '', '_': '', '.': '', ':': '', '/': '', '\\': '',
        '|': '1', '·': '', '•': '', '*': '', ',': '', ';': ''
    })
    LETTER_TO_DIGIT = {
        'O': '0', 'Q': '0', 'D': '0', 'I': '1', 'L': '1', 'T': '1',
        'Z': '2', 'S': '5', 'B': '8', 'G': '6', 'A': '4'
    }
    DIGIT_TO_LETTER = {
        '0': 'O', '1': 'I', '2': 'Z', '5': 'S', '8': 'B', '6': 'G', '4': 'A'
    }
    VALID_RE = re.compile(r'^[A-Z0-9ÄÖÜ\- ]{2,14}$')

    # Keep detect_format() returning a string for existing code.
    COUNTRY_PATTERNS = [
        ('FL', re.compile(r'^FL\s?\d{1,5}$')),                         # FL 12345
        ('CH', re.compile(r'^[A-Z]{2}\s?\d{1,6}$')),                    # OW 12345
        ('DE', re.compile(r'^[A-ZÄÖÜ]{1,3}\s?[A-Z]{1,2}\s?\d{1,4}[EH]?$')),
        ('AT', re.compile(r'^[A-Z]{1,2}\s?\d{1,5}\s?[A-Z]{1,2}$')),
        ('FR/IT/Generic-EU', re.compile(r'^[A-Z]{2}\s?\d{3}\s?[A-Z]{2}$')),
        ('NL', re.compile(r'^[A-Z0-9]{2}\s?[A-Z0-9]{2}\s?[A-Z0-9]{2}$')),
        ('Generic', re.compile(r'^[A-Z0-9]{3,12}$')),
    ]

    @classmethod
    def normalize(cls, text, compact=True):
        if text is None:
            return ''
        value = str(text).upper().strip()
        value = value.replace('\n', ' ').replace('\r', ' ')
        value = re.sub(r'[^A-Z0-9ÄÖÜÀÁÂÈÉÊÌÍÎÒÓÔÙÚÛ\- _./:|·•*,;]', '', value)
        value = re.sub(r'\s+', ' ', value)
        if compact:
            value = value.translate(cls.OCR_CONFUSIONS)
            value = re.sub(r'[^A-Z0-9]', '', value)
        return value

    @classmethod
    def pretty(cls, text):
        value = cls.normalize(text, compact=True)
        if not value:
            return ''
        ch = re.match(r'^([A-Z]{2})(\d{1,6})$', value)
        if ch:
            return f"{ch.group(1)} {ch.group(2)}"
        fl = re.match(r'^(FL)(\d{1,5})$', value)
        if fl:
            return f"{fl.group(1)} {fl.group(2)}"
        eu = re.match(r'^([A-Z]{2})(\d{3})([A-Z]{2})$', value)
        if eu:
            return f"{eu.group(1)}-{eu.group(2)}-{eu.group(3)}"
        de = re.match(r'^([A-ZÄÖÜ]{1,3})([A-Z]{1,2})(\d{1,4}[EH]?)$', value)
        if de:
            return f"{de.group(1)} {de.group(2)} {de.group(3)}"
        return value

    @classmethod
    def detect_format(cls, text):
        raw = str(text or '').upper().strip()
        compact = cls.normalize(raw, compact=True)
        candidates = [raw, compact, cls.pretty(compact)]
        for name, pattern in cls.COUNTRY_PATTERNS:
            for candidate in candidates:
                candidate = str(candidate).replace('-', ' ')
                if pattern.match(candidate) or pattern.match(cls.normalize(candidate, compact=True)):
                    return name
        return 'Unbekannt'

    @classmethod
    def similarity(cls, left, right):
        a = cls.normalize(left, compact=True)
        b = cls.normalize(right, compact=True)
        if not a or not b:
            return 0.0
        return SequenceMatcher(None, a, b).ratio()

    @classmethod
    def is_valid(cls, text, min_len=3, max_len=12, regex=None):
        compact = cls.normalize(text, compact=True)
        if len(compact) < int(min_len or 0) or len(compact) > int(max_len or 99):
            return False
        if regex:
            try:
                return re.match(regex, compact) is not None
            except re.error:
                return False
        if not (cls.VALID_RE.match(str(text or '').upper()) or compact.isalnum()):
            return False
        return cls.detect_format(compact) != 'Unbekannt' or compact.isalnum()

    @classmethod
    def smart_correct(cls, text, country_hint='auto'):
        """Korrigiert typische OCR-Verwechslungen mit Positions- und Länderlogik."""
        compact = cls.normalize(text, compact=True)
        if not compact:
            return ''
        country_hint = (country_hint or 'auto').upper()
        chars = list(compact)
        for i, ch in enumerate(chars):
            prev_digit = i > 0 and chars[i - 1].isdigit()
            next_digit = i < len(chars) - 1 and chars[i + 1].isdigit()
            prev_alpha = i > 0 and chars[i - 1].isalpha()
            next_alpha = i < len(chars) - 1 and chars[i + 1].isalpha()
            if ch in cls.LETTER_TO_DIGIT and (prev_digit or next_digit):
                chars[i] = cls.LETTER_TO_DIGIT[ch]
            elif ch in cls.DIGIT_TO_LETTER and prev_alpha and next_alpha:
                chars[i] = cls.DIGIT_TO_LETTER[ch]
        corrected = ''.join(chars)
        if country_hint in ('AUTO', 'FL'):
            fl = re.match(r'^F[L1IT]([A-Z0-9]{1,5})$', corrected)
            if fl:
                number = ''.join(cls.LETTER_TO_DIGIT.get(c, c) for c in fl.group(1))
                if number.isdigit():
                    return 'FL' + number
        if country_hint in ('AUTO', 'CH', 'FL'):
            m = re.match(r'^([A-Z0-9]{2})([A-Z0-9]{1,6})$', corrected)
            if m:
                prefix = ''.join(cls.DIGIT_TO_LETTER.get(c, c) for c in m.group(1))
                number = ''.join(cls.LETTER_TO_DIGIT.get(c, c) for c in m.group(2))
                if prefix.isalpha() and number.isdigit():
                    return prefix + number
        if country_hint == 'DE':
            # German plates generally start with one to three letters.
            m = re.match(r'^([A-Z0-9]{1,3})([A-Z0-9]{1,2})([A-Z0-9]{1,4}[A-Z0-9]?)$', corrected)
            if m:
                region = ''.join(cls.DIGIT_TO_LETTER.get(c, c) for c in m.group(1))
                letters = ''.join(cls.DIGIT_TO_LETTER.get(c, c) for c in m.group(2))
                number = ''.join(cls.LETTER_TO_DIGIT.get(c, c) for c in m.group(3))
                candidate = region + letters + number
                if cls.detect_format(candidate) in ('DE', 'Generic'):
                    return candidate
        return corrected

    @classmethod
    def generate_candidates(cls, text, country_hint='auto', max_candidates=12):
        """Erzeugt mehrere plausible Kennzeichen-Kandidaten aus OCR-Rohtext."""
        raw = cls.normalize(text, compact=True)
        if not raw:
            return []
        seeds = {raw, cls.smart_correct(raw, country_hint)}
        # Common OCR noise at plate edges.
        seeds.add(raw.strip('I1L|[](){}'))
        seeds.add(raw.strip('O0QD'))
        # Build positional variants.
        for seed in list(seeds):
            if not seed:
                continue
            chars = list(seed)
            for i, ch in enumerate(chars):
                if ch in cls.LETTER_TO_DIGIT:
                    variant = chars.copy()
                    variant[i] = cls.LETTER_TO_DIGIT[ch]
                    seeds.add(''.join(variant))
                if ch in cls.DIGIT_TO_LETTER:
                    variant = chars.copy()
                    variant[i] = cls.DIGIT_TO_LETTER[ch]
                    seeds.add(''.join(variant))
        scored = []
        seen = set()
        for candidate in seeds:
            candidate = cls.normalize(candidate, compact=True)
            if not candidate or candidate in seen:
                continue
            seen.add(candidate)
            scored.append(cls.analyze(candidate, country_hint=country_hint))
        scored.sort(key=lambda item: (item['valid'], item['score'], item['similarity_to_input']), reverse=True)
        return scored[:max_candidates]

    @classmethod
    def analyze(cls, text, country_hint='auto'):
        compact = cls.normalize(text, compact=True)
        corrected = cls.smart_correct(compact, country_hint)
        fmt = cls.detect_format(corrected) if corrected else 'Unbekannt'
        valid = cls.is_valid(corrected)
        score = 0.0
        if corrected:
            score += min(len(corrected) / 10, 1) * 0.25
        if fmt != 'Unbekannt':
            score += 0.45
        if valid:
            score += 0.20
        if corrected and corrected != compact:
            score += 0.05
        if (country_hint or '').upper() in ('CH', 'FL') and fmt in ('CH', 'FL'):
            score += 0.10
        if (country_hint or '').upper() == 'DE' and fmt == 'DE':
            score += 0.10
        return {
            'input': text,
            'normalized': compact,
            'corrected': corrected,
            'pretty': cls.pretty(corrected),
            'format': fmt,
            'valid': bool(valid),
            'score': round(min(score, 1.0), 3),
            'length': len(corrected or compact),
            'similarity_to_input': round(cls.similarity(text, corrected), 3) if corrected else 0,
            'masked': cls.mask(corrected),
        }

    @classmethod
    def best_candidate(cls, text, country_hint='auto'):
        candidates = cls.generate_candidates(text, country_hint=country_hint, max_candidates=10)
        if candidates:
            return candidates[0]
        return cls.analyze(text, country_hint=country_hint)

    @classmethod
    def mask(cls, text, visible_start=2, visible_end=2):
        compact = cls.normalize(text, compact=True)
        if len(compact) <= visible_start + visible_end:
            return compact
        return compact[:visible_start] + '•' * (len(compact) - visible_start - visible_end) + compact[-visible_end:]


# ============================================================
# KONFIGURATIONSMANAGER
# ============================================================

class ConfigManager:
    """Verwaltet alle Einstellungen der Anwendung"""
    
    CONFIG_FILE = 'data/config.json'
    
    DEFAULT_CONFIG = {
        'rtsp': {
            'url': 'rtsp://admin:password@192.168.1.100:554/stream1',
            'enabled': False,
            'reconnect_delay': 5,
            'buffer_size': 10,
            'resolution': {
                'width': 1280,
                'height': 720
            },
            'analysis_area': {
                'enabled': False,
                'mode': 'polygon',
                'mask_outside': True,
                # RTSP CPU-Sparmodus: Erkennung standardmäßig auf den gespeicherten
                # Straßenbereich beschränken. Der Crop bekommt bewusst Zusatzrand,
                # damit Fahrzeuge/LKW/Motorräder nicht abgeschnitten werden.
                'crop_before_detection': True,
                'mask_before_detection': False,
                'crop_padding_percent': 25.0,
                'crop_min_padding_px': 120,
                'motion_gate_enabled': True,
                'motion_gate_threshold_percent': 0.20,
                'motion_gate_hold_seconds': 2.0,
                'motion_gate_idle_scan_seconds': 5.0,
                'coordinate_width': 1280,
                'coordinate_height': 720,
                'area': {
                    'x': 0,
                    'y': 0,
                    'width': 1280,
                    'height': 720
                },
                'polygon': [
                    {'x': 230, 'y': 150},
                    {'x': 900, 'y': 150},
                    {'x': 960, 'y': 720},
                    {'x': 120, 'y': 720}
                ]
            }
        },
        'detection': {
            'confidence_threshold': 0.5,
            'car_detection_enabled': True,
            'zoom_enabled': True,
            'zoom_factor': 2.5,
            'zoom_padding': 100,
            'process_interval': 0.5,
            'save_detected_plates': True,
            'save_detected_vehicles': True,
            'save_full_frame': True,
            'min_plate_width': 20,
            'min_plate_height': 8,
            'plate_aspect_ratio_min': 1.2,
            'plate_aspect_ratio_max': 8.0,
            'vehicle_class_filter': ['car', 'truck', 'bus', 'motorcycle', 'bicycle'],
            'max_detections_per_frame': 8,
            'plate_detector_confidence_factor': 0.6,
            'plate_detector_confidence': 0.25,
            'plate_detector_iou': 0.45,
            'plate_detector_imgsz': 960,
            'plate_detector_max_det': 8,
            'plate_crop_padding_percent': 8.0,
            'plate_scan_strategy': 'full_frame_first',
            'annotate_frames': True,
            'draw_confidence': True,
            'scan_full_frame_when_vehicle_found': True,
            'min_vehicle_width': 80,
            'min_vehicle_height': 60,
            'duplicate_cooldown_per_frame': True,
        },
        'ocr': {
            'languages': ['en', 'de'],
            'gpu_enabled': False,
            'min_confidence': 0.25,
            'allowed_characters': 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789- ÄÖÜ',
            'preprocessing': {
                'enabled': True,
                'resize_factor': 4.0,
                'target_height': 120,
                'min_width': 200,
                'denoise': True,
                'sharpen': True,
                'contrast_enhance': True,
                'adaptive_threshold': True,
                'deskew': True,
                'morphology': True,
                'gamma_correction': False,
                'gamma': 1.2,
                'clahe_clip_limit': 3.0,
                'clahe_tile_grid': 8,
                'denoise_strength': 10,
                'threshold_block_size': 11,
                'threshold_c': 2,
                'invert_variant': True,
                'bilateral_filter': False,
                'perspective_correction': False,
                'border_padding': 6,
            },
            'active_mode': 'enhanced',
            'retry_on_fail': True,
            'max_retries': 3,
            'engine': 'fast_plate_ocr',
            'easyocr_backup_enabled': True,
            'fast_plate_model': 'cct-s-v2-global-model',
            'fast_plate_device': 'auto',
            'fast_plate_providers': '',
            'fast_plate_remove_pad_char': True,
            'fast_plate_return_confidence': True,
            'decoder': 'greedy',
            'paragraph_mode': False,
            'rotation_variants': True,
            'early_stop_confidence': 0.85,
            'max_variants_to_read': 8,
            'min_text_length': 2,
            'merge_fragments': True,
            'uppercase_output': True,
            'use_allowlist': True,
        },
        'general': {
            'theme': 'dark',
            'language': 'de',
            'available_languages': ['de', 'en', 'fr', 'it'],
            'timezone': 'Europe/Zurich',
            'date_format': 'dd.mm.yyyy HH:MM',
            'auto_save_history': True,
            'max_history_entries': 1000,
            'notification_enabled': True,
            'debug_mode': False,
            'startup_page': 'dashboard',
            'time_24h': True,
            'auto_refresh_enabled': True,
            'log_level': 'INFO',
            'accessibility_mode': False,
            'compact_numbers': False,
            'operator_name': '',
            'site_name': 'PlateVision Standort'
        },
        'history': {
            'filter_duplicates': True,
            'duplicate_timeout': 60,
            'min_confidence_to_save': 0.35,
            'save_vehicle_image': True,
            'save_plate_image': True,
            'fuzzy_duplicate_detection': True,
            'fuzzy_duplicate_similarity': 0.88,
            'store_raw_ocr': True,
            'store_candidates': True,
            'store_location_data': True,
            'group_by_visit': True,
            'visit_gap_minutes': 15,
            'export_default_format': 'csv',
            'auto_cleanup_enabled': False,
            'cleanup_days': 365,
            'mark_low_confidence': True,
            'low_confidence_threshold': 0.45,
        },
        'storage': {
            'jpeg_quality_plate': 95,
            'jpeg_quality_vehicle': 90,
            'jpeg_quality_frame': 88,
            'create_daily_folders': True,
            'save_metadata_json': True,
            'auto_cleanup_images': False,
            'cleanup_images_days': 90,
            'max_storage_mb': 0,
            'compress_exports': False,
            'include_thumbnails': True,
            'thumbnail_width': 320,
            'filename_pattern': '{date}_{time}_{plate}_{confidence}',
            'separate_unknown_folder': True
        },
        'plate_recognition': {
            'country_hint': 'CH',
            'min_length': 3,
            'max_length': 12,
            'validation_regex': '',
            'smart_ocr_correction': True,
            'format_pretty_output': True,
            'save_low_confidence_candidates': False,
            'watchlist_enabled': True
        },
        'search': {
            'default_limit': 50,
            'enable_fuzzy_search': True,
            'fuzzy_similarity': 0.72,
            'allow_regex_search': True,
            'remember_last_filters': True
        },
        'dashboard': {
            'auto_refresh_seconds': 10,
            'show_confidence_chart': True,
            'show_hourly_chart': True,
            'show_watchlist': True,
            'show_storage': True,
            'compact_mode': False,
            'default_range_days': 7
        },
        'traffic': {
            'visit_gap_minutes': 15,
            'active_timeout_minutes': 30,
            'daily_count_mode': 'visits',
            'direction_mode': 'auto',
            'min_confidence': 0.0,
            'ignore_unknown_plates': True,
            'include_duplicate_events': False,
            'movement_axis': 'x',
            'movement_threshold_percent': 8,
            'arrival_label': 'gekommen',
            'departure_label': 'gegangen'
        },
        'people': {
            'enabled': True,
            'history_enabled': True,
            'show_on_live': True,
            'draw_boxes': True,
            'confidence_threshold': 0.55,
            'model_mode': 'coco_person',
            'model_path': 'models/best.pt',
            'selected_model_file': 'models/best.pt',
            'fallback_model_files': ['models/best.pt', 'models/last.pt', 'models/human_best.pt'],
            'model_auto_scan': True,
            'model_choices': {
                'coco_person': 'Standard YOLOv8 COCO Person-Klasse',
                'custom_human': 'Eigenes YOLOv8 Human Modell',
                'model_file': 'Modell aus Liste auswählen',
                'custom_path': 'Benutzerdefinierter Modellpfad'
            },
            'custom_model_path': 'models/best.pt',
            'class_ids': [0],
            'class_names': ['person', 'human'],
            'min_person_width': 20,
            'min_person_height': 40,
            'max_persons_per_frame': 30,
            'image_size': 640,
            'nms_iou_threshold': 0.45,
            'min_area_percent': 0.05,
            'max_area_percent': 45.0,
            'roi_filter_enabled': True,
            'roi_filter_mode': 'foot_and_center',
            'line_relative_to_roi': True,
            'reject_partial_outside_roi': True,
            'min_roi_overlap_percent': 25,
            'min_aspect_ratio': 0.25,
            'max_aspect_ratio': 1.2,
            'zone_enabled': False,
            'zone': {'x': 0, 'y': 0, 'width': 100, 'height': 100, 'unit': 'percent'},
            'tracker_enabled': True,
            'tracker_max_distance': 120,
            'tracker_timeout_seconds': 8,
            'min_track_age_frames': 1,
            'count_debounce_seconds': 2,
            'count_strategy': 'line_crossing',
            'count_once_per_track': True,
            'line_crossing_enabled': True,
            'virtual_line_position_percent': 50,
            'movement_axis': 'y',
            'crossing_direction': 'both',
            'session_gap_minutes': 5,
            'present_timeout_minutes': 10,
            'presence_enabled': True,
            'occupancy_estimation': True,
            'save_all_detections': True,
            'save_person_crops': True,
            'save_full_frame': False,
            'privacy_blur_people': False,
            'blur_strength': 35,
            'person_recount_block_enabled': True,
            'person_recount_block_minutes': 15,
            'person_recount_identity_mode': 'track_or_position',
            'person_recount_position_tolerance_percent': 12,
            'image_history_enabled': True,
            'image_history_store_crop': True,
            'image_history_store_annotated': False,
            'image_history_store_full_frame': False,
            'image_history_jpeg_quality': 85,
            'image_history_retention_days': 90,
            'image_history_auto_cleanup_enabled': False,
            'image_history_cleanup_on_add': True,
            'image_history_last_cleanup': '',
            'test_environment_enabled': True,
            'test_image_upload_enabled': True,
            'test_force_enable_people': True,
            'test_save_uploads': False,
            'test_save_to_history_default': True,
            'simulation_enabled': False,
            'retention_days': 90,
            'auto_cleanup_enabled': False,
            'export_default_format': 'csv',
            'alert_threshold_per_hour': 0,
            'settings_preview_enabled': True,
            'settings_preview_refresh_seconds': 5,
            'settings_preview_show_fallback': True,
            'settings_preview_fallback_label': 'RTSP Stream nicht erreichbar - Kalibrierungsbild',
            'calibration_preview_enabled': True,
            'test_apply_saved_settings': True,
            'note': 'Personenzählung ist ohne klare Zähllinie oder zweite Kamera heuristisch.'
        },
        'alerts': {
            'watchlist_notifications': True,
            'unknown_plate_notifications': False,
            'low_confidence_notifications': False,
            'min_alert_confidence': 0.65,
            'webhook_url': '',
            'mqtt_topic_prefix': 'platevision'
        },
        'ui': {
            'accent_color': '#6366f1',
            'density': 'comfortable',
            'animations': True,
            'sidebar_labels': True,
            'card_style': 'glass',
            'show_help_text': True
        },
        'privacy': {
            'mask_plate_numbers': False,
            'blur_plate_images': False,
            'retention_days': 0,
            'export_include_images': False
        },
        'recognition_profiles': {
            'active': 'balanced',
            'profiles': {
                'fast': {'confidence_threshold': 0.6, 'ocr_min_confidence': 0.35, 'process_interval': 1.0},
                'balanced': {'confidence_threshold': 0.5, 'ocr_min_confidence': 0.25, 'process_interval': 0.5},
                'strict': {'confidence_threshold': 0.7, 'ocr_min_confidence': 0.55, 'process_interval': 0.75},
                'night': {'confidence_threshold': 0.35, 'ocr_min_confidence': 0.18, 'process_interval': 0.8}
            }
        },
        'models': {
            'license_plate_detector': 'models/license_plate_detector.pt',
            'vehicle_detector': 'models/yolov8n.pt',
            'auto_reload_on_change': False,
            'warmup_on_start': True,
            'device': 'auto',
            'half_precision': False,
            'model_size_hint': 'nano',
            'download_missing_models': False,
            'fallback_to_cpu': True,
            'custom_model_directory': 'models',
            'additional_model_directories': ['/data/models', '/app/models', 'platevision/src/models'],
            'vehicle_model_labels': 'COCO',
            'plate_model_labels': 'license_plate',
            'person_detector': 'models/best.pt',
            'person_model_labels': 'person,human',
            'person_model_source': 'COCO class 0 or custom YOLOv8 human model',
            'person_model_scan_enabled': True,
            'person_model_extensions': ['.pt', '.onnx', '.engine'],
            'model_upload_enabled': True,
            'model_upload_directory': '/data/models',
            'model_upload_max_mb': 500,
            'model_upload_allow_overwrite': False,
            'model_upload_select_after_upload': True,
            'last_uploaded_model': None,
            'last_model_scan_at': None,
            'last_reload_at': None
        },
        'about': {
            'show_version_banner': True,
            'show_system_links': True,
            'support_url': '',
            'documentation_url': '',
            'release_channel': 'stable',
            'license_notice': 'MIT'
        }
    }
    
    def __init__(self):
        self.config = self.load_config()
    
    def load_config(self):
        if os.path.exists(self.CONFIG_FILE):
            try:
                with open(self.CONFIG_FILE, 'r', encoding='utf-8') as f:
                    saved_config = json.load(f)
                merged = self._merge_configs(self.DEFAULT_CONFIG, saved_config)
                merged = self._migrate_config_for_0821(merged, saved_config)
                return self._normalize_analysis_area(merged)
            except Exception as e:
                logger.error(f"Fehler beim Laden der Konfiguration: {e}")
                return self._normalize_analysis_area(json.loads(json.dumps(self.DEFAULT_CONFIG)))
        return self._normalize_analysis_area(json.loads(json.dumps(self.DEFAULT_CONFIG)))
    
    def _migrate_config_for_0821(self, config, saved_config=None):
        """Apply safe defaults for the 0.8.23 OCR/vehicle pipeline.

        This keeps EasyOCR available, but makes fast-plate-ocr the default OCR
        engine for both RTSP and upload analysis. Existing configs from older
        versions may contain engine=easyocr because that was the only option.
        """
        try:
            ocr_cfg = config.setdefault('ocr', {})
            migration_cfg = config.setdefault('migration', {})
            old_engine = str(ocr_cfg.get('engine') or '').strip().lower()
            # Apply the new 0.8.23 default once for old installations. After the
            # flag is stored, a user can switch back to EasyOCR in the settings
            # and the next restart will keep that explicit choice.
            if not migration_cfg.get('fast_plate_ocr_default_applied'):
                if old_engine in ('', 'easyocr'):
                    ocr_cfg['engine'] = 'fast_plate_ocr'
                migration_cfg['fast_plate_ocr_default_applied'] = True
            ocr_cfg.setdefault('easyocr_backup_enabled', True)
            ocr_cfg.setdefault('fast_plate_model', 'cct-s-v2-global-model')
            ocr_cfg.setdefault('fast_plate_device', 'auto')
            ocr_cfg.setdefault('fast_plate_providers', '')
            ocr_cfg.setdefault('fast_plate_remove_pad_char', True)
            ocr_cfg.setdefault('fast_plate_return_confidence', True)

            models = config.setdefault('models', {})
            models.setdefault('license_plate_detector', 'models/license_plate_detector.pt')
            models.setdefault('vehicle_detector', 'models/yolov8n.pt')

            detection = config.setdefault('detection', {})
            classes = detection.get('vehicle_class_filter') or []
            if 'bicycle' not in classes:
                detection['vehicle_class_filter'] = list(classes) + ['bicycle']

            # 0.8.23b: make live/test upload use the same robust plate-detector
            # settings that worked in the demo UI. This is applied once to older
            # saved configs because otherwise old thresholds can override the new
            # DEFAULT_CONFIG after merge_config().
            if not migration_cfg.get('demo_yolo_plate_settings_applied_v2'):
                detection['scan_full_frame_when_vehicle_found'] = True
                detection['plate_scan_strategy'] = 'full_frame_first'
                detection['plate_detector_confidence'] = 0.25
                detection['plate_detector_iou'] = 0.45
                detection['plate_detector_imgsz'] = 960
                detection['plate_detector_max_det'] = max(8, int(detection.get('max_detections_per_frame') or 0))
                detection['plate_crop_padding_percent'] = 8.0
                if int(detection.get('min_plate_width') or 0) >= 60:
                    detection['min_plate_width'] = 20
                if int(detection.get('min_plate_height') or 0) >= 15:
                    detection['min_plate_height'] = 8
                if float(detection.get('plate_aspect_ratio_min') or 0) >= 2.0:
                    detection['plate_aspect_ratio_min'] = 1.2
                if float(detection.get('plate_aspect_ratio_max') or 0) <= 6.5:
                    detection['plate_aspect_ratio_max'] = 8.0
                migration_cfg['demo_yolo_plate_settings_applied_v2'] = True
            else:
                detection.setdefault('plate_detector_confidence', 0.25)
                detection.setdefault('plate_detector_iou', 0.45)
                detection.setdefault('plate_detector_imgsz', 960)
                detection.setdefault('plate_detector_max_det', 8)
                detection.setdefault('plate_crop_padding_percent', 8.0)
                detection.setdefault('plate_scan_strategy', 'full_frame_first')

            area = config.setdefault('rtsp', {}).setdefault('analysis_area', {})
            area.setdefault('crop_padding_percent', 25.0)
            area.setdefault('crop_min_padding_px', 120)
            area.setdefault('motion_gate_enabled', True)
            area.setdefault('motion_gate_threshold_percent', 0.20)
            area.setdefault('motion_gate_hold_seconds', 2.0)
            area.setdefault('motion_gate_idle_scan_seconds', 5.0)
            if not migration_cfg.get('rtsp_cpu_saver_area_applied_v1'):
                # 0.8.23 CPU-Fix: Nicht mehr permanent das komplette Bild durch
                # drei YOLO-Stufen schicken. Der verbindliche Straßenbereich wird
                # vor der RTSP-Erkennung als gepolsterter Crop genutzt. Das Demo-
                # Verhalten für Foto-Uploads bleibt unverändert.
                area['crop_before_detection'] = True
                area['mask_before_detection'] = False
                area['crop_padding_percent'] = float(area.get('crop_padding_percent') or 25.0)
                area['crop_min_padding_px'] = int(area.get('crop_min_padding_px') or 120)
                area['motion_gate_enabled'] = True
                area['motion_gate_threshold_percent'] = float(area.get('motion_gate_threshold_percent') or 0.20)
                area['motion_gate_hold_seconds'] = float(area.get('motion_gate_hold_seconds') or 2.0)
                area['motion_gate_idle_scan_seconds'] = float(area.get('motion_gate_idle_scan_seconds') or 5.0)
                detection_cfg = config.setdefault('detection', {})
                try:
                    if float(detection_cfg.get('process_interval') or 0.5) < 0.8:
                        detection_cfg['process_interval'] = 0.8
                except Exception:
                    detection_cfg['process_interval'] = 0.8
                migration_cfg['rtsp_cpu_saver_area_applied_v1'] = True
            else:
                area.setdefault('crop_before_detection', True)
                area.setdefault('mask_before_detection', False)

            # 0.8.23c: persons should be stored with images like vehicle/plate detections.
            # This is applied once to existing installations so the /people page gets
            # useful image history from Test & Upload and from the RTSP loop.
            people_cfg = config.setdefault('people', {})
            if not migration_cfg.get('people_images_default_applied_v1'):
                people_cfg['image_history_enabled'] = True
                people_cfg['save_person_crops'] = True
                people_cfg['image_history_store_crop'] = True
                people_cfg['image_history_store_annotated'] = False
                people_cfg['save_full_frame'] = False
                people_cfg['image_history_store_full_frame'] = False
                people_cfg['save_all_detections'] = True
                people_cfg.setdefault('enabled', True)
                migration_cfg['people_images_default_applied_v1'] = True
            if not migration_cfg.get('people_crop_only_display_applied_v1'):
                people_cfg['image_history_enabled'] = True
                people_cfg['save_person_crops'] = True
                people_cfg['image_history_store_crop'] = True
                people_cfg['image_history_store_annotated'] = False
                people_cfg['save_full_frame'] = False
                people_cfg['image_history_store_full_frame'] = False
                migration_cfg['people_crop_only_display_applied_v1'] = True
        except Exception as exc:
            logger.warning(f"0.8.23 Config-Migration konnte nicht vollständig angewendet werden: {exc}")
        return config

    def _normalize_analysis_area(self, config):
        """Keep one canonical RTSP analysis area: the road polygon.

        The ROI is stored in one coordinate system. Older versions only had
        rtsp.resolution, which could differ from the real RTSP frame. That made
        the same polygon look different in /live and /rtsp-settings. Newer
        configs therefore store coordinate_width / coordinate_height directly on
        the analysis_area. Processing scales from that saved coordinate system to
        the current camera frame when needed.
        """
        try:
            rtsp = config.setdefault('rtsp', {})
            resolution = rtsp.get('resolution') or {}
            area_cfg = rtsp.setdefault('analysis_area', {})

            def to_int(value, fallback):
                try:
                    return int(round(float(value)))
                except Exception:
                    return fallback

            frame_w = to_int(area_cfg.get('coordinate_width'), 0) or to_int(resolution.get('width'), 1280)
            frame_h = to_int(area_cfg.get('coordinate_height'), 0) or to_int(resolution.get('height'), 720)
            frame_w = max(1, frame_w)
            frame_h = max(1, frame_h)

            def clamp_int(value, minimum, maximum):
                try:
                    return max(minimum, min(int(round(float(value))), maximum))
                except Exception:
                    return minimum

            raw_polygon = area_cfg.get('polygon') or []
            polygon = []
            if isinstance(raw_polygon, list):
                for point in raw_polygon:
                    if isinstance(point, dict):
                        px = point.get('x', 0)
                        py = point.get('y', 0)
                    elif isinstance(point, (list, tuple)) and len(point) >= 2:
                        px, py = point[0], point[1]
                    else:
                        continue
                    polygon.append({
                        'x': clamp_int(px, 0, frame_w - 1),
                        'y': clamp_int(py, 0, frame_h - 1)
                    })

            if len(polygon) < 3:
                old_area = area_cfg.get('area') or {}
                try:
                    x = clamp_int(old_area.get('x', 0), 0, frame_w - 1)
                    y = clamp_int(old_area.get('y', 0), 0, frame_h - 1)
                    width = clamp_int(old_area.get('width', frame_w), 1, frame_w - x)
                    height = clamp_int(old_area.get('height', frame_h), 1, frame_h - y)
                    polygon = [
                        {'x': x, 'y': y},
                        {'x': min(frame_w - 1, x + width), 'y': y},
                        {'x': min(frame_w - 1, x + width), 'y': min(frame_h - 1, y + height)},
                        {'x': x, 'y': min(frame_h - 1, y + height)},
                    ]
                except Exception:
                    polygon = [
                        {'x': int(frame_w * 0.18), 'y': int(frame_h * 0.26)},
                        {'x': int(frame_w * 0.72), 'y': int(frame_h * 0.25)},
                        {'x': int(frame_w * 0.78), 'y': frame_h - 1},
                        {'x': int(frame_w * 0.10), 'y': frame_h - 1},
                    ]

            xs = [p['x'] for p in polygon]
            ys = [p['y'] for p in polygon]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            area_cfg['mode'] = 'polygon'
            area_cfg['mask_outside'] = True
            area_cfg['coordinate_width'] = frame_w
            area_cfg['coordinate_height'] = frame_h
            area_cfg['polygon'] = polygon
            area_cfg['area'] = {
                'x': int(min_x),
                'y': int(min_y),
                'width': int(max(1, max_x - min_x)),
                'height': int(max(1, max_y - min_y))
            }
        except Exception as e:
            logger.warning(f"Analysebereich konnte nicht normalisiert werden: {e}")
        return config

    def _merge_configs(self, default, saved):
        """Merge defaults with saved config without dropping older or custom keys.

        Earlier builds only returned keys known by DEFAULT_CONFIG. That could silently
        hide user-defined settings after an update. This version keeps every saved
        key and only fills missing defaults.
        """
        result = {}
        saved = saved or {}
        for key, value in default.items():
            if key in saved:
                if isinstance(value, dict) and isinstance(saved[key], dict):
                    result[key] = self._merge_configs(value, saved[key])
                else:
                    result[key] = saved[key]
            else:
                result[key] = value
        for key, value in saved.items():
            if key not in result:
                result[key] = value
        return result
    
    def save_config(self):
        try:
            self.config = self._normalize_analysis_area(self.config)
            with open(self.CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=4, ensure_ascii=False)
            return True
        except Exception as e:
            logger.error(f"Fehler beim Speichern der Konfiguration: {e}")
            return False
    
    def get(self, *keys):
        value = self.config
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
            else:
                return None
            if value is None:
                return None
        return value
    
    def set(self, value, *keys):
        config = self.config
        for key in keys[:-1]:
            config = config.setdefault(key, {})
        config[keys[-1]] = value
        self.save_config()


# ============================================================
# HISTORY MANAGER
# ============================================================

class HistoryManager:
    """Verwaltet die Erkennungshistorie"""
    
    HISTORY_FILE = 'data/history/detections.json'
    
    def __init__(self):
        self.history = self.load_history()
        self.lock = threading.Lock()
    
    def load_history(self):
        if os.path.exists(self.HISTORY_FILE):
            try:
                with open(self.HISTORY_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Fehler beim Laden der Historie: {e}")
        return []
    
    def save_history(self):
        try:
            with open(self.HISTORY_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.history, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            logger.error(f"Fehler beim Speichern der Historie: {e}")
            return False
    
    def _normalize_plate(self, plate_text):
        return PlateUtils.normalize(plate_text, compact=True)
    
    def _is_duplicate_in_history(self, plate_text, timeout_seconds=60):
        if not plate_text:
            return True

        normalized = self._normalize_plate(plate_text)
        if not normalized or len(normalized) < 3:
            return True

        current_time = datetime.now()
        fuzzy_enabled = config_manager.get('history', 'fuzzy_duplicate_detection')
        fuzzy_similarity = config_manager.get('history', 'fuzzy_duplicate_similarity') or 0.88

        for entry in self.history[:200]:
            entry_plate = self._normalize_plate(entry.get('plate_text', ''))
            if not entry_plate:
                continue
            try:
                entry_time = datetime.fromisoformat(entry.get('timestamp', ''))
                time_diff = (current_time - entry_time).total_seconds()
            except Exception:
                time_diff = timeout_seconds + 1
            if time_diff >= timeout_seconds:
                continue
            if entry_plate == normalized:
                return True
            if fuzzy_enabled and PlateUtils.similarity(entry_plate, normalized) >= fuzzy_similarity:
                return True

        return False

    def add_entry(self, entry, check_duplicate=True):
        with self.lock:
            if check_duplicate:
                timeout = config_manager.get('history', 'duplicate_timeout') or 60
                filter_enabled = config_manager.get('history', 'filter_duplicates')
                
                if filter_enabled and self._is_duplicate_in_history(entry.get('plate_text'), timeout):
                    logger.debug(f"Duplikat in Historie übersprungen: {entry.get('plate_text')}")
                    return None
            
            plate_text = entry.get('plate_text', '')
            normalized = PlateUtils.normalize(plate_text, compact=True)
            pretty = PlateUtils.pretty(plate_text)
            entry['plate_text_normalized'] = normalized
            entry['plate_format'] = PlateUtils.detect_format(plate_text)
            entry['is_valid_plate'] = PlateUtils.is_valid(
                plate_text,
                config_manager.get('plate_recognition', 'min_length') or 3,
                config_manager.get('plate_recognition', 'max_length') or 12,
                config_manager.get('plate_recognition', 'validation_regex') or None
            )
            if config_manager.get('plate_recognition', 'format_pretty_output') and pretty:
                entry['plate_text'] = pretty
            try:
                if config_manager.get('plate_recognition', 'watchlist_enabled'):
                    entry['watchlist_match'] = watchlist_manager.check(entry.get('plate_text', plate_text))
            except Exception:
                entry['watchlist_match'] = None
            entry['id'] = str(uuid.uuid4())
            entry['timestamp'] = datetime.now().isoformat()
            self.history.insert(0, entry)
            
            max_entries = config_manager.get('general', 'max_history_entries') or 1000
            if len(self.history) > max_entries:
                self.history = self.history[:max_entries]
            
            self.save_history()
            return entry
    
    def get_all(self, limit=100, offset=0, unique_only=False):
        if unique_only:
            seen = set()
            unique_entries = []
            
            for entry in self.history:
                normalized = self._normalize_plate(entry.get('plate_text', ''))
                if normalized and normalized not in seen:
                    seen.add(normalized)
                    unique_entries.append(entry)
            
            return unique_entries[offset:offset + limit]
        
        return self.history[offset:offset + limit]
    
    def get_by_id(self, entry_id):
        for entry in self.history:
            if entry.get('id') == entry_id:
                return entry
        return None
    
    def delete_entry(self, entry_id):
        with self.lock:
            self.history = [e for e in self.history if e.get('id') != entry_id]
            self.save_history()
    
    def clear_history(self):
        with self.lock:
            self.history = []
            self.save_history()
    
    def search(self, query):
        query = self._normalize_plate(query)
        return [e for e in self.history 
                if query in self._normalize_plate(e.get('plate_text', ''))]
    
    def _parse_datetime(self, value):
        if not value:
            return None
        try:
            return datetime.fromisoformat(value.replace('Z', '+00:00'))
        except Exception:
            try:
                return datetime.fromisoformat(value[:10])
            except Exception:
                return None

    def _entry_matches(self, entry, filters):
        query = (filters.get('q') or filters.get('search') or '').strip()
        normalized = self._normalize_plate(entry.get('plate_text', ''))
        raw = str(entry.get('plate_text', '')).upper()

        if query:
            q_norm = self._normalize_plate(query)
            if filters.get('regex'):
                if not config_manager.get('search', 'allow_regex_search'):
                    return False
                try:
                    if not re.search(query, raw, flags=re.IGNORECASE):
                        return False
                except re.error:
                    return False
            elif q_norm not in normalized:
                fuzzy_enabled = filters.get('fuzzy') or config_manager.get('search', 'enable_fuzzy_search')
                similarity = PlateUtils.similarity(query, normalized)
                threshold = float(filters.get('fuzzy_similarity') or config_manager.get('search', 'fuzzy_similarity') or 0.72)
                if not fuzzy_enabled or similarity < threshold:
                    return False

        for key in ('source', 'vehicle_type', 'vehicle_color', 'plate_format'):
            val = filters.get(key)
            if val and val != 'all' and str(entry.get(key, '')).lower() != str(val).lower():
                return False

        if filters.get('valid_only') and not entry.get('is_valid_plate', True):
            return False
        if filters.get('watchlist_only') and not entry.get('watchlist_match'):
            return False

        min_conf = filters.get('min_confidence')
        max_conf = filters.get('max_confidence')
        confidence = float(entry.get('confidence') or 0)
        if min_conf not in (None, '') and confidence < float(min_conf):
            return False
        if max_conf not in (None, '') and confidence > float(max_conf):
            return False

        ts = self._parse_datetime(entry.get('timestamp'))
        if filters.get('date_from'):
            start = self._parse_datetime(filters.get('date_from'))
            if ts and start and ts < start:
                return False
        if filters.get('date_to'):
            end = self._parse_datetime(filters.get('date_to'))
            if ts and end:
                # date-only values include the full day
                if len(str(filters.get('date_to'))) <= 10:
                    end = end.replace(hour=23, minute=59, second=59)
                if ts > end:
                    return False

        return True

    def search_advanced(self, filters=None):
        filters = filters or {}
        entries = [e for e in self.history if self._entry_matches(e, filters)]

        if filters.get('unique'):
            seen = set()
            unique_entries = []
            for entry in entries:
                normalized = self._normalize_plate(entry.get('plate_text', ''))
                if normalized and normalized not in seen:
                    seen.add(normalized)
                    unique_entries.append(entry)
            entries = unique_entries

        sort = filters.get('sort') or 'timestamp'
        reverse = (filters.get('order') or 'desc').lower() != 'asc'
        if sort == 'confidence':
            entries.sort(key=lambda e: float(e.get('confidence') or 0), reverse=reverse)
        elif sort == 'plate':
            entries.sort(key=lambda e: self._normalize_plate(e.get('plate_text', '')), reverse=reverse)
        else:
            entries.sort(key=lambda e: e.get('timestamp', ''), reverse=reverse)

        total = len(entries)
        limit = int(filters.get('limit') or config_manager.get('search', 'default_limit') or 50)
        offset = int(filters.get('offset') or 0)
        return {'entries': entries[offset:offset + limit], 'total': total, 'limit': limit, 'offset': offset}

    def get_facets(self):
        def counts(key):
            c = Counter(str(e.get(key, 'Unbekannt') or 'Unbekannt') for e in self.history)
            return [{'value': k, 'count': v} for k, v in c.most_common()]
        return {
            'sources': counts('source'),
            'vehicle_types': counts('vehicle_type'),
            'vehicle_colors': counts('vehicle_color'),
            'plate_formats': counts('plate_format')
        }


    def _safe_float(self, value, default=0.0):
        try:
            return float(value)
        except Exception:
            return default

    def _safe_int(self, value, default=0):
        try:
            return int(value)
        except Exception:
            return default

    def _entry_timestamp(self, entry):
        return self._parse_datetime(entry.get('timestamp'))

    def _entry_confidence(self, entry):
        return self._safe_float(entry.get('confidence'), 0.0)

    def _entry_center(self, entry, axis='x'):
        """Returns the best available center coordinate for movement heuristics."""
        axis = (axis or 'x').lower()
        direct_keys = ['vehicle_center_x', 'plate_center_x', 'center_x'] if axis == 'x' else ['vehicle_center_y', 'plate_center_y', 'center_y']
        for key in direct_keys:
            if entry.get(key) is not None:
                return self._safe_float(entry.get(key), None)
        bbox_keys = ['vehicle_bbox', 'plate_bbox', 'bbox']
        for key in bbox_keys:
            bbox = entry.get(key)
            if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                try:
                    x1, y1, x2, y2 = [float(v) for v in bbox]
                    return (x1 + x2) / 2.0 if axis == 'x' else (y1 + y2) / 2.0
                except Exception:
                    continue
        return None

    def _direction_from_entry(self, entry):
        raw = str(entry.get('direction') or entry.get('movement') or entry.get('event_type') or '').strip().lower()
        if not raw:
            return None
        if raw in ('in', 'enter', 'entered', 'arrival', 'arrived', 'kommen', 'gekommen', 'rein', 'einfahrt'):
            return 'arrival'
        if raw in ('out', 'exit', 'exited', 'departure', 'departed', 'gehen', 'gegangen', 'raus', 'ausfahrt'):
            return 'departure'
        if 'ein' in raw or 'komm' in raw or 'arrival' in raw or 'enter' in raw:
            return 'arrival'
        if 'aus' in raw or 'geh' in raw or 'depart' in raw or 'exit' in raw:
            return 'departure'
        return None

    def _traffic_filters_from_request(self, filters=None):
        filters = filters or {}
        cfg = config_manager.get('traffic') or {}
        now = datetime.now()
        days = self._safe_int(filters.get('days'), self._safe_int(config_manager.get('dashboard', 'default_range_days'), 7) or 7)
        explicit_to = filters.get('date_to')
        explicit_from = filters.get('date_from')
        date_to = self._parse_datetime(explicit_to) if explicit_to else now
        if date_to and explicit_to and len(str(explicit_to)) <= 10:
            date_to = date_to.replace(hour=23, minute=59, second=59)
        if explicit_from:
            date_from = self._parse_datetime(explicit_from)
        else:
            base = date_to or now
            # Default ranges are calendar-day based, not last N*24 hours from the current minute.
            date_from = base.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=max(days - 1, 0))
        return {
            'date_from': date_from,
            'date_to': date_to,
            'min_confidence': self._safe_float(filters.get('min_confidence'), self._safe_float(cfg.get('min_confidence'), 0.0)),
            'ignore_unknown_plates': str(filters.get('ignore_unknown_plates', cfg.get('ignore_unknown_plates', True))).lower() not in ('0', 'false', 'no', 'nein'),
            'include_duplicate_events': str(filters.get('include_duplicate_events', cfg.get('include_duplicate_events', False))).lower() in ('1', 'true', 'yes', 'ja'),
            'visit_gap_minutes': self._safe_int(filters.get('visit_gap_minutes'), self._safe_int(cfg.get('visit_gap_minutes'), 15) or 15),
            'active_timeout_minutes': self._safe_int(filters.get('active_timeout_minutes'), self._safe_int(cfg.get('active_timeout_minutes'), 30) or 30),
            'daily_count_mode': filters.get('daily_count_mode') or cfg.get('daily_count_mode') or 'visits',
            'movement_axis': filters.get('movement_axis') or cfg.get('movement_axis') or 'x',
            'movement_threshold_percent': self._safe_float(filters.get('movement_threshold_percent'), self._safe_float(cfg.get('movement_threshold_percent'), 8.0)),
        }

    def _traffic_entries(self, filters=None):
        f = self._traffic_filters_from_request(filters)
        entries = []
        seen_event_ids = set()
        for entry in self.history:
            ts = self._entry_timestamp(entry)
            if not ts:
                continue
            ts_naive = ts.replace(tzinfo=None) if getattr(ts, 'tzinfo', None) else ts
            if f['date_from'] and ts_naive < f['date_from'].replace(tzinfo=None):
                continue
            if f['date_to'] and ts_naive > f['date_to'].replace(tzinfo=None):
                continue
            if self._entry_confidence(entry) < f['min_confidence']:
                continue
            plate = self._normalize_plate(entry.get('plate_text', ''))
            if f['ignore_unknown_plates'] and (not plate or plate in ('UNKNOWN', 'UNBEKANNT')):
                continue
            if not f['include_duplicate_events']:
                event_key = entry.get('id') or f"{plate}:{entry.get('timestamp')}"
                if event_key in seen_event_ids:
                    continue
                seen_event_ids.add(event_key)
            enriched = dict(entry)
            enriched['_ts'] = ts_naive
            enriched['_plate_norm'] = plate or 'UNKNOWN'
            enriched['_center_x'] = self._entry_center(entry, 'x')
            enriched['_center_y'] = self._entry_center(entry, 'y')
            enriched['_direction'] = self._direction_from_entry(entry)
            entries.append(enriched)
        entries.sort(key=lambda e: e['_ts'])
        return entries, f

    def _build_traffic_sessions(self, filters=None):
        entries, f = self._traffic_entries(filters)
        by_plate = defaultdict(list)
        for entry in entries:
            by_plate[entry['_plate_norm']].append(entry)

        sessions = []
        gap = timedelta(minutes=max(f['visit_gap_minutes'], 1))
        active_timeout = timedelta(minutes=max(f['active_timeout_minutes'], 1))
        now = datetime.now()
        for plate, plate_entries in by_plate.items():
            current = []
            for entry in plate_entries:
                if not current or (entry['_ts'] - current[-1]['_ts']) <= gap:
                    current.append(entry)
                else:
                    sessions.append(self._session_summary(plate, current, now, active_timeout, f))
                    current = [entry]
            if current:
                sessions.append(self._session_summary(plate, current, now, active_timeout, f))
        sessions.sort(key=lambda s: s['start_time'], reverse=True)
        return sessions, entries, f

    def _session_summary(self, plate, entries, now, active_timeout, filters):
        first = entries[0]
        last = entries[-1]
        start = first['_ts']
        end = last['_ts']
        duration_seconds = max(0, int((end - start).total_seconds()))
        directions = [e.get('_direction') for e in entries if e.get('_direction')]
        direction_quality = 'explicit' if directions else 'heuristic'
        arrival_detected = 'arrival' in directions
        departure_detected = 'departure' in directions

        movement = None
        movement_label = 'unbekannt'
        movement_delta = None
        axis = filters.get('movement_axis') or 'x'
        centers = [self._entry_center(e, axis) for e in entries]
        centers = [c for c in centers if c is not None]
        if len(centers) >= 2:
            movement_delta = centers[-1] - centers[0]
            frame_size = None
            size_keys = ['frame_width', 'image_width', 'source_width'] if axis == 'x' else ['frame_height', 'image_height', 'source_height']
            for key in size_keys:
                for e in entries:
                    if e.get(key):
                        frame_size = self._safe_float(e.get(key), None)
                        break
                if frame_size:
                    break
            threshold = self._safe_float(filters.get('movement_threshold_percent'), 8.0)
            min_delta = (frame_size * threshold / 100.0) if frame_size else max(40.0, abs(centers[0]) * threshold / 100.0)
            if abs(movement_delta) >= min_delta:
                movement = 'positive' if movement_delta > 0 else 'negative'
                movement_label = 'rechts/abwärts' if movement == 'positive' else 'links/aufwärts'

        status = 'present_recently' if (now - end) <= active_timeout else 'departed_assumed'
        if departure_detected:
            status = 'departed_detected'
        elif arrival_detected and len(entries) == 1 and (now - end) <= active_timeout:
            status = 'arrived_detected'

        avg_confidence = sum(self._entry_confidence(e) for e in entries) / len(entries) if entries else 0
        return {
            'plate_text': last.get('plate_text') or first.get('plate_text') or PlateUtils.pretty(plate),
            'plate_text_normalized': plate,
            'start_time': start.isoformat(),
            'end_time': end.isoformat(),
            'date': start.date().isoformat(),
            'duration_seconds': duration_seconds,
            'duration_label': self._format_duration(duration_seconds),
            'detections': len(entries),
            'average_confidence': round(avg_confidence, 4),
            'first_entry_id': first.get('id'),
            'last_entry_id': last.get('id'),
            'vehicle_type': last.get('vehicle_type') or first.get('vehicle_type') or 'Unbekannt',
            'vehicle_color': last.get('vehicle_color') or first.get('vehicle_color') or 'Unbekannt',
            'source': last.get('source') or first.get('source') or 'unknown',
            'status': status,
            'arrival_detected': bool(arrival_detected or len(entries) >= 1),
            'departure_detected': bool(departure_detected),
            'departure_assumed': status == 'departed_assumed',
            'direction_quality': direction_quality,
            'movement': movement,
            'movement_label': movement_label,
            'movement_delta': round(movement_delta, 2) if movement_delta is not None else None,
            'has_images': bool(last.get('plate_image') or last.get('vehicle_image') or last.get('full_frame')),
            'watchlist_match': last.get('watchlist_match') or first.get('watchlist_match')
        }

    def _format_duration(self, seconds):
        seconds = int(seconds or 0)
        if seconds < 60:
            return f"{seconds}s"
        minutes = seconds // 60
        if minutes < 60:
            return f"{minutes}m {seconds % 60}s"
        hours = minutes // 60
        return f"{hours}h {minutes % 60}m"

    def get_traffic_statistics(self, filters=None):
        sessions, entries, f = self._build_traffic_sessions(filters)
        daily = defaultdict(lambda: {
            'date': '', 'detections': 0, 'visits': 0, 'unique_vehicles': set(),
            'arrivals': 0, 'departures_detected': 0, 'departures_assumed': 0,
            'present_recently': 0, 'repeat_visits': 0
        })
        hourly = defaultdict(int)
        plate_stats = {}

        for entry in entries:
            day = entry['_ts'].date().isoformat()
            daily[day]['date'] = day
            daily[day]['detections'] += 1
            daily[day]['unique_vehicles'].add(entry['_plate_norm'])
            hourly[entry['_ts'].strftime('%Y-%m-%d %H:00')] += 1

        plate_day_visits = defaultdict(lambda: defaultdict(int))
        for session in sessions:
            day = session['date']
            daily[day]['date'] = day
            daily[day]['visits'] += 1
            daily[day]['unique_vehicles'].add(session['plate_text_normalized'])
            daily[day]['arrivals'] += 1 if session.get('arrival_detected') else 0
            daily[day]['departures_detected'] += 1 if session.get('departure_detected') else 0
            daily[day]['departures_assumed'] += 1 if session.get('departure_assumed') else 0
            daily[day]['present_recently'] += 1 if session.get('status') in ('present_recently', 'arrived_detected') else 0
            plate_day_visits[day][session['plate_text_normalized']] += 1

            p = plate_stats.setdefault(session['plate_text_normalized'], {
                'plate_text': session['plate_text'], 'normalized': session['plate_text_normalized'],
                'detections': 0, 'visits': 0, 'days_seen': set(), 'first_seen': session['start_time'],
                'last_seen': session['end_time'], 'average_confidence_sum': 0.0,
                'last_status': session['status'], 'vehicle_type': session['vehicle_type'],
                'vehicle_color': session['vehicle_color'], 'watchlist_match': session.get('watchlist_match')
            })
            p['detections'] += session['detections']
            p['visits'] += 1
            p['days_seen'].add(day)
            p['first_seen'] = min(p['first_seen'], session['start_time'])
            p['last_seen'] = max(p['last_seen'], session['end_time'])
            p['average_confidence_sum'] += session['average_confidence']
            p['last_status'] = session['status']
            p['vehicle_type'] = session['vehicle_type'] or p['vehicle_type']
            p['vehicle_color'] = session['vehicle_color'] or p['vehicle_color']

        daily_items = []
        for day, row in sorted(daily.items()):
            repeats = sum(1 for _, count in plate_day_visits[day].items() if count > 1)
            daily_items.append({
                'date': day,
                'detections': row['detections'],
                'visits': row['visits'],
                'unique_vehicles': len(row['unique_vehicles']),
                'arrivals': row['arrivals'],
                'departures_detected': row['departures_detected'],
                'departures_assumed': row['departures_assumed'],
                'present_recently': row['present_recently'],
                'repeat_vehicles': repeats
            })

        plates = []
        for p in plate_stats.values():
            visits = max(p['visits'], 1)
            plates.append({
                'plate_text': p['plate_text'],
                'normalized': p['normalized'],
                'detections': p['detections'],
                'visits': p['visits'],
                'days_seen': len(p['days_seen']),
                'first_seen': p['first_seen'],
                'last_seen': p['last_seen'],
                'average_confidence': round(p['average_confidence_sum'] / visits, 4),
                'last_status': p['last_status'],
                'vehicle_type': p['vehicle_type'],
                'vehicle_color': p['vehicle_color'],
                'watchlist_match': p.get('watchlist_match'),
                'repeat_vehicle': p['visits'] > 1 or p['detections'] > 1
            })
        plates.sort(key=lambda p: (p['visits'], p['detections'], p['last_seen']), reverse=True)
        repeat_vehicles = [p for p in plates if p['repeat_vehicle']]
        current_present = [s for s in sessions if s['status'] in ('present_recently', 'arrived_detected')]

        total_visits = len(sessions)
        total_detections = len(entries)
        unique_vehicles = len(plate_stats)
        busiest_day = max(daily_items, key=lambda d: d['visits'], default=None)
        busiest_hour = max([{'hour': k, 'count': v} for k, v in hourly.items()], key=lambda h: h['count'], default=None)

        return {
            'filters': {
                'date_from': f['date_from'].isoformat() if f['date_from'] else None,
                'date_to': f['date_to'].isoformat() if f['date_to'] else None,
                'visit_gap_minutes': f['visit_gap_minutes'],
                'active_timeout_minutes': f['active_timeout_minutes'],
                'daily_count_mode': f['daily_count_mode'],
                'min_confidence': f['min_confidence']
            },
            'summary': {
                'total_detections': total_detections,
                'total_visits': total_visits,
                'unique_vehicles': unique_vehicles,
                'repeat_vehicles': len(repeat_vehicles),
                'currently_present': len(current_present),
                'departures_detected': sum(1 for s in sessions if s.get('departure_detected')),
                'departures_assumed': sum(1 for s in sessions if s.get('departure_assumed')),
                'average_detections_per_visit': round(total_detections / total_visits, 2) if total_visits else 0,
                'busiest_day': busiest_day,
                'busiest_hour': busiest_hour,
                'note': 'Kommen/Gehen wird sicher erkannt, wenn explizite Richtungsdaten vorhanden sind. Ohne Richtungsdaten wird Gehen nach Timeout angenommen.'
            },
            'daily': daily_items,
            'hourly': [{'hour': k, 'count': v} for k, v in sorted(hourly.items())],
            'top_plates': plates[:50],
            'repeat_vehicles': repeat_vehicles[:50],
            'sessions': sessions[:500],
            'currently_present': current_present[:100]
        }

    def get_plate_profile(self, plate_text, filters=None):
        norm = self._normalize_plate(plate_text)
        sessions, entries, f = self._build_traffic_sessions(filters)
        plate_sessions = [s for s in sessions if s['plate_text_normalized'] == norm]
        plate_entries = [e for e in entries if e['_plate_norm'] == norm]
        days = sorted({s['date'] for s in plate_sessions})
        return {
            'plate_text': PlateUtils.pretty(norm),
            'normalized': norm,
            'detections': len(plate_entries),
            'visits': len(plate_sessions),
            'days_seen': len(days),
            'days': days,
            'first_seen': min([s['start_time'] for s in plate_sessions], default=None),
            'last_seen': max([s['end_time'] for s in plate_sessions], default=None),
            'sessions': plate_sessions,
            'entries': [{k: v for k, v in e.items() if not k.startswith('_')} for e in plate_entries[-100:]]
        }

    def get_statistics(self):
        total = len(self.history)
        today = datetime.now().date().isoformat()
        now = datetime.now()
        today_count = sum(1 for e in self.history if e.get('timestamp', '').startswith(today))
        last_hour_count = 0
        unique_plates = set()
        vehicle_types = Counter()
        vehicle_colors = Counter()
        sources = Counter()
        plate_counts = Counter()
        confidence_values = []
        hourly = defaultdict(int)
        daily = defaultdict(int)
        confidence_buckets = {'0-40': 0, '40-60': 0, '60-80': 0, '80-100': 0}

        for e in self.history:
            normalized = self._normalize_plate(e.get('plate_text', ''))
            if normalized:
                unique_plates.add(normalized)
                plate_counts[e.get('plate_text', normalized)] += 1

            vehicle_types[e.get('vehicle_type', 'Unbekannt') or 'Unbekannt'] += 1
            vehicle_colors[e.get('vehicle_color', 'Unbekannt') or 'Unbekannt'] += 1
            sources[e.get('source', 'unknown') or 'unknown'] += 1

            confidence = float(e.get('confidence') or 0)
            confidence_values.append(confidence)
            if confidence < 0.4:
                confidence_buckets['0-40'] += 1
            elif confidence < 0.6:
                confidence_buckets['40-60'] += 1
            elif confidence < 0.8:
                confidence_buckets['60-80'] += 1
            else:
                confidence_buckets['80-100'] += 1

            ts = self._parse_datetime(e.get('timestamp'))
            if ts:
                daily[ts.date().isoformat()] += 1
                hourly[ts.strftime('%H:00')] += 1
                if (now - ts.replace(tzinfo=None)).total_seconds() <= 3600:
                    last_hour_count += 1

        top_plates = plate_counts.most_common(10)
        avg_conf = sum(confidence_values) / len(confidence_values) if confidence_values else 0
        last_detection = self.history[0] if self.history else None

        traffic_preview = self.get_traffic_statistics({'days': 1})
        return {
            'total_detections': total,
            'today_detections': today_count,
            'last_hour_detections': last_hour_count,
            'unique_plates': len(unique_plates),
            'average_confidence': avg_conf,
            'vehicle_types': dict(vehicle_types),
            'vehicle_colors': dict(vehicle_colors),
            'sources': dict(sources),
            'top_plates': top_plates,
            'hourly': dict(sorted(hourly.items())),
            'daily': dict(sorted(daily.items())[-30:]),
            'confidence_buckets': confidence_buckets,
            'last_detection': last_detection,
            'valid_plates': sum(1 for e in self.history if e.get('is_valid_plate', True)),
            'watchlist_hits': sum(1 for e in self.history if e.get('watchlist_match')),
            'traffic_today': traffic_preview.get('summary', {})
        }


# ============================================================
# WATCHLIST MANAGER
# ============================================================

class WatchlistManager:
    """Bekannte, erlaubte oder gesuchte Kennzeichen verwalten."""

    WATCHLIST_FILE = 'data/watchlist.json'

    def __init__(self):
        self.items = self.load()
        self.lock = threading.Lock()

    def load(self):
        if os.path.exists(self.WATCHLIST_FILE):
            try:
                with open(self.WATCHLIST_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Fehler beim Laden der Watchlist: {e}")
        return []

    def save(self):
        try:
            with open(self.WATCHLIST_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.items, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            logger.error(f"Fehler beim Speichern der Watchlist: {e}")
            return False

    def list(self):
        return self.items

    def add(self, plate_text, label='', category='known', notes='', notify=True):
        normalized = PlateUtils.normalize(plate_text, compact=True)
        if not normalized:
            raise ValueError('Leeres Kennzeichen')
        with self.lock:
            for item in self.items:
                if item.get('normalized') == normalized:
                    item.update({'plate_text': PlateUtils.pretty(plate_text), 'label': label or item.get('label', ''), 'category': category or item.get('category', 'known'), 'notes': notes or item.get('notes', ''), 'notify': bool(notify)})
                    self.save()
                    return item
            item = {
                'id': str(uuid.uuid4()),
                'plate_text': PlateUtils.pretty(plate_text),
                'normalized': normalized,
                'label': label,
                'category': category,
                'notes': notes,
                'notify': bool(notify),
                'created_at': datetime.now().isoformat()
            }
            self.items.insert(0, item)
            self.save()
            return item

    def delete(self, item_id):
        with self.lock:
            before = len(self.items)
            self.items = [i for i in self.items if i.get('id') != item_id and i.get('normalized') != PlateUtils.normalize(item_id, compact=True)]
            self.save()
            return len(self.items) != before

    def check(self, plate_text):
        normalized = PlateUtils.normalize(plate_text, compact=True)
        if not normalized:
            return None
        for item in self.items:
            if item.get('normalized') == normalized:
                return item
        fuzzy_threshold = config_manager.get('history', 'fuzzy_duplicate_similarity') or 0.88
        for item in self.items:
            if PlateUtils.similarity(item.get('normalized'), normalized) >= fuzzy_threshold:
                match = item.copy()
                match['fuzzy_match'] = True
                match['similarity'] = PlateUtils.similarity(item.get('normalized'), normalized)
                return match
        return None


# ============================================================
# PERSONEN-HISTORY & PERSONENSTATISTIK
# ============================================================

class PersonHistoryManager:
    """Speichert Personen-Zählereignisse und erstellt Tages-/Stundenstatistiken."""

    HISTORY_FILE = 'data/people/history.json'

    def __init__(self):
        self.history = self.load_history()
        self.lock = threading.RLock()
        self._last_auto_cleanup_ts = 0

    def load_history(self):
        if os.path.exists(self.HISTORY_FILE):
            try:
                with open(self.HISTORY_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Fehler beim Laden der Personen-Historie: {e}")
        return []

    def save_history(self):
        try:
            Path(self.HISTORY_FILE).parent.mkdir(parents=True, exist_ok=True)
            with open(self.HISTORY_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.history, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            logger.error(f"Fehler beim Speichern der Personen-Historie: {e}")
            return False

    def _parse_datetime(self, value):
        if not value:
            return None
        try:
            return datetime.fromisoformat(str(value).replace('Z', '+00:00'))
        except Exception:
            return None

    IMAGE_ROOT = Path('data/people/images')

    def _cfg(self):
        try:
            return config_manager.get('people') or {}
        except Exception:
            return {}

    def _event_dt(self, item):
        return self._parse_datetime((item or {}).get('timestamp'))

    def _position_signature(self, item, tolerance_percent=None):
        """Build a privacy-friendly signature from track/location without face recognition."""
        cfg = self._cfg()
        tol = float(tolerance_percent or cfg.get('person_recount_position_tolerance_percent') or 12)
        tol = max(1.0, min(50.0, tol))
        fw = float(item.get('frame_width') or 100)
        fh = float(item.get('frame_height') or 100)
        cx = float(item.get('center_x') or 0)
        cy = float(item.get('center_y') or 0)
        bbox = item.get('bbox') or []
        if len(bbox) == 4:
            area = max(0, float(bbox[2]) - float(bbox[0])) * max(0, float(bbox[3]) - float(bbox[1]))
        else:
            area = float(item.get('area_percent') or 0) * fw * fh / 100.0
        bx = round((cx / max(1.0, fw) * 100.0) / tol)
        by = round((cy / max(1.0, fh) * 100.0) / tol)
        ba = round(((area / max(1.0, fw * fh)) * 100.0) / max(1.0, tol / 2.0))
        source = item.get('source') or 'unknown'
        return f"pos:{source}:{bx}:{by}:{ba}"

    def _identity_keys(self, item):
        cfg = self._cfg()
        mode = cfg.get('person_recount_identity_mode') or 'track_or_position'
        source = item.get('source') or 'unknown'
        keys = []
        track_id = item.get('track_id')
        if track_id is not None and str(track_id) != '' and mode in ('track', 'track_only', 'track_or_position'):
            keys.append(f"track:{source}:{track_id}")
        if mode in ('position', 'position_only', 'track_or_position'):
            keys.append(self._position_signature(item))
        return keys

    def _recent_counted_match(self, item):
        cfg = self._cfg()
        if not cfg.get('person_recount_block_enabled', True):
            return None
        if not item.get('counted', True):
            return None
        minutes = float(cfg.get('person_recount_block_minutes') or 0)
        if minutes <= 0:
            return None
        ts = self._event_dt(item) or datetime.now()
        ts = ts.replace(tzinfo=None) if getattr(ts, 'tzinfo', None) else ts
        cutoff = ts - timedelta(minutes=minutes)
        keys = set(self._identity_keys(item))
        if not keys:
            return None
        for old in self.history:
            if not old.get('counted'):
                continue
            old_ts = self._event_dt(old)
            if not old_ts:
                continue
            old_ts = old_ts.replace(tzinfo=None) if getattr(old_ts, 'tzinfo', None) else old_ts
            if old_ts < cutoff:
                # history is newest first; older entries can be skipped.
                continue
            if keys.intersection(self._identity_keys(old)):
                return old
        return None

    def _prepare_image(self, image, item=None, blur_people=False):
        if image is None:
            return None
        out = image.copy()
        if blur_people and item:
            try:
                bbox = item.get('bbox') or []
                if len(bbox) == 4:
                    x1, y1, x2, y2 = [int(float(v)) for v in bbox]
                    h, w = out.shape[:2]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    if x2 > x1 and y2 > y1:
                        k = int(self._cfg().get('blur_strength') or 35)
                        k = k if k % 2 == 1 else k + 1
                        out[y1:y2, x1:x2] = cv2.GaussianBlur(out[y1:y2, x1:x2], (k, k), 0)
            except Exception as exc:
                logger.warning(f"Personen-Weichzeichnung fehlgeschlagen: {exc}")
        return out

    def _save_jpeg(self, image, relative_path, quality=85):
        if image is None:
            return None
        target = self.IMAGE_ROOT / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        ok, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
        if not ok:
            return None
        target.write_bytes(buffer.tobytes())
        return str(relative_path).replace('\\\\', '/')

    def attach_images(self, item, frame=None, annotated_frame=None):
        cfg = self._cfg()
        if not cfg.get('image_history_enabled', True):
            return {}
        if frame is None and annotated_frame is None:
            return {}
        quality = int(cfg.get('image_history_jpeg_quality') or 85)
        quality = max(35, min(100, quality))
        date_part = str(item.get('timestamp') or datetime.now().isoformat())[:10]
        safe_id = str(item.get('id') or uuid.uuid4()).replace('/', '_')
        images = dict(item.get('images') or {})
        blur = bool(cfg.get('privacy_blur_people'))
        try:
            if cfg.get('save_person_crops') or cfg.get('image_history_store_crop'):
                bbox = item.get('bbox') or []
                if frame is not None and len(bbox) == 4:
                    h, w = frame.shape[:2]
                    x1, y1, x2, y2 = [int(float(v)) for v in bbox]
                    pad = int(cfg.get('image_history_crop_padding_px') or 8)
                    x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
                    x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
                    if x2 > x1 and y2 > y1:
                        crop = frame[y1:y2, x1:x2].copy()
                        if blur:
                            k = int(cfg.get('blur_strength') or 35)
                            k = k if k % 2 == 1 else k + 1
                            crop = cv2.GaussianBlur(crop, (k, k), 0)
                        rel = Path('crops') / date_part / f'{safe_id}.jpg'
                        saved = self._save_jpeg(crop, rel, quality)
                        if saved:
                            images['crop'] = saved
            if cfg.get('save_full_frame') or cfg.get('image_history_store_full_frame'):
                img = self._prepare_image(frame, item, blur) if frame is not None else None
                saved = self._save_jpeg(img, Path('full_frames') / date_part / f'{safe_id}.jpg', quality)
                if saved:
                    images['full_frame'] = saved
            if cfg.get('image_history_store_annotated', True):
                img = self._prepare_image(annotated_frame, item, blur) if annotated_frame is not None else None
                saved = self._save_jpeg(img, Path('annotated') / date_part / f'{safe_id}.jpg', quality)
                if saved:
                    images['annotated'] = saved
        except Exception as exc:
            logger.warning(f"Personenbild konnte nicht gespeichert werden: {exc}")
        if images:
            item['images'] = images
        return images

    def _delete_images_for_item(self, item):
        images = (item or {}).get('images') or {}
        deleted = 0
        for rel in list(images.values()):
            try:
                target = (self.IMAGE_ROOT / str(rel)).resolve()
                root = self.IMAGE_ROOT.resolve()
                if str(target).startswith(str(root)) and target.exists() and target.is_file():
                    target.unlink()
                    deleted += 1
            except Exception as exc:
                logger.warning(f"Personenbild konnte nicht gelöscht werden: {rel} - {exc}")
        return deleted

    def _maybe_auto_cleanup_images(self):
        cfg = self._cfg()
        if not cfg.get('image_history_auto_cleanup_enabled') and not cfg.get('auto_cleanup_enabled'):
            return
        if not cfg.get('image_history_cleanup_on_add', True):
            return
        now = time.time()
        if now - float(getattr(self, '_last_auto_cleanup_ts', 0) or 0) < 3600:
            return
        self._last_auto_cleanup_ts = now
        try:
            self.cleanup_images(cfg.get('image_history_retention_days') or cfg.get('retention_days'))
        except Exception as exc:
            logger.warning(f"Auto-Cleanup Personenbilder fehlgeschlagen: {exc}")

    def add_event(self, event, check_duplicate=True, frame=None, annotated_frame=None):
        if not (config_manager.get('people', 'history_enabled') is not False):
            return None
        with self.lock:
            item = dict(event or {})
            item.setdefault('id', str(uuid.uuid4()))
            item.setdefault('timestamp', datetime.now().isoformat())
            item.setdefault('source', 'unknown')
            item.setdefault('event_type', 'person_detected')
            item.setdefault('direction', 'unknown')
            item.setdefault('counted', True)
            item.setdefault('confidence', 0)
            item.setdefault('track_id', None)
            if item.get('counted'):
                match = self._recent_counted_match(item)
                if match:
                    item['counted_original'] = True
                    item['counted'] = False
                    item['repeat_blocked'] = True
                    item['repeat_block_minutes'] = int(float(self._cfg().get('person_recount_block_minutes') or 15))
                    item['repeat_match_id'] = match.get('id')
                    item['event_type'] = 'repeat_blocked'
                    item['note'] = f"Nicht erneut gezählt: ähnliche Person/Track innerhalb von {item['repeat_block_minutes']} Minuten."
            self.attach_images(item, frame=frame, annotated_frame=annotated_frame)
            self.history.insert(0, item)
            max_entries = int(config_manager.get('general', 'max_history_entries') or 1000)
            if len(self.history) > max_entries:
                removed = self.history[max_entries:]
                for old in removed:
                    self._delete_images_for_item(old)
                self.history = self.history[:max_entries]
            self.save_history()
            self._maybe_auto_cleanup_images()
            return item

    def get_all(self, limit=100, offset=0):
        return self.history[offset:offset + limit]

    def clear_history(self, delete_images=True):
        with self.lock:
            deleted_images = 0
            if delete_images:
                for item in self.history:
                    deleted_images += self._delete_images_for_item(item)
            self.history = []
            self.save_history()
            return {'deleted_images': deleted_images}

    def purge_simulation_events(self, delete_images=True):
        """Remove old demo/simulation people entries from earlier builds."""
        with self.lock:
            deleted_images = 0
            kept = []
            removed = 0
            for item in self.history:
                is_sim = (item or {}).get('event_type') == 'simulation' or (item or {}).get('source') == 'test_environment' or str((item or {}).get('track_id') or '').startswith('sim-')
                if is_sim:
                    removed += 1
                    if delete_images:
                        deleted_images += self._delete_images_for_item(item)
                else:
                    kept.append(item)
            if removed:
                self.history = kept
                self.save_history()
                logger.info(f"Personenanalyse Demo-Daten entfernt: entries={removed}, images={deleted_images}")
            return {'removed': removed, 'deleted_images': deleted_images}

    def delete_event(self, event_id, delete_images=True):
        with self.lock:
            kept = []
            deleted = None
            deleted_images = 0
            for item in self.history:
                if str(item.get('id')) == str(event_id):
                    deleted = item
                    if delete_images:
                        deleted_images += self._delete_images_for_item(item)
                else:
                    kept.append(item)
            if deleted is None:
                return {'success': False, 'deleted': 0, 'deleted_images': 0}
            self.history = kept
            self.save_history()
            return {'success': True, 'deleted': 1, 'deleted_images': deleted_images, 'item': deleted}

    def image_history(self, filters=None):
        filters = filters or {}
        limit = int(filters.get('limit') or 60)
        offset = int(filters.get('offset') or 0)
        rows = []
        for item in self.search(filters):
            images = item.get('images') or {}
            if not images:
                continue
            row = dict(item)
            row['image_urls'] = {k: f"/api/people/images/{v}" for k, v in images.items()}
            rows.append(row)
        return {'entries': rows[offset:offset + limit], 'total': len(rows), 'limit': limit, 'offset': offset}

    def cleanup_images(self, retention_days=None, delete_orphan_files=True, delete_records=False):
        """Delete old person image files. By default keep the statistical person events."""
        cfg = self._cfg()
        days = int(retention_days or cfg.get('image_history_retention_days') or cfg.get('retention_days') or 0)
        if days <= 0:
            return {'cleared_image_events': 0, 'removed_events': 0, 'deleted_images': 0, 'remaining': len(self.history), 'retention_days': days}
        cutoff = datetime.now() - timedelta(days=days)
        deleted_images = 0
        removed_events = 0
        cleared_image_events = 0
        kept = []
        with self.lock:
            for item in self.history:
                ts = self._event_dt(item)
                ts = ts.replace(tzinfo=None) if ts and getattr(ts, 'tzinfo', None) else ts
                if ts and ts < cutoff and item.get('images'):
                    deleted_images += self._delete_images_for_item(item)
                    cleared_image_events += 1
                    if delete_records:
                        removed_events += 1
                        continue
                    item['images'] = {}
                    item['images_deleted_at'] = datetime.now().isoformat()
                kept.append(item)
            self.history = kept
            self.save_history()
        # delete orphan files older than cutoff
        if delete_orphan_files and self.IMAGE_ROOT.exists():
            for file in self.IMAGE_ROOT.rglob('*.jpg'):
                try:
                    mtime = datetime.fromtimestamp(file.stat().st_mtime)
                    if mtime < cutoff:
                        file.unlink()
                        deleted_images += 1
                except Exception:
                    pass
        return {'cleared_image_events': cleared_image_events, 'removed_events': removed_events, 'deleted_images': deleted_images, 'remaining': len(self.history), 'retention_days': days}

    def _filters(self, filters=None):
        filters = filters or {}
        now = datetime.now()
        days = int(filters.get('days') or config_manager.get('dashboard', 'default_range_days') or 7)
        def parse(value):
            if not value:
                return None
            try:
                dt = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
                if len(str(value)) <= 10:
                    dt = dt.replace(hour=0, minute=0, second=0, microsecond=0)
                return dt.replace(tzinfo=None) if getattr(dt, 'tzinfo', None) else dt
            except Exception:
                return None
        date_to = parse(filters.get('date_to')) or now
        if filters.get('date_to') and len(str(filters.get('date_to'))) <= 10:
            date_to = date_to.replace(hour=23, minute=59, second=59)
        date_from = parse(filters.get('date_from'))
        if not date_from:
            date_from = date_to.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=max(days - 1, 0))
        return {
            'date_from': date_from,
            'date_to': date_to,
            'min_confidence': float(filters.get('min_confidence') or config_manager.get('people', 'confidence_threshold') or 0),
            'counted_only': str(filters.get('counted_only', 'true')).lower() not in ('0', 'false', 'no', 'nein'),
            'event_type': filters.get('event_type') or '',
            'direction': filters.get('direction') or ''
        }

    def search(self, filters=None):
        f = self._filters(filters)
        rows = []
        for item in self.history:
            ts = self._parse_datetime(item.get('timestamp'))
            if not ts:
                continue
            ts = ts.replace(tzinfo=None) if getattr(ts, 'tzinfo', None) else ts
            if f['date_from'] and ts < f['date_from']:
                continue
            if f['date_to'] and ts > f['date_to']:
                continue
            if float(item.get('confidence') or 0) < f['min_confidence']:
                continue
            if f['counted_only'] and not item.get('counted', True):
                continue
            if f['event_type'] and item.get('event_type') != f['event_type']:
                continue
            if f['direction'] and f['direction'] != 'all' and item.get('direction') != f['direction']:
                continue
            rows.append(item)
        return rows

    def cleanup(self, retention_days=None):
        retention_days = int(retention_days or config_manager.get('people', 'retention_days') or 0)
        if retention_days <= 0:
            return {'removed': 0, 'remaining': len(self.history), 'retention_days': retention_days}
        cutoff = datetime.now() - timedelta(days=retention_days)
        with self.lock:
            before = len(self.history)
            kept = []
            for item in self.history:
                ts = self._parse_datetime(item.get('timestamp'))
                if ts and ts.replace(tzinfo=None) < cutoff:
                    continue
                kept.append(item)
            self.history = kept
            self.save_history()
        return {'removed': before - len(self.history), 'remaining': len(self.history), 'retention_days': retention_days}

    def get_presence(self):
        timeout = int(config_manager.get('people', 'present_timeout_minutes') or 10)
        cutoff = datetime.now() - timedelta(minutes=timeout)
        active_tracks = {}
        for item in self.history:
            ts = self._parse_datetime(item.get('timestamp'))
            if not ts:
                continue
            ts = ts.replace(tzinfo=None) if getattr(ts, 'tzinfo', None) else ts
            if ts < cutoff:
                continue
            track_key = str(item.get('track_id') if item.get('track_id') is not None else item.get('id'))
            if track_key not in active_tracks or ts > active_tracks[track_key]['timestamp_dt']:
                active_tracks[track_key] = dict(item, timestamp_dt=ts)
        rows = []
        for row in active_tracks.values():
            row.pop('timestamp_dt', None)
            rows.append(row)
        rows.sort(key=lambda x: x.get('timestamp') or '', reverse=True)
        return {'active_count': len(rows), 'timeout_minutes': timeout, 'active': rows[:100]}

    def get_statistics(self, filters=None):
        rows = self.search(filters)
        daily = defaultdict(lambda: {'date': '', 'persons': 0, 'detections': 0, 'line_crossings': 0, 'appearances': 0, 'directions': Counter()})
        hourly = defaultdict(int)
        directions = Counter()
        event_types = Counter()
        tracks = set()
        confidences = []
        for item in rows:
            ts = self._parse_datetime(item.get('timestamp'))
            if not ts:
                continue
            ts = ts.replace(tzinfo=None) if getattr(ts, 'tzinfo', None) else ts
            day = ts.date().isoformat()
            daily[day]['date'] = day
            daily[day]['detections'] += 1
            if item.get('counted', True):
                daily[day]['persons'] += 1
                hourly[ts.strftime('%Y-%m-%d %H:00')] += 1
            if item.get('event_type') == 'line_crossing':
                daily[day]['line_crossings'] += 1
            if item.get('event_type') == 'appearance':
                daily[day]['appearances'] += 1
            direction = item.get('direction') or 'unknown'
            daily[day]['directions'][direction] += 1
            directions[direction] += 1
            event_types[item.get('event_type') or 'unknown'] += 1
            if item.get('track_id') is not None:
                tracks.add(str(item.get('track_id')))
            confidences.append(float(item.get('confidence') or 0))
        daily_items = []
        for _, row in sorted(daily.items()):
            daily_items.append({
                'date': row['date'],
                'persons': row['persons'],
                'detections': row['detections'],
                'line_crossings': row['line_crossings'],
                'appearances': row['appearances'],
                'directions': dict(row['directions'])
            })
        today = datetime.now().date().isoformat()
        today_persons = next((d['persons'] for d in daily_items if d['date'] == today), 0)
        return {
            'summary': {
                'total_persons': sum(d['persons'] for d in daily_items),
                'today_persons': today_persons,
                'events': len(rows),
                'unique_tracks': len(tracks),
                'average_confidence': round(sum(confidences) / len(confidences), 4) if confidences else 0,
                'busiest_day': max(daily_items, key=lambda x: x['persons'], default=None),
                'note': 'Für genaue Durchgangszählung sollte die virtuelle Linie passend zur Kameraposition eingestellt werden.'
            },
            'daily': daily_items,
            'hourly': [{'hour': k, 'count': v} for k, v in sorted(hourly.items())],
            'directions': dict(directions),
            'event_types': dict(event_types),
            'latest': rows[:100],
            'config': config_manager.get('people') or {}
        }


# ============================================================
# KENNZEICHEN-DETEKTOR
# ============================================================

class LicensePlateDetector:
    """Haupt-Erkennungsklasse"""
    
    VEHICLE_CLASSES = {1: 'Fahrrad', 2: 'Auto / PKW', 3: 'Motorrad', 5: 'Bus', 7: 'LKW'}
    VEHICLE_CLASSES_EN = {1: 'bicycle', 2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
    COUNTRY_LABELS_DE = {
        'Unknown': 'Unbekannt',
        'Switzerland': 'Schweiz',
        'Germany': 'Deutschland',
        'Austria': 'Österreich',
        'France': 'Frankreich',
        'Italy': 'Italien',
        'Liechtenstein': 'Liechtenstein',
        'Netherlands': 'Niederlande',
        'Belgium': 'Belgien',
        'Luxembourg': 'Luxemburg',
        'United Kingdom': 'Vereinigtes Königreich',
        'United States': 'Vereinigte Staaten',
        'Spain': 'Spanien',
        'Portugal': 'Portugal',
        'Poland': 'Polen',
        'Czech Republic': 'Tschechien',
        'Slovakia': 'Slowakei',
        'Slovenia': 'Slowenien',
        'Croatia': 'Kroatien',
        'Hungary': 'Ungarn',
        'Romania': 'Rumänien',
        'Bulgaria': 'Bulgarien',
        'Denmark': 'Dänemark',
        'Sweden': 'Schweden',
        'Norway': 'Norwegen',
        'Finland': 'Finnland',
        'Ireland': 'Irland',
        'Czechia': 'Tschechien',
    }
    
    def __init__(self, config_manager):
        self.config_manager = config_manager
        self.coco_model = None
        self.license_model = None
        self.human_model = None
        self.ocr_reader = None
        self.fast_plate_recognizer = None
        self.fast_plate_cache_key = None
        self.models_loaded = False
        self.load_lock = threading.Lock()
        self.recent_plates = {}
        self.person_tracks = {}
        self.person_next_track_id = 1
    
    def load_models(self):
        with self.load_lock:
            if self.models_loaded:
                return True
            
            try:
                logger.info("Lade ML-Modelle...")
                
                vehicle_model_path = _resolve_model_path(self.config_manager.get('models', 'vehicle_detector'))
                license_model_path = _resolve_model_path(self.config_manager.get('models', 'license_plate_detector'))
                
                if vehicle_model_path and os.path.exists(vehicle_model_path):
                    self.coco_model = YOLO(vehicle_model_path)
                    logger.info(f"Fahrzeug-Modell geladen: {vehicle_model_path}")
                else:
                    logger.warning(f"Fahrzeug-Modell nicht gefunden: {vehicle_model_path}")
                
                if license_model_path and os.path.exists(license_model_path):
                    self.license_model = YOLO(license_model_path)
                    logger.info(f"Kennzeichen-Modell geladen: {license_model_path}")
                else:
                    logger.warning(f"Kennzeichen-Modell nicht gefunden: {license_model_path}")

                self.human_model = None
                people_cfg = self.config_manager.get('people') or {}
                human_path = (
                    people_cfg.get('selected_model_file')
                    or people_cfg.get('custom_model_path')
                    or people_cfg.get('model_path')
                    or self.config_manager.get('models', 'person_detector')
                )
                people_mode = people_cfg.get('model_mode') or 'coco_person'
                if people_cfg.get('enabled') and people_mode in ('custom_human', 'custom_path', 'model_file'):
                    human_path = _resolve_model_path(human_path)
                    if human_path and os.path.exists(human_path):
                        self.human_model = YOLO(human_path)
                        logger.info(f"Personen-Modell geladen: {human_path}")
                    else:
                        logger.warning(f"Personen-Modell nicht gefunden: {human_path}; Fallback auf COCO-Personenklasse")
                
                ocr_engine = str(self.config_manager.get('ocr', 'engine') or 'fast_plate_ocr').lower()
                easyocr_backup = bool(self.config_manager.get('ocr', 'easyocr_backup_enabled'))
                self.ocr_reader = None
                if ocr_engine == 'easyocr' or easyocr_backup:
                    try:
                        languages = self.config_manager.get('ocr', 'languages') or ['en']
                        gpu_enabled = self.config_manager.get('ocr', 'gpu_enabled') or False
                        self.ocr_reader = easyocr.Reader(languages, gpu=gpu_enabled)
                        logger.info(f"EasyOCR geladen mit Sprachen: {languages}")
                    except Exception as exc:
                        logger.warning(f"EasyOCR konnte nicht geladen werden: {exc}")
                if ocr_engine == 'fast_plate_ocr':
                    logger.info(f"fast-plate-ocr ist als Standard-OCR aktiv: {self.config_manager.get('ocr', 'fast_plate_model') or 'cct-s-v2-global-model'}")
                
                self.models_loaded = True
                return True
                
            except Exception as e:
                logger.error(f"Fehler beim Laden der Modelle: {e}")
                import traceback
                traceback.print_exc()
                return False
    
    def _yolo_runtime_kwargs(self):
        kwargs = {}
        device = self.config_manager.get('models', 'device') or 'auto'
        if device and device != 'auto':
            kwargs['device'] = device
        if self.config_manager.get('models', 'half_precision') and device not in ('cpu', 'auto'):
            kwargs['half'] = True
        return kwargs

    def _is_duplicate(self, plate_text):
        if not plate_text or len(plate_text) < 3:
            return True
            
        filter_enabled = self.config_manager.get('history', 'filter_duplicates')
        if not filter_enabled:
            return False
        
        timeout = self.config_manager.get('history', 'duplicate_timeout') or 60
        current_time = time.time()
        
        self.recent_plates = {
            k: v for k, v in self.recent_plates.items() 
            if current_time - v < timeout
        }
        
        normalized = PlateUtils.normalize(plate_text, compact=True)
        
        if normalized in self.recent_plates:
            return True
        if self.config_manager.get('history', 'fuzzy_duplicate_detection'):
            threshold = self.config_manager.get('history', 'fuzzy_duplicate_similarity') or 0.88
            for recent_plate in self.recent_plates.keys():
                if PlateUtils.similarity(recent_plate, normalized) >= threshold:
                    return True
        
        self.recent_plates[normalized] = current_time
        return False
    
    def _rgb_to_hex(self, rgb):
        return "#" + "".join(f"{max(0, min(255, int(v))):02X}" for v in rgb)

    def _neutral_gray_name(self, val):
        if val < 45:
            return "Schwarz", "black"
        if val < 90:
            return "Dunkelgrau", "dark_gray"
        if val < 155:
            return "Grau", "gray"
        if val < 215:
            return "Silber / Hellgrau", "silver"
        return "Weiss", "white"

    def _classify_rgb_color(self, rgb):
        """Vehicle-paint color name from RGB.

        Metallic cars often have a blue/green camera tint. This classifier uses
        saturation/chroma first, so muted paint is named Blaugrau/Silber/Grau
        instead of plain Blau.
        """
        r, g, b = [max(0, min(255, int(v))) for v in rgb]
        maxc = max(r, g, b)
        minc = min(r, g, b)
        chroma = maxc - minc
        hsv_pixel = cv2.cvtColor(np.array([[[b, g, r]]], dtype=np.uint8), cv2.COLOR_BGR2HSV)[0, 0]
        hue = int(hsv_pixel[0]) * 2
        sat = int(hsv_pixel[1])
        val = int(hsv_pixel[2])

        if val < 42:
            return "Schwarz", "black"
        if sat < 38 or chroma < 22:
            return self._neutral_gray_name(val)
        if sat < 115 and chroma < 70:
            if 165 <= hue < 255:
                return ("Dunkles Blaugrau", "dark_blue_gray") if val < 125 else ("Blaugrau / Silber", "blue_gray")
            if 65 <= hue < 165:
                return ("Dunkles Graugrün", "dark_green_gray") if val < 125 else ("Graugrün", "green_gray")
            if 35 <= hue < 65:
                return ("Beige / Champagner", "beige") if val > 145 else ("Braungrau", "brown_gray")
            if 15 <= hue < 35:
                return ("Braun / Beige", "brown_beige") if val > 120 else ("Dunkelbraun", "dark_brown")
            if hue < 15 or hue >= 330:
                return ("Bordeaux / Rotbraun", "red_brown") if val < 150 else ("Rotgrau", "red_gray")
            if 255 <= hue < 300:
                return "Violettgrau", "purple_gray"

        if hue < 15 or hue >= 345:
            return "Rot", "red"
        if 15 <= hue < 35:
            return ("Braun", "brown") if val < 140 else ("Orange", "orange")
        if 35 <= hue < 65:
            return ("Gold / Gelb", "yellow") if val > 115 else ("Braun", "brown")
        if 65 <= hue < 170:
            return ("Dunkelgrün", "dark_green") if val < 95 else ("Grün", "green")
        if 170 <= hue < 255:
            return ("Dunkelblau", "dark_blue") if val < 95 else ("Blau", "blue")
        if 255 <= hue < 290:
            return "Violett", "purple"
        if 290 <= hue < 345:
            return "Rot", "red"
        return "Unbekannt", "unknown"

    def _classify_hsv_color(self, h, s, v):
        bgr = cv2.cvtColor(np.array([[[int(h), int(s), int(v)]]], dtype=np.uint8), cv2.COLOR_HSV2BGR)[0, 0]
        rgb = [int(bgr[2]), int(bgr[1]), int(bgr[0])]
        return self._classify_rgb_color(rgb)

    def _estimate_vehicle_color(self, vehicle_crop, plate_box_relative=None):
        """Return detailed dominant vehicle color info.

        The returned dict contains the display name plus HEX/RGB swatch. The
        public/history fields still keep a plain vehicle_color string for
        compatibility, while HEX/RGB are stored separately.
        """
        try:
            if vehicle_crop is None or vehicle_crop.size == 0:
                return None

            h, w = vehicle_crop.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            mx = max(1, int(w * 0.08))
            my = max(1, int(h * 0.12))
            mask[my:max(my + 1, h - my), mx:max(mx + 1, w - mx)] = 1

            if plate_box_relative:
                try:
                    px1, py1, px2, py2 = [int(v) for v in plate_box_relative]
                    pad_x = int(w * 0.03)
                    pad_y = int(h * 0.03)
                    rx1 = max(0, px1 - pad_x)
                    ry1 = max(0, py1 - pad_y)
                    rx2 = min(w, px2 + pad_x)
                    ry2 = min(h, py2 + pad_y)
                    if rx2 > rx1 and ry2 > ry1:
                        mask[ry1:ry2, rx1:rx2] = 0
                except Exception:
                    pass

            pixels = vehicle_crop[mask.astype(bool)]
            if pixels.size == 0:
                pixels = vehicle_crop.reshape(-1, 3)
            if len(pixels) > 60000:
                step = max(1, len(pixels) // 60000)
                pixels = pixels[::step]

            hsv = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
            buckets = {}
            for bgr, hsv_pixel in zip(pixels, hsv):
                name_de, name_en = self._classify_hsv_color(int(hsv_pixel[0]), int(hsv_pixel[1]), int(hsv_pixel[2]))
                if name_en not in buckets:
                    buckets[name_en] = {"name": name_de, "english": name_en, "count": 0, "bgr_sum": np.zeros(3, dtype=np.float64)}
                buckets[name_en]["count"] += 1
                buckets[name_en]["bgr_sum"] += bgr.astype(np.float64)

            if not buckets:
                return None

            total = sum(int(item["count"]) for item in buckets.values())
            palette = []
            for item in sorted(buckets.values(), key=lambda value: int(value["count"]), reverse=True):
                count = int(item["count"])
                avg_bgr = item["bgr_sum"] / max(1, count)
                rgb = [int(round(avg_bgr[2])), int(round(avg_bgr[1])), int(round(avg_bgr[0]))]
                display_name, display_key = self._classify_rgb_color(rgb)
                palette.append({
                    "name": display_name,
                    "english": display_key,
                    "raw_bucket_name": item["name"],
                    "raw_bucket_english": item["english"],
                    "coverage": round(count / max(1, total), 4),
                    "rgb": rgb,
                    "hex": self._rgb_to_hex(rgb),
                })

            # IMPORTANT: never attach the original palette list to palette[0].
            # That creates a self-reference: best -> palette -> best, and Flask's
            # jsonify/json.dumps then fails with "Circular reference detected".
            best = dict(palette[0])
            palette_preview = [dict(item) for item in palette[:5]]
            best["method"] = "dominant vehicle paint swatch + RGB-based color naming; heuristic"
            best["note"] = "Schätzung aus Pixeln, kein spezialisiertes Lackfarben-Modell. HEX/RGB ist die wichtigste Messung."
            best["palette"] = palette_preview
            return best
        except Exception as e:
            logger.debug(f"Farb-Ermittlung fehlgeschlagen: {e}")
            return None

    def _apply_vehicle_color_fields(self, vehicle_info, color_info):
        if not color_info:
            vehicle_info.update({
                'color': 'Unbekannt',
                'color_hex': None,
                'color_rgb': None,
                'color_coverage': None,
                'color_info': None,
            })
            return vehicle_info
        vehicle_info.update({
            'color': color_info.get('name') or 'Unbekannt',
            'color_hex': color_info.get('hex'),
            'color_rgb': color_info.get('rgb'),
            'color_coverage': color_info.get('coverage'),
            'color_info': color_info,
        })
        return vehicle_info

    def _preprocess_plate_image(self, plate_image):
        if plate_image is None or plate_image.size == 0:
            return None, []
        
        config = self.config_manager.get('ocr', 'preprocessing') or {}
        
        if not config.get('enabled', True):
            return plate_image, [plate_image]
        
        try:
            processed = plate_image.copy()
            height, width = processed.shape[:2]
            
            target_height = config.get('target_height', 120)
            min_width = config.get('min_width', 200)
            resize_factor = config.get('resize_factor', 4.0)
            
            if height < target_height or width < min_width:
                scale_h = target_height / height if height < target_height else 1
                scale_w = min_width / width if width < min_width else 1
                scale = max(scale_h, scale_w)
                scale = min(scale, resize_factor)
                
                new_width = int(width * scale)
                new_height = int(height * scale)
                
                processed = cv2.resize(processed, (new_width, new_height), 
                                       interpolation=cv2.INTER_CUBIC)
            
            border_padding = int(config.get('border_padding', 0) or 0)
            if border_padding > 0:
                processed = cv2.copyMakeBorder(processed, border_padding, border_padding, border_padding, border_padding, cv2.BORDER_REPLICATE)

            if len(processed.shape) == 3:
                gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
            else:
                gray = processed
            
            variants = [gray.copy()]

            if config.get('gamma_correction'):
                gamma = float(config.get('gamma', 1.2) or 1.2)
                gamma = max(0.2, min(gamma, 5.0))
                inv_gamma = 1.0 / gamma
                table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype('uint8')
                gray = cv2.LUT(gray, table)
                variants.append(gray.copy())
            
            if config.get('denoise', True):
                denoise_strength = int(config.get('denoise_strength', 10) or 10)
                gray = cv2.fastNlMeansDenoising(gray, None, denoise_strength, 7, 21)

            if config.get('bilateral_filter'):
                gray = cv2.bilateralFilter(gray, 7, 50, 50)
            
            if config.get('contrast_enhance', True):
                clip_limit = float(config.get('clahe_clip_limit', 3.0) or 3.0)
                tile = int(config.get('clahe_tile_grid', 8) or 8)
                clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile))
                gray = clahe.apply(gray)
            
            variants.append(gray.copy())
            
            if config.get('sharpen', True):
                kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
                sharpened = cv2.filter2D(gray, -1, kernel)
                variants.append(sharpened)
            
            if config.get('adaptive_threshold', True):
                block_size = int(config.get('threshold_block_size', 11) or 11)
                if block_size % 2 == 0:
                    block_size += 1
                block_size = max(3, block_size)
                c_value = int(config.get('threshold_c', 2) or 2)
                thresh1 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                cv2.THRESH_BINARY, block_size, c_value)
                variants.append(thresh1)
                
                _, thresh3 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                variants.append(thresh3)

            if config.get('morphology', True):
                kernel = np.ones((2, 2), np.uint8)
                variants.append(cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel))
            
            if config.get('invert_variant', True):
                inverted = cv2.bitwise_not(gray)
                variants.append(inverted)
            
            return gray, variants
            
        except Exception as e:
            return plate_image, [plate_image]
    
    def _read_plate_enhanced(self, plate_image):
        if not self.ocr_reader or plate_image is None or plate_image.size == 0:
            return None, 0
        
        try:
            result = self._preprocess_plate_image(plate_image)
            
            if result is None:
                return None, 0
            
            processed, variants = result
            
            min_confidence = self.config_manager.get('ocr', 'min_confidence') or 0.25
            allowed_chars = self.config_manager.get('ocr', 'allowed_characters') or ''
            use_allowlist = self.config_manager.get('ocr', 'use_allowlist')
            max_variants = int(self.config_manager.get('ocr', 'max_variants_to_read') or 8)
            early_stop = float(self.config_manager.get('ocr', 'early_stop_confidence') or 0.85)
            paragraph_mode = bool(self.config_manager.get('ocr', 'paragraph_mode'))
            decoder = self.config_manager.get('ocr', 'decoder') or 'greedy'
            rotation_variants = bool(self.config_manager.get('ocr', 'rotation_variants'))
            
            if rotation_variants:
                rotated = []
                for variant in variants[:3]:
                    try:
                        rotated.append(cv2.rotate(variant, cv2.ROTATE_180))
                    except Exception:
                        pass
                variants.extend(rotated)

            best_result = None
            best_confidence = 0
            
            for i, variant in enumerate(variants[:max_variants]):
                try:
                    kwargs = {'detail': 1, 'paragraph': paragraph_mode, 'decoder': decoder}
                    if use_allowlist and allowed_chars:
                        kwargs['allowlist'] = allowed_chars
                    results = self.ocr_reader.readtext(variant, **kwargs)
                    text, confidence = self._process_ocr_results(results, min_confidence * 0.7, allowed_chars)
                    
                    if text and confidence > best_confidence:
                        best_confidence = confidence
                        best_result = text
                    
                    if best_confidence >= early_stop:
                        break
                        
                except Exception:
                    continue
            
            if best_result:
                country_hint = self.config_manager.get('plate_recognition', 'country_hint') or 'auto'
                analysis = PlateUtils.best_candidate(best_result, country_hint=country_hint)
                best_result = analysis.get('corrected') or self._correct_common_errors(best_result)
                if not PlateUtils.is_valid(
                    best_result,
                    self.config_manager.get('plate_recognition', 'min_length') or 3,
                    self.config_manager.get('plate_recognition', 'max_length') or 12,
                    self.config_manager.get('plate_recognition', 'validation_regex') or None
                ):
                    return None, 0
                if self.config_manager.get('plate_recognition', 'format_pretty_output'):
                    best_result = analysis.get('pretty') or PlateUtils.pretty(best_result)
            
            return best_result, best_confidence
            
        except Exception as e:
            logger.error(f"OCR Fehler: {e}")
            return None, 0
    
    def _parse_fast_plate_providers(self):
        raw = self.config_manager.get('ocr', 'fast_plate_providers') or ''
        providers = [item.strip() for item in str(raw).split(',') if item.strip()]
        return providers or None

    def _get_fast_plate_recognizer(self):
        model_name = self.config_manager.get('ocr', 'fast_plate_model') or 'cct-s-v2-global-model'
        device = self.config_manager.get('ocr', 'fast_plate_device') or self.config_manager.get('models', 'device') or 'auto'
        providers = self._parse_fast_plate_providers()
        cache_key = (model_name, device, tuple(providers or []))
        if self.fast_plate_recognizer is not None and self.fast_plate_cache_key == cache_key:
            return self.fast_plate_recognizer
        try:
            from fast_plate_ocr import LicensePlateRecognizer
        except ModuleNotFoundError as exc:
            raise RuntimeError('fast-plate-ocr ist nicht installiert. Installiere: pip install fast-plate-ocr[onnx]') from exc
        self.fast_plate_recognizer = LicensePlateRecognizer(
            hub_ocr_model=model_name,
            device=device,
            providers=providers,
        )
        self.fast_plate_cache_key = cache_key
        logger.info(f"fast-plate-ocr geladen: {model_name} / device={device} / providers={providers or 'auto'}")
        return self.fast_plate_recognizer

    def _prediction_to_plate_meta(self, prediction, recognizer):
        cfg = getattr(recognizer, 'config', None)
        pad_char = getattr(cfg, 'pad_char', '_') if cfg is not None else '_'
        remove_pad = self.config_manager.get('ocr', 'fast_plate_remove_pad_char') is not False
        raw_plate = getattr(prediction, 'plate', None) or ''
        plate_text = raw_plate.replace(pad_char, '') if remove_pad and pad_char else raw_plate
        plate_text = PlateUtils.normalize(plate_text, compact=True)

        char_probs = getattr(prediction, 'char_probs', None)
        char_prob_list = None
        mean_all = None
        mean_visible = None
        if char_probs is not None:
            try:
                char_prob_list = [float(x) for x in np.asarray(char_probs).ravel()]
                mean_all = float(np.mean(char_prob_list)) if char_prob_list else None
                visible = []
                for idx, prob in enumerate(char_prob_list):
                    char = raw_plate[idx] if idx < len(raw_plate) else ''
                    if not pad_char or char != pad_char:
                        visible.append(float(prob))
                mean_visible = float(np.mean(visible)) if visible else None
            except Exception:
                char_prob_list = None

        region = getattr(prediction, 'region', None)
        region_prob = getattr(prediction, 'region_prob', None)
        try:
            region_prob = float(region_prob) if region_prob is not None else None
        except Exception:
            region_prob = None
        country_display = self.COUNTRY_LABELS_DE.get(region, region) if region else None
        only_padding = bool(raw_plate and pad_char and raw_plate.replace(pad_char, '') == '')
        confidence = mean_visible if mean_visible is not None else (mean_all if mean_all is not None else (region_prob or 0.0))
        return {
            'plate_text': plate_text,
            'raw_plate': raw_plate,
            'only_padding': only_padding,
            'pad_char': pad_char,
            'char_probs': char_prob_list,
            'mean_char_prob_all': mean_all,
            'mean_char_prob_visible': mean_visible,
            'plate_country': region,
            'plate_country_display': country_display,
            'plate_country_prob': region_prob,
            'plate_region': region,
            'plate_region_prob': region_prob,
            'ocr_engine': 'fast_plate_ocr',
            'ocr_model': self.config_manager.get('ocr', 'fast_plate_model') or 'cct-s-v2-global-model',
            'ocr_confidence': confidence or 0.0,
        }

    def _read_plate_fast_plate_ocr(self, plate_image):
        if plate_image is None or plate_image.size == 0:
            return None, 0, {'ocr_engine': 'fast_plate_ocr'}
        recognizer = self._get_fast_plate_recognizer()
        cfg = getattr(recognizer, 'config', None)
        color_mode = getattr(cfg, 'image_color_mode', 'rgb') if cfg is not None else 'rgb'
        if color_mode == 'rgb' and plate_image.ndim == 3:
            ocr_input = cv2.cvtColor(plate_image, cv2.COLOR_BGR2RGB)
        else:
            ocr_input = plate_image
        started = time.perf_counter()
        predictions = recognizer.run(
            ocr_input,
            return_confidence=self.config_manager.get('ocr', 'fast_plate_return_confidence') is not False,
            remove_pad_char=False,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000
        if not predictions:
            return None, 0, {'ocr_engine': 'fast_plate_ocr', 'ocr_elapsed_ms': round(elapsed_ms, 3), 'error': 'Keine Prediction'}
        meta = self._prediction_to_plate_meta(predictions[0], recognizer)
        meta['ocr_elapsed_ms'] = round(elapsed_ms, 3)
        plate_text = meta.get('plate_text') or ''
        if meta.get('only_padding') or not plate_text:
            return None, 0, meta
        confidence = float(meta.get('ocr_confidence') or 0)
        return plate_text, confidence, meta

    def _read_plate_ocr(self, plate_image):
        engine = str(self.config_manager.get('ocr', 'engine') or 'fast_plate_ocr').lower().replace('-', '_')
        if engine in ('fast_plate_ocr', 'fastplateocr', 'fast'):
            try:
                text, confidence, meta = self._read_plate_fast_plate_ocr(plate_image)
                min_conf = float(self.config_manager.get('ocr', 'min_confidence') or 0.0)
                if text and confidence >= min_conf:
                    return text, confidence, meta
                if self.config_manager.get('ocr', 'easyocr_backup_enabled') and self.ocr_reader is not None:
                    easy_text, easy_conf = self._read_plate_enhanced(plate_image)
                    if easy_text and easy_conf > confidence:
                        easy_meta = dict(meta or {})
                        easy_meta.update({'ocr_engine': 'easyocr_backup', 'ocr_confidence': easy_conf, 'fast_plate_failed_text': text})
                        return easy_text, easy_conf, easy_meta
                return text, confidence, meta
            except Exception as exc:
                logger.warning(f"fast-plate-ocr Fehler, versuche EasyOCR-Backup: {exc}")
                if self.config_manager.get('ocr', 'easyocr_backup_enabled') and self.ocr_reader is not None:
                    easy_text, easy_conf = self._read_plate_enhanced(plate_image)
                    return easy_text, easy_conf, {'ocr_engine': 'easyocr_backup', 'ocr_confidence': easy_conf, 'fast_plate_error': str(exc)}
                return None, 0, {'ocr_engine': 'fast_plate_ocr', 'error': str(exc)}

        easy_text, easy_conf = self._read_plate_enhanced(plate_image)
        return easy_text, easy_conf, {'ocr_engine': 'easyocr', 'ocr_confidence': easy_conf}

    def _bbox_center(self, box):
        x1, y1, x2, y2 = [float(v) for v in box]
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    def _bbox_area(self, box):
        x1, y1, x2, y2 = [int(v) for v in box]
        return max(0, x2 - x1) * max(0, y2 - y1)

    def _intersection_area(self, a, b):
        ax1, ay1, ax2, ay2 = [int(v) for v in a]
        bx1, by1, bx2, by2 = [int(v) for v in b]
        x1, y1 = max(ax1, bx1), max(ay1, by1)
        x2, y2 = min(ax2, bx2), min(ay2, by2)
        return max(0, x2 - x1) * max(0, y2 - y1)

    def _match_vehicle_for_plate(self, plate_box, vehicles):
        if not plate_box or not vehicles:
            return None
        pcx, pcy = self._bbox_center(plate_box)
        containing = []
        for vehicle in vehicles:
            box = vehicle.get('bbox')
            if not box:
                continue
            x1, y1, x2, y2 = [int(v) for v in box]
            if x1 <= pcx <= x2 and y1 <= pcy <= y2:
                containing.append((self._bbox_area(box), vehicle))
        if containing:
            _, vehicle = min(containing, key=lambda item: item[0])
            vehicle['match_reason'] = 'plate_center_inside_vehicle'
            return vehicle
        best = None
        for vehicle in vehicles:
            box = vehicle.get('bbox')
            if not box:
                continue
            inter = self._intersection_area(plate_box, box)
            if inter > 0 and (best is None or inter > best[0]):
                best = (inter, vehicle)
        if best is not None:
            _, vehicle = best
            vehicle['match_reason'] = 'bbox_intersection'
            return vehicle
        nearest = None
        for vehicle in vehicles:
            box = vehicle.get('bbox')
            if not box:
                continue
            vcx, vcy = self._bbox_center(box)
            dist = ((pcx - vcx) ** 2 + (pcy - vcy) ** 2) ** 0.5
            if nearest is None or dist < nearest[0]:
                nearest = (dist, vehicle)
        if nearest is None:
            return None
        _, vehicle = nearest
        vehicle['match_reason'] = 'nearest_vehicle'
        return vehicle

    def _refresh_vehicle_color_excluding_plate(self, frame, vehicle, plate_box):
        if not vehicle or not plate_box:
            return vehicle
        try:
            vx1, vy1, vx2, vy2 = [int(v) for v in vehicle.get('bbox')]
            px1, py1, px2, py2 = [int(v) for v in plate_box]
            rel_plate = [px1 - vx1, py1 - vy1, px2 - vx1, py2 - vy1]
            vehicle_crop = frame[vy1:vy2, vx1:vx2].copy()
            color_info = self._estimate_vehicle_color(vehicle_crop, rel_plate)
            self._apply_vehicle_color_fields(vehicle, color_info)
        except Exception:
            pass
        return vehicle

    def _process_ocr_results(self, results, min_confidence, allowed_chars):
        if not results:
            return None, 0

        candidates = []
        allowed_upper = allowed_chars.upper() if allowed_chars else ''
        for result in results:
            if len(result) >= 3:
                text, confidence = result[1], float(result[2])
            elif len(result) == 2:
                text, confidence = result[0], float(result[1])
            else:
                continue
            if confidence < min_confidence:
                continue
            raw_text = str(text).upper() if self.config_manager.get('ocr', 'uppercase_output') else str(text)
            clean_text = ''.join(c for c in raw_text if not allowed_upper or c.upper() in allowed_upper)
            clean_text = PlateUtils.normalize(clean_text, compact=True)
            min_text_length = int(self.config_manager.get('ocr', 'min_text_length') or 2)
            if len(clean_text) >= min_text_length:
                candidates.append((clean_text, confidence))

        if not candidates:
            return None, 0

        # Einzelne OCR-Fragmente und zusammengesetzte Variante bewerten.
        if self.config_manager.get('ocr', 'merge_fragments'):
            combined = ''.join(c[0] for c in candidates)
            avg = sum(c[1] for c in candidates) / len(candidates)
            candidates.append((combined, avg))
        best_text, best_conf = max(candidates, key=lambda item: (item[1], len(item[0])))
        return best_text, best_conf

    def _correct_common_errors(self, text):
        if not text or len(text) < 2:
            return text
        if not self.config_manager.get('plate_recognition', 'smart_ocr_correction'):
            return PlateUtils.normalize(text, compact=True)
        country_hint = self.config_manager.get('plate_recognition', 'country_hint') or 'auto'
        return PlateUtils.smart_correct(text, country_hint=country_hint)


    def _person_config(self):
        return self.config_manager.get('people') or {}

    def _person_runtime_model(self):
        cfg = self._person_config()
        if cfg.get('model_mode') in ('custom_human', 'custom_path', 'model_file') and self.human_model is not None:
            return self.human_model, False
        return self.coco_model, True

    def _track_people(self, detections, frame_w, frame_h):
        cfg = self._person_config()
        if not cfg.get('tracker_enabled', True):
            for index, det in enumerate(detections, start=1):
                det['track_id'] = index
                det['counted'] = True
                det['event_type'] = 'person_detected'
                det['direction'] = 'unknown'
            return detections

        now = time.time()
        max_distance = float(cfg.get('tracker_max_distance') or 120)
        timeout = float(cfg.get('tracker_timeout_seconds') or 8)
        axis = (cfg.get('movement_axis') or 'y').lower()
        line_percent = float(cfg.get('virtual_line_position_percent') or 50)
        line_value = (frame_w if axis == 'x' else frame_h) * line_percent / 100.0
        crossing_enabled = bool(cfg.get('line_crossing_enabled', True))
        crossing_direction = cfg.get('crossing_direction') or 'both'
        count_once = bool(cfg.get('count_once_per_track', True))
        count_strategy = cfg.get('count_strategy') or 'line_crossing'

        # purge stale tracks
        self.person_tracks = {tid: tr for tid, tr in self.person_tracks.items() if now - tr.get('last_seen', now) <= timeout}

        used_tracks = set()
        for det in detections:
            cx, cy = det['center_x'], det['center_y']
            best_tid, best_dist = None, None
            for tid, tr in self.person_tracks.items():
                if tid in used_tracks:
                    continue
                px, py = tr.get('center', (cx, cy))
                dist = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
                if dist <= max_distance and (best_dist is None or dist < best_dist):
                    best_tid, best_dist = tid, dist

            new_track = best_tid is None
            if new_track:
                best_tid = self.person_next_track_id
                self.person_next_track_id += 1
                self.person_tracks[best_tid] = {
                    'center': (cx, cy), 'previous_center': None, 'first_seen': now,
                    'last_seen': now, 'counted': False, 'last_side': None,
                    'hits': 0, 'last_counted_at': 0
                }
            tr = self.person_tracks[best_tid]
            previous = tr.get('center')
            tr['previous_center'] = previous
            tr['center'] = (cx, cy)
            tr['last_seen'] = now
            tr['hits'] = int(tr.get('hits') or 0) + 1
            used_tracks.add(best_tid)

            prev_value = previous[0] if previous and axis == 'x' else (previous[1] if previous else None)
            current_value = cx if axis == 'x' else cy
            prev_side = None if prev_value is None else ('positive' if prev_value >= line_value else 'negative')
            current_side = 'positive' if current_value >= line_value else 'negative'
            direction = 'unknown'
            event_type = 'person_detected'
            counted = False

            if new_track and count_strategy in ('first_seen', 'appearance'):
                event_type = 'appearance'
                counted = True
            elif crossing_enabled and prev_side is not None and prev_side != current_side:
                event_type = 'line_crossing'
                if axis == 'y':
                    direction = 'down' if current_value > prev_value else 'up'
                else:
                    direction = 'right' if current_value > prev_value else 'left'
                if crossing_direction == 'both' or direction == crossing_direction:
                    counted = True
            elif count_strategy == 'every_detection':
                counted = True

            min_age = max(1, int(cfg.get('min_track_age_frames') or 1))
            debounce = float(cfg.get('count_debounce_seconds') or 0)
            if counted and int(tr.get('hits') or 0) < min_age:
                counted = False
                event_type = 'waiting_min_track_age'
            if count_once and tr.get('counted') and counted:
                counted = False
                event_type = 'already_counted'
            if counted and debounce > 0 and now - float(tr.get('last_counted_at') or 0) < debounce:
                counted = False
                event_type = 'debounced'
            if counted:
                tr['counted'] = True
                tr['last_counted_at'] = now

            tr['last_side'] = current_side
            det.update({
                'track_id': best_tid,
                'counted': bool(counted),
                'event_type': event_type,
                'direction': direction,
                'line_value': round(line_value, 2),
                'tracking_status': 'new' if new_track else 'matched'
            })
        return detections

    def _detect_people(self, frame, annotated=None, runtime_roi_polygon=None):
        cfg = self._person_config()
        if not cfg.get('enabled'):
            return []
        model, is_coco = self._person_runtime_model()
        if model is None:
            return []
        frame_h, frame_w = frame.shape[:2]

        roi_contour = None
        if runtime_roi_polygon and cfg.get('roi_filter_enabled') is not False:
            try:
                roi_pts = []
                for pt in runtime_roi_polygon:
                    if isinstance(pt, dict):
                        roi_pts.append([float(pt.get('x', 0)), float(pt.get('y', 0))])
                    else:
                        roi_pts.append([float(pt[0]), float(pt[1])])
                if len(roi_pts) >= 3:
                    roi_contour = np.array(roi_pts, dtype=np.float32)
            except Exception:
                roi_contour = None

        def person_bbox_allowed_by_runtime_roi(x1, y1, x2, y2):
            if roi_contour is None:
                return True
            mode = str(cfg.get('roi_filter_mode') or 'foot_and_center').lower()
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            foot_x = cx
            foot_y = y2
            center_ok = cv2.pointPolygonTest(roi_contour, (float(cx), float(cy)), False) >= 0
            foot_ok = cv2.pointPolygonTest(roi_contour, (float(foot_x), float(foot_y)), False) >= 0
            if mode == 'center':
                return center_ok
            if mode == 'foot':
                return foot_ok
            if mode == 'center_or_foot':
                return center_ok or foot_ok
            return center_ok and foot_ok

        confidence = float(cfg.get('confidence_threshold') or 0.45)
        max_persons = int(cfg.get('max_persons_per_frame') or 0)
        min_w = int(cfg.get('min_person_width') or 0)
        min_h = int(cfg.get('min_person_height') or 0)
        runtime_kwargs = self._yolo_runtime_kwargs()
        yolo_kwargs = {'conf': confidence, 'verbose': False}
        try:
            if cfg.get('nms_iou_threshold') is not None:
                yolo_kwargs['iou'] = float(cfg.get('nms_iou_threshold'))
            if cfg.get('image_size'):
                yolo_kwargs['imgsz'] = int(cfg.get('image_size'))
        except Exception:
            pass
        yolo_kwargs.update(runtime_kwargs)
        if is_coco:
            yolo_kwargs['classes'] = [0]
        people = []
        try:
            results = model(frame, **yolo_kwargs)[0]
            names = getattr(model, 'names', {}) or {}
            allowed_names = {str(x).strip().lower() for x in (cfg.get('class_names') or ['person', 'human'])}
            allowed_ids = set(int(x) for x in (cfg.get('class_ids') or [0]) if str(x).strip() != '')
            for raw in results.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = raw[:6]
                class_id = int(class_id)
                class_name = str(names.get(class_id, 'person' if class_id == 0 else class_id)).lower()
                if not is_coco and allowed_ids and class_id not in allowed_ids and class_name not in allowed_names:
                    continue
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1
                if w < min_w or h < min_h:
                    continue
                area_percent = ((w * h) / max(1, frame_w * frame_h)) * 100.0
                min_area = float(cfg.get('min_area_percent') or 0)
                max_area = float(cfg.get('max_area_percent') or 100)
                if area_percent < min_area or area_percent > max_area:
                    continue
                aspect = w / max(1, h)
                min_aspect = float(cfg.get('min_aspect_ratio') or 0)
                max_aspect = float(cfg.get('max_aspect_ratio') or 99)
                if aspect < min_aspect or aspect > max_aspect:
                    continue
                cx_val = (x1 + x2) / 2
                cy_val = (y1 + y2) / 2
                zone_match = True
                if cfg.get('zone_enabled'):
                    z = cfg.get('zone') or {}
                    zx, zy = float(z.get('x') or 0), float(z.get('y') or 0)
                    zw, zh = float(z.get('width') or 100), float(z.get('height') or 100)
                    if (z.get('unit') or 'percent') == 'percent':
                        zx, zw = frame_w * zx / 100.0, frame_w * zw / 100.0
                        zy, zh = frame_h * zy / 100.0, frame_h * zh / 100.0
                    zone_match = (zx <= cx_val <= zx + zw and zy <= cy_val <= zy + zh)
                if not zone_match:
                    continue
                if not person_bbox_allowed_by_runtime_roi(x1, y1, x2, y2):
                    continue
                people.append({
                    'bbox': [x1, y1, x2, y2],
                    'center_x': round(cx_val, 2),
                    'center_y': round(cy_val, 2),
                    'width': int(w),
                    'height': int(h),
                    'area_percent': round(area_percent, 4),
                    'aspect_ratio': round(aspect, 4),
                    'zone_match': zone_match,
                    'confidence': float(score),
                    'class_id': class_id,
                    'class_name': class_name,
                    'frame_width': frame_w,
                    'frame_height': frame_h,
                    'source_model': 'coco_person' if is_coco else 'custom_human'
                })
                if max_persons and len(people) >= max_persons:
                    break
        except Exception as e:
            logger.error(f"Personenerkennung Fehler: {e}")
            return []

        people = self._track_people(people, frame_w, frame_h)
        if annotated is not None and cfg.get('draw_boxes', True):
            axis = (cfg.get('movement_axis') or 'y').lower()
            line_percent = float(cfg.get('virtual_line_position_percent') or 50)
            if cfg.get('line_crossing_enabled', True):
                if axis == 'y':
                    y = int(frame_h * line_percent / 100.0)
                    cv2.line(annotated, (0, y), (frame_w, y), (34, 211, 238), 2)
                    cv2.putText(annotated, 'Personen-Zaehllinie', (10, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (34, 211, 238), 2)
                else:
                    x = int(frame_w * line_percent / 100.0)
                    cv2.line(annotated, (x, 0), (x, frame_h), (34, 211, 238), 2)
                    cv2.putText(annotated, 'Personen-Zaehllinie', (min(frame_w - 220, x + 8), 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (34, 211, 238), 2)
            for person in people:
                x1, y1, x2, y2 = person['bbox']
                color = (16, 185, 129) if person.get('counted') else (245, 158, 11)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                label = f"P#{person.get('track_id')} {person.get('confidence', 0):.2f}"
                if person.get('counted'):
                    label += ' gezählt'
                cv2.putText(annotated, label, (x1, max(18, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        return people

    def process_frame(self, frame, apply_analysis_area=False, runtime_roi_polygon=None, filter_duplicates=True):
        """Verarbeitet einen einzelnen Frame"""
        if not self.models_loaded:
            self.load_models()
        
        if frame is None or frame.size == 0:
            return {
                'annotated_frame': np.zeros((480, 640, 3), dtype=np.uint8),
                'detections': [],
                'vehicles': [],
                'people': [],
                'processing_time': 0
            }
        
        result = {
            'annotated_frame': frame.copy(),
            'detections': [],
            'vehicles': [],
            'people': [],
            'processing_time': 0
        }
        
        start_time = time.time()
        
        try:
            confidence_threshold = self.config_manager.get('detection', 'confidence_threshold') or 0.5
            zoom_enabled = self.config_manager.get('detection', 'zoom_enabled') is not False
            zoom_factor = self.config_manager.get('detection', 'zoom_factor') or 2.5
            zoom_padding = self.config_manager.get('detection', 'zoom_padding') or 100
            max_detections_per_frame = int(self.config_manager.get('detection', 'max_detections_per_frame') or 0)
            allowed_vehicle_names = set(self.config_manager.get('detection', 'vehicle_class_filter') or ['car', 'truck', 'bus', 'motorcycle', 'bicycle'])
            min_vehicle_width = int(self.config_manager.get('detection', 'min_vehicle_width') or 0)
            min_vehicle_height = int(self.config_manager.get('detection', 'min_vehicle_height') or 0)
            annotate_frames = self.config_manager.get('detection', 'annotate_frames') is not False
            draw_confidence = self.config_manager.get('detection', 'draw_confidence') is not False
            
            annotated = frame.copy()
            detected_vehicles = []
            frame_h, frame_w = frame.shape[:2]
            
            # Fahrzeugerkennung
            if self.coco_model and self.config_manager.get('detection', 'car_detection_enabled'):
                vehicle_results = self.coco_model(frame, conf=confidence_threshold, verbose=False, **self._yolo_runtime_kwargs())[0]
                
                for detection in vehicle_results.boxes.data.tolist():
                    x1, y1, x2, y2, score, class_id = detection
                    class_id = int(class_id)
                    
                    if class_id in self.VEHICLE_CLASSES:
                        class_name_en = self.VEHICLE_CLASSES_EN.get(class_id, 'unknown')
                        if allowed_vehicle_names and class_name_en not in allowed_vehicle_names:
                            continue
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        if (x2 - x1) < min_vehicle_width or (y2 - y1) < min_vehicle_height:
                            continue
                        
                        vehicle_crop = frame[y1:y2, x1:x2].copy()
                        vehicle_color_info = self._estimate_vehicle_color(vehicle_crop)
                        
                        vehicle_info = {
                            'bbox': [x1, y1, x2, y2],
                            'confidence': score,
                            'type': self.VEHICLE_CLASSES[class_id],
                            'type_en': self.VEHICLE_CLASSES_EN[class_id],
                            'crop': vehicle_crop
                        }
                        self._apply_vehicle_color_fields(vehicle_info, vehicle_color_info)
                        detected_vehicles.append(vehicle_info)
                        
                        if annotate_frames:
                            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 0), 2)
                            label = f"{self.VEHICLE_CLASSES[class_id]} ({vehicle_info.get('color', 'Unbekannt')})"
                            if draw_confidence:
                                label += f" {score:.2f}"
                            cv2.putText(annotated, label, (x1, y1 - 10),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            result['vehicles'] = detected_vehicles
            
            # Kennzeichenerkennung
            if self.license_model:
                frames_to_process = []

                # Demo-compatible plate scan: the license-plate model first sees
                # the complete original image. This is important because the old
                # live loop sometimes cropped to the vehicle/ROI before the plate
                # detector. In that case the detector settings no longer matched
                # the demo mode and small plates were missed.
                plate_scan_strategy = str(self.config_manager.get('detection', 'plate_scan_strategy') or 'full_frame_first')
                scan_full = bool(self.config_manager.get('detection', 'scan_full_frame_when_vehicle_found'))
                full_frame_info = {
                    'frame': frame,
                    'offset': (0, 0),
                    'scale': 1,
                    'vehicle': None,
                    'source': 'full_frame'
                }
                if plate_scan_strategy in ('full_frame_first', 'demo', 'full_frame_only') or not (zoom_enabled and detected_vehicles):
                    frames_to_process.append(full_frame_info)

                if zoom_enabled and detected_vehicles and plate_scan_strategy != 'full_frame_only':
                    for vehicle in detected_vehicles:
                        x1, y1, x2, y2 = vehicle['bbox']
                        
                        pad = zoom_padding
                        zx1 = max(0, x1 - pad)
                        zy1 = max(0, y1 - pad)
                        zx2 = min(frame_w, x2 + pad)
                        zy2 = min(frame_h, y2 + pad)
                        
                        vehicle_region = frame[zy1:zy2, zx1:zx2]
                        
                        if vehicle_region.size == 0:
                            continue
                        
                        crop_h, crop_w = vehicle_region.shape[:2]
                        scale = max(800 / max(crop_w, 1), 800 / max(crop_h, 1), zoom_factor)
                        scale = min(scale, 5.0)
                        
                        vehicle_region_scaled = cv2.resize(
                            vehicle_region, 
                            (int(crop_w * scale), int(crop_h * scale)), 
                            interpolation=cv2.INTER_CUBIC
                        )
                        
                        frames_to_process.append({
                            'frame': vehicle_region_scaled,
                            'offset': (zx1, zy1),
                            'scale': scale,
                            'vehicle': vehicle,
                            'source': 'vehicle_zoom'
                        })

                if scan_full and not any(item.get('source') == 'full_frame' for item in frames_to_process):
                    frames_to_process.append(full_frame_info)
                
                for frame_info in frames_to_process:
                    proc_frame = frame_info['frame']
                    off_x, off_y = frame_info['offset']
                    scale = frame_info['scale']
                    vehicle = frame_info['vehicle']
                    
                    if proc_frame is None or proc_frame.size == 0:
                        continue
                    
                    plate_conf_factor = float(self.config_manager.get('detection', 'plate_detector_confidence_factor') or 0.6)
                    configured_plate_conf = self.config_manager.get('detection', 'plate_detector_confidence')
                    plate_conf = float(configured_plate_conf) if configured_plate_conf is not None else max(0.05, confidence_threshold * plate_conf_factor)
                    plate_iou = float(self.config_manager.get('detection', 'plate_detector_iou') or 0.45)
                    plate_imgsz = int(self.config_manager.get('detection', 'plate_detector_imgsz') or 960)
                    plate_max_det = int(self.config_manager.get('detection', 'plate_detector_max_det') or max_detections_per_frame or 8)
                    plate_kwargs = {
                        'conf': max(0.01, min(1.0, plate_conf)),
                        'iou': max(0.01, min(1.0, plate_iou)),
                        'imgsz': max(32, plate_imgsz),
                        'max_det': max(1, plate_max_det),
                        'verbose': False,
                    }
                    plate_kwargs.update(self._yolo_runtime_kwargs())
                    license_results = self.license_model.predict(proc_frame, **plate_kwargs)[0]
                    
                    if getattr(license_results, 'boxes', None) is None:
                        continue

                    crop_padding_pct = float(self.config_manager.get('detection', 'plate_crop_padding_percent') or 0.0)
                    proc_h, proc_w = proc_frame.shape[:2]
                    for plate_detection in license_results.boxes.data.tolist():
                        px1, py1, px2, py2, plate_score, _ = plate_detection

                        # Use the raw YOLO box for geometry checks, but crop with
                        # the same small padding as the demo. fast-plate-ocr reads
                        # cropped plates more reliably when borders are not cut off.
                        bw = max(1.0, float(px2) - float(px1))
                        bh = max(1.0, float(py2) - float(py1))
                        pad_x = bw * max(0.0, crop_padding_pct) / 100.0
                        pad_y = bh * max(0.0, crop_padding_pct) / 100.0
                        cpx1 = max(0, int(round(float(px1) - pad_x)))
                        cpy1 = max(0, int(round(float(py1) - pad_y)))
                        cpx2 = min(proc_w, int(round(float(px2) + pad_x)))
                        cpy2 = min(proc_h, int(round(float(py2) + pad_y)))
                        if cpx2 <= cpx1 or cpy2 <= cpy1:
                            continue

                        orig_raw_px1 = int(px1 / scale + off_x)
                        orig_raw_py1 = int(py1 / scale + off_y)
                        orig_raw_px2 = int(px2 / scale + off_x)
                        orig_raw_py2 = int(py2 / scale + off_y)
                        orig_px1 = int(cpx1 / scale + off_x)
                        orig_py1 = int(cpy1 / scale + off_y)
                        orig_px2 = int(cpx2 / scale + off_x)
                        orig_py2 = int(cpy2 / scale + off_y)
                        
                        plate_crop_scaled = proc_frame[cpy1:cpy2, cpx1:cpx2]
                        
                        if plate_crop_scaled.size == 0:
                            continue
                        plate_w = max(1, orig_px2 - orig_px1)
                        plate_h = max(1, orig_py2 - orig_py1)
                        min_plate_w = self.config_manager.get('detection', 'min_plate_width') or 0
                        min_plate_h = self.config_manager.get('detection', 'min_plate_height') or 0
                        if plate_w < min_plate_w or plate_h < min_plate_h:
                            continue
                        aspect = plate_w / plate_h
                        aspect_min = float(self.config_manager.get('detection', 'plate_aspect_ratio_min') or 0)
                        aspect_max = float(self.config_manager.get('detection', 'plate_aspect_ratio_max') or 99)
                        if aspect < aspect_min or aspect > aspect_max:
                            continue
                        if max_detections_per_frame and len(result['detections']) >= max_detections_per_frame:
                            break
                        
                        plate_text, ocr_confidence, ocr_meta = self._read_plate_ocr(plate_crop_scaled)
                        ocr_meta = ocr_meta or {}
                        
                        min_save_conf = self.config_manager.get('history', 'min_confidence_to_save') or 0.35
                        
                        if not plate_text or ocr_confidence < min_save_conf:
                            if annotate_frames:
                                cv2.rectangle(annotated, (orig_px1, orig_py1), (orig_px2, orig_py2), (0, 165, 255), 2)
                            continue
                        
                        duplicate_detection = False
                        if filter_duplicates:
                            duplicate_detection = self._is_duplicate(plate_text)
                            if duplicate_detection:
                                continue
                        
                        if vehicle is None:
                            vehicle = self._match_vehicle_for_plate([orig_px1, orig_py1, orig_px2, orig_py2], detected_vehicles)
                        if vehicle is not None:
                            self._refresh_vehicle_color_excluding_plate(frame, vehicle, [orig_px1, orig_py1, orig_px2, orig_py2])

                        if annotate_frames:
                            cv2.rectangle(annotated, (orig_px1, orig_py1), (orig_px2, orig_py2), (0, 255, 0), 3)
                            country_label = ocr_meta.get('plate_country_display') or ''
                            plate_label = f"{plate_text} ({country_label})" if country_label and country_label != 'Unbekannt' else plate_text
                            label_text = f"{plate_label} {ocr_confidence:.2f}" if draw_confidence else plate_label
                            text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
                            cv2.rectangle(annotated, (orig_px1, orig_py1 - text_size[1] - 15),
                                         (orig_px1 + text_size[0] + 10, orig_py1), (0, 255, 0), -1)
                            cv2.putText(annotated, label_text, (orig_px1 + 5, orig_py1 - 8),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
                        
                        # Bilder speichern
                        plate_image_b64 = None
                        vehicle_image_b64 = None
                        
                        if self.config_manager.get('detection', 'save_detected_plates'):
                            plate_quality = int(self.config_manager.get('storage', 'jpeg_quality_plate') or 95)
                            _, buffer = cv2.imencode('.jpg', plate_crop_scaled, [cv2.IMWRITE_JPEG_QUALITY, plate_quality])
                            plate_image_b64 = base64.b64encode(buffer).decode('utf-8')
                        
                        if self.config_manager.get('detection', 'save_detected_vehicles') and vehicle:
                            vehicle_crop = vehicle.get('crop')
                            if vehicle_crop is not None and vehicle_crop.size > 0:
                                vehicle_quality = int(self.config_manager.get('storage', 'jpeg_quality_vehicle') or 90)
                                _, buffer = cv2.imencode('.jpg', vehicle_crop, [cv2.IMWRITE_JPEG_QUALITY, vehicle_quality])
                                vehicle_image_b64 = base64.b64encode(buffer).decode('utf-8')
                        
                        detection_info = {
                            'plate_text': plate_text,
                            'confidence': ocr_confidence,
                            'plate_bbox': [orig_px1, orig_py1, orig_px2, orig_py2],
                            'plate_center_x': round((orig_px1 + orig_px2) / 2, 2),
                            'plate_center_y': round((orig_py1 + orig_py2) / 2, 2),
                            'frame_width': frame_w,
                            'frame_height': frame_h,
                            'vehicle_bbox': vehicle['bbox'] if vehicle else None,
                            'vehicle_center_x': round((vehicle['bbox'][0] + vehicle['bbox'][2]) / 2, 2) if vehicle else None,
                            'vehicle_center_y': round((vehicle['bbox'][1] + vehicle['bbox'][3]) / 2, 2) if vehicle else None,
                            'plate_score': plate_score,
                            'plate_detector_source': frame_info.get('source'),
                            'plate_bbox_raw': [orig_raw_px1, orig_raw_py1, orig_raw_px2, orig_raw_py2],
                            'duplicate_suppressed': duplicate_detection,
                            'plate_image_base64': plate_image_b64,
                            'vehicle_image_base64': vehicle_image_b64,
                            'vehicle_type': vehicle['type'] if vehicle else 'Unbekannt',
                            'vehicle_type_en': vehicle['type_en'] if vehicle else 'unknown',
                            'vehicle_confidence': vehicle.get('confidence') if vehicle else None,
                            'vehicle_match_reason': vehicle.get('match_reason') if vehicle else None,
                            'vehicle_color': vehicle.get('color') if vehicle else 'Unbekannt',
                            'vehicle_color_hex': vehicle.get('color_hex') if vehicle else None,
                            'vehicle_color_rgb': vehicle.get('color_rgb') if vehicle else None,
                            'vehicle_color_coverage': vehicle.get('color_coverage') if vehicle else None,
                            'vehicle_color_info': vehicle.get('color_info') if vehicle else None,
                            'plate_text_normalized': PlateUtils.normalize(plate_text, compact=True),
                            'plate_format': PlateUtils.detect_format(plate_text),
                            'plate_country': ocr_meta.get('plate_country'),
                            'plate_country_display': ocr_meta.get('plate_country_display'),
                            'plate_country_prob': ocr_meta.get('plate_country_prob'),
                            'plate_region': ocr_meta.get('plate_region'),
                            'plate_region_prob': ocr_meta.get('plate_region_prob'),
                            'ocr_engine': ocr_meta.get('ocr_engine'),
                            'ocr_model': ocr_meta.get('ocr_model'),
                            'ocr_raw_plate': ocr_meta.get('raw_plate'),
                            'ocr_elapsed_ms': ocr_meta.get('ocr_elapsed_ms'),
                            'mean_char_prob_all': ocr_meta.get('mean_char_prob_all'),
                            'mean_char_prob_visible': ocr_meta.get('mean_char_prob_visible'),
                            'is_valid_plate': PlateUtils.is_valid(plate_text),
                            'watchlist_match': watchlist_manager.check(plate_text) if self.config_manager.get('plate_recognition', 'watchlist_enabled') else None,
                        }
                        
                        result['detections'].append(detection_info)
                        logger.info(f"Erkannt: {plate_text} | Konfidenz: {ocr_confidence:.2f}")
            
            result['people'] = self._detect_people(frame, annotated, runtime_roi_polygon=runtime_roi_polygon)
            result['annotated_frame'] = annotated
            
        except Exception as e:
            logger.error(f"Verarbeitungsfehler: {e}")
            import traceback
            traceback.print_exc()
        
        result['processing_time'] = time.time() - start_time
        return result
    
    def process_image(self, image_path_or_array):
        if isinstance(image_path_or_array, str):
            frame = cv2.imread(image_path_or_array)
        else:
            frame = image_path_or_array
        
        if frame is None:
            return None
        
        return self.process_frame(frame)



# ============================================================
# MODELL-PFAD-HILFEN (müssen vor dem initialen Modell-Load existieren)
# ============================================================

def _resolve_model_path(path_value):
    """Resolve model paths stored as models/best.pt, /data/models/best.pt or repo-relative paths."""
    if not path_value:
        return None
    raw = str(path_value).strip().replace('\\', '/')
    if not raw:
        return None
    p = Path(raw).expanduser()
    app_dir = Path(__file__).resolve().parent
    candidates = []
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.extend([Path.cwd() / p, app_dir / p, Path('/app') / p, Path('/data') / p])
        if p.name:
            candidates.extend([Path('/data/models') / p.name, Path('/app/models') / p.name, app_dir / 'models' / p.name, Path.cwd() / 'models' / p.name])
    for candidate in candidates:
        try:
            if candidate.exists():
                return str(candidate)
        except Exception:
            continue
    return raw


def _model_path_exists(path_value):
    resolved = _resolve_model_path(path_value)
    return bool(resolved and Path(resolved).exists())



# ============================================================
# GLOBALE INSTANZEN
# ============================================================

config_manager = ConfigManager()
history_manager = HistoryManager()
watchlist_manager = WatchlistManager()
person_history_manager = PersonHistoryManager()
try:
    person_history_manager.purge_simulation_events(delete_images=True)
except Exception as exc:
    logger.warning(f"Personenanalyse Demo-Daten konnten nicht automatisch bereinigt werden: {exc}")
detector = LicensePlateDetector(config_manager)

# RTSP Handler importieren
from rtsp_handler import RTSPHandler
stream_manager = RTSPHandler(config_manager, history_manager, detector, person_history_manager)

def init_models():
    detector.load_models()

threading.Thread(target=init_models, daemon=True).start()


# ============================================================
# FLASK ROUTEN - SEITEN
# ============================================================

@app.route('/')
def index():
    return render_template('dashboard.html', 
                          page='dashboard',
                          stats=history_manager.get_statistics(),
                          stream_status=stream_manager.get_status(),
                          config=config_manager.config)

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html',
                          page='dashboard',
                          stats=history_manager.get_statistics(),
                          stream_status=stream_manager.get_status(),
                          config=config_manager.config)

@app.route('/history')
def history():
    # Parameter aus URL lesen
    page_num = request.args.get('page', 1, type=int)
    per_page = 20
    search = request.args.get('search', '')
    unique_only = request.args.get('unique', 'false').lower() == 'true'
    source_filter = request.args.get('source', '')
    vehicle_type_filter = request.args.get('vehicle_type', '')
    sort_order = request.args.get('sort', 'newest')

    # Basis-Einträge holen
    if search:
        entries = history_manager.search(search)
    else:
        entries = history_manager.get_all(limit=10000, unique_only=unique_only)

    # ============================================
    # FILTER ANWENDEN
    # ============================================
    
    # Quelle filtern
    if source_filter:
        entries = [e for e in entries if e.get('source', '') == source_filter]

    # Fahrzeugtyp filtern (case-insensitive)
    if vehicle_type_filter:
        entries = [e for e in entries 
                   if e.get('vehicle_type', '').upper() == vehicle_type_filter.upper()]

    # ============================================
    # SORTIERUNG ANWENDEN
    # ============================================
    
    if sort_order == 'oldest':
        entries = sorted(entries, key=lambda x: x.get('timestamp', ''), reverse=False)
    elif sort_order == 'confidence':
        entries = sorted(entries, key=lambda x: x.get('confidence', 0) or 0, reverse=True)
    else:  # 'newest' (default)
        entries = sorted(entries, key=lambda x: x.get('timestamp', ''), reverse=True)

    # ============================================
    # PAGINATION
    # ============================================
    
    total_filtered = len(entries)
    total_all = len(history_manager.history)
    
    # Pagination anwenden
    start_idx = (page_num - 1) * per_page
    end_idx = start_idx + per_page
    paginated_entries = entries[start_idx:end_idx]

    # Seitenzahl berechnen
    total_pages = max(1, (total_filtered + per_page - 1) // per_page)

    return render_template('history.html',
                          page='history',
                          entries=paginated_entries,
                          current_page=page_num,
                          total_pages=total_pages,
                          total_entries=total_all,
                          total_filtered=total_filtered,
                          search=search,
                          unique_only=unique_only,
                          source_filter=source_filter,
                          vehicle_type_filter=vehicle_type_filter,
                          sort_order=sort_order)

@app.route('/search')
def search_page():
    return render_template('search.html',
                          page='search',
                          stream_status=stream_manager.get_status(),
                          config=config_manager.config)

@app.route('/statistics')
def statistics_page():
    return render_template('statistics.html',
                          page='statistics',
                          stream_status=stream_manager.get_status(),
                          config=config_manager.config)

@app.route('/people')
def people_page():
    return render_template('people.html',
                          page='people',
                          stream_status=stream_manager.get_status(),
                          config=config_manager.config)

@app.route('/rtsp-settings')
def rtsp_settings():
    return render_template('rtsp_settings.html',
                          page='rtsp',
                          config=config_manager.config.get('rtsp', {}),
                          stream_status=stream_manager.get_status())

@app.route('/settings')
def settings():
    return render_template('settings.html',
                          page='settings',
                          config=config_manager.config)

@app.route('/test')
def test_page():
    return render_template('test.html', page='test', jobs=video_processing_jobs, config=config_manager.config)

@app.route('/live')
def live_view():
    return render_template('live.html',
                          page='live',
                          stream_status=stream_manager.get_status(),
                          config=config_manager.config)

@app.route('/latest')
def latest_detection_page():
    return render_template('latest.html', page='latest')


# ============================================================
# API ROUTEN - STREAM KONTROLLE
# ============================================================

@app.route('/api/stream/start', methods=['POST'])
def api_stream_start():
    success = stream_manager.start()
    return jsonify({'success': success, 'status': stream_manager.get_status()})

@app.route('/api/stream/stop', methods=['POST'])
def api_stream_stop():
    stream_manager.stop()
    return jsonify({'success': True, 'status': stream_manager.get_status()})

@app.route('/api/stream/status')
def api_stream_status():
    return jsonify(stream_manager.get_status())

@app.route('/api/stream/resolution')
def api_stream_resolution():
    """Gibt die aktuelle Stream-Auflösung zurück"""
    return jsonify(stream_manager.get_stream_resolution())


def _frame_dimensions(frame):
    if frame is None:
        return None
    try:
        h, w = frame.shape[:2]
        return {'width': int(w), 'height': int(h)}
    except Exception:
        return None


def _current_stream_geometry():
    """One geometry source for settings, live view and processing."""
    raw = stream_manager.get_raw_frame()
    annotated = stream_manager.get_current_frame()
    configured = config_manager.get('rtsp', 'resolution') or {}
    active = _frame_dimensions(raw) or _frame_dimensions(annotated)
    if active is None:
        active = {
            'width': int(configured.get('width') or 1280),
            'height': int(configured.get('height') or 720)
        }
    area = config_manager.get('rtsp', 'analysis_area') or {}
    coordinate = {
        'width': int(area.get('coordinate_width') or active.get('width') or configured.get('width') or 1280),
        'height': int(area.get('coordinate_height') or active.get('height') or configured.get('height') or 720)
    }
    return {
        'success': True,
        'active': active,
        'raw': _frame_dimensions(raw),
        'annotated': _frame_dimensions(annotated),
        'configured': {
            'width': int(configured.get('width') or active.get('width') or 1280),
            'height': int(configured.get('height') or active.get('height') or 720)
        },
        'analysis_area_coordinate': coordinate,
        'connected': stream_manager.is_connected(),
        'status': stream_manager.get_status()
    }


@app.route('/api/stream/geometry')
def api_stream_geometry():
    """Returns the exact frame geometry used by RTSP settings and live output."""
    return jsonify(_current_stream_geometry())


@app.route('/api/rtsp/analysis-area', methods=['GET', 'POST'])
def api_rtsp_analysis_area():
    """Canonical endpoint for the one road ROI used everywhere."""
    if request.method == 'GET':
        area = config_manager.get('rtsp', 'analysis_area') or {}
        return jsonify({'success': True, 'analysis_area': area, 'geometry': _current_stream_geometry()})

    try:
        payload = request.get_json(force=True) or {}
        area = payload.get('analysis_area') or payload
        if not isinstance(area, dict):
            return jsonify({'success': False, 'error': 'analysis_area muss ein Objekt sein.'}), 400

        geometry = _current_stream_geometry().get('active') or {}
        coord_w = int(area.get('coordinate_width') or geometry.get('width') or config_manager.get('rtsp', 'resolution', 'width') or 1280)
        coord_h = int(area.get('coordinate_height') or geometry.get('height') or config_manager.get('rtsp', 'resolution', 'height') or 720)
        coord_w = max(1, coord_w)
        coord_h = max(1, coord_h)

        clean_points = []
        for point in area.get('polygon') or []:
            if isinstance(point, dict):
                px, py = point.get('x'), point.get('y')
            elif isinstance(point, (list, tuple)) and len(point) >= 2:
                px, py = point[0], point[1]
            else:
                continue
            try:
                clean_points.append({
                    'x': max(0, min(int(round(float(px))), coord_w - 1)),
                    'y': max(0, min(int(round(float(py))), coord_h - 1)),
                })
            except Exception:
                continue

        if len(clean_points) < 3:
            return jsonify({'success': False, 'error': 'Mindestens 3 Polygonpunkte erforderlich.'}), 400

        xs = [p['x'] for p in clean_points]
        ys = [p['y'] for p in clean_points]
        existing_area = config_manager.get('rtsp', 'analysis_area') or {}
        normalized = {
            'enabled': bool(area.get('enabled', True)),
            'mode': 'polygon',
            'mask_outside': True,
            'coordinate_width': coord_w,
            'coordinate_height': coord_h,
            'polygon': clean_points,
            'area': {
                'x': int(min(xs)),
                'y': int(min(ys)),
                'width': int(max(1, max(xs) - min(xs))),
                'height': int(max(1, max(ys) - min(ys))),
            },
            # Keep RTSP CPU-saver keys when the user only saves the polygon.
            'crop_before_detection': area.get('crop_before_detection', existing_area.get('crop_before_detection', True)),
            'mask_before_detection': area.get('mask_before_detection', existing_area.get('mask_before_detection', False)),
            'crop_padding_percent': area.get('crop_padding_percent', existing_area.get('crop_padding_percent', 25.0)),
            'crop_min_padding_px': area.get('crop_min_padding_px', existing_area.get('crop_min_padding_px', 120)),
            'motion_gate_enabled': area.get('motion_gate_enabled', existing_area.get('motion_gate_enabled', True)),
            'motion_gate_threshold_percent': area.get('motion_gate_threshold_percent', existing_area.get('motion_gate_threshold_percent', 0.20)),
            'motion_gate_hold_seconds': area.get('motion_gate_hold_seconds', existing_area.get('motion_gate_hold_seconds', 2.0)),
            'motion_gate_idle_scan_seconds': area.get('motion_gate_idle_scan_seconds', existing_area.get('motion_gate_idle_scan_seconds', 5.0)),
        }
        config_manager.config.setdefault('rtsp', {})['analysis_area'] = normalized
        config_manager.config['rtsp']['resolution'] = {'width': coord_w, 'height': coord_h}
        config_manager.config = config_manager._normalize_analysis_area(config_manager.config)
        config_manager.save_config()
        return jsonify({'success': True, 'analysis_area': config_manager.config['rtsp']['analysis_area'], 'geometry': _current_stream_geometry()})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/stream/feed')
def stream_feed():
    raw_mode = str(request.args.get('raw', '')).lower() in ('1', 'true', 'yes')
    def generate():
        while True:
            frame = stream_manager.get_raw_frame() if raw_mode else stream_manager.get_current_frame()
            if frame is not None:
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            else:
                placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(placeholder, "Warte auf Stream...", (150, 240),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                _, buffer = cv2.imencode('.jpg', placeholder)
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.033)
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/stream/snapshot')
def api_stream_snapshot():
    """Rohes Einzel-Snapshot vom Stream oder ein kalibrierbares Fallback-Bild.

    Wichtig: Für die RTSP-Kalibrierung wird bewusst das rohe Frame verwendet.
    Dadurch wird nicht zusätzlich der bereits im Livebild eingezeichnete
    Analysebereich angezeigt. Im Editor gibt es nur noch eine sichtbare Zone:
    das Browser-Overlay, das später 1:1 gespeichert und angewendet wird.
    """
    frame = stream_manager.get_raw_frame()
    if frame is None:
        frame = stream_manager.get_current_frame()
    if frame is not None:
        h, w = frame.shape[:2]
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        resp = Response(buffer.tobytes(), mimetype='image/jpeg')
        resp.headers['Cache-Control'] = 'no-store, max-age=0'
        resp.headers['X-Frame-Width'] = str(w)
        resp.headers['X-Frame-Height'] = str(h)
        return resp

    # Fallback in konfigurierter Stream-Auflösung, damit ROI-Linie/Polygon trotzdem sauber passt.
    try:
        res = config_manager.get('rtsp', 'resolution') or {}
        width = int(res.get('width') or 1280)
        height = int(res.get('height') or 720)
    except Exception:
        width, height = 1280, 720
    width = max(320, min(width, 3840))
    height = max(240, min(height, 2160))

    placeholder = np.zeros((height, width, 3), dtype=np.uint8)
    placeholder[:] = (24, 31, 46)
    grid_color = (55, 65, 81)
    for gx in range(0, width, max(80, width // 16)):
        cv2.line(placeholder, (gx, 0), (gx, height), grid_color, 1)
    for gy in range(0, height, max(60, height // 12)):
        cv2.line(placeholder, (0, gy), (width, gy), grid_color, 1)

    road = np.array([
        [int(width * 0.28), int(height * 0.20)],
        [int(width * 0.70), int(height * 0.20)],
        [int(width * 0.92), height],
        [int(width * 0.08), height],
    ], dtype=np.int32)
    overlay = placeholder.copy()
    cv2.fillPoly(overlay, [road], (54, 63, 80))
    placeholder = cv2.addWeighted(overlay, 0.55, placeholder, 0.45, 0)
    cv2.polylines(placeholder, [road], True, (148, 163, 184), 2)

    cv2.putText(placeholder, "RTSP nicht erreichbar - Kalibrierungsbild", (max(20, width // 18), max(50, height // 12)),
                cv2.FONT_HERSHEY_SIMPLEX, max(0.7, width / 1900), (226, 232, 240), 2)
    cv2.putText(placeholder, "Analysebereich/Strasse kann trotzdem eingezeichnet werden", (max(20, width // 18), max(90, height // 12 + 42)),
                cv2.FONT_HERSHEY_SIMPLEX, max(0.55, width / 2500), (34, 211, 238), 2)

    _, buffer = cv2.imencode('.jpg', placeholder, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return Response(buffer.tobytes(), mimetype='image/jpeg')


# ============================================================
# API ROUTEN - KONFIGURATION
# ============================================================

@app.route('/api/config/reset', methods=['POST'])
def api_reset_config():
    config_manager.config = json.loads(json.dumps(config_manager.DEFAULT_CONFIG))
    config_manager.save_config()
    return jsonify({'success': True})

@app.route('/api/config', methods=['GET'])
def api_get_config():
    return jsonify(config_manager.config)

@app.route('/api/config', methods=['POST'])
def api_save_config():
    try:
        data = request.json
        
        def deep_update(d, u):
            for k, v in u.items():
                if isinstance(v, dict):
                    d[k] = deep_update(d.get(k, {}), v)
                else:
                    d[k] = v
            return d
        
        deep_update(config_manager.config, data)
        config_manager.save_config()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/config/rtsp', methods=['POST'])
def api_save_rtsp_config():
    try:
        data = request.json
        
        # Deep merge für analysis_area. coordinate_width/height keep ROI geometry stable
        # even when the camera aspect ratio is not the old 16:9 default.
        if 'analysis_area' in data:
            if 'analysis_area' not in config_manager.config['rtsp']:
                config_manager.config['rtsp']['analysis_area'] = {}
            area_payload = data['analysis_area'] or {}
            for key, value in area_payload.items():
                config_manager.config['rtsp']['analysis_area'][key] = value
            if area_payload.get('coordinate_width') and area_payload.get('coordinate_height'):
                data.setdefault('resolution', {
                    'width': int(area_payload.get('coordinate_width')),
                    'height': int(area_payload.get('coordinate_height'))
                })
            del data['analysis_area']
        
        config_manager.config['rtsp'].update(data)
        config_manager.config = config_manager._normalize_analysis_area(config_manager.config)
        config_manager.save_config()
        
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/config/detection', methods=['POST'])
def api_save_detection_config():
    try:
        data = request.json
        config_manager.config['detection'].update(data)
        config_manager.save_config()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/config/ocr', methods=['POST'])
def api_save_ocr_config():
    try:
        data = request.json
        
        if 'preprocessing' in data:
            if 'preprocessing' not in config_manager.config['ocr']:
                config_manager.config['ocr']['preprocessing'] = {}
            config_manager.config['ocr']['preprocessing'].update(data['preprocessing'])
            del data['preprocessing']
        
        config_manager.config['ocr'].update(data)
        config_manager.save_config()
        
        detector.ocr_reader = None
        detector.fast_plate_recognizer = None
        detector.fast_plate_cache_key = None
        detector.models_loaded = False
        threading.Thread(target=detector.load_models, daemon=True).start()
        
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/config/history', methods=['POST'])
def api_save_history_config():
    try:
        data = request.json
        config_manager.config['history'].update(data)
        config_manager.save_config()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/config/storage', methods=['POST'])
def api_save_storage_config():
    try:
        data = request.get_json(silent=True) or {}
        config_manager.config.setdefault('storage', {})
        config_manager.config['storage'].update(data)
        config_manager.save_config()
        return jsonify({'success': True, 'config': _public_config()})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400



@app.route('/api/config/people', methods=['POST'])
def api_save_people_config():
    try:
        data = request.get_json(silent=True) or {}
        config_manager.config.setdefault('people', {})
        config_manager.config['people'].update(data)
        # Keep model setting in sync for the model overview.
        selected_path = data.get('selected_model_file') or data.get('custom_model_path') or data.get('model_path')
        if selected_path:
            config_manager.config.setdefault('models', {})['person_detector'] = selected_path
            config_manager.config.setdefault('people', {})['model_path'] = selected_path
        config_manager.save_config()
        if data.get('reload'):
            detector.models_loaded = False
            detector.human_model = None
            threading.Thread(target=detector.load_models, daemon=True).start()
        return jsonify({'success': True, 'config': _public_config()})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/config/models', methods=['POST'])
def api_save_models_config():
    try:
        data = request.get_json(silent=True) or {}
        reload_requested = bool(data.pop('reload', False))
        config_manager.config.setdefault('models', {})
        config_manager.config['models'].update(data)
        if data.get('person_detector'):
            config_manager.config.setdefault('people', {})['custom_model_path'] = data.get('person_detector')
            config_manager.config.setdefault('people', {})['model_path'] = data.get('person_detector')
        config_manager.save_config()
        if reload_requested:
            detector.models_loaded = False
            detector.human_model = None
            config_manager.config['models']['last_reload_at'] = datetime.now().isoformat()
            config_manager.save_config()
            threading.Thread(target=detector.load_models, daemon=True).start()
        return jsonify({'success': True, 'config': _public_config(), 'reload': reload_requested})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/config/about', methods=['POST'])
def api_save_about_config():
    try:
        data = request.get_json(silent=True) or {}
        config_manager.config.setdefault('about', {})
        config_manager.config['about'].update(data)
        config_manager.save_config()
        return jsonify({'success': True, 'config': _public_config()})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


# ============================================================
# API ROUTEN - HISTORIE
# ============================================================

@app.route('/api/history')
def api_get_history():
    limit = request.args.get('limit', 100, type=int)
    offset = request.args.get('offset', 0, type=int)
    search = request.args.get('search', '')
    unique_only = request.args.get('unique', 'false').lower() == 'true'
    
    if search:
        entries = history_manager.search(search)
    else:
        entries = history_manager.get_all(limit=limit, offset=offset, unique_only=unique_only)
    
    return jsonify({'entries': entries, 'total': len(history_manager.history)})

@app.route('/api/history/<entry_id>', methods=['GET'])
def api_get_history_entry(entry_id):
    entry = history_manager.get_by_id(entry_id)
    if entry:
        return jsonify(entry)
    return jsonify({'error': 'Nicht gefunden'}), 404

@app.route('/api/history/<entry_id>', methods=['DELETE'])
def api_delete_history_entry(entry_id):
    history_manager.delete_entry(entry_id)
    return jsonify({'success': True})

@app.route('/api/history/clear', methods=['POST'])
def api_clear_history():
    history_manager.clear_history()
    return jsonify({'success': True})

@app.route('/api/history/statistics')
def api_history_statistics():
    return jsonify(history_manager.get_statistics())

@app.route('/api/history/search', methods=['GET', 'POST'])
def api_history_search():
    filters = request.get_json(silent=True) if request.method == 'POST' else request.args.to_dict()
    filters = filters or {}
    for key in ('unique', 'regex', 'fuzzy', 'valid_only', 'watchlist_only'):
        if key in filters:
            filters[key] = str(filters[key]).lower() in ('true', '1', 'yes', 'on')
    return jsonify(history_manager.search_advanced(filters))

@app.route('/api/history/facets')
def api_history_facets():
    return jsonify(history_manager.get_facets())

@app.route('/api/history/export')
def api_history_export():
    fmt = request.args.get('format', 'csv').lower()
    filters = request.args.to_dict()
    filters['limit'] = int(filters.get('limit') or 100000)
    rows = history_manager.search_advanced(filters)['entries']
    filename = f"platevision_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    fields = ['timestamp', 'plate_text', 'plate_text_normalized', 'confidence', 'source', 'vehicle_type', 'vehicle_color', 'vehicle_color_hex', 'plate_country_display', 'plate_country', 'plate_country_prob', 'ocr_engine', 'ocr_model', 'plate_format', 'is_valid_plate', 'filename']
    if fmt == 'json':
        return Response(json.dumps(rows, indent=2, ensure_ascii=False), mimetype='application/json', headers={'Content-Disposition': f'attachment; filename={filename}.json'})
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fields, extrasaction='ignore')
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return Response(output.getvalue(), mimetype='text/csv; charset=utf-8', headers={'Content-Disposition': f'attachment; filename={filename}.csv'})

@app.route('/api/dashboard/overview')
def api_dashboard_overview():
    stats = history_manager.get_statistics()
    latest = history_manager.get_all(limit=12)
    status = stream_manager.get_status()
    return jsonify({
        'statistics': stats,
        'latest': latest,
        'stream': status,
        'facets': history_manager.get_facets(),
        'people': person_history_manager.get_statistics({'days': 1}).get('summary', {}),
        'config': {
            'dashboard': config_manager.get('dashboard'),
            'search': config_manager.get('search'),
            'plate_recognition': config_manager.get('plate_recognition')
        }
    })

@app.route('/api/plate/normalize', methods=['POST'])
def api_plate_normalize():
    data = request.get_json(silent=True) or {}
    plate = data.get('plate_text') or data.get('plate') or ''
    return jsonify({
        'input': plate,
        'normalized': PlateUtils.normalize(plate, compact=True),
        'pretty': PlateUtils.pretty(plate),
        'corrected': PlateUtils.smart_correct(plate, config_manager.get('plate_recognition', 'country_hint') or 'auto'),
        'format': PlateUtils.detect_format(plate),
        'valid': PlateUtils.is_valid(plate)
    })

@app.route('/api/watchlist', methods=['GET', 'POST'])
def api_watchlist():
    if request.method == 'GET':
        return jsonify({'items': watchlist_manager.list(), 'total': len(watchlist_manager.list())})
    data = request.get_json(silent=True) or {}
    try:
        item = watchlist_manager.add(
            data.get('plate_text') or data.get('plate') or '',
            data.get('label', ''),
            data.get('category', 'known'),
            data.get('notes', ''),
            data.get('notify', True)
        )
        return jsonify({'success': True, 'item': item})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/watchlist/<item_id>', methods=['DELETE'])
def api_watchlist_delete(item_id):
    return jsonify({'success': watchlist_manager.delete(item_id)})

@app.route('/api/config/advanced', methods=['POST'])
def api_save_advanced_config():
    try:
        data = request.get_json(silent=True) or {}
        for section in ('general', 'ui', 'privacy', 'plate_recognition', 'search', 'dashboard', 'alerts', 'traffic', 'people', 'recognition_profiles'):
            if section in data:
                config_manager.config.setdefault(section, {}).update(data[section])
        config_manager.save_config()
        return jsonify({'success': True, 'config': config_manager.config})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400



# ============================================================
# API ROUTEN - SPEICHER INFO
# ============================================================

@app.route('/api/storage/info')
def api_storage_info():
    """Gibt Speicherplatz-Informationen zurück"""
    import os
    from pathlib import Path
    
    def count_files(directory):
        """Zählt Dateien in einem Verzeichnis"""
        path = Path(directory)
        if not path.exists():
            return 0
        return len([f for f in path.iterdir() if f.is_file()])
    
    def get_dir_size(directory):
        """Berechnet Größe eines Verzeichnisses in Bytes"""
        total_size = 0
        path = Path(directory)
        if not path.exists():
            return 0
        
        for file in path.rglob('*'):
            if file.is_file():
                try:
                    total_size += file.stat().st_size
                except (OSError, IOError):
                    pass
        return total_size
    
    def format_size(bytes_size):
        """Formatiert Bytes in lesbare Größe"""
        if bytes_size < 1024:
            return f"{bytes_size} B"
        elif bytes_size < 1024 * 1024:
            return f"{bytes_size / 1024:.1f} KB"
        elif bytes_size < 1024 * 1024 * 1024:
            return f"{bytes_size / (1024 * 1024):.1f} MB"
        else:
            return f"{bytes_size / (1024 * 1024 * 1024):.2f} GB"
    
    # Verzeichnisse
    plates_dir = 'data/plates_detected'
    vehicles_dir = 'data/vehicles_detected'
    uploads_dir = 'uploads'
    data_dir = 'data'
    
    # Dateien zählen
    plates_count = count_files(plates_dir)
    vehicles_count = count_files(vehicles_dir)
    
    # Historie-Einträge
    history_count = len(history_manager.history)
    
    # Speicherplatz berechnen
    plates_size = get_dir_size(plates_dir)
    vehicles_size = get_dir_size(vehicles_dir)
    uploads_size = get_dir_size(uploads_dir)
    data_size = get_dir_size(data_dir)
    total_size = plates_size + vehicles_size + uploads_size
    
    return jsonify({
        'plates_count': plates_count,
        'vehicles_count': vehicles_count,
        'history_count': history_count,
        'plates_size': plates_size,
        'plates_size_formatted': format_size(plates_size),
        'vehicles_size': vehicles_size,
        'vehicles_size_formatted': format_size(vehicles_size),
        'uploads_size': uploads_size,
        'uploads_size_formatted': format_size(uploads_size),
        'data_size': data_size,
        'data_size_formatted': format_size(data_size),
        'total_size': total_size,
        'total_size_formatted': format_size(total_size)
    })


@app.route('/api/storage/clear/plates', methods=['POST'])
def api_clear_plates():
    """Löscht alle Kennzeichen-Bilder"""
    import shutil
    from pathlib import Path
    
    plates_dir = Path('data/plates_detected')
    deleted_count = 0
    
    if plates_dir.exists():
        for file in plates_dir.iterdir():
            if file.is_file():
                try:
                    file.unlink()
                    deleted_count += 1
                except Exception as e:
                    logger.warning(f"Konnte Datei nicht löschen: {file} - {e}")
    
    return jsonify({
        'success': True,
        'deleted_count': deleted_count,
        'message': f'{deleted_count} Kennzeichen-Bilder gelöscht'
    })


@app.route('/api/storage/clear/vehicles', methods=['POST'])
def api_clear_vehicles():
    """Löscht alle Fahrzeug-Bilder"""
    from pathlib import Path
    
    vehicles_dir = Path('data/vehicles_detected')
    deleted_count = 0
    
    if vehicles_dir.exists():
        for file in vehicles_dir.iterdir():
            if file.is_file():
                try:
                    file.unlink()
                    deleted_count += 1
                except Exception as e:
                    logger.warning(f"Konnte Datei nicht löschen: {file} - {e}")
    
    return jsonify({
        'success': True,
        'deleted_count': deleted_count,
        'message': f'{deleted_count} Fahrzeug-Bilder gelöscht'
    })


@app.route('/api/storage/clear/uploads', methods=['POST'])
def api_clear_uploads():
    """Löscht alle hochgeladenen und verarbeiteten Dateien"""
    from pathlib import Path
    
    dirs_to_clear = [
        'uploads/images',
        'uploads/videos',
        'uploads/processed'
    ]
    
    deleted_count = 0
    
    for dir_path in dirs_to_clear:
        path = Path(dir_path)
        if path.exists():
            for file in path.iterdir():
                if file.is_file():
                    try:
                        file.unlink()
                        deleted_count += 1
                    except Exception as e:
                        logger.warning(f"Konnte Datei nicht löschen: {file} - {e}")
    
    # Video-Jobs auch leeren
    global video_processing_jobs
    video_processing_jobs = {}
    
    return jsonify({
        'success': True,
        'deleted_count': deleted_count,
        'message': f'{deleted_count} Upload-Dateien gelöscht'
    })


@app.route('/api/storage/clear/all', methods=['POST'])
def api_clear_all_storage():
    """Löscht alle gespeicherten Daten"""
    from pathlib import Path
    
    dirs_to_clear = [
        'data/plates_detected',
        'data/vehicles_detected',
        'uploads/images',
        'uploads/videos',
        'uploads/processed'
    ]
    
    deleted_count = 0
    
    for dir_path in dirs_to_clear:
        path = Path(dir_path)
        if path.exists():
            for file in path.iterdir():
                if file.is_file():
                    try:
                        file.unlink()
                        deleted_count += 1
                    except Exception as e:
                        logger.warning(f"Konnte Datei nicht löschen: {file} - {e}")
    
    # Historie löschen
    history_manager.clear_history()
    
    # Video-Jobs leeren
    global video_processing_jobs
    video_processing_jobs = {}
    
    return jsonify({
        'success': True,
        'deleted_count': deleted_count,
        'message': f'Alle Daten gelöscht ({deleted_count} Dateien + Historie)'
    })


# ============================================================
# API ROUTEN - BILD VERARBEITUNG
# ============================================================

@app.route('/api/process/image', methods=['POST'])
def api_process_image():
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'Keine Datei'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'Keine Datei ausgewählt'}), 400
    
    try:
        file_bytes = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'success': False, 'error': 'Ungültiges Bild'}), 400
        
        logger.info('[TestUpload] Bildanalyse gestartet: filename=%s shape=%s plate_model=%s vehicle_model=%s ocr=%s',
                    file.filename, getattr(image, 'shape', None), config_manager.get('models', 'license_plate_detector'),
                    config_manager.get('models', 'vehicle_detector'), config_manager.get('ocr', 'engine'))
        result = detector.process_frame(image, filter_duplicates=False)
        logger.info('[TestUpload] Ergebnis: plates=%s vehicles=%s people=%s time=%.3fs',
                    len(result.get('detections', []) or []), len(result.get('vehicles', []) or []),
                    len(result.get('people', []) or []), float(result.get('processing_time') or 0))
        
        _, buffer = cv2.imencode('.jpg', result['annotated_frame'])
        result_image_b64 = base64.b64encode(buffer).decode('utf-8')
        
        for detection in result['detections']:
            if detection.get('plate_text'):
                entry = {
                    'plate_text': detection['plate_text'],
                    'confidence': detection.get('confidence', 0),
                    'source': 'image_upload',
                    'filename': file.filename,
                    'plate_image': detection.get('plate_image_base64'),
                    'vehicle_image': detection.get('vehicle_image_base64'),
                    'vehicle_type': detection.get('vehicle_type', 'Unbekannt'),
                    'vehicle_type_en': detection.get('vehicle_type_en', 'unknown'),
                    'vehicle_color': detection.get('vehicle_color', 'Unbekannt'),
                    'vehicle_color_hex': detection.get('vehicle_color_hex'),
                    'vehicle_color_rgb': detection.get('vehicle_color_rgb'),
                    'vehicle_color_coverage': detection.get('vehicle_color_coverage'),
                    'plate_country': detection.get('plate_country'),
                    'plate_country_display': detection.get('plate_country_display'),
                    'plate_country_prob': detection.get('plate_country_prob'),
                    'ocr_engine': detection.get('ocr_engine'),
                    'ocr_model': detection.get('ocr_model'),
                }
                history_manager.add_entry(entry)

        for person in result.get('people', []):
            save_person_event = bool(person.get('counted') or config_manager.get('people', 'save_all_detections') or config_manager.get('people', 'image_history_enabled'))
            if save_person_event:
                event = dict(person)
                event.update({'source': 'image_upload', 'filename': file.filename})
                saved_person_event = person_history_manager.add_event(event, frame=image, annotated_frame=result.get('annotated_frame'))
                if saved_person_event:
                    person.update({k: saved_person_event.get(k) for k in ('id', 'counted', 'event_type', 'repeat_blocked', 'repeat_block_minutes', 'repeat_match_id', 'images', 'note') if k in saved_person_event})
        
        _attach_person_crop_previews(result.get('people', []), image)
        safe_detections = _json_safe(result.get('detections', []))
        safe_vehicles = _json_safe([{k: v for k, v in v.items() if k != 'crop'} for v in result.get('vehicles', [])])
        safe_people = _json_safe(result.get('people', []))

        return jsonify({
            'success': True,
            'result_image': result_image_b64,
            'detections': safe_detections,
            'vehicles': safe_vehicles,
            'people': safe_people,
            'people_counted': sum(1 for p in result.get('people', []) if p.get('counted')),
            'processing_time': _json_safe(result.get('processing_time', 0))
        })
        
    except Exception as e:
        logger.exception(f"Bildverarbeitung Fehler: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


# ============================================================
# API ROUTEN - LETZTE ERKENNUNG
# ============================================================

@app.route('/api/latest')
def api_latest_detection():
    entries = history_manager.get_all(limit=1)
    if entries:
        return jsonify(entries[0])
    return jsonify({'error': 'Keine Erkennung vorhanden', 'plate_text': None})

@app.route('/api/latest/plate')
def api_latest_plate_text():
    entries = history_manager.get_all(limit=1)
    if entries:
        entry = entries[0]
        return jsonify({
            'plate_text': entry.get('plate_text', ''),
            'confidence': entry.get('confidence', 0),
            'vehicle_type': entry.get('vehicle_type', 'unknown'),
            'vehicle_color': entry.get('vehicle_color', 'unknown'),
            'vehicle_color_hex': entry.get('vehicle_color_hex'),
            'vehicle_color_rgb': entry.get('vehicle_color_rgb'),
            'plate_country': entry.get('plate_country'),
            'plate_country_display': entry.get('plate_country_display'),
            'plate_country_prob': entry.get('plate_country_prob'),
            'timestamp': entry.get('timestamp', ''),
            'source': entry.get('source', '')
        })
    return jsonify({'plate_text': '', 'confidence': 0})

@app.route('/api/latest/plate/image')
def api_latest_plate_image():
    entries = history_manager.get_all(limit=1)
    if entries:
        plate_image = entries[0].get('plate_image')
        if plate_image:
            try:
                return Response(base64.b64decode(plate_image), mimetype='image/jpeg')
            except:
                pass
    
    img = np.zeros((100, 400, 3), dtype=np.uint8)
    img[:] = (30, 30, 30)
    cv2.putText(img, "Kein Kennzeichen", (100, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)
    _, buffer = cv2.imencode('.jpg', img)
    return Response(buffer.tobytes(), mimetype='image/jpeg')

# ============================================================
# API ROUTEN - FAHRZEUG (LETZTE ERKENNUNG)
# ============================================================

@app.route('/api/latest/vehicle')
@app.route('/api/latest/vehicle/')
def api_latest_vehicle():
    """Gibt die letzte Fahrzeug-Erkennung als JSON zurück"""
    entries = history_manager.get_all(limit=1)
    if entries:
        entry = entries[0]
        return jsonify({
            'plate_text': entry.get('plate_text', ''),
            'confidence': entry.get('confidence', 0),
            'vehicle_type': entry.get('vehicle_type', 'unknown'),
            'vehicle_type_en': entry.get('vehicle_type_en', 'unknown'),
            'vehicle_color': entry.get('vehicle_color', 'unknown'),
            'vehicle_color_hex': entry.get('vehicle_color_hex'),
            'vehicle_color_rgb': entry.get('vehicle_color_rgb'),
            'plate_country': entry.get('plate_country'),
            'plate_country_display': entry.get('plate_country_display'),
            'plate_country_prob': entry.get('plate_country_prob'),
            'timestamp': entry.get('timestamp', ''),
            'source': entry.get('source', ''),
            'has_vehicle_image': entry.get('vehicle_image') is not None,
            'has_plate_image': entry.get('plate_image') is not None
        })
    return jsonify({
        'error': 'Keine Erkennung vorhanden',
        'plate_text': None,
        'vehicle_type': None
    })


@app.route('/api/latest/vehicle/image')
@app.route('/api/latest/vehicle/image/')
def api_latest_vehicle_image():
    """Gibt das letzte Fahrzeug-Bild als JPEG zurück"""
    entries = history_manager.get_all(limit=1)
    if entries:
        vehicle_image = entries[0].get('vehicle_image')
        if vehicle_image:
            try:
                return Response(base64.b64decode(vehicle_image), mimetype='image/jpeg')
            except:
                pass
    
    # Placeholder-Bild
    img = np.zeros((200, 400, 3), dtype=np.uint8)
    img[:] = (30, 30, 30)
    cv2.putText(img, "Kein Fahrzeug", (120, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)
    cv2.putText(img, "vorhanden", (140, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)
    _, buffer = cv2.imencode('.jpg', img)
    return Response(buffer.tobytes(), mimetype='image/jpeg')


@app.route('/api/latest/full')
@app.route('/api/latest/full/')
def api_latest_full():
    """Gibt alle Daten der letzten Erkennung inkl. Base64-Bilder zurück"""
    entries = history_manager.get_all(limit=1)
    if entries:
        entry = entries[0]
        return jsonify({
            'success': True,
            'plate_text': entry.get('plate_text', ''),
            'confidence': entry.get('confidence', 0),
            'vehicle_type': entry.get('vehicle_type', 'unknown'),
            'vehicle_type_en': entry.get('vehicle_type_en', 'unknown'),
            'vehicle_color': entry.get('vehicle_color', 'unknown'),
            'vehicle_color_hex': entry.get('vehicle_color_hex'),
            'vehicle_color_rgb': entry.get('vehicle_color_rgb'),
            'plate_country': entry.get('plate_country'),
            'plate_country_display': entry.get('plate_country_display'),
            'plate_country_prob': entry.get('plate_country_prob'),
            'timestamp': entry.get('timestamp', ''),
            'source': entry.get('source', ''),
            'plate_image_base64': entry.get('plate_image', None),
            'vehicle_image_base64': entry.get('vehicle_image', None),
            'id': entry.get('id', '')
        })
    return jsonify({
        'success': False,
        'error': 'Keine Erkennung vorhanden'
    })


@app.route('/api/latest/image')
@app.route('/api/latest/image/')
def api_latest_full_image():
    """Gibt das volle Bild (Fahrzeug mit Kennzeichen) der letzten Erkennung zurück"""
    entries = history_manager.get_all(limit=1)
    if entries:
        # Zuerst Fahrzeugbild versuchen, dann Kennzeichenbild
        vehicle_image = entries[0].get('vehicle_image')
        plate_image = entries[0].get('plate_image')
        
        image_data = vehicle_image or plate_image
        if image_data:
            try:
                return Response(base64.b64decode(image_data), mimetype='image/jpeg')
            except:
                pass
    
    # Placeholder
    img = np.zeros((200, 400, 3), dtype=np.uint8)
    img[:] = (30, 30, 30)
    cv2.putText(img, "Kein Bild", (130, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (100, 100, 100), 2)
    _, buffer = cv2.imencode('.jpg', img)
    return Response(buffer.tobytes(), mimetype='image/jpeg')



# ============================================================
# API ROUTEN - UTILITIES
# ============================================================


# ============================================================
# MODEL DISCOVERY & SETTINGS VALIDATION
# ============================================================

def _safe_relpath(path_value):
    try:
        p = Path(str(path_value))
        if not p.is_absolute():
            return str(p).replace('\\', '/')
        return str(p.relative_to(Path.cwd())).replace('\\', '/')
    except Exception:
        return str(path_value or '').replace('\\', '/')

def _guess_model_kind(path_value):
    name = Path(str(path_value)).name.lower()
    if any(token in name for token in ('human', 'person', 'people', 'pedestrian')):
        return 'people'
    if any(token in name for token in ('plate', 'license', 'kennzeichen')):
        return 'plate'
    if any(token in name for token in ('yolov8', 'coco', 'vehicle', 'car')):
        return 'vehicle/general'
    return 'custom'

def _candidate_model_directories(*extra_values):
    """Return model directories that can exist in dev, Docker and HA runtime."""
    cfg_models = config_manager.get('models') or {}
    cfg_people = config_manager.get('people') or {}
    app_dir = Path(__file__).resolve().parent
    raw_candidates = [
        'models',
        '/data/models',
        '/app/models',
        str(app_dir / 'models'),
        str(app_dir.parent / 'src' / 'models'),
        'platevision/src/models',
        cfg_models.get('custom_model_directory'),
        cfg_models.get('model_upload_directory'),
        *(cfg_models.get('additional_model_directories') or []),
        os.path.dirname(cfg_models.get('person_detector') or ''),
        os.path.dirname(cfg_people.get('selected_model_file') or ''),
        os.path.dirname(cfg_people.get('custom_model_path') or ''),
        os.path.dirname(cfg_people.get('model_path') or ''),
        *[os.path.dirname(str(v)) for v in extra_values if v],
    ]
    dirs, seen = [], set()
    for raw in raw_candidates:
        if not raw:
            continue
        p = Path(str(raw)).expanduser()
        candidates = [p] if p.is_absolute() else [Path.cwd() / p, app_dir / p, Path('/app') / p]
        # If a repo-relative path is configured, also try the runtime /app/models and /data/models by basename.
        if 'models' in p.parts:
            candidates.extend([Path('/data/models'), Path('/app/models'), app_dir / 'models'])
        for c in candidates:
            try:
                key = str(c.resolve()) if c.exists() else str(c)
            except Exception:
                key = str(c)
            if key not in seen:
                seen.add(key)
                dirs.append(c)
    return dirs


def _resolve_model_path(path_value):
    """Resolve model paths stored as models/best.pt, /data/models/best.pt or repo-relative paths."""
    if not path_value:
        return None
    raw = str(path_value).strip().replace('\\', '/')
    if not raw:
        return None
    p = Path(raw).expanduser()
    app_dir = Path(__file__).resolve().parent
    candidates = []
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.extend([
            Path.cwd() / p,
            app_dir / p,
            Path('/app') / p,
            Path('/data') / p,
        ])
        # Common user/GitHub layout: platevision/src/models/best.pt. At runtime this becomes /app/models/best.pt or /data/models/best.pt.
        if p.name:
            candidates.extend([Path('/data/models') / p.name, Path('/app/models') / p.name, app_dir / 'models' / p.name, Path.cwd() / 'models' / p.name])
    for candidate in candidates:
        try:
            if candidate.exists():
                return str(candidate)
        except Exception:
            continue
    return raw


def _model_path_exists(path_value):
    resolved = _resolve_model_path(path_value)
    return bool(resolved and Path(resolved).exists())


def _bool_from_request(value, default=False):
    if value is None:
        return bool(default)
    return str(value).strip().lower() in ('1', 'true', 'yes', 'on', 'ja', 'y')


def _json_safe(value, _seen=None):
    """Convert detector results into plain JSON-safe values.

    The live detector keeps OpenCV crops as numpy arrays and some nested helper
    dicts for debugging. Flask cannot serialize numpy objects, and a bug in the
    first 0.8.23 build created a color palette self-reference. This helper also
    protects future API responses against circular references.
    """
    if _seen is None:
        _seen = set()

    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, np.ndarray):
        return None

    if isinstance(value, (datetime,)):
        return value.isoformat()

    if isinstance(value, dict):
        obj_id = id(value)
        if obj_id in _seen:
            return None
        _seen.add(obj_id)
        out = {}
        for k, v in value.items():
            key = str(k)
            if key in {'crop', 'annotated_frame'}:
                continue
            out[key] = _json_safe(v, _seen)
        _seen.remove(obj_id)
        return out

    if isinstance(value, (list, tuple, set)):
        obj_id = id(value)
        if obj_id in _seen:
            return None
        _seen.add(obj_id)
        out = [_json_safe(v, _seen) for v in value]
        _seen.remove(obj_id)
        return out

    return str(value)


def _allowed_model_extensions():
    cfg = config_manager.get('models') or {}
    configured = cfg.get('person_model_extensions') or ['.pt', '.onnx', '.engine']
    return {str(ext).lower() if str(ext).startswith('.') else f'.{str(ext).lower()}' for ext in configured}


def _model_upload_directory():
    cfg = config_manager.get('models') or {}
    preferred = Path(str(cfg.get('model_upload_directory') or '/data/models')).expanduser()
    fallback = Path.cwd() / 'models'
    for candidate in (preferred, Path('/data/models'), fallback):
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            if os.access(candidate, os.W_OK):
                return candidate
        except Exception:
            continue
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


def _model_info_from_path(path_value, selected_for_people=False):
    p = Path(path_value)
    try:
        resolved = str(p.resolve())
        stat = p.stat()
    except Exception:
        resolved = str(p)
        stat = None
    try:
        rel = str(p.resolve().relative_to(Path.cwd().resolve())).replace('\\', '/')
    except Exception:
        try:
            rel = str(p.resolve().relative_to(Path(__file__).resolve().parent)).replace('\\', '/')
        except Exception:
            rel = str(p).replace('\\', '/')
    return {
        'path': rel,
        'resolved_path': resolved,
        'name': p.name,
        'directory': _safe_relpath(p.parent),
        'extension': p.suffix.lower(),
        'size_mb': round((stat.st_size if stat else 0) / 1024 / 1024, 2),
        'modified_at': datetime.fromtimestamp(stat.st_mtime).isoformat() if stat else None,
        'kind_guess': _guess_model_kind(p),
        'exists': p.exists(),
        'selected_for_people': bool(selected_for_people)
    }


def _safe_image_upload(file_storage):
    if not file_storage or not getattr(file_storage, 'filename', ''):
        return None, 'Keine Datei ausgewählt'
    filename = secure_filename(file_storage.filename)
    suffix = Path(filename).suffix.lower()
    if suffix not in {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}:
        return None, 'Nur Bilddateien JPG, PNG, WEBP oder BMP sind erlaubt'
    data = file_storage.read()
    if not data:
        return None, 'Leere Datei'
    arr = np.frombuffer(data, np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        return None, 'Ungültiges oder beschädigtes Bild'
    return {'filename': filename, 'data': data, 'image': image, 'suffix': suffix}, None



def _attach_person_crop_previews(people, frame, quality=88):
    """Attach JPEG/base64 previews containing only detected persons.

    This is used by Test & Upload and people snapshot APIs so the UI can show
    person crops instead of the full camera frame. Stored history images still
    use PersonHistoryManager, but this helper gives an immediate preview even
    before/without history persistence.
    """
    if frame is None or not people:
        return people
    try:
        h, w = frame.shape[:2]
    except Exception:
        return people
    for person in people:
        try:
            bbox = person.get('bbox') or []
            if len(bbox) != 4:
                continue
            x1, y1, x2, y2 = [int(float(v)) for v in bbox]
            pad = 8
            try:
                pad = int((config_manager.get('people') or {}).get('image_history_crop_padding_px') or 8)
            except Exception:
                pass
            x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
            x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
            if x2 <= x1 or y2 <= y1:
                continue
            crop = frame[y1:y2, x1:x2].copy()
            ok, buffer = cv2.imencode('.jpg', crop, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
            if ok:
                b64 = base64.b64encode(buffer).decode('utf-8')
                person['person_image_base64'] = b64
                person['crop_image_base64'] = b64
        except Exception as exc:
            logger.debug(f'Personen-Crop Preview konnte nicht erstellt werden: {exc}')
    return people


def _people_config_with_overrides(overrides=None):
    """Return a deep-copied people configuration with optional temporary overrides."""
    base = json.loads(json.dumps(config_manager.get('people') or {}, ensure_ascii=False))
    if isinstance(overrides, dict):
        _deep_update(base, overrides)
    return base


def _people_cfg_from_preview_args(args):
    """Build safe people preview overrides from query parameters."""
    def as_bool(value, default=False):
        return _bool_from_request(value, default)
    def as_float(value, default):
        try:
            return float(value)
        except Exception:
            return default
    cfg = {}
    if 'enabled' in args:
        cfg['enabled'] = as_bool(args.get('enabled'))
    if 'draw_boxes' in args:
        cfg['draw_boxes'] = as_bool(args.get('draw_boxes'), True)
    if 'line_enabled' in args:
        cfg['line_crossing_enabled'] = as_bool(args.get('line_enabled'), True)
    if 'line_percent' in args:
        cfg['virtual_line_position_percent'] = max(1, min(99, as_float(args.get('line_percent'), 50)))
    if 'movement_axis' in args:
        cfg['movement_axis'] = 'x' if str(args.get('movement_axis')).lower() == 'x' else 'y'
    if 'crossing_direction' in args:
        cfg['crossing_direction'] = str(args.get('crossing_direction') or 'both')
    if 'count_strategy' in args:
        cfg['count_strategy'] = str(args.get('count_strategy') or 'line_crossing')
    if 'confidence' in args:
        cfg['confidence_threshold'] = max(0, min(1, as_float(args.get('confidence'), 0.45)))
    if 'roi_filter_enabled' in args:
        cfg['roi_filter_enabled'] = as_bool(args.get('roi_filter_enabled'), True)
    if 'roi_filter_mode' in args:
        mode = str(args.get('roi_filter_mode') or 'foot_and_center')
        cfg['roi_filter_mode'] = mode if mode in ('foot_and_center', 'foot', 'center', 'center_or_foot') else 'foot_and_center'
    if 'line_relative_to_roi' in args:
        cfg['line_relative_to_roi'] = as_bool(args.get('line_relative_to_roi'), True)
    if 'zone_enabled' in args:
        cfg['zone_enabled'] = as_bool(args.get('zone_enabled'))
    zone = {}
    for key, target in [('zone_x','x'), ('zone_y','y'), ('zone_w','width'), ('zone_h','height')]:
        if key in args:
            zone[target] = max(0, min(100, as_float(args.get(key), 0 if target in ('x','y') else 100)))
    if zone:
        existing = cfg.get('zone') if isinstance(cfg.get('zone'), dict) else {}
        existing.update(zone)
        existing['unit'] = 'percent'
        cfg['zone'] = existing
    return cfg


def _create_people_preview_fallback(width=1280, height=720, message=None):
    """Create a readable fallback frame when no RTSP frame is available."""
    width = int(width or 1280)
    height = int(height or 720)
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    # Soft dark gradient
    for y in range(height):
        shade = int(18 + (y / max(1, height - 1)) * 32)
        frame[y, :] = (shade, max(10, shade - 8), max(18, shade - 2))
    # Grid and entrance hint
    grid_color = (65, 75, 95)
    for x in range(0, width, max(80, width // 12)):
        cv2.line(frame, (x, 0), (x, height), grid_color, 1)
    for y in range(0, height, max(60, height // 10)):
        cv2.line(frame, (0, y), (width, y), grid_color, 1)
    # Simple corridor/path visual
    pts = np.array([[int(width*0.18), height], [int(width*0.43), int(height*0.35)], [int(width*0.57), int(height*0.35)], [int(width*0.82), height]], np.int32)
    overlay = frame.copy()
    cv2.fillPoly(overlay, [pts], (45, 60, 90))
    frame = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)
    label = message or (config_manager.get('people', 'settings_preview_fallback_label') or 'RTSP Stream nicht erreichbar - Kalibrierungsbild')
    cv2.putText(frame, 'PlateVision Personen-Kalibrierung', (40, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (241, 245, 249), 2)
    cv2.putText(frame, label, (40, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (148, 163, 184), 2)
    cv2.putText(frame, 'Linie, Zone und Richtung werden trotzdem anhand deiner Einstellungen angezeigt.', (40, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (148, 163, 184), 2)
    return frame



def _scaled_road_roi_for_frame(width, height):
    """Return the unified RTSP road ROI scaled to the given frame size."""
    try:
        cfg = config_manager.get('rtsp', 'analysis_area') or {}
        if not cfg.get('enabled'):
            return {'enabled': False, 'polygon': [], 'x': 0, 'y': 0, 'width': int(width), 'height': int(height)}

        def safe_int(value, fallback):
            try:
                return int(round(float(value)))
            except Exception:
                return fallback

        coord_w = max(1, safe_int(cfg.get('coordinate_width'), width))
        coord_h = max(1, safe_int(cfg.get('coordinate_height'), height))
        sx = float(width) / coord_w
        sy = float(height) / coord_h
        points = []
        for point in (cfg.get('polygon') or []):
            try:
                if isinstance(point, dict):
                    px, py = point.get('x', 0), point.get('y', 0)
                else:
                    px, py = point[0], point[1]
                points.append([
                    max(0, min(int(round(float(px) * sx)), int(width) - 1)),
                    max(0, min(int(round(float(py) * sy)), int(height) - 1)),
                ])
            except Exception:
                continue
        if len(points) < 3:
            area = cfg.get('area') or {}
            x = max(0, min(int(round(float(area.get('x', 0)) * sx)), int(width) - 1))
            y = max(0, min(int(round(float(area.get('y', 0)) * sy)), int(height) - 1))
            w = max(10, min(int(round(float(area.get('width', width)) * sx)), int(width) - x))
            h = max(10, min(int(round(float(area.get('height', height)) * sy)), int(height) - y))
            points = [[x, y], [min(int(width)-1, x+w), y], [min(int(width)-1, x+w), min(int(height)-1, y+h)], [x, min(int(height)-1, y+h)]]
        xs = [p[0] for p in points]; ys = [p[1] for p in points]
        x1, y1 = min(xs), min(ys)
        x2, y2 = max(xs), max(ys)
        return {'enabled': True, 'polygon': points, 'x': x1, 'y': y1, 'width': max(10, x2-x1), 'height': max(10, y2-y1), 'coordinate_width': coord_w, 'coordinate_height': coord_h}
    except Exception:
        return {'enabled': False, 'polygon': [], 'x': 0, 'y': 0, 'width': int(width), 'height': int(height)}


def _draw_unified_road_roi_on_image(img, area_info):
    """Draw the same road ROI that is used by the live stream."""
    if not area_info or not area_info.get('enabled') or len(area_info.get('polygon') or []) < 3:
        return img
    pts = np.array(area_info['polygon'], dtype=np.int32)
    overlay = img.copy()
    cv2.fillPoly(overlay, [pts], (0, 255, 255))
    img[:] = cv2.addWeighted(overlay, 0.14, img, 0.86, 0)
    cv2.polylines(img, [pts], True, (0, 255, 255), 3)
    label_x = max(8, int(min(p[0] for p in area_info['polygon'])) + 8)
    label_y = max(28, int(min(p[1] for p in area_info['polygon'])) + 28)
    cv2.putText(img, 'Analysebereich Strasse', (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 255, 255), 2)
    return img


def _people_line_bounds_for_area(width, height, cfg, area_info=None):
    """Return the count line coordinates using the same basis as live processing."""
    axis = str(cfg.get('movement_axis') or 'y').lower()
    line_percent = max(1, min(99, float(cfg.get('virtual_line_position_percent') or 50)))
    use_roi = cfg.get('line_relative_to_roi') is not False and area_info and area_info.get('enabled')
    if use_roi:
        ax = int(area_info.get('x') or 0); ay = int(area_info.get('y') or 0)
        aw = int(area_info.get('width') or width); ah = int(area_info.get('height') or height)
    else:
        ax, ay, aw, ah = 0, 0, int(width), int(height)
    if axis == 'x':
        x = int(ax + aw * line_percent / 100.0)
        return {'axis': 'x', 'x': x, 'y1': ay, 'y2': min(int(height)-1, ay + ah), 'line_percent': line_percent, 'relative_to_roi': use_roi}
    y = int(ay + ah * line_percent / 100.0)
    return {'axis': 'y', 'y': y, 'x1': ax, 'x2': min(int(width)-1, ax + aw), 'line_percent': line_percent, 'relative_to_roi': use_roi}

def _draw_people_calibration_overlay(frame, cfg=None, source_label='Live/RTSP'):
    """Draw line/zone/counting settings on a frame for settings and test previews."""
    if frame is None:
        frame = _create_people_preview_fallback()
    cfg = cfg or (config_manager.get('people') or {})
    img = frame.copy()
    h, w = img.shape[:2]
    road_roi = _scaled_road_roi_for_frame(w, h)
    img = _draw_unified_road_roi_on_image(img, road_roi)
    overlay = img.copy()
    # Person zone
    zone = cfg.get('zone') or {}
    zone_enabled = bool(cfg.get('zone_enabled'))
    if zone_enabled:
        zx = float(zone.get('x') or 0); zy = float(zone.get('y') or 0)
        zw = float(zone.get('width') or 100); zh = float(zone.get('height') or 100)
        if (zone.get('unit') or 'percent') == 'percent':
            x1 = int(w * zx / 100); y1 = int(h * zy / 100)
            x2 = int(w * min(100, zx + zw) / 100); y2 = int(h * min(100, zy + zh) / 100)
        else:
            x1, y1, x2, y2 = int(zx), int(zy), int(zx + zw), int(zy + zh)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w-1, x2), min(h-1, y2)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (99, 102, 241), -1)
        img = cv2.addWeighted(overlay, 0.16, img, 0.84, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), (129, 140, 248), 3)
        cv2.putText(img, 'Personen-Zone', (x1 + 10, max(28, y1 + 28)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (199, 210, 254), 2)
    # Count line: same basis as live processing. By default the line is relative to the road ROI.
    line_enabled = cfg.get('line_crossing_enabled') is not False
    line_color = (34, 211, 238)
    direction = str(cfg.get('crossing_direction') or 'both')
    if line_enabled:
        lb = _people_line_bounds_for_area(w, h, cfg, road_roi)
        basis_label = 'Strasse' if lb.get('relative_to_roi') else 'Bild'
        if lb['axis'] == 'x':
            x = int(lb['x'])
            cv2.line(img, (x, int(lb['y1'])), (x, int(lb['y2'])), line_color, 4)
            cv2.putText(img, f'Zaehllinie X={lb["line_percent"]:.0f}% / {direction} / {basis_label}', (min(w-680, max(20, x + 12)), max(42, int(lb['y1']) + 34)), cv2.FONT_HERSHEY_SIMPLEX, 0.68, line_color, 2)
            mid_y = int((int(lb['y1']) + int(lb['y2'])) / 2)
            cv2.arrowedLine(img, (max(20, x - 130), mid_y), (max(20, x - 30), mid_y), (16,185,129), 3, tipLength=0.25)
            cv2.arrowedLine(img, (min(w-20, x + 130), min(h-20, mid_y + 55)), (min(w-20, x + 30), min(h-20, mid_y + 55)), (245,158,11), 3, tipLength=0.25)
        else:
            y = int(lb['y'])
            cv2.line(img, (int(lb['x1']), y), (int(lb['x2']), y), line_color, 4)
            cv2.putText(img, f'Zaehllinie Y={lb["line_percent"]:.0f}% / {direction} / {basis_label}', (max(20, int(lb['x1']) + 10), max(36, y - 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.68, line_color, 2)
            mid_x = int((int(lb['x1']) + int(lb['x2'])) / 2)
            cv2.arrowedLine(img, (max(20, mid_x - 80), max(20, y - 110)), (max(20, mid_x - 80), max(20, y - 25)), (16,185,129), 3, tipLength=0.25)
            cv2.arrowedLine(img, (min(w-20, mid_x + 60), min(h-20, y + 110)), (min(w-20, mid_x + 60), min(h-20, y + 25)), (245,158,11), 3, tipLength=0.25)
    # Status strip
    strip_h = 86
    strip = img.copy()
    cv2.rectangle(strip, (0, h-strip_h), (w, h), (15, 23, 42), -1)
    img = cv2.addWeighted(strip, 0.72, img, 0.28, 0)
    status = 'AKTIV' if cfg.get('enabled') else 'INAKTIV'
    model = cfg.get('selected_model_file') or cfg.get('custom_model_path') or cfg.get('model_path') or 'COCO Person-Klasse'
    text1 = f'Personenanalyse: {status} | Strategie: {cfg.get("count_strategy") or "line_crossing"} | Konfidenz: {float(cfg.get("confidence_threshold") or 0):.2f}'
    text2 = f'Modell: {Path(str(model)).name} | Quelle: {source_label}'
    cv2.putText(img, text1, (24, h-48), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (241,245,249), 2)
    cv2.putText(img, text2, (24, h-18), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (148,163,184), 2)
    return img


def _people_settings_summary(cfg=None):
    cfg = cfg or (config_manager.get('people') or {})
    zone = cfg.get('zone') or {}
    return {
        'enabled': bool(cfg.get('enabled')),
        'history_enabled': cfg.get('history_enabled') is not False,
        'model_mode': cfg.get('model_mode') or 'coco_person',
        'selected_model': cfg.get('selected_model_file') or cfg.get('custom_model_path') or cfg.get('model_path') or 'COCO Person-Klasse',
        'confidence_threshold': cfg.get('confidence_threshold'),
        'count_strategy': cfg.get('count_strategy'),
        'line_crossing_enabled': cfg.get('line_crossing_enabled') is not False,
        'roi_filter_enabled': cfg.get('roi_filter_enabled') is not False,
        'roi_filter_mode': cfg.get('roi_filter_mode') or 'foot_and_center',
        'line_relative_to_roi': cfg.get('line_relative_to_roi') is not False,
        'min_roi_overlap_percent': cfg.get('min_roi_overlap_percent'),
        'virtual_line_position_percent': cfg.get('virtual_line_position_percent'),
        'movement_axis': cfg.get('movement_axis'),
        'crossing_direction': cfg.get('crossing_direction'),
        'zone_enabled': bool(cfg.get('zone_enabled')),
        'zone': zone,
        'session_gap_minutes': cfg.get('session_gap_minutes'),
        'present_timeout_minutes': cfg.get('present_timeout_minutes'),
        'save_all_detections': bool(cfg.get('save_all_detections')),
        'test_environment_enabled': bool(cfg.get('test_environment_enabled')),
        'save_person_crops': bool(cfg.get('save_person_crops')),
        'save_full_frame': bool(cfg.get('save_full_frame')),
        'privacy_blur_people': bool(cfg.get('privacy_blur_people')),
        'export_default_format': cfg.get('export_default_format'),
        'retention_days': cfg.get('retention_days'),
        'auto_cleanup_enabled': bool(cfg.get('auto_cleanup_enabled')),
        'simulation_enabled': bool(cfg.get('simulation_enabled')),
        'alert_threshold_per_hour': cfg.get('alert_threshold_per_hour')
    }

def _scan_model_files():
    cfg_models = config_manager.get('models') or {}
    cfg_people = config_manager.get('people') or {}
    extensions = cfg_models.get('person_model_extensions') or ['.pt', '.onnx', '.engine']
    extensions = {str(ext).lower() if str(ext).startswith('.') else f'.{str(ext).lower()}' for ext in extensions}
    directories = _candidate_model_directories(
        cfg_models.get('person_detector'),
        cfg_people.get('selected_model_file'),
        cfg_people.get('custom_model_path'),
        cfg_people.get('model_path')
    )
    found = []
    seen = set()
    selected_values = {
        cfg_people.get('selected_model_file'),
        cfg_people.get('custom_model_path'),
        cfg_people.get('model_path'),
        cfg_models.get('person_detector')
    }
    selected_resolved = {_resolve_model_path(v) for v in selected_values if v}

    for directory in directories:
        try:
            d = Path(directory)
            if not d.exists() or not d.is_dir():
                continue
            for p in sorted(d.rglob('*')):
                if not p.is_file() or p.suffix.lower() not in extensions:
                    continue
                resolved = str(p.resolve())
                if resolved in seen:
                    continue
                seen.add(resolved)
                stat = p.stat()
                # Prefer a runtime-safe relative path when the model is under the active working directory.
                try:
                    rel = str(p.resolve().relative_to(Path.cwd().resolve())).replace('\\', '/')
                except Exception:
                    try:
                        rel = str(p.resolve().relative_to(Path(__file__).resolve().parent)).replace('\\', '/')
                    except Exception:
                        rel = str(p).replace('\\', '/')
                kind = _guess_model_kind(p)
                # best.pt and last.pt from a HumanDetection repo are intentionally usable as people models.
                if p.name.lower() in ('best.pt', 'last.pt') and 'models' in [part.lower() for part in p.parts]:
                    kind = 'people'
                found.append({
                    'path': rel,
                    'resolved_path': resolved,
                    'name': p.name,
                    'directory': _safe_relpath(p.parent),
                    'extension': p.suffix.lower(),
                    'size_mb': round(stat.st_size / 1024 / 1024, 2),
                    'modified_at': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    'kind_guess': kind,
                    'exists': True,
                    'selected_for_people': rel in selected_values or resolved in selected_resolved or str(p) in selected_values
                })
        except Exception as e:
            logger.warning(f'Modellscan fehlgeschlagen für {directory}: {e}')

    # Add expected standard HumanDetection names as clear missing hints instead of silently hiding them.
    for expected in (cfg_people.get('fallback_model_files') or ['models/best.pt', 'models/last.pt', 'models/human_best.pt']):
        resolved = _resolve_model_path(expected)
        if resolved and Path(resolved).exists():
            continue
        name = Path(str(expected)).name
        if name and name.lower() in ('best.pt', 'last.pt', 'human_best.pt') and not any(m['name'] == name for m in found):
            found.append({
                'path': str(expected).replace('\\', '/'),
                'resolved_path': resolved or str(expected),
                'name': name,
                'directory': str(Path(str(expected)).parent).replace('\\', '/'),
                'extension': Path(str(expected)).suffix.lower(),
                'size_mb': 0,
                'modified_at': None,
                'kind_guess': 'people',
                'exists': False,
                'missing_hint': 'Datei nicht gefunden. Beim Add-on-Start werden neue Modelle aus /app/models nach /data/models synchronisiert.',
                'selected_for_people': expected in selected_values
            })

    found.sort(key=lambda m: (not m.get('exists'), 0 if m.get('kind_guess') == 'people' else 1, m.get('name','').lower()))
    config_manager.config.setdefault('models', {})['last_model_scan_at'] = datetime.now().isoformat()
    config_manager.config.setdefault('models', {})['last_model_scan_count'] = len([m for m in found if m.get('exists')])
    return found

def _validate_current_config():
    checks = []
    def add(key, ok, message, level='ok'):
        checks.append({'key': key, 'ok': bool(ok), 'level': level if ok else 'warning', 'message': message})
    people = config_manager.get('people') or {}
    models = config_manager.get('models') or {}
    add('people_enabled', True, 'Personenmodul ist aktivierbar; aktueller Status: ' + ('aktiv' if people.get('enabled') else 'inaktiv'))
    mode = people.get('model_mode') or 'coco_person'
    selected = people.get('selected_model_file') or people.get('custom_model_path') or models.get('person_detector')
    add('people_model_mode', mode in ('coco_person', 'custom_human', 'custom_path', 'model_file'), f'Personen-Modus: {mode}')
    if mode == 'coco_person':
        add('people_model_path', _model_path_exists(models.get('vehicle_detector') or ''), 'COCO-Personenklasse nutzt das Fahrzeug/YOLOv8-Modell.')
    else:
        add('people_model_path', bool(selected and _model_path_exists(selected)), f'Ausgewähltes Personenmodell: {selected}')
    add('line_position', 1 <= float(people.get('virtual_line_position_percent') or 50) <= 99, 'Virtuelle Zähllinie liegt zwischen 1% und 99%.')
    add('confidence', 0.01 <= float(people.get('confidence_threshold') or 0.45) <= 0.99, 'Personen-Konfidenz liegt im gültigen Bereich.')
    add('history_enabled', people.get('history_enabled') is not False, 'Personen-Historie ist gespeichert, wenn aktiviert.')
    return {'success': True, 'checks': checks, 'warnings': [c for c in checks if not c['ok']]}

@app.route('/api/models/available')
def api_models_available():
    return jsonify({'success': True, 'models': _scan_model_files(), 'config': config_manager.get('models') or {}})


@app.route('/api/models/upload', methods=['POST'])
def api_models_upload():
    """Upload custom YOLO/ONNX/TensorRT models from the settings UI.
    Files are saved to /data/models when possible so they survive add-on updates.
    """
    try:
        models_cfg = config_manager.get('models') or {}
        if models_cfg.get('model_upload_enabled') is False:
            return jsonify({'success': False, 'error': 'Modell-Upload ist in den Einstellungen deaktiviert.'}), 403
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'Keine Modelldatei erhalten.'}), 400
        upload = request.files['file']
        original_name = secure_filename(upload.filename or '')
        if not original_name:
            return jsonify({'success': False, 'error': 'Keine Datei ausgewählt.'}), 400
        ext = Path(original_name).suffix.lower()
        allowed = _allowed_model_extensions()
        if ext not in allowed:
            return jsonify({'success': False, 'error': f'Ungültiger Dateityp {ext}. Erlaubt: {", ".join(sorted(allowed))}'}), 400

        role = (request.form.get('role') or 'people').strip().lower()
        select_after_upload = _bool_from_request(request.form.get('select_after_upload'), models_cfg.get('model_upload_select_after_upload', True))
        reload_after_upload = _bool_from_request(request.form.get('reload'), False)
        overwrite = _bool_from_request(request.form.get('overwrite'), models_cfg.get('model_upload_allow_overwrite', False))
        max_mb = float(models_cfg.get('model_upload_max_mb') or 500)
        content_length = request.content_length or 0
        if max_mb > 0 and content_length and content_length > max_mb * 1024 * 1024:
            return jsonify({'success': False, 'error': f'Datei ist größer als das Limit von {max_mb:g} MB.'}), 413

        target_dir = _model_upload_directory()
        target = target_dir / original_name
        if target.exists() and not overwrite:
            target = target_dir / f"{target.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{target.suffix}"
        upload.save(str(target))

        info = _model_info_from_path(target, selected_for_people=(role == 'people' and select_after_upload))
        cfg_models = config_manager.config.setdefault('models', {})
        cfg_models['last_uploaded_model'] = info
        cfg_models['last_model_upload_at'] = datetime.now().isoformat()
        cfg_models['model_upload_directory'] = str(target_dir)

        if role in ('people', 'person', 'human') and select_after_upload:
            cfg_people = config_manager.config.setdefault('people', {})
            cfg_people['model_mode'] = 'model_file'
            cfg_people['selected_model_file'] = info['path']
            cfg_people['custom_model_path'] = info['path']
            cfg_people['model_path'] = info['path']
            cfg_models['person_detector'] = info['path']
        elif role in ('vehicle', 'car', 'coco') and select_after_upload:
            cfg_models['vehicle_detector'] = info['path']
        elif role in ('plate', 'license', 'kennzeichen') and select_after_upload:
            cfg_models['license_plate_detector'] = info['path']

        config_manager.save_config()
        if reload_after_upload:
            detector.models_loaded = False
            detector.human_model = None
            cfg_models['last_reload_at'] = datetime.now().isoformat()
            config_manager.save_config()
            threading.Thread(target=detector.load_models, daemon=True).start()

        return jsonify({
            'success': True,
            'model': info,
            'selected': select_after_upload,
            'role': role,
            'reload': reload_after_upload,
            'models': _scan_model_files(),
            'config': _public_config()
        })
    except Exception as e:
        logger.error(f'Modell-Upload Fehler: {e}')
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/models/people/options')
def api_people_model_options():
    models = _scan_model_files()
    recommended = [m for m in models if m.get('kind_guess') == 'people']
    return jsonify({
        'success': True,
        'models': models,
        'recommended': recommended,
        'current': config_manager.get('people') or {},
        'status': {
            'human_model_loaded': detector.human_model is not None,
            'coco_model_loaded': detector.coco_model is not None,
            'models_loaded': detector.models_loaded
        }
    })

@app.route('/api/models/people/select', methods=['POST'])
def api_people_model_select():
    try:
        data = request.get_json(silent=True) or {}
        path = data.get('path') or data.get('selected_model_file')
        mode = data.get('model_mode') or ('model_file' if path else 'coco_person')
        resolved_path = _resolve_model_path(path) if path else None
        if path and not _model_path_exists(path):
            return jsonify({'success': False, 'error': f'Modell nicht gefunden: {path}. Gefundene Ordner: models/, /data/models, /app/models'}), 400
        config_manager.config.setdefault('people', {})['model_mode'] = mode
        if path:
            config_manager.config['people']['selected_model_file'] = path
            config_manager.config['people']['custom_model_path'] = path
            config_manager.config['people']['model_path'] = path
            config_manager.config.setdefault('models', {})['person_detector'] = path
        config_manager.save_config()
        if data.get('reload', True):
            detector.models_loaded = False
            detector.human_model = None
            threading.Thread(target=detector.load_models, daemon=True).start()
        return jsonify({'success': True, 'config': _public_config(), 'resolved_path': resolved_path})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/config/validate')
def api_config_validate():
    return jsonify(_validate_current_config())

@app.route('/api/models/status')
def api_models_status():
    return jsonify({
        'loaded': detector.models_loaded,
        'coco_model': detector.coco_model is not None,
        'license_model': detector.license_model is not None,
        'human_model': detector.human_model is not None,
        'person_detection_enabled': bool(config_manager.get('people', 'enabled')),
        'person_model_mode': config_manager.get('people', 'model_mode'),
        'ocr_reader': detector.ocr_reader is not None,
        'fast_plate_ocr': detector.fast_plate_recognizer is not None,
        'ocr_engine': config_manager.get('ocr', 'engine') or 'fast_plate_ocr'
    })

@app.route('/api/models/reload', methods=['POST'])
def api_reload_models():
    detector.models_loaded = False
    detector.coco_model = None
    detector.license_model = None
    detector.human_model = None
    detector.ocr_reader = None
    detector.fast_plate_recognizer = None
    detector.fast_plate_cache_key = None
    threading.Thread(target=detector.load_models, daemon=True).start()
    return jsonify({'success': True, 'message': 'Modelle werden neu geladen...'})

@app.route('/api/system/info')
def api_system_info():
    import platform
    return jsonify({
        'platform': platform.system(),
        'python_version': platform.python_version(),
        'opencv_version': cv2.__version__,
        'models_loaded': detector.models_loaded,
        'stream_status': stream_manager.get_status(),
        'history_count': len(history_manager.history),
        'people_history_count': len(person_history_manager.history)
    })


# ============================================================
# API ROUTEN - VIDEO VERARBEITUNG
# ============================================================

def process_video_job(job_id, video_path, original_filename):
    """Hintergrund-Thread für Video-Verarbeitung"""
    global video_processing_jobs
    
    try:
        job = video_processing_jobs[job_id]
        job['status'] = 'processing'
        job['started_at'] = datetime.now().isoformat()
        
        # Video öffnen
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception("Video konnte nicht geöffnet werden")
        
        # Video-Eigenschaften
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        job['total_frames'] = total_frames
        job['fps'] = fps
        job['resolution'] = f"{width}x{height}"
        
        # Output Video Writer
        output_path = f"uploads/processed/{job_id}_output.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Verarbeitungs-Einstellungen
        process_every_n_frames = max(1, int(fps / 2))  # ~2 FPS Analyse
        
        frame_count = 0
        detections_count = 0
        start_time = time.time()
        all_detections = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            annotated_frame = frame.copy()
            
            # Nur jeden n-ten Frame analysieren
            if frame_count % process_every_n_frames == 0:
                try:
                    result = detector.process_frame(frame)
                    annotated_frame = result['annotated_frame']
                    
                    # Neue Erkennungen speichern
                    for detection in result['detections']:
                        if detection.get('plate_text'):
                            detections_count += 1
                            
                            entry = {
                                'plate_text': detection['plate_text'],
                                'confidence': detection.get('confidence', 0),
                                'source': 'video_upload',
                                'filename': original_filename,
                                'frame_number': frame_count,
                                'timestamp_video': frame_count / fps,
                                'plate_image': detection.get('plate_image_base64'),
                                'vehicle_image': detection.get('vehicle_image_base64'),
                                'vehicle_type': detection.get('vehicle_type', 'Unbekannt'),
                                'vehicle_type_en': detection.get('vehicle_type_en', 'unknown'),
                                'vehicle_color': detection.get('vehicle_color', 'Unbekannt'),
                                'vehicle_color_hex': detection.get('vehicle_color_hex'),
                                'vehicle_color_rgb': detection.get('vehicle_color_rgb'),
                                'vehicle_color_coverage': detection.get('vehicle_color_coverage'),
                                'plate_country': detection.get('plate_country'),
                                'plate_country_display': detection.get('plate_country_display'),
                                'plate_country_prob': detection.get('plate_country_prob'),
                                'ocr_engine': detection.get('ocr_engine'),
                                'ocr_model': detection.get('ocr_model'),
                            }
                            history_manager.add_entry(entry)
                            all_detections.append(entry)
                            
                except Exception as e:
                    logger.error(f"Frame-Verarbeitung Fehler: {e}")
            
            # Frame schreiben
            out.write(annotated_frame)
            
            # Progress Update
            elapsed = time.time() - start_time
            progress = int((frame_count / total_frames) * 100) if total_frames > 0 else 0
            fps_processing = round(frame_count / elapsed, 1) if elapsed > 0 else 0
            eta = int((total_frames - frame_count) / fps_processing) if fps_processing > 0 else 0
            
            job['current_frame'] = frame_count
            job['progress'] = progress
            job['detections_count'] = detections_count
            job['elapsed_time'] = int(elapsed)
            job['fps_processing'] = fps_processing
            job['eta'] = eta
            
            # WebSocket Progress (alle 30 Frames)
            if frame_count % 30 == 0:
                try:
                    socketio.emit('video_progress', {
                        'job_id': job_id,
                        'progress': progress,
                        'current_frame': frame_count,
                        'total_frames': total_frames,
                        'detections_count': detections_count,
                        'elapsed_time': int(elapsed),
                        'fps_processing': fps_processing,
                        'eta': eta,
                        'status': 'processing'
                    })
                except:
                    pass
        
        # Aufräumen
        cap.release()
        out.release()
        
        # Job abschließen
        job['status'] = 'completed'
        job['completed_at'] = datetime.now().isoformat()
        job['progress'] = 100
        job['current_frame'] = total_frames
        job['output_path'] = output_path
        job['all_detections'] = all_detections
        
        # WebSocket Completion
        try:
            socketio.emit('video_completed', {
                'job_id': job_id,
                'status': 'completed',
                'detections_count': detections_count,
                'total_frames': total_frames,
                'processing_time': int(time.time() - start_time)
            })
        except:
            pass
        
        logger.info(f"Video-Job {job_id} abgeschlossen: {detections_count} Erkennungen in {total_frames} Frames")
        
    except Exception as e:
        logger.error(f"Video-Verarbeitung Fehler: {e}")
        import traceback
        traceback.print_exc()
        
        video_processing_jobs[job_id]['status'] = 'error'
        video_processing_jobs[job_id]['error'] = str(e)
        
        try:
            socketio.emit('video_completed', {
                'job_id': job_id,
                'status': 'error',
                'error': str(e)
            })
        except:
            pass


@app.route('/api/process/video', methods=['POST'])
def api_process_video():
    """Video-Datei hochladen und Verarbeitung starten"""
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'Keine Datei'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'Keine Datei ausgewählt'}), 400
    
    # Dateiendung prüfen
    allowed_extensions = {'mp4', 'avi', 'mkv', 'mov', 'wmv', 'flv', 'webm'}
    file_ext = file.filename.rsplit('.', 1)[-1].lower() if '.' in file.filename else ''
    
    if file_ext not in allowed_extensions:
        return jsonify({'success': False, 'error': f'Ungültiges Videoformat. Erlaubt: {", ".join(allowed_extensions)}'}), 400
    
    try:
        # Job erstellen
        job_id = str(uuid.uuid4())
        
        # Video speichern
        video_filename = f"{job_id}.{file_ext}"
        video_path = os.path.join('uploads/videos', video_filename)
        file.save(video_path)
        
        # Job registrieren
        video_processing_jobs[job_id] = {
            'id': job_id,
            'filename': file.filename,
            'video_path': video_path,
            'status': 'queued',
            'progress': 0,
            'current_frame': 0,
            'total_frames': 0,
            'detections_count': 0,
            'created_at': datetime.now().isoformat(),
            'error': None
        }
        
        # Verarbeitung im Hintergrund starten
        thread = threading.Thread(
            target=process_video_job,
            args=(job_id, video_path, file.filename),
            daemon=True
        )
        thread.start()
        
        logger.info(f"Video-Job gestartet: {job_id} - {file.filename}")
        
        return jsonify({
            'success': True,
            'job_id': job_id,
            'message': 'Video-Verarbeitung gestartet'
        })
        
    except Exception as e:
        logger.error(f"Video-Upload Fehler: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/process/video/<job_id>/status')
def api_video_job_status(job_id):
    """Status eines Video-Jobs abrufen"""
    if job_id not in video_processing_jobs:
        return jsonify({'error': 'Job nicht gefunden'}), 404
    
    job = video_processing_jobs[job_id]
    
    return jsonify({
        'job_id': job_id,
        'status': job.get('status'),
        'progress': job.get('progress', 0),
        'current_frame': job.get('current_frame', 0),
        'total_frames': job.get('total_frames', 0),
        'detections_count': job.get('detections_count', 0),
        'elapsed_time': job.get('elapsed_time', 0),
        'fps_processing': job.get('fps_processing', 0),
        'eta': job.get('eta', 0),
        'error': job.get('error'),
        'filename': job.get('filename'),
        'created_at': job.get('created_at'),
        'completed_at': job.get('completed_at')
    })


@app.route('/api/process/video/<job_id>/output')
def api_video_job_output(job_id):
    """Verarbeitetes Video herunterladen"""
    if job_id not in video_processing_jobs:
        return jsonify({'error': 'Job nicht gefunden'}), 404
    
    job = video_processing_jobs[job_id]
    
    if job.get('status') != 'completed':
        return jsonify({'error': 'Video noch nicht fertig'}), 400
    
    output_path = job.get('output_path')
    if not output_path or not os.path.exists(output_path):
        return jsonify({'error': 'Output-Datei nicht gefunden'}), 404
    
    directory = os.path.dirname(output_path)
    filename = os.path.basename(output_path)
    
    # Originaler Dateiname für Download
    original_name = job.get('filename', 'video')
    if '.' in original_name:
        original_name = original_name.rsplit('.', 1)[0]
    download_name = f"{original_name}_processed.mp4"
    
    return send_from_directory(
        directory,
        filename,
        as_attachment=True,
        download_name=download_name
    )


@app.route('/api/process/jobs')
def api_get_jobs():
    """Alle Video-Jobs abrufen"""
    # Jobs als Dictionary zurückgeben (so erwartet es das Frontend)
    return jsonify(video_processing_jobs)


@app.route('/api/process/jobs/<job_id>', methods=['DELETE'])
def api_delete_job(job_id):
    """Video-Job löschen"""
    if job_id not in video_processing_jobs:
        return jsonify({'error': 'Job nicht gefunden'}), 404
    
    job = video_processing_jobs[job_id]
    
    # Dateien löschen
    try:
        video_path = job.get('video_path')
        if video_path and os.path.exists(video_path):
            os.remove(video_path)
        
        output_path = job.get('output_path')
        if output_path and os.path.exists(output_path):
            os.remove(output_path)
    except Exception as e:
        logger.warning(f"Fehler beim Löschen der Job-Dateien: {e}")
    
    # Job entfernen
    del video_processing_jobs[job_id]
    
    return jsonify({'success': True})





@app.route('/api/system/about')
def api_system_about():
    return jsonify({
        'name': 'PlateVision',
        'version': '0.8.23',
        'edition': 'FastPlateOCR + YOLO Vehicle Intelligence',
        'features': [
            'RTSP Live Stream', 'Einheitlicher Straßen-ROI', 'Fahrzeugerkennung', 'Kennzeichenerkennung',
            'Statistik', 'Suche', 'Watchlist', 'Mehrsprachige Einstellungen',
            'Erweiterte Original-Einstellungen', 'Personenzählung und Personenanalyse'
        ],
        'config': config_manager.get('about') or {},
        'models_loaded': detector.models_loaded,
        'history_count': len(history_manager.history),
        'people_history_count': len(person_history_manager.history)
    })


# ============================================================
# API ROUTEN - PRO ERWEITERUNGEN, SPRACHE, EINSTELLUNGEN
# ============================================================

TRANSLATIONS = {
    'de': {
        'dashboard': 'Dashboard', 'history': 'Historie', 'search': 'Suche & Analyse',
        'statistics': 'Statistik', 'people': 'Personenanalyse', 'settings': 'Einstellungen', 'diagnostics': 'Diagnose', 'live': 'Live Stream',
        'latest': 'Letzte Erkennung', 'test': 'Test & Upload', 'stream': 'Stream',
        'plates': 'Kennzeichen', 'confidence': 'Konfidenz', 'watchlist': 'Watchlist', 'traffic': 'Verkehr', 'arrived': 'Gekommen', 'departed': 'Gegangen'
    },
    'en': {
        'dashboard': 'Dashboard', 'history': 'History', 'search': 'Search & Analysis',
        'statistics': 'Statistics', 'people': 'People analysis', 'settings': 'Settings', 'diagnostics': 'Diagnostics', 'live': 'Live Stream',
        'latest': 'Latest Detection', 'test': 'Test & Upload', 'stream': 'Stream',
        'plates': 'License plates', 'confidence': 'Confidence', 'watchlist': 'Watchlist', 'traffic': 'Traffic', 'arrived': 'Arrived', 'departed': 'Departed'
    },
    'fr': {
        'dashboard': 'Tableau de bord', 'history': 'Historique', 'search': 'Recherche & Analyse',
        'statistics': 'Statistiques', 'people': 'Analyse personnes', 'settings': 'Paramètres', 'diagnostics': 'Diagnostic', 'live': 'Flux en direct',
        'latest': 'Dernière détection', 'test': 'Test & Upload', 'stream': 'Flux',
        'plates': 'Plaques', 'confidence': 'Confiance', 'watchlist': 'Liste', 'traffic': 'Trafic', 'arrived': 'Arrivé', 'departed': 'Parti'
    },
    'it': {
        'dashboard': 'Dashboard', 'history': 'Cronologia', 'search': 'Ricerca & Analisi',
        'statistics': 'Statistiche', 'people': 'Analisi persone', 'settings': 'Impostazioni', 'diagnostics': 'Diagnostica', 'live': 'Live Stream',
        'latest': 'Ultimo rilevamento', 'test': 'Test & Upload', 'stream': 'Stream',
        'plates': 'Targhe', 'confidence': 'Confidenza', 'watchlist': 'Watchlist', 'traffic': 'Traffico', 'arrived': 'Arrivato', 'departed': 'Uscito'
    }
}




@app.context_processor
def inject_i18n_helpers():
    lang = config_manager.get('general', 'language') or 'de'
    translations = TRANSLATIONS.get(lang, TRANSLATIONS['de'])
    def t(key):
        return translations.get(key, TRANSLATIONS['de'].get(key, key))
    return {'lang': lang, 't': t, 'translations': translations}


def _deep_update(target, updates):
    for key, value in (updates or {}).items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value
    return target


def _public_config():
    cfg = json.loads(json.dumps(config_manager.config, ensure_ascii=False))
    rtsp = cfg.get('rtsp', {})
    if rtsp.get('url'):
        rtsp['url_masked'] = re.sub(r'//([^:/@]+):([^@]+)@', r'//\1:***@', rtsp['url'])
    return cfg


def _setting_schema():
    return {
        'general': {
            'title': 'Allgemein & Sprache',
            'fields': {
                'language': {'type': 'select', 'options': ['de', 'en', 'fr', 'it'], 'label': 'Sprache'},
                'theme': {'type': 'select', 'options': ['dark', 'light', 'auto'], 'label': 'Design'},
                'timezone': {'type': 'text', 'label': 'Zeitzone'},
                'max_history_entries': {'type': 'number', 'label': 'Max. Historie'},
                'notification_enabled': {'type': 'boolean', 'label': 'Benachrichtigungen'},
                'debug_mode': {'type': 'boolean', 'label': 'Debug-Modus'}
            }
        },
        'ui': {
            'title': 'Layout & Oberfläche',
            'fields': {
                'accent_color': {'type': 'color', 'label': 'Akzentfarbe'},
                'density': {'type': 'select', 'options': ['compact', 'comfortable', 'spacious'], 'label': 'Dichte'},
                'animations': {'type': 'boolean', 'label': 'Animationen'},
                'sidebar_labels': {'type': 'boolean', 'label': 'Sidebar-Text'},
                'card_style': {'type': 'select', 'options': ['flat', 'glass', 'bordered'], 'label': 'Kartenstil'},
                'show_help_text': {'type': 'boolean', 'label': 'Hilfetexte'}
            }
        },
        'plate_recognition': {
            'title': 'Kennzeichen-Erkennung',
            'fields': {
                'country_hint': {'type': 'select', 'options': ['auto', 'CH', 'FL', 'DE', 'AT', 'FR', 'IT'], 'label': 'Länder-Hinweis'},
                'min_length': {'type': 'number', 'label': 'Min. Länge'},
                'max_length': {'type': 'number', 'label': 'Max. Länge'},
                'validation_regex': {'type': 'text', 'label': 'Regex-Validierung'},
                'smart_ocr_correction': {'type': 'boolean', 'label': 'Intelligente OCR-Korrektur'},
                'format_pretty_output': {'type': 'boolean', 'label': 'Schöne Ausgabe'},
                'watchlist_enabled': {'type': 'boolean', 'label': 'Watchlist prüfen'}
            }
        },
        'search': {
            'title': 'Suche',
            'fields': {
                'default_limit': {'type': 'number', 'label': 'Standard-Limit'},
                'enable_fuzzy_search': {'type': 'boolean', 'label': 'Fuzzy-Suche'},
                'fuzzy_similarity': {'type': 'number', 'step': 0.01, 'label': 'Fuzzy-Schwelle'},
                'allow_regex_search': {'type': 'boolean', 'label': 'Regex erlauben'},
                'remember_last_filters': {'type': 'boolean', 'label': 'Filter merken'}
            }
        },
        'traffic': {
            'title': 'Verkehrsstatistik & Sessions',
            'fields': {
                'visit_gap_minutes': {'type': 'number', 'label': 'Besuch trennen nach Minuten'},
                'active_timeout_minutes': {'type': 'number', 'label': 'Als gegangen nach Minuten ohne Erkennung'},
                'daily_count_mode': {'type': 'select', 'options': ['visits', 'detections', 'unique_vehicles'], 'label': 'Tageszählung'},
                'direction_mode': {'type': 'select', 'options': ['auto', 'explicit', 'spatial', 'timeout'], 'label': 'Kommen/Gehen-Modus'},
                'min_confidence': {'type': 'number', 'step': 0.01, 'label': 'Min. Konfidenz für Statistik'},
                'ignore_unknown_plates': {'type': 'boolean', 'label': 'Unbekannte Kennzeichen ignorieren'},
                'include_duplicate_events': {'type': 'boolean', 'label': 'Duplikat-Events mitzählen'},
                'movement_axis': {'type': 'select', 'options': ['x', 'y'], 'label': 'Bewegungsachse'},
                'movement_threshold_percent': {'type': 'number', 'label': 'Bewegungs-Schwelle in %'}
            }
        },
        'privacy': {
            'title': 'Datenschutz & Aufbewahrung',
            'fields': {
                'mask_plate_numbers': {'type': 'boolean', 'label': 'Kennzeichen maskieren'},
                'blur_plate_images': {'type': 'boolean', 'label': 'Kennzeichenbilder weichzeichnen'},
                'retention_days': {'type': 'number', 'label': 'Aufbewahrung in Tagen (0 = unbegrenzt)'},
                'export_include_images': {'type': 'boolean', 'label': 'Bilder in Exporten erlauben'}
            }
        },
        'storage': {
            'title': 'Speicher',
            'fields': {
                'jpeg_quality_plate': {'type': 'number', 'label': 'JPEG Qualität Kennzeichen'},
                'jpeg_quality_vehicle': {'type': 'number', 'label': 'JPEG Qualität Fahrzeuge'},
                'auto_cleanup_images': {'type': 'boolean', 'label': 'Bilder automatisch bereinigen'},
                'cleanup_images_days': {'type': 'number', 'label': 'Bild-Aufbewahrung Tage'},
                'max_storage_mb': {'type': 'number', 'label': 'Max. Speicher MB'}
            }
        },
        'detection': {
            'title': 'YOLO Kennzeichen & Fahrzeuge',
            'fields': {
                'confidence_threshold': {'type': 'number', 'step': 0.01, 'label': 'Fahrzeug YOLO Confidence'},
                'plate_detector_confidence': {'type': 'number', 'step': 0.01, 'label': 'Kennzeichen YOLO Confidence'},
                'plate_detector_iou': {'type': 'number', 'step': 0.01, 'label': 'Kennzeichen YOLO IoU'},
                'plate_detector_imgsz': {'type': 'number', 'label': 'Kennzeichen YOLO Bildgröße'},
                'plate_crop_padding_percent': {'type': 'number', 'step': 0.5, 'label': 'Kennzeichen Crop Padding %'},
                'scan_full_frame_when_vehicle_found': {'type': 'boolean', 'label': 'Immer ganzes Bild nach Kennzeichen scannen'},
                'zoom_enabled': {'type': 'boolean', 'label': 'Zusätzlicher Fahrzeug-Zoom Scan'},
                'min_plate_width': {'type': 'number', 'label': 'Min. Kennzeichenbreite'},
                'min_plate_height': {'type': 'number', 'label': 'Min. Kennzeichenhöhe'}
            }
        },
        'ocr': {
            'title': 'OCR Engine',
            'fields': {
                'engine': {'type': 'select', 'options': ['fast_plate_ocr', 'easyocr'], 'label': 'OCR Engine'},
                'fast_plate_model': {'type': 'select', 'options': ['cct-s-v2-global-model', 'cct-xs-v2-global-model', 'cct-s-v1-global-model', 'cct-xs-v1-global-model'], 'label': 'fast-plate-ocr Modell'},
                'fast_plate_device': {'type': 'select', 'options': ['auto', 'cpu', 'cuda'], 'label': 'fast-plate-ocr Gerät'},
                'easyocr_backup_enabled': {'type': 'boolean', 'label': 'EasyOCR als Backup verwenden'},
                'min_confidence': {'type': 'number', 'step': 0.01, 'label': 'OCR Mindest-Konfidenz'},
                'fast_plate_remove_pad_char': {'type': 'boolean', 'label': 'Padding-Zeichen entfernen'}
            }
        },
        'models': {
            'title': 'Modelle',
            'fields': {
                'device': {'type': 'select', 'options': ['auto', 'cpu', 'cuda', 'mps'], 'label': 'Gerät'},
                'half_precision': {'type': 'boolean', 'label': 'FP16'},
                'auto_reload_on_change': {'type': 'boolean', 'label': 'Auto Reload'},
                'fallback_to_cpu': {'type': 'boolean', 'label': 'CPU-Fallback'}
            }
        },
        'about': {
            'title': 'Über',
            'fields': {
                'release_channel': {'type': 'select', 'options': ['stable', 'beta', 'dev'], 'label': 'Release Kanal'},
                'support_url': {'type': 'text', 'label': 'Support URL'},
                'documentation_url': {'type': 'text', 'label': 'Dokumentation URL'}
            }
        }
    }


@app.route('/diagnostics')
def diagnostics_page():
    return render_template('diagnostics.html', page='diagnostics', stream_status=stream_manager.get_status(), config=config_manager.config)


@app.route('/api/i18n')
def api_i18n():
    lang = request.args.get('lang') or config_manager.get('general', 'language') or 'de'
    return jsonify({'language': lang, 'available': list(TRANSLATIONS.keys()), 'translations': TRANSLATIONS.get(lang, TRANSLATIONS['de'])})



@app.route('/api/test/settings', methods=['GET', 'POST'])
def api_test_settings():
    """Central recognition profile used by Test & Upload and by RTSP.

    The web UI writes the practical detector/OCR/person/traffic values here so
    the same values are used for uploaded images, uploaded videos and the live
    RTSP loop. This keeps the old Settings page focused on global app/storage
    options while the tunable recognition values live in Test & Upload.
    """
    if request.method == 'GET':
        return jsonify({
            'success': True,
            'config': _public_config(),
            'models': _scan_model_files(),
            'model_status': {
                'models_loaded': bool(getattr(detector, 'models_loaded', False)),
                'plate_model_loaded': bool(getattr(detector, 'plate_model', None)),
                'vehicle_model_loaded': bool(getattr(detector, 'vehicle_model', None)),
                'human_model_loaded': bool(getattr(detector, 'human_model', None)),
                'ocr_engine': config_manager.get('ocr', 'engine') or 'fast_plate_ocr'
            }
        })

    try:
        data = request.get_json(silent=True) or {}
        reload_requested = bool(data.pop('reload', False))

        for section in ('detection', 'ocr', 'models', 'people', 'traffic'):
            payload = data.get(section)
            if not isinstance(payload, dict):
                continue
            config_manager.config.setdefault(section, {})
            if section == 'ocr' and isinstance(payload.get('preprocessing'), dict):
                config_manager.config[section].setdefault('preprocessing', {})
                config_manager.config[section]['preprocessing'].update(payload.pop('preprocessing'))
            config_manager.config[section].update(payload)

        # Keep person model path synchronized with the generic models section.
        people_cfg = config_manager.config.setdefault('people', {})
        models_cfg = config_manager.config.setdefault('models', {})
        selected_person = (people_cfg.get('selected_model_file') or people_cfg.get('custom_model_path') or models_cfg.get('person_detector'))
        if selected_person:
            people_cfg['model_path'] = selected_person
            people_cfg['custom_model_path'] = selected_person
            people_cfg['selected_model_file'] = selected_person
            models_cfg['person_detector'] = selected_person

        # Person images are intentionally enabled by default for the new workflow.
        if people_cfg.get('image_history_enabled', True):
            people_cfg['save_person_crops'] = True
            people_cfg['image_history_store_crop'] = True
            people_cfg.setdefault('image_history_store_annotated', True)

        config_manager.save_config()

        detector.ocr_reader = None
        detector.fast_plate_recognizer = None
        detector.fast_plate_cache_key = None
        detector.models_loaded = False
        detector.human_model = None
        if reload_requested:
            config_manager.config.setdefault('models', {})['last_reload_at'] = datetime.now().isoformat()
            config_manager.save_config()
            threading.Thread(target=detector.load_models, daemon=True).start()

        logger.info('[TestSettings] Gespeichert: plate_model=%s vehicle_model=%s person_model=%s plate_conf=%s ocr=%s people=%s',
                    config_manager.get('models', 'license_plate_detector'),
                    config_manager.get('models', 'vehicle_detector'),
                    config_manager.get('models', 'person_detector'),
                    config_manager.get('detection', 'plate_detector_confidence'),
                    config_manager.get('ocr', 'engine'),
                    'aktiv' if config_manager.get('people', 'enabled') else 'inaktiv')
        return jsonify({'success': True, 'config': _public_config(), 'models': _scan_model_files()})
    except Exception as e:
        logger.exception(f'Test-Einstellungen konnten nicht gespeichert werden: {e}')
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/settings/schema')
def api_settings_schema():
    return jsonify({'schema': _setting_schema(), 'config': _public_config()})


@app.route('/api/config/general', methods=['POST'])
def api_save_general_config():
    try:
        data = request.get_json(silent=True) or {}
        config_manager.config.setdefault('general', {})
        config_manager.config['general'].update(data)
        config_manager.save_config()
        return jsonify({'success': True, 'config': _public_config()})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/config/ui', methods=['POST'])
def api_save_ui_config():
    try:
        data = request.get_json(silent=True) or {}
        config_manager.config.setdefault('ui', {})
        config_manager.config['ui'].update(data)
        config_manager.save_config()
        return jsonify({'success': True, 'config': _public_config()})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/config/privacy', methods=['POST'])
def api_save_privacy_config():
    try:
        data = request.get_json(silent=True) or {}
        config_manager.config.setdefault('privacy', {})
        config_manager.config['privacy'].update(data)
        config_manager.save_config()
        return jsonify({'success': True, 'config': _public_config()})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/config/export')
def api_config_export():
    filename = f"platevision_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    return Response(json.dumps(config_manager.config, indent=2, ensure_ascii=False), mimetype='application/json', headers={'Content-Disposition': f'attachment; filename={filename}'})


@app.route('/api/config/import', methods=['POST'])
def api_config_import():
    try:
        payload = None
        if request.files.get('file'):
            payload = json.load(request.files['file'].stream)
        else:
            payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            raise ValueError('Keine gültige JSON-Konfiguration')
        merged = config_manager._merge_configs(config_manager.DEFAULT_CONFIG, payload)
        config_manager.config = merged
        config_manager.save_config()
        return jsonify({'success': True, 'config': _public_config()})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/profile/apply/<profile_name>', methods=['POST'])
def api_apply_profile(profile_name):
    profiles = config_manager.get('recognition_profiles', 'profiles') or {}
    profile = profiles.get(profile_name)
    if not profile:
        return jsonify({'success': False, 'error': 'Profil nicht gefunden'}), 404
    config_manager.config.setdefault('detection', {})['confidence_threshold'] = profile.get('confidence_threshold', config_manager.get('detection', 'confidence_threshold'))
    config_manager.config.setdefault('detection', {})['process_interval'] = profile.get('process_interval', config_manager.get('detection', 'process_interval'))
    config_manager.config.setdefault('ocr', {})['min_confidence'] = profile.get('ocr_min_confidence', config_manager.get('ocr', 'min_confidence'))
    config_manager.config.setdefault('recognition_profiles', {})['active'] = profile_name
    config_manager.save_config()
    return jsonify({'success': True, 'active': profile_name, 'config': _public_config()})


@app.route('/api/plate/analyze', methods=['POST'])
def api_plate_analyze():
    data = request.get_json(silent=True) or {}
    plate = data.get('plate_text') or data.get('plate') or ''
    country_hint = data.get('country_hint') or config_manager.get('plate_recognition', 'country_hint') or 'auto'
    return jsonify({'analysis': PlateUtils.analyze(plate, country_hint=country_hint), 'candidates': PlateUtils.generate_candidates(plate, country_hint=country_hint)})


@app.route('/api/plate/batch-analyze', methods=['POST'])
def api_plate_batch_analyze():
    data = request.get_json(silent=True) or {}
    values = data.get('plates') or data.get('values') or []
    if isinstance(values, str):
        values = re.split(r'[\n,;]+', values)
    country_hint = data.get('country_hint') or config_manager.get('plate_recognition', 'country_hint') or 'auto'
    rows = [{'input': value, 'best': PlateUtils.best_candidate(value, country_hint=country_hint), 'candidates': PlateUtils.generate_candidates(value, country_hint=country_hint, max_candidates=5)} for value in values if str(value).strip()]
    return jsonify({'rows': rows, 'total': len(rows)})


@app.route('/api/history/autocomplete')
def api_history_autocomplete():
    q = PlateUtils.normalize(request.args.get('q', ''), compact=True)
    limit = request.args.get('limit', 12, type=int)
    counter = Counter()
    for entry in history_manager.history:
        plate = PlateUtils.normalize(entry.get('plate_text', ''), compact=True)
        if plate and (not q or q in plate or PlateUtils.similarity(q, plate) >= 0.65):
            counter[entry.get('plate_text', plate)] += 1
    return jsonify({'items': [{'plate_text': plate, 'count': count} for plate, count in counter.most_common(limit)]})


@app.route('/api/history/timeline')
def api_history_timeline():
    bucket = request.args.get('bucket', 'day')
    limit = request.args.get('limit', 60, type=int)
    counts = Counter()
    for entry in history_manager.history:
        ts = history_manager._parse_datetime(entry.get('timestamp'))
        if not ts:
            continue
        key = ts.strftime('%Y-%m-%d %H:00') if bucket == 'hour' else ts.date().isoformat()
        counts[key] += 1
    items = [{'time': k, 'count': v} for k, v in sorted(counts.items())[-limit:]]
    return jsonify({'bucket': bucket, 'items': items})


@app.route('/api/history/duplicates')
def api_history_duplicates():
    threshold = request.args.get('similarity', config_manager.get('history', 'fuzzy_duplicate_similarity') or 0.88, type=float)
    groups = []
    used = set()
    entries = history_manager.history[:1000]
    for idx, entry in enumerate(entries):
        if idx in used:
            continue
        plate = PlateUtils.normalize(entry.get('plate_text', ''), compact=True)
        if not plate:
            continue
        group = [entry]
        used.add(idx)
        for j, other in enumerate(entries[idx + 1:], start=idx + 1):
            if j in used:
                continue
            other_plate = PlateUtils.normalize(other.get('plate_text', ''), compact=True)
            if other_plate and PlateUtils.similarity(plate, other_plate) >= threshold:
                group.append(other)
                used.add(j)
        if len(group) > 1:
            groups.append({'plate': entry.get('plate_text', plate), 'normalized': plate, 'count': len(group), 'entries': group[:10]})
    return jsonify({'groups': groups, 'total_groups': len(groups), 'similarity': threshold})


@app.route('/api/history/cleanup', methods=['POST'])
def api_history_cleanup():
    data = request.get_json(silent=True) or {}
    retention_days = int(data.get('retention_days') or config_manager.get('privacy', 'retention_days') or 0)
    dry_run = str(data.get('dry_run', True)).lower() in ('true', '1', 'yes', 'on')
    if retention_days <= 0:
        return jsonify({'success': True, 'deleted': 0, 'kept': len(history_manager.history), 'dry_run': dry_run})
    cutoff = datetime.now().timestamp() - retention_days * 86400
    keep, delete = [], []
    for entry in history_manager.history:
        ts = history_manager._parse_datetime(entry.get('timestamp'))
        if ts and ts.timestamp() < cutoff:
            delete.append(entry)
        else:
            keep.append(entry)
    if not dry_run:
        with history_manager.lock:
            history_manager.history = keep
            history_manager.save_history()
    return jsonify({'success': True, 'deleted': len(delete), 'kept': len(keep), 'dry_run': dry_run, 'retention_days': retention_days})


@app.route('/api/watchlist/export')
def api_watchlist_export():
    filename = f"platevision_watchlist_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    return Response(json.dumps(watchlist_manager.list(), indent=2, ensure_ascii=False), mimetype='application/json', headers={'Content-Disposition': f'attachment; filename={filename}'})


@app.route('/api/watchlist/import', methods=['POST'])
def api_watchlist_import():
    try:
        payload = None
        if request.files.get('file'):
            payload = json.load(request.files['file'].stream)
        else:
            payload = request.get_json(silent=True)
        if isinstance(payload, dict):
            payload = payload.get('items') or payload.get('watchlist') or []
        if not isinstance(payload, list):
            raise ValueError('Watchlist muss eine Liste sein')
        added = 0
        for item in payload:
            if not isinstance(item, dict):
                continue
            plate = item.get('plate_text') or item.get('plate') or item.get('normalized')
            if plate:
                watchlist_manager.add(plate, item.get('label', ''), item.get('category', 'known'), item.get('notes', ''), item.get('notify', True))
                added += 1
        return jsonify({'success': True, 'added': added, 'items': watchlist_manager.list()})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400



@app.route('/api/statistics/traffic')
def api_statistics_traffic():
    filters = dict(request.args)
    return jsonify(history_manager.get_traffic_statistics(filters))


@app.route('/api/statistics/plate/<plate_text>')
def api_statistics_plate(plate_text):
    filters = dict(request.args)
    return jsonify(history_manager.get_plate_profile(plate_text, filters))


@app.route('/api/statistics/export')
def api_statistics_export():
    fmt = (request.args.get('format') or 'csv').lower()
    data = history_manager.get_traffic_statistics(dict(request.args))
    filename = f"platevision_traffic_statistics_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if fmt == 'json':
        response = app.response_class(
            response=json.dumps(data, ensure_ascii=False, indent=2),
            status=200,
            mimetype='application/json'
        )
        response.headers['Content-Disposition'] = f'attachment; filename={filename}.json'
        return response
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['date', 'detections', 'visits', 'unique_vehicles', 'arrivals', 'departures_detected', 'departures_assumed', 'present_recently', 'repeat_vehicles'])
    for row in data.get('daily', []):
        writer.writerow([row.get('date'), row.get('detections'), row.get('visits'), row.get('unique_vehicles'), row.get('arrivals'), row.get('departures_detected'), row.get('departures_assumed'), row.get('present_recently'), row.get('repeat_vehicles')])
    writer.writerow([])
    writer.writerow(['plate_text', 'visits', 'detections', 'days_seen', 'first_seen', 'last_seen', 'last_status', 'vehicle_type', 'vehicle_color'])
    for row in data.get('top_plates', []):
        writer.writerow([row.get('plate_text'), row.get('visits'), row.get('detections'), row.get('days_seen'), row.get('first_seen'), row.get('last_seen'), row.get('last_status'), row.get('vehicle_type'), row.get('vehicle_color')])
    response = app.response_class(response=output.getvalue(), status=200, mimetype='text/csv')
    response.headers['Content-Disposition'] = f'attachment; filename={filename}.csv'
    return response


@app.route('/api/config/traffic', methods=['POST'])
def api_save_traffic_config():
    try:
        data = request.get_json(silent=True) or {}
        config_manager.config.setdefault('traffic', {})
        config_manager.config['traffic'].update(data)
        config_manager.save_config()
        return jsonify({'success': True, 'config': _public_config()})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400




@app.route('/api/people/settings-state')
def api_people_settings_state():
    cfg = config_manager.get('people') or {}
    return jsonify({
        'success': True,
        'people': cfg,
        'summary': _people_settings_summary(cfg),
        'stream_status': stream_manager.get_status(),
        'models': _scan_model_files(),
        'model_status': {
            'models_loaded': detector.models_loaded,
            'human_model_loaded': detector.human_model is not None,
            'coco_model_loaded': detector.coco_model is not None
        }
    })


@app.route('/api/people/preview/image')
def api_people_preview_image():
    """Live/fallback preview image with people counting line, zone and selected settings."""
    cfg = _people_config_with_overrides(_people_cfg_from_preview_args(request.args))
    frame = stream_manager.get_raw_frame()
    source = 'RTSP Live-Frame'
    if frame is None:
        frame = stream_manager.get_current_frame()
    if frame is None:
        frame = _create_people_preview_fallback(message=cfg.get('settings_preview_fallback_label'))
        source = 'Fallback-Bild'
    annotated = _draw_people_calibration_overlay(frame, cfg, source_label=source)
    ok, buffer = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        return jsonify({'success': False, 'error': 'Vorschaubild konnte nicht erstellt werden.'}), 500
    resp = Response(buffer.tobytes(), mimetype='image/jpeg')
    resp.headers['Cache-Control'] = 'no-store, max-age=0'
    resp.headers['X-PlateVision-Preview-Source'] = source
    return resp


@app.route('/api/people/preview/status')
def api_people_preview_status():
    stream_status = stream_manager.get_status()
    frame_available = stream_manager.get_raw_frame() is not None or stream_manager.get_current_frame() is not None
    return jsonify({
        'success': True,
        'frame_available': frame_available,
        'source': 'rtsp' if frame_available else 'fallback',
        'stream_status': stream_status,
        'summary': _people_settings_summary()
    })

@app.route('/api/people/statistics')
def api_people_statistics():
    return jsonify(person_history_manager.get_statistics(request.args.to_dict()))

@app.route('/api/people/history')
def api_people_history():
    limit = request.args.get('limit', 100, type=int)
    offset = request.args.get('offset', 0, type=int)
    return jsonify({'entries': person_history_manager.get_all(limit=limit, offset=offset), 'total': len(person_history_manager.history)})

@app.route('/api/people/history/clear', methods=['POST'])
def api_people_history_clear():
    data = request.get_json(silent=True) or {}
    result = person_history_manager.clear_history(delete_images=data.get('delete_images', True))
    return jsonify({'success': True, **(result or {})})

@app.route('/api/people/history/<event_id>/delete', methods=['POST', 'DELETE'])
def api_people_history_delete_event(event_id):
    data = request.get_json(silent=True) or {}
    result = person_history_manager.delete_event(event_id, delete_images=data.get('delete_images', True))
    status = 200 if result.get('success') else 404
    return jsonify(result), status

@app.route('/api/people/images/history')
def api_people_images_history():
    return jsonify(person_history_manager.image_history(request.args.to_dict()))

@app.route('/api/people/images/<path:filename>')
def api_people_images_file(filename):
    root = person_history_manager.IMAGE_ROOT.resolve()
    target = (root / filename).resolve()
    if not str(target).startswith(str(root)) or not target.exists() or not target.is_file():
        return jsonify({'success': False, 'error': 'Bild nicht gefunden'}), 404
    return send_from_directory(str(root), filename)

@app.route('/api/people/images/delete', methods=['POST'])
def api_people_images_delete():
    data = request.get_json(silent=True) or {}
    ids = data.get('ids') or data.get('event_ids') or []
    if isinstance(ids, str):
        ids = [ids]
    deleted_events = 0
    deleted_images = 0
    for event_id in ids:
        result = person_history_manager.delete_event(event_id, delete_images=True)
        if result.get('success'):
            deleted_events += 1
            deleted_images += int(result.get('deleted_images') or 0)
    return jsonify({'success': True, 'deleted_events': deleted_events, 'deleted_images': deleted_images})

@app.route('/api/people/images/cleanup', methods=['POST'])
def api_people_images_cleanup():
    data = request.get_json(silent=True) or {}
    result = person_history_manager.cleanup_images(data.get('retention_days'), delete_records=bool(data.get('delete_records', False)))
    return jsonify({'success': True, **result})


@app.route('/api/people/export')
def api_people_export():
    fmt = request.args.get('format', config_manager.get('people', 'export_default_format') or 'csv').lower()
    rows = person_history_manager.search(request.args.to_dict())
    filename = f"platevision_people_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if fmt == 'json':
        return Response(json.dumps(rows, indent=2, ensure_ascii=False), mimetype='application/json', headers={'Content-Disposition': f'attachment; filename={filename}.json'})
    fields = ['timestamp', 'event_type', 'counted', 'direction', 'track_id', 'confidence', 'bbox', 'source', 'filename']
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fields, extrasaction='ignore')
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return Response(output.getvalue(), mimetype='text/csv; charset=utf-8', headers={'Content-Disposition': f'attachment; filename={filename}.csv'})

@app.route('/api/people/test/snapshot', methods=['POST'])
def api_people_test_snapshot():
    frame = stream_manager.get_raw_frame()
    if frame is None:
        return jsonify({'success': False, 'error': 'Kein Live-Frame verfügbar. Es wird in den Einstellungen automatisch ein Fallback-Bild für die Kalibrierung angezeigt.'}), 400
    previous_enabled = config_manager.get('people', 'enabled')
    config_manager.config.setdefault('people', {})['enabled'] = True
    try:
        result = detector.process_frame(frame)
        annotated = _draw_people_calibration_overlay(result.get('annotated_frame'), config_manager.get('people') or {}, source_label='RTSP Live-Frame')
        _, buffer = cv2.imencode('.jpg', annotated)
        _attach_person_crop_previews(result.get('people', []), frame)
        return jsonify({
            'success': True,
            'people': _json_safe(result.get('people', [])),
            'people_counted': sum(1 for p in result.get('people', []) if p.get('counted')),
            'result_image': '',
            'processing_time': result.get('processing_time', 0)
        })
    finally:
        config_manager.config.setdefault('people', {})['enabled'] = previous_enabled




def _handle_people_image_analysis(source_name='people_upload'):
    """Analyze an uploaded image with the saved/overridden people settings."""
    try:
        people_cfg = config_manager.get('people') or {}
        if people_cfg.get('test_image_upload_enabled') is False:
            return jsonify({'success': False, 'error': 'Personen-Bildanalyse ist in den Einstellungen deaktiviert.'}), 403
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'Keine Bilddatei erhalten.'}), 400
        payload, error = _safe_image_upload(request.files['file'])
        if error:
            return jsonify({'success': False, 'error': error}), 400

        override_cfg = {}
        raw_settings = request.form.get('settings_json')
        if raw_settings:
            try:
                parsed = json.loads(raw_settings)
                if isinstance(parsed, dict):
                    override_cfg = parsed.get('people') if isinstance(parsed.get('people'), dict) else parsed
            except Exception as exc:
                logger.warning(f'Ungueltige settings_json fuer Personenanalyse: {exc}')
        effective_cfg = _people_config_with_overrides(override_cfg)

        force_enable = _bool_from_request(request.form.get('force_enable'), effective_cfg.get('test_force_enable_people', True))
        save_to_history = _bool_from_request(request.form.get('save_history'), effective_cfg.get('test_save_to_history_default', True))
        save_upload = _bool_from_request(request.form.get('save_upload'), effective_cfg.get('test_save_uploads', False))
        draw_boxes = _bool_from_request(request.form.get('draw_boxes'), effective_cfg.get('draw_boxes', True))
        if force_enable:
            effective_cfg['enabled'] = True
        effective_cfg['draw_boxes'] = draw_boxes

        saved_filename = payload['filename']
        if save_upload or effective_cfg.get('test_save_uploads'):
            target_dir = Path('uploads/people_tests') / datetime.now().strftime('%Y-%m-%d')
            target_dir.mkdir(parents=True, exist_ok=True)
            saved_filename = f"{datetime.now().strftime('%H%M%S')}_{uuid.uuid4().hex[:8]}_{payload['filename']}"
            (target_dir / saved_filename).write_bytes(payload['data'])

        original_people = json.loads(json.dumps(config_manager.config.get('people') or {}, ensure_ascii=False))
        config_manager.config['people'] = effective_cfg
        try:
            result = detector.process_frame(payload['image'])
        finally:
            config_manager.config['people'] = original_people

        annotated = result.get('annotated_frame') if result.get('annotated_frame') is not None else payload['image'].copy()
        # Ensure the calibration line and zone are visible even when no person is detected.
        annotated = _draw_people_calibration_overlay(annotated, effective_cfg, source_label='Upload-Bild')

        history_saved = 0
        if save_to_history and (effective_cfg.get('history_enabled') is not False):
            for person in result.get('people', []):
                if person.get('counted') or effective_cfg.get('save_all_detections') or effective_cfg.get('image_history_enabled'):
                    event = dict(person)
                    event.update({'source': source_name, 'filename': saved_filename})
                    saved_person_event = person_history_manager.add_event(event, frame=payload['image'], annotated_frame=annotated)
                    if saved_person_event:
                        person.update({k: saved_person_event.get(k) for k in ('id', 'counted', 'event_type', 'repeat_blocked', 'repeat_block_minutes', 'repeat_match_id', 'images', 'note') if k in saved_person_event})
                        history_saved += 1

        people = result.get('people', []) or []
        _attach_person_crop_previews(people, payload['image'])
        result_image_b64 = ''
        return jsonify({
            'success': True,
            'filename': saved_filename,
            'people': _json_safe(people),
            'people_count': len(people),
            'people_counted': sum(1 for p in people if p.get('counted')),
            'history_saved': history_saved,
            'result_image': result_image_b64,
            'processing_time': result.get('processing_time', 0),
            'forced_enabled': force_enable,
            'applied_settings': _people_settings_summary(effective_cfg),
            'model_mode': effective_cfg.get('model_mode'),
            'selected_model': effective_cfg.get('selected_model_file') or effective_cfg.get('custom_model_path') or effective_cfg.get('model_path') or (config_manager.get('models') or {}).get('person_detector'),
            'status': {
                'human_model_loaded': detector.human_model is not None,
                'coco_model_loaded': detector.coco_model is not None,
                'models_loaded': detector.models_loaded
            }
        })
    except Exception as e:
        logger.error(f'Personen-Bildanalyse Fehler: {e}')
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/people/analyze/image', methods=['POST'])
def api_people_analyze_image_upload():
    """Normal project endpoint for uploaded-image people analysis."""
    return _handle_people_image_analysis('people_image_analysis')


@app.route('/api/people/test/image', methods=['POST'])
def api_people_test_image_upload():
    """Backward-compatible endpoint used by the /test page."""
    return _handle_people_image_analysis('people_test_upload')


@app.route('/api/people/presence')
def api_people_presence():
    return jsonify(person_history_manager.get_presence())

@app.route('/api/people/cleanup', methods=['POST'])
def api_people_cleanup():
    data = request.get_json(silent=True) or {}
    return jsonify({'success': True, **person_history_manager.cleanup(data.get('retention_days'))})

@app.route('/api/people/test/simulate', methods=['POST'])
def api_people_test_simulate():
    return jsonify({'success': False, 'error': 'Demo-Daten wurden in Version 0.8.23 entfernt. Bitte echte Foto-/RTSP-Tests verwenden.'}), 410

@app.route('/api/system/health')
def api_system_health():
    checks = []
    def add(name, ok, detail=''):
        checks.append({'name': name, 'ok': bool(ok), 'detail': detail})
    add('models_directory', os.path.isdir('models'), 'models/')
    add('vehicle_model', os.path.exists(config_manager.get('models', 'vehicle_detector') or ''), config_manager.get('models', 'vehicle_detector') or '')
    add('plate_model', os.path.exists(config_manager.get('models', 'license_plate_detector') or ''), config_manager.get('models', 'license_plate_detector') or '')
    add('history_writable', os.access(os.path.dirname(history_manager.HISTORY_FILE), os.W_OK), history_manager.HISTORY_FILE)
    add('watchlist_writable', os.access(os.path.dirname(watchlist_manager.WATCHLIST_FILE), os.W_OK), watchlist_manager.WATCHLIST_FILE)
    add('people_history_writable', os.access(os.path.dirname(person_history_manager.HISTORY_FILE), os.W_OK), person_history_manager.HISTORY_FILE)
    add('person_model_path', bool(config_manager.get('people', 'model_mode') == 'coco_person' or os.path.exists(config_manager.get('people', 'custom_model_path') or '')), config_manager.get('people', 'custom_model_path') or 'COCO person class')
    add('rtsp_configured', bool(config_manager.get('rtsp', 'url')), _public_config().get('rtsp', {}).get('url_masked', ''))
    add('stream_connected', stream_manager.is_connected(), stream_manager.get_status().get('error') or '')
    ok = all(c['ok'] for c in checks if c['name'] not in ('stream_connected',))
    return jsonify({'ok': ok, 'checks': checks, 'status': stream_manager.get_status(), 'version': '0.8.23'})


@app.route('/api/system/audit')
def api_system_audit():
    issues = []
    suggestions = []
    if not os.path.exists(config_manager.get('models', 'license_plate_detector') or ''):
        issues.append({'level': 'warning', 'title': 'Kennzeichen-Modell fehlt', 'detail': 'Ohne license_plate_detector.pt kann keine Kennzeichenerkennung laufen.'})
    if not os.path.exists(config_manager.get('models', 'vehicle_detector') or ''):
        issues.append({'level': 'warning', 'title': 'Fahrzeug-Modell fehlt', 'detail': 'Ohne yolov8n.pt ist Fahrzeug-Zoom eingeschränkt.'})
    if (config_manager.get('ocr', 'min_confidence') or 0) < 0.15:
        issues.append({'level': 'info', 'title': 'OCR-Konfidenz sehr niedrig', 'detail': 'Das erhöht Treffer, aber auch Fehlalarme.'})
    if config_manager.get('history', 'duplicate_timeout') and config_manager.get('history', 'duplicate_timeout') < 10:
        suggestions.append('Duplikat-Timeout auf 30-120 Sekunden setzen, wenn derselbe Wagen mehrfach erkannt wird.')
    if not config_manager.get('plate_recognition', 'smart_ocr_correction'):
        suggestions.append('Intelligente OCR-Korrektur aktivieren, um O/0, I/1 und S/5 besser zu behandeln.')
    if not config_manager.get('search', 'enable_fuzzy_search'):
        suggestions.append('Fuzzy-Suche aktivieren, damit ähnliche Kennzeichen gefunden werden.')
    if (config_manager.get('traffic', 'active_timeout_minutes') or 0) < 5:
        suggestions.append('Traffic-Timeout auf mindestens 10-30 Minuten setzen, damit Kommen/Gehen realistischer ausgewertet wird.')
    return jsonify({'issues': issues, 'suggestions': suggestions, 'stats': history_manager.get_statistics(), 'config': _public_config()})


# ============================================================
# WEBSOCKET EVENTS
# ============================================================

@socketio.on('connect')
def handle_connect():
    emit('connected', {'status': 'ok', 'stream_status': stream_manager.get_status()})
    logger.info("WebSocket Client verbunden")

@socketio.on('disconnect')
def handle_disconnect():
    logger.info("WebSocket Client getrennt")

@socketio.on('request_frame')
def handle_frame_request():
    frame = stream_manager.get_current_frame()
    if frame is not None:
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        frame_b64 = base64.b64encode(buffer).decode('utf-8')
        emit('frame', {'image': frame_b64})


# ============================================================
# FEHLERHANDLER
# ============================================================

@app.errorhandler(404)
def page_not_found(e):
    if request.path.startswith('/api/'):
        return jsonify({'error': 'Nicht gefunden'}), 404
    return render_template('404.html', page='error'), 404

@app.errorhandler(500)
def internal_error(e):
    logger.error(f"Internal Error: {e}")
    if request.path.startswith('/api/'):
        return jsonify({'error': 'Interner Serverfehler'}), 500
    return render_template('500.html', page='error'), 500


# ============================================================
# HAUPTPROGRAMM
# ============================================================


@app.route('/api/system/compatibility')
def api_system_compatibility():
    """Reports whether original PlateVision routes and config sections are still present."""
    original_config_sections = ['rtsp', 'detection', 'ocr', 'general', 'history', 'models']
    original_routes = [
        '/', '/dashboard', '/history', '/rtsp-settings', '/settings', '/test', '/live', '/latest',
        '/api/stream/start', '/api/stream/stop', '/api/stream/status', '/api/stream/resolution', '/api/stream/feed', '/api/stream/snapshot',
        '/api/config', '/api/config/rtsp', '/api/config/detection', '/api/config/ocr', '/api/config/history',
        '/api/history', '/api/history/statistics', '/api/storage/info', '/api/process/image', '/api/latest',
        '/api/models/status', '/api/models/reload', '/api/models/available', '/api/models/people/options', '/api/models/people/select', '/api/config/validate', '/api/people/presence', '/api/people/cleanup', '/api/system/info', '/api/process/video', '/api/process/jobs'
    ]
    available_routes = sorted(str(rule.rule) for rule in app.url_map.iter_rules())
    missing_sections = [section for section in original_config_sections if section not in config_manager.config]
    missing_routes = [route for route in original_routes if route not in available_routes]
    return jsonify({
        'success': True,
        'original_config_sections_present': len(missing_sections) == 0,
        'original_routes_present': len(missing_routes) == 0,
        'missing_config_sections': missing_sections,
        'missing_routes': missing_routes,
        'mode': 'additive-extension',
        'note': 'Original sections/routes are preserved; ProTraffic settings are added on top.'
    })

if __name__ == '__main__':
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║     PLATEVISION - LICENSE PLATE DETECTION SYSTEM         ║
    ║     Version 0.8.23 FastPlateOCR Vehicle Intelligence                                    ║
    ╠══════════════════════════════════════════════════════════╣
    ║     Dashboard:     http://localhost:5000                 ║
    ║     Live Stream:   http://localhost:5000/live            ║
    ║     RTSP Settings: http://localhost:5000/rtsp-settings   ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    socketio.run(app, host='0.0.0.0', port=5000, 
                 debug=config_manager.get('general', 'debug_mode') or False,
                 allow_unsafe_werkzeug=True)





