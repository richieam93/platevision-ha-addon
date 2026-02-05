"""
PlateVision - License Plate Detection System
Flask-based Web Application with RTSP Support
Version 2.1 - Fixed RTSP & Analysis Area Support
"""

from flask import Flask, render_template, request, jsonify, Response, send_from_directory
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
import threading
import time
import json
import os
import uuid
from datetime import datetime
import base64
from PIL import Image
import io
import queue
import logging
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
    'static',
    'static/css',
    'static/js',
    'static/images',
    'data',
    'data/plates_detected',
    'data/vehicles_detected',
    'data/history',
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
                'area': {
                    'x': 0,
                    'y': 0,
                    'width': 1280,
                    'height': 720
                }
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
            'min_plate_width': 60,
            'min_plate_height': 15,
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
            },
            'active_mode': 'enhanced',
            'retry_on_fail': True,
            'max_retries': 3,
        },
        'general': {
            'theme': 'dark',
            'language': 'de',
            'auto_save_history': True,
            'max_history_entries': 1000,
            'notification_enabled': True,
            'debug_mode': False
        },
        'history': {
            'filter_duplicates': True,
            'duplicate_timeout': 60,
            'min_confidence_to_save': 0.35,
            'save_vehicle_image': True,
            'save_plate_image': True,
        },
        'models': {
            'license_plate_detector': 'models/license_plate_detector.pt',
            'vehicle_detector': 'models/yolov8n.pt'
        }
    }
    
    def __init__(self):
        self.config = self.load_config()
    
    def load_config(self):
        if os.path.exists(self.CONFIG_FILE):
            try:
                with open(self.CONFIG_FILE, 'r', encoding='utf-8') as f:
                    saved_config = json.load(f)
                return self._merge_configs(self.DEFAULT_CONFIG, saved_config)
            except Exception as e:
                logger.error(f"Fehler beim Laden der Konfiguration: {e}")
                return self.DEFAULT_CONFIG.copy()
        return self.DEFAULT_CONFIG.copy()
    
    def _merge_configs(self, default, saved):
        result = {}
        for key, value in default.items():
            if key in saved:
                if isinstance(value, dict) and isinstance(saved[key], dict):
                    result[key] = self._merge_configs(value, saved[key])
                else:
                    result[key] = saved[key]
            else:
                result[key] = value
        return result
    
    def save_config(self):
        try:
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
        if not plate_text:
            return ""
        return plate_text.upper().replace(' ', '').replace('-', '').strip()
    
    def _is_duplicate_in_history(self, plate_text, timeout_seconds=60):
        if not plate_text:
            return True
        
        normalized = self._normalize_plate(plate_text)
        if not normalized or len(normalized) < 3:
            return True
            
        current_time = datetime.now()
        
        for entry in self.history[:100]:
            entry_plate = self._normalize_plate(entry.get('plate_text', ''))
            
            if entry_plate == normalized:
                try:
                    entry_time = datetime.fromisoformat(entry.get('timestamp', ''))
                    time_diff = (current_time - entry_time).total_seconds()
                    
                    if time_diff < timeout_seconds:
                        return True
                except:
                    pass
        
        return False
    
    def add_entry(self, entry, check_duplicate=True):
        with self.lock:
            if check_duplicate:
                timeout = config_manager.get('history', 'duplicate_timeout') or 60
                filter_enabled = config_manager.get('history', 'filter_duplicates')
                
                if filter_enabled and self._is_duplicate_in_history(entry.get('plate_text'), timeout):
                    logger.debug(f"Duplikat in Historie übersprungen: {entry.get('plate_text')}")
                    return None
            
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
    
    def get_statistics(self):
        total = len(self.history)
        today = datetime.now().date().isoformat()
        today_count = sum(1 for e in self.history if e.get('timestamp', '').startswith(today))
        
        unique_plates = set()
        vehicle_types = {}
        
        for e in self.history:
            normalized = self._normalize_plate(e.get('plate_text', ''))
            if normalized:
                unique_plates.add(normalized)
            
            v_type = e.get('vehicle_type', 'unknown')
            vehicle_types[v_type] = vehicle_types.get(v_type, 0) + 1
        
        plate_counts = {}
        for e in self.history:
            plate = e.get('plate_text', '')
            if plate:
                plate_counts[plate] = plate_counts.get(plate, 0) + 1
        
        top_plates = sorted(plate_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            'total_detections': total,
            'today_detections': today_count,
            'unique_plates': len(unique_plates),
            'vehicle_types': vehicle_types,
            'top_plates': top_plates
        }


# ============================================================
# KENNZEICHEN-DETEKTOR
# ============================================================

class LicensePlateDetector:
    """Haupt-Erkennungsklasse"""
    
    VEHICLE_CLASSES = {2: 'PKW', 3: 'Motorrad', 5: 'Bus', 7: 'LKW'}
    VEHICLE_CLASSES_EN = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
    
    def __init__(self, config_manager):
        self.config_manager = config_manager
        self.coco_model = None
        self.license_model = None
        self.ocr_reader = None
        self.models_loaded = False
        self.load_lock = threading.Lock()
        self.recent_plates = {}
    
    def load_models(self):
        with self.load_lock:
            if self.models_loaded:
                return True
            
            try:
                logger.info("Lade ML-Modelle...")
                
                vehicle_model_path = self.config_manager.get('models', 'vehicle_detector')
                license_model_path = self.config_manager.get('models', 'license_plate_detector')
                
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
                
                languages = self.config_manager.get('ocr', 'languages') or ['en']
                gpu_enabled = self.config_manager.get('ocr', 'gpu_enabled') or False
                self.ocr_reader = easyocr.Reader(languages, gpu=gpu_enabled)
                logger.info(f"OCR geladen mit Sprachen: {languages}")
                
                self.models_loaded = True
                return True
                
            except Exception as e:
                logger.error(f"Fehler beim Laden der Modelle: {e}")
                import traceback
                traceback.print_exc()
                return False
    
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
        
        normalized = plate_text.upper().replace(' ', '').replace('-', '')
        
        if normalized in self.recent_plates:
            return True
        
        self.recent_plates[normalized] = current_time
        return False
    
    def _estimate_vehicle_color(self, vehicle_crop):
        try:
            if vehicle_crop is None or vehicle_crop.size == 0:
                return "unbekannt"
            
            hsv = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2HSV)
            
            center_y = vehicle_crop.shape[0] // 2
            center_x = vehicle_crop.shape[1] // 2
            roi_size = min(vehicle_crop.shape[0], vehicle_crop.shape[1]) // 4
            
            roi = hsv[
                max(0, center_y - roi_size):center_y + roi_size,
                max(0, center_x - roi_size):center_x + roi_size
            ]
            
            if roi.size == 0:
                return "unbekannt"
            
            avg_h = np.mean(roi[:, :, 0])
            avg_s = np.mean(roi[:, :, 1])
            avg_v = np.mean(roi[:, :, 2])
            
            if avg_s < 30:
                if avg_v < 50:
                    return "Schwarz"
                elif avg_v > 200:
                    return "Weiß"
                else:
                    return "Grau"
            elif avg_s < 80:
                return "Silber"
            else:
                if avg_h < 15 or avg_h > 165:
                    return "Rot"
                elif avg_h < 25:
                    return "Orange"
                elif avg_h < 35:
                    return "Gelb"
                elif avg_h < 85:
                    return "Grün"
                elif avg_h < 130:
                    return "Blau"
                else:
                    return "Lila"
                    
        except Exception as e:
            return "unbekannt"
    
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
            
            if len(processed.shape) == 3:
                gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
            else:
                gray = processed
            
            variants = [gray.copy()]
            
            if config.get('denoise', True):
                gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
            
            if config.get('contrast_enhance', True):
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                gray = clahe.apply(gray)
            
            variants.append(gray.copy())
            
            if config.get('sharpen', True):
                kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
                sharpened = cv2.filter2D(gray, -1, kernel)
                variants.append(sharpened)
            
            if config.get('adaptive_threshold', True):
                thresh1 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                cv2.THRESH_BINARY, 11, 2)
                variants.append(thresh1)
                
                _, thresh3 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                variants.append(thresh3)
            
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
            
            best_result = None
            best_confidence = 0
            
            for i, variant in enumerate(variants[:6]):
                try:
                    results = self.ocr_reader.readtext(variant, detail=1)
                    text, confidence = self._process_ocr_results(results, min_confidence * 0.7, allowed_chars)
                    
                    if text and confidence > best_confidence:
                        best_confidence = confidence
                        best_result = text
                    
                    if best_confidence >= 0.85:
                        break
                        
                except:
                    continue
            
            if best_result:
                best_result = self._correct_common_errors(best_result)
            
            return best_result, best_confidence
            
        except Exception as e:
            logger.error(f"OCR Fehler: {e}")
            return None, 0
    
    def _process_ocr_results(self, results, min_confidence, allowed_chars):
        if not results:
            return None, 0
        
        texts = []
        confidences = []
        
        for result in results:
            if len(result) >= 3:
                bbox, text, confidence = result[0], result[1], result[2]
            elif len(result) == 2:
                text, confidence = result[0], result[1]
            else:
                continue
                
            if confidence >= min_confidence:
                clean_text = ''.join(c for c in text.upper() 
                                    if not allowed_chars or c in allowed_chars.upper())
                if clean_text and len(clean_text) >= 2:
                    texts.append(clean_text)
                    confidences.append(confidence)
        
        if texts:
            combined_text = ''.join(texts)
            combined_text = ' '.join(combined_text.split())
            avg_confidence = sum(confidences) / len(confidences)
            return combined_text, avg_confidence
        
        return None, 0
    
    def _correct_common_errors(self, text):
        if not text or len(text) < 2:
            return text
        
        result = list(text)
        
        for i, char in enumerate(result):
            prev_is_digit = i > 0 and result[i-1].isdigit()
            next_is_digit = i < len(result)-1 and result[i+1].isdigit()
            
            if char == 'O' and (prev_is_digit or next_is_digit):
                result[i] = '0'
            elif char == 'I' and (prev_is_digit or next_is_digit):
                result[i] = '1'
            elif char == 'l' and (prev_is_digit or next_is_digit):
                result[i] = '1'
        
        return ''.join(result)
    
    def process_frame(self, frame, apply_analysis_area=False):
        """Verarbeitet einen einzelnen Frame"""
        if not self.models_loaded:
            self.load_models()
        
        if frame is None or frame.size == 0:
            return {
                'annotated_frame': np.zeros((480, 640, 3), dtype=np.uint8),
                'detections': [],
                'vehicles': [],
                'processing_time': 0
            }
        
        result = {
            'annotated_frame': frame.copy(),
            'detections': [],
            'vehicles': [],
            'processing_time': 0
        }
        
        start_time = time.time()
        
        try:
            confidence_threshold = self.config_manager.get('detection', 'confidence_threshold') or 0.5
            zoom_enabled = self.config_manager.get('detection', 'zoom_enabled') or True
            zoom_factor = self.config_manager.get('detection', 'zoom_factor') or 2.5
            zoom_padding = self.config_manager.get('detection', 'zoom_padding') or 100
            
            annotated = frame.copy()
            detected_vehicles = []
            frame_h, frame_w = frame.shape[:2]
            
            # Fahrzeugerkennung
            if self.coco_model and self.config_manager.get('detection', 'car_detection_enabled'):
                vehicle_results = self.coco_model(frame, conf=confidence_threshold, verbose=False)[0]
                
                for detection in vehicle_results.boxes.data.tolist():
                    x1, y1, x2, y2, score, class_id = detection
                    class_id = int(class_id)
                    
                    if class_id in self.VEHICLE_CLASSES:
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        
                        vehicle_crop = frame[y1:y2, x1:x2].copy()
                        vehicle_color = self._estimate_vehicle_color(vehicle_crop)
                        
                        vehicle_info = {
                            'bbox': [x1, y1, x2, y2],
                            'confidence': score,
                            'type': self.VEHICLE_CLASSES[class_id],
                            'type_en': self.VEHICLE_CLASSES_EN[class_id],
                            'color': vehicle_color,
                            'crop': vehicle_crop
                        }
                        detected_vehicles.append(vehicle_info)
                        
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        label = f"{self.VEHICLE_CLASSES[class_id]} ({vehicle_color})"
                        cv2.putText(annotated, label, (x1, y1 - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            result['vehicles'] = detected_vehicles
            
            # Kennzeichenerkennung
            if self.license_model:
                frames_to_process = []
                
                if zoom_enabled and detected_vehicles:
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
                            'vehicle': vehicle
                        })
                else:
                    frames_to_process.append({
                        'frame': frame,
                        'offset': (0, 0),
                        'scale': 1,
                        'vehicle': None
                    })
                
                for frame_info in frames_to_process:
                    proc_frame = frame_info['frame']
                    off_x, off_y = frame_info['offset']
                    scale = frame_info['scale']
                    vehicle = frame_info['vehicle']
                    
                    if proc_frame is None or proc_frame.size == 0:
                        continue
                    
                    license_results = self.license_model(proc_frame, conf=0.3, verbose=False)[0]
                    
                    for plate_detection in license_results.boxes.data.tolist():
                        px1, py1, px2, py2, plate_score, _ = plate_detection
                        
                        orig_px1 = int(px1 / scale + off_x)
                        orig_py1 = int(py1 / scale + off_y)
                        orig_px2 = int(px2 / scale + off_x)
                        orig_py2 = int(py2 / scale + off_y)
                        
                        plate_crop_scaled = proc_frame[int(py1):int(py2), int(px1):int(px2)]
                        
                        if plate_crop_scaled.size == 0:
                            continue
                        
                        plate_text, ocr_confidence = self._read_plate_enhanced(plate_crop_scaled)
                        
                        min_save_conf = self.config_manager.get('history', 'min_confidence_to_save') or 0.35
                        
                        if not plate_text or ocr_confidence < min_save_conf:
                            cv2.rectangle(annotated, (orig_px1, orig_py1), (orig_px2, orig_py2), (0, 165, 255), 2)
                            continue
                        
                        if self._is_duplicate(plate_text):
                            continue
                        
                        cv2.rectangle(annotated, (orig_px1, orig_py1), (orig_px2, orig_py2), (0, 255, 0), 3)
                        
                        text_size = cv2.getTextSize(plate_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
                        cv2.rectangle(annotated, (orig_px1, orig_py1 - text_size[1] - 15),
                                     (orig_px1 + text_size[0] + 10, orig_py1), (0, 255, 0), -1)
                        cv2.putText(annotated, plate_text, (orig_px1 + 5, orig_py1 - 8),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
                        
                        # Bilder speichern
                        plate_image_b64 = None
                        vehicle_image_b64 = None
                        
                        if self.config_manager.get('detection', 'save_detected_plates'):
                            _, buffer = cv2.imencode('.jpg', plate_crop_scaled, [cv2.IMWRITE_JPEG_QUALITY, 95])
                            plate_image_b64 = base64.b64encode(buffer).decode('utf-8')
                        
                        if self.config_manager.get('detection', 'save_detected_vehicles') and vehicle:
                            vehicle_crop = vehicle.get('crop')
                            if vehicle_crop is not None and vehicle_crop.size > 0:
                                _, buffer = cv2.imencode('.jpg', vehicle_crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
                                vehicle_image_b64 = base64.b64encode(buffer).decode('utf-8')
                        
                        detection_info = {
                            'plate_text': plate_text,
                            'confidence': ocr_confidence,
                            'plate_bbox': [orig_px1, orig_py1, orig_px2, orig_py2],
                            'plate_score': plate_score,
                            'plate_image_base64': plate_image_b64,
                            'vehicle_image_base64': vehicle_image_b64,
                            'vehicle_type': vehicle['type'] if vehicle else 'Unbekannt',
                            'vehicle_type_en': vehicle['type_en'] if vehicle else 'unknown',
                            'vehicle_color': vehicle['color'] if vehicle else 'Unbekannt',
                        }
                        
                        result['detections'].append(detection_info)
                        logger.info(f"Erkannt: {plate_text} | Konfidenz: {ocr_confidence:.2f}")
            
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
# GLOBALE INSTANZEN
# ============================================================

config_manager = ConfigManager()
history_manager = HistoryManager()
detector = LicensePlateDetector(config_manager)

# RTSP Handler importieren
from rtsp_handler import RTSPHandler
stream_manager = RTSPHandler(config_manager, history_manager, detector)

def init_models():
    detector.load_models()

threading.Thread(target=init_models, daemon=True).start()


# ============================================================
# FLASK ROUTEN - SEITEN
# ============================================================

@app.route('/')
def index():
    return render_template('index.html', 
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
    page_num = request.args.get('page', 1, type=int)
    per_page = 20
    search = request.args.get('search', '')
    unique_only = request.args.get('unique', 'false').lower() == 'true'
    
    if search:
        entries = history_manager.search(search)
    else:
        entries = history_manager.get_all(limit=per_page, offset=(page_num-1)*per_page, unique_only=unique_only)
    
    total = len(history_manager.history)
    
    return render_template('history.html',
                          page='history',
                          entries=entries,
                          current_page=page_num,
                          total_pages=(total // per_page) + 1,
                          total_entries=total,
                          search=search,
                          unique_only=unique_only)

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
    return render_template('test.html', page='test', jobs=video_processing_jobs)

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

@app.route('/api/stream/feed')
def stream_feed():
    def generate():
        while True:
            frame = stream_manager.get_current_frame()
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
    """Einzelnes Snapshot vom Stream"""
    frame = stream_manager.get_current_frame()
    if frame is not None:
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return Response(buffer.tobytes(), mimetype='image/jpeg')
    
    # Placeholder
    placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(placeholder, "Kein Stream", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    _, buffer = cv2.imencode('.jpg', placeholder)
    return Response(buffer.tobytes(), mimetype='image/jpeg')


# ============================================================
# API ROUTEN - KONFIGURATION
# ============================================================

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
        
        # Deep merge für analysis_area
        if 'analysis_area' in data:
            if 'analysis_area' not in config_manager.config['rtsp']:
                config_manager.config['rtsp']['analysis_area'] = {}
            
            for key, value in data['analysis_area'].items():
                config_manager.config['rtsp']['analysis_area'][key] = value
            del data['analysis_area']
        
        config_manager.config['rtsp'].update(data)
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
        
        result = detector.process_frame(image)
        
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
                    'vehicle_color': detection.get('vehicle_color', 'Unbekannt'),
                }
                history_manager.add_entry(entry)
        
        return jsonify({
            'success': True,
            'result_image': result_image_b64,
            'detections': result['detections'],
            'vehicles': [{k: v for k, v in v.items() if k != 'crop'} for v in result['vehicles']],
            'processing_time': result['processing_time']
        })
        
    except Exception as e:
        logger.error(f"Bildverarbeitung Fehler: {e}")
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
# API ROUTEN - UTILITIES
# ============================================================

@app.route('/api/models/status')
def api_models_status():
    return jsonify({
        'loaded': detector.models_loaded,
        'coco_model': detector.coco_model is not None,
        'license_model': detector.license_model is not None,
        'ocr_reader': detector.ocr_reader is not None
    })

@app.route('/api/models/reload', methods=['POST'])
def api_reload_models():
    detector.models_loaded = False
    detector.coco_model = None
    detector.license_model = None
    detector.ocr_reader = None
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
        'history_count': len(history_manager.history)
    })


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

if __name__ == '__main__':
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║     PLATEVISION - LICENSE PLATE DETECTION SYSTEM         ║
    ║     Version 2.1                                          ║
    ╠══════════════════════════════════════════════════════════╣
    ║     Dashboard:     http://localhost:5000                 ║
    ║     Live Stream:   http://localhost:5000/live            ║
    ║     RTSP Settings: http://localhost:5000/rtsp-settings   ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    socketio.run(app, host='0.0.0.0', port=5000, 
                 debug=config_manager.get('general', 'debug_mode') or False,
                 allow_unsafe_werkzeug=True)
