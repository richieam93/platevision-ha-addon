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

video_processing_jobs = {}  # {job_id: {'progress': 0, 'status': 'processing', ...}}

# ============================================================
# KONFIGURATIONSMANAGER
# ============================================================

class ConfigManager:
    """Verwaltet alle Einstellungen der Anwendung"""
    
    CONFIG_FILE = 'data/config.json'
    
    DEFAULT_CONFIG = {
        # RTSP Einstellungen
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
        # Erkennungseinstellungen
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
        # Erweiterte OCR Einstellungen
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
        # Allgemeine Einstellungen
        'general': {
            'theme': 'dark',
            'language': 'de',
            'auto_save_history': True,
            'max_history_entries': 1000,
            'notification_enabled': True,
            'debug_mode': False
        },
        # Historie Einstellungen
        'history': {
            'filter_duplicates': True,
            'duplicate_timeout': 60,
            'min_confidence_to_save': 0.35,
            'save_vehicle_image': True,
            'save_plate_image': True,
        },
        # Modell Pfade
        'models': {
            'license_plate_detector': 'models/license_plate_detector.pt',
            'vehicle_detector': 'models/yolov8n.pt'
        }
    }
    
    def __init__(self):
        self.config = self.load_config()
    
    def load_config(self):
        """Lädt die Konfiguration aus der Datei"""
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
        """Merged gespeicherte Config mit Default für fehlende Werte"""
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
        """Speichert die Konfiguration"""
        try:
            with open(self.CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=4, ensure_ascii=False)
            return True
        except Exception as e:
            logger.error(f"Fehler beim Speichern der Konfiguration: {e}")
            return False
    
    def get(self, *keys):
        """Holt einen Konfigurationswert"""
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
        """Setzt einen Konfigurationswert"""
        config = self.config
        for key in keys[:-1]:
            config = config.setdefault(key, {})
        config[keys[-1]] = value
        self.save_config()


# ============================================================
# HISTORY MANAGER
# ============================================================

class HistoryManager:
    """Verwaltet die Erkennungshistorie mit Duplikat-Filter"""
    
    HISTORY_FILE = 'data/history/detections.json'
    
    def __init__(self):
        self.history = self.load_history()
        self.lock = threading.Lock()
    
    def load_history(self):
        """Lädt die Historie"""
        if os.path.exists(self.HISTORY_FILE):
            try:
                with open(self.HISTORY_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Fehler beim Laden der Historie: {e}")
        return []
    
    def save_history(self):
        """Speichert die Historie"""
        try:
            with open(self.HISTORY_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.history, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            logger.error(f"Fehler beim Speichern der Historie: {e}")
            return False
    
    def _normalize_plate(self, plate_text):
        """Normalisiert Kennzeichentext für Vergleiche"""
        if not plate_text:
            return ""
        return plate_text.upper().replace(' ', '').replace('-', '').strip()
    
    def _is_duplicate_in_history(self, plate_text, timeout_seconds=60):
        """Prüft ob das Kennzeichen kürzlich in der Historie ist"""
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
        """Fügt einen neuen Eintrag hinzu"""
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
        """Holt alle Einträge"""
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
        """Holt einen Eintrag nach ID"""
        for entry in self.history:
            if entry.get('id') == entry_id:
                return entry
        return None
    
    def delete_entry(self, entry_id):
        """Löscht einen Eintrag"""
        with self.lock:
            self.history = [e for e in self.history if e.get('id') != entry_id]
            self.save_history()
    
    def clear_history(self):
        """Löscht die gesamte Historie"""
        with self.lock:
            self.history = []
            self.save_history()
    
    def search(self, query):
        """Sucht in der Historie"""
        query = self._normalize_plate(query)
        return [e for e in self.history 
                if query in self._normalize_plate(e.get('plate_text', ''))]
    
    def get_statistics(self):
        """Holt Statistiken"""
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
# KENNZEICHEN-DETEKTOR (MIT ANALYSIS AREA SUPPORT)
# ============================================================

class LicensePlateDetector:
    """Haupt-Erkennungsklasse mit verbesserter OCR und Fahrzeugerkennung"""
    
    VEHICLE_CLASSES = {
        2: 'PKW',
        3: 'Motorrad',
        5: 'Bus',
        7: 'LKW'
    }
    
    VEHICLE_CLASSES_EN = {
        2: 'car',
        3: 'motorcycle', 
        5: 'bus',
        7: 'truck'
    }
    
    def __init__(self, config_manager):
        self.config_manager = config_manager
        self.coco_model = None
        self.license_model = None
        self.ocr_reader = None
        self.models_loaded = False
        self.load_lock = threading.Lock()
        self.recent_plates = {}
    
    def load_models(self):
        """Lädt die ML-Modelle"""
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
        """Prüft ob das Kennzeichen kürzlich erkannt wurde"""
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
        """Schätzt die Fahrzeugfarbe"""
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
            logger.error(f"Fehler bei Farbbestimmung: {e}")
            return "unbekannt"
    
    def _preprocess_plate_image(self, plate_image):
        """Erweiterte Vorverarbeitung für bessere OCR"""
        if plate_image is None or plate_image.size == 0:
            return None, []
        
        config = self.config_manager.get('ocr', 'preprocessing') or {}
        
        if not config.get('enabled', True):
            return plate_image, [plate_image]
        
        try:
            processed = plate_image.copy()
            height, width = processed.shape[:2]
            
            logger.debug(f"Original Kennzeichen-Größe: {width}x{height}")
            
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
                
                logger.debug(f"Kennzeichen skaliert: {width}x{height} -> {new_width}x{new_height}")
            
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
                kernel = np.array([[-1, -1, -1],
                                   [-1,  9, -1],
                                   [-1, -1, -1]])
                sharpened = cv2.filter2D(gray, -1, kernel)
                variants.append(sharpened)
            
            if config.get('adaptive_threshold', True):
                thresh1 = cv2.adaptiveThreshold(
                    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                    cv2.THRESH_BINARY, 11, 2
                )
                variants.append(thresh1)
                
                thresh2 = cv2.adaptiveThreshold(
                    gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                    cv2.THRESH_BINARY, 15, 4
                )
                variants.append(thresh2)
                
                _, thresh3 = cv2.threshold(
                    gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                )
                variants.append(thresh3)
            
            if config.get('morphology', True):
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
                morph = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
                variants.append(morph)
            
            inverted = cv2.bitwise_not(gray)
            variants.append(inverted)
            
            if config.get('deskew', True):
                deskewed = self._deskew_image(gray)
                if deskewed is not None:
                    variants.append(deskewed)
            
            return gray, variants
            
        except Exception as e:
            logger.error(f"Fehler bei Bildvorverarbeitung: {e}")
            return plate_image, [plate_image]
    
    def _deskew_image(self, image):
        """Begradigt ein schräges Bild"""
        try:
            coords = np.column_stack(np.where(image > 0))
            if len(coords) < 10:
                return image
                
            angle = cv2.minAreaRect(coords)[-1]
            
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle
            
            if abs(angle) < 0.5 or abs(angle) > 15:
                return image
            
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(image, M, (w, h), 
                                     flags=cv2.INTER_CUBIC,
                                     borderMode=cv2.BORDER_REPLICATE)
            
            return rotated
            
        except Exception as e:
            return image
    
    def _read_plate_enhanced(self, plate_image):
        """Verbesserte OCR mit mehreren Durchgängen"""
        if not self.ocr_reader or plate_image is None or plate_image.size == 0:
            return None, 0
        
        try:
            result = self._preprocess_plate_image(plate_image)
            
            if result is None:
                return None, 0
            
            processed, variants = result
            
            min_confidence = self.config_manager.get('ocr', 'min_confidence') or 0.25
            allowed_chars = self.config_manager.get('ocr', 'allowed_characters') or ''
            active_mode = self.config_manager.get('ocr', 'active_mode') or 'enhanced'
            
            best_result = None
            best_confidence = 0
            
            for i, variant in enumerate(variants[:8]):
                try:
                    results = self.ocr_reader.readtext(variant, detail=1)
                    
                    text, confidence = self._process_ocr_results(results, min_confidence * 0.7, allowed_chars)
                    
                    if text and confidence > best_confidence:
                        best_confidence = confidence
                        best_result = text
                        logger.debug(f"OCR Variante {i}: '{text}' (Konfidenz: {confidence:.2f})")
                    
                    if best_confidence >= 0.85:
                        break
                        
                except Exception as e:
                    logger.debug(f"OCR Fehler bei Variante {i}: {e}")
                    continue
            
            if active_mode in ['enhanced', 'multi_scale'] and best_confidence < 0.7:
                for scale in [1.5, 2.0, 0.75]:
                    try:
                        h, w = processed.shape[:2]
                        scaled = cv2.resize(processed, (int(w * scale), int(h * scale)),
                                           interpolation=cv2.INTER_CUBIC)
                        results = self.ocr_reader.readtext(scaled, detail=1)
                        text, confidence = self._process_ocr_results(results, min_confidence * 0.7, allowed_chars)
                        
                        if text and confidence > best_confidence:
                            best_confidence = confidence
                            best_result = text
                            logger.debug(f"OCR Scale {scale}: '{text}' (Konfidenz: {confidence:.2f})")
                    except:
                        continue
            
            if best_result is None and self.config_manager.get('ocr', 'retry_on_fail'):
                max_retries = self.config_manager.get('ocr', 'max_retries') or 3
                
                for retry in range(max_retries):
                    try:
                        enhanced = self._aggressive_preprocessing(plate_image, retry)
                        results = self.ocr_reader.readtext(enhanced, detail=1)
                        text, confidence = self._process_ocr_results(results, min_confidence * 0.6, allowed_chars)
                        
                        if text and confidence > best_confidence:
                            best_confidence = confidence
                            best_result = text
                            logger.debug(f"OCR Retry {retry}: '{text}' (Konfidenz: {confidence:.2f})")
                            break
                    except:
                        continue
            
            if best_result:
                best_result = self._correct_common_errors(best_result)
            
            return best_result, best_confidence
            
        except Exception as e:
            logger.error(f"OCR Fehler: {e}")
            import traceback
            traceback.print_exc()
            return None, 0
    
    def _aggressive_preprocessing(self, image, level):
        """Aggressive Vorverarbeitung für schwierige Fälle"""
        try:
            scale = 3.0 + level * 0.5
            h, w = image.shape[:2]
            new_w, new_h = int(w * scale), int(h * scale)
            img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img
            
            if level == 0:
                gray = cv2.bilateralFilter(gray, 9, 75, 75)
                kernel = np.array([[-1, -1, -1, -1, -1],
                                   [-1,  2,  2,  2, -1],
                                   [-1,  2,  8,  2, -1],
                                   [-1,  2,  2,  2, -1],
                                   [-1, -1, -1, -1, -1]]) / 8
                gray = cv2.filter2D(gray, -1, kernel)
            
            elif level == 1:
                gray = cv2.GaussianBlur(gray, (3, 3), 0)
                gray = cv2.adaptiveThreshold(gray, 255, 
                                            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY, 11, 2)
            
            elif level >= 2:
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
                gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
            
            return gray
            
        except Exception as e:
            return image
    
    def _process_ocr_results(self, results, min_confidence, allowed_chars):
        """Verarbeitet OCR-Ergebnisse"""
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
        """Korrigiert häufige OCR-Fehler"""
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
            elif char == 'S' and (prev_is_digit and next_is_digit):
                result[i] = '5'
            elif char == 'B' and (prev_is_digit and next_is_digit):
                result[i] = '8'
            elif char == 'D' and next_is_digit and prev_is_digit:
                result[i] = '0'
        
        return ''.join(result)
    
    def _get_analysis_area(self, frame):
        """
        Holt und validiert den Analysebereich
        
        Returns:
            Tuple (enabled, x, y, width, height) oder (False, 0, 0, frame_w, frame_h)
        """
        h, w = frame.shape[:2]
        
        area_enabled = self.config_manager.get('rtsp', 'analysis_area', 'enabled')
        if not area_enabled:
            return False, 0, 0, w, h
        
        area = self.config_manager.get('rtsp', 'analysis_area', 'area')
        if not area:
            return False, 0, 0, w, h
        
        x = int(area.get('x', 0))
        y = int(area.get('y', 0))
        width = int(area.get('width', w))
        height = int(area.get('height', h))
        
        # Grenzen prüfen und korrigieren
        x = max(0, min(x, w - 10))
        y = max(0, min(y, h - 10))
        width = max(10, min(width, w - x))
        height = max(10, min(height, h - y))
        
        return True, x, y, width, height
    
    def process_frame(self, frame, apply_analysis_area=False):
        """
        Verarbeitet einen einzelnen Frame
        
        Args:
            frame: Input Frame (BGR)
            apply_analysis_area: Ob Analysis Area angewendet werden soll
            
        Returns:
            Dictionary mit Erkennungsergebnissen
        """
        if not self.models_loaded:
            self.load_models()
        
        if frame is None or frame.size == 0:
            return {
                'annotated_frame': np.zeros((480, 640, 3), dtype=np.uint8),
                'detections': [],
                'vehicles': [],
                'processing_time': 0,
                'error': 'Invalid frame'
            }
        
        result = {
            'annotated_frame': frame.copy(),
            'detections': [],
            'vehicles': [],
            'processing_time': 0,
            'analysis_area_applied': False
        }
        
        start_time = time.time()
        
        try:
            # ============= ANALYSIS AREA HANDLING =============
            process_frame = frame.copy()
            offset_x, offset_y = 0, 0
            full_frame = frame.copy()
            
            if apply_analysis_area:
                area_enabled, ax, ay, aw, ah = self._get_analysis_area(frame)
                if area_enabled:
                    process_frame = frame[ay:ay+ah, ax:ax+aw].copy()
                    offset_x, offset_y = ax, ay
                    result['analysis_area_applied'] = True
                    logger.debug(f"Analysis Area: x={ax}, y={ay}, w={aw}, h={ah}")
            
            confidence_threshold = self.config_manager.get('detection', 'confidence_threshold') or 0.5
            zoom_enabled = self.config_manager.get('detection', 'zoom_enabled') or True
            zoom_factor = self.config_manager.get('detection', 'zoom_factor') or 2.5
            zoom_padding = self.config_manager.get('detection', 'zoom_padding') or 100
            
            annotated = full_frame.copy()
            detected_vehicles = []
            
            # ============= ANALYSE-BEREICH MARKIEREN =============
            if result['analysis_area_applied']:
                area_enabled, ax, ay, aw, ah = self._get_analysis_area(frame)
                cv2.rectangle(annotated, (ax, ay), (ax + aw, ay + ah), (0, 255, 255), 2)
                cv2.putText(annotated, "Analysebereich", (ax + 5, ay + 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # ============= FAHRZEUGERKENNUNG =============
            if self.coco_model and self.config_manager.get('detection', 'car_detection_enabled'):
                vehicle_results = self.coco_model(process_frame, conf=confidence_threshold, verbose=False)[0]
                
                for detection in vehicle_results.boxes.data.tolist():
                    x1, y1, x2, y2, score, class_id = detection
                    class_id = int(class_id)
                    
                    if class_id in self.VEHICLE_CLASSES:
                        # Offset für Analysis Area hinzufügen
                        x1 = int(x1) + offset_x
                        y1 = int(y1) + offset_y
                        x2 = int(x2) + offset_x
                        y2 = int(y2) + offset_y
                        
                        # Vehicle crop vom Original-Frame
                        crop_x1 = max(0, x1)
                        crop_y1 = max(0, y1)
                        crop_x2 = min(full_frame.shape[1], x2)
                        crop_y2 = min(full_frame.shape[0], y2)
                        
                        vehicle_crop = full_frame[crop_y1:crop_y2, crop_x1:crop_x2].copy()
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
            
            # ============= KENNZEICHENERKENNUNG =============
            if self.license_model:
                frames_to_process = []
                
                if zoom_enabled and detected_vehicles:
                    # Zoome auf jedes erkannte Fahrzeug
                    for vehicle in detected_vehicles:
                        x1, y1, x2, y2 = vehicle['bbox']
                        height, width = full_frame.shape[:2]
                        
                        pad = zoom_padding
                        zx1 = max(0, x1 - pad)
                        zy1 = max(0, y1 - pad)
                        zx2 = min(width, x2 + pad)
                        zy2 = min(height, y2 + pad)
                        
                        vehicle_region = full_frame[zy1:zy2, zx1:zx2]
                        
                        if vehicle_region.size == 0:
                            continue
                        
                        crop_h, crop_w = vehicle_region.shape[:2]
                        min_size = 800
                        scale = max(min_size / max(crop_w, 1), min_size / max(crop_h, 1), zoom_factor)
                        scale = min(scale, 5.0)
                        
                        new_width = int(crop_w * scale)
                        new_height = int(crop_h * scale)
                        
                        vehicle_region_scaled = cv2.resize(
                            vehicle_region, 
                            (new_width, new_height), 
                            interpolation=cv2.INTER_CUBIC
                        )
                        
                        frames_to_process.append({
                            'frame': vehicle_region_scaled,
                            'offset': (zx1, zy1),
                            'scale': scale,
                            'vehicle': vehicle
                        })
                else:
                    # Ohne Zoom: Verarbeite den zugeschnittenen Bereich oder ganzes Frame
                    frames_to_process.append({
                        'frame': process_frame,
                        'offset': (offset_x, offset_y),
                        'scale': 1,
                        'vehicle': None
                    })
                
                # Verarbeite alle Frames
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
                        
                        # Originale Koordinaten zurückrechnen
                        orig_px1 = int(px1 / scale + off_x)
                        orig_py1 = int(py1 / scale + off_y)
                        orig_px2 = int(px2 / scale + off_x)
                        orig_py2 = int(py2 / scale + off_y)
                        
                        plate_w = orig_px2 - orig_px1
                        plate_h = orig_py2 - orig_py1
                        
                        # Kennzeichen aus skaliertem Frame
                        plate_crop_scaled = proc_frame[int(py1):int(py2), int(px1):int(px2)]
                        
                        if plate_crop_scaled.size == 0:
                            continue
                        
                        scaled_h, scaled_w = plate_crop_scaled.shape[:2]
                        
                        # Zusätzlich skalieren für OCR wenn nötig
                        target_height = self.config_manager.get('ocr', 'preprocessing', 'target_height') or 120
                        if scaled_h < target_height:
                            additional_scale = target_height / scaled_h
                            additional_scale = min(additional_scale, 4.0)
                            
                            plate_crop_scaled = cv2.resize(
                                plate_crop_scaled,
                                (int(scaled_w * additional_scale), int(scaled_h * additional_scale)),
                                interpolation=cv2.INTER_CUBIC
                            )
                        
                        # OCR durchführen
                        plate_text, ocr_confidence = self._read_plate_enhanced(plate_crop_scaled)
                        
                        min_save_conf = self.config_manager.get('history', 'min_confidence_to_save') or 0.35
                        
                        if not plate_text or ocr_confidence < min_save_conf:
                            # Markiere unerkannte Kennzeichen orange
                            color = (0, 165, 255)
                            cv2.rectangle(annotated, (orig_px1, orig_py1), (orig_px2, orig_py2), color, 2)
                            continue
                        
                        if self._is_duplicate(plate_text):
                            continue
                        
                        # Markiere erkannte Kennzeichen grün
                        color = (0, 255, 0)
                        cv2.rectangle(annotated, (orig_px1, orig_py1), (orig_px2, orig_py2), color, 3)
                        
                        # Text über Kennzeichen
                        text_size = cv2.getTextSize(plate_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
                        cv2.rectangle(annotated,
                                     (orig_px1, orig_py1 - text_size[1] - 15),
                                     (orig_px1 + text_size[0] + 10, orig_py1),
                                     color, -1)
                        cv2.putText(annotated, plate_text,
                                   (orig_px1 + 5, orig_py1 - 8),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
                        
                        # ============= BILDER SPEICHERN =============
                        plate_image_b64 = None
                        vehicle_image_b64 = None
                        full_frame_b64 = None
                        
                        if self.config_manager.get('detection', 'save_detected_plates'):
                            _, buffer = cv2.imencode('.jpg', plate_crop_scaled, 
                                                    [cv2.IMWRITE_JPEG_QUALITY, 95])
                            plate_image_b64 = base64.b64encode(buffer).decode('utf-8')
                            
                            plate_filename = f"data/plates_detected/{uuid.uuid4()}.jpg"
                            cv2.imwrite(plate_filename, plate_crop_scaled)
                        
                        if self.config_manager.get('detection', 'save_detected_vehicles') and vehicle:
                            vehicle_crop = vehicle.get('crop')
                            if vehicle_crop is not None and vehicle_crop.size > 0:
                                vh, vw = vehicle_crop.shape[:2]
                                if vh < 200:
                                    v_scale = 200 / vh
                                    vehicle_crop = cv2.resize(vehicle_crop, 
                                                             (int(vw * v_scale), int(vh * v_scale)),
                                                             interpolation=cv2.INTER_CUBIC)
                                
                                _, buffer = cv2.imencode('.jpg', vehicle_crop, 
                                                        [cv2.IMWRITE_JPEG_QUALITY, 90])
                                vehicle_image_b64 = base64.b64encode(buffer).decode('utf-8')
                                
                                vehicle_filename = f"data/vehicles_detected/{uuid.uuid4()}.jpg"
                                cv2.imwrite(vehicle_filename, vehicle_crop)
                        
                        if self.config_manager.get('detection', 'save_full_frame'):
                            _, buffer = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
                            full_frame_b64 = base64.b64encode(buffer).decode('utf-8')
                        
                        # ============= ERGEBNIS SPEICHERN =============
                        detection_info = {
                            'plate_text': plate_text,
                            'confidence': ocr_confidence,
                            'plate_bbox': [orig_px1, orig_py1, orig_px2, orig_py2],
                            'plate_score': plate_score,
                            'plate_image_base64': plate_image_b64,
                            'vehicle_image_base64': vehicle_image_b64,
                            'full_frame_base64': full_frame_b64,
                            'vehicle_type': vehicle['type'] if vehicle else 'Unbekannt',
                            'vehicle_type_en': vehicle['type_en'] if vehicle else 'unknown',
                            'vehicle_confidence': vehicle['confidence'] if vehicle else 0,
                            'vehicle_color': vehicle['color'] if vehicle else 'Unbekannt',
                            'plate_size': f"{plate_crop_scaled.shape[1]}x{plate_crop_scaled.shape[0]}",
                            'original_plate_size': f"{plate_w}x{plate_h}"
                        }
                        
                        result['detections'].append(detection_info)
                        logger.info(f"Erkannt: {plate_text} | Fahrzeug: {vehicle['type'] if vehicle else 'N/A'} | Farbe: {vehicle['color'] if vehicle else 'N/A'} | Konfidenz: {ocr_confidence:.2f}")
            
            result['annotated_frame'] = annotated
            
        except Exception as e:
            logger.error(f"Verarbeitungsfehler: {e}")
            import traceback
            traceback.print_exc()
        
        result['processing_time'] = time.time() - start_time
        return result
    
    # Alias für Kompatibilität mit rtsp_handler.py
    def detect(self, frame, source="unknown"):
        """Alias für process_frame - für Kompatibilität"""
        result = self.process_frame(frame, apply_analysis_area=True)
        
        # Konvertiere zu altem Format wenn nötig
        return {
            'plates': result.get('detections', []),
            'cars': result.get('vehicles', []),
            'car_detected': len(result.get('vehicles', [])) > 0,
            'cars_count': len(result.get('vehicles', [])),
            'annotated_image': result.get('annotated_frame'),
            'analysis_area': result.get('analysis_area_applied')
        }
    
    def process_image(self, image_path_or_array):
        """Verarbeitet ein einzelnes Bild"""
        if isinstance(image_path_or_array, str):
            frame = cv2.imread(image_path_or_array)
        else:
            frame = image_path_or_array
        
        if frame is None:
            return None
        
        return self.process_frame(frame)
    
    def process_video(self, video_path, output_path=None, job_id=None):
        """Verarbeitet ein Video mit Fortschrittsanzeige"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            if job_id:
                video_processing_jobs[job_id]['status'] = 'error'
                video_processing_jobs[job_id]['error'] = 'Video konnte nicht geöffnet werden'
            return None
        
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if job_id:
            video_processing_jobs[job_id]['total_frames'] = total_frames
            video_processing_jobs[job_id]['fps'] = fps
            video_processing_jobs[job_id]['resolution'] = f"{width}x{height}"
        
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        all_detections = []
        frame_count = 0
        start_time = time.time()
        
        process_every_n = max(1, fps // 5)
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % process_every_n == 0:
                    result = self.process_frame(frame)
                    processed_frame = result['annotated_frame']
                    
                    if result['detections']:
                        for det in result['detections']:
                            det['frame_number'] = frame_count
                            det['timestamp'] = frame_count / fps
                            all_detections.append(det)
                else:
                    processed_frame = frame
                
                if writer:
                    writer.write(processed_frame)
                
                frame_count += 1
                
                if job_id:
                    progress = int((frame_count / total_frames) * 100)
                    elapsed = time.time() - start_time
                    frames_per_sec = frame_count / elapsed if elapsed > 0 else 0
                    eta = (total_frames - frame_count) / frames_per_sec if frames_per_sec > 0 else 0
                    
                    video_processing_jobs[job_id].update({
                        'progress': progress,
                        'current_frame': frame_count,
                        'detections_count': len(all_detections),
                        'elapsed_time': int(elapsed),
                        'eta': int(eta),
                        'fps_processing': round(frames_per_sec, 1)
                    })
                    
                    if frame_count % 30 == 0:
                        socketio.emit('video_progress', {
                            'job_id': job_id,
                            **video_processing_jobs[job_id]
                        })
        
        except Exception as e:
            logger.error(f"Video-Verarbeitungsfehler: {e}")
            if job_id:
                video_processing_jobs[job_id]['status'] = 'error'
                video_processing_jobs[job_id]['error'] = str(e)
        
        finally:
            cap.release()
            if writer:
                writer.release()
        
        if job_id:
            video_processing_jobs[job_id]['status'] = 'completed'
            video_processing_jobs[job_id]['progress'] = 100
            socketio.emit('video_completed', {
                'job_id': job_id,
                'detections_count': len(all_detections),
                'output_path': output_path
            })
        
        return {
            'total_frames': total_frames,
            'processed_frames': frame_count,
            'detections': all_detections,
            'output_path': output_path,
            'processing_time': time.time() - start_time
        }


# ============================================================
# RTSP STREAM MANAGER
# ============================================================

class RTSPHandler:
    """Handler für RTSP Videostreams"""
    
    def __init__(self, config_manager, history_manager, detector):
        self.config_manager = config_manager
        self.history_manager = history_manager
        self.detector = detector
        
        # Stream-Variablen
        self.cap = None
        self.current_frame = None
        self.annotated_frame = None
        self.frame_lock = threading.Lock()
        
        # Thread-Variablen
        self.capture_thread = None
        self.process_thread = None
        self.running = False
        self.connected = False
        
        # Statistiken
        self.frame_count = 0
        self.detection_count = 0
        self.fps = 0
        self.last_fps_time = time.time()
        self.fps_frame_count = 0
        self.last_error = None
        
        # Frame Queue für Verarbeitung
        self.frame_queue = queue.Queue(maxsize=10)
        
        # Duplikat-Erkennung
        self.recent_plates = {}
        
        logger.info("RTSP Handler initialisiert")
    
    def update_config(self, config):
        """Konfiguration aktualisieren"""
        pass  # Config wird direkt vom config_manager gelesen
    
    def get_rtsp_url(self):
        """RTSP URL aus Konfiguration holen"""
        return self.config_manager.get('rtsp', 'url') or ''
    
    def is_running(self):
        """Prüft ob Stream läuft"""
        return self.running
    
    def is_connected(self):
        """Prüft ob Verbindung besteht"""
        return self.connected
    
    def get_fps(self):
        """Aktuelle FPS zurückgeben"""
        return round(self.fps, 1)
    
    def get_frame_count(self):
        """Anzahl verarbeiteter Frames"""
        return self.frame_count
    
    def get_status(self):
        """Gibt den aktuellen Stream-Status zurück"""
        return {
            'status': 'running' if self.running else 'stopped',
            'connected': self.connected,
            'fps': self.get_fps(),
            'frame_count': self.get_frame_count(),
            'detection_count': self.detection_count,
            'url': self.get_rtsp_url(),
            'error': self.last_error if not self.connected else None,
            'analysis_area_enabled': self.config_manager.get('rtsp', 'analysis_area', 'enabled') or False
        }
    
    def get_current_frame(self):
        """Aktuelles Frame (annotiert) zurückgeben"""
        with self.frame_lock:
            if self.annotated_frame is not None:
                return self.annotated_frame.copy()
            elif self.current_frame is not None:
                return self.current_frame.copy()
        return None
    
    def connect(self):
        """Verbindung zum RTSP Stream herstellen"""
        url = self.get_rtsp_url()
        if not url:
            self.last_error = "Keine RTSP URL konfiguriert"
            logger.warning(self.last_error)
            return False
        
        try:
            logger.info(f"Verbinde zu RTSP: {url}")
            
            # OpenCV VideoCapture mit RTSP
            self.cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
            
            # Puffer-Einstellungen
            buffer_size = self.config_manager.get('rtsp', 'buffer_size') or 1
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, buffer_size)
            
            if self.cap.isOpened():
                # Test-Frame lesen
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    self.connected = True
                    self.last_error = None
                    
                    with self.frame_lock:
                        self.current_frame = frame
                    
                    logger.info(f"RTSP Verbindung hergestellt: {url} - Frame Size: {frame.shape}")
                    return True
                else:
                    self.last_error = "Konnte keinen Frame lesen"
            else:
                self.last_error = "Stream konnte nicht geöffnet werden"
            
            logger.warning(f"Verbindung fehlgeschlagen: {self.last_error}")
            return False
            
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"RTSP Verbindungsfehler: {e}")
            return False
    
    def disconnect(self):
        """Verbindung trennen"""
        if self.cap:
            self.cap.release()
            self.cap = None
        self.connected = False
        logger.info("RTSP Verbindung getrennt")
    
    def start(self):
        """Stream starten"""
        if self.running:
            logger.warning("Stream läuft bereits")
            return True
        
        self.running = True
        self.last_error = None
        
        # Capture Thread starten
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        
        # Processing Thread starten
        self.process_thread = threading.Thread(target=self._process_loop, daemon=True)
        self.process_thread.start()
        
        logger.info("RTSP Handler gestartet")
        return True
    
    def stop(self):
        """Stream stoppen"""
        logger.info("Stoppe RTSP Handler...")
        self.running = False
        
        # Queue leeren
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except:
                pass
        
        # Auf Thread-Ende warten
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2)
        
        if self.process_thread and self.process_thread.is_alive():
            self.process_thread.join(timeout=2)
        
        self.disconnect()
        logger.info("RTSP Handler gestoppt")
    
    def _capture_loop(self):
        """Capture-Schleife für RTSP Stream"""
        reconnect_delay = self.config_manager.get('rtsp', 'reconnect_delay') or 5
        
        while self.running:
            if not self.connected:
                if not self.connect():
                    logger.debug(f"Reconnect in {reconnect_delay} Sekunden...")
                    time.sleep(reconnect_delay)
                    continue
            
            try:
                ret, frame = self.cap.read()
                
                if not ret or frame is None:
                    logger.warning("Frame konnte nicht gelesen werden, reconnecting...")
                    self.disconnect()
                    time.sleep(reconnect_delay)
                    continue
                
                # Frame speichern
                with self.frame_lock:
                    self.current_frame = frame.copy()
                
                # Frame zur Verarbeitung in Queue
                if not self.frame_queue.full():
                    self.frame_queue.put(frame.copy())
                
                # FPS berechnen
                self.fps_frame_count += 1
                current_time = time.time()
                if current_time - self.last_fps_time >= 1.0:
                    self.fps = self.fps_frame_count / (current_time - self.last_fps_time)
                    self.fps_frame_count = 0
                    self.last_fps_time = current_time
                
                time.sleep(0.01)
                    
            except Exception as e:
                logger.error(f"Capture Fehler: {e}")
                self.last_error = str(e)
                self.disconnect()
                time.sleep(reconnect_delay)
    
    def _process_loop(self):
        """Verarbeitungs-Schleife für Nummernschilderkennung"""
        process_interval = self.config_manager.get('detection', 'process_interval') or 0.5
        
        while self.running:
            try:
                # Frame aus Queue holen
                try:
                    frame = self.frame_queue.get(timeout=1)
                except queue.Empty:
                    continue
                
                if frame is None:
                    continue
                
                # Erkennung durchführen
                if self.detector:
                    # Modelle laden falls nötig
                    if not self.detector.models_loaded:
                        self.detector.load_models()
                        time.sleep(1)
                        continue
                    
                    try:
                        # process_frame mit Analysis Area aufrufen
                        results = self.detector.process_frame(frame, apply_analysis_area=True)
                        
                        # Annotiertes Frame speichern
                        with self.frame_lock:
                            annotated = results.get('annotated_frame', frame)
                            
                            # Status-Info einzeichnen
                            status_text = f"FPS: {self.get_fps()} | Frames: {self.frame_count} | Erkennungen: {self.detection_count}"
                            cv2.putText(annotated, status_text, (10, 30),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                            
                            self.annotated_frame = annotated
                        
                        self.frame_count += 1
                        
                        # Erkennungen verarbeiten
                        for detection in results.get('detections', []):
                            if detection.get('plate_text'):
                                self._handle_detection(detection, results)
                                
                    except Exception as e:
                        logger.error(f"Erkennungsfehler: {e}")
                        import traceback
                        traceback.print_exc()
                else:
                    # Wenn kein Detector, nur Frame anzeigen
                    with self.frame_lock:
                        self.annotated_frame = frame
                
                time.sleep(process_interval)
                
            except Exception as e:
                logger.error(f"Processing Fehler: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(1)
    
    def _handle_detection(self, detection, full_results):
        """Verarbeitet erkanntes Nummernschild"""
        plate_text = detection.get('plate_text', '')
        if not plate_text or len(plate_text) < 3:
            return
        
        current_time = time.time()
        
        # Duplikat-Timeout
        timeout = self.config_manager.get('history', 'duplicate_timeout') or 30
        filter_enabled = self.config_manager.get('history', 'filter_duplicates')
        
        normalized = plate_text.upper().replace(' ', '').replace('-', '')
        
        if filter_enabled:
            if normalized in self.recent_plates:
                last_seen = self.recent_plates[normalized]
                if current_time - last_seen < timeout:
                    logger.debug(f"Duplikat übersprungen: {plate_text}")
                    return
        
        self.recent_plates[normalized] = current_time
        
        # Alte Einträge bereinigen
        self.recent_plates = {k: v for k, v in self.recent_plates.items() 
                             if current_time - v < timeout * 2}
        
        # Entry für Historie
        entry = {
            "plate_text": plate_text,
            "confidence": detection.get('confidence', 0),
            "source": "rtsp",
            "plate_image": detection.get('plate_image_base64'),
            "vehicle_image": detection.get('vehicle_image_base64'),
            "full_frame": detection.get('full_frame_base64'),
            "vehicle_type": detection.get('vehicle_type', 'Unbekannt'),
            "vehicle_color": detection.get('vehicle_color', 'Unbekannt'),
        }
        
        saved_entry = self.history_manager.add_entry(entry, check_duplicate=False)
        
        if saved_entry:
            self.detection_count += 1
            logger.info(f"RTSP Erkennung: {plate_text} (Konfidenz: {detection.get('confidence', 0):.2f})")
            
            # WebSocket Event senden
            socketio.emit('plate_detected', {
                'plate_text': plate_text,
                'confidence': detection.get('confidence', 0),
                'vehicle_type': detection.get('vehicle_type', 'Unbekannt'),
                'vehicle_color': detection.get('vehicle_color', 'Unbekannt'),
                'timestamp': datetime.now().isoformat()
            })


# ============================================================
# GLOBALE INSTANZEN
# ============================================================

config_manager = ConfigManager()
history_manager = HistoryManager()
detector = LicensePlateDetector(config_manager)
stream_manager = RTSPHandler(config_manager, history_manager, detector)

def init_models():
    """Lädt Modelle im Hintergrund"""
    detector.load_models()

threading.Thread(target=init_models, daemon=True).start()


# ============================================================
# FLASK ROUTEN - SEITEN
# ============================================================

@app.route('/')
def index():
    """Startseite - Redirect zu Dashboard"""
    return render_template('index.html', 
                          page='dashboard',
                          stats=history_manager.get_statistics(),
                          stream_status=stream_manager.get_status(),
                          config=config_manager.config)


@app.route('/dashboard')
def dashboard():
    """Dashboard Seite"""
    return render_template('dashboard.html',
                          page='dashboard',
                          stats=history_manager.get_statistics(),
                          stream_status=stream_manager.get_status(),
                          config=config_manager.config)


@app.route('/history')
def history():
    """Historie Seite"""
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
    """RTSP Einstellungen Seite"""
    return render_template('rtsp_settings.html',
                          page='rtsp',
                          config=config_manager.config.get('rtsp', {}),
                          stream_status=stream_manager.get_status())


@app.route('/settings')
def settings():
    """Allgemeine Einstellungen Seite"""
    return render_template('settings.html',
                          page='settings',
                          config=config_manager.config)


@app.route('/test')
def test_page():
    """Test-Seite für Bild/Video Upload"""
    return render_template('test.html', 
                          page='test',
                          jobs=video_processing_jobs)


@app.route('/live')
def live_view():
    """Live Stream Ansicht"""
    return render_template('live.html',
                          page='live',
                          stream_status=stream_manager.get_status(),
                          config=config_manager.config)


# ============================================================
# API ROUTEN - STREAM KONTROLLE
# ============================================================

@app.route('/api/stream/start', methods=['POST'])
def api_stream_start():
    """Startet den RTSP Stream"""
    success = stream_manager.start()
    return jsonify({
        'success': success,
        'status': stream_manager.get_status()
    })


@app.route('/api/stream/stop', methods=['POST'])
def api_stream_stop():
    """Stoppt den RTSP Stream"""
    stream_manager.stop()
    return jsonify({
        'success': True,
        'status': stream_manager.get_status()
    })


@app.route('/api/stream/status')
def api_stream_status():
    """Holt den Stream Status"""
    return jsonify(stream_manager.get_status())


@app.route('/api/stream/feed')
def stream_feed():
    """MJPEG Stream für Live-Ansicht"""
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
    
    return Response(generate(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')


# ============================================================
# API ROUTEN - KONFIGURATION
# ============================================================

@app.route('/api/config', methods=['GET'])
def api_get_config():
    """Holt die gesamte Konfiguration"""
    return jsonify(config_manager.config)


@app.route('/api/config', methods=['POST'])
def api_save_config():
    """Speichert die Konfiguration"""
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
    """Speichert RTSP Einstellungen"""
    try:
        data = request.json
        config_manager.config['rtsp'].update(data)
        config_manager.save_config()
        
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/config/detection', methods=['POST'])
def api_save_detection_config():
    """Speichert Erkennungs-Einstellungen"""
    try:
        data = request.json
        config_manager.config['detection'].update(data)
        config_manager.save_config()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/config/ocr', methods=['POST'])
def api_save_ocr_config():
    """Speichert OCR Einstellungen"""
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
    """Speichert Historie-Einstellungen"""
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
    """Holt die Historie"""
    limit = request.args.get('limit', 100, type=int)
    offset = request.args.get('offset', 0, type=int)
    search = request.args.get('search', '')
    unique_only = request.args.get('unique', 'false').lower() == 'true'
    
    if search:
        entries = history_manager.search(search)
    else:
        entries = history_manager.get_all(limit=limit, offset=offset, unique_only=unique_only)
    
    return jsonify({
        'entries': entries,
        'total': len(history_manager.history)
    })


@app.route('/api/history/<entry_id>', methods=['GET'])
def api_get_history_entry(entry_id):
    """Holt einen einzelnen Historie-Eintrag"""
    entry = history_manager.get_by_id(entry_id)
    if entry:
        return jsonify(entry)
    return jsonify({'error': 'Nicht gefunden'}), 404


@app.route('/api/history/<entry_id>', methods=['DELETE'])
def api_delete_history_entry(entry_id):
    """Löscht einen Historie-Eintrag"""
    history_manager.delete_entry(entry_id)
    return jsonify({'success': True})


@app.route('/api/history/clear', methods=['POST'])
def api_clear_history():
    """Löscht die gesamte Historie"""
    history_manager.clear_history()
    return jsonify({'success': True})


@app.route('/api/history/statistics')
def api_history_statistics():
    """Holt Statistiken"""
    return jsonify(history_manager.get_statistics())


@app.route('/api/history/export')
def api_export_history():
    """Exportiert die Historie als JSON"""
    return jsonify(history_manager.history)


# ============================================================
# API ROUTEN - BILD/VIDEO VERARBEITUNG
# ============================================================

@app.route('/api/process/image', methods=['POST'])
def api_process_image():
    """Verarbeitet ein hochgeladenes Bild"""
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
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/process/video', methods=['POST'])
def api_process_video():
    """Verarbeitet ein hochgeladenes Video"""
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'Keine Datei'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'Keine Datei ausgewählt'}), 400
    
    try:
        job_id = str(uuid.uuid4())
        input_path = f"uploads/videos/{job_id}_input.mp4"
        output_path = f"uploads/processed/{job_id}_output.mp4"
        
        file.save(input_path)
        
        video_processing_jobs[job_id] = {
            'status': 'processing',
            'progress': 0,
            'current_frame': 0,
            'total_frames': 0,
            'detections_count': 0,
            'elapsed_time': 0,
            'eta': 0,
            'fps_processing': 0,
            'filename': file.filename,
            'input_path': input_path,
            'output_path': output_path,
            'created_at': datetime.now().isoformat()
        }
        
        def process_video_async():
            try:
                result = detector.process_video(input_path, output_path, job_id)
                
                if result and result['detections']:
                    for detection in result['detections']:
                        if detection.get('plate_text'):
                            entry = {
                                'plate_text': detection['plate_text'],
                                'confidence': detection.get('confidence', 0),
                                'source': 'video_upload',
                                'filename': file.filename,
                                'frame_number': detection.get('frame_number', 0),
                                'video_timestamp': detection.get('timestamp', 0),
                                'plate_image': detection.get('plate_image_base64'),
                                'vehicle_image': detection.get('vehicle_image_base64'),
                                'vehicle_type': detection.get('vehicle_type', 'Unbekannt'),
                                'vehicle_color': detection.get('vehicle_color', 'Unbekannt'),
                            }
                            history_manager.add_entry(entry, check_duplicate=True)
                
            except Exception as e:
                logger.error(f"Async Video-Verarbeitung Fehler: {e}")
                video_processing_jobs[job_id]['status'] = 'error'
                video_processing_jobs[job_id]['error'] = str(e)
        
        threading.Thread(target=process_video_async, daemon=True).start()
        
        return jsonify({
            'success': True,
            'job_id': job_id,
            'message': 'Video wird verarbeitet...'
        })
        
    except Exception as e:
        logger.error(f"Videoverarbeitung Fehler: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/process/video/<job_id>/status')
def api_video_job_status(job_id):
    """Holt den Status eines Video-Verarbeitungsjobs"""
    if job_id in video_processing_jobs:
        return jsonify(video_processing_jobs[job_id])
    return jsonify({'error': 'Job nicht gefunden'}), 404


@app.route('/api/process/video/<job_id>/output')
def api_get_processed_video(job_id):
    """Liefert das verarbeitete Video"""
    output_path = f"uploads/processed/{job_id}_output.mp4"
    if os.path.exists(output_path):
        return send_from_directory('uploads/processed', f"{job_id}_output.mp4",
                                  mimetype='video/mp4')
    return jsonify({'error': 'Video nicht gefunden'}), 404


@app.route('/api/process/jobs')
def api_get_all_jobs():
    """Holt alle Video-Verarbeitungsjobs"""
    return jsonify(video_processing_jobs)


@app.route('/api/process/jobs/<job_id>', methods=['DELETE'])
def api_delete_job(job_id):
    """Löscht einen Job und zugehörige Dateien"""
    if job_id in video_processing_jobs:
        job = video_processing_jobs[job_id]
        
        for path in [job.get('input_path'), job.get('output_path')]:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except:
                    pass
        
        del video_processing_jobs[job_id]
        return jsonify({'success': True})
    
    return jsonify({'error': 'Job nicht gefunden'}), 404


# ============================================================
# API ROUTEN - UTILITIES
# ============================================================

@app.route('/api/models/status')
def api_models_status():
    """Status der geladenen Modelle"""
    return jsonify({
        'loaded': detector.models_loaded,
        'coco_model': detector.coco_model is not None,
        'license_model': detector.license_model is not None,
        'ocr_reader': detector.ocr_reader is not None
    })


@app.route('/api/models/reload', methods=['POST'])
def api_reload_models():
    """Lädt die Modelle neu"""
    detector.models_loaded = False
    detector.coco_model = None
    detector.license_model = None
    detector.ocr_reader = None
    
    threading.Thread(target=detector.load_models, daemon=True).start()
    return jsonify({'success': True, 'message': 'Modelle werden neu geladen...'})


@app.route('/api/system/info')
def api_system_info():
    """System-Informationen"""
    import platform
    
    return jsonify({
        'platform': platform.system(),
        'python_version': platform.python_version(),
        'opencv_version': cv2.__version__,
        'models_loaded': detector.models_loaded,
        'stream_status': stream_manager.get_status(),
        'history_count': len(history_manager.history),
        'active_jobs': len([j for j in video_processing_jobs.values() if j['status'] == 'processing'])
    })


# ============================================================
# API ROUTEN - LETZTE ERKENNUNG (FÜR HOME ASSISTANT)
# ============================================================

@app.route('/api/latest')
def api_latest_detection():
    """Holt die letzte Erkennung als JSON"""
    entries = history_manager.get_all(limit=1)
    
    if entries and len(entries) > 0:
        return jsonify(entries[0])
    
    return jsonify({'error': 'Keine Erkennung vorhanden', 'plate_text': None})


@app.route('/api/latest/plate')
def api_latest_plate_text():
    """Holt nur den Kennzeichen-Text der letzten Erkennung"""
    entries = history_manager.get_all(limit=1)
    
    if entries and len(entries) > 0:
        entry = entries[0]
        return jsonify({
            'plate_text': entry.get('plate_text', ''),
            'confidence': entry.get('confidence', 0),
            'vehicle_type': entry.get('vehicle_type', 'unknown'),
            'vehicle_color': entry.get('vehicle_color', 'unknown'),
            'timestamp': entry.get('timestamp', ''),
            'source': entry.get('source', '')
        })
    
    return jsonify({
        'plate_text': '',
        'confidence': 0,
        'vehicle_type': 'unknown',
        'vehicle_color': 'unknown',
        'timestamp': '',
        'source': ''
    })


@app.route('/api/latest/plate/image')
def api_latest_plate_image():
    """Liefert das Kennzeichen-Bild der letzten Erkennung als JPEG"""
    entries = history_manager.get_all(limit=1)
    
    if entries and len(entries) > 0:
        entry = entries[0]
        plate_image = entry.get('plate_image')
        
        if plate_image:
            try:
                image_data = base64.b64decode(plate_image)
                return Response(image_data, mimetype='image/jpeg')
            except Exception as e:
                logger.error(f"Fehler beim Dekodieren des Kennzeichenbildes: {e}")
    
    placeholder = create_placeholder_image("Kein Kennzeichen", 400, 100)
    return Response(placeholder, mimetype='image/jpeg')


@app.route('/api/latest/vehicle/image')
def api_latest_vehicle_image():
    """Liefert das Fahrzeug-Bild der letzten Erkennung als JPEG"""
    entries = history_manager.get_all(limit=1)
    
    if entries and len(entries) > 0:
        entry = entries[0]
        vehicle_image = entry.get('vehicle_image')
        
        if vehicle_image:
            try:
                image_data = base64.b64decode(vehicle_image)
                return Response(image_data, mimetype='image/jpeg')
            except Exception as e:
                logger.error(f"Fehler beim Dekodieren des Fahrzeugbildes: {e}")
    
    placeholder = create_placeholder_image("Kein Fahrzeug", 640, 480)
    return Response(placeholder, mimetype='image/jpeg')


@app.route('/api/latest/full/image')
def api_latest_full_image():
    """Liefert das Vollbild der letzten Erkennung als JPEG"""
    entries = history_manager.get_all(limit=1)
    
    if entries and len(entries) > 0:
        entry = entries[0]
        full_frame = entry.get('full_frame')
        
        if full_frame:
            try:
                image_data = base64.b64decode(full_frame)
                return Response(image_data, mimetype='image/jpeg')
            except Exception as e:
                logger.error(f"Fehler beim Dekodieren des Vollbildes: {e}")
    
    placeholder = create_placeholder_image("Kein Bild", 1280, 720)
    return Response(placeholder, mimetype='image/jpeg')


def create_placeholder_image(text, width, height):
    """Erstellt ein Placeholder-Bild mit Text"""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = (30, 30, 30)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = min(width, height) / 300
    thickness = max(1, int(font_scale * 2))
    
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = (width - text_size[0]) // 2
    text_y = (height + text_size[1]) // 2
    
    cv2.putText(img, text, (text_x, text_y), font, font_scale, (100, 100, 100), thickness)
    
    _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return buffer.tobytes()


# ============================================================
# SEITEN-ROUTE FÜR LETZTE ERKENNUNG
# ============================================================

@app.route('/latest')
def latest_detection_page():
    """Seite für die letzte Erkennung"""
    return render_template('latest.html', page='latest')


# ============================================================
# WEBSOCKET EVENTS
# ============================================================

@socketio.on('connect')
def handle_connect():
    """Client verbindet sich"""
    emit('connected', {
        'status': 'ok',
        'stream_status': stream_manager.get_status()
    })
    logger.info("WebSocket Client verbunden")


@socketio.on('disconnect')
def handle_disconnect():
    """Client trennt sich"""
    logger.info("WebSocket Client getrennt")


@socketio.on('request_frame')
def handle_frame_request():
    """Client fordert aktuellen Frame an"""
    frame = stream_manager.get_current_frame()
    if frame is not None:
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        frame_b64 = base64.b64encode(buffer).decode('utf-8')
        emit('frame', {'image': frame_b64})


@socketio.on('get_job_status')
def handle_job_status(data):
    """Client fragt Job-Status ab"""
    job_id = data.get('job_id')
    if job_id and job_id in video_processing_jobs:
        emit('job_status', {
            'job_id': job_id,
            **video_processing_jobs[job_id]
        })


# ============================================================
# FEHLERHANDLER
# ============================================================

@app.errorhandler(404)
def page_not_found(e):
    """404 Fehlerseite"""
    if request.path.startswith('/api/'):
        return jsonify({'error': 'Nicht gefunden'}), 404
    return render_template('404.html', page='error'), 404


@app.errorhandler(500)
def internal_error(e):
    """500 Fehlerseite"""
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
    ║     Version 2.1 - Fixed RTSP & Analysis Area             ║
    ╠══════════════════════════════════════════════════════════╣
    ║     Dashboard:     http://localhost:5000                 ║
    ║     Live Stream:   http://localhost:5000/live            ║
    ║     History:       http://localhost:5000/history         ║
    ║     RTSP Settings: http://localhost:5000/rtsp-settings   ║
    ║     Settings:      http://localhost:5000/settings        ║
    ║     Test:          http://localhost:5000/test            ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    socketio.run(app, 
                 host='0.0.0.0', 
                 port=5000, 
                 debug=config_manager.get('general', 'debug_mode') or False,
                 allow_unsafe_werkzeug=True)
