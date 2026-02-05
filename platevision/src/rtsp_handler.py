#!/usr/bin/env python3
"""
RTSP Stream Handler
Verwaltet RTSP Videostreams im Hintergrund
Version 2.1 - Mit korrekter Analysis Area Unterstützung
"""

import cv2
import numpy as np
import threading
import time
import queue
import base64
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class RTSPHandler:
    """Handler für RTSP Videostreams"""
    
    def __init__(self, config_manager, history_manager, detector):
        """
        Initialisiert den RTSP Handler
        
        Args:
            config_manager: ConfigManager Instanz
            history_manager: HistoryManager Instanz
            detector: LicensePlateDetector Instanz
        """
        self.config_manager = config_manager
        self.history_manager = history_manager
        self.detector = detector
        
        # Stream-Variablen
        self.cap = None
        self.current_frame = None
        self.annotated_frame = None
        self.frame_lock = threading.Lock()
        
        # Stream Auflösung (wird beim Connect gesetzt)
        self.stream_width = 1280
        self.stream_height = 720
        
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
    
    def get_stream_resolution(self):
        """Gibt die aktuelle Stream-Auflösung zurück"""
        return {
            'width': self.stream_width,
            'height': self.stream_height
        }
    
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
            'resolution': self.get_stream_resolution(),
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
    
    def get_raw_frame(self):
        """Aktuelles rohes Frame (ohne Annotationen) zurückgeben"""
        with self.frame_lock:
            if self.current_frame is not None:
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
                    
                    # Stream-Auflösung speichern
                    self.stream_height, self.stream_width = frame.shape[:2]
                    
                    with self.frame_lock:
                        self.current_frame = frame
                    
                    logger.info(f"RTSP Verbindung hergestellt: {url} - Auflösung: {self.stream_width}x{self.stream_height}")
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
    
    def _get_analysis_area(self, frame_height, frame_width):
        """
        Holt und validiert den Analysebereich
        
        Returns:
            Tuple (enabled, x, y, width, height)
        """
        area_enabled = self.config_manager.get('rtsp', 'analysis_area', 'enabled')
        if not area_enabled:
            return False, 0, 0, frame_width, frame_height
        
        area = self.config_manager.get('rtsp', 'analysis_area', 'area')
        if not area:
            return False, 0, 0, frame_width, frame_height
        
        # Koordinaten aus Config (können Float sein)
        x = int(float(area.get('x', 0)))
        y = int(float(area.get('y', 0)))
        width = int(float(area.get('width', frame_width)))
        height = int(float(area.get('height', frame_height)))
        
        # Grenzen prüfen und korrigieren
        x = max(0, min(x, frame_width - 10))
        y = max(0, min(y, frame_height - 10))
        width = max(10, min(width, frame_width - x))
        height = max(10, min(height, frame_height - y))
        
        logger.debug(f"Analysis Area: x={x}, y={y}, w={width}, h={height} (Frame: {frame_width}x{frame_height})")
        
        return True, x, y, width, height
    
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
                
                # Stream-Auflösung aktualisieren falls sich was geändert hat
                h, w = frame.shape[:2]
                if w != self.stream_width or h != self.stream_height:
                    self.stream_width = w
                    self.stream_height = h
                    logger.info(f"Stream-Auflösung geändert: {w}x{h}")
                
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
                
                frame_h, frame_w = frame.shape[:2]
                
                # Erkennung durchführen
                if self.detector:
                    # Modelle laden falls nötig
                    if not self.detector.models_loaded:
                        self.detector.load_models()
                        time.sleep(1)
                        continue
                    
                    try:
                        # Analysis Area holen
                        area_enabled, ax, ay, aw, ah = self._get_analysis_area(frame_h, frame_w)
                        
                        # Frame für Verarbeitung vorbereiten
                        if area_enabled:
                            # Bereich ausschneiden
                            process_frame = frame[ay:ay+ah, ax:ax+aw].copy()
                            offset_x, offset_y = ax, ay
                        else:
                            process_frame = frame
                            offset_x, offset_y = 0, 0
                        
                        # Erkennung auf (zugeschnittenem) Frame
                        results = self.detector.process_frame(process_frame, apply_analysis_area=False)
                        
                        # Annotiertes Frame erstellen
                        annotated = frame.copy()
                        
                        # Analysis Area einzeichnen
                        if area_enabled:
                            cv2.rectangle(annotated, (ax, ay), (ax + aw, ay + ah), (0, 255, 255), 2)
                            cv2.putText(annotated, "Analysebereich", (ax + 5, ay + 25),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        
                        # Fahrzeuge mit Offset einzeichnen
                        for vehicle in results.get('vehicles', []):
                            bbox = vehicle.get('bbox', [])
                            if len(bbox) == 4:
                                vx1, vy1, vx2, vy2 = bbox
                                # Offset addieren
                                vx1 += offset_x
                                vy1 += offset_y
                                vx2 += offset_x
                                vy2 += offset_y
                                
                                cv2.rectangle(annotated, (vx1, vy1), (vx2, vy2), (255, 0, 0), 2)
                                label = f"{vehicle.get('type', 'Fahrzeug')} ({vehicle.get('color', '')})"
                                cv2.putText(annotated, label, (vx1, vy1 - 10),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                        
                        # Erkennungen mit Offset einzeichnen
                        for detection in results.get('detections', []):
                            bbox = detection.get('plate_bbox', [])
                            if len(bbox) == 4:
                                px1, py1, px2, py2 = bbox
                                # Offset addieren
                                px1 += offset_x
                                py1 += offset_y
                                px2 += offset_x
                                py2 += offset_y
                                
                                # Aktualisiere bbox für Historie
                                detection['plate_bbox'] = [px1, py1, px2, py2]
                                
                                plate_text = detection.get('plate_text', '')
                                if plate_text:
                                    cv2.rectangle(annotated, (px1, py1), (px2, py2), (0, 255, 0), 3)
                                    text_size = cv2.getTextSize(plate_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
                                    cv2.rectangle(annotated,
                                                 (px1, py1 - text_size[1] - 15),
                                                 (px1 + text_size[0] + 10, py1),
                                                 (0, 255, 0), -1)
                                    cv2.putText(annotated, plate_text,
                                               (px1 + 5, py1 - 8),
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
                                else:
                                    cv2.rectangle(annotated, (px1, py1), (px2, py2), (0, 165, 255), 2)
                        
                        # Status-Info einzeichnen
                        status_text = f"FPS: {self.get_fps()} | Frames: {self.frame_count} | Erkennungen: {self.detection_count}"
                        cv2.putText(annotated, status_text, (10, 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        
                        if area_enabled:
                            area_text = f"Area: {ax},{ay} {aw}x{ah}"
                            cv2.putText(annotated, area_text, (10, 60),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                        
                        # Annotiertes Frame speichern
                        with self.frame_lock:
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
            
            # WebSocket Event senden (falls socketio verfügbar)
            try:
                from flask_socketio import emit
                emit('plate_detected', {
                    'plate_text': plate_text,
                    'confidence': detection.get('confidence', 0),
                    'vehicle_type': detection.get('vehicle_type', 'Unbekannt'),
                    'vehicle_color': detection.get('vehicle_color', 'Unbekannt'),
                    'timestamp': datetime.now().isoformat()
                }, broadcast=True, namespace='/')
            except:
                pass  # SocketIO nicht verfügbar oder Fehler
