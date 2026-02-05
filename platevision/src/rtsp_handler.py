#!/usr/bin/env python3
"""
RTSP Stream Handler
Verwaltet RTSP Videostreams im Hintergrund
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
        self.plate_cooldown = 30  # Sekunden bis gleiches Kennzeichen erneut gespeichert wird
        
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
            
            # Timeout setzen
            self.cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000)
            self.cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
            
            if self.cap.isOpened():
                # Test-Frame lesen
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    self.connected = True
                    self.last_error = None
                    
                    # Initiales Frame speichern
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
        
        # Prüfe ob RTSP aktiviert ist
        if not self.config_manager.get('rtsp', 'enabled'):
            logger.warning("RTSP ist nicht aktiviert in der Konfiguration")
            # Trotzdem starten, falls gewünscht
        
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
    
    def _apply_analysis_area(self, frame):
        """
        Wendet den Analysebereich auf das Frame an
        
        Args:
            frame: Eingabe-Frame
            
        Returns:
            Tuple (cropped_frame, offset_x, offset_y, annotated_frame)
        """
        h, w = frame.shape[:2]
        annotated = frame.copy()
        
        area_enabled = self.config_manager.get('rtsp', 'analysis_area', 'enabled')
        
        if not area_enabled:
            return frame, 0, 0, annotated
        
        area = self.config_manager.get('rtsp', 'analysis_area', 'area')
        
        if not area:
            return frame, 0, 0, annotated
        
        # Koordinaten extrahieren
        x = int(area.get('x', 0))
        y = int(area.get('y', 0))
        width = int(area.get('width', w))
        height = int(area.get('height', h))
        
        # Grenzen prüfen
        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))
        width = max(10, min(width, w - x))
        height = max(10, min(height, h - y))
        
        # Analysebereich auf annotiertem Frame markieren
        cv2.rectangle(annotated, (x, y), (x + width, y + height), (0, 255, 255), 2)
        cv2.putText(annotated, "Analysebereich", (x + 5, y + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Bereich ausschneiden
        cropped = frame[y:y+height, x:x+width].copy()
        
        logger.debug(f"Analysis Area angewendet: x={x}, y={y}, w={width}, h={height}")
        
        return cropped, x, y, annotated
    
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
                
                # Kleine Pause um CPU zu schonen
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
                
                # Analysebereich anwenden
                cropped_frame, offset_x, offset_y, base_annotated = self._apply_analysis_area(frame)
                
                # Prüfen ob Frame gültig ist
                if cropped_frame is None or cropped_frame.size == 0:
                    logger.warning("Ungültiger Frame nach Area-Zuschnitt")
                    continue
                
                # Erkennung durchführen
                if self.detector and self.detector.models_loaded:
                    try:
                        # process_frame auf zugeschnittenem Bereich aufrufen
                        results = self.detector.process_frame(cropped_frame)
                        
                        # Annotiertes Frame erstellen
                        annotated = base_annotated.copy()
                        
                        # Erkennungen auf Original-Frame übertragen (mit Offset)
                        if results.get('detections'):
                            for detection in results['detections']:
                                # Bounding Box mit Offset anpassen
                                bbox = detection.get('plate_bbox', [])
                                if len(bbox) == 4:
                                    px1, py1, px2, py2 = bbox
                                    # Offset addieren
                                    px1 += offset_x
                                    py1 += offset_y
                                    px2 += offset_x
                                    py2 += offset_y
                                    
                                    # Auf annotiertem Frame zeichnen
                                    cv2.rectangle(annotated, (px1, py1), (px2, py2), (0, 255, 0), 3)
                                    
                                    plate_text = detection.get('plate_text', '')
                                    if plate_text:
                                        # Hintergrund für Text
                                        text_size = cv2.getTextSize(plate_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
                                        cv2.rectangle(annotated,
                                                     (px1, py1 - text_size[1] - 15),
                                                     (px1 + text_size[0] + 10, py1),
                                                     (0, 255, 0), -1)
                                        cv2.putText(annotated, plate_text,
                                                   (px1 + 5, py1 - 8),
                                                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
                        
                        # Fahrzeuge einzeichnen (mit Offset)
                        if results.get('vehicles'):
                            for vehicle in results['vehicles']:
                                bbox = vehicle.get('bbox', [])
                                if len(bbox) == 4:
                                    vx1, vy1, vx2, vy2 = bbox
                                    vx1 += offset_x
                                    vy1 += offset_y
                                    vx2 += offset_x
                                    vy2 += offset_y
                                    
                                    cv2.rectangle(annotated, (vx1, vy1), (vx2, vy2), (255, 0, 0), 2)
                                    label = f"{vehicle.get('type', 'Fahrzeug')}"
                                    cv2.putText(annotated, label, (vx1, vy1 - 10),
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                        
                        # Status-Info einzeichnen
                        status_text = f"FPS: {self.get_fps()} | Erkennungen: {self.detection_count}"
                        cv2.putText(annotated, status_text, (10, 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        
                        # Annotiertes Frame speichern
                        with self.frame_lock:
                            self.annotated_frame = annotated
                        
                        self.frame_count += 1
                        
                        # Erkannte Nummernschilder verarbeiten
                        for detection in results.get('detections', []):
                            plate_text = detection.get('plate_text')
                            if plate_text:
                                self._handle_detection(detection, results)
                                
                    except Exception as e:
                        logger.error(f"Erkennungsfehler: {e}")
                        import traceback
                        traceback.print_exc()
                else:
                    # Wenn Modelle noch nicht geladen, nur Frame anzeigen
                    with self.frame_lock:
                        self.annotated_frame = base_annotated
                        cv2.putText(self.annotated_frame, "Modelle werden geladen...", (10, 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                
                time.sleep(process_interval)
                
            except Exception as e:
                logger.error(f"Processing Fehler: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(1)
    
    def _handle_detection(self, detection, full_results):
        """
        Verarbeitet erkanntes Nummernschild
        
        Args:
            detection: Dictionary mit Kennzeichen-Infos
            full_results: Vollständige Erkennungsergebnisse
        """
        plate_text = detection.get('plate_text', '')
        if not plate_text or len(plate_text) < 3:
            return
        
        current_time = time.time()
        
        # Duplikat-Timeout aus Konfiguration
        timeout = self.config_manager.get('history', 'duplicate_timeout') or 30
        filter_enabled = self.config_manager.get('history', 'filter_duplicates')
        
        # Normalisieren für Vergleich
        normalized = plate_text.upper().replace(' ', '').replace('-', '')
        
        if filter_enabled:
            # Prüfen ob Kennzeichen kürzlich erkannt wurde
            if normalized in self.recent_plates:
                last_seen = self.recent_plates[normalized]
                if current_time - last_seen < timeout:
                    logger.debug(f"Duplikat übersprungen: {plate_text}")
                    return
        
        # Zeitstempel aktualisieren
        self.recent_plates[normalized] = current_time
        
        # Alte Einträge bereinigen
        self.recent_plates = {k: v for k, v in self.recent_plates.items() 
                             if current_time - v < timeout * 2}
        
        # Entry für Historie erstellen
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
            logger.info(f"RTSP Erkennung gespeichert: {plate_text} (Konfidenz: {detection.get('confidence', 0):.2f})")
