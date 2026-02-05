#!/usr/bin/env python3
"""
RTSP Stream Handler
Verwaltet RTSP Videostreams im Hintergrund
"""

import cv2
import threading
import time
import queue
from datetime import datetime


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
        self.fps = 0
        self.last_fps_time = time.time()
        self.fps_frame_count = 0
        
        # Frame Queue für Verarbeitung
        self.frame_queue = queue.Queue(maxsize=10)
        
        # Duplikat-Erkennung
        self.recent_plates = {}
        self.plate_cooldown = 30  # Sekunden bis gleiches Kennzeichen erneut gespeichert wird
    
    def update_config(self, config):
        """Konfiguration aktualisieren"""
        self.config = config
    
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
            'url': self.get_rtsp_url(),
            'error': None if self.connected else 'Keine Verbindung'
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
            print("Keine RTSP URL konfiguriert")
            return False
        
        try:
            # OpenCV VideoCapture mit RTSP
            self.cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
            
            # Puffer-Einstellungen
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 
                        self.config_manager.get('rtsp', 'buffer_size') or 1)
            
            if self.cap.isOpened():
                # Test-Frame lesen
                ret, frame = self.cap.read()
                if ret:
                    self.connected = True
                    print(f"RTSP Verbindung hergestellt: {url}")
                    return True
            
            print(f"Konnte keine Verbindung herstellen: {url}")
            return False
            
        except Exception as e:
            print(f"RTSP Verbindungsfehler: {e}")
            return False
    
    def disconnect(self):
        """Verbindung trennen"""
        if self.cap:
            self.cap.release()
            self.cap = None
        self.connected = False
    
    def start(self):
        """Stream starten"""
        if self.running:
            return
        
        self.running = True
        
        # Capture Thread starten
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        
        # Processing Thread starten
        self.process_thread = threading.Thread(target=self._process_loop, daemon=True)
        self.process_thread.start()
        
        print("RTSP Handler gestartet")
    
    def stop(self):
        """Stream stoppen"""
        self.running = False
        
        # Auf Thread-Ende warten
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2)
        
        if self.process_thread and self.process_thread.is_alive():
            self.process_thread.join(timeout=2)
        
        self.disconnect()
        print("RTSP Handler gestoppt")
    
    def _capture_loop(self):
        """Capture-Schleife für RTSP Stream"""
        reconnect_delay = self.config_manager.get('rtsp', 'reconnect_delay') or 5
        
        while self.running:
            if not self.connected:
                if not self.connect():
                    time.sleep(reconnect_delay)
                    continue
            
            try:
                ret, frame = self.cap.read()
                
                if not ret:
                    print("Frame konnte nicht gelesen werden, reconnecting...")
                    self.disconnect()
                    time.sleep(reconnect_delay)
                    continue
                
                # Frame speichern
                with self.frame_lock:
                    self.current_frame = frame
                
                # Frame zur Verarbeitung in Queue
                if not self.frame_queue.full():
                    self.frame_queue.put(frame)
                
                # FPS berechnen
                self.fps_frame_count += 1
                current_time = time.time()
                if current_time - self.last_fps_time >= 1.0:
                    self.fps = self.fps_frame_count / (current_time - self.last_fps_time)
                    self.fps_frame_count = 0
                    self.last_fps_time = current_time
                    
            except Exception as e:
                print(f"Capture Fehler: {e}")
                self.disconnect()
                time.sleep(reconnect_delay)
    
    def _process_loop(self):
        """Verarbeitungs-Schleife für Nummernschilderkennung"""
        process_interval = 0.5  # Alle 0.5 Sekunden ein Frame verarbeiten
        
        while self.running:
            try:
                # Frame aus Queue holen
                try:
                    frame = self.frame_queue.get(timeout=1)
                except queue.Empty:
                    continue
                
                # Erkennung durchführen
                if self.detector:
                    # Update detector config with current RTSP config
                    if hasattr(self.detector, 'update_config'):
                        self.detector.update_config(self.config_manager.config)
                    
                    results = self.detector.detect(frame)
                    
                    # Annotiertes Frame speichern
                    with self.frame_lock:
                        self.annotated_frame = results['annotated_image']
                    
                    self.frame_count += 1
                    
                    # Erkannte Nummernschilder verarbeiten
                    for plate in results.get('plates', []):
                        if plate['text']:
                            self._handle_detection(plate, results['car_detected'])
                
                time.sleep(process_interval)
                
            except Exception as e:
                print(f"Processing Fehler: {e}")
                time.sleep(1)
    
    def _handle_detection(self, plate_info, car_detected):
        """
        Verarbeitet erkanntes Nummernschild
        
        Args:
            plate_info: Dictionary mit Kennzeichen-Infos
            car_detected: Ob ein Auto erkannt wurde
        """
        plate_text = plate_info['text']
        current_time = time.time()
        
        # Prüfen ob Kennzeichen kürzlich erkannt wurde
        if plate_text in self.recent_plates:
            last_seen = self.recent_plates[plate_text]
            if current_time - last_seen < self.plate_cooldown:
                return  # Duplikat, nicht speichern
        
        # Zeitstempel aktualisieren
        self.recent_plates[plate_text] = current_time
        
        # Alte Einträge bereinigen
        self.recent_plates = {k: v for k, v in self.recent_plates.items() 
                             if current_time - v < self.plate_cooldown * 2}
        
        # Zur Historie hinzufügen
        entry = {
            "plate_text": plate_text,
            "confidence": plate_info.get('confidence', 0),
            "source": "rtsp",
            "car_detected": car_detected
        }
        
        saved_entry = self.history_manager.add_entry(entry)
        if saved_entry:
            print(f"Nummernschild erkannt: {plate_text}")
