#!/usr/bin/env python3
"""
RTSP Stream Handler
Verwaltet RTSP Videostreams im Hintergrund
Version 2.2 - Einheitliche ROI-Geometrie für Live und Einstellungen
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
    
    def __init__(self, config_manager, history_manager, detector, person_history_manager=None):
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
        self.person_history_manager = person_history_manager
        
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
            'analysis_area_enabled': self.config_manager.get('rtsp', 'analysis_area', 'enabled') or False,
            'people_enabled': bool(self.config_manager.get('people', 'enabled')),
            'people_history_count': len(self.person_history_manager.history) if self.person_history_manager else 0
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
        Holt und validiert den einen Analysebereich.

        Die gespeicherten Polygonpunkte können aus einer anderen, gespeicherten
        Koordinatenbasis stammen (coordinate_width/height). Hier werden sie auf
        die aktuelle Framegröße skaliert. Dadurch sehen /live und
        /rtsp-settings gleich aus und die Analyse nutzt denselben Bereich.
        """
        cfg = self.config_manager.get('rtsp', 'analysis_area') or {}
        if not cfg.get('enabled'):
            return {
                'enabled': False,
                'mode': 'polygon',
                'x': 0, 'y': 0, 'width': frame_width, 'height': frame_height,
                'polygon': [], 'crop_polygon': [], 'mask_outside': False,
                'coordinate_width': frame_width, 'coordinate_height': frame_height
            }

        mask_outside = True

        def safe_int(value, fallback):
            try:
                return int(round(float(value)))
            except Exception:
                return fallback

        coord_w = max(1, safe_int(cfg.get('coordinate_width'), frame_width))
        coord_h = max(1, safe_int(cfg.get('coordinate_height'), frame_height))
        scale_x = frame_width / coord_w
        scale_y = frame_height / coord_h

        def clamp_int(value, minimum, maximum):
            try:
                return max(minimum, min(int(round(float(value))), maximum))
            except Exception:
                return minimum

        points = []
        raw_points = cfg.get('polygon') or []
        if isinstance(raw_points, list):
            for point in raw_points:
                try:
                    if isinstance(point, dict):
                        px, py = point.get('x', 0), point.get('y', 0)
                    elif isinstance(point, (list, tuple)) and len(point) >= 2:
                        px, py = point[0], point[1]
                    else:
                        continue
                    sx = float(px) * scale_x
                    sy = float(py) * scale_y
                    points.append([
                        clamp_int(sx, 0, frame_width - 1),
                        clamp_int(sy, 0, frame_height - 1)
                    ])
                except Exception:
                    continue

        # Legacy-Fallback: alte Rechteckwerte in Polygon umwandeln.
        if len(points) < 3:
            area = cfg.get('area') or {}
            x = clamp_int(float(area.get('x', 0)) * scale_x, 0, frame_width - 1)
            y = clamp_int(float(area.get('y', 0)) * scale_y, 0, frame_height - 1)
            width = clamp_int(float(area.get('width', frame_width)) * scale_x, 1, frame_width - x)
            height = clamp_int(float(area.get('height', frame_height)) * scale_y, 1, frame_height - y)
            points = [
                [x, y],
                [min(frame_width - 1, x + width), y],
                [min(frame_width - 1, x + width), min(frame_height - 1, y + height)],
                [x, min(frame_height - 1, y + height)]
            ]

        if len(points) >= 3:
            xs = [pt[0] for pt in points]
            ys = [pt[1] for pt in points]
            x = max(0, min(xs))
            y = max(0, min(ys))
            x2 = min(frame_width, max(xs) + 1)
            y2 = min(frame_height, max(ys) + 1)
            width = max(10, x2 - x)
            height = max(10, y2 - y)
            crop_polygon = [[pt[0] - x, pt[1] - y] for pt in points]
            logger.debug(f"Unified Analysis ROI: frame={frame_width}x{frame_height}, basis={coord_w}x{coord_h}, bbox={x},{y},{width},{height}, points={points}")
            return {
                'enabled': True,
                'mode': 'polygon',
                'x': x, 'y': y, 'width': width, 'height': height,
                'polygon': points,
                'crop_polygon': crop_polygon,
                'mask_outside': mask_outside,
                'coordinate_width': coord_w,
                'coordinate_height': coord_h
            }

        return {
            'enabled': False,
            'mode': 'polygon',
            'x': 0, 'y': 0, 'width': frame_width, 'height': frame_height,
            'polygon': [], 'crop_polygon': [], 'mask_outside': False,
            'coordinate_width': frame_width, 'coordinate_height': frame_height
        }

    def _apply_analysis_mask(self, frame, area_info):
        """
        Gibt standardmäßig das komplette Frame an YOLO weiter.

        Fix 0.8.21: Der Analysebereich darf nicht vor der YOLO-Erkennung
        ausgeschnitten werden, sonst sieht das Fahrzeugmodell nicht mehr das
        komplette Auto/LKW/Motorrad und es wird später kein Fahrzeugbild
        gespeichert. Der ROI wird deshalb nach der Erkennung gefiltert.

        Optionales Legacy-Verhalten:
        - rtsp.analysis_area.crop_before_detection = true  -> alter Crop
        - rtsp.analysis_area.mask_before_detection = true  -> Vollbild maskieren
        """
        if not area_info.get('enabled'):
            return frame, 0, 0

        crop_before = bool(self.config_manager.get('rtsp', 'analysis_area', 'crop_before_detection'))
        mask_before = bool(self.config_manager.get('rtsp', 'analysis_area', 'mask_before_detection'))

        if crop_before:
            ax = area_info['x']; ay = area_info['y']; aw = area_info['width']; ah = area_info['height']
            process_frame = frame[ay:ay + ah, ax:ax + aw].copy()
            if mask_before and area_info.get('mode') == 'polygon' and len(area_info.get('crop_polygon') or []) >= 3:
                mask = np.zeros(process_frame.shape[:2], dtype=np.uint8)
                pts = np.array(area_info['crop_polygon'], dtype=np.int32)
                cv2.fillPoly(mask, [pts], 255)
                process_frame = cv2.bitwise_and(process_frame, process_frame, mask=mask)
            return process_frame, ax, ay

        process_frame = frame.copy()
        if mask_before and area_info.get('mode') == 'polygon' and len(area_info.get('polygon') or []) >= 3:
            mask = np.zeros(process_frame.shape[:2], dtype=np.uint8)
            pts = np.array(area_info['polygon'], dtype=np.int32)
            cv2.fillPoly(mask, [pts], 255)
            process_frame = cv2.bitwise_and(process_frame, process_frame, mask=mask)
        return process_frame, 0, 0

    def _draw_analysis_area(self, frame, area_info):
        """Zeichnet die eine verbindliche Straßen-Analyse-Zone auf das Livebild."""
        if not area_info.get('enabled'):
            return
        color = (0, 255, 255)
        pts_raw = area_info.get('polygon') or []
        if len(pts_raw) >= 3:
            pts = np.array(pts_raw, dtype=np.int32)
            overlay = frame.copy()
            cv2.fillPoly(overlay, [pts], color)
            cv2.addWeighted(overlay, 0.14, frame, 0.86, 0, frame)
            cv2.polylines(frame, [pts], True, color, 3)
            for idx, pt in enumerate(pts_raw, start=1):
                px, py = int(pt[0]), int(pt[1])
                cv2.circle(frame, (px, py), 7, (255, 255, 255), -1)
                cv2.circle(frame, (px, py), 7, color, 2)
                cv2.putText(frame, str(idx), (px + 8, py - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            label_x = int(min(pt[0] for pt in pts_raw)) + 5
            label_y = max(25, int(min(pt[1] for pt in pts_raw)) + 25)
            cv2.putText(frame, "Analysebereich Strasse", (label_x, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        else:
            ax = area_info['x']; ay = area_info['y']; aw = area_info['width']; ah = area_info['height']
            fallback_pts = np.array([[ax, ay], [ax + aw, ay], [ax + aw, ay + ah], [ax, ay + ah]], dtype=np.int32)
            cv2.polylines(frame, [fallback_pts], True, color, 2)
            cv2.putText(frame, "Analysebereich", (ax + 5, ay + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    def _point_inside_analysis_area(self, x, y, area_info):
        """True if a point in full-frame coordinates is inside the unified road ROI."""
        if not area_info or not area_info.get('enabled'):
            return True
        pts = area_info.get('polygon') or []
        if len(pts) < 3:
            ax = area_info.get('x', 0); ay = area_info.get('y', 0)
            aw = area_info.get('width', 0); ah = area_info.get('height', 0)
            return ax <= x <= ax + aw and ay <= y <= ay + ah
        contour = np.array(pts, dtype=np.float32)
        return cv2.pointPolygonTest(contour, (float(x), float(y)), False) >= 0

    def _bbox_roi_overlap_percent(self, bbox, area_info):
        """Approximate percentage of bbox area that lies inside the road ROI."""
        if not area_info or not area_info.get('enabled'):
            return 100.0
        try:
            x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
            if x2 <= x1 or y2 <= y1:
                return 0.0
            w = max(1, x2 - x1); h = max(1, y2 - y1)
            mask = np.zeros((h, w), dtype=np.uint8)
            pts = np.array([[int(px - x1), int(py - y1)] for px, py in (area_info.get('polygon') or [])], dtype=np.int32)
            if len(pts) >= 3:
                cv2.fillPoly(mask, [pts], 255)
                return float(np.count_nonzero(mask)) * 100.0 / float(w * h)
            return 100.0
        except Exception:
            return 0.0

    def _bbox_allowed_in_analysis_area(self, bbox, area_info, kind='object'):
        """Filter detections so only the one road ROI is analyzed and stored."""
        if not area_info or not area_info.get('enabled'):
            return True
        try:
            x1, y1, x2, y2 = [float(v) for v in bbox]
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            foot_x = cx
            foot_y = y2
            if kind == 'person':
                cfg = self.config_manager.get('people') or {}
                if cfg.get('roi_filter_enabled') is False:
                    return True
                mode = str(cfg.get('roi_filter_mode') or 'foot_and_center').lower()
                center_ok = self._point_inside_analysis_area(cx, cy, area_info)
                foot_ok = self._point_inside_analysis_area(foot_x, foot_y, area_info)
                if mode == 'center':
                    ok = center_ok
                elif mode == 'foot':
                    ok = foot_ok
                elif mode == 'center_or_foot':
                    ok = center_ok or foot_ok
                else:
                    ok = center_ok and foot_ok
                if not ok:
                    return False
                min_overlap = float(cfg.get('min_roi_overlap_percent') or 0)
                if cfg.get('reject_partial_outside_roi') and min_overlap > 0:
                    return self._bbox_roi_overlap_percent([x1, y1, x2, y2], area_info) >= min_overlap
                return True
            # Vehicles/plates: center must be inside the road polygon.
            return self._point_inside_analysis_area(cx, cy, area_info)
        except Exception:
            return True

    def _draw_people_line(self, frame, area_info):
        """Draw the person counting line in the same coordinate basis used for detection."""
        cfg = self.config_manager.get('people') or {}
        if not cfg.get('enabled') or cfg.get('show_on_live') is False or cfg.get('line_crossing_enabled') is False:
            return
        h, w = frame.shape[:2]
        axis = str(cfg.get('movement_axis') or 'y').lower()
        line_percent = max(1, min(99, float(cfg.get('virtual_line_position_percent') or 50)))
        use_roi = cfg.get('line_relative_to_roi') is not False and area_info and area_info.get('enabled')
        if use_roi:
            ax = int(area_info.get('x') or 0); ay = int(area_info.get('y') or 0)
            aw = int(area_info.get('width') or w); ah = int(area_info.get('height') or h)
        else:
            ax, ay, aw, ah = 0, 0, w, h
        color = (34, 211, 238)
        if axis == 'x':
            x = int(ax + aw * line_percent / 100.0)
            y1 = max(0, ay); y2 = min(h - 1, ay + ah)
            cv2.line(frame, (x, y1), (x, y2), color, 2)
            cv2.putText(frame, f"Personenlinie X {line_percent:.0f}%", (min(w - 240, x + 8), max(24, y1 + 24)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        else:
            y = int(ay + ah * line_percent / 100.0)
            x1 = max(0, ax); x2 = min(w - 1, ax + aw)
            cv2.line(frame, (x1, y), (x2, y), color, 2)
            cv2.putText(frame, f"Personenlinie Y {line_percent:.0f}%", (max(8, x1 + 8), max(24, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

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
                        # Analysis Area holen und ggf. auf Straße/Polygon maskieren
                        area_info = self._get_analysis_area(frame_h, frame_w)
                        area_enabled = area_info.get('enabled', False)
                        process_frame, offset_x, offset_y = self._apply_analysis_mask(frame, area_info)
                        
                        # Erkennung auf Vollbild; ROI wird danach gefiltert. Nur im Legacy-Crop-Modus werden Crop-Koordinaten genutzt.
                        runtime_polygon = None
                        if area_enabled:
                            runtime_polygon = area_info.get('crop_polygon') if (offset_x or offset_y) else area_info.get('polygon')
                        results = self.detector.process_frame(process_frame, apply_analysis_area=False, runtime_roi_polygon=runtime_polygon)
                        
                        # Annotiertes Frame erstellen
                        annotated = frame.copy()
                        
                        # Analysis Area einzeichnen
                        self._draw_analysis_area(annotated, area_info)
                        
                        # Personen-Zähllinie in derselben Basis wie die Erkennung zeichnen
                        self._draw_people_line(annotated, area_info)

                        # Fahrzeuge mit Offset einzeichnen und gegen die einheitliche ROI filtern
                        filtered_vehicles = []
                        for vehicle in results.get('vehicles', []):
                            bbox = vehicle.get('bbox', [])
                            if len(bbox) == 4:
                                vx1, vy1, vx2, vy2 = [int(v) for v in bbox]
                                vx1 += offset_x
                                vy1 += offset_y
                                vx2 += offset_x
                                vy2 += offset_y
                                if not self._bbox_allowed_in_analysis_area([vx1, vy1, vx2, vy2], area_info, kind='vehicle'):
                                    continue
                                vehicle['bbox'] = [vx1, vy1, vx2, vy2]
                                vehicle['center_x'] = round((vx1 + vx2) / 2, 2)
                                vehicle['center_y'] = round((vy1 + vy2) / 2, 2)
                                vehicle['frame_width'] = frame_w
                                vehicle['frame_height'] = frame_h
                                filtered_vehicles.append(vehicle)
                                cv2.rectangle(annotated, (vx1, vy1), (vx2, vy2), (255, 0, 0), 2)
                                label = f"{vehicle.get('type', 'Fahrzeug')} ({vehicle.get('color', '')})"
                                cv2.putText(annotated, label, (vx1, max(18, vy1 - 10)),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                        results['vehicles'] = filtered_vehicles

                        # Kennzeichen mit Offset einzeichnen und gegen die einheitliche ROI filtern
                        filtered_detections = []
                        for detection in results.get('detections', []):
                            bbox = detection.get('plate_bbox', [])
                            if len(bbox) == 4:
                                px1, py1, px2, py2 = [int(v) for v in bbox]
                                px1 += offset_x
                                py1 += offset_y
                                px2 += offset_x
                                py2 += offset_y
                                if not self._bbox_allowed_in_analysis_area([px1, py1, px2, py2], area_info, kind='plate'):
                                    continue
                                detection['plate_bbox'] = [px1, py1, px2, py2]
                                detection['plate_center_x'] = round((px1 + px2) / 2, 2)
                                detection['plate_center_y'] = round((py1 + py2) / 2, 2)
                                detection['frame_width'] = frame_w
                                detection['frame_height'] = frame_h
                                filtered_detections.append(detection)
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
                        results['detections'] = filtered_detections

                        # Personenerkennungen mit Offset einzeichnen und Fußpunkt/Zentrum gegen Straße prüfen
                        filtered_people = []
                        for person in results.get('people', []):
                            bbox = person.get('bbox', [])
                            if len(bbox) == 4:
                                x1, y1, x2, y2 = [int(v) for v in bbox]
                                x1 += offset_x
                                y1 += offset_y
                                x2 += offset_x
                                y2 += offset_y
                                if not self._bbox_allowed_in_analysis_area([x1, y1, x2, y2], area_info, kind='person'):
                                    person['roi_filtered'] = True
                                    continue
                                person['bbox'] = [x1, y1, x2, y2]
                                person['center_x'] = round((x1 + x2) / 2, 2)
                                person['center_y'] = round((y1 + y2) / 2, 2)
                                person['foot_x'] = round((x1 + x2) / 2, 2)
                                person['foot_y'] = y2
                                person['frame_width'] = frame_w
                                person['frame_height'] = frame_h
                                person['roi_filter_status'] = 'accepted'
                                filtered_people.append(person)
                                if self.config_manager.get('people', 'show_on_live') is not False:
                                    color = (16, 185, 129) if person.get('counted') else (245, 158, 11)
                                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                                    label = f"Person #{person.get('track_id', '?')} {person.get('confidence', 0):.2f}"
                                    if person.get('counted'):
                                        label += " gezählt"
                                    cv2.putText(annotated, label, (x1, max(18, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
                        results['people'] = filtered_people

                        # Status-Info einzeichnen
                        status_text = f"FPS: {self.get_fps()} | Frames: {self.frame_count} | Erkennungen: {self.detection_count} | Personen: {len(results.get('people', []))}"
                        cv2.putText(annotated, status_text, (10, 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        
                        if area_enabled:
                            area_text = f"ROI: Strasse {area_info.get('x')},{area_info.get('y')} {area_info.get('width')}x{area_info.get('height')}"
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

                        for person in results.get('people', []):
                            self._handle_person_detection(person, results, frame=frame, annotated=annotated)
                                
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
            "vehicle_type_en": detection.get('vehicle_type_en', 'unknown'),
            "vehicle_color": detection.get('vehicle_color', 'Unbekannt'),
            "vehicle_color_hex": detection.get('vehicle_color_hex'),
            "vehicle_color_rgb": detection.get('vehicle_color_rgb'),
            "vehicle_color_coverage": detection.get('vehicle_color_coverage'),
            "plate_country": detection.get('plate_country'),
            "plate_country_display": detection.get('plate_country_display'),
            "plate_country_prob": detection.get('plate_country_prob'),
            "ocr_engine": detection.get('ocr_engine'),
            "ocr_model": detection.get('ocr_model'),
            "plate_bbox": detection.get('plate_bbox'),
            "vehicle_bbox": detection.get('vehicle_bbox'),
            "plate_center_x": detection.get('plate_center_x'),
            "plate_center_y": detection.get('plate_center_y'),
            "vehicle_center_x": detection.get('vehicle_center_x'),
            "vehicle_center_y": detection.get('vehicle_center_y'),
            "frame_width": detection.get('frame_width'),
            "frame_height": detection.get('frame_height'),
        }
        
        saved_entry = self.history_manager.add_entry(entry, check_duplicate=True)
        
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
                    'vehicle_color_hex': detection.get('vehicle_color_hex'),
                    'plate_country_display': detection.get('plate_country_display'),
                    'timestamp': datetime.now().isoformat()
                }, broadcast=True, namespace='/')
            except:
                pass  # SocketIO nicht verfügbar oder Fehler


    def _handle_person_detection(self, person, full_results, frame=None, annotated=None):
        """Speichert Personenereignisse in separater Historie."""
        if not self.person_history_manager:
            return
        if not self.config_manager.get('people', 'enabled'):
            return
        save_all = bool(self.config_manager.get('people', 'save_all_detections'))
        if not person.get('counted') and not save_all:
            return
        event = {
            'event_type': person.get('event_type', 'person_detected'),
            'counted': bool(person.get('counted')),
            'direction': person.get('direction', 'unknown'),
            'track_id': person.get('track_id'),
            'confidence': person.get('confidence', 0),
            'bbox': person.get('bbox'),
            'center_x': person.get('center_x'),
            'center_y': person.get('center_y'),
            'frame_width': person.get('frame_width'),
            'frame_height': person.get('frame_height'),
            'source': 'rtsp',
            'source_model': person.get('source_model'),
        }
        saved = self.person_history_manager.add_event(event, frame=frame, annotated_frame=annotated)
        if saved:
            try:
                from flask_socketio import emit
                emit('person_detected', {
                    'counted': saved.get('counted'),
                    'direction': saved.get('direction'),
                    'track_id': saved.get('track_id'),
                    'confidence': saved.get('confidence'),
                    'timestamp': saved.get('timestamp')
                }, broadcast=True, namespace='/')
            except Exception:
                pass
