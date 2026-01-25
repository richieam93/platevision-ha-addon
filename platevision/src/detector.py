#!/usr/bin/env python3
"""
License Plate Detection - Erkennungsmodul
YOLO für Auto/Nummernschilderkennung + EasyOCR für Texterkennung
Mit Auto-Zoom Funktion
"""

import cv2
import numpy as np
import easyocr
from ultralytics import YOLO
import os
from datetime import datetime


class LicensePlateDetector:
    """Klasse für Nummernschilderkennung mit YOLO und EasyOCR"""
    
    # Fahrzeug-Klassen in COCO Dataset
    VEHICLE_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck
    
    def __init__(self, coco_model_path, plate_model_path, config=None):
        """
        Initialisiert den Detektor
        
        Args:
            coco_model_path: Pfad zum COCO YOLO Model
            plate_model_path: Pfad zum Nummernschilderkennung Model
            config: Konfigurationsdictionary
        """
        self.config = config or {}
        
        # YOLO Modelle laden
        self.coco_model = YOLO(coco_model_path)
        self.plate_model = YOLO(plate_model_path)
        
        # EasyOCR Reader initialisieren
        self.reader = easyocr.Reader(['en', 'de'], gpu=False)
        
        # Konfiguration setzen
        self.update_config(config)
        
        print("Detector initialized successfully")
    
    def update_config(self, config):
        """Konfiguration aktualisieren"""
        self.config = config or {}
        
        detection_config = self.config.get('detection', {})
        self.confidence_threshold = detection_config.get('confidence_threshold', 0.25)
        self.car_zoom_factor = detection_config.get('car_zoom_factor', 1.2)
        self.plate_zoom_factor = detection_config.get('plate_zoom_factor', 1.5)
        self.enable_car_zoom = detection_config.get('enable_car_zoom', True)
        self.save_detections = detection_config.get('save_detections', True)
        self.save_cropped_plates = detection_config.get('save_cropped_plates', True)
        
        display_config = self.config.get('display', {})
        self.show_bounding_boxes = display_config.get('show_bounding_boxes', True)
        self.show_labels = display_config.get('show_labels', True)
        self.box_color_car = tuple(display_config.get('box_color_car', [0, 0, 255]))
        self.box_color_plate = tuple(display_config.get('box_color_plate', [0, 255, 0]))
        self.font_scale = display_config.get('font_scale', 0.8)
    
    def zoom_to_region(self, image, bbox, zoom_factor, padding_percent=0.1):
        """
        Zoomt auf eine Region im Bild
        
        Args:
            image: Eingabebild
            bbox: Bounding Box [x1, y1, x2, y2]
            zoom_factor: Zoom-Faktor (1.0 = kein Zoom, 2.0 = doppelter Zoom)
            padding_percent: Zusätzlicher Rand in Prozent
            
        Returns:
            Ausgeschnittenes und gezoomtes Bild
        """
        h, w = image.shape[:2]
        x1, y1, x2, y2 = map(int, bbox)
        
        # Bereich mit Padding berechnen
        box_w = x2 - x1
        box_h = y2 - y1
        
        # Padding hinzufügen
        padding_x = int(box_w * padding_percent)
        padding_y = int(box_h * padding_percent)
        
        # Erweiterte Koordinaten berechnen
        new_x1 = max(0, x1 - padding_x)
        new_y1 = max(0, y1 - padding_y)
        new_x2 = min(w, x2 + padding_x)
        new_y2 = min(h, y2 + padding_y)
        
        # Region ausschneiden
        cropped = image[new_y1:new_y2, new_x1:new_x2]
        
        # Auf Originalgröße zoomen wenn gewünscht
        if zoom_factor > 1.0:
            new_size = (int(cropped.shape[1] * zoom_factor), int(cropped.shape[0] * zoom_factor))
            cropped = cv2.resize(cropped, new_size, interpolation=cv2.INTER_CUBIC)
        
        return cropped, (new_x1, new_y1, new_x2, new_y2)
    
    def preprocess_plate_image(self, image):
        """
        Vorverarbeitung für bessere OCR-Ergebnisse
        
        Args:
            image: Nummernschildbild
            
        Returns:
            Vorverarbeitetes Bild
        """
        # Zu Graustufen konvertieren
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Kontrastanpassung
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Rauschreduzierung
        denoised = cv2.bilateralFilter(enhanced, 11, 17, 17)
        
        # Optional: Thresholding für bessere Texterkennung
        _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return thresh
    
    def read_license_plate(self, plate_image):
        """
        Liest Text vom Nummernschild
        
        Args:
            plate_image: Bild des Nummernschilds
            
        Returns:
            Tuple (Text, Konfidenz)
        """
        try:
            # Vorverarbeitung
            preprocessed = self.preprocess_plate_image(plate_image)
            
            # OCR durchführen
            detections = self.reader.readtext(preprocessed)
            
            if not detections:
                # Auch mit Originalbild versuchen
                detections = self.reader.readtext(plate_image)
            
            if not detections:
                return None, 0
            
            # Alle erkannten Textteile sammeln
            plate_parts = []
            total_score = 0
            
            for detection in detections:
                bbox, text, score = detection
                
                # Nur Text mit ausreichender Konfidenz
                if score > 0.2:
                    # Text bereinigen
                    text = text.upper().strip()
                    text = ''.join(c for c in text if c.isalnum() or c in '-– ')
                    
                    if len(text) >= 1:
                        plate_parts.append(text)
                        total_score += score
            
            if not plate_parts:
                return None, 0
            
            # Text zusammenfügen
            full_text = ' '.join(plate_parts)
            avg_score = total_score / len(plate_parts)
            
            return full_text, avg_score
            
        except Exception as e:
            print(f"OCR Fehler: {e}")
            return None, 0
    
    def detect_in_car_region(self, image, car_bbox):
        """
        Erkennt Nummernschilder im Auto-Bereich mit Zoom
        
        Args:
            image: Gesamtbild
            car_bbox: Bounding Box des Autos
            
        Returns:
            Liste der erkannten Nummernschilder
        """
        plates = []
        
        # Auto-Region extrahieren und zoomen
        car_region, region_coords = self.zoom_to_region(
            image, car_bbox, 
            self.car_zoom_factor if self.enable_car_zoom else 1.0,
            padding_percent=0.1
        )
        
        # Nummernschilderkennung in Auto-Region
        plate_detections = self.plate_model(car_region, conf=self.confidence_threshold)[0]
        
        for detection in plate_detections.boxes.data.tolist():
            px1, py1, px2, py2, score, class_id = detection
            
            # Nummernschild ausschneiden (mit extra Zoom für OCR)
            plate_region, _ = self.zoom_to_region(
                car_region, 
                [px1, py1, px2, py2], 
                self.plate_zoom_factor,
                padding_percent=0.05
            )
            
            # OCR durchführen
            plate_text, text_score = self.read_license_plate(plate_region)
            
            # Originale Koordinaten zurückrechnen
            scale_x = image.shape[1] / car_region.shape[1] if self.enable_car_zoom else 1
            scale_y = image.shape[0] / car_region.shape[0] if self.enable_car_zoom else 1
            
            original_px1 = int(region_coords[0] + px1 / self.car_zoom_factor)
            original_py1 = int(region_coords[1] + py1 / self.car_zoom_factor)
            original_px2 = int(region_coords[0] + px2 / self.car_zoom_factor)
            original_py2 = int(region_coords[1] + py2 / self.car_zoom_factor)
            
            plates.append({
                'bbox': [original_px1, original_py1, original_px2, original_py2],
                'text': plate_text,
                'confidence': score,
                'text_confidence': text_score,
                'cropped_image': plate_region if self.save_cropped_plates else None
            })
        
        return plates
    
    def detect(self, image, source="unknown"):
        """
        Haupterkennungsfunktion
        
        Args:
            image: Eingabebild (BGR)
            source: Quelle des Bildes (für Historie)
            
        Returns:
            Dictionary mit Erkennungsergebnissen
        """
        results = {
            'plates': [],
            'cars': [],
            'car_detected': False,
            'cars_count': 0,
            'annotated_image': image.copy()
        }
        
        h, w = image.shape[:2]
        
        # Area selection logic
        area_config = self.config.get('rtsp', {}).get('analysis_area', {})
        area_enabled = area_config.get('enabled', False)
        area_data = area_config.get('area', {})
        
        if area_enabled and area_data:
            # Extract area coordinates
            x = area_data.get('x', 0)
            y = area_data.get('y', 0)
            width = area_data.get('width', w)
            height = area_data.get('height', h)
            
            # Ensure area is within image bounds
            x = max(0, min(x, w - 1))
            y = max(0, min(y, h - 1))
            width = max(1, min(width, w - x))
            height = max(1, min(height, h - y))
            
            # Create mask for area selection
            mask = np.zeros((h, w), dtype=np.uint8)
            mask[y:y+height, x:x+width] = 255
            
            # Apply mask to create area-only image for detection
            area_image = cv2.bitwise_and(image, image, mask=mask)
            
            # Store area info for debugging/visualization
            results['analysis_area'] = {
                'x': x, 'y': y, 'width': width, 'height': height
            }
        else:
            # Use full image
            area_image = image
            results['analysis_area'] = None
        
        # Fahrzeugerkennung
        car_detections = self.coco_model(image, conf=self.confidence_threshold)[0]
        
        cars_found = []
        for detection in car_detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            
            if int(class_id) in self.VEHICLE_CLASSES:
                cars_found.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': score,
                    'class_id': int(class_id)
                })
                
                # Bounding Box zeichnen
                if self.show_bounding_boxes:
                    cv2.rectangle(results['annotated_image'], 
                                (int(x1), int(y1)), (int(x2), int(y2)), 
                                self.box_color_car, 2)
                    
                    if self.show_labels:
                        label = f"Car {score:.2f}"
                        cv2.putText(results['annotated_image'], label,
                                  (int(x1), int(y1) - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, self.font_scale,
                                  self.box_color_car, 2)
        
        results['cars'] = cars_found
        results['cars_count'] = len(cars_found)
        results['car_detected'] = len(cars_found) > 0
        
        # Nummernschilderkennung
        if self.enable_car_zoom and cars_found:
            # Suche Nummernschilder nur in Auto-Regionen (mit Zoom)
            for car in cars_found:
                plates = self.detect_in_car_region(image, car['bbox'])
                results['plates'].extend(plates)
        else:
            # Direktsuche im gesamten Bild
            plate_detections = self.plate_model(image, conf=self.confidence_threshold)[0]
            
            for detection in plate_detections.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = detection
                
                # Nummernschild ausschneiden
                plate_crop = image[int(y1):int(y2), int(x1):int(x2)]
                
                # Mit Zoom für bessere OCR
                if self.plate_zoom_factor > 1.0:
                    new_size = (int(plate_crop.shape[1] * self.plate_zoom_factor), 
                              int(plate_crop.shape[0] * self.plate_zoom_factor))
                    plate_crop = cv2.resize(plate_crop, new_size, interpolation=cv2.INTER_CUBIC)
                
                # OCR
                plate_text, text_score = self.read_license_plate(plate_crop)
                
                results['plates'].append({
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'text': plate_text,
                    'confidence': score,
                    'text_confidence': text_score,
                    'cropped_image': plate_crop if self.save_cropped_plates else None
                })
        
        # Nummernschilder annotieren
        for plate in results['plates']:
            x1, y1, x2, y2 = plate['bbox']
            
            if self.show_bounding_boxes:
                cv2.rectangle(results['annotated_image'],
                            (x1, y1), (x2, y2),
                            self.box_color_plate, 2)
                
                if self.show_labels and plate['text']:
                    # Hintergrund für Text
                    text = plate['text']
                    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 
                                                          self.font_scale, 2)
                    cv2.rectangle(results['annotated_image'],
                                (x1, y1 - text_h - 10), (x1 + text_w + 10, y1),
                                (255, 255, 255), -1)
                    cv2.putText(results['annotated_image'], text,
                              (x1 + 5, y1 - 5),
                              cv2.FONT_HERSHEY_SIMPLEX, self.font_scale,
                              (0, 0, 0), 2)
        
        return results
    
    def process_video(self, video_path, history_callback=None):
        """
        Verarbeitet ein Video und erkennt Nummernschilder
        
        Args:
            video_path: Pfad zur Videodatei
            history_callback: Callback-Funktion für Historie
            
        Returns:
            Dictionary mit Verarbeitungsergebnissen
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return {"error": "Video konnte nicht geöffnet werden"}
        
        results = {
            'frames_processed': 0,
            'plates_detected': 0,
            'unique_plates': set()
        }
        
        frame_skip = 5  # Nur jeden 5. Frame verarbeiten
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Frames überspringen für Performance
            if frame_count % frame_skip != 0:
                continue
            
            # Erkennung durchführen
            detection_results = self.detect(frame, source="video")
            results['frames_processed'] += 1
            
            for plate in detection_results['plates']:
                if plate['text']:
                    results['plates_detected'] += 1
                    results['unique_plates'].add(plate['text'])
                    
                    # Zur Historie hinzufügen
                    if history_callback:
                        history_callback({
                            "license_plate": plate['text'],
                            "confidence": plate['confidence'],
                            "source": "video",
                            "car_detected": detection_results['car_detected']
                        })
        
        cap.release()
        
        results['unique_plates'] = list(results['unique_plates'])
        return results