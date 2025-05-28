"""
Mall Blueprint Mapping GUI

A user-friendly interface for mapping mall blueprints to CCTV footage.
Provides step-by-step guidance and visual tools for:
1. Loading and calibrating blueprint images
2. Placing and orienting cameras
3. Defining store boundaries
4. Testing with CCTV footage
"""

import sys
import cv2
import numpy as np
import json
import os
import boto3
import time
from datetime import datetime
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QMessageBox, QToolBar,
    QStatusBar, QDockWidget, QTextEdit, QSpinBox, QDoubleSpinBox,
    QComboBox, QGroupBox, QFormLayout, QTabWidget, QToolTip,
    QSplitter, QFrame, QSizePolicy, QDialog, QLineEdit, QDialogButtonBox
)
from PyQt6.QtCore import Qt, QPoint, QRect, pyqtSignal, QTimer
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont, QAction
from blueprint_mapping import BlueprintProcessor

class CalibrationDialog(QMessageBox):
    """Dialog for camera calibration"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Camera Calibration")
        self.setText("Camera Calibration Required")
        self.setInformativeText(
            "To map the blueprint to CCTV footage, we need to calibrate the camera.\n\n"
            "IMPORTANT: The blueprint shows stores vertically, but in the video they appear horizontally.\n\n"
            "Calibration Steps:\n"
            "1. In the blueprint, click 4 points in this order:\n"
            "   • Top-left corner of the store area\n"
            "   • Top-right corner of the store area\n"
            "   • Bottom-right corner of the store area\n"
            "   • Bottom-left corner of the store area\n"
            "2. In the video, click the same 4 points in the same order\n"
            "   (The points should form a rectangle around the stores)\n"
            "3. The system will calculate the correct perspective transformation"
        )
        self.setStandardButtons(QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel)
        self.setDefaultButton(QMessageBox.StandardButton.Ok)

class BlueprintView(QWidget):
    """Custom widget for displaying and interacting with the blueprint"""
    camera_placed = pyqtSignal(str, int, int, float)  # camera_id, x, y, orientation
    store_defined = pyqtSignal(str, list)  # store_id, polygon_points
    calibration_points_selected = pyqtSignal(list)  # List of (x, y) points
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.blueprint_image = None
        self.blueprint_path = None  # Add path storage
        self.scaled_image = None
        self.current_tool = "select"  # select, camera, store, calibrate
        self.drawing_points = []
        self.cameras = {}
        self.stores = {}
        self.calibration_points = []
        self.setMouseTracking(True)
        self.setToolTip("Click to place cameras or define store boundaries")
        
        # Add store mapping status
        self.mapped_stores = set()  # Track which stores have been mapped
        
        # Set minimum size to ensure the widget is visible
        self.setMinimumSize(400, 300)
        
        # Set size policy to allow widget to expand
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
    
    def set_tool(self, tool):
        """Set the current interaction tool"""
        self.current_tool = tool
        self.drawing_points = []
        if tool == "calibrate":
            self.calibration_points = []
            self.setToolTip("Click 4 points for camera calibration")
        elif tool == "camera":
            self.setToolTip("Click to place camera, drag to set orientation")
        elif tool == "store":
            self.setToolTip("Click to add polygon points, double-click to complete")
        else:
            self.setToolTip("Click to select and move elements")
        self.update()
    
    def load_image(self, image_path):
        """Load and display the blueprint image"""
        self.blueprint_image = cv2.imread(image_path)
        if self.blueprint_image is None:
            return False
        
        # Store the blueprint path
        self.blueprint_path = image_path
        
        # Convert to RGB for display
        self.blueprint_image = cv2.cvtColor(self.blueprint_image, cv2.COLOR_BGR2RGB)
        
        # Update the scaled image
        self.update_scaled_image()
        
        # Update the widget size to match the image
        self.updateGeometry()
        
        return True
    
    def update_scaled_image(self):
        """Update the scaled image to fit the widget size"""
        if self.blueprint_image is None:
            return
        
        # Get the widget size
        widget_size = self.size()
        
        # Calculate the scaling factor to fit the image in the widget
        img_height, img_width = self.blueprint_image.shape[:2]
        scale_w = widget_size.width() / img_width
        scale_h = widget_size.height() / img_height
        scale = min(scale_w, scale_h)
        
        # Calculate new dimensions
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        
        # Resize the image
        self.scaled_image = cv2.resize(self.blueprint_image, (new_width, new_height))
        
        # Calculate the offset to center the image
        self.image_offset_x = (widget_size.width() - new_width) // 2
        self.image_offset_y = (widget_size.height() - new_height) // 2
        
        # Store the scale factor for coordinate conversion
        self.scale_factor = scale
    
    def resizeEvent(self, event):
        """Handle widget resize events"""
        super().resizeEvent(event)
        self.update_scaled_image()
        self.update()
    
    def paintEvent(self, event):
        """Draw the blueprint and any overlays"""
        if self.scaled_image is None:
            return
        
        painter = QPainter(self)
        
        # Draw blueprint
        height, width = self.scaled_image.shape[:2]
        qimage = QImage(self.scaled_image.data, width, height, 
                       self.scaled_image.strides[0], QImage.Format.Format_RGB888)
        painter.drawImage(self.image_offset_x, self.image_offset_y, qimage)
        
        # Convert coordinates for overlays
        def to_widget_coords(x, y):
            return (int(x * self.scale_factor + self.image_offset_x),
                   int(y * self.scale_factor + self.image_offset_y))
        
        # Draw cameras
        for camera_id, camera in self.cameras.items():
            x, y = to_widget_coords(*camera["position"])
            orientation = camera["orientation"]
            
            # Draw camera marker
            painter.setPen(QPen(QColor(255, 0, 0), 2))
            painter.drawEllipse(QPoint(x, y), 5, 5)
            
            # Draw orientation line
            line_length = 20 * self.scale_factor
            end_x = int(x + line_length * np.cos(np.radians(orientation)))
            end_y = int(y - line_length * np.sin(np.radians(orientation)))
            painter.drawLine(x, y, end_x, end_y)
            
            # Draw camera ID
            painter.setFont(QFont("Arial", 8))
            painter.drawText(x + 10, y, camera_id)
            
            # Draw field of view cone
            if "fov_angle" in camera and "fov_range" in camera:
                fov_angle = camera["fov_angle"]
                fov_range = camera["fov_range"] * self.scale_factor
                half_angle = fov_angle / 2
                
                # Draw cone edges
                angle1 = np.radians(orientation - half_angle)
                angle2 = np.radians(orientation + half_angle)
                
                edge1_x = int(x + fov_range * np.cos(angle1))
                edge1_y = int(y - fov_range * np.sin(angle1))
                edge2_x = int(x + fov_range * np.cos(angle2))
                edge2_y = int(y - fov_range * np.sin(angle2))
                
                painter.setPen(QPen(QColor(255, 0, 0), 1, Qt.PenStyle.DashLine))
                painter.drawLine(x, y, edge1_x, edge1_y)
                painter.drawLine(x, y, edge2_x, edge2_y)
        
        # Draw stores
        for store_id, store in self.stores.items():
            points = store["polygon"]
            if len(points) > 1:
                # Convert points to widget coordinates
                widget_points = [to_widget_coords(x, y) for x, y in points]
                
                # Set color based on mapping status
                if store_id in self.mapped_stores:
                    painter.setPen(QPen(QColor(0, 255, 0), 2))  # Green for mapped stores
                else:
                    painter.setPen(QPen(QColor(255, 165, 0), 2))  # Orange for unmapped stores
                
                # Draw polygon
                for i in range(len(widget_points) - 1):
                    painter.drawLine(widget_points[i][0], widget_points[i][1],
                                   widget_points[i+1][0], widget_points[i+1][1])
                if len(widget_points) > 2:
                    painter.drawLine(widget_points[-1][0], widget_points[-1][1],
                                   widget_points[0][0], widget_points[0][1])
                
                # Draw store name and mapping status
                if "name" in store:
                    centroid_x = int(np.mean([p[0] for p in widget_points]))
                    centroid_y = int(np.mean([p[1] for p in widget_points]))
                    painter.setFont(QFont("Arial", 8))
                    status = "✓" if store_id in self.mapped_stores else "?"
                    painter.drawText(centroid_x, centroid_y, f"{store['name']} {status}")
        
        # Draw current drawing points
        if self.current_tool == "store" and len(self.drawing_points) > 0:
            # Convert points to widget coordinates
            widget_points = [to_widget_coords(x, y) for x, y in self.drawing_points]
            
            painter.setPen(QPen(QColor(0, 255, 0), 2))
            for i in range(len(widget_points) - 1):
                painter.drawLine(widget_points[i][0], widget_points[i][1],
                               widget_points[i+1][0], widget_points[i+1][1])
            if len(widget_points) > 2:
                painter.drawLine(widget_points[-1][0], widget_points[-1][1],
                               widget_points[0][0], widget_points[0][1])
        
        # Draw calibration points
        if self.current_tool == "calibrate":
            for i, point in enumerate(self.calibration_points):
                x, y = to_widget_coords(*point)
                painter.setPen(QPen(QColor(255, 255, 0), 2))
                painter.drawEllipse(QPoint(int(x), int(y)), 5, 5)
                painter.setFont(QFont("Arial", 8))
                painter.drawText(int(x + 10), int(y), str(i + 1))
    
    def mousePressEvent(self, event):
        """Handle mouse clicks for different tools"""
        if self.scaled_image is None:
            return
        
        # Convert mouse coordinates to image coordinates
        x = (event.position().x() - self.image_offset_x) / self.scale_factor
        y = (event.position().y() - self.image_offset_y) / self.scale_factor
        
        # Check if click is within image bounds
        img_height, img_width = self.blueprint_image.shape[:2]
        if not (0 <= x < img_width and 0 <= y < img_height):
            return
        
        if self.current_tool == "camera":
            # Place a new camera
            camera_id = f"cam{len(self.cameras) + 1:03d}"
            self.cameras[camera_id] = {
                "position": (x, y),
                "orientation": 0,
                "fov_angle": 70,  # Default FOV
                "fov_range": 100  # Default range in pixels
            }
            self.camera_placed.emit(camera_id, x, y, 0)
            self.update()
        
        elif self.current_tool == "store":
            # Add point to store polygon
            self.drawing_points.append((x, y))
            self.update()
        
        elif self.current_tool == "calibrate":
            # Add calibration point
            if len(self.calibration_points) < 4:
                self.calibration_points.append((x, y))
                if len(self.calibration_points) == 4:
                    self.calibration_points_selected.emit(self.calibration_points)
                self.update()
    
    def mouseMoveEvent(self, event):
        """Handle mouse movement for camera orientation"""
        if self.current_tool == "camera" and event.buttons() & Qt.MouseButton.LeftButton:
            # Convert mouse coordinates to image coordinates
            mouse_x = (event.position().x() - self.image_offset_x) / self.scale_factor
            mouse_y = (event.position().y() - self.image_offset_y) / self.scale_factor
            
            # Update camera orientation
            for camera_id, camera in self.cameras.items():
                cam_x, cam_y = camera["position"]
                dx = mouse_x - cam_x
                dy = cam_y - mouse_y  # Invert y-axis
                orientation = np.degrees(np.arctan2(dy, dx))
                camera["orientation"] = orientation
                self.camera_placed.emit(camera_id, cam_x, cam_y, orientation)
                self.update()
    
    def mouseDoubleClickEvent(self, event):
        """Handle double clicks for completing store polygons"""
        if self.current_tool == "store" and len(self.drawing_points) > 2:
            # Create store dialog
            store_dialog = QDialog(self)
            store_dialog.setWindowTitle("Store Information")
            store_dialog.setModal(True)
            
            # Create form layout
            form_layout = QFormLayout(store_dialog)
            
            # Add input fields
            name_input = QLineEdit()
            category_input = QLineEdit()
            form_layout.addRow("Store Name:", name_input)
            form_layout.addRow("Category:", category_input)
            
            # Add buttons
            button_box = QDialogButtonBox(
                QDialogButtonBox.StandardButton.Ok | 
                QDialogButtonBox.StandardButton.Cancel
            )
            button_box.accepted.connect(store_dialog.accept)
            button_box.rejected.connect(store_dialog.reject)
            form_layout.addRow(button_box)
            
            # Show dialog
            if store_dialog.exec() == QDialog.DialogCode.Accepted:
                store_name = name_input.text().strip()
                store_category = category_input.text().strip()
                
                if not store_name:
                    QMessageBox.warning(self, "Error", "Store name cannot be empty")
                    return
                
                # Generate store ID
                store_id = f"store{len(self.stores) + 1:03d}"
                
                # Add store
                self.stores[store_id] = {
                    "polygon": self.drawing_points.copy(),
                    "name": store_name,
                    "category": store_category
                }
                self.store_defined.emit(store_id, self.drawing_points)
                self.drawing_points = []
                self.update()
            else:
                # If dialog was cancelled, clear the drawing points
                self.drawing_points = []
                self.update()

class PersonTracker:
    """Tracks people using OpenCV's person detector"""
    def __init__(self):
        # Load pre-trained person detector
        self.person_detector = cv2.HOGDescriptor_getDefaultPeopleDetector()
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(self.person_detector)
        
        # Tracking settings
        self.min_confidence = 0.3  # Minimum confidence for person detection
        self.tracked_people = {}   # Format: {track_id: {'box': (x,y,w,h), 'frames': count, 'last_pos': (x,y), 'velocity': (dx,dy)}}
        self.track_counter = 0
        self.tracking_settings = {
            'max_tracking_frames': 30,     # Maximum frames to track a person
            'iou_threshold': 0.3,          # IOU threshold for matching detections
            'max_movement': 0.2,           # Maximum allowed movement between frames
            'smooth_factor': 0.7,          # Smoothing factor for position updates
            'velocity_decay': 0.95,        # Velocity decay factor per frame
            'min_velocity': 0.001          # Minimum velocity threshold
        }
    
    def detect_people(self, frame):
        """Detect people in a frame using HOG detector"""
        # Resize frame for faster detection while maintaining aspect ratio
        scale = min(1.0, 800 / max(frame.shape[:2]))
        if scale < 1.0:
            small_frame = cv2.resize(frame, None, fx=scale, fy=scale)
        else:
            small_frame = frame
        
        # Detect people
        boxes, weights = self.hog.detectMultiScale(
            small_frame,
            winStride=(8, 8),
            padding=(4, 4),
            scale=1.05,
            hitThreshold=0,
            groupThreshold=0.3
        )
        
        # Convert boxes back to original scale
        if scale < 1.0:
            boxes = boxes / scale
        
        # Filter detections by confidence
        detections = []
        for box, confidence in zip(boxes, weights):
            if confidence[0] >= self.min_confidence:
                x, y, w, h = box
                detections.append({
                    'box': (int(x), int(y), int(x + w), int(y + h)),
                    'confidence': float(confidence[0])
                })
        
        return detections
    
    def update_tracking(self, frame):
        """Update tracking for all detected people"""
        if frame is None or frame.size == 0:
            return {}
        
        # Get new detections
        detections = self.detect_people(frame)
        
        # Update existing tracks with predictions
        updated_tracks = {}
        matched_detection_indices = set()
        
        # First pass: Update existing tracks with predictions
        for track_id, track_data in list(self.tracked_people.items()):
            # Apply velocity decay
            if 'velocity' in track_data:
                dx, dy = track_data['velocity']
                dx *= self.tracking_settings['velocity_decay']
                dy *= self.tracking_settings['velocity_decay']
                
                # Stop very small movements
                if abs(dx) < self.tracking_settings['min_velocity']:
                    dx = 0
                if abs(dy) < self.tracking_settings['min_velocity']:
                    dy = 0
                
                track_data['velocity'] = (dx, dy)
            
            # Predict new position
            predicted_box = self.predict_position(track_data, frame.shape)
            
            if predicted_box:
                # Find best matching detection
                best_iou = 0
                best_match_index = -1
                
                for i, detection in enumerate(detections):
                    if i in matched_detection_indices:
                        continue
                    
                    # Calculate IOU with predicted position
                    iou = self.calculate_iou(predicted_box, detection['box'])
                    
                    if iou > self.tracking_settings['iou_threshold'] and iou > best_iou:
                        best_iou = iou
                        best_match_index = i
                
                if best_match_index != -1:
                    # Update track with matched detection
                    matched_detection_indices.add(best_match_index)
                    detection = detections[best_match_index]
                    
                    # Calculate new velocity
                    new_velocity = self.calculate_velocity(track_data['box'], detection['box'], frame.shape)
                    
                    # Smooth the velocity update
                    if 'velocity' in track_data:
                        old_dx, old_dy = track_data['velocity']
                        new_dx, new_dy = new_velocity
                        smoothed_dx = old_dx * (1 - self.tracking_settings['smooth_factor']) + new_dx * self.tracking_settings['smooth_factor']
                        smoothed_dy = old_dy * (1 - self.tracking_settings['smooth_factor']) + new_dy * self.tracking_settings['smooth_factor']
                        new_velocity = (smoothed_dx, smoothed_dy)
                    
                    # Update track data
                    updated_tracks[track_id] = {
                        'box': detection['box'],
                        'confidence': detection['confidence'],
                        'frames': self.tracking_settings['max_tracking_frames'],
                        'velocity': new_velocity,
                        'last_pos': ((detection['box'][0] + detection['box'][2])/2,
                                   (detection['box'][1] + detection['box'][3])/2)
                    }
                else:
                    # Use predicted position with decaying confidence
                    track_data['confidence'] *= 0.95
                    track_data['frames'] -= 1
                    
                    if track_data['frames'] > 0:
                        # Update position based on velocity
                        predicted_box = self.predict_position(track_data, frame.shape)
                        if predicted_box:
                            track_data['box'] = predicted_box
                            track_data['last_pos'] = ((predicted_box[0] + predicted_box[2])/2,
                                                    (predicted_box[1] + predicted_box[3])/2)
                        updated_tracks[track_id] = track_data
        
        # Second pass: Create new tracks for unmatched detections
        for i, detection in enumerate(detections):
            if i not in matched_detection_indices:
                new_id = self.track_counter
                self.track_counter += 1
                
                # Initialize new track
                updated_tracks[new_id] = {
                    'box': detection['box'],
                    'confidence': detection['confidence'],
                    'frames': self.tracking_settings['max_tracking_frames'],
                    'velocity': (0, 0),  # Initial velocity
                    'last_pos': ((detection['box'][0] + detection['box'][2])/2,
                               (detection['box'][1] + detection['box'][3])/2)
                }
        
        self.tracked_people = updated_tracks
        return updated_tracks
    
    def predict_position(self, track_data, frame_shape):
        """Predict person position based on velocity"""
        if 'velocity' not in track_data or 'last_pos' not in track_data:
            return None
            
        height, width = frame_shape[:2]
        last_x, last_y = track_data['last_pos']
        dx, dy = track_data['velocity']
        
        # Get original box dimensions
        box = track_data['box']
        box_w = box[2] - box[0]
        box_h = box[3] - box[1]
        
        # Calculate new center position
        new_x = last_x + (dx * width)
        new_y = last_y + (dy * height)
        
        # Ensure the box stays within frame bounds
        new_x = max(box_w/2, min(width - box_w/2, new_x))
        new_y = max(box_h/2, min(height - box_h/2, new_y))
        
        # Calculate new box coordinates
        left = int(new_x - box_w/2)
        top = int(new_y - box_h/2)
        right = int(new_x + box_w/2)
        bottom = int(new_y + box_h/2)
        
        # Final bounds check
        left = max(0, min(width - box_w, left))
        top = max(0, min(height - box_h, top))
        right = left + box_w
        bottom = top + box_h
        
        return (left, top, right, bottom)
    
    def calculate_velocity(self, old_box, new_box, frame_shape):
        """Calculate velocity between two positions"""
        height, width = frame_shape[:2]
        
        # Calculate center points
        old_center = ((old_box[0] + old_box[2])/2, (old_box[1] + old_box[3])/2)
        new_center = ((new_box[0] + new_box[2])/2, (new_box[1] + new_box[3])/2)
        
        # Calculate velocity (normalized by frame size)
        dx = (new_center[0] - old_center[0]) / width
        dy = (new_center[1] - old_center[1]) / height
        
        return (dx, dy)
    
    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union between two boxes"""
        # Get coordinates of intersection rectangle
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        # Calculate intersection area
        if x2 < x1 or y2 < y1:
            return 0.0
        
        intersection_area = (x2 - x1) * (y2 - y1)
        
        # Calculate union area
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - intersection_area
        
        return intersection_area / union_area if union_area > 0 else 0.0

class StoreEntryTracker:
    """Tracks and validates store entries to prevent false positives"""
    def __init__(self):
        self.entry_threshold = 15  # Minimum frames a person must be in store to count as entry
        self.exit_threshold = 10   # Minimum frames a person must be outside to count as exit
        self.min_confidence = 0.3  # Minimum confidence for tracking
        self.tracked_entries = {}  # Format: {track_id: {'store_id': store_id, 'frames_in_store': count, 'frames_outside': count, 'last_store': store_id}}
        self.store_entries = []    # List of confirmed store entries: [(person_id, store_id, timestamp, confidence)]
        
    def update_tracking(self, tracked_people, store_polygons, frame_number):
        """Update tracking for all detected people"""
        current_entries = set()  # Track which people are currently in stores
        
        for track_id, person_data in tracked_people.items():
            if person_data.get('confidence', 0) < self.min_confidence:
                continue
                
            # Get person center point (use bottom center for better ground position)
            box = person_data['box']
            person_center = ((box[0] + box[2])/2, box[3])  # Use bottom center point
            
            # Check which store (if any) contains this point
            current_store = None
            for store_id, polygon in store_polygons.items():
                if self.point_in_polygon(person_center, polygon):
                    current_store = store_id
                    current_entries.add(track_id)
                    break
            
            # Update tracking data
            if track_id not in self.tracked_entries:
                self.tracked_entries[track_id] = {
                    'store_id': current_store,
                    'frames_in_store': 1 if current_store else 0,
                    'frames_outside': 0 if current_store else 1,
                    'last_store': current_store,
                    'entry_time': frame_number if current_store else None,
                    'exit_time': None
                }
            else:
                track = self.tracked_entries[track_id]
                
                if current_store:
                    # Person is in a store
                    if track['last_store'] == current_store:
                        # Same store as before
                        track['frames_in_store'] += 1
                        track['frames_outside'] = 0
                    else:
                        # Different store or first entry
                        track['frames_in_store'] = 1
                        track['frames_outside'] = 0
                        track['entry_time'] = frame_number
                else:
                    # Person is outside stores
                    track['frames_outside'] += 1
                    if track['last_store'] is not None:
                        # Just exited a store
                        track['exit_time'] = frame_number
                
                # Check for confirmed entries/exits
                if track['frames_in_store'] >= self.entry_threshold and track['last_store'] != current_store:
                    # Confirmed entry into a new store
                    self.store_entries.append({
                        'person_id': track_id,
                        'store_id': current_store,
                        'entry_time': track['entry_time'],
                        'exit_time': None,
                        'confidence': person_data.get('confidence', 0)
                    })
                    track['last_store'] = current_store
                
                elif track['frames_outside'] >= self.exit_threshold and track['last_store'] is not None:
                    # Confirmed exit from a store
                    if self.store_entries and self.store_entries[-1]['exit_time'] is None:
                        self.store_entries[-1]['exit_time'] = track['exit_time']
                    track['last_store'] = None
                
                track['store_id'] = current_store
        
        # Clean up tracking data for people no longer detected
        for track_id in list(self.tracked_entries.keys()):
            if track_id not in tracked_people:
                del self.tracked_entries[track_id]
    
    def point_in_polygon(self, point, polygon):
        """Check if a point is inside a polygon using ray casting algorithm"""
        x, y = point
        n = len(polygon)
        inside = False
        
        p1x, p1y = polygon[0]
        for i in range(n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside
    
    def get_store_entries(self):
        """Get list of confirmed store entries"""
        return self.store_entries

class AWSProcessor:
    """Processes video using AWS Rekognition for person detection"""
    def __init__(self):
        self.rekognition = boto3.client('rekognition')
        self.tracked_people = {}  # Format: {person_id: {'store_id': store_id, 'frames_in_store': count}}
        self.person_counter = 0
        self.entry_threshold = 15  # Minimum frames to confirm store entry
        self.min_confidence = 0.7  # Minimum confidence for person detection
        
    def process_frame(self, frame, store_polygons, frame_number):
        """Process a single frame using AWS Rekognition"""
        try:
            # Convert frame to bytes
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            
            # Call AWS Rekognition
            response = self.rekognition.detect_labels(
                Image={'Bytes': frame_bytes},
                MaxLabels=20,
                MinConfidence=50.0
            )
            
            # Find person detections
            person_detections = []
            for label in response['Labels']:
                if label['Name'].lower() == 'person' and label['Confidence'] >= self.min_confidence * 100:
                    for instance in label.get('Instances', []):
                        if instance['Confidence'] >= self.min_confidence * 100:
                            # Get bounding box
                            box = instance['BoundingBox']
                            x = int(box['Left'] * frame.shape[1])
                            y = int(box['Top'] * frame.shape[0])
                            w = int(box['Width'] * frame.shape[1])
                            h = int(box['Height'] * frame.shape[0])
                            
                            # Calculate bottom center point (feet position)
                            feet_x = x + w//2
                            feet_y = y + h
                            
                            person_detections.append({
                                'box': (x, y, x + w, y + h),
                                'feet_pos': (feet_x, feet_y),
                                'confidence': instance['Confidence'] / 100.0
                            })
            
            # Update tracking for each detected person
            current_entries = set()
            for detection in person_detections:
                feet_pos = detection['feet_pos']
                
                # Check which store contains this person
                current_store = None
                for store_id, polygon in store_polygons.items():
                    if self.point_in_polygon(feet_pos, polygon):
                        current_store = store_id
                        current_entries.add(self.person_counter)
                        break
                
                # Assign new ID if this is a new person
                person_id = self.person_counter
                self.person_counter += 1
                
                # Update tracking data
                if person_id not in self.tracked_people:
                    self.tracked_people[person_id] = {
                        'store_id': current_store,
                        'frames_in_store': 1 if current_store else 0,
                        'entry_time': frame_number if current_store else None,
                        'last_store': current_store
                    }
                else:
                    track = self.tracked_people[person_id]
                    
                    if current_store:
                        if track['last_store'] == current_store:
                            track['frames_in_store'] += 1
                        else:
                            track['frames_in_store'] = 1
                            track['entry_time'] = frame_number
                    else:
                        if track['frames_in_store'] >= self.entry_threshold:
                            # Person has exited store
                            store_name = next((store['name'] for store_id, store in self.stores.items() 
                                            if store_id == track['last_store']), "Unknown Store")
                            print(f"Person {person_id} exited {store_name} at frame {frame_number}")
                        
                        track['frames_in_store'] = 0
                        track['entry_time'] = None
                    
                    # Check for confirmed entry
                    if track['frames_in_store'] == self.entry_threshold:
                        store_name = next((store['name'] for store_id, store in self.stores.items() 
                                        if store_id == current_store), "Unknown Store")
                        print(f"Person {person_id} entered {store_name} at frame {frame_number}")
                    
                    track['last_store'] = current_store
                
                # Draw detection
                x, y, x2, y2 = detection['box']
                cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Person {person_id}", (x, y-10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            return frame
            
        except Exception as e:
            print(f"Error in AWS processing: {str(e)}")
            return frame
    
    def point_in_polygon(self, point, polygon):
        """Check if a point is inside a polygon using ray casting algorithm"""
        x, y = point
        n = len(polygon)
        inside = False
        
        p1x, p1y = polygon[0]
        for i in range(n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside

class CCTVPreview(QWidget):
    """Widget for displaying and testing CCTV footage"""
    calibration_points_selected = pyqtSignal(list)  # List of (x, y) points
    status_message = pyqtSignal(str)  # Signal for status updates
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.video_capture = None
        self.current_frame = None
        self.scaled_frame = None
        self.stores = {}
        self.cameras = {}
        self.calibration_points = []
        self.store_perspective_matrices = {}  # Store individual matrices for each store
        self.calibration_mode = False
        self.test_mode = False
        self.calibration_point_labels = ["Top-Left", "Top-Right", "Bottom-Right", "Bottom-Left"]
        self.setMinimumSize(640, 480)
        self.setToolTip("CCTV footage preview with store detection")
        
        # Set size policy to allow widget to expand
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        # Set up video update timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(33)  # ~30 FPS
        
        # Replace face analyzer with person tracker
        self.person_tracker = PersonTracker()
        self.store_tracker = StoreEntryTracker()
        self.entry_results = []
        
        # Add AWS processor
        self.aws_processor = None
        self.processing_video = False
        self.processed_frame = None
    
    def load_video(self, video_path):
        """Load a video file for testing"""
        self.video_capture = cv2.VideoCapture(video_path)
        if not self.video_capture.isOpened():
            return False
        self.update_frame()
        return True
    
    def update_frame(self):
        """Update the current frame from video"""
        if self.video_capture is None:
            return
        
        ret, frame = self.video_capture.read()
        if not ret:
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop video
            ret, frame = self.video_capture.read()
            if not ret:
                return
        
        # Convert to RGB for display
        self.current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame with AWS if enabled
        if self.processing_video and self.aws_processor:
            try:
                # Get store polygons
                video_polygons = {}
                for store_id, store in self.stores.items():
                    if "video_polygon" in store:
                        video_polygons[store_id] = store["video_polygon"]
                
                # Process frame
                self.processed_frame = self.aws_processor.process_frame(
                    frame.copy(),
                    video_polygons,
                    int(self.video_capture.get(cv2.CAP_PROP_POS_FRAMES))
                )
                self.current_frame = cv2.cvtColor(self.processed_frame, cv2.COLOR_BGR2RGB)
            except Exception as e:
                print(f"Error in AWS frame processing: {str(e)}")
        
        # Scale frame to fit widget while maintaining aspect ratio
        self.update_scaled_frame()
        self.update()
    
    def update_scaled_frame(self):
        """Scale the current frame to fit the widget while maintaining aspect ratio"""
        if self.current_frame is None:
            return
            
        # Get widget size
        widget_size = self.size()
        
        # Calculate scaling factors
        frame_height, frame_width = self.current_frame.shape[:2]
        scale_w = widget_size.width() / frame_width
        scale_h = widget_size.height() / frame_height
        self.scale_factor = min(scale_w, scale_h)
        
        # Calculate new dimensions
        new_width = int(frame_width * self.scale_factor)
        new_height = int(frame_height * self.scale_factor)
        
        # Resize frame
        self.scaled_frame = cv2.resize(self.current_frame, (new_width, new_height))
        
        # Calculate offset to center the frame
        self.frame_offset_x = (widget_size.width() - new_width) // 2
        self.frame_offset_y = (widget_size.height() - new_height) // 2
    
    def resizeEvent(self, event):
        """Handle widget resize events"""
        super().resizeEvent(event)
        self.update_scaled_frame()
        self.update()
    
    def paintEvent(self, event):
        """Draw the video frame and store bounding boxes"""
        if self.scaled_frame is None:
            return
        
        painter = QPainter(self)
        
        # Draw video frame
        height, width = self.scaled_frame.shape[:2]
        qimage = QImage(self.scaled_frame.data, width, height,
                       self.scaled_frame.strides[0], QImage.Format.Format_RGB888)
        painter.drawImage(self.frame_offset_x, self.frame_offset_y, qimage)
        
        # Draw tracked people
        if self.test_mode and hasattr(self, 'person_tracker'):
            for track_id, person_data in self.person_tracker.tracked_people.items():
                box = person_data['box']
                left, top, right, bottom = box
                
                # Draw person box
                painter.setPen(QPen(QColor(0, 255, 0), 2))
                painter.drawRect(left, top, right - left, bottom - top)
                
                # Draw person ID
                painter.setFont(QFont("Arial", 10, QFont.Weight.Bold))
                painter.setPen(QColor(255, 255, 255))
                # Draw text background
                text = f"Person {track_id}"
                text_rect = painter.fontMetrics().boundingRect(text)
                text_rect.moveCenter(QPoint(left + (right - left)//2, top - 10))
                text_rect.adjust(-5, -2, 5, 2)
                painter.fillRect(text_rect, QColor(0, 0, 0, 180))
                # Draw text
                painter.drawText(left + (right - left)//2, top - 10, text)
        
        # Draw calibration points and lines if in calibration mode
        if self.calibration_mode:
            # Draw existing points
            for i, point in enumerate(self.calibration_points):
                x = int(point[0] * self.scale_factor + self.frame_offset_x)
                y = int(point[1] * self.scale_factor + self.frame_offset_y)
                
                # Draw point
                painter.setPen(QPen(QColor(255, 0, 0), 3))
                painter.drawEllipse(QPoint(x, y), 5, 5)
                
                # Draw label
                painter.setFont(QFont("Arial", 10, QFont.Weight.Bold))
                painter.setPen(QColor(255, 255, 255))
                # Draw text background
                text = f"{i+1}. {self.calibration_point_labels[i]}"
                text_rect = painter.fontMetrics().boundingRect(text)
                text_rect.moveCenter(QPoint(x, y - 20))
                text_rect.adjust(-5, -2, 5, 2)
                painter.fillRect(text_rect, QColor(0, 0, 0, 180))
                # Draw text
                painter.drawText(x, y - 20, text)
            
            # Draw lines between points
            if len(self.calibration_points) > 1:
                painter.setPen(QPen(QColor(255, 0, 0), 2, Qt.PenStyle.DashLine))
                for i in range(len(self.calibration_points) - 1):
                    x1 = int(self.calibration_points[i][0] * self.scale_factor + self.frame_offset_x)
                    y1 = int(self.calibration_points[i][1] * self.scale_factor + self.frame_offset_y)
                    x2 = int(self.calibration_points[i+1][0] * self.scale_factor + self.frame_offset_x)
                    y2 = int(self.calibration_points[i+1][1] * self.scale_factor + self.frame_offset_y)
                    painter.drawLine(x1, y1, x2, y2)
                
                # Draw line from last point to first point if we have 3 points
                if len(self.calibration_points) == 3:
                    x1 = int(self.calibration_points[-1][0] * self.scale_factor + self.frame_offset_x)
                    y1 = int(self.calibration_points[-1][1] * self.scale_factor + self.frame_offset_y)
                    x2 = int(self.calibration_points[0][0] * self.scale_factor + self.frame_offset_x)
                    y2 = int(self.calibration_points[0][1] * self.scale_factor + self.frame_offset_y)
                    painter.drawLine(x1, y1, x2, y2)
            
            # Draw next point indicator
            if len(self.calibration_points) < 4:
                next_point = self.calibration_point_labels[len(self.calibration_points)]
                painter.setFont(QFont("Arial", 12, QFont.Weight.Bold))
                painter.setPen(QColor(255, 255, 255))
                # Draw text background
                text = f"Click to place {next_point} point"
                text_rect = painter.fontMetrics().boundingRect(text)
                text_rect.moveCenter(QPoint(self.width() // 2, 30))
                text_rect.adjust(-10, -5, 10, 5)
                painter.fillRect(text_rect, QColor(0, 0, 0, 180))
                # Draw text
                painter.drawText(self.width() // 2, 30, text)
        
        # Draw store polygons in test mode
        if self.test_mode and self.stores:
            # Debug information
            print(f"Drawing {len(self.stores)} stores in test mode")
            
            for store_id, store in self.stores.items():
                if "video_polygon" in store and len(store["video_polygon"]) > 2:
                    try:
                        # Scale the transformed points to widget coordinates
                        video_polygon = []
                        for x, y in store["video_polygon"]:
                            scaled_x = int(x * self.scale_factor + self.frame_offset_x)
                            scaled_y = int(y * self.scale_factor + self.frame_offset_y)
                            video_polygon.append((scaled_x, scaled_y))
                        
                        if len(video_polygon) > 2:
                            # Draw filled semi-transparent polygon
                            painter.setPen(Qt.PenStyle.NoPen)
                            painter.setBrush(QColor(0, 255, 0, 50))  # Semi-transparent green
                            painter.drawPolygon([QPoint(x, y) for x, y in video_polygon])
                            
                            # Draw polygon outline
                            painter.setPen(QPen(QColor(0, 255, 0), 2))
                            painter.setBrush(Qt.BrushStyle.NoBrush)
                            for i in range(len(video_polygon) - 1):
                                painter.drawLine(video_polygon[i][0], video_polygon[i][1],
                                               video_polygon[i+1][0], video_polygon[i+1][1])
                            painter.drawLine(video_polygon[-1][0], video_polygon[-1][1],
                                           video_polygon[0][0], video_polygon[0][1])
                            
                            # Draw store name
                            if "name" in store:
                                centroid_x = int(np.mean([p[0] for p in video_polygon]))
                                centroid_y = int(np.mean([p[1] for p in video_polygon]))
                                painter.setFont(QFont("Arial", 10, QFont.Weight.Bold))
                                # Draw text background
                                text_rect = painter.fontMetrics().boundingRect(store["name"])
                                text_rect.moveCenter(QPoint(centroid_x, centroid_y))
                                text_rect.adjust(-5, -2, 5, 2)
                                painter.fillRect(text_rect, QColor(0, 0, 0, 180))
                                # Draw text
                                painter.setPen(QColor(255, 255, 255))
                                painter.drawText(centroid_x, centroid_y, store["name"])
                    except Exception as e:
                        print(f"Error drawing store {store_id}: {str(e)}")
        
        # Draw store entry information if in test mode
        if self.test_mode and self.entry_results:
            # Draw entry statistics
            painter.setFont(QFont("Arial", 12, QFont.Weight.Bold))
            painter.setPen(QColor(255, 255, 255))
            
            # Create entry summary
            store_entries = {}
            for entry in self.entry_results:
                store_id = entry['store_id']
                if store_id not in store_entries:
                    store_entries[store_id] = []
                store_entries[store_id].append(entry)
            
            # Draw summary at top of frame
            y_offset = 30
            for store_id, entries in store_entries.items():
                store_name = self.stores[store_id]['name']
                text = f"{store_name}: {len(entries)} entries"
                # Draw text background
                text_rect = painter.fontMetrics().boundingRect(text)
                text_rect.moveCenter(QPoint(self.width() // 2, y_offset))
                text_rect.adjust(-10, -5, 10, 5)
                painter.fillRect(text_rect, QColor(0, 0, 0, 180))
                # Draw text
                painter.drawText(self.width() // 2, y_offset, text)
                y_offset += 25
    
    def mousePressEvent(self, event):
        """Handle mouse clicks for calibration"""
        if not self.calibration_mode:
            return
        
        # Convert widget coordinates to original video coordinates
        x = (event.position().x() - self.frame_offset_x) / self.scale_factor
        y = (event.position().y() - self.frame_offset_y) / self.scale_factor
        
        # Check if click is within video bounds
        if not (0 <= x < self.current_frame.shape[1] and 0 <= y < self.current_frame.shape[0]):
            QMessageBox.warning(self, "Invalid Point", 
                "Please click within the video frame.")
            return
        
        if len(self.calibration_points) < 4:
            # Add point with validation
            if len(self.calibration_points) > 0:
                # Check if point is too close to existing points
                for existing_point in self.calibration_points:
                    distance = np.sqrt((x - existing_point[0])**2 + (y - existing_point[1])**2)
                    if distance < 20:  # Minimum 20 pixels between points
                        QMessageBox.warning(self, "Invalid Point", 
                            "Please click further away from existing points.")
                        return
            
            self.calibration_points.append((x, y))
            if len(self.calibration_points) == 4:
                # Validate the quadrilateral
                points = np.array(self.calibration_points)
                # Check if points form a reasonable quadrilateral
                if not self.validate_quadrilateral(points):
                    QMessageBox.warning(self, "Invalid Points", 
                        "The selected points do not form a valid quadrilateral.\n"
                        "Please try again, making sure to click points in the correct order:\n"
                        "1. Top-Left\n2. Top-Right\n3. Bottom-Right\n4. Bottom-Left")
                    self.calibration_points = []
                    return
                self.calibration_points_selected.emit(self.calibration_points)
            self.update()

    def validate_quadrilateral(self, points):
        """Validate that the points form a reasonable quadrilateral"""
        try:
            # Calculate distances between consecutive points
            distances = []
            for i in range(4):
                j = (i + 1) % 4
                dist = np.linalg.norm(points[i] - points[j])
                distances.append(dist)
            
            # Check if any side is too short
            if min(distances) < 20:  # Minimum 20 pixels
                return False
            
            # Check if the quadrilateral is too skewed
            # Calculate angles between consecutive sides
            angles = []
            for i in range(4):
                j = (i + 1) % 4
                k = (i + 2) % 4
                v1 = points[j] - points[i]
                v2 = points[k] - points[j]
                angle = np.abs(np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))))
                angles.append(np.degrees(angle))
            
            # Check if any angle is too small or too large
            if min(angles) < 30 or max(angles) > 150:  # Angles should be between 30 and 150 degrees
                return False
            
            return True
        except Exception as e:
            print(f"Error validating quadrilateral: {str(e)}")
            return False

    def calculate_perspective_transform(self, blueprint_points, video_points, store_id):
        """Calculate perspective transformation matrix for a specific store"""
        if len(blueprint_points) != 4 or len(video_points) != 4:
            return False
        
        try:
            # Convert points to numpy arrays
            src_points = np.array(blueprint_points, dtype=np.float32)
            dst_points = np.array(video_points, dtype=np.float32)
            
            # Calculate perspective transform matrix
            perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
            
            # Basic validation that the matrix is valid
            if perspective_matrix is None or perspective_matrix.shape != (3, 3):
                print(f"Error: Failed to create valid perspective matrix for store {store_id}")
                return False
            
            # Store the matrix for this specific store
            self.store_perspective_matrices[store_id] = perspective_matrix
            print(f"Perspective transformation matrix created successfully for store {store_id}")
            return True
            
        except Exception as e:
            print(f"Error in perspective transformation for store {store_id}: {str(e)}")
            return False

    def set_calibration_mode(self, enabled):
        """Enable/disable calibration point selection"""
        self.calibration_mode = enabled
        self.calibration_points = []  # Clear existing calibration points
        if enabled:
            self.setToolTip("Click 4 points in the video in order: Top-Left, Top-Right, Bottom-Right, Bottom-Left")
        else:
            self.setToolTip("CCTV footage preview with store detection")
        self.update()

    def set_test_mode(self, enabled):
        """Enable/disable test mode for store detection"""
        self.test_mode = enabled
        if enabled:
            # Debug information
            print(f"Test mode enabled. Stores: {len(self.stores)}, Perspective matrices: {len(self.store_perspective_matrices)}")
            if not self.stores:
                print("Warning: No stores defined")
            self.setToolTip("Test mode: Store boundaries are highlighted in video")
        else:
            self.setToolTip("CCTV footage preview with store detection")
        self.update()

    def start_aws_processing(self):
        """Start AWS-based video processing"""
        if not self.video_capture or not self.stores:
            self.status_message.emit("Error: Please load both video and store mapping first")
            return False
        
        if not self.aws_processor:
            self.aws_processor = AWSProcessor()
        
        self.processing_video = True
        self.status_message.emit("Processing video with AWS Rekognition...")
        return True
    
    def stop_aws_processing(self):
        """Stop AWS-based video processing"""
        self.processing_video = False
        self.status_message.emit("AWS processing stopped")

class MainWindow(QMainWindow):
    """Main application window"""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mall Blueprint Mapping Tool")
        self.setMinimumSize(1200, 800)
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QHBoxLayout(central_widget)
        layout.setSpacing(10)  # Add some spacing between widgets
        
        # Create splitter for blueprint and video views
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Create main view with fixed width ratio
        self.blueprint_view = BlueprintView()
        splitter.addWidget(self.blueprint_view)
        
        # Create video preview
        self.video_preview = CCTVPreview()
        splitter.addWidget(self.video_preview)
        
        # Set initial splitter sizes (60% blueprint, 40% video)
        total_width = self.width()
        splitter.setSizes([int(total_width * 0.6), int(total_width * 0.4)])
        
        # Create side panel with fixed width
        side_panel = QWidget()
        side_panel.setFixedWidth(300)  # Fixed width for side panel
        side_layout = QVBoxLayout(side_panel)
        side_layout.setContentsMargins(10, 10, 10, 10)  # Add some margins
        
        # Add tool selection
        tool_group = QGroupBox("Tools")
        tool_layout = QVBoxLayout(tool_group)
        tool_layout.setSpacing(5)  # Reduce spacing between tools
        
        self.select_tool_btn = QPushButton("Select")
        self.camera_tool_btn = QPushButton("Add Camera")
        self.store_tool_btn = QPushButton("Define Store")
        self.calibrate_btn = QPushButton("Calibrate Camera")
        self.test_mode_btn = QPushButton("Test Mapping")
        self.aws_process_btn = QPushButton("Process with AWS")
        
        # Add tooltips
        self.select_tool_btn.setToolTip("Select and move cameras or stores")
        self.camera_tool_btn.setToolTip("Click to place camera, drag to set orientation")
        self.store_tool_btn.setToolTip("Click to create store polygon, double-click to complete")
        self.calibrate_btn.setToolTip("Calibrate camera view with blueprint")
        self.test_mode_btn.setToolTip("Toggle test mode to see store boundaries in video")
        self.aws_process_btn.setToolTip("Process video using AWS Rekognition for person detection")
        
        # Set fixed height for buttons
        button_height = 30
        for btn in [self.select_tool_btn, self.camera_tool_btn, 
                   self.store_tool_btn, self.calibrate_btn, self.test_mode_btn, self.aws_process_btn]:
            btn.setFixedHeight(button_height)
            tool_layout.addWidget(btn)
        
        # Add help text with scroll area
        help_group = QGroupBox("Help")
        help_layout = QVBoxLayout(help_group)
        
        help_text = QTextEdit()
        help_text.setReadOnly(True)
        help_text.setHtml("""
            <h3>How to Use This Tool</h3>
            <ol>
                <li><b>Load Blueprint</b>
                    <ul>
                        <li>Click "Load Blueprint" in the toolbar</li>
                        <li>Select your mall blueprint image</li>
                        <li>The image should be clear and show store boundaries</li>
                    </ul>
                </li>
                <li><b>Add Cameras</b>
                    <ul>
                        <li>Click the "Add Camera" tool</li>
                        <li>Click on the blueprint where each camera is located</li>
                        <li>Drag from the camera to set its orientation</li>
                        <li>The red cone shows the camera's field of view</li>
                    </ul>
                </li>
                <li><b>Define Stores</b>
                    <ul>
                        <li>Click the "Define Store" tool</li>
                        <li>Click to create polygon points around each store</li>
                        <li>Double-click to complete the polygon</li>
                        <li>Enter store name and category when prompted</li>
                    </ul>
                </li>
                <li><b>Calibrate Camera View</b>
                    <ul>
                        <li>Load CCTV footage from a camera</li>
                        <li>Click "Calibrate Camera"</li>
                        <li>Click 4 points in the blueprint that you can identify in the video</li>
                        <li>Click the same 4 points in the video</li>
                        <li>The system will map the blueprint to the video view</li>
                    </ul>
                </li>
            </ol>
        """)
        
        help_layout.addWidget(help_text)
        
        # Add groups to side layout
        side_layout.addWidget(tool_group)
        side_layout.addWidget(help_group)
        
        # Add widgets to main layout with proper proportions
        layout.addWidget(splitter, stretch=60)  # 60% of space
        layout.addWidget(side_panel, stretch=40)  # 40% of space
        
        # Create toolbar
        toolbar = QToolBar()
        toolbar.setMovable(False)  # Prevent toolbar from being moved
        self.addToolBar(toolbar)
        
        # Add toolbar actions with tooltips
        load_blueprint_action = QAction("Load Blueprint", self)
        load_blueprint_action.setToolTip("Load a mall blueprint image")
        load_blueprint_action.triggered.connect(self.load_blueprint)
        
        load_video_action = QAction("Load CCTV", self)
        load_video_action.setToolTip("Load CCTV footage for testing")
        load_video_action.triggered.connect(self.load_video)
        
        export_action = QAction("Export", self)
        export_action.setToolTip("Export the mapping data to JSON")
        export_action.triggered.connect(self.export_data)
        
        load_mapping_action = QAction("Load Mapping", self)
        load_mapping_action.setToolTip("Load previously exported mapping data")
        load_mapping_action.triggered.connect(self.load_mapping_data)
        
        toolbar.addAction(load_blueprint_action)
        toolbar.addAction(load_video_action)
        toolbar.addAction(export_action)
        toolbar.addAction(load_mapping_action)
        
        # Create status bar
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("Ready")
        
        # Connect signals
        self.select_tool_btn.clicked.connect(lambda: self.blueprint_view.set_tool("select"))
        self.camera_tool_btn.clicked.connect(lambda: self.blueprint_view.set_tool("camera"))
        self.store_tool_btn.clicked.connect(lambda: self.blueprint_view.set_tool("store"))
        self.calibrate_btn.clicked.connect(self.prepare_calibration)
        self.test_mode_btn.clicked.connect(self.toggle_test_mode)
        self.aws_process_btn.clicked.connect(self.toggle_aws_processing)
        
        self.blueprint_view.calibration_points_selected.connect(self.on_blueprint_calibration_points)
        self.video_preview.calibration_points_selected.connect(self.on_video_calibration_points)
        
        # Create processor
        self.processor = BlueprintProcessor()
        
        # Add calibration state tracking
        self.calibration_blueprint_points = None
        self.current_calibration_store = None  # Track which store is being calibrated
        
        # Remove video analyzer initialization
        self.video_analyzer = None
        
        # Connect CCTVPreview status signal
        self.video_preview.status_message.connect(self.statusBar.showMessage)
        
        # Add AWS processor status to status bar
        self.aws_status_label = QLabel("AWS: Disabled")
        self.statusBar.addPermanentWidget(self.aws_status_label)
    
    def load_blueprint(self):
        """Load a blueprint image"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Blueprint", "", "Image Files (*.png *.jpg *.jpeg)"
        )
        if file_path:
            if self.blueprint_view.load_image(file_path):
                self.statusBar.showMessage(f"Loaded blueprint: {file_path}")
            else:
                QMessageBox.critical(self, "Error", "Failed to load blueprint image")
    
    def load_video(self):
        """Load CCTV footage for testing"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load CCTV Footage", "", "Video Files (*.mp4 *.avi *.mov)"
        )
        if file_path:
            if self.video_preview.load_video(file_path):
                self.statusBar.showMessage(f"Loaded video: {file_path}")
                
                # If we have mapping data, suggest testing
                if self.video_preview.store_perspective_matrices:
                    reply = QMessageBox.question(self, "Test Mapping",
                        "Mapping data is available. Would you like to test the mapping with this video?",
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
                    
                    if reply == QMessageBox.StandardButton.Yes:
                        self.test_mode_btn.setChecked(True)
                        self.toggle_test_mode(True)
            else:
                QMessageBox.critical(self, "Error", "Failed to load video file")
    
    def prepare_calibration(self):
        """Prepare for calibration by selecting a store"""
        if self.blueprint_view.blueprint_image is None or self.video_preview.current_frame is None:
            QMessageBox.warning(self, "Error", "Please load both blueprint and video first")
            return
        
        # Check if there are any stores defined
        if not self.blueprint_view.stores:
            QMessageBox.warning(self, "Error", "Please define at least one store before calibration")
            return
        
        # Create store selection dialog
        store_dialog = QDialog(self)
        store_dialog.setWindowTitle("Select Store for Calibration")
        store_dialog.setModal(True)
        
        layout = QVBoxLayout(store_dialog)
        
        # Add store selection combo box
        store_combo = QComboBox()
        for store_id, store in self.blueprint_view.stores.items():
            store_combo.addItem(f"{store['name']} ({store_id})", store_id)
        layout.addWidget(QLabel("Select store to calibrate:"))
        layout.addWidget(store_combo)
        
        # Add calibration instructions
        instructions = QLabel(
            "IMPORTANT: The blueprint shows stores vertically, but in the video they appear horizontally.\n\n"
            "When selecting points:\n"
            "1. In the blueprint, click points in this order:\n"
            "   • Top-left corner of the store area\n"
            "   • Top-right corner of the store area\n"
            "   • Bottom-right corner of the store area\n"
            "   • Bottom-left corner of the store area\n"
            "2. In the video, click the same 4 points in the same order\n"
            "   (The points should form a rectangle around the stores)"
        )
        instructions.setWordWrap(True)
        layout.addWidget(instructions)
        
        # Add buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | 
            QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(store_dialog.accept)
        button_box.rejected.connect(store_dialog.reject)
        layout.addWidget(button_box)
        
        if store_dialog.exec() == QDialog.DialogCode.Accepted:
            selected_store_id = store_combo.currentData()
            self.current_calibration_store = selected_store_id
            
            # Show calibration instructions
            dialog = CalibrationDialog(self)
            dialog.setText(f"Calibrating Camera for {self.blueprint_view.stores[selected_store_id]['name']}")
            
            if dialog.exec() == QMessageBox.StandardButton.Ok:
                self.blueprint_view.set_tool("calibrate")
                self.video_preview.set_calibration_mode(True)
                self.statusBar.showMessage(
                    f"Step 1: Click 4 points around {self.blueprint_view.stores[selected_store_id]['name']} "
                    "in the blueprint (top-left, top-right, bottom-right, bottom-left)"
                )

    def on_blueprint_calibration_points(self, points):
        """Handle blueprint calibration points selection"""
        self.calibration_blueprint_points = points
        self.statusBar.showMessage(
            "Step 2: Now click the same 4 points in the video in the same order "
            "(top-left, top-right, bottom-right, bottom-left)"
        )

    def on_video_calibration_points(self, points):
        """Handle video calibration points selection"""
        if self.calibration_blueprint_points is None or self.current_calibration_store is None:
            return
        
        # Calculate perspective transform for this specific store
        if self.video_preview.calculate_perspective_transform(
            self.calibration_blueprint_points, points, self.current_calibration_store):
            
            # Update mapping status for the current store only
            self.blueprint_view.mapped_stores.add(self.current_calibration_store)
            
            store_name = self.blueprint_view.stores[self.current_calibration_store]['name']
            self.statusBar.showMessage(f"Camera calibration complete - {store_name} mapped")
            
            # Show transformation details
            QMessageBox.information(self, "Calibration Complete", 
                f"Camera calibration successful!\n\n"
                f"Store '{store_name}' has been mapped to the video view.\n\n"
                f"Each store now has its own perspective transformation, ensuring correct positioning "
                f"relative to other stores in the CCTV view.\n\n"
                f"You can now test the mapping to verify the store boundaries are correctly positioned.")
        else:
            QMessageBox.warning(self, "Error", 
                "Calibration failed. Please try again, making sure to:\n\n"
                "1. Click points in the correct order (top-left, top-right, bottom-right, bottom-left)\n"
                "2. Select points that form a proper rectangle around the stores\n"
                "3. Ensure the points in the video correspond to the same locations as in the blueprint")
        
        # Reset calibration mode
        self.blueprint_view.set_tool("select")
        self.video_preview.set_calibration_mode(False)
        self.calibration_blueprint_points = None
        self.current_calibration_store = None
        self.blueprint_view.update()
    
    def toggle_test_mode(self, enabled):
        """Toggle test mode for store mapping visualization"""
        if self.video_preview.current_frame is None:
            QMessageBox.warning(self, "Error", "Please load a video first")
            self.test_mode_btn.setChecked(False)
            return
        
        if not self.video_preview.store_perspective_matrices:
            QMessageBox.warning(self, "Error", "Please calibrate at least one store first")
            self.test_mode_btn.setChecked(False)
            return
        
        if not self.video_preview.stores:
            QMessageBox.warning(self, "Error", "No stores defined for mapping")
            self.test_mode_btn.setChecked(False)
            return
        
        print(f"Toggling test mode: {enabled}")
        print(f"Stores available: {len(self.video_preview.stores)}")
        print(f"Perspective matrices available: {len(self.video_preview.store_perspective_matrices)}")
        
        self.video_preview.set_test_mode(enabled)
        if enabled:
            self.statusBar.showMessage("Test mode: Store boundaries are highlighted in video")
        else:
            self.statusBar.showMessage("Test mode disabled")
    
    def export_data(self):
        """Export the mapping data"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Data", "", "JSON Files (*.json)"
        )
        if file_path:
            try:
                # Prepare data for export
                export_data = {
                    "blueprint": {
                        "image_path": self.blueprint_view.blueprint_path if hasattr(self.blueprint_view, 'blueprint_path') else None,
                        "scale_factor": self.blueprint_view.scale_factor if hasattr(self.blueprint_view, 'scale_factor') else 1.0
                    },
                    "cameras": {
                        camera_id: {
                            "position": camera["position"],
                            "orientation": camera["orientation"],
                            "fov_angle": camera.get("fov_angle", 70),
                            "fov_range": camera.get("fov_range", 100)
                        }
                        for camera_id, camera in self.blueprint_view.cameras.items()
                    },
                    "stores": {
                        store_id: {
                            "name": store["name"],
                            "category": store.get("category", ""),
                            "polygon": store["polygon"],
                            "is_mapped": store_id in self.blueprint_view.mapped_stores
                        }
                        for store_id, store in self.blueprint_view.stores.items()
                    },
                    "calibration": {
                        "store_matrices": {
                            store_id: matrix.tolist() for store_id, matrix in self.video_preview.store_perspective_matrices.items()
                        },
                        "blueprint_points": self.calibration_blueprint_points if self.calibration_blueprint_points else None,
                        "video_points": self.video_preview.calibration_points if self.video_preview.calibration_points else None
                    },
                    "test_results": {
                        "total_stores": len(self.blueprint_view.stores),
                        "mapped_stores": len(self.blueprint_view.mapped_stores),
                        "mapping_status": {
                            store_id: {
                                "name": store["name"],
                                "is_mapped": store_id in self.blueprint_view.mapped_stores,
                                "calibration_points": self.calibration_blueprint_points if store_id == self.current_calibration_store else None
                            }
                            for store_id, store in self.blueprint_view.stores.items()
                        }
                    }
                }
                
                # Save to JSON file
                with open(file_path, 'w') as f:
                    json.dump(export_data, f, indent=2)
                
                self.statusBar.showMessage(f"Successfully exported data to: {file_path}")
                QMessageBox.information(self, "Export Successful", 
                    f"Mapping data has been exported to:\n{file_path}\n\n"
                    f"Exported {len(export_data['stores'])} stores and {len(export_data['cameras'])} cameras.")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to export data: {str(e)}")
                self.statusBar.showMessage("Export failed")

    def load_mapping_data(self):
        """Load previously exported mapping data"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Mapping Data", "", "JSON Files (*.json)"
        )
        if not file_path:
            return
            
        try:
            with open(file_path, 'r') as f:
                mapping_data = json.load(f)
            
            # Debug information
            print(f"Loading mapping data from: {file_path}")
            print(f"Found {len(mapping_data.get('stores', {}))} stores")
            print(f"Found {len(mapping_data.get('cameras', {}))} cameras")
            
            # Load blueprint if path exists
            if mapping_data.get("blueprint", {}).get("image_path"):
                blueprint_path = mapping_data["blueprint"]["image_path"]
                if os.path.exists(blueprint_path):
                    if not self.blueprint_view.load_image(blueprint_path):
                        QMessageBox.warning(self, "Warning", 
                            "Could not load original blueprint image. Using current blueprint if available.")
            
            # Load stores first (needed for mapping status)
            self.blueprint_view.stores = {
                store_id: {
                    "name": store["name"],
                    "category": store.get("category", ""),
                    "polygon": store["polygon"]
                }
                for store_id, store in mapping_data.get("stores", {}).items()
            }
            
            # Update mapped stores status
            self.blueprint_view.mapped_stores = {
                store_id for store_id, store in mapping_data.get("stores", {}).items()
                if store.get("is_mapped", False)
            }
            
            # Load cameras
            self.blueprint_view.cameras = {
                camera_id: {
                    "position": tuple(camera["position"]),
                    "orientation": camera["orientation"],
                    "fov_angle": camera.get("fov_angle", 70),
                    "fov_range": camera.get("fov_range", 100)
                }
                for camera_id, camera in mapping_data.get("cameras", {}).items()
            }
            
            # Load calibration data if available
            calibration_data = mapping_data.get("calibration", {})
            if calibration_data.get("store_matrices"):
                self.video_preview.store_perspective_matrices = {
                    store_id: np.array(matrix, dtype=np.float32) for store_id, matrix in calibration_data["store_matrices"].items()
                }
                print("Loaded perspective matrices for stores")
            else:
                print("No perspective matrices found in mapping data")
            
            # Copy stores to video preview
            self.video_preview.stores = self.blueprint_view.stores.copy()
            
            # Update UI
            self.blueprint_view.update()
            self.video_preview.update()
            self.statusBar.showMessage(f"Loaded mapping data from: {file_path}")
            
            # Show summary
            QMessageBox.information(self, "Mapping Data Loaded",
                f"Successfully loaded mapping data:\n\n"
                f"• {len(self.blueprint_view.stores)} stores\n"
                f"• {len(self.blueprint_view.cameras)} cameras\n"
                f"• {len(self.blueprint_view.mapped_stores)} mapped stores\n"
                f"• Perspective matrices: {len(self.video_preview.store_perspective_matrices)} stores\n"
                f"• Matrix shapes: {[matrix.shape for matrix in self.video_preview.store_perspective_matrices.values()]}\n\n"
                "You can now load a new video to test the mapping.")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load mapping data: {str(e)}")
            self.statusBar.showMessage("Failed to load mapping data")
            print(f"Error loading mapping data: {str(e)}")

    def toggle_aws_processing(self, enabled):
        """Toggle AWS-based video processing"""
        if enabled:
            if self.video_preview.start_aws_processing():
                self.aws_process_btn.setText("Stop AWS Processing")
                self.aws_status_label.setText("AWS: Processing")
            else:
                self.aws_process_btn.setChecked(False)
        else:
            self.video_preview.stop_aws_processing()
            self.aws_process_btn.setText("Process with AWS")
            self.aws_status_label.setText("AWS: Disabled")

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main() 