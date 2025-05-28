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
from botocore.exceptions import ClientError
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QMessageBox, QToolBar,
    QStatusBar, QDockWidget, QTextEdit, QSpinBox, QDoubleSpinBox,
    QComboBox, QGroupBox, QFormLayout, QTabWidget, QToolTip,
    QSplitter, QFrame, QSizePolicy, QDialog, QLineEdit, QDialogButtonBox,
    QProgressDialog
)
from PyQt6.QtCore import Qt, QPoint, QRect, pyqtSignal, QTimer
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont, QAction
from blueprint_mapping import BlueprintProcessor
from datetime import datetime
from shapely.geometry import Polygon, Point

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
    def __init__(self):
        self.next_id = 1
        self.tracked_people = {}  # id -> {bbox, last_seen, current_store, history}
        self.store_entry_threshold = 0.5  # How much of the person needs to be in store to count as entry
        self.max_frames_missing = 30  # How many frames a person can be missing before removing
        self.iou_threshold = 0.3  # IOU threshold for matching detections to tracked people
    
    def calculate_iou(self, bbox1, bbox2):
        """Calculate Intersection over Union between two bounding boxes"""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[0] + bbox1[2], bbox2[0] + bbox2[2])
        y2 = min(bbox1[1] + bbox1[3], bbox2[1] + bbox2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        bbox1_area = bbox1[2] * bbox1[3]
        bbox2_area = bbox2[2] * bbox2[3]
        union = bbox1_area + bbox2_area - intersection
        
        return intersection / union if union > 0 else 0
    
    def is_person_in_store(self, person_bbox, store_polygon):
        """Check if a person is inside a store using polygon intersection"""
        try:
            # Create person bounding box polygon
            x, y, w, h = person_bbox
            person_polygon = Polygon([
                (x, y), (x + w, y), (x + w, y + h), (x, y + h)
            ])
            
            # Create store polygon
            store_polygon = Polygon(store_polygon)
            
            # Calculate intersection
            if person_polygon.intersects(store_polygon):
                intersection = person_polygon.intersection(store_polygon)
                # If more than threshold of person is in store, count as entry
                return (intersection.area / person_polygon.area) > self.store_entry_threshold
            
            return False
        except Exception as e:
            print(f"Error checking store entry: {str(e)}")
            return False
    
    def update(self, detected_people, stores, frame_number):
        """Update tracked people with new detections"""
        current_time = datetime.now()
        
        # Update existing tracks
        for person_id in list(self.tracked_people.keys()):
            person = self.tracked_people[person_id]
            person['last_seen'] = frame_number
            person['current_store'] = None  # Reset current store, will update if still in one
        
        # Match new detections to existing tracks
        matched_detections = set()
        for person_id, person in self.tracked_people.items():
            best_iou = 0
            best_detection = None
            
            for i, detection in enumerate(detected_people):
                if i in matched_detections:
                    continue
                
                iou = self.calculate_iou(person['bbox'], detection['bbox'])
                if iou > self.iou_threshold and iou > best_iou:
                    best_iou = iou
                    best_detection = (i, detection)
            
            if best_detection:
                idx, detection = best_detection
                matched_detections.add(idx)
                person['bbox'] = detection['bbox']
                person['confidence'] = detection['confidence']
                
                # Check store entry
                for store_id, store in stores.items():
                    if "video_polygon" in store and len(store["video_polygon"]) > 2:
                        if self.is_person_in_store(detection['bbox'], store["video_polygon"]):
                            if person['current_store'] != store_id:
                                # Person entered a new store
                                entry_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
                                person['history'].append({
                                    'store_id': store_id,
                                    'store_name': store.get('name', 'Unknown'),
                                    'entry_time': entry_time,
                                    'frame': frame_number
                                })
                                # Update the last store entry for notification
                                if hasattr(self, 'parent') and isinstance(self.parent, CCTVPreview):
                                    self.parent.last_store_entry = {
                                        'person_id': person_id,
                                        'store_name': store.get('name', 'Unknown'),
                                        'entry_time': entry_time
                                    }
                                    self.parent.store_entry_display_time = self.parent.notification_duration
                                print(f"Person {person_id} entered {store.get('name', 'Unknown')} at {entry_time}")
                            person['current_store'] = store_id
                            break
        
        # Add new tracks for unmatched detections
        for i, detection in enumerate(detected_people):
            if i not in matched_detections:
                person_id = self.next_id
                self.next_id += 1
                
                # Check initial store
                current_store = None
                for store_id, store in stores.items():
                    if "video_polygon" in store and len(store["video_polygon"]) > 2:
                        if self.is_person_in_store(detection['bbox'], store["video_polygon"]):
                            current_store = store_id
                            break
                
                self.tracked_people[person_id] = {
                    'bbox': detection['bbox'],
                    'confidence': detection['confidence'],
                    'last_seen': frame_number,
                    'current_store': current_store,
                    'history': []
                }
                
                if current_store:
                    entry_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
                    self.tracked_people[person_id]['history'].append({
                        'store_id': current_store,
                        'store_name': stores[current_store].get('name', 'Unknown'),
                        'entry_time': entry_time,
                        'frame': frame_number
                    })
                    print(f"Person {person_id} entered {stores[current_store].get('name', 'Unknown')} at {entry_time}")
        
        # Remove old tracks
        self.tracked_people = {
            pid: person for pid, person in self.tracked_people.items()
            if frame_number - person['last_seen'] < self.max_frames_missing
        }
        
        return self.tracked_people

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
        
        # AWS Rekognition setup
        self.rekognition_client = None
        self.aws_enabled = False
        self.detected_people = []  # List of current detected people with bounding boxes
        self.frame_count = 0
        self.detection_interval = 29  # Process every 5th frame for performance
        
        # Store entry notification setup
        self.last_store_entry = None  # Track the last store entry
        self.store_entry_display_time = 0  # How long to show the entry message (in frames)
        self.notification_duration = 30  # Show notification for 30 frames (about 1 second)
        
        # Set size policy to allow widget to expand
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        # Set up video update timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(33)  # ~30 FPS
        
        self.video_writer = None
        self.is_exporting = False
        self.export_progress = 0
        self.total_frames = 0
        
        # Add person tracker
        self.person_tracker = PersonTracker()
        self.frame_number = 0
    
    def enable_aws_rekognition(self, aws_region='ap-south-1'):
        """Enable AWS Rekognition using default credentials"""
        try:
            # Use default credentials with specified region
            self.rekognition_client = boto3.client('rekognition', region_name=aws_region)
            
            # Test the connection
            self.rekognition_client.detect_labels(
                Image={'Bytes': cv2.imencode('.jpg', np.zeros((100, 100, 3), dtype=np.uint8))[1].tobytes()},
                MaxLabels=1
            )
            
            self.aws_enabled = True
            self.status_message.emit("AWS Rekognition enabled successfully")
            return True
            
        except ClientError as e:
            self.aws_enabled = False
            self.rekognition_client = None
            self.status_message.emit(f"AWS Rekognition error: {str(e)}")
            return False
        except Exception as e:
            self.aws_enabled = False
            self.rekognition_client = None
            self.status_message.emit(f"Error enabling AWS Rekognition: {str(e)}")
            return False
    
    def detect_people(self, frame):
        """Detect people in the frame using AWS Rekognition"""
        if not self.aws_enabled or self.rekognition_client is None:
            return []
        
        try:
            # Convert frame to JPEG bytes
            _, jpeg_bytes = cv2.imencode('.jpg', frame)
            
            # Call AWS Rekognition
            response = self.rekognition_client.detect_labels(
                Image={'Bytes': jpeg_bytes.tobytes()},
                MaxLabels=10,
                MinConfidence=70.0
            )
            
            # Filter for people and extract bounding boxes
            people = []
            for label in response['Labels']:
                if label['Name'].lower() == 'person':
                    for instance in label.get('Instances', []):
                        if instance['Confidence'] >= 70.0:  # Only include high confidence detections
                            bbox = instance['BoundingBox']
                            # Convert normalized coordinates to pixel coordinates
                            x = int(bbox['Left'] * frame.shape[1])
                            y = int(bbox['Top'] * frame.shape[0])
                            width = int(bbox['Width'] * frame.shape[1])
                            height = int(bbox['Height'] * frame.shape[0])
                            confidence = instance['Confidence']
                            people.append({
                                'bbox': (x, y, width, height),
                                'confidence': confidence
                            })
            
            return people
            
        except Exception as e:
            print(f"Error in person detection: {str(e)}")
            return []
    
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
        
        # Process frame for person detection if AWS is enabled
        if self.aws_enabled:
            self.frame_count += 1
            if self.frame_count % self.detection_interval == 0:
                detected_people = self.detect_people(frame)
                # Update person tracking
                self.tracked_people = self.person_tracker.update(
                    detected_people, self.stores, self.frame_number
                )
        
        self.frame_number += 1
        
        # Convert to RGB for display
        self.current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Scale frame to fit widget while maintaining aspect ratio
        self.update_scaled_frame()
        
        # Apply perspective transformation if available
        if self.stores:
            try:
                # Transform store polygons to video coordinates
                for store_id, store in self.stores.items():
                    if "polygon" in store and len(store["polygon"]) > 2:
                        # Get the store's perspective matrix
                        perspective_matrix = self.store_perspective_matrices.get(store_id)
                        if perspective_matrix is None:
                            continue
                            
                        # Convert points to numpy array with correct shape (N,1,2)
                        points = np.array(store["polygon"], dtype=np.float32).reshape(-1, 1, 2)
                        
                        # Ensure perspective matrix is 3x3
                        if perspective_matrix.shape != (3, 3):
                            print(f"Warning: Invalid perspective matrix shape for store {store_id}: {perspective_matrix.shape}")
                            continue
                        
                        # Transform points
                        transformed = cv2.perspectiveTransform(points, perspective_matrix)
                        
                        # Store transformed points
                        store["video_polygon"] = [(int(p[0][0]), int(p[0][1])) for p in transformed]
            except Exception as e:
                print(f"Error in perspective transformation: {str(e)}")
        
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
        """Draw the video frame and overlays"""
        if self.scaled_frame is None:
            return
        
        painter = QPainter(self)
        
        # Draw video frame
        height, width = self.scaled_frame.shape[:2]
        qimage = QImage(self.scaled_frame.data, width, height,
                       self.scaled_frame.strides[0], QImage.Format.Format_RGB888)
        painter.drawImage(self.frame_offset_x, self.frame_offset_y, qimage)
        
        # Draw tracked people (only those in stores)
        if self.aws_enabled and hasattr(self, 'tracked_people'):
            for person_id, person in self.tracked_people.items():
                if person['current_store'] is not None:  # Only draw if person is in a store
                    x, y, w, h = person['bbox']
                    current_store = person['current_store']
                    store_name = self.stores[current_store]['name']
                    
                    # Scale coordinates to widget size
                    scaled_x = int(x * self.scale_factor + self.frame_offset_x)
                    scaled_y = int(y * self.scale_factor + self.frame_offset_y)
                    scaled_w = int(w * self.scale_factor)
                    scaled_h = int(h * self.scale_factor)
                    
                    # Draw bounding box with thinner line
                    painter.setPen(QPen(QColor(255, 0, 0), 1))
                    painter.drawRect(scaled_x, scaled_y, scaled_w, scaled_h)
                    
                    # Draw small ID and store info
                    label = f"{person_id}|{store_name[:5]}"  # Truncate store name to 5 chars
                    painter.setFont(QFont("Arial", 7))  # Smaller font size
                    painter.setPen(QColor(255, 255, 255))
                    # Draw text background
                    text_rect = painter.fontMetrics().boundingRect(label)
                    text_rect.moveTop(scaled_y - text_rect.height())
                    text_rect.moveLeft(scaled_x)
                    text_rect.adjust(-1, -1, 1, 1)  # Smaller padding
                    painter.fillRect(text_rect, QColor(0, 0, 0, 180))
                    # Draw text
                    painter.drawText(scaled_x, scaled_y - 2, label)  # Reduced vertical offset
        
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

    def export_video(self, output_path):
        """Export the processed video with drawings"""
        if not self.video_capture or self.current_frame is None:
            self.status_message.emit("Error: No video loaded")
            return False
        
        try:
            # Get video properties
            frame_width = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.video_capture.get(cv2.CAP_PROP_FPS)
            total_frames = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps  # Duration in seconds
            
            print(f"Input video properties:")
            print(f"- Resolution: {frame_width}x{frame_height}")
            print(f"- FPS: {fps}")
            print(f"- Total frames: {total_frames}")
            print(f"- Duration: {duration:.2f} seconds")
            
            # Store current position to restore later
            current_position = self.video_capture.get(cv2.CAP_PROP_POS_FRAMES)
            
            # Create video writer with H.264 codec
            fourcc = cv2.VideoWriter_fourcc(*'avc1')  # Use H.264 codec
            self.video_writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
            
            if not self.video_writer.isOpened():
                self.status_message.emit("Error: Could not create output video file")
                return False
            
            # Reset video capture to beginning and ensure we're at frame 0
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, _ = self.video_capture.read()  # Read first frame to ensure we're at start
            if not ret:
                raise Exception("Could not read first frame")
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset again to be sure
            
            self.is_exporting = True
            self.export_progress = 0
            self.frame_number = 0  # Reset frame counter
            
            # Process each frame
            frame_count = 0
            last_frame_time = 0  # For timing verification
            
            while frame_count < total_frames:
                # Get current frame timestamp
                current_time = self.video_capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0  # Convert to seconds
                
                ret, frame = self.video_capture.read()
                if not ret:
                    print(f"Warning: Could not read frame {frame_count} at time {current_time:.2f}s")
                    break
                
                # Verify frame timing
                if frame_count > 0:
                    expected_time = frame_count / fps
                    if abs(current_time - expected_time) > 0.1:  # More than 0.1s difference
                        print(f"Warning: Frame timing mismatch at frame {frame_count}")
                        print(f"Expected time: {expected_time:.2f}s, Actual time: {current_time:.2f}s")
                
                try:
                    # Create a copy for drawing
                    frame_draw = frame.copy()
                    
                    # Apply perspective transformation for stores
                    if self.stores:
                        try:
                            # Transform store polygons to video coordinates
                            for store_id, store in self.stores.items():
                                if "polygon" in store and len(store["polygon"]) > 2:
                                    perspective_matrix = self.store_perspective_matrices.get(store_id)
                                    if perspective_matrix is None:
                                        continue
                                    
                                    points = np.array(store["polygon"], dtype=np.float32).reshape(-1, 1, 2)
                                    if perspective_matrix.shape != (3, 3):
                                        continue
                                    
                                    transformed = cv2.perspectiveTransform(points, perspective_matrix)
                                    store["video_polygon"] = [(int(p[0][0]), int(p[0][1])) for p in transformed]
                        except Exception as e:
                            print(f"Error in perspective transformation: {str(e)}")
                    
                    # Draw store polygons
                    if self.stores:
                        for store_id, store in self.stores.items():
                            if "video_polygon" in store and len(store["video_polygon"]) > 2:
                                try:
                                    points = np.array(store["video_polygon"], np.int32).reshape((-1, 1, 2))
                                    
                                    # Draw filled semi-transparent polygon
                                    overlay = frame_draw.copy()
                                    cv2.fillPoly(overlay, [points], (0, 255, 0))
                                    cv2.addWeighted(overlay, 0.3, frame_draw, 0.7, 0, frame_draw)
                                    
                                    # Draw polygon outline
                                    cv2.polylines(frame_draw, [points], True, (0, 255, 0), 1)
                                    
                                    # Draw store name
                                    if "name" in store:
                                        centroid_x = int(np.mean([p[0] for p in store["video_polygon"]]))
                                        centroid_y = int(np.mean([p[1] for p in store["video_polygon"]]))
                                        
                                        text = store["name"]
                                        font = cv2.FONT_HERSHEY_SIMPLEX
                                        font_scale = 0.5
                                        thickness = 1
                                        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
                                        
                                        cv2.rectangle(frame_draw, 
                                                    (centroid_x - text_width//2 - 2, centroid_y - text_height//2 - 2),
                                                    (centroid_x + text_width//2 + 2, centroid_y + text_height//2 + 2),
                                                    (0, 0, 0), -1)
                                        
                                        cv2.putText(frame_draw, text,
                                                  (centroid_x - text_width//2, centroid_y + text_height//2),
                                                  font, font_scale, (255, 255, 255), thickness)
                                except Exception as e:
                                    print(f"Error drawing store {store_id}: {str(e)}")
                    
                    # Process person detection if AWS is enabled
                    if self.aws_enabled:
                        try:
                            detected_people = self.detect_people(frame)
                            tracked_people = self.person_tracker.update(detected_people, self.stores, self.frame_number)
                            
                            # Draw tracked people (only those in stores)
                            for person_id, person in tracked_people.items():
                                if person['current_store'] is not None:
                                    x, y, w, h = person['bbox']
                                    current_store = person['current_store']
                                    store_name = self.stores[current_store]['name']
                                    
                                    cv2.rectangle(frame_draw, (x, y), (x + w, y + h), (0, 0, 255), 1)
                                    
                                    label = f"{person_id}|{store_name[:5]}"
                                    font = cv2.FONT_HERSHEY_SIMPLEX
                                    font_scale = 0.4
                                    thickness = 1
                                    (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
                                    
                                    cv2.rectangle(frame_draw, 
                                                (x, y - text_height - 2),
                                                (x + text_width, y),
                                                (0, 0, 0), -1)
                                    
                                    cv2.putText(frame_draw, label,
                                              (x, y - 2),
                                              font, font_scale, (255, 255, 255), thickness)
                            
                            # Draw store entry notification
                            if self.last_store_entry and self.store_entry_display_time > 0:
                                text = f"Person {self.last_store_entry['person_id']} entered {self.last_store_entry['store_name']}"
                                font = cv2.FONT_HERSHEY_SIMPLEX
                                font_scale = 0.6
                                thickness = 1
                                (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
                                
                                cv2.rectangle(frame_draw,
                                            (frame_draw.shape[1]//2 - text_width//2 - 5, 10),
                                            (frame_draw.shape[1]//2 + text_width//2 + 5, 10 + text_height + 5),
                                            (0, 0, 0), -1)
                                
                                cv2.putText(frame_draw, text,
                                          (frame_draw.shape[1]//2 - text_width//2, 10 + text_height),
                                          font, font_scale, (255, 255, 255), thickness)
                                
                                self.store_entry_display_time -= 1
                        except Exception as e:
                            print(f"Error in person detection/tracking: {str(e)}")
                    
                    # Write frame
                    self.video_writer.write(frame_draw)
                    
                    # Update progress
                    frame_count += 1
                    progress = (frame_count / total_frames) * 100
                    self.status_message.emit(f"Exporting video: {progress:.1f}%")
                    
                    self.frame_number += 1
                    last_frame_time = current_time
                    
                except Exception as e:
                    print(f"Error processing frame {frame_count}: {str(e)}")
                    continue
            
            # Clean up
            self.video_writer.release()
            self.video_writer = None
            self.is_exporting = False
            self.export_progress = 0
            
            # Verify output video
            output_cap = cv2.VideoCapture(output_path)
            if output_cap.isOpened():
                output_frames = int(output_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                output_fps = output_cap.get(cv2.CAP_PROP_FPS)
                output_duration = output_frames / output_fps
                output_cap.release()
                
                print(f"Output video properties:")
                print(f"- Resolution: {frame_width}x{frame_height}")
                print(f"- FPS: {output_fps}")
                print(f"- Total frames: {output_frames}")
                print(f"- Duration: {output_duration:.2f} seconds")
                
                # Verify frame count and duration
                if output_frames != total_frames:
                    print(f"Error: Frame count mismatch - Input: {total_frames}, Output: {output_frames}")
                    # Delete the incorrect output file
                    os.remove(output_path)
                    raise Exception("Frame count mismatch in output video")
                
                if abs(output_duration - duration) > 0.1:
                    print(f"Error: Duration mismatch - Input: {duration:.2f}s, Output: {output_duration:.2f}s")
                    # Delete the incorrect output file
                    os.remove(output_path)
                    raise Exception("Duration mismatch in output video")
            else:
                print("Warning: Could not verify output video properties")
                raise Exception("Could not verify output video")
            
            # Restore video capture position
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, current_position)
            self.update_frame()
            
            self.status_message.emit(f"Video exported successfully to: {output_path}")
            return True
            
        except Exception as e:
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None
            self.is_exporting = False
            self.export_progress = 0
            # Restore video capture position on error
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, current_position)
            self.status_message.emit(f"Error exporting video: {str(e)}")
            print(f"Export error details: {str(e)}")
            return False

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
        
        # Add tooltips
        self.select_tool_btn.setToolTip("Select and move cameras or stores")
        self.camera_tool_btn.setToolTip("Click to place camera, drag to set orientation")
        self.store_tool_btn.setToolTip("Click to create store polygon, double-click to complete")
        self.calibrate_btn.setToolTip("Calibrate camera view with blueprint")
        self.test_mode_btn.setToolTip("Toggle test mode to see store boundaries in video")
        
        # Set fixed height for buttons
        button_height = 30
        for btn in [self.select_tool_btn, self.camera_tool_btn, 
                   self.store_tool_btn, self.calibrate_btn, self.test_mode_btn]:
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
        
        export_video_action = QAction("Export Video", self)
        export_video_action.setToolTip("Export processed video with store boundaries")
        export_video_action.triggered.connect(self.export_video)
        
        aws_action = QAction("Enable AWS", self)
        aws_action.setToolTip("Enable AWS Rekognition for person detection")
        aws_action.triggered.connect(self.enable_aws)
        
        export_log_action = QAction("Export Movement Log", self)
        export_log_action.setToolTip("Export person movement log")
        export_log_action.triggered.connect(self.export_movement_log)
        
        toolbar.addAction(load_blueprint_action)
        toolbar.addAction(load_video_action)
        toolbar.addAction(export_action)
        toolbar.addAction(load_mapping_action)
        toolbar.addAction(export_video_action)
        toolbar.addAction(aws_action)
        toolbar.addAction(export_log_action)
        
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
        
        self.blueprint_view.calibration_points_selected.connect(self.on_blueprint_calibration_points)
        self.video_preview.calibration_points_selected.connect(self.on_video_calibration_points)
        
        # Create processor
        self.processor = BlueprintProcessor()
        
        # Add calibration state tracking
        self.calibration_blueprint_points = None
        self.current_calibration_store = None  # Track which store is being calibrated
    
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

    def export_video(self):
        """Handle video export request"""
        if not self.video_preview.video_capture:
            QMessageBox.warning(self, "Error", "Please load a video first")
            return
        
        if not self.video_preview.stores:
            QMessageBox.warning(self, "Error", "Please define and map stores first")
            return
        
        # Get output file path
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Video", "", "Video Files (*.mp4)"
        )
        
        if file_path:
            # Ensure .mp4 extension
            if not file_path.lower().endswith('.mp4'):
                file_path += '.mp4'
            
            # Show progress dialog
            progress = QProgressDialog("Exporting video...", "Cancel", 0, 100, self)
            progress.setWindowModality(Qt.WindowModality.WindowModal)
            progress.setAutoClose(True)
            progress.setAutoReset(True)
            
            # Connect status message to progress dialog
            def update_progress(message):
                if "Exporting video:" in message:
                    try:
                        percent = float(message.split(":")[1].strip().rstrip("%"))
                        progress.setValue(int(percent))
                    except:
                        pass
            
            self.video_preview.status_message.connect(update_progress)
            
            # Start export
            if self.video_preview.export_video(file_path):
                QMessageBox.information(self, "Success", 
                    f"Video exported successfully to:\n{file_path}")
            else:
                QMessageBox.critical(self, "Error", "Failed to export video")
            
            # Disconnect status message handler
            self.video_preview.status_message.disconnect(update_progress)

    def enable_aws(self):
        """Enable AWS Rekognition with default credentials"""
        if self.video_preview.enable_aws_rekognition(aws_region='ap-south-1'):
            QMessageBox.information(self, "Success", 
                "AWS Rekognition enabled successfully.\n\n"
                "Person detection is now active in the video preview.")
        else:
            QMessageBox.warning(self, "Error", 
                "Failed to enable AWS Rekognition.\n\n"
                "Please check your AWS credentials configuration.")

    def export_movement_log(self):
        """Export the movement log of tracked people"""
        if not hasattr(self.video_preview, 'person_tracker'):
            QMessageBox.warning(self, "Error", "No movement data available")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Movement Log", "", "CSV Files (*.csv)"
        )
        
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    f.write("Person ID,Store Name,Entry Time,Frame Number\n")
                    for person_id, person in self.video_preview.person_tracker.tracked_people.items():
                        for entry in person['history']:
                            f.write(f"{person_id},{entry['store_name']},{entry['entry_time']},{entry['frame']}\n")
                
                QMessageBox.information(self, "Success", 
                    f"Movement log exported successfully to:\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to export movement log: {str(e)}")

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main() 