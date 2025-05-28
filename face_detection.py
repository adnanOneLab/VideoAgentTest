import boto3
import cv2
import time
import os
import numpy as np
from datetime import datetime
import json
from sklearn.metrics.pairwise import cosine_similarity
import face_recognition
import uuid
from collections import defaultdict
import matplotlib.pyplot as plt
import shutil
import zipfile
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
from queue import Queue
import csv
from PIL import Image, ImageTk

class VideoAnalyzer:
    def __init__(self):
        # AWS Rekognition client
        self.rekognition = boto3.client('rekognition', region_name='ap-south-1')
        self.collection_id = 'my-face-collection'
        
        # Video quality settings - focus on larger size for better face visibility
        self.video_settings = {
            'target_width': 2560,    # Increased target width for larger display
            'target_height': 1440,   # Increased target height for larger display
            'min_width': 1920,       # Increased minimum width
            'min_height': 1080,      # Increased minimum height
            'frame_skip': 10,         # Keep frame skip at 3 for smooth tracking
            'quality_factor': 95,    # Keep high quality
            'upscale_factor': 2.0    # Increased upscale factor for small faces
        }
        
        # Recognition settings with improved thresholds
        self.recognition_settings = {
            'default_threshold': 75,      # Increased threshold for better accuracy
            'min_confidence': 85,         # Increased minimum confidence
            'max_faces': 10,             # Keep max faces for better coverage
            'quality_threshold': 70,      # Increased quality threshold
            'tracking_cache_time': 30,    # Number of frames to cache recognized faces
            'min_tracking_confidence': 80  # Increased minimum confidence for tracking
        }
        
        # Face tracking cache
        self.tracking_cache = {}  # Format: {face_id: {'name': name, 'confidence': conf, 'frames': count}}
        
        # Enhanced preprocessing settings
        self.preprocessing_settings = {
            'clahe_clip_limit': 3.0,      # Increased contrast enhancement
            'clahe_grid_size': (8, 8),
            'bilateral_d': 9,             # Bilateral filter parameters
            'bilateral_sigma_color': 75,
            'bilateral_sigma_space': 75,
            'sharpen_kernel': np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]]),
            'contrast_alpha': 1.4,        # Increased contrast
            'contrast_beta': 15           # Increased brightness
        }
        
        # Create directories
        self.unrecognized_dir = "unrecognized_faces"
        self.indexed_dir = "indexed_faces"
        os.makedirs(self.unrecognized_dir, exist_ok=True)
        os.makedirs(self.indexed_dir, exist_ok=True)
        
        # Load or create face mapping
        self.face_mapping_file = "face_mapping.json"
        self.face_mapping = self.load_face_mapping()
        
        # Add UUIDs to existing entries if not present
        self.update_mapping_with_uuids()
        
        # Sync with AWS collection
        self.sync_collection()
        
        # Visualization settings
        self.face_color = (0, 255, 0)
        self.unrecognized_color = (0, 165, 255)
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.7
        self.font_thickness = 2
        self.box_thickness = 2
        
        # Processing control
        self.frame_skip = 5
        self.frame_counter = 0
        self.last_results = {'faces': []}
        self.tracked_faces = {}
        self.tracking_counter = 0
        self.total_faces_detected = 0
        self.recognized_faces = set()
        
        # Face tracking settings
        self.tracking_iou_threshold = 0.3
        self.max_tracking_frames = 30
        
        # Quality assessment settings
        self.quality_metrics = {
            'blur_threshold': 100,
            'min_face_size': 100,
            'max_face_size': 1000,
            'min_confidence': 90,
            'max_angle': 30,
            'min_lighting': 40,
            'max_lighting': 200
        }
        
        # Face comparison settings
        self.face_similarity_threshold = 0.6
        self.known_face_encodings = {}
        
        # Load historical data
        self.load_historical_data()
        
        # Enhanced tracking settings
        self.tracking_settings = {
            'max_tracking_frames': 30,     # Maximum frames to track a face
            'min_tracking_confidence': 75,  # Minimum confidence to start tracking
            'iou_threshold': 0.3,          # IOU threshold for matching faces
            'max_movement': 0.2,           # Maximum allowed movement between frames (as fraction of frame size)
            'prediction_frames': 3,        # Number of frames to predict movement
            'cache_duration': 30,          # Base cache duration
            'confidence_boost': 1.2,       # Confidence multiplier for tracked faces
            'smooth_factor': 0.7,          # Smoothing factor for position updates (0-1)
            'velocity_decay': 0.95,        # Velocity decay factor per frame
            'min_velocity': 0.001         # Minimum velocity threshold
        }
        
        # Enhanced tracking data structures
        self.tracked_faces = {}  # Format: {track_id: {'box': (x,y,w,h), 'name': name, 'confidence': conf, 'frames': count, 'velocity': (dx,dy), 'last_pos': (x,y)}}
        self.tracking_counter = 0
        self.last_frame_size = None

    def load_face_mapping(self):
        """Load existing face mappings from JSON file"""
        if os.path.exists(self.face_mapping_file):
            try:
                with open(self.face_mapping_file, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                print("⚠️ Error reading face mapping file, creating new one")
                return {}
        return {}

    def save_face_mapping(self):
        """Save face mappings to JSON file"""
        with open(self.face_mapping_file, 'w') as f:
            json.dump(self.face_mapping, f, indent=2)

    def update_mapping_with_uuids(self):
        """Add UUIDs to existing entries that don't have them"""
        updated = False
        for person, data in self.face_mapping.items():
            if 'uuid' not in data:
                data['uuid'] = str(uuid.uuid4())
                updated = True
        if updated:
            self.save_face_mapping()
            print("✅ Updated face mapping with UUIDs")

    def index_face(self, image_path, person_name):
        """Index a face image into the AWS Rekognition collection"""
        try:
            # Read the image
            with open(image_path, 'rb') as image:
                image_bytes = image.read()
            
            # Check if the image contains a face
            response = self.rekognition.detect_faces(
                Image={'Bytes': image_bytes},
                Attributes=['ALL']
            )
            
            if not response['FaceDetails']:
                print(f"❌ No face detected in {image_path}")
                return False
            
            # Clean up person name (remove any UUID suffix if present)
            base_name = person_name.split('_')[0] if '_' in person_name else person_name
            
            # Generate a unique ID for this person if new
            if base_name not in self.face_mapping:
                self.face_mapping[base_name] = {
                    'uuid': str(uuid.uuid4()),
                    'face_ids': [],
                    'images': [],
                    'created_at': datetime.now().isoformat()
                }
            
            # Use the person's name as the external ID
            external_id = base_name
            
            # Index the face
            response = self.rekognition.index_faces(
                CollectionId=self.collection_id,
                Image={'Bytes': image_bytes},
                ExternalImageId=external_id,
                DetectionAttributes=['ALL']
            )
            
            if response['FaceRecords']:
                face_id = response['FaceRecords'][0]['Face']['FaceId']
                
                # Only add face_id if not already present
                if face_id not in self.face_mapping[base_name]['face_ids']:
                    self.face_mapping[base_name]['face_ids'].append(face_id)
                
                # Move the image to indexed directory with person name in filename
                indexed_filename = f"{base_name}_{len(self.face_mapping[base_name]['images'])}.jpg"
                indexed_path = os.path.join(self.indexed_dir, indexed_filename)
                
                # Only move and add image if not already present
                if not any(img == indexed_path for img in self.face_mapping[base_name]['images']):
                    os.rename(image_path, indexed_path)
                    self.face_mapping[base_name]['images'].append(indexed_path)
                
                self.face_mapping[base_name]['last_updated'] = datetime.now().isoformat()
                
                self.save_face_mapping()
                print(f"✅ Successfully indexed face for {base_name}")
                return True
            else:
                print(f"❌ Failed to index face in {image_path}")
                return False
                
        except Exception as e:
            print(f"❌ Error indexing face: {e}")
            return False

    def check_existing_face(self, image_path, threshold=80):
        """Check if a face might already be in the collection"""
        try:
            # Read the image
            with open(image_path, 'rb') as image:
                image_bytes = image.read()
            
            # Search for the face in the collection
            response = self.rekognition.search_faces_by_image(
                CollectionId=self.collection_id,
                Image={'Bytes': image_bytes},
                MaxFaces=5,  # Get top 5 matches
                FaceMatchThreshold=threshold
            )
            
            if response.get('FaceMatches'):
                matches = []
                for match in response['FaceMatches']:
                    similarity = match['Similarity']
                    external_id = match['Face']['ExternalImageId']
                    matches.append({
                        'name': external_id,
                        'similarity': similarity
                    })
                return matches
            return None
            
        except Exception as e:
            print(f"⚠️ Error checking existing face: {e}")
            return None

    def preprocess_frame(self, frame):
        """Enhanced frame preprocessing for better face detection"""
        if frame is None or frame.size == 0:
            return frame, None

        # Convert to grayscale for processing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply enhanced CLAHE for better contrast
        clahe = cv2.createCLAHE(
            clipLimit=self.preprocessing_settings['clahe_clip_limit'],
            tileGridSize=self.preprocessing_settings['clahe_grid_size']
        )
        gray = clahe.apply(gray)
        
        # Apply bilateral filter with optimized parameters
        gray = cv2.bilateralFilter(
            gray,
            self.preprocessing_settings['bilateral_d'],
            self.preprocessing_settings['bilateral_sigma_color'],
            self.preprocessing_settings['bilateral_sigma_space']
        )
        
        # Enhanced contrast and brightness
        alpha = self.preprocessing_settings['contrast_alpha']
        beta = self.preprocessing_settings['contrast_beta']
        gray = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)
        
        # Apply sharpening
        gray = cv2.filter2D(gray, -1, self.preprocessing_settings['sharpen_kernel'])
        
        # Convert back to color for display
        enhanced = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        
        return enhanced, gray

    def assess_face_quality(self, face_image):
        """Assess the quality of a face image with detailed metrics
        
        Args:
            face_image: numpy array of the face image
            
        Returns:
            tuple: (quality_score, issues_list) where quality_score is 0-100 and issues_list contains quality problems
        """
        try:
            # Convert to grayscale if needed
            if len(face_image.shape) == 3:
                gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = face_image
                
            issues = []
            metrics = {}
            
            # 1. Calculate Laplacian variance (measure of sharpness)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            metrics['sharpness'] = min(100, laplacian_var)
            if laplacian_var < 50:
                issues.append("Image is blurry")
            elif laplacian_var > 500:
                issues.append("Image is too sharp/noisy")
                
            # 2. Calculate face size score
            height, width = gray.shape
            face_size = height * width
            metrics['size'] = min(100, (face_size / 10000) * 100)  # Normalize to 100x100
            if face_size < 5000:  # Less than 70x70 pixels
                issues.append("Face is too small")
            elif face_size > 100000:  # More than 300x300 pixels
                issues.append("Face is too large")
                
            # 3. Calculate brightness and contrast
            brightness = np.mean(gray)
            contrast = np.std(gray)
            metrics['brightness'] = min(100, (brightness / 255) * 100)
            metrics['contrast'] = min(100, (contrast / 128) * 100)
            
            if brightness < 40:
                issues.append("Image is too dark")
            elif brightness > 200:
                issues.append("Image is too bright")
            if contrast < 30:
                issues.append("Low contrast")
            elif contrast > 150:
                issues.append("Too much contrast")
                
            # 4. Check face angle (if we have facial landmarks)
            try:
                # Convert to RGB for face_recognition
                rgb_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
                face_landmarks = face_recognition.face_landmarks(rgb_image)
                if face_landmarks:
                    # Get nose bridge points
                    nose_bridge = face_landmarks[0]['nose_bridge']
                    if len(nose_bridge) >= 2:
                        # Calculate angle from vertical
                        dx = nose_bridge[-1][0] - nose_bridge[0][0]
                        dy = nose_bridge[-1][1] - nose_bridge[0][1]
                        angle = abs(np.degrees(np.arctan2(dx, dy)))
                        metrics['angle'] = 100 - min(100, angle * 2)  # Convert to score
                        if angle > 15:
                            issues.append("Face is tilted")
                else:
                    metrics['angle'] = 0
                    issues.append("Could not detect facial landmarks")
            except Exception as e:
                metrics['angle'] = 0
                issues.append("Could not analyze face angle")
                
            # 5. Check for occlusions (basic check using face_recognition)
            try:
                face_encodings = face_recognition.face_encodings(rgb_image)
                if not face_encodings:
                    issues.append("Face may be partially occluded")
                    metrics['occlusion'] = 0
                else:
                    metrics['occlusion'] = 100
            except:
                metrics['occlusion'] = 0
                
            # Calculate final quality score with weights
            weights = {
                'sharpness': 0.25,
                'size': 0.20,
                'brightness': 0.15,
                'contrast': 0.15,
                'angle': 0.15,
                'occlusion': 0.10
            }
            
            quality_score = sum(metrics[k] * weights[k] for k in weights.keys())
            
            # Add guidance if there are issues
            if issues:
                guidance = []
                if "Image is blurry" in issues:
                    guidance.append("Hold the camera steady")
                if "Face is too small" in issues:
                    guidance.append("Move closer to the camera")
                if "Face is too large" in issues:
                    guidance.append("Move further from the camera")
                if "Image is too dark" in issues:
                    guidance.append("Move to a better lit area")
                if "Image is too bright" in issues:
                    guidance.append("Reduce lighting or move to a darker area")
                if "Face is tilted" in issues:
                    guidance.append("Keep your head straight")
                if "Face may be partially occluded" in issues:
                    guidance.append("Remove any obstructions from your face")
                    
                issues.extend([f"Tip: {tip}" for tip in guidance])
                
            return quality_score, issues, metrics
                
        except Exception as e:
            print(f"Error assessing face quality: {e}")
            return 0, ["Error analyzing face quality"], {}

    def get_face_encoding(self, image):
        """Get face encoding from image (file path or numpy array)"""
        try:
            # If image is a path, load it
            if isinstance(image, str):
                if not os.path.exists(image):
                    return None
                image = cv2.imread(image)
                if image is None:
                    return None
                # Convert BGR to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Get face locations
            face_locations = face_recognition.face_locations(image)
            if not face_locations:
                return None
                
            # Get face encodings
            face_encodings = face_recognition.face_encodings(image, face_locations)
            if not face_encodings:
                return None
                
            return face_encodings[0]  # Return first face encoding
            
        except Exception as e:
            print(f"Error getting face encoding: {e}")
            return None

    def compare_faces(self, encoding1, encoding2):
        """Compare two face encodings and return similarity score"""
        try:
            # Calculate cosine similarity
            similarity = 1 - distance.cosine(encoding1, encoding2)
            return similarity
        except Exception as e:
            print(f"Error comparing faces: {e}")
            return 0.0

    def process_video(self, video_path, progress_callback=None):
        """Process a video file with focus on larger display size"""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        # Get original video properties
        orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Always upscale to target size while maintaining aspect ratio
        aspect_ratio = orig_width / orig_height
        if aspect_ratio > 1:
            # Landscape video
            target_width = self.video_settings['target_width']
            target_height = int(target_width / aspect_ratio)
        else:
            # Portrait video
            target_height = self.video_settings['target_height']
            target_width = int(target_height * aspect_ratio)

        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = 0

        # Reset tracking and performance metrics
        self.tracked_faces = {}
        self.tracking_counter = 0
        self.performance_metrics = {
            'total_frames_processed': 0,
            'total_faces_detected': 0,
            'total_faces_recognized': 0,
            'recognition_times': [],
            'confidence_scores': []
        }
        self.recognized_faces = set()

        # Create output directory
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)

        # Create output file path
        output_filename = os.path.basename(video_path).split('.')[0] + '_analyzed.mp4'
        output_path = os.path.join(output_dir, output_filename)

        # Define video writer with high quality settings
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (target_width, target_height))

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Always resize to target size using INTER_CUBIC for better quality
                frame = cv2.resize(frame, (target_width, target_height), 
                                 interpolation=cv2.INTER_CUBIC)

                if frame_count % self.video_settings['frame_skip'] == 0:
                    self.analyze_frame(frame)

                self.update_tracked_faces(frame)
                display_frame = frame.copy()
                display_frame = self.draw_detections(display_frame)

                if progress_callback:
                    progress_callback(display_frame, frame_count, total_frames)

                out.write(display_frame)
                frame_count += 1
                self.performance_metrics['total_frames_processed'] = frame_count

        finally:
            cap.release()
            out.release()
            self.save_historical_data()

        return {
            'total_frames': total_frames,
            'processed_frames': frame_count,
            'faces_detected': self.performance_metrics['total_faces_detected'],
            'faces_recognized': len(self.recognized_faces),
            'resolution': f"{target_width}x{target_height}"
        }

    def enhance_frame(self, frame):
        """Simplified frame processing - just return the frame as is"""
        return frame  # Return frame without any enhancement

    def analyze_frame(self, frame):
        """Analyze a single frame with optimized recognition and caching"""
        start_time = time.time()

        if frame is None or frame.size == 0:
            return {'faces': []}

        # Enhanced preprocessing
        enhanced_frame = self.enhance_frame(frame)

        # Convert frame to JPEG bytes with higher quality
        try:
            _, img_encoded = cv2.imencode('.jpg', enhanced_frame, 
                                        [cv2.IMWRITE_JPEG_QUALITY, self.video_settings['quality_factor']])
            image_bytes = img_encoded.tobytes()
        except Exception as e:
            print(f"⚠️ Frame encoding error: {e}")
            return {'faces': []}

        results = {'faces': []}

        try:
            # Face detection with improved parameters
            detect_response = self.rekognition.detect_faces(
                Image={'Bytes': image_bytes},
                Attributes=['ALL']
            )

            self.performance_metrics['total_faces_detected'] += len(detect_response.get('FaceDetails', []))

            for face_detail in detect_response.get('FaceDetails', []):
                face_box = face_detail['BoundingBox']
                confidence = face_detail['Confidence']

                # Skip if confidence is too low
                if confidence < self.recognition_settings['min_confidence']:
                    continue

                height, width = frame.shape[:2]
                left = int(face_box['Left'] * width)
                top = int(face_box['Top'] * height)
                right = left + int(face_box['Width'] * width)
                bottom = top + int(face_box['Height'] * height)

                # Add padding to face region (30% on each side for better context)
                padding_x = int((right - left) * 0.3)
                padding_y = int((bottom - top) * 0.3)
                
                # Ensure the crop is within frame bounds with padding
                left = max(0, left - padding_x)
                top = max(0, top - padding_y)
                right = min(width, right + padding_x)
                bottom = min(height, bottom + padding_y)

                if right <= left or bottom <= top:
                    continue

                # Extract face region with padding
                face_region = frame[top:bottom, left:right]
                if face_region.size == 0 or face_region.shape[0] < 20 or face_region.shape[1] < 20:
                    continue

                # Ensure minimum face size and upscale if needed
                min_face_size = 200  # increased minimum size
                if face_region.shape[0] < min_face_size or face_region.shape[1] < min_face_size:
                    # Upscale small faces
                    scale = self.video_settings['upscale_factor']
                    face_region = cv2.resize(face_region, None, 
                                           fx=scale, fy=scale, 
                                           interpolation=cv2.INTER_LANCZOS4)

                # Additional preprocessing for face region
                face_region = self.enhance_frame(face_region)  # Apply enhancement to face region
                
                # Convert to RGB for AWS
                face_region_rgb = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)

                # Encode face region with high quality
                try:
                    _, face_encoded = cv2.imencode('.jpg', face_region_rgb, 
                                                 [cv2.IMWRITE_JPEG_QUALITY, self.video_settings['quality_factor']])
                    face_bytes = face_encoded.tobytes()
                except Exception as e:
                    print(f"⚠️ Face region encoding error: {e}")
                    continue

                # Perform recognition with AWS Rekognition
                try:
                    recognize_response = self.rekognition.search_faces_by_image(
                        CollectionId=self.collection_id,
                        Image={'Bytes': face_bytes},
                        MaxFaces=self.recognition_settings['max_faces'],
                        FaceMatchThreshold=self.recognition_settings['default_threshold']
                    )

                    if recognize_response.get('FaceMatches'):
                        match = recognize_response['FaceMatches'][0]
                        if match['Similarity'] >= self.recognition_settings['min_tracking_confidence']:
                            # Add to tracking cache
                            face_id = face_detail.get('FaceId')
                            if face_id:
                                self.tracking_cache[face_id] = {
                                    'name': match['Face']['ExternalImageId'],
                                    'confidence': match['Similarity'],
                                    'frames': self.recognition_settings['tracking_cache_time']
                                }
                        
                        results['faces'].append({
                            'Face': {
                                'BoundingBox': face_box,
                                'ExternalImageId': match['Face']['ExternalImageId'],
                                'Confidence': confidence,
                                'FaceId': face_detail.get('FaceId')
                            },
                            'Similarity': match['Similarity'],
                            'Recognized': True
                        })
                    else:
                        results['faces'].append({
                            'Face': {
                                'BoundingBox': face_box,
                                'ExternalImageId': 'Unknown',
                                'Confidence': confidence,
                                'FaceId': face_detail.get('FaceId')
                            },
                            'Similarity': 0,
                            'Recognized': False
                        })

                except Exception as e:
                    print(f"⚠️ Recognition error for detected face: {e}")
                    results['faces'].append({
                        'Face': {
                            'BoundingBox': face_box,
                            'ExternalImageId': 'Unknown',
                            'Confidence': confidence,
                            'FaceId': face_detail.get('FaceId')
                        },
                        'Similarity': 0,
                        'Recognized': False
                    })

        except Exception as e:
            print(f"⚠️ Face detection error: {e}")

        processing_time = time.time() - start_time
        self.update_performance_metrics(results, processing_time)
        self.last_results = results

        return results

    def update_tracked_faces(self, frame):
        """Enhanced face tracking with smooth movement and prediction"""
        if frame is None or frame.size == 0:
            return

        height, width = frame.shape[:2]
        self.last_frame_size = (width, height)
        
        # Update tracking cache counters
        for face_id in list(self.tracking_cache.keys()):
            self.tracking_cache[face_id]['frames'] -= 1
            if self.tracking_cache[face_id]['frames'] <= 0:
                del self.tracking_cache[face_id]
        
        # Prepare current frame's detection results
        current_detections = []
        if self.last_results.get('faces'):
            for face_match in self.last_results['faces']:
                box = face_match['Face']['BoundingBox']
                left = int(box['Left'] * width)
                top = int(box['Top'] * height)
                right = left + int(box['Width'] * width)
                bottom = top + int(box['Height'] * height)
                
                # Ensure box is within frame bounds
                left = max(0, min(width - 1, left))
                top = max(0, min(height - 1, top))
                right = max(left + 1, min(width, right))
                bottom = max(top + 1, min(height, bottom))
                
                detection = {
                    'box': (left, top, right, bottom),
                    'name': face_match['Face']['ExternalImageId'],
                    'confidence': face_match.get('Similarity', face_match['Face'].get('Confidence', 0)),
                    'recognized': face_match.get('Recognized', False),
                    'face_id': face_match.get('Face', {}).get('FaceId')
                }
                current_detections.append(detection)
        
        # Update existing tracks with predictions
        updated_tracks = {}
        matched_detection_indices = set()
        
        # First pass: Update existing tracks with predictions
        for track_id, track_data in list(self.tracked_faces.items()):
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
            predicted_box = self.predict_face_position(track_data, frame.shape)
            
            if predicted_box:
                # Find best matching detection
                best_iou = 0
                best_match_index = -1
                best_match_box = None
                
                for i, detection in enumerate(current_detections):
                    if i in matched_detection_indices:
                        continue
                    
                    # Calculate IOU with predicted position
                    iou = self.calculate_iou(predicted_box, detection['box'])
                    
                    if iou > self.tracking_settings['iou_threshold'] and iou > best_iou:
                        best_iou = iou
                        best_match_index = i
                        best_match_box = detection['box']
                
                if best_match_index != -1:
                    # Update track with matched detection
                    matched_detection_indices.add(best_match_index)
                    detection = current_detections[best_match_index]
                    
                    # Calculate new velocity based on actual movement
                    new_velocity = self.calculate_velocity(track_data['box'], detection['box'], frame.shape)
                    
                    # Smooth the velocity update
                    if 'velocity' in track_data:
                        old_dx, old_dy = track_data['velocity']
                        new_dx, new_dy = new_velocity
                        smoothed_dx = old_dx * (1 - self.tracking_settings['smooth_factor']) + new_dx * self.tracking_settings['smooth_factor']
                        smoothed_dy = old_dy * (1 - self.tracking_settings['smooth_factor']) + new_dy * self.tracking_settings['smooth_factor']
                        new_velocity = (smoothed_dx, smoothed_dy)
                    
                    # Smooth the position update while ensuring bounds
                    old_box = track_data['box']
                    new_box = detection['box']
                    
                    # Calculate smoothed box with bounds checking
                    smoothed_box = (
                        max(0, min(width - (new_box[2] - new_box[0]), 
                            int(old_box[0] * (1 - self.tracking_settings['smooth_factor']) + 
                                new_box[0] * self.tracking_settings['smooth_factor']))),
                        max(0, min(height - (new_box[3] - new_box[1]), 
                            int(old_box[1] * (1 - self.tracking_settings['smooth_factor']) + 
                                new_box[1] * self.tracking_settings['smooth_factor']))),
                        min(width, max(new_box[2] - new_box[0], 
                            int(old_box[2] * (1 - self.tracking_settings['smooth_factor']) + 
                                new_box[2] * self.tracking_settings['smooth_factor']))),
                        min(height, max(new_box[3] - new_box[1], 
                            int(old_box[3] * (1 - self.tracking_settings['smooth_factor']) + 
                                new_box[3] * self.tracking_settings['smooth_factor'])))
                    )
                    
                    # Update track data with smoothed values
                    updated_tracks[track_id] = {
                        'box': smoothed_box,
                        'name': detection['name'],
                        'confidence': detection['confidence'] * self.tracking_settings['confidence_boost'],
                        'frames': self.tracking_settings['cache_duration'],
                        'velocity': new_velocity,
                        'last_pos': ((smoothed_box[0] + smoothed_box[2])/2,
                                   (smoothed_box[1] + smoothed_box[3])/2),
                        'recognized': detection['recognized']
                    }
                else:
                    # Use predicted position with decaying confidence
                    track_data['confidence'] *= 0.95
                    track_data['frames'] -= 1
                    
                    if track_data['frames'] > 0:
                        # Update position based on velocity
                        predicted_box = self.predict_face_position(track_data, frame.shape)
                        if predicted_box:
                            track_data['box'] = predicted_box
                            track_data['last_pos'] = ((predicted_box[0] + predicted_box[2])/2,
                                                    (predicted_box[1] + predicted_box[3])/2)
                        updated_tracks[track_id] = track_data
            else:
                # Remove track if prediction fails
                track_data['frames'] -= 1
                if track_data['frames'] > 0:
                    updated_tracks[track_id] = track_data
        
        # Second pass: Create new tracks for unmatched detections
        for i, detection in enumerate(current_detections):
            if i not in matched_detection_indices:
                new_id = self.tracking_counter
                self.tracking_counter += 1
                
                # Initialize new track
                updated_tracks[new_id] = {
                    'box': detection['box'],
                    'name': detection['name'],
                    'confidence': detection['confidence'],
                    'frames': self.tracking_settings['cache_duration'],
                    'velocity': (0, 0),  # Initial velocity
                    'last_pos': ((detection['box'][0] + detection['box'][2])/2,
                               (detection['box'][1] + detection['box'][3])/2),
                    'recognized': detection['recognized']
                }
        
        self.tracked_faces = updated_tracks

    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union for two bounding boxes"""
        # box format: (left, top, right, bottom) - pixel coordinates
        # box format: (left, top, right, bottom) - pixel coordinates
        x_left = max(box1[0], box2[0])
        y_top = max(box1[1], box2[1])
        x_right = min(box1[2], box2[2])
        y_bottom = min(box1[3], box2[3])


        if x_right < x_left or y_bottom < y_top:
            return 0.0


        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - intersection_area

        if union_area == 0:
             return 0.0


        if union_area == 0:
             return 0.0

        return intersection_area / union_area

    def draw_detections(self, frame):
        """Draw tracked face detections on the frame, skipping unknown faces"""
        if frame is None or frame.size == 0:
            return frame

        # height, width = frame.shape[:2] # Not needed if using pixel coordinates from tracking

        # Draw tracked faces
        for track_id, track_data in self.tracked_faces.items():
            # Skip unknown faces
            if track_data.get('name') == 'Unknown':
                continue
                
            box = track_data['box']
            left, top, right, bottom = box


            # Choose color based on recognition status
            if track_data.get('recognized', False):
                color = (0, 255, 0)  # Green for recognized faces
                label_text = f"{track_data['name']} ({track_data['confidence']:.0f}%)"
            else:
                color = (0, 165, 255)  # Orange for detected but not recognized
                label_text = f"Detecting... ({track_data['confidence']:.0f}%)"

            thickness = self.box_thickness
            cv2.rectangle(frame, (left, top), (right, bottom), color, thickness)

            # Draw name label background
            (text_width, text_height), _ = cv2.getTextSize(label_text, self.font, self.font_scale, self.font_thickness)

            # Ensure label background is within frame bounds
            label_bg_top = max(0, top - text_height - 10)
            label_bg_bottom = top
            label_bg_right = min(frame.shape[1], left + text_width + 10)

            cv2.rectangle(frame, (left, label_bg_top), (label_bg_right, label_bg_bottom), color, -1)

            # Draw name and confidence
            cv2.putText(frame, label_text,
                        (left + 5, top - 10), self.font, self.font_scale, (255, 255, 255), self.font_thickness)

        return frame

    def show_person_images(self):
        """Display all images of a specific person"""
        if not self.face_mapping:
            print("No indexed faces found!")
            return
        
        # Show list of all persons
        print("\n=== Indexed Persons ===")
        persons = sorted(self.face_mapping.keys())
        for i, person in enumerate(persons, 1):
            print(f"{i}. {person} ({len(self.face_mapping[person]['images'])} images)")
        
        while True:
            try:
                choice = input("\nSelect person number (or 'q' to quit): ").strip()
                if choice.lower() == 'q':
                    return
                
                choice = int(choice)
                if 1 <= choice <= len(persons):
                    person = persons[choice - 1]
                    break
                else:
                    print("❌ Invalid choice!")
            except ValueError:
                print("❌ Please enter a valid number")
        
        # Get all images for selected person
        images = self.face_mapping[person]['images']
        if not images:
            print(f"\nNo images found for {person}")
            return
        
        print(f"\n=== Images of {person} ===")
        print(f"Found {len(images)} images")
        
        # Create a window to display images
        window_name = f"Images of {person} (Press 'n' for next, 'p' for previous, 'q' to quit)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        current_idx = 0
        while True:
            # Load and display current image
            img_path = images[current_idx]
            img = cv2.imread(img_path)
            if img is not None:
                # Resize if too large
                max_height = 800
                if img.shape[0] > max_height:
                    scale = max_height / img.shape[0]
                    img = cv2.resize(img, None, fx=scale, fy=scale)
                
                # Add image number and total
                cv2.putText(img, f"Image {current_idx + 1}/{len(images)}", 
                           (10, 30), self.font, 1, (255, 255, 255), 2)
                
                cv2.imshow(window_name, img)
            
            # Wait for key press
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):  # Quit
                break
            elif key == ord('n'):  # Next image
                current_idx = (current_idx + 1) % len(images)
            elif key == ord('p'):  # Previous image
                current_idx = (current_idx - 1) % len(images)
            elif key == ord('d'):  # Delete current image
                if len(images) > 1:  # Keep at least one image
                    try:
                        # Delete from AWS collection
                        face_id = self.face_mapping[person]['face_ids'][current_idx]
                        self.rekognition.delete_faces(
                            CollectionId=self.collection_id,
                            FaceIds=[face_id]
                        )
                        
                        # Delete file
                        os.remove(img_path)
                        
                        # Update mapping
                        del self.face_mapping[person]['face_ids'][current_idx]
                        del self.face_mapping[person]['images'][current_idx]
                        self.save_face_mapping()
                        
                        print(f"✅ Deleted image {current_idx + 1}")
                        images = self.face_mapping[person]['images']  # Update images list
                        if not images:
                            print("No more images left for this person")
                            break
                        current_idx = current_idx % len(images)
                    except Exception as e:
                        print(f"❌ Error deleting image: {e}")
                else:
                    print("❌ Cannot delete the last image of a person")
        
        cv2.destroyAllWindows()

    def manage_collection(self):
        """Manage the entire face collection"""
        if not self.face_mapping:
            print("No indexed faces found!")
            return
        
        while True:
            print("\n=== Collection Management ===")
            print("1. List all users in collection")
            print("2. Delete specific user")
            print("3. Delete entire collection")
            print("4. Return to main menu")
            
            choice = input("\nEnter choice (1-4): ").strip()
            
            if choice == '1':
                # List all users with their details
                print("\n=== Collection Contents ===")
                total_faces = 0
                for person, data in sorted(self.face_mapping.items()):
                    face_count = len(data['face_ids'])
                    total_faces += face_count
                    print(f"\n{person}:")
                    print(f"  • UUID: {data['uuid']}")
                    print(f"  • Face IDs: {face_count}")
                    print(f"  • Images: {len(data['images'])}")
                    print(f"  • Created: {data.get('created_at', 'Unknown')}")
                    print(f"  • Last Updated: {data.get('last_updated', 'Unknown')}")
                    print("  • Image paths:")
                    for img_path in data['images']:
                        print(f"    - {os.path.basename(img_path)}")
                print(f"\nTotal users: {len(self.face_mapping)}")
                print(f"Total faces: {total_faces}")
                
            elif choice == '2':
                # Delete specific user
                print("\n=== Delete User ===")
                persons = sorted(self.face_mapping.keys())
                for i, person in enumerate(persons, 1):
                    data = self.face_mapping[person]
                    print(f"{i}. {person} (UUID: {data['uuid']}, {len(data['face_ids'])} faces)")
                
                try:
                    user_choice = input("\nEnter user number to delete (or 'q' to cancel): ").strip()
                    if user_choice.lower() == 'q':
                        continue
                    
                    user_choice = int(user_choice)
                    if 1 <= user_choice <= len(persons):
                        person = persons[user_choice - 1]
                        person_uuid = self.face_mapping[person]['uuid']
                        
                        # Confirm deletion
                        confirm = input(f"\nAre you sure you want to delete {person} (UUID: {person_uuid})? This will remove {len(self.face_mapping[person]['face_ids'])} faces. (y/n): ").strip().lower()
                        if confirm == 'y':
                            try:
                                # Delete from AWS collection
                                self.rekognition.delete_faces(
                                    CollectionId=self.collection_id,
                                    FaceIds=self.face_mapping[person]['face_ids']
                                )
                                
                                # Delete image files
                                for img_path in self.face_mapping[person]['images']:
                                    try:
                                        os.remove(img_path)
                                    except Exception as e:
                                        print(f"⚠️ Could not delete file {img_path}: {e}")
                                
                                # Remove from mapping
                                del self.face_mapping[person]
                                self.save_face_mapping()
                                
                                print(f"✅ Successfully deleted {person} (UUID: {person_uuid}) from collection")
                            except Exception as e:
                                print(f"❌ Error deleting user: {e}")
                        else:
                            print("Deletion cancelled")
                    else:
                        print("❌ Invalid choice!")
                except ValueError:
                    print("❌ Please enter a valid number")
                
            elif choice == '3':
                # Delete entire collection
                confirm = input("\n⚠️ WARNING: This will delete ALL faces from the collection and remove all image files. Are you sure? (y/n): ").strip().lower()
                if confirm == 'y':
                    try:
                        # Get all face IDs
                        all_face_ids = []
                        for person_data in self.face_mapping.values():
                            all_face_ids.extend(person_data['face_ids'])
                        
                        if all_face_ids:
                            # Delete from AWS collection
                            self.rekognition.delete_faces(
                                CollectionId=self.collection_id,
                                FaceIds=all_face_ids
                            )
                        
                        # Delete all image files
                        for person_data in self.face_mapping.values():
                            for img_path in person_data['images']:
                                try:
                                    os.remove(img_path)
                                except Exception as e:
                                    print(f"⚠️ Could not delete file {img_path}: {e}")
                        
                        # Clear mapping
                        self.face_mapping = {}
                        self.save_face_mapping()
                        
                        print("✅ Successfully deleted entire collection")
                    except Exception as e:
                        print(f"❌ Error deleting collection: {e}")
                else:
                    print("Deletion cancelled")
                
            elif choice == '4':
                return
            else:
                print("❌ Invalid choice, please try again")

    def register_faces_from_folder(self):
        """Register all faces from a specified folder"""
        print("\n=== Register Faces from Folder ===")
        folder_path = input("Enter the folder path containing face images: ").strip()
        
        if not os.path.exists(folder_path):
            print(f"❌ Folder not found: {folder_path}")
            return
        
        # Get all image files from the folder
        image_files = []
        for ext in ('.jpg', '.jpeg', '.png'):
            image_files.extend([f for f in os.listdir(folder_path) if f.lower().endswith(ext)])
        
        if not image_files:
            print("❌ No image files found in the folder!")
            return
        
        print(f"\nFound {len(image_files)} image files")
        
        # Ask for naming convention
        print("\nHow would you like to name the faces?")
        print("1. Use folder name as base (e.g., folder_name_1, folder_name_2)")
        print("2. Use image filenames (without extension)")
        print("3. Enter custom prefix")
        
        naming_choice = input("Enter choice (1-3): ").strip()
        
        base_name = ""
        if naming_choice == '1':
            base_name = os.path.basename(folder_path)
        elif naming_choice == '2':
            # Will use individual filenames
            pass
        elif naming_choice == '3':
            base_name = input("Enter custom prefix for names: ").strip()
        else:
            print("❌ Invalid choice!")
            return
        
        # Process each image
        successful = 0
        failed = 0
        skipped = 0
        
        print("\nProcessing images...")
        for i, image_file in enumerate(image_files, 1):
            image_path = os.path.join(folder_path, image_file)
            
            # Determine the name for this face
            if naming_choice == '2':
                person_name = os.path.splitext(image_file)[0]
            else:
                person_name = f"{base_name}_{i}"
            
            print(f"\nProcessing {i}/{len(image_files)}: {image_file}")
            print(f"Will be registered as: {person_name}")
            
            # Check if this person already exists
            if person_name in self.face_mapping:
                print(f"⚠️ {person_name} already exists in collection")
                action = input("What would you like to do?\n"
                             "1. Add as new person with different name\n"
                             "2. Add to existing person\n"
                             "3. Skip this image\n"
                             "Enter choice (1-3): ").strip()
                
                if action == '1':
                    person_name = input("Enter new name: ").strip()
                    if not person_name:
                        print("❌ Name cannot be empty, skipping...")
                        skipped += 1
                        continue
                elif action == '2':
                    # Will use existing person_name
                    pass
                else:
                    print("Skipping this image...")
                    skipped += 1
                    continue
            
            # Try to index the face
            if self.index_face(image_path, person_name):
                successful += 1
            else:
                failed += 1
        
        # Print summary
        print("\n=== Registration Summary ===")
        print(f"Total images processed: {len(image_files)}")
        print(f"Successfully registered: {successful}")
        print(f"Failed to register: {failed}")
        print(f"Skipped: {skipped}")
        
        if successful > 0:
            print("\nNew entries in collection:")
            for person, data in self.face_mapping.items():
                if data.get('created_at', '').startswith(datetime.now().strftime('%Y-%m-%d')):
                    print(f"  • {person} (UUID: {data['uuid']})")

    def cleanup_duplicate_entries(self):
        """Clean up and merge duplicate entries in the collection"""
        print("\n=== Cleaning Up Duplicate Entries ===")
        
        # Find potential duplicates (same name, different UUIDs)
        name_groups = {}
        for person, data in self.face_mapping.items():
            # Extract base name without UUID
            if '_' in person:
                # If name contains UUID, split and take first part
                base_name = person.split('_')[0]
            else:
                # If name doesn't contain UUID, use as is
                base_name = person
            
            # Normalize the name (remove any extra spaces, convert to lowercase)
            base_name = base_name.strip().lower()
            
            if base_name not in name_groups:
                name_groups[base_name] = []
            name_groups[base_name].append((person, data))
        
        # Find groups with multiple entries
        duplicates = {name: entries for name, entries in name_groups.items() if len(entries) > 1}
        
        if not duplicates:
            print("No duplicate entries found!")
            # Print current collection state for verification
            print("\nCurrent collection state:")
            for person, data in sorted(self.face_mapping.items()):
                print(f"\n{person}:")
                print(f"  • UUID: {data['uuid']}")
                print(f"  • Face IDs: {len(data['face_ids'])}")
                print(f"  • Images: {len(data['images'])}")
            return
        
        print(f"\nFound {len(duplicates)} names with multiple entries:")
        for name, entries in duplicates.items():
            print(f"\n{name}:")
            for person, data in entries:
                print(f"  • {person} (UUID: {data['uuid']}, {len(data['face_ids'])} faces)")
                print(f"    Images: {[os.path.basename(img) for img in data['images']]}")
        
        # Process each group
        for name, entries in duplicates.items():
            print(f"\nProcessing duplicates for: {name}")
            print("What would you like to do?")
            print("1. Merge all entries into one")
            print("2. Keep the entry with most faces")
            print("3. Skip this group")
            
            action = input("Enter choice (1-3): ").strip()
            
            if action == '1':
                # Merge all entries
                # Use the first entry as the base
                base_person, base_data = entries[0]
                base_name = base_person.split('_')[0] if '_' in base_person else base_person
                
                print(f"\nMerging into: {base_name}")
                print("Current entries to merge:")
                for person, data in entries:
                    print(f"  • {person}: {len(data['face_ids'])} faces, {len(data['images'])} images")
                
                # Collect all face IDs and images
                all_face_ids = base_data['face_ids'].copy()
                all_images = base_data['images'].copy()
                created_at = base_data.get('created_at', datetime.now().isoformat())
                
                # Merge other entries
                for person, data in entries[1:]:
                    print(f"\nMerging {person}:")
                    print(f"  • Adding {len(data['face_ids'])} face IDs")
                    print(f"  • Adding {len(data['images'])} images")
                    all_face_ids.extend(data['face_ids'])
                    all_images.extend(data['images'])
                    # Delete the old entry
                    del self.face_mapping[person]
                
                # Update the base entry
                self.face_mapping[base_name] = {
                    'uuid': base_data['uuid'],
                    'face_ids': all_face_ids,
                    'images': all_images,
                    'created_at': created_at,
                    'last_updated': datetime.now().isoformat()
                }
                
                print(f"\n✅ Merged all entries into: {base_name}")
                print(f"  • Total faces: {len(all_face_ids)}")
                print(f"  • Total images: {len(all_images)}")
                
            elif action == '2':
                # Keep the entry with most faces
                best_entry = max(entries, key=lambda x: len(x[1]['face_ids']))
                best_person, best_data = best_entry
                best_name = best_person.split('_')[0] if '_' in best_person else best_person
                
                print(f"\nKeeping entry: {best_person}")
                print(f"Faces: {len(best_data['face_ids'])}, Images: {len(best_data['images'])}")
                
                # Delete other entries
                for person, data in entries:
                    if person != best_person:
                        print(f"\nDeleting {person}:")
                        print(f"  • Removing {len(data['face_ids'])} face IDs from AWS")
                        print(f"  • Deleting {len(data['images'])} image files")
                        
                        # Delete from AWS collection
                        try:
                            self.rekognition.delete_faces(
                                CollectionId=self.collection_id,
                                FaceIds=data['face_ids']
                            )
                        except Exception as e:
                            print(f"⚠️ Error deleting faces from AWS: {e}")
                        
                        # Delete image files
                        for img_path in data['images']:
                            try:
                                os.remove(img_path)
                            except Exception as e:
                                print(f"⚠️ Error deleting file {img_path}: {e}")
                        
                        # Remove from mapping
                        del self.face_mapping[person]
                
                # Rename the best entry if needed
                if best_person != best_name:
                    print(f"\nRenaming {best_person} to {best_name}")
                    self.face_mapping[best_name] = self.face_mapping.pop(best_person)
                
                print(f"\n✅ Kept entry: {best_name}")
                print(f"  • Faces: {len(best_data['face_ids'])}")
                print(f"  • Images: {len(best_data['images'])}")
                
            else:
                print("Skipping this group...")
        
        # Save changes
        self.save_face_mapping()
        print("\n✅ Cleanup complete!")
        
        # Print final collection state
        print("\nFinal collection state:")
        for person, data in sorted(self.face_mapping.items()):
            print(f"\n{person}:")
            print(f"  • UUID: {data['uuid']}")
            print(f"  • Face IDs: {len(data['face_ids'])}")
            print(f"  • Images: {len(data['images'])}")

    def load_historical_data(self):
        """Load historical performance data"""
        historical_file = "face_recognition_history.json"
        if os.path.exists(historical_file):
            try:
                with open(historical_file, 'r') as f:
                    loaded_data = json.load(f)
                    # Convert the loaded data back to defaultdict structure
                    self.historical_data = {
                        'daily_stats': defaultdict(lambda: {
                            'faces_detected': 0,
                            'faces_recognized': 0,
                            'false_positives': 0,
                            'false_negatives': 0,
                            'avg_confidence': 0,
                            'avg_quality': 0
                        })
                    }
                    # Update with loaded data
                    if 'daily_stats' in loaded_data:
                        for date, stats in loaded_data['daily_stats'].items():
                            self.historical_data['daily_stats'][date].update(stats)
            except Exception as e:
                print(f"⚠️ Error loading historical data: {e}")
                # Initialize with empty defaultdict if loading fails
                self.historical_data = {
                    'daily_stats': defaultdict(lambda: {
                        'faces_detected': 0,
                        'faces_recognized': 0,
                        'false_positives': 0,
                        'false_negatives': 0,
                        'avg_confidence': 0,
                        'avg_quality': 0
                    })
                }

    def save_historical_data(self):
        """Save historical performance data"""
        historical_file = "face_recognition_history.json"
        try:
            with open(historical_file, 'w') as f:
                json.dump(self.historical_data, f, indent=2)
        except Exception as e:
            print(f"⚠️ Error saving historical data: {e}")

    def update_performance_metrics(self, frame_results, processing_time):
        """Update performance metrics with new frame results"""
        self.performance_metrics['total_frames_processed'] += 1
        
        faces_detected = len(frame_results.get('faces', []))
        self.performance_metrics['total_faces_detected'] += faces_detected
        
        recognized_faces = [f for f in frame_results.get('faces', []) if f.get('Recognized', False)]
        self.performance_metrics['total_faces_recognized'] += len(recognized_faces)
        
        for face in frame_results.get('faces', []):
            if 'Face' in face and 'Confidence' in face['Face']:
                self.performance_metrics['confidence_scores'].append(face['Face']['Confidence'])
        
        self.performance_metrics['recognition_times'].append(processing_time)
        
        try:
            today = datetime.now().strftime('%Y-%m-%d')
            self.historical_data['daily_stats'][today]['faces_detected'] += faces_detected
            self.historical_data['daily_stats'][today]['faces_recognized'] += len(recognized_faces)
            
            if self.performance_metrics['confidence_scores']:
                avg_confidence = np.mean(self.performance_metrics['confidence_scores'])
                self.historical_data['daily_stats'][today]['avg_confidence'] = avg_confidence
            
            if self.performance_metrics['total_frames_processed'] % 100 == 0:
                self.save_historical_data()
        except Exception as e:
            print(f"Error updating metrics: {e}")

    def generate_performance_report(self):
        """Generate a comprehensive performance report"""
        print("\n=== Face Recognition Performance Report ===")
        
        # Overall statistics
        print("\nOverall Statistics:")
        print(f"Total frames processed: {self.performance_metrics['total_frames_processed']}")
        print(f"Total faces detected: {self.performance_metrics['total_faces_detected']}")
        print(f"Total faces recognized: {self.performance_metrics['total_faces_recognized']}")
        
        if self.performance_metrics['total_faces_detected'] > 0:
            recognition_rate = (self.performance_metrics['total_faces_recognized'] / 
                             self.performance_metrics['total_faces_detected'] * 100)
            print(f"Recognition rate: {recognition_rate:.1f}%")
        
        # Confidence statistics
        if self.performance_metrics['confidence_scores']:
            avg_confidence = np.mean(self.performance_metrics['confidence_scores'])
            std_confidence = np.std(self.performance_metrics['confidence_scores'])
            print(f"\nConfidence Statistics:")
            print(f"Average confidence: {avg_confidence:.1f}%")
            print(f"Confidence standard deviation: {std_confidence:.1f}%")
        
        # Processing time statistics
        if self.performance_metrics['recognition_times']:
            avg_time = np.mean(self.performance_metrics['recognition_times'])
            print(f"\nProcessing Time:")
            print(f"Average processing time per frame: {avg_time*1000:.1f}ms")
        
        # Daily statistics
        print("\nDaily Statistics:")
        for date, stats in sorted(self.historical_data['daily_stats'].items()):
            print(f"\n{date}:")
            print(f"  • Faces detected: {stats['faces_detected']}")
            print(f"  • Faces recognized: {stats['faces_recognized']}")
            if stats['faces_detected'] > 0:
                daily_rate = (stats['faces_recognized'] / stats['faces_detected'] * 100)
                print(f"  • Recognition rate: {daily_rate:.1f}%")
            print(f"  • Average confidence: {stats['avg_confidence']:.1f}%")
        
        # Generate plots
        self.generate_performance_plots()

    def generate_performance_plots(self):
        """Generate performance visualization plots"""
        # Create plots directory if it doesn't exist
        plots_dir = "performance_plots"
        os.makedirs(plots_dir, exist_ok=True)
        
        # Daily recognition rates
        dates = []
        recognition_rates = []
        for date, stats in sorted(self.historical_data['daily_stats'].items()):
            dates.append(date)
            if stats['faces_detected'] > 0:
                rate = (stats['faces_recognized'] / stats['faces_detected'] * 100)
                recognition_rates.append(rate)
            else:
                recognition_rates.append(0)
        
        plt.figure(figsize=(12, 6))
        plt.plot(dates, recognition_rates, marker='o')
        plt.title('Daily Face Recognition Rates')
        plt.xlabel('Date')
        plt.ylabel('Recognition Rate (%)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'daily_recognition_rates.png'))
        plt.close()
        
        # Confidence distribution
        if self.performance_metrics['confidence_scores']:
            plt.figure(figsize=(10, 6))
            plt.hist(self.performance_metrics['confidence_scores'], bins=20)
            plt.title('Face Recognition Confidence Distribution')
            plt.xlabel('Confidence Score')
            plt.ylabel('Frequency')
            plt.savefig(os.path.join(plots_dir, 'confidence_distribution.png'))
            plt.close()
        
        # Processing time distribution
        if self.performance_metrics['recognition_times']:
            plt.figure(figsize=(10, 6))
            plt.hist([t*1000 for t in self.performance_metrics['recognition_times']], bins=20)
            plt.title('Frame Processing Time Distribution')
            plt.xlabel('Processing Time (ms)')
            plt.ylabel('Frequency')
            plt.savefig(os.path.join(plots_dir, 'processing_time_distribution.png'))
            plt.close()

    def load_person_settings(self):
        """Load person-specific recognition settings"""
        settings_file = "person_settings.json"
        if os.path.exists(settings_file):
            try:
                with open(settings_file, 'r') as f:
                    loaded_settings = json.load(f)
                    for person, settings in loaded_settings.items():
                        self.person_settings[person].update(settings)
            except Exception as e:
                print(f"⚠️ Error loading person settings: {e}")

    def save_person_settings(self):
        """Save person-specific recognition settings"""
        settings_file = "person_settings.json"
        try:
            with open(settings_file, 'w') as f:
                json.dump(dict(self.person_settings), f, indent=2)
        except Exception as e:
            print(f"⚠️ Error saving person settings: {e}")

    def backup_collection(self):
        """Create a backup of the entire face collection"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"collection_backup_{timestamp}"
        backup_path = os.path.join(self.collection_settings['backup_dir'], backup_name)
        
        try:
            # Create backup directory
            os.makedirs(backup_path, exist_ok=True)
            
            # Backup face mapping
            shutil.copy2(self.face_mapping_file, os.path.join(backup_path, "face_mapping.json"))
            
            # Backup historical data
            if os.path.exists("face_recognition_history.json"):
                shutil.copy2("face_recognition_history.json", 
                           os.path.join(backup_path, "face_recognition_history.json"))
            
            # Backup indexed faces
            indexed_backup = os.path.join(backup_path, "indexed_faces")
            os.makedirs(indexed_backup, exist_ok=True)
            for person, data in self.face_mapping.items():
                for img_path in data['images']:
                    if os.path.exists(img_path):
                        shutil.copy2(img_path, os.path.join(indexed_backup, os.path.basename(img_path)))
            
            # Create zip archive
            zip_path = f"{backup_path}.zip"
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, _, files in os.walk(backup_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, backup_path)
                        zipf.write(file_path, arcname)
            
            # Remove temporary backup directory
            shutil.rmtree(backup_path)
            
            print(f"✅ Collection backed up to: {zip_path}")
            return zip_path
            
        except Exception as e:
            print(f"❌ Error creating backup: {e}")
            if os.path.exists(backup_path):
                shutil.rmtree(backup_path)
            return None

    def restore_collection(self, backup_path):
        """Restore face collection from a backup"""
        if not os.path.exists(backup_path):
            print(f"❌ Backup file not found: {backup_path}")
            return False
        
        try:
            # Create temporary directory for extraction
            temp_dir = "temp_restore"
            os.makedirs(temp_dir, exist_ok=True)
            
            # Extract backup
            with zipfile.ZipFile(backup_path, 'r') as zipf:
                zipf.extractall(temp_dir)
            
            # Restore face mapping
            mapping_file = os.path.join(temp_dir, "face_mapping.json")
            if os.path.exists(mapping_file):
                shutil.copy2(mapping_file, self.face_mapping_file)
                self.face_mapping = self.load_face_mapping()
            
            # Restore historical data
            history_file = os.path.join(temp_dir, "face_recognition_history.json")
            if os.path.exists(history_file):
                shutil.copy2(history_file, "face_recognition_history.json")
            
            # Restore indexed faces
            indexed_dir = os.path.join(temp_dir, "indexed_faces")
            if os.path.exists(indexed_dir):
                for file in os.listdir(indexed_dir):
                    src = os.path.join(indexed_dir, file)
                    dst = os.path.join(self.indexed_dir, file)
                    shutil.copy2(src, dst)
            
            # Cleanup
            shutil.rmtree(temp_dir)
            
            print("✅ Collection restored successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error restoring backup: {e}")
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            return False

    def export_collection(self, format='json'):
        """Export face collection in specified format"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_name = f"collection_export_{timestamp}"
        export_path = os.path.join(self.collection_settings['export_dir'], export_name)
        
        try:
            os.makedirs(export_path, exist_ok=True)
            
            if format == 'json':
                # Export as JSON
                export_file = f"{export_path}.json"
                export_data = {
                    'version': self.collection_settings['version'],
                    'export_date': datetime.now().isoformat(),
                    'face_mapping': self.face_mapping,
                    'performance_metrics': self.performance_metrics
                }
                with open(export_file, 'w') as f:
                    json.dump(export_data, f, indent=2)
                    
            elif format == 'csv':
                # Export as CSV
                export_file = f"{export_path}.csv"
                with open(export_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Person', 'UUID', 'Face Count', 'Image Count', 
                                   'Created At', 'Last Updated'])
                    for person, data in self.face_mapping.items():
                        writer.writerow([
                            person,
                            data['uuid'],
                            len(data['face_ids']),
                            len(data['images']),
                            data.get('created_at', ''),
                            data.get('last_updated', '')
                        ])
            
            print(f"✅ Collection exported to: {export_file}")
            return export_file
            
        except Exception as e:
            print(f"❌ Error exporting collection: {e}")
            if os.path.exists(export_path):
                shutil.rmtree(export_path)
            return None

    def process_video_batch(self, video_paths, output_format='mp4', report_format='csv'):
        """Process multiple videos in batch"""
        if not video_paths:
            print("❌ No videos provided")
            return
        
        print(f"\n=== Processing {len(video_paths)} Videos ===")
        
        # Create output directories
        output_dir = "output_videos"
        report_dir = "video_reports"
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(report_dir, exist_ok=True)
        
        # Process videos in parallel
        results = []
        threads = []
        result_queue = Queue()
        
        def process_video_thread(video_path):
            try:
                # Process single video
                start_time = time.time()
                self.process_video(video_path)
                processing_time = time.time() - start_time
                
                # Get results
                result = {
                    'video': video_path,
                    'processing_time': processing_time,
                    'faces_detected': self.performance_metrics['total_faces_detected'],
                    'faces_recognized': self.performance_metrics['total_faces_recognized'],
                    'recognized_persons': list(self.recognized_faces)
                }
                result_queue.put(result)
                
            except Exception as e:
                print(f"❌ Error processing {video_path}: {e}")
                result_queue.put({'video': video_path, 'error': str(e)})
        
        # Start processing threads
        for video_path in video_paths:
            thread = threading.Thread(target=process_video_thread, args=(video_path,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Collect results
        while not result_queue.empty():
            results.append(result_queue.get())
        
        # Generate batch report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(report_dir, f"batch_report_{timestamp}")
        
        if report_format == 'csv':
            report_file += '.csv'
            with open(report_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Video', 'Processing Time', 'Faces Detected',
                               'Faces Recognized', 'Recognized Persons', 'Status'])
                for result in results:
                    if 'error' in result:
                        writer.writerow([result['video'], '', '', '', '', f"Error: {result['error']}"])
                    else:
                        writer.writerow([
                            result['video'],
                            f"{result['processing_time']:.1f}s",
                            result['faces_detected'],
                            result['faces_recognized'],
                            ', '.join(result['recognized_persons']),
                            'Success'
                        ])
        
        elif report_format == 'json':
            report_file += '.json'
            with open(report_file, 'w') as f:
                json.dump({
                    'timestamp': timestamp,
                    'videos_processed': len(video_paths),
                    'results': results
                }, f, indent=2)
        
        print(f"\n✅ Batch processing complete!")
        print(f"Report saved to: {report_file}")

    def update_person_settings(self, person_name, settings):
        """Update recognition settings for a specific person"""
        if person_name not in self.face_mapping:
            print(f"❌ Person not found: {person_name}")
            return False
        
        try:
            # Update settings
            self.person_settings[person_name].update(settings)
            self.save_person_settings()
            print(f"✅ Updated settings for {person_name}")
            return True
        except Exception as e:
            print(f"❌ Error updating settings: {e}")
            return False

    def get_person_settings(self, person_name):
        """Get current settings for a specific person"""
        if person_name not in self.face_mapping:
            print(f"❌ Person not found: {person_name}")
            return None
        return dict(self.person_settings[person_name])

    def sync_collection(self):
        """Sync local data with AWS Rekognition collection"""
        print("\n=== Syncing with AWS Collection ===")
        try:
            # Get all faces from AWS collection
            aws_faces = []
            paginator = self.rekognition.get_paginator('list_faces')
            
            for page in paginator.paginate(CollectionId=self.collection_id):
                aws_faces.extend(page['Faces'])
            
            print(f"Found {len(aws_faces)} faces in AWS collection")
            
            # Create a mapping of face IDs to external IDs (person names)
            aws_face_map = {}
            for face in aws_faces:
                face_id = face['FaceId']
                external_id = face['ExternalImageId']
                # Use the external ID directly as the person name
                person_name = external_id
                if person_name not in aws_face_map:
                    aws_face_map[person_name] = []
                aws_face_map[person_name].append(face_id)
            
            # Check for mismatches and sync
            changes_made = False
            
            # 1. Check for faces in AWS but not in local mapping
            for person_name, face_ids in aws_face_map.items():
                if person_name not in self.face_mapping:
                    print(f"⚠️ Found person in AWS but not locally: {person_name}")
                    # Check if we have any images for this person in the indexed directory
                    person_images = []
                    for img_file in os.listdir(self.indexed_dir):
                        if img_file.startswith(person_name + '_'):
                            img_path = os.path.join(self.indexed_dir, img_file)
                            if os.path.exists(img_path):
                                person_images.append(img_path)
                    
                    # Create new entry in local mapping
                    self.face_mapping[person_name] = {
                        'uuid': str(uuid.uuid4()),
                        'face_ids': face_ids,
                        'images': person_images,  # Use found images if any
                        'created_at': datetime.now().isoformat(),
                        'last_updated': datetime.now().isoformat()
                    }
                    changes_made = True
                else:
                    # Check for missing face IDs
                    local_face_ids = set(self.face_mapping[person_name]['face_ids'])
                    aws_face_ids = set(face_ids)
                    
                    # Find faces in AWS but not locally
                    missing_locally = aws_face_ids - local_face_ids
                    if missing_locally:
                        print(f"⚠️ Found {len(missing_locally)} faces in AWS but not locally for {person_name}")
                        self.face_mapping[person_name]['face_ids'].extend(missing_locally)
                        changes_made = True
                    
                    # Find faces locally but not in AWS
                    missing_in_aws = local_face_ids - aws_face_ids
                    if missing_in_aws:
                        print(f"⚠️ Found {len(missing_in_aws)} faces locally but not in AWS for {person_name}")
                        # Remove these face IDs from local mapping
                        self.face_mapping[person_name]['face_ids'] = list(aws_face_ids)
                        # Keep only images that exist
                        self.face_mapping[person_name]['images'] = [
                            img for img in self.face_mapping[person_name]['images']
                            if os.path.exists(img)
                        ]
                        changes_made = True
            
            # 2. Check for people in local mapping but not in AWS
            local_people = set(self.face_mapping.keys())
            aws_people = set(aws_face_map.keys())
            missing_in_aws = local_people - aws_people
            
            if missing_in_aws:
                print(f"⚠️ Found {len(missing_in_aws)} people locally but not in AWS")
                for person in missing_in_aws:
                    print(f"  • {person}")
                    # Remove from local mapping
                    del self.face_mapping[person]
                    changes_made = True
            
            # Save changes if any were made
            if changes_made:
                self.save_face_mapping()
                print("✅ Local mapping updated to match AWS collection")
            else:
                print("✅ Local mapping is in sync with AWS collection")
            
            # Calculate summary
            total_people = len(self.face_mapping)
            total_faces = sum(len(data['face_ids']) for data in self.face_mapping.values())
            total_images = sum(len(data['images']) for data in self.face_mapping.values())
            
            # Print summary
            print("\n=== Collection Summary ===")
            print(f"Total people: {total_people}")
            print(f"Total faces: {total_faces}")
            print(f"Total images: {total_images}")
            
            return {
                'success': True,
                'changes_made': changes_made,
                'total_people': total_people,
                'total_faces': total_faces,
                'total_images': total_images,
                'aws_faces': len(aws_faces)
            }
            
        except Exception as e:
            print(f"❌ Error syncing with AWS collection: {e}")
            if hasattr(e, 'response'):
                print(f"AWS Response: {e.response}")
            return {
                'success': False,
                'error': str(e)
            }

    def predict_face_position(self, track_data, frame_shape):
        """Predict face position based on velocity and last position, ensuring it stays within frame bounds"""
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
        new_x = last_x + (dx * width)  # Scale velocity by frame width
        new_y = last_y + (dy * height)  # Scale velocity by frame height
        
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
        """Calculate velocity between two face positions"""
        height, width = frame_shape[:2]
        
        # Calculate center points
        old_center = ((old_box[0] + old_box[2])/2, (old_box[1] + old_box[3])/2)
        new_center = ((new_box[0] + new_box[2])/2, (new_box[1] + new_box[3])/2)
        
        # Calculate velocity (normalized by frame size)
        dx = (new_center[0] - old_center[0]) / width
        dy = (new_center[1] - old_center[1]) / height
        
        return (dx, dy)

class CollectionManagerGUI:
    def __init__(self, root, analyzer):
        """Initialize the GUI with improved styling and organization"""
        self.root = root
        self.analyzer = analyzer  # Store analyzer reference
        
        # Create main container
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title frame
        title_frame = ttk.Frame(main_frame)
        title_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Title with icon
        title_label = ttk.Label(title_frame, 
                              text="👥 Face Recognition System",
                              style="Title.TLabel",
                              font=("Segoe UI", 24, "bold"))
        title_label.pack(side=tk.LEFT)
        
        # Add sync button to title frame
        sync_btn = ttk.Button(title_frame, 
                             text="Sync with AWS 🔄",
                             style="Info.TButton",
                             command=self.sync_with_aws)
        sync_btn.pack(side=tk.RIGHT)
        
        # Configure styles
        self._configure_styles()
        
        # Collection info
        info_frame = ttk.LabelFrame(main_frame, text="Collection Information", padding="10")
        info_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Info grid
        info_grid = ttk.Frame(info_frame)
        info_grid.pack(fill=tk.X)
        
        # Total faces
        total_frame = ttk.Frame(info_grid)
        total_frame.grid(row=0, column=0, padx=10)
        ttk.Label(total_frame, text="Total Faces:", 
                 style="Subheader.TLabel").pack(anchor="w")
        self.total_faces_label = ttk.Label(total_frame, text="0")
        self.total_faces_label.pack(anchor="w")
        
        # Unique people
        unique_frame = ttk.Frame(info_grid)
        unique_frame.grid(row=0, column=1, padx=10)
        ttk.Label(unique_frame, text="Unique People:", 
                 style="Subheader.TLabel").pack(anchor="w")
        self.unique_people_label = ttk.Label(unique_frame, text="0")
        self.unique_people_label.pack(anchor="w")
        
        # Last update
        update_frame = ttk.Frame(info_grid)
        update_frame.grid(row=0, column=2, padx=10)
        ttk.Label(update_frame, text="Last Update:", 
                 style="Subheader.TLabel").pack(anchor="w")
        self.last_update_label = ttk.Label(update_frame, text="Never")
        self.last_update_label.pack(anchor="w")
        
        # Action buttons in two columns
        action_frame = ttk.LabelFrame(main_frame, text="Actions", padding="10")
        action_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Create two columns
        left_column = ttk.Frame(action_frame)
        left_column.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        right_column = ttk.Frame(action_frame)
        right_column.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        # Left column buttons
        ttk.Button(left_column, text="Process Video (V)", 
                  style="Primary.TButton",
                  command=self.process_video).pack(fill=tk.X, pady=2)
        
        ttk.Button(left_column, text="Capture Face (C)", 
                  style="Success.TButton",
                  command=self.capture_face_from_webcam).pack(fill=tk.X, pady=2)
        
        # Right column buttons
        ttk.Button(right_column, text="Show Collection (S)", 
                  style="Info.TButton",
                  command=self.show_collection).pack(fill=tk.X, pady=2)
        

        ttk.Button(right_column, text="Register Faces (R)", 
                  style="Info.TButton",
                  command=self.register_faces).pack(fill=tk.X, pady=2)
        
        # Exit button at bottom
        ttk.Button(main_frame, text="Exit (Q)", 
                  style="Danger.TButton",
                  command=self.root.quit).pack(fill=tk.X, pady=(20, 0))
        
        # Status bar
        status_frame = ttk.Frame(main_frame)
        status_frame.pack(fill=tk.X, pady=(20, 0))
        
        self.status_label = ttk.Label(status_frame, 
                                    text="Ready",
                                    style="Status.TLabel")
        self.status_label.pack(side=tk.LEFT)
        
        # Load initial data
        self.update_collection_info()
        
        # Bind keyboard shortcuts
        self.root.bind('v', lambda e: self.process_video())
        self.root.bind('c', lambda e: self.capture_face_from_webcam())
        self.root.bind('r', lambda e: self.register_faces())
        self.root.bind('s', lambda e: self.show_collection())
        self.root.bind('q', lambda e: self.root.quit())
        self.root.bind('y', lambda e: self.sync_with_aws())  # Add sync shortcut
        
        # Add tooltips
        self._add_tooltips()
        
        # Center window
        self.center_window()
        
    
    def _configure_styles(self):
        """Configure ttk styles for the application"""
        style = ttk.Style()
        
        # Configure theme
        style.theme_use('clam')  # Use clam theme as base
        
        # Title style
        style.configure("Title.TLabel",
                       font=("Segoe UI", 24, "bold"),
                       padding=10)
        
        # Subheader style
        style.configure("Subheader.TLabel",
                       font=("Segoe UI", 10, "bold"),
                       padding=2)
        
        # Status style
        style.configure("Status.TLabel",
                       font=("Segoe UI", 9),
                       padding=5)
        
        # Button styles
        style.configure("Primary.TButton",
                       font=("Segoe UI", 10),
                       padding=10)
        
        style.configure("Success.TButton",
                       font=("Segoe UI", 10),
                       padding=10)
        
        style.configure("Info.TButton",
                       font=("Segoe UI", 10),
                       padding=10)
        
        style.configure("Warning.TButton",
                       font=("Segoe UI", 10),
                       padding=10)
        
        style.configure("Danger.TButton",
                       font=("Segoe UI", 10),
                       padding=10)
        
        # Message style
        style.configure("Message.TLabel",
                       font=("Segoe UI", 10),
                       padding=5)
    
    def _add_tooltips(self):
        """Add tooltips to buttons"""
        tooltips = {
            "Process Video (V)": "Process a video file to detect and recognize faces",
            "Capture Face (C)": "Capture a face from webcam",
            "Register Faces (R)": "Register faces from a folder",
            "Show Collection (S)": "View and manage the face collection",
            "Cleanup Duplicates (D)": "Find and merge duplicate entries",
            "Index Faces (I)": "Index unrecognized faces",
            "Sync with AWS (Y)": "Sync collection with AWS Rekognition",
            "Exit (Q)": "Exit the application"
        }
        
        def create_tooltip(widget, text):
            def show_tooltip(event):
                tooltip = tk.Toplevel()
                tooltip.wm_overrideredirect(True)
                tooltip.wm_geometry(f"+{event.x_root+10}+{event.y_root+10}")
                
                label = ttk.Label(tooltip, text=text,
                                background="#ffffe0",
                                relief="solid",
                                borderwidth=1,
                                padding=5)
                label.pack()
                
                def hide_tooltip():
                    tooltip.destroy()
                
                widget.tooltip = tooltip
                widget.bind("<Leave>", lambda e: hide_tooltip())
                widget.bind("<ButtonPress>", lambda e: hide_tooltip())
            
            widget.bind("<Enter>", show_tooltip)
        
        # Add tooltips to all buttons
        for widget in self.root.winfo_children():
            if isinstance(widget, ttk.Frame):
                for child in widget.winfo_children():
                    if isinstance(child, ttk.Button):
                        text = child.cget("text")
                        if text in tooltips:
                            create_tooltip(child, tooltips[text])

    def center_window(self, window=None):
        """Center the window on the screen"""
        if window is None:
            window = self.root
        window.update_idletasks()
        width = window.winfo_width()
        height = window.winfo_height()
        x = (window.winfo_screenwidth() // 2) - (width // 2)
        y = (window.winfo_screenheight() // 2) - (height // 2)
        window.geometry(f'{width}x{height}+{x}+{y}')

    def process_video(self):
        """Process a video file with progress dialog"""
        video_path = filedialog.askopenfilename(
            title="Select Video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov")]
        )
        if not video_path:
            return

        # Create a new window for video display
        video_window = tk.Toplevel(self.root)
        video_window.title(f"Processing: {os.path.basename(video_path)}")
        video_window.geometry("800x600")
        self.center_window(video_window)

        # Canvas to display video frames
        video_canvas = tk.Canvas(video_window, bg='black')
        video_canvas.pack(fill=tk.BOTH, expand=True)

        # Status label
        status_label = ttk.Label(video_window, text="Initializing...")
        status_label.pack(side=tk.BOTTOM, fill=tk.X, pady=5)

        # Store reference to PhotoImage to prevent garbage collection
        video_canvas.image = None

        def display_frame(frame, frame_num, total_frames):
            """Callback function to display processed frames"""
            if frame is None or frame.size == 0:
                return

            try:
                # Convert OpenCV frame to PhotoImage
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                # Resize image to fit canvas while maintaining aspect ratio
                canvas_width = video_canvas.winfo_width()
                canvas_height = video_canvas.winfo_height()

                scale = min(canvas_width/img.width, canvas_height/img.height)
                new_width = int(img.width * scale)
                new_height = int(img.height * scale)

                img = img.resize((new_width, new_height))

                img_tk = ImageTk.PhotoImage(image=img)

                # Update canvas
                video_canvas.create_image(canvas_width//2, canvas_height//2,
                                          image=img_tk, anchor=tk.CENTER)
                video_canvas.image = img_tk  # Keep reference

                # Update status label
                status_label.config(text=f"Processing frame {frame_num}/{total_frames}...")

            except Exception as e:
                print(f"Error displaying frame: {e}")

        def process():
            try:
                # Pass the display_frame callback to the analyzer
                results = self.analyzer.process_video(video_path, progress_callback=display_frame)

                # Processing finished
                video_window.destroy()
                self.show_message("Success", "Video processing complete!")
                self.update_collection_info()

                # Optionally display a report based on results
                # self.show_video_report(results)

            except Exception as e:
                video_window.destroy()
                self.show_message("Error", str(e), "error")

        # Start processing in a separate thread
        threading.Thread(target=process, daemon=True).start()

        # Handle window closing
        def on_closing():
            # Optional: Add confirmation dialog
            # if messagebox.askokcancel("Quit", "Do you want to stop video processing?"):
            #     # Signal processing thread to stop if needed (requires changes in VideoAnalyzer)
            video_window.destroy()

        video_window.protocol("WM_DELETE_WINDOW", on_closing)

        # Set focus to the video window
        video_window.focus_set()

    def show_collection(self):
        """Display the face collection with a clean, modern interface"""
        try:
            # Create collection window
            collection_window = tk.Toplevel(self.root)
            collection_window.title("Face Collection")
            collection_window.geometry("1200x800")
            self.center_window(collection_window)
            
            # Make window modal
            collection_window.transient(self.root)
            collection_window.grab_set()
            
            # Main container with padding
            main_frame = ttk.Frame(collection_window, padding="10")
            main_frame.pack(fill=tk.BOTH, expand=True)
            
            # Search bar at top
            search_frame = ttk.Frame(main_frame)
            search_frame.pack(fill=tk.X, pady=(0, 10))
            
            ttk.Label(search_frame, text="Search:").pack(side=tk.LEFT, padx=(0, 5))
            search_var = tk.StringVar()
            search_entry = ttk.Entry(search_frame, textvariable=search_var, width=40)
            search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
            
            refresh_btn = ttk.Button(search_frame, text="Refresh", width=10)
            refresh_btn.pack(side=tk.RIGHT)
            
            # Main content area with two panels
            content_frame = ttk.Frame(main_frame)
            content_frame.pack(fill=tk.BOTH, expand=True)
            
            # Left panel - Person list
            left_panel = ttk.Frame(content_frame, width=200)  # Reduced from 300 to 200
            left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))
            left_panel.pack_propagate(False)  # Prevent panel from shrinking
            
            # Person list header
            list_header = ttk.Frame(left_panel)
            list_header.pack(fill=tk.X, pady=(0, 5))
            ttk.Label(list_header, text="People", style="Subheader.TLabel").pack(side=tk.LEFT)
            ttk.Label(list_header, text="Faces", style="Subheader.TLabel").pack(side=tk.RIGHT)
            
            # Person list with scrollbar
            list_container = ttk.Frame(left_panel)
            list_container.pack(fill=tk.BOTH, expand=True)
            
            person_canvas = tk.Canvas(list_container)
            scrollbar = ttk.Scrollbar(list_container, orient="vertical", command=person_canvas.yview)
            person_list_frame = ttk.Frame(person_canvas)
            
            person_list_frame.bind(
                "<Configure>",
                lambda e: person_canvas.configure(scrollregion=person_canvas.bbox("all"))
            )
            
            person_canvas.create_window((0, 0), window=person_list_frame, anchor="nw", width=180)  # Reduced from 280 to 180
            person_canvas.configure(yscrollcommand=scrollbar.set)
            
            scrollbar.pack(side="right", fill="y")
            person_canvas.pack(side="left", fill="both", expand=True)
            
            # Right panel - Details and faces
            right_panel = ttk.Frame(content_frame)
            right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0))
            
            # Details section
            details_frame = ttk.LabelFrame(right_panel, text="Details", padding="10")
            details_frame.pack(fill=tk.X, pady=(0, 10))
            
            details_text = tk.Text(details_frame, wrap=tk.WORD, height=4)
            details_text.pack(fill=tk.X)
            
            # Action buttons
            action_frame = ttk.Frame(right_panel)
            action_frame.pack(fill=tk.X, pady=(0, 10))
            
            # Create a frame for delete buttons
            delete_frame = ttk.Frame(action_frame)
            delete_frame.pack(side=tk.LEFT, padx=(0, 5))
            
            delete_btn = ttk.Button(delete_frame, text="Delete Person", state=tk.DISABLED)
            delete_btn.pack(side=tk.LEFT, padx=(0, 5))
            
            delete_all_btn = ttk.Button(delete_frame, 
                                      text="Delete All Users",
                                      style="Danger.TButton",
                                      command=lambda: self._delete_all_users(collection_window))
            delete_all_btn.pack(side=tk.LEFT)
            
            # Faces grid
            faces_frame = ttk.LabelFrame(right_panel, text="Faces", padding="10")
            faces_frame.pack(fill=tk.X, expand=False)  # Changed from fill=tk.BOTH to fill=tk.X and expand=False
            
            # Face grid with scrollbar
            grid_container = ttk.Frame(faces_frame)
            grid_container.pack(fill=tk.X, expand=False)  # Changed from fill=tk.BOTH to fill=tk.X and expand=False
            
            face_canvas = tk.Canvas(grid_container)
            face_scrollbar = ttk.Scrollbar(grid_container, orient="vertical", command=face_canvas.yview)
            face_grid_frame = ttk.Frame(face_canvas)
            
            face_grid_frame.bind(
                "<Configure>",
                lambda e: face_canvas.configure(scrollregion=face_canvas.bbox("all"))
            )
            
            face_canvas.create_window((0, 0), window=face_grid_frame, anchor="nw")
            face_canvas.configure(yscrollcommand=face_scrollbar.set)
            
            face_scrollbar.pack(side="right", fill="y")
            face_canvas.pack(side="left", fill="both", expand=True)
            
            # Store references
            selected_person = None
            
            def update_collection_display(search_text=""):
                # Clear person list
                for widget in person_list_frame.winfo_children():
                    widget.destroy()
                
                # Get collection data
                collection = self.analyzer.face_mapping
                if not collection:
                    ttk.Label(person_list_frame, 
                             text="No faces in collection",
                             style="Subheader.TLabel").pack(pady=20)
                    return
                
                # Create person list
                for person, data in sorted(collection.items()):
                    # Skip if search text doesn't match
                    if search_text and search_text.lower() not in person.lower():
                        continue
                    
                    # Person frame with hover effect
                    person_frame = ttk.Frame(person_list_frame)
                    person_frame.pack(fill=tk.X, pady=1)
                    
                    # Make entire frame clickable
                    def on_enter(e, frame=person_frame):
                        frame.configure(style="Selected.TFrame")
                    
                    def on_leave(e, frame=person_frame):
                        frame.configure(style="TFrame")
                    
                    person_frame.bind("<Enter>", on_enter)
                    person_frame.bind("<Leave>", on_leave)
                    
                    # Person name and face count in a more compact layout
                    name_label = ttk.Label(person_frame, text=person, width=15, anchor="w")
                    name_label.pack(side=tk.LEFT, padx=(5, 0))
                    
                    count_label = ttk.Label(person_frame, text=str(len(data['images'])), width=3)
                    count_label.pack(side=tk.RIGHT, padx=(0, 5))
                    
                    # Make entire frame clickable
                    def on_click(e, p=person):
                        view_person(p)
                    
                    person_frame.bind("<Button-1>", on_click)
                    name_label.bind("<Button-1>", on_click)
                    count_label.bind("<Button-1>", on_click)
            
            def view_person(person):
                nonlocal selected_person
                selected_person = person
                
                # Get person data
                data = self.analyzer.face_mapping[person]
                
                # Update details
                details_text.delete(1.0, tk.END)
                details_text.insert(tk.END, f"Name: {person}\n")
                details_text.insert(tk.END, f"UUID: {data['uuid']}\n")
                details_text.insert(tk.END, f"Created: {data.get('created_at', 'Unknown')}\n")
                details_text.insert(tk.END, f"Last Updated: {data.get('last_updated', 'Unknown')}")
                
                # Enable/disable delete button
                delete_btn.config(state=tk.NORMAL if len(data['images']) > 0 else tk.DISABLED)
                
                # Clear face grid
                for widget in face_grid_frame.winfo_children():
                    widget.destroy()
                
                # Calculate grid dimensions
                num_faces = len(data['images'])
                if num_faces == 0:
                    # Show message if no faces
                    ttk.Label(face_grid_frame, 
                            text="No faces available",
                            style="Subheader.TLabel").pack(pady=20)
                    return
                
                # Calculate number of rows needed (4 faces per row)
                num_rows = (num_faces + 3) // 4  # Ceiling division
                
                # Set canvas height based on number of rows
                row_height = 160  # Height for each row (150px image + 10px padding)
                canvas_height = min(num_rows * row_height, 400)  # Max height of 400px
                face_canvas.configure(height=canvas_height)
                
                # Store references to PhotoImage objects
                photo_references = []
                
                # Create face grid
                for i, face_path in enumerate(data['images']):
                    face_frame = ttk.Frame(face_grid_frame)
                    face_frame.grid(row=i//4, column=i%4, padx=5, pady=5)
                    
                    try:
                        # Load and process image
                        img = Image.open(face_path)
                        
                        # Fix orientation based on EXIF data
                        try:
                            exif = img._getexif()
                            if exif:
                                orientation = exif.get(274)
                                if orientation:
                                    if orientation == 3:
                                        img = img.rotate(180, expand=True)
                                    elif orientation == 6:
                                        img = img.rotate(270, expand=True)
                                    elif orientation == 8:
                                        img = img.rotate(90, expand=True)
                        except Exception:
                            pass
                        
                        # Resize image to be smaller
                        img.thumbnail((120, 120))
                        photo = ImageTk.PhotoImage(img)
                        
                        # Store reference to prevent garbage collection
                        photo_references.append(photo)
                        
                        # Create image label with border
                        img_label = ttk.Label(face_frame, image=photo, style="Face.TLabel")
                        img_label.image = photo
                        img_label.pack(padx=2, pady=2)
                        
                        # Add click handlers
                        def on_face_click(event, path=face_path):
                            if messagebox.askyesno("Confirm Delete", 
                                f"Delete this face from {person}?"):
                                self.delete_face(path, person)
                                view_person(person)  # Refresh view
                        
                        def on_face_double_click(event, path=face_path):
                            self.show_full_image(path)
                        
                        img_label.bind("<Button-1>", on_face_click)
                        img_label.bind("<Double-Button-1>", on_face_double_click)
                    except Exception as e:
                        print(f"Error loading image {face_path}: {e}")
                        ttk.Label(face_frame, 
                                text="Error loading image",
                                foreground="red").pack()
                
                # Store photo references in the frame to prevent garbage collection
                face_grid_frame.photo_references = photo_references
            
            def delete_selected_person():
                if not selected_person:
                    return
                
                if messagebox.askyesno("Confirm Delete", 
                    f"Are you sure you want to delete {selected_person} and all their faces?"):
                    try:
                        # Delete from AWS collection
                        self.analyzer.rekognition.delete_faces(
                            CollectionId=self.analyzer.collection_id,
                            FaceIds=self.analyzer.face_mapping[selected_person]['face_ids']
                        )
                        
                        # Delete image files
                        for img_path in self.analyzer.face_mapping[selected_person]['images']:
                            try:
                                os.remove(img_path)
                            except Exception as e:
                                print(f"Warning: Could not delete file {img_path}: {e}")
                        
                        # Remove from mapping
                        del self.analyzer.face_mapping[selected_person]
                        self.analyzer.save_face_mapping()
                        
                        # Update display
                        self.update_collection_info()
                        update_collection_display(search_var.get())
                        
                        self.show_message("Success", f"Deleted {selected_person} and all their faces")
                    except Exception as e:
                        self.show_message("Error", f"Error deleting person: {str(e)}", "error")
            
            # Configure button commands
            delete_btn.config(command=delete_selected_person)
            delete_all_btn.config(command=lambda: self._delete_all_users(collection_window))
            refresh_btn.config(command=lambda: update_collection_display(search_var.get()))
            
            # Configure styles
            style = ttk.Style()
            style.configure("Selected.TFrame", background="#e0e0e0")
            style.configure("Face.TLabel", relief="solid", borderwidth=1)
            
            # Bind search
            search_var.trace("w", lambda *args: update_collection_display(search_var.get()))
            
            # Bind keyboard shortcuts
            collection_window.bind('r', lambda e: update_collection_display(search_var.get()))
            collection_window.bind('<Control-f>', lambda e: search_entry.focus())
            collection_window.bind('<Escape>', lambda e: collection_window.destroy())
            
            # Initial display
            update_collection_display()
            
            # Set focus to search
            search_entry.focus()
            
        except Exception as e:
            print(f"Error creating collection window: {e}")
            self.show_message("Error", f"Error creating collection window: {str(e)}", "error")

    def show_full_image(self, image_path):
        """Show full size image in a new window"""
        try:
            # Create image window
            img_window = tk.Toplevel(self.root)
            img_window.title(os.path.basename(image_path))
            
            # Load and display image
            img = Image.open(image_path)
            photo = ImageTk.PhotoImage(img)
            
            # Create label with image
            label = ttk.Label(img_window, image=photo)
            label.image = photo  # Keep reference
            label.pack()
            
            # Center window
            self.center_window(img_window)
            
        except Exception as e:
            self.show_message("Error", f"Error displaying image: {str(e)}", "error")

    def delete_face(self, face_path, person_name):
        """Delete a face from the collection"""
        try:
            if messagebox.askyesno("Confirm Delete", 
                f"Are you sure you want to delete this face from {person_name}?"):
                
                # Get face ID from the path
                face_id = None
                for idx, path in enumerate(self.analyzer.face_mapping[person_name]['images']):
                    if path == face_path:
                        face_id = self.analyzer.face_mapping[person_name]['face_ids'][idx]
                        break
                
                if face_id:
                    # Delete from AWS collection
                    self.analyzer.rekognition.delete_faces(
                        CollectionId=self.analyzer.collection_id,
                        FaceIds=[face_id]
                    )
                    
                    # Remove from mapping
                    self.analyzer.face_mapping[person_name]['face_ids'].remove(face_id)
                    self.analyzer.face_mapping[person_name]['images'].remove(face_path)
                    
                    # Delete file
                    os.remove(face_path)
                    
                    # Update last_updated
                    self.analyzer.face_mapping[person_name]['last_updated'] = datetime.now().isoformat()
                    
                    # Save changes
                    self.analyzer.save_face_mapping()
                    
                    # Update display
                    self.update_collection_info()
                    self.show_message("Success", "Face deleted successfully")
                else:
                    self.show_message("Error", "Could not find face ID", "error")
                    
        except Exception as e:
            self.show_message("Error", f"Error deleting face: {str(e)}", "error")

    def run(self):
        """Run the GUI application"""
        self.root.mainloop()

    def capture_face_from_webcam(self):
        """Launch webcam capture window with improved face quality assessment"""
        # Create capture window
        capture_window = tk.Toplevel(self.root)
        capture_window.title("Capture Face")
        capture_window.geometry("1000x800")  # Increased window size
        self.center_window(capture_window)
        
        # Main container
        main_container = ttk.Frame(capture_window)
        main_container.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        # Preview frame
        preview_frame = ttk.LabelFrame(main_container, text="Webcam Preview", padding="10")
        preview_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        # Preview canvas
        preview_canvas = tk.Canvas(preview_frame, width=640, height=480, bg='black')
        preview_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Quality metrics frame
        metrics_frame = ttk.LabelFrame(main_container, text="Face Quality Metrics", padding="10")
        metrics_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        
        # Create metric labels
        metric_labels = {}
        metrics = ['sharpness', 'size', 'brightness', 'contrast', 'angle', 'occlusion']
        for i, metric in enumerate(metrics):
            ttk.Label(metrics_frame, text=f"{metric.title()}:").grid(row=i, column=0, sticky="w", pady=2)
            metric_labels[metric] = ttk.Label(metrics_frame, text="--")
            metric_labels[metric].grid(row=i, column=1, sticky="w", pady=2)
        
        # Overall quality score
        ttk.Separator(metrics_frame, orient='horizontal').grid(row=len(metrics), column=0, columnspan=2, sticky="ew", pady=10)
        ttk.Label(metrics_frame, text="Overall Quality:").grid(row=len(metrics)+1, column=0, sticky="w", pady=2)
        quality_label = ttk.Label(metrics_frame, text="--", font=('TkDefaultFont', 12, 'bold'))
        quality_label.grid(row=len(metrics)+1, column=1, sticky="w", pady=2)
        
        # Issues list
        issues_frame = ttk.LabelFrame(main_container, text="Quality Issues & Tips", padding="10")
        issues_frame.grid(row=1, column=0, columnspan=2, sticky="ew", padx=5, pady=5)
        issues_text = tk.Text(issues_frame, height=4, wrap=tk.WORD, width=80)
        issues_text.pack(fill=tk.BOTH, expand=True)
        
        # Status frame
        status_frame = ttk.Frame(main_container)
        status_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=5)
        
        status_label = ttk.Label(status_frame, text="Press 'C' to capture, 'Q' to quit")
        status_label.pack(side=tk.LEFT)
        
        # Control frame
        control_frame = ttk.Frame(main_container)
        control_frame.grid(row=3, column=0, columnspan=2, sticky="ew", pady=5)
        
        # Initialize webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.show_message("Error", "Could not open webcam", "error")
            capture_window.destroy()
            return
        
        # Set webcam resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Store current frame and face detection state
        current_frame = None
        face_detected = False
        face_box = None
        capture_running = True
        last_quality_update = 0
        quality_update_interval = 0.5  # Update quality metrics every 0.5 seconds
        
        def update_metrics(quality_score, issues, metrics):
            """Update the quality metrics display"""
            # Update individual metrics
            for metric, value in metrics.items():
                if metric in metric_labels:
                    metric_labels[metric].config(text=f"{value:.0f}%")
            
            # Update overall quality
            quality_label.config(text=f"{quality_score:.0f}%")
            
            # Update issues text
            issues_text.delete(1.0, tk.END)
            if issues:
                issues_text.insert(tk.END, "\n".join(issues))
            else:
                issues_text.insert(tk.END, "No quality issues detected")
            
            # Set text color based on quality
            if quality_score >= 80:
                quality_label.config(foreground="green")
            elif quality_score >= 60:
                quality_label.config(foreground="orange")
            else:
                quality_label.config(foreground="red")
        
        def update_preview():
            nonlocal current_frame, face_detected, face_box, last_quality_update
            
            # Define metrics list and initialize quality_score at the start of the function
            metrics = ['sharpness', 'size', 'brightness', 'contrast', 'angle', 'occlusion']
            quality_score = 0  # Initialize quality_score
            
            if not capture_running:
                return
            
            ret, frame = cap.read()
            if not ret:
                return
            
            # Mirror the frame horizontally
            frame = cv2.flip(frame, 1)
            current_frame = frame.copy()
            
            # Convert to RGB for display
            display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            try:
                # Convert frame to JPEG bytes
                _, img_encoded = cv2.imencode('.jpg', frame)
                image_bytes = img_encoded.tobytes()
                
                # Detect faces using AWS Rekognition
                response = self.analyzer.rekognition.detect_faces(
                    Image={'Bytes': image_bytes},
                    Attributes=['ALL']
                )
                
                face_detected = len(response['FaceDetails']) > 0
                
                # Draw face boxes and quality indicators
                for face in response['FaceDetails']:
                    box = face['BoundingBox']
                    confidence = face['Confidence']
                    
                    # Convert normalized coordinates to pixel coordinates
                    height, width = frame.shape[:2]
                    left = int(box['Left'] * width)
                    top = int(box['Top'] * height)
                    right = left + int(box['Width'] * width)
                    bottom = top + int(box['Height'] * height)
                    
                    # Store face box for capture
                    face_box = (left, top, right, bottom)
                    
                    # Update quality metrics periodically
                    current_time = time.time()
                    if current_time - last_quality_update >= quality_update_interval:
                        try:
                            face_region = frame[top:bottom, left:right]
                            quality_score, issues, metrics_dict = self.analyzer.assess_face_quality(face_region)
                            update_metrics(quality_score, issues, metrics_dict)
                            last_quality_update = current_time
                        except Exception as e:
                            print(f"Error assessing face quality: {e}")
                            quality_score = 0
                            update_metrics(0, ["Error assessing face quality"], {k: 0 for k in metrics})
                    
                    # Draw box with color based on quality
                    color = (0, 255, 0) if quality_score >= 60 else (0, 165, 255)
                    cv2.rectangle(display_frame, (left, top), (right, bottom), color, 2)
                    
                    # Draw quality score
                    cv2.putText(display_frame, f"Quality: {quality_score:.0f}%",
                              (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
            except Exception as e:
                print(f"Face detection error: {e}")
                face_detected = False
                face_box = None
                quality_score = 0
                update_metrics(0, ["Error detecting face"], {k: 0 for k in metrics})
            
            # Convert to PhotoImage
            img = Image.fromarray(display_frame)
            img_tk = ImageTk.PhotoImage(image=img)
            
            # Update canvas
            preview_canvas.create_image(320, 240, image=img_tk, anchor=tk.CENTER)
            preview_canvas.image = img_tk  # Keep reference
            
            # Schedule next update
            capture_window.after(30, update_preview)
        
        def capture_face():
            if not face_detected or face_box is None:
                self.show_message("No Face", "No face detected in frame", "warning")
                return
            
            # Extract face region
            left, top, right, bottom = face_box
            face_region = current_frame[top:bottom, left:right]
            
            # Assess quality
            quality_score, issues, metrics = self.analyzer.assess_face_quality(face_region)
            if quality_score < 60:
                if not messagebox.askyesno("Low Quality", 
                    f"Face quality is low ({quality_score:.0f}%).\n\nIssues:\n" + 
                    "\n".join(issues) + "\n\nDo you want to capture anyway?"):
                    return
            
            # Save face image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"face_{timestamp}_qual{quality_score:.0f}.jpg"
            filepath = os.path.join(self.analyzer.unrecognized_dir, filename)
            
            try:
                cv2.imwrite(filepath, face_region)
                
                # Ask if user wants to index the face immediately
                if messagebox.askyesno("Success", 
                    "Face captured successfully! Would you like to index it now?"):
                    # Create name entry dialog
                    name_dialog = tk.Toplevel(capture_window)
                    name_dialog.title("Enter Person Name")
                    name_dialog.geometry("300x150")
                    self.center_window(name_dialog)
                    
                    ttk.Label(name_dialog, text="Enter name for this face:").pack(pady=10)
                    name_entry = ttk.Entry(name_dialog, width=30)
                    name_entry.pack(pady=10)
                    name_entry.focus()
                    
                    def do_index():
                        name = name_entry.get().strip()
                        if not name:
                            self.show_message("Error", "Name cannot be empty", "error")
                            return
                        
                        if self.analyzer.index_face(filepath, name):
                            self.show_message("Success", f"Face indexed as {name}")
                            self.update_collection_info()
                            name_dialog.destroy()
                        else:
                            self.show_message("Error", "Failed to index face", "error")
                    
                    ttk.Button(name_dialog, text="Index", command=do_index).pack(pady=10)
                    name_entry.bind('<Return>', lambda e: do_index())
                else:
                    self.show_message("Success", "Face saved to unrecognized faces folder")
            
            except Exception as e:
                self.show_message("Error", f"Failed to save face: {e}", "error")
        
        def on_closing():
            nonlocal capture_running
            capture_running = False
            cap.release()
            capture_window.destroy()
        
        # Bind keyboard shortcuts
        capture_window.bind('space', lambda e: capture_face())
        capture_window.bind('q', lambda e: on_closing())
        capture_window.protocol("WM_DELETE_WINDOW", on_closing)
        
        # Add control buttons
        ttk.Button(control_frame, text="Capture (Space)", 
                  command=capture_face).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Quit (Q)", 
                  command=on_closing).pack(side=tk.LEFT, padx=5)
        
        # Configure grid weights
        capture_window.grid_columnconfigure(0, weight=1)
        capture_window.grid_rowconfigure(0, weight=1)
        main_container.grid_columnconfigure(0, weight=3)
        main_container.grid_columnconfigure(1, weight=1)
        main_container.grid_rowconfigure(0, weight=3)
        main_container.grid_rowconfigure(1, weight=1)
        
        # Start preview
        update_preview()

    def register_faces(self):
        """Launch register faces from folder dialog with GUI interface"""
        # Select folder
        folder_path = filedialog.askdirectory(title="Select Folder with Face Images")
        if not folder_path:
            return
        
        # Get all image files from the folder
        image_files = []
        for ext in ('.jpg', '.jpeg', '.png'):
            image_files.extend([f for f in os.listdir(folder_path) if f.lower().endswith(ext)])
        
        if not image_files:
            self.show_message("No Images", "No image files found in the selected folder!", "warning")
            return
        
        # Create registration window
        reg_window = tk.Toplevel(self.root)
        reg_window.title("Register Faces from Folder")
        reg_window.geometry("1200x800")
        self.center_window(reg_window)
        
        # Preview frame
        preview_frame = ttk.LabelFrame(reg_window, text="Face Preview", padding="10")
        preview_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        # Preview canvas with scrollbar
        canvas_frame = ttk.Frame(preview_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        canvas = tk.Canvas(canvas_frame, bg='white')
        scrollbar = ttk.Scrollbar(canvas_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Options frame
        options_frame = ttk.LabelFrame(reg_window, text="Registration Options", padding="10")
        options_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        
        # Registration mode
        mode_var = tk.StringVar(value="same")
        ttk.Label(options_frame, text="Registration Mode:").pack(anchor="w", pady=(0, 5))
        ttk.Radiobutton(options_frame, text="All faces belong to same person", 
                        variable=mode_var, value="same").pack(anchor="w")
        ttk.Radiobutton(options_frame, text="Select faces for each person", 
                        variable=mode_var, value="select").pack(anchor="w")
        
        # Name entry frame
        name_frame = ttk.LabelFrame(options_frame, text="Person Name", padding="10")
        name_frame.pack(fill="x", pady=10)
        
        name_var = tk.StringVar()
        name_entry = ttk.Entry(name_frame, textvariable=name_var, width=30)
        name_entry.pack(fill="x", pady=5)
        
        # Help text
        help_label = ttk.Label(name_frame, 
                              text="Enter name for selected faces",
                              wraplength=200)
        help_label.pack(fill="x", pady=5)
        
        # Status frame
        status_frame = ttk.LabelFrame(reg_window, text="Status", padding="10")
        status_frame.grid(row=1, column=0, columnspan=2, sticky="ew", padx=10, pady=10)
        
        status_text = tk.Text(status_frame, height=6, width=80, wrap=tk.WORD)
        status_text.pack(fill="both", expand=True)
        
        # Store photo references and selection state
        photo_references = []
        selected_faces = set()
        
        def update_preview():
            # Clear previous previews
            for widget in scrollable_frame.winfo_children():
                widget.destroy()
            photo_references.clear()
            
            # Create grid of face previews
            row = 0
            col = 0
            max_cols = 4
            
            for i, image_file in enumerate(image_files):
                image_path = os.path.join(folder_path, image_file)
                try:
                    # Load and resize image
                    img = cv2.imread(image_path)
                    if img is None:
                        continue
                        
                    # Convert to RGB
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    # Resize maintaining aspect ratio
                    max_size = 200
                    h, w = img.shape[:2]
                    scale = min(max_size/w, max_size/h)
                    new_size = (int(w*scale), int(h*scale))
                    img = cv2.resize(img, new_size)
                    
                    # Convert to PhotoImage
                    photo = ImageTk.PhotoImage(image=Image.fromarray(img))
                    photo_references.append(photo)
                    
                    # Create frame for this face
                    face_frame = ttk.Frame(scrollable_frame)
                    face_frame.grid(row=row, column=col, padx=5, pady=5)
                    
                    # Add selection checkbox
                    selected = image_path in selected_faces
                    var = tk.BooleanVar(value=selected)
                    
                    def on_select(path=image_path, var=var):
                        if var.get():
                            selected_faces.add(path)
                        else:
                            selected_faces.discard(path)
                        update_register_button()
                    
                    checkbox = ttk.Checkbutton(face_frame, variable=var, 
                                             command=lambda p=image_path, v=var: on_select(p, v))
                    checkbox.pack()
                    
                    # Add image
                    img_label = ttk.Label(face_frame, image=photo)
                    img_label.pack()
                    
                    # Add filename
                    ttk.Label(face_frame, text=os.path.basename(image_file),
                             wraplength=200).pack()
                    
                    # Update grid position
                    col += 1
                    if col >= max_cols:
                        col = 0
                        row += 1
                        
                except Exception as e:
                    print(f"Error loading {image_file}: {e}")
        
        def update_register_button():
            # Enable/disable register button based on selection and name
            if mode_var.get() == "same":
                register_btn.config(state=tk.NORMAL if name_var.get().strip() else tk.DISABLED)
            else:
                register_btn.config(state=tk.NORMAL if selected_faces and name_var.get().strip() else tk.DISABLED)
        
        def update_help(*args):
            if mode_var.get() == "same":
                help_label.config(text="Enter the name for all faces")
                # Clear selection
                selected_faces.clear()
                update_preview()
            else:
                help_label.config(text="Select faces and enter name for the selected person")
            update_register_button()
        
        # Update when mode changes
        mode_var.trace("w", update_help)
        name_var.trace("w", lambda *args: update_register_button())
        
        def register_faces():
            mode = mode_var.get()
            name = name_var.get().strip()
            
            if not name:
                self.show_message("Error", "Please enter a name", "error")
                return
            
            # Process images
            successful = 0
            failed = 0
            skipped = 0
            
            status_text.delete(1.0, tk.END)
            status_text.insert(tk.END, "Processing faces...\n\n")
            reg_window.update()
            
            # Get list of images to process
            if mode == "same":
                images_to_process = [os.path.join(folder_path, f) for f in image_files]
            else:
                images_to_process = list(selected_faces)
            
            for image_path in images_to_process:
                # Check if person exists
                if name in self.analyzer.face_mapping:
                    if not messagebox.askyesno("Person Exists",
                        f"{name} already exists in collection.\n"
                        "Would you like to add these faces to this person?"):
                        skipped += 1
                        status_text.insert(tk.END, f"Skipped {os.path.basename(image_path)} (person exists)\n")
                        continue
                
                # Try to index the face
                if self.analyzer.index_face(image_path, name):
                    successful += 1
                    status_text.insert(tk.END, f"Added {os.path.basename(image_path)} as {name}\n")
                else:
                    failed += 1
                    status_text.insert(tk.END, f"Failed to add {os.path.basename(image_path)}\n")
                
                reg_window.update()
            
            # Show summary
            status_text.insert(tk.END, f"\nRegistration complete!\n")
            status_text.insert(tk.END, f"Successfully registered: {successful}\n")
            status_text.insert(tk.END, f"Failed to register: {failed}\n")
            status_text.insert(tk.END, f"Skipped: {skipped}\n")
            
            # Update collection info
            self.update_collection_info()
            
            # Clear selection and name for next person
            if mode == "select":
                selected_faces.clear()
                name_var.set("")
                update_preview()
                update_register_button()
            else:
                # Enable close button if in same person mode
                close_btn.config(state=tk.NORMAL)
        
        # Action buttons
        btn_frame = ttk.Frame(reg_window)
        btn_frame.grid(row=2, column=0, columnspan=2, pady=10)
        
        register_btn = ttk.Button(btn_frame, text="Register Faces", 
                                 command=register_faces,
                                 state=tk.DISABLED)
        register_btn.pack(side=tk.LEFT, padx=5)
        
        close_btn = ttk.Button(btn_frame, text="Close", 
                              command=reg_window.destroy,
                              state=tk.DISABLED)
        close_btn.pack(side=tk.LEFT, padx=5)
        
        # Configure grid weights
        reg_window.grid_columnconfigure(0, weight=2)
        reg_window.grid_columnconfigure(1, weight=1)
        reg_window.grid_rowconfigure(0, weight=1)
        
        # Initial preview
        update_preview()
        update_help()  # Initial help text

    def update_collection_info(self):
        """Update the collection information display"""
        try:
            # Calculate total faces and unique people
            total_faces = sum(len(data['face_ids']) for data in self.analyzer.face_mapping.values())
            unique_people = len(self.analyzer.face_mapping)
            
            # Get the most recent update time
            last_update = max(
                (data.get('last_updated', '') for data in self.analyzer.face_mapping.values()),
                default='Never'
            )
            
            # Update the labels
            self.total_faces_label.config(text=str(total_faces))
            self.unique_people_label.config(text=str(unique_people))
            self.last_update_label.config(text=last_update)
            
            # Update status
            self.status_label.config(text="Collection info updated")
            
        except Exception as e:
            self.status_label.config(text=f"Error updating info: {str(e)}")
            print(f"Error updating collection info: {e}")

    def show_message(self, title, message, message_type="info"):
        """Show a message dialog with appropriate styling
        
        Args:
            title (str): The title of the message
            message (str): The message to display
            message_type (str): Type of message - "info", "warning", "error", or "success"
        """
        # Configure icon based on message type
        icon = {
            "info": "ℹ️",
            "warning": "⚠️",
            "error": "❌",
            "success": "✅"
        }.get(message_type.lower(), "ℹ️")
        
        # Configure background color based on message type
        bg_color = {
            "info": "#e3f2fd",      # Light blue
            "warning": "#fff3e0",    # Light orange
            "error": "#ffebee",      # Light red
            "success": "#e8f5e9"     # Light green
        }.get(message_type.lower(), "#e3f2fd")
        
        # Create message window
        msg_window = tk.Toplevel(self.root)
        msg_window.title(title)
        msg_window.geometry("400x200")
        self.center_window(msg_window)
        
        # Make window modal
        msg_window.transient(self.root)
        msg_window.grab_set()
        
        # Main frame with padding
        main_frame = ttk.Frame(msg_window, padding="20", style="Message.TFrame")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Configure style for message frame
        style = ttk.Style()
        style.configure("Message.TFrame", background=bg_color)
        
        # Icon and title
        title_frame = ttk.Frame(main_frame)
        title_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(title_frame, 
                 text=f"{icon} {title}",
                 style="Message.TLabel",
                 font=("Segoe UI", 12, "bold")).pack(side=tk.LEFT)
        
        # Message text
        message_text = tk.Text(main_frame, 
                             wrap=tk.WORD,
                             height=4,
                             font=("Segoe UI", 10),
                             relief=tk.FLAT,
                             background=bg_color)
        message_text.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        message_text.insert(tk.END, message)
        message_text.config(state=tk.DISABLED)
        
        # OK button
        ttk.Button(main_frame,
                  text="OK",
                  command=msg_window.destroy).pack(pady=(10, 0))
        
        # Bind Enter key to close window
        msg_window.bind('<Return>', lambda e: msg_window.destroy())
        
        # Set focus to window
        msg_window.focus_set()
        
        # Wait for window to be closed
        self.root.wait_window(msg_window)

    def sync_with_aws(self):
        """Sync the local collection with AWS Rekognition"""
        try:
            # Show syncing message
            self.status_label.config(text="Syncing with AWS...")
            self.root.update()
            
            # Perform sync
            result = self.analyzer.sync_collection()
            
            if result['success']:
                # Update collection info
                self.update_collection_info()
                
                # Show success message with details
                message = (
                    f"Successfully synced with AWS!\n\n"
                    f"Total People: {result.get('total_people', 0)}\n"
                    f"Total Faces: {result.get('total_faces', 0)}\n"
                    f"AWS Faces: {result.get('aws_faces', 0)}\n\n"
                    f"Changes made:\n"
                )
                
                # Add change details if available
                if 'added' in result:
                    message += f"- Added: {result['added']}\n"
                if 'removed' in result:
                    message += f"- Removed: {result['removed']}\n"
                if 'updated' in result:
                    message += f"- Updated: {result['updated']}\n"
                
                self.show_message("Sync Complete", message, "info")
            else:
                self.show_message(
                    "Sync Failed",
                    f"Error syncing with AWS: {result.get('error', 'Unknown error')}",
                    "error"
                )
                
        except Exception as e:
            self.show_message(
                "Sync Error",
                f"Unexpected error during sync: {str(e)}",
                "error"
            )
        finally:
            self.status_label.config(text="Ready")

    def _delete_all_users(self, parent_window):
        """Delete all users from the collection"""
        if messagebox.askyesno(
            "Delete All Users",
            "This will delete ALL users and faces from the collection.\n"
            "This action cannot be undone.\n\n"
            "Are you sure you want to proceed?"):
            try:
                # Get all face IDs
                all_face_ids = []
                for person_data in self.analyzer.face_mapping.values():
                    if 'face_ids' in person_data:
                        all_face_ids.extend(person_data['face_ids'])
                
                # Delete all face IDs from AWS collection
                if all_face_ids:
                    try:
                        self.analyzer.rekognition.delete_faces(
                            CollectionId=self.analyzer.collection_id,
                            FaceIds=all_face_ids
                        )
                    except Exception as e:
                        print(f"Warning: Error deleting faces from AWS: {e}")
                
                # Remove all image files
                for person_data in self.analyzer.face_mapping.values():
                    # Handle both old and new format of face paths
                    if 'face_paths' in person_data:
                        # New format: face_paths dictionary
                        for face_id, image_path in person_data['face_paths'].items():
                            if image_path and os.path.exists(image_path):
                                try:
                                    os.remove(image_path)
                                except Exception as e:
                                    print(f"Warning: Could not delete file {image_path}: {e}")
                    elif 'images' in person_data:
                        # Old format: images list
                        for image_path in person_data['images']:
                            if image_path and os.path.exists(image_path):
                                try:
                                    os.remove(image_path)
                                except Exception as e:
                                    print(f"Warning: Could not delete file {image_path}: {e}")
                
                # Clear face mapping
                self.analyzer.face_mapping.clear()
                
                # Save changes
                self.analyzer.save_face_mapping()
                
                # Update display
                self.update_collection_info()
                
                # Close collection window
                parent_window.destroy()
                
                self.show_message("Success", "Successfully deleted all users from the collection")
                
            except Exception as e:
                self.show_message(
                    "Error", 
                    f"Error deleting all users: {str(e)}\n\n"
                    "Some files may have been deleted. Please check the collection status.",
                    "error"
                )

def main():
    root = tk.Tk()
    root.title("Face Recognition System")
    analyzer = VideoAnalyzer()
    gui = CollectionManagerGUI(root, analyzer)
    gui.run()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)