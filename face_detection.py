import boto3
import cv2
import time
import os
import sys  # Add sys import
import numpy as np
from datetime import datetime
import json
from sklearn.metrics.pairwise import cosine_similarity
import face_recognition  # You'll need to install this: pip install face_recognition
import uuid  # Add UUID import
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import distance
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
        
        # Create directories
        self.unrecognized_dir = "unrecognized_faces"
        self.indexed_dir = "indexed_faces"  # For faces that have been added to collection
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
        self.face_color = (0, 255, 0)  # Green for face boxes
        self.unrecognized_color = (0, 165, 255)  # Orange for unrecognized faces
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.7
        self.font_thickness = 2
        self.box_thickness = 2
        
        # Processing control
        self.frame_skip = 3  # Process every 3rd frame for better detection
        self.frame_counter = 0
        self.last_results = {
            'faces': []
        }
        self.tracked_faces = {}  # To maintain face tracking between frames
        self.tracking_counter = 0
        self.total_faces_detected = 0  # Track total faces detected
        self.recognized_faces = set()  # Track unique recognized faces
        self.unrecognized_faces = []  # Track unrecognized faces for review
        
        # Face comparison settings
        self.face_similarity_threshold = 0.6  # Threshold for considering faces as the same person
        self.known_face_encodings = {}  # Cache for face encodings
        
        # Add new attributes for quality assessment and performance tracking
        self.quality_metrics = {
            'blur_threshold': 100,  # Laplacian variance threshold
            'min_face_size': 100,   # Minimum face size in pixels
            'max_face_size': 1000,  # Maximum face size in pixels
            'min_confidence': 90,   # Minimum detection confidence
            'max_angle': 30,        # Maximum face angle in degrees
            'min_lighting': 40,     # Minimum average brightness
            'max_lighting': 200     # Maximum average brightness
        }
        
        # Performance tracking
        self.performance_metrics = {
            'total_frames_processed': 0,
            'total_faces_detected': 0,
            'total_faces_recognized': 0,
            'false_positives': 0,
            'false_negatives': 0,
            'recognition_times': [],
            'confidence_scores': [],
            'quality_scores': []
        }
        
        # Historical data
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
        
        # Load historical data if exists
        self.load_historical_data()
        
        # Add new attributes for collection management
        self.collection_settings = {
            'backup_dir': 'collection_backups',
            'export_dir': 'collection_exports',
            'version': '1.0'
        }
        
        # Add video processing settings
        self.video_settings = {
            'batch_size': 5,  # Number of videos to process in parallel
            'default_threshold': 80,  # Default recognition threshold
            'min_confidence': 90,  # Minimum confidence for face detection
            'frame_skip': 3,  # Process every Nth frame
            'output_formats': ['mp4', 'avi', 'mov'],
            'report_formats': ['csv', 'json', 'xlsx']
        }
        
        # Add new attributes for face recognition settings
        self.recognition_settings = {
            'default_threshold': 80,  # Default recognition threshold
            'min_confidence': 90,  # Minimum confidence for face detection
            'max_faces': 5,  # Maximum number of faces to detect per frame
            'tracking_enabled': True,  # Enable face tracking
            'quality_filtering': True,  # Enable quality filtering
            'batch_size': 5,  # Number of videos to process in parallel
            'frame_skip': 3,  # Process every Nth frame
            'output_formats': ['mp4', 'avi', 'mov'],
            'report_formats': ['csv', 'json', 'xlsx']
        }
        
        # Add per-person recognition settings
        self.person_settings = defaultdict(lambda: {
            'recognition_threshold': self.recognition_settings['default_threshold'],
            'min_confidence': self.recognition_settings['min_confidence'],
            'priority': 1,  # Higher number = higher priority
            'enabled': True
        })
        
        # Create necessary directories
        os.makedirs(self.collection_settings['backup_dir'], exist_ok=True)
        os.makedirs(self.collection_settings['export_dir'], exist_ok=True)
        
        # Load person settings if exists
        self.load_person_settings()
        
        # Add new attributes for optimized face tracking
        self.known_faces_cache = {}  # Cache for recognized faces
        self.face_tracking_history = {}  # Track face positions and recognition status
        self.tracking_confidence_threshold = 0.95  # Threshold for considering a face as "known"
        self.tracking_iou_threshold = 0.3  # IOU threshold for tracking
        self.max_tracking_frames = 30  # Maximum frames to track without re-recognition
        self.tracking_frame_counter = 0  # Counter for tracking frames

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
        """Index a face image into the AWS Rekognition collection with UUID"""
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
            
            # Create a unique external ID using just the UUID
            external_id = self.face_mapping[base_name]['uuid']
            
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
                
                # Move the image to indexed directory with UUID in filename
                indexed_filename = f"{base_name}_{self.face_mapping[base_name]['uuid']}_{len(self.face_mapping[base_name]['images'])}.jpg"
                indexed_path = os.path.join(self.indexed_dir, indexed_filename)
                
                # Only move and add image if not already present
                if not any(img == indexed_path for img in self.face_mapping[base_name]['images']):
                    os.rename(image_path, indexed_path)
                    self.face_mapping[base_name]['images'].append(indexed_path)
                
                self.face_mapping[base_name]['last_updated'] = datetime.now().isoformat()
                
                # Clean up any duplicate entries
                self.cleanup_duplicate_entries()
                
                self.save_face_mapping()
                print(f"✅ Successfully indexed face for {base_name} (UUID: {self.face_mapping[base_name]['uuid']})")
                return True
            else:
                print(f"❌ Failed to index face in {image_path}")
                return False
                
        except Exception as e:
            print(f"❌ Error indexing face: {e}")
            return False

    def list_unrecognized_faces(self):
        """List all unrecognized faces available for indexing"""
        faces = []
        for filename in os.listdir(self.unrecognized_dir):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                path = os.path.join(self.unrecognized_dir, filename)
                faces.append({
                    'path': path,
                    'filename': filename,
                    'timestamp': os.path.getmtime(path)
                })
        return sorted(faces, key=lambda x: x['timestamp'], reverse=True)

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

    def index_unrecognized_faces(self):
        """Interactive method to index unrecognized faces"""
        faces = self.list_unrecognized_faces()
        
        if not faces:
            print("No unrecognized faces found!")
            return
        
        print("\n=== Unrecognized Faces ===")
        for i, face in enumerate(faces, 1):
            print(f"\n{i}. {face['filename']}")
            print(f"   Path: {face['path']}")
            print(f"   Detected: {datetime.fromtimestamp(face['timestamp']).strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Display the face image
            img = cv2.imread(face['path'])
            if img is not None:
                # Resize if too large
                max_height = 300
                if img.shape[0] > max_height:
                    scale = max_height / img.shape[0]
                    img = cv2.resize(img, None, fx=scale, fy=scale)
                cv2.imshow("Face to Index (Press any key to continue)", img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            
            # Check if face might already be in collection
            existing_matches = self.check_existing_face(face['path'])
            if existing_matches:
                print("\n⚠️ This face might already be in the collection!")
                print("Possible matches:")
                for match in existing_matches:
                    print(f"  • {match['name']} ({match['similarity']:.1f}% similarity)")
                
                while True:
                    action = input("\nWhat would you like to do?\n"
                                 "1. Add as new person anyway\n"
                                 "2. Add to existing person\n"
                                 "3. Skip this face\n"
                                 "4. Delete this face\n"
                                 "5. Exit indexing\n"
                                 "Enter choice (1-5): ").strip()
                    
                    if action == '1':
                        person_name = input("Enter new person's name: ").strip()
                        if person_name:
                            if self.index_face(face['path'], person_name):
                                print(f"✅ Face indexed as new person: {person_name}")
                            break
                        else:
                            print("❌ Name cannot be empty!")
                    elif action == '2':
                        print("\nSelect person to add to:")
                        for idx, match in enumerate(existing_matches, 1):
                            print(f"{idx}. {match['name']} ({match['similarity']:.1f}%)")
                        try:
                            choice = int(input("Enter number (1-{}): ".format(len(existing_matches))))
                            if 1 <= choice <= len(existing_matches):
                                person_name = existing_matches[choice-1]['name']
                                if self.index_face(face['path'], person_name):
                                    print(f"✅ Face added to existing person: {person_name}")
                                break
                            else:
                                print("❌ Invalid choice!")
                        except ValueError:
                            print("❌ Please enter a valid number!")
                    elif action == '3':
                        print("Skipping this face...")
                        break
                    elif action == '4':
                        try:
                            os.remove(face['path'])
                            print("✅ Face deleted")
                            break
                        except Exception as e:
                            print(f"❌ Error deleting face: {e}")
                    elif action == '5':
                        print("Exiting face indexing...")
                        return
                    else:
                        print("❌ Invalid choice, please try again")
            else:
                # No existing matches found, proceed with normal indexing
                while True:
                    action = input("\nWhat would you like to do?\n"
                                 "1. Index this face\n"
                                 "2. Skip this face\n"
                                 "3. Delete this face\n"
                                 "4. Exit indexing\n"
                                 "Enter choice (1-4): ").strip()
                    
                    if action == '1':
                        person_name = input("Enter person's name: ").strip()
                        if person_name:
                            if self.index_face(face['path'], person_name):
                                print(f"✅ Face indexed as {person_name}")
                            break
                        else:
                            print("❌ Name cannot be empty!")
                    elif action == '2':
                        print("Skipping this face...")
                        break
                    elif action == '3':
                        try:
                            os.remove(face['path'])
                            print("✅ Face deleted")
                            break
                        except Exception as e:
                            print(f"❌ Error deleting face: {e}")
                    elif action == '4':
                        print("Exiting face indexing...")
                        return
                    else:
                        print("❌ Invalid choice, please try again")
        
        print("\n=== Face Indexing Complete ===")
        print("Current collection status:")
        for person, data in self.face_mapping.items():
            print(f"\n{person}:")
            print(f"  • UUID: {data['uuid']}")
            print(f"  • Face IDs: {len(data['face_ids'])}")
            print(f"  • Images: {len(data['images'])}")
            print(f"  • Created: {data.get('created_at', 'Unknown')}")
            print(f"  • Last Updated: {data.get('last_updated', 'Unknown')}")

    def preprocess_frame(self, frame):
        """Enhance frame quality for better face detection"""
        if frame is None or frame.size == 0:
            return frame, None

        # Convert to grayscale for processing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply histogram equalization to improve contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        
        # Apply bilateral filter to reduce noise while preserving edges
        gray = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Enhance contrast
        alpha = 1.3  # Contrast control
        beta = 10    # Brightness control
        gray = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)
        
        # Sharpen the image
        kernel = np.array([[-1, -1, -1], 
                          [-1, 9, -1], 
                          [-1, -1, -1]])
        gray = cv2.filter2D(gray, -1, kernel)
        
        # Convert back to color for display
        enhanced = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        
        return enhanced, gray

    def save_unrecognized_face(self, frame, face_box, confidence):
        """Save unrecognized face for review with quality assessment"""
        height, width = frame.shape[:2]
        left = int(face_box['Left'] * width)
        top = int(face_box['Top'] * height)
        right = left + int(face_box['Width'] * width)
        bottom = top + int(face_box['Height'] * height)
        
        # Add margin around face
        margin = 20
        left = max(0, left - margin)
        top = max(0, top - margin)
        right = min(width, right + margin)
        bottom = min(height, bottom + margin)
        
        face_region = frame[top:bottom, left:right]
        if face_region.size == 0:
            return
        
        # Assess face quality before saving
        quality_score, issues = self.assess_face_quality(face_region)
        
        # Only save if quality is acceptable
        if quality_score >= 60:  # Minimum quality threshold
            # Save with timestamp, confidence, and quality score
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"face_{timestamp}_conf{confidence:.0f}_qual{quality_score:.0f}.jpg"
            filepath = os.path.join(self.unrecognized_dir, filename)
            cv2.imwrite(filepath, face_region)
            
            # Log quality issues
            if issues:
                print(f"⚠️ Face quality issues: {', '.join(issues)}")
                print(f"Quality score: {quality_score}/100")
        else:
            print(f"❌ Face quality too low ({quality_score}/100): {', '.join(issues)}")

    def process_video(self, video_path, progress_callback=None):
        """Process a video file with optional progress callback"""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = 0

        # Reset tracking and performance metrics for the new video
        self.tracked_faces = {}
        self.tracking_counter = 0
        self.performance_metrics = {
            'total_frames_processed': 0,
            'total_faces_detected': 0,
            'total_faces_recognized': 0,
            'false_positives': 0,
            'false_negatives': 0,
            'recognition_times': [],
            'confidence_scores': [],
            'quality_scores': []
        }
        self.recognized_faces = set() # Reset recognized faces set per video

        # Define and create output directory
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)

        # Create output file path within the output directory
        output_filename = os.path.basename(video_path).split('.')[0] + '_analyzed.mp4'
        output_path = os.path.join(output_dir, output_filename)

        # Define video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Process every nth frame (detection and recognition)
                if frame_count % self.frame_skip == 0:
                    self.analyze_frame(frame)

                # Update tracking for every frame (using last_results)
                self.update_tracked_faces(frame)

                # Draw detections on a copy of the frame to avoid modifying original
                display_frame = frame.copy()
                display_frame = self.draw_detections(display_frame)

                # Call progress callback if provided, passing the frame with detections
                if progress_callback:
                    progress_callback(display_frame, frame_count, total_frames)

                # Write frame to output video
                out.write(display_frame)

                frame_count += 1
                self.performance_metrics['total_frames_processed'] = frame_count # Update total frames processed

        finally:
                cap.release()
                out.release()
                # Save historical data after processing each video
                self.save_historical_data()

        return {
            'total_frames': total_frames,
            'processed_frames': frame_count,
            'faces_detected': self.performance_metrics['total_faces_detected'], # Report total detected across analyzed frames
            'faces_recognized': len(self.recognized_faces) # Report unique recognized faces across the video
        }

    def analyze_frame(self, frame):
        """Analyze a single frame with optimized recognition"""
        start_time = time.time()

        # Skip empty frames
        if frame is None or frame.size == 0:
            print("⚠️ Empty frame received")
            return {'faces': []}

        # Preprocess frame for better detection
        enhanced_frame, gray_frame = self.preprocess_frame(frame)

        # Convert frame to JPEG bytes
        try:
            _, img_encoded = cv2.imencode('.jpg', enhanced_frame)
            image_bytes = img_encoded.tobytes()
            # print(f"✅ Frame encoded successfully, size: {len(image_bytes)} bytes") # Reduce logging noise
        except Exception as e:
            print(f"⚠️ Frame encoding error: {e}")
            return {'faces': []}

        results = {
            'faces': []
        }

        try:
            # print("🔍 Attempting face detection with AWS Rekognition...") # Reduce logging noise
            # Face detection with improved parameters
            detect_response = self.rekognition.detect_faces(
                Image={'Bytes': image_bytes},
                Attributes=['DEFAULT', 'ALL']
            )

            # print(f"📊 Face detection response: {len(detect_response.get('FaceDetails', []))} faces detected") # Reduce logging noise

            # Update total faces detected (cumulative)
            self.performance_metrics['total_faces_detected'] += len(detect_response.get('FaceDetails', []))

            # Only perform recognition for new faces or when tracking is lost
            for face_detail in detect_response.get('FaceDetails', []):
                face_box = face_detail['BoundingBox']
                confidence = face_detail['Confidence']
                # print(f"👤 Detected Face - Confidence: {confidence:.1f}%, Box: {face_box}") # Reduce logging noise

                # Log details of the detected face (optional for debugging)
                # print(f"   Details: {face_detail}")

                height, width = frame.shape[:2]
                left = int(face_box['Left'] * width)
                top = int(face_box['Top'] * height)
                right = left + int(face_box['Width'] * width)
                bottom = top + int(face_box['Height'] * height)

                # Ensure the crop is within frame bounds
                left = max(0, left)
                top = max(0, top)
                right = min(width, right)
                bottom = min(height, bottom)

                if right <= left or bottom <= top:
                    continue

                face_region = frame[top:bottom, left:right]
                if face_region.size == 0:
                    continue

                # Attempt recognition for detected face
                _, face_encoded = cv2.imencode('.jpg', face_region)
                face_bytes = face_encoded.tobytes()

                try:
                    recognize_response = self.rekognition.search_faces_by_image(
                        CollectionId=self.collection_id,
                        Image={'Bytes': face_bytes},
                        MaxFaces=1,
                        FaceMatchThreshold=self.recognition_settings['default_threshold'] # Use configurable threshold
                    )

                    if recognize_response.get('FaceMatches'):
                        match = recognize_response['FaceMatches'][0]
                        user_id = match['Face']['ExternalImageId']
                        self.recognized_faces.add(user_id) # Track unique recognized faces
                        results['faces'].append({
                            'Face': {
                                'BoundingBox': face_box,
                                'ExternalImageId': user_id,
                                'Confidence': confidence # Use detection confidence
                            },
                            'Similarity': match['Similarity'],
                            'Recognized': True
                        })
                    else:
                        # No match found
                        # Optionally save unrecognized face if confidence is high enough
                        if confidence > self.recognition_settings['min_confidence']:
                            self.save_unrecognized_face(frame, face_box, confidence)
                        # Add unrecognized face to results for drawing and potential future indexing
                        results['faces'].append({
                            'Face': {
                                'BoundingBox': face_box,
                                'ExternalImageId': 'Unknown', # Label as Unknown
                                'Confidence': confidence
                            },
                            'Similarity': 0, # No similarity if unrecognized
                            'Recognized': False
                        })
                except Exception as e:
                    print(f"⚠️ Recognition error for detected face: {e}")
                    # Add unrecognized face to results even if recognition API fails
                    if confidence > self.recognition_settings['min_confidence']: # Still save if detection confidence is high
                         self.save_unrecognized_face(frame, face_box, confidence)
                    results['faces'].append({
                        'Face': {
                            'BoundingBox': face_box,
                            'ExternalImageId': 'Unknown', # Label as Unknown
                            'Confidence': confidence
                        },
                        'Similarity': 0, # No similarity if unrecognized
                        'Recognized': False
                    })

        except Exception as e:
            print(f"⚠️ Face detection error: {e}")
            print(f"Error type: {type(e).__name__}")
            if hasattr(e, 'response'):
                print(f"AWS Response: {e.response}")

        # Update performance metrics (only count recognized faces from this frame's results)
        recognized_in_frame = len([f for f in results.get('faces', []) if f.get('Recognized', False)])
        self.performance_metrics['total_faces_recognized'] += recognized_in_frame
        processing_time = time.time() - start_time
        self.update_performance_metrics(results, processing_time)

        # Store current frame's detection results for tracking in the next frame
        self.last_results = results

        return results

    def update_tracked_faces(self, frame):
        """Update tracked faces using IoU and detection results."""
        height, width = frame.shape[:2]
        current_detections_px = []
        
        # Prepare current frame's detection results in pixel coordinates
        if self.last_results.get('faces'):
            for face_match in self.last_results['faces']:
                box = face_match['Face']['BoundingBox']
                left = int(box['Left'] * width)
                top = int(box['Top'] * height)
                right = left + int(box['Width'] * width)
                bottom = top + int(box['Height'] * height)
                current_detections_px.append({
                    'box': (left, top, right, bottom),
                    'name': face_match['Face']['ExternalImageId'],
                    'confidence': face_match.get('Similarity', face_match['Face'].get('Confidence', 0)), # Use similarity if recognized, else detection confidence
                    'recognized': face_match.get('Recognized', False)
                })

        updated_tracks = {}
        matched_detection_indices = set()

        # Try to match existing tracks with current detections using IoU
        for track_id, track_data in list(self.tracked_faces.items()): # Iterate on a copy to allow modification
            best_iou = 0
            best_match_index = -1

            for i, detection in enumerate(current_detections_px):
                if i in matched_detection_indices:
                    continue

                iou = self.calculate_iou(detection['box'], track_data['box'])

                # Consider both IoU and potentially name match for better tracking stability
                # If a track was recognized, prioritize matching with recognized detections of the same name
                if track_data.get('recognized', False) and detection.get('recognized', False) and track_data['name'] == detection['name']:
                     if iou > best_iou:
                        best_iou = iou
                        best_match_index = i
                # Otherwise, match based purely on IoU above a threshold
                elif iou > self.tracking_iou_threshold:
                     if iou > best_iou:
                         best_iou = iou
                         best_match_index = i


            if best_match_index != -1:
                # Update existing track with new detection data
                matched_detection_indices.add(best_match_index)
                updated_tracks[track_id] = {
                    'box': current_detections_px[best_match_index]['box'],
                    'name': current_detections_px[best_match_index]['name'],
                    'confidence': current_detections_px[best_match_index]['confidence'],
                    'frames_since_update': 0, # Reset frame counter
                    'recognized': current_detections_px[best_match_index]['recognized']
                }
            else:
                # Increment frames since update for unmatched tracks
                track_data['frames_since_update'] += 1
                # Keep track for a few frames after losing detection, but mark as not recognized if it was
                if track_data['frames_since_update'] < self.max_tracking_frames:
                     updated_tracks[track_id] = track_data
                # If track was recognized and lost, mark as not recognized to potentially re-recognize later
                elif track_data.get('recognized', False):
                    track_data['recognized'] = False
                    updated_tracks[track_id] = track_data

        # Add new tracks for unmatched detections
        for i, detection in enumerate(current_detections_px):
            if i not in matched_detection_indices:
                new_id = self.tracking_counter
                updated_tracks[new_id] = {
                    'box': detection['box'],
                    'name': detection['name'],
                    'confidence': detection['confidence'],
                    'frames_since_update': 0,
                    'recognized': detection['recognized']
                }
                self.tracking_counter += 1

        self.tracked_faces = updated_tracks

    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union for two bounding boxes"""
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

        return intersection_area / union_area

    def draw_detections(self, frame):
        """Draw tracked face detections on the frame"""
        if frame is None or frame.size == 0:
            return frame

        # height, width = frame.shape[:2] # Not needed if using pixel coordinates from tracking

        # Draw tracked faces
        for track_id, track_data in self.tracked_faces.items():
            # Skip drawing if the person is unknown
            if track_data.get('name') == 'Unknown':
                continue

            box = track_data['box']
            left, top, right, bottom = box

            # Choose color based on recognition status
            color = (0, 255, 0) if track_data.get('recognized', False) else (0, 165, 255) # Green for recognized, Orange for unknown
            thickness = self.box_thickness

            cv2.rectangle(frame, (left, top), (right, bottom), color, thickness)

            # Draw name label background
            label_text = f"{track_data['name']} ({track_data['confidence']:.0f}%)"
            (text_width, text_height), _ = cv2.getTextSize(label_text, self.font, self.font_scale, self.font_thickness)

            # Ensure label background is within frame bounds
            label_bg_top = max(0, top - text_height - 10)
            label_bg_bottom = top
            label_bg_right = min(frame.shape[1], left + text_width + 10)

            cv2.rectangle(frame, (left, label_bg_top), (label_bg_right, label_bg_bottom), color, -1)

            # Draw name and confidence
            cv2.putText(frame, label_text,
                        (left + 5, top - 10), self.font, self.font_scale, (255, 255, 255), self.font_thickness)

            # Optional: Draw tracking ID for debugging
            # cv2.putText(frame, f"ID: {track_id}",
            #             (left + 5, bottom + 15), self.font, self.font_scale, color, self.font_thickness)

        # No longer drawing raw detections here, only tracked faces

        return frame

    def force_recheck_unrecognized_faces(self, threshold=70):
        """Force recheck all unrecognized faces with a lower threshold"""
        faces = self.list_unrecognized_faces()
        
        if not faces:
            print("No unrecognized faces found!")
            return
        
        print(f"\n=== Rechecking Unrecognized Faces (Threshold: {threshold}%) ===")
        found_matches = []
        
        for face in faces:
            print(f"\nChecking: {face['filename']}")
            matches = self.check_existing_face(face['path'], threshold=threshold)
            
            if matches:
                print("Found potential matches:")
                for match in matches:
                    print(f"  • {match['name']} ({match['similarity']:.1f}% similarity)")
                found_matches.append({
                    'face': face,
                    'matches': matches
                })
            else:
                print("No matches found even with lower threshold")
        
        if found_matches:
            print("\n=== Found Potential Matches ===")
            for i, data in enumerate(found_matches, 1):
                face = data['face']
                matches = data['matches']
                print(f"\n{i}. {face['filename']}")
                print("   Possible matches:")
                for match in matches:
                    print(f"      • {match['name']} ({match['similarity']:.1f}%)")
                
                # Display the face image
                img = cv2.imread(face['path'])
                if img is not None:
                    max_height = 300
                    if img.shape[0] > max_height:
                        scale = max_height / img.shape[0]
                        img = cv2.resize(img, None, fx=scale, fy=scale)
                    cv2.imshow("Face to Process (Press any key to continue)", img)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                
                while True:
                    action = input("\nWhat would you like to do?\n"
                                 "1. Add to existing person\n"
                                 "2. Skip this face\n"
                                 "3. Delete this face\n"
                                 "4. Exit rechecking\n"
                                 "Enter choice (1-4): ").strip()
                    
                    if action == '1':
                        print("\nSelect person to add to:")
                        for idx, match in enumerate(matches, 1):
                            print(f"{idx}. {match['name']} ({match['similarity']:.1f}%)")
                        try:
                            choice = int(input("Enter number (1-{}): ".format(len(matches))))
                            if 1 <= choice <= len(matches):
                                person_name = matches[choice-1]['name']
                                if self.index_face(face['path'], person_name):
                                    print(f"✅ Face added to existing person: {person_name}")
                                break
                            else:
                                print("❌ Invalid choice!")
                        except ValueError:
                            print("❌ Please enter a valid number!")
                    elif action == '2':
                        print("Skipping this face...")
                        break
                    elif action == '3':
                        try:
                            os.remove(face['path'])
                            print("✅ Face deleted")
                            break
                        except Exception as e:
                            print(f"❌ Error deleting face: {e}")
                    elif action == '4':
                        print("Exiting recheck...")
                        return
                    else:
                        print("❌ Invalid choice, please try again")
        else:
            print("\nNo matches found even with lower threshold. You might want to:")
            print("1. Try an even lower threshold")
            print("2. Check if the face is clearly visible")
            print("3. Add the face as a new entry to the collection")

    def get_face_encoding(self, image_path):
        """Get face encoding using face_recognition library"""
        try:
            # Load image
            image = face_recognition.load_image_file(image_path)
            
            # Get face locations
            face_locations = face_recognition.face_locations(image)
            if not face_locations:
                return None
            
            # Get face encodings
            face_encodings = face_recognition.face_encodings(image, face_locations)
            if not face_encodings:
                return None
            
            # Return the first face encoding
            return face_encodings[0]
            
        except Exception as e:
            print(f"⚠️ Error getting face encoding for {image_path}: {e}")
            return None

    def compare_faces(self, encoding1, encoding2):
        """Compare two face encodings using cosine similarity"""
        if encoding1 is None or encoding2 is None:
            return 0.0
        return cosine_similarity([encoding1], [encoding2])[0][0]

    def find_duplicate_faces(self):
        """Find and handle duplicate faces in unrecognized_faces directory"""
        print("\n=== Checking for Duplicate Faces ===")
        
        # Get all face images
        faces = self.list_unrecognized_faces()
        if not faces:
            print("No unrecognized faces found!")
            return
        
        # Get encodings for all faces
        print("Analyzing faces...")
        face_data = []
        for face in faces:
            encoding = self.get_face_encoding(face['path'])
            if encoding is not None:
                face_data.append({
                    'path': face['path'],
                    'encoding': encoding,
                    'timestamp': face['timestamp']
                })
        
        if not face_data:
            print("No valid faces found to compare!")
            return
        
        # Find duplicates
        duplicates = []
        processed = set()
        
        print("\nComparing faces...")
        for i, face1 in enumerate(face_data):
            if face1['path'] in processed:
                continue
                
            current_group = []
            
            for face2 in face_data[i+1:]:
                if face2['path'] in processed:
                    continue
                    
                similarity = self.compare_faces(face1['encoding'], face2['encoding'])
                if similarity > self.face_similarity_threshold:
                    if not current_group:
                        current_group.append(face1)
                    current_group.append(face2)
                    processed.add(face2['path'])
            
            if current_group:
                processed.add(face1['path'])
                duplicates.append(current_group)
        
        if not duplicates:
            print("No duplicate faces found!")
            return
        
        # Handle duplicates
        print(f"\nFound {len(duplicates)} groups of similar faces")
        for group_idx, group in enumerate(duplicates, 1):
            print(f"\nGroup {group_idx}:")
            
            # Show all faces in the group
            for face in group:
                print(f"\n  • {os.path.basename(face['path'])}")
                print(f"    Detected: {datetime.fromtimestamp(face['timestamp']).strftime('%Y-%m-%d %H:%M:%S')}")
                
                # Display the face
                img = cv2.imread(face['path'])
                if img is not None:
                    max_height = 300
                    if img.shape[0] > max_height:
                        scale = max_height / img.shape[0]
                        img = cv2.resize(img, None, fx=scale, fy=scale)
                    cv2.imshow("Similar Face (Press any key to continue)", img)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
            
            while True:
                action = input("\nWhat would you like to do with this group?\n"
                             "1. Keep the most recent face\n"
                             "2. Keep the highest quality face\n"
                             "3. Keep all faces\n"
                             "4. Delete all faces\n"
                             "5. Skip this group\n"
                             "Enter choice (1-5): ").strip()
                
                if action == '1':
                    # Keep the most recent face
                    most_recent = max(group, key=lambda x: x['timestamp'])
                    for face in group:
                        if face['path'] != most_recent['path']:
                            try:
                                os.remove(face['path'])
                                print(f"✅ Deleted: {os.path.basename(face['path'])}")
                            except Exception as e:
                                print(f"❌ Error deleting {face['path']}: {e}")
                    print(f"✅ Kept most recent face: {os.path.basename(most_recent['path'])}")
                    break
                    
                elif action == '2':
                    # Keep the highest quality face (based on image size as a simple metric)
                    highest_quality = max(group, key=lambda x: os.path.getsize(x['path']))
                    for face in group:
                        if face['path'] != highest_quality['path']:
                            try:
                                os.remove(face['path'])
                                print(f"✅ Deleted: {os.path.basename(face['path'])}")
                            except Exception as e:
                                print(f"❌ Error deleting {face['path']}: {e}")
                    print(f"✅ Kept highest quality face: {os.path.basename(highest_quality['path'])}")
                    break
                    
                elif action == '3':
                    print("Keeping all faces in this group")
                    break
                    
                elif action == '4':
                    # Delete all faces in the group
                    for face in group:
                        try:
                            os.remove(face['path'])
                            print(f"✅ Deleted: {os.path.basename(face['path'])}")
                        except Exception as e:
                            print(f"❌ Error deleting {face['path']}: {e}")
                    break
                    
                elif action == '5':
                    print("Skipping this group")
                    break
                    
                else:
                    print("❌ Invalid choice, please try again")

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

    def assess_face_quality(self, face_image):
        """Assess the quality of a face image"""
        if face_image is None or face_image.size == 0:
            return 0, "Invalid image"
        
        quality_score = 100
        issues = []
        
        # Check image size
        height, width = face_image.shape[:2]
        if height < self.quality_metrics['min_face_size'] or width < self.quality_metrics['min_face_size']:
            quality_score -= 20
            issues.append("Face too small")
        elif height > self.quality_metrics['max_face_size'] or width > self.quality_metrics['max_face_size']:
            quality_score -= 10
            issues.append("Face too large")
        
        # Check blur
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        if blur_score < self.quality_metrics['blur_threshold']:
            quality_score -= 30
            issues.append("Image too blurry")
        
        # Check lighting
        avg_brightness = np.mean(gray)
        if avg_brightness < self.quality_metrics['min_lighting']:
            quality_score -= 20
            issues.append("Image too dark")
        elif avg_brightness > self.quality_metrics['max_lighting']:
            quality_score -= 20
            issues.append("Image too bright")
        
        # Check face angle (using facial landmarks)
        try:
            face_landmarks = face_recognition.face_landmarks(face_image)
            if face_landmarks:
                # Calculate face angle using eye positions
                left_eye = np.mean(face_landmarks[0]['left_eye'], axis=0)
                right_eye = np.mean(face_landmarks[0]['right_eye'], axis=0)
                angle = np.degrees(np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]))
                if abs(angle) > self.quality_metrics['max_angle']:
                    quality_score -= 15
                    issues.append(f"Face angle too large ({angle:.1f}°)")
        except Exception:
            quality_score -= 10
            issues.append("Could not detect facial landmarks")
        
        return max(0, quality_score), issues

    def update_performance_metrics(self, frame_results, processing_time):
        """Update performance metrics with new frame results"""
        self.performance_metrics['total_frames_processed'] += 1
        
        # Update face detection metrics
        faces_detected = len(frame_results.get('faces', []))
        self.performance_metrics['total_faces_detected'] += faces_detected
        
        # Update recognition metrics
        recognized_faces = [f for f in frame_results.get('faces', []) if f.get('Recognized', False)]
        self.performance_metrics['total_faces_recognized'] += len(recognized_faces)
        
        # Update confidence scores
        for face in frame_results.get('faces', []):
            if 'Face' in face and 'Confidence' in face['Face']:
                self.performance_metrics['confidence_scores'].append(face['Face']['Confidence'])
        
        # Update recognition times
        self.performance_metrics['recognition_times'].append(processing_time)
        
        # Update daily stats with debug logging
        try:
            today = datetime.now().strftime('%Y-%m-%d')
            print(f"📅 Updating stats for date: {today}")
            print(f"📊 Current stats before update: {self.historical_data['daily_stats'][today]}")
            
            self.historical_data['daily_stats'][today]['faces_detected'] += faces_detected
            self.historical_data['daily_stats'][today]['faces_recognized'] += len(recognized_faces)
            
            # Calculate and update averages
            if self.performance_metrics['confidence_scores']:
                avg_confidence = np.mean(self.performance_metrics['confidence_scores'])
                self.historical_data['daily_stats'][today]['avg_confidence'] = avg_confidence
            
            print(f"📈 Updated stats: {self.historical_data['daily_stats'][today]}")
        except Exception as e:
            print(f"⚠️ Error updating daily stats: {e}")
            print(f"Current historical data structure: {self.historical_data}")
        
        # Save historical data periodically
        if self.performance_metrics['total_frames_processed'] % 100 == 0:
            try:
                self.save_historical_data()
                print("💾 Saved historical data")
            except Exception as e:
                print(f"⚠️ Error saving historical data: {e}")

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
                # Extract person name from external ID (format: person_name_uuid)
                person_name = external_id.split('_')[0]
                if person_name not in aws_face_map:
                    aws_face_map[person_name] = []
                aws_face_map[person_name].append(face_id)
            
            # Check for mismatches and sync
            changes_made = False
            
            # 1. Check for faces in AWS but not in local mapping
            for person_name, face_ids in aws_face_map.items():
                if person_name not in self.face_mapping:
                    print(f"⚠️ Found person in AWS but not locally: {person_name}")
                    # Create new entry in local mapping
                    self.face_mapping[person_name] = {
                        'uuid': str(uuid.uuid4()),
                        'face_ids': face_ids,
                        'images': [],  # We can't recover the original images
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
                        # Also remove corresponding images if they exist
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
            
            # Print summary
            print("\n=== Collection Summary ===")
            print(f"Total people: {len(self.face_mapping)}")
            total_faces = sum(len(data['face_ids']) for data in self.face_mapping.values())
            print(f"Total faces: {total_faces}")
            
        except Exception as e:
            print(f"❌ Error syncing with AWS collection: {e}")
            if hasattr(e, 'response'):
                print(f"AWS Response: {e.response}")
            raise  # Re-raise the exception to handle it in the calling code

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
        
        ttk.Button(left_column, text="Register Faces (R)", 
                  style="Info.TButton",
                  command=self.register_faces).pack(fill=tk.X, pady=2)
        
        # Right column buttons
        ttk.Button(right_column, text="Show Collection (S)", 
                  style="Info.TButton",
                  command=self.show_collection).pack(fill=tk.X, pady=2)
        
        ttk.Button(right_column, text="Cleanup Duplicates (D)", 
                  style="Warning.TButton",
                  command=self.cleanup_duplicates).pack(fill=tk.X, pady=2)
        
        ttk.Button(right_column, text="Index Faces (I)", 
                  style="Warning.TButton",
                  command=self.index_faces).pack(fill=tk.X, pady=2)
        
        # Add delete users button to right column
        ttk.Button(right_column, text="Delete Users (X)", 
                  style="Danger.TButton",
                  command=self.delete_users).pack(fill=tk.X, pady=2)
        
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
        self.root.bind('d', lambda e: self.cleanup_duplicates())
        self.root.bind('i', lambda e: self.index_faces())  # Changed from 'u' to 'i'
        self.root.bind('q', lambda e: self.root.quit())
        self.root.bind('x', lambda e: self.delete_users())
        
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
            "Capture Face (C)": "Capture a face from webcam for registration",
            "Register Faces (R)": "Register multiple faces from a folder",
            "Show Collection (S)": "View and manage the face collection",
            "Cleanup Duplicates (D)": "Find and remove duplicate faces",
            "Index Faces (I)": "Review and index unrecognized faces",  # Updated tooltip
            "Exit (Q)": "Close the application",
            "Delete Users (X)": "Delete specific users from the collection"
        }
        
        def create_tooltip(widget, text):
            def show_tooltip(event):
                tooltip = tk.Toplevel(self.root)
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
                
                label.bind("<Leave>", lambda e: hide_tooltip())
                tooltip.bind("<Leave>", lambda e: hide_tooltip())
            
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

    def index_faces(self):
        """Show face indexing dialog with similarity checking and auto-deletion of low-quality matches"""
        faces = self.analyzer.list_unrecognized_faces()
        if not faces:
            self.show_message("No Faces", "No unrecognized faces found!", "warning")
            return
        
        # Create indexing window
        index_window = tk.Toplevel(self.root)
        index_window.title("Index Faces")
        index_window.geometry("1200x800")
        self.center_window(index_window)
        
        # Face preview
        preview_frame = ttk.LabelFrame(index_window, text="Face Preview", padding="10")
        preview_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        # Preview canvas
        preview_canvas = tk.Canvas(preview_frame, width=400, height=400, bg='white')
        preview_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Face list
        list_frame = ttk.LabelFrame(index_window, text="Unrecognized Faces", padding="10")
        list_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        
        # Add scrollbar to list
        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        face_list = tk.Listbox(list_frame, height=20, width=40, yscrollcommand=scrollbar.set)
        face_list.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=face_list.yview)
        
        # Face details and matches
        details_frame = ttk.LabelFrame(index_window, text="Face Details & Matches", padding="10")
        details_frame.grid(row=0, column=2, sticky="nsew", padx=10, pady=10)
        
        # Details text
        details_text = tk.Text(details_frame, height=6, width=40, wrap=tk.WORD)
        details_text.pack(fill=tk.X, expand=True)
        
        # Matches text
        matches_text = tk.Text(details_frame, height=8, width=40, wrap=tk.WORD)
        matches_text.pack(fill=tk.X, expand=True, pady=(10, 0))
        
        # Store current preview image and matches
        current_preview = None
        current_matches = None
        current_face = None
        
        def show_preview(event):
            nonlocal current_preview, current_matches, current_face
            selection = face_list.curselection()
            if not selection:
                return
            
            face = faces[selection[0]]
            current_face = face
            
            # Load and display image
            img = cv2.imread(face['path'])
            if img is not None:
                # Convert to RGB for tkinter
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Resize image to fit canvas while maintaining aspect ratio
                canvas_width = preview_canvas.winfo_width()
                canvas_height = preview_canvas.winfo_height()
                
                scale = min(canvas_width/img.shape[1], canvas_height/img.shape[0])
                new_width = int(img.shape[1] * scale)
                new_height = int(img.shape[0] * scale)
                
                img = cv2.resize(img, (new_width, new_height))
                
                # Convert to PhotoImage
                img_tk = ImageTk.PhotoImage(image=Image.fromarray(img))
                
                # Update canvas
                preview_canvas.delete("all")
                preview_canvas.create_image(canvas_width//2, canvas_height//2, 
                                          image=img_tk, anchor=tk.CENTER)
                
                # Keep reference
                current_preview = img_tk
            
            # Update details
            details = f"Filename: {face['filename']}\n"
            details += f"Detected: {datetime.fromtimestamp(face['timestamp']).strftime('%Y-%m-%d %H:%M:%S')}\n"
            details += f"Path: {face['path']}"
            
            details_text.delete(1.0, tk.END)
            details_text.insert(tk.END, details)
            
            # Check for matches with lower threshold
            matches = self.analyzer.check_existing_face(face['path'], threshold=60)
            current_matches = matches
            
            # Update matches text and buttons
            matches_text.delete(1.0, tk.END)
            if matches:
                matches_text.insert(tk.END, "Found potential matches:\n\n")
                for match in matches:
                    matches_text.insert(tk.END, f"• {match['name']} ({match['similarity']:.1f}%)\n")
                create_match_buttons()
            else:
                matches_text.insert(tk.END, "No matches found even with 60% threshold.\nThis face will be deleted.")
                # Clear any existing match buttons
                for widget in details_frame.winfo_children():
                    if isinstance(widget, ttk.Frame):
                        widget.destroy()
                
                # Auto-delete face with no matches
                try:
                    os.remove(face['path'])
                    face_list.delete(selection[0])
                    faces.pop(selection[0])
                    preview_canvas.delete("all")
                    details_text.delete(1.0, tk.END)
                    matches_text.delete(1.0, tk.END)
                    self.show_message("Info", "Face deleted due to no matches found")
                except Exception as e:
                    self.show_message("Error", f"Error deleting face: {str(e)}", "error")
        
        face_list.bind('<<ListboxSelect>>', show_preview)
        
        # Populate list
        for face in faces:
            face_list.insert(tk.END, face['filename'])
        
        def add_to_match(match_name):
            if not current_face:
                return
            
            if self.analyzer.index_face(current_face['path'], match_name):
                self.show_message("Success", f"Face added to {match_name}")
                # Remove from list
                selection = face_list.curselection()
                if selection:
                    face_list.delete(selection[0])
                    faces.pop(selection[0])
                # Clear preview and details
                preview_canvas.delete("all")
                details_text.delete(1.0, tk.END)
                matches_text.delete(1.0, tk.END)
                # Clear match buttons
                for widget in details_frame.winfo_children():
                    if isinstance(widget, ttk.Frame):
                        widget.destroy()
                self.update_collection_info()
            else:
                self.show_message("Error", "Failed to add face", "error")
        
        def create_match_buttons():
            # Clear existing buttons
            for widget in details_frame.winfo_children():
                if isinstance(widget, ttk.Frame):
                    widget.destroy()
            
            if current_matches:
                btn_container = ttk.Frame(details_frame)
                btn_container.pack(fill=tk.X, pady=5)
                
                for match in current_matches:
                    ttk.Button(btn_container, 
                             text=f"Add to {match['name']}",
                             command=lambda m=match['name']: add_to_match(m)
                             ).pack(side=tk.LEFT, padx=2)
        
        def index_as_new():
            if not current_face:
                self.show_message("No Selection", "Please select a face to index", "warning")
                return
            
            # Create name entry dialog
            name_dialog = tk.Toplevel(index_window)
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
                
                if self.analyzer.index_face(current_face['path'], name):
                    self.show_message("Success", f"Face indexed as {name}")
                    # Remove from list
                    selection = face_list.curselection()
                    if selection:
                        face_list.delete(selection[0])
                        faces.pop(selection[0])
                    # Clear preview and details
                    preview_canvas.delete("all")
                    details_text.delete(1.0, tk.END)
                    matches_text.delete(1.0, tk.END)
                    # Clear match buttons
                    for widget in details_frame.winfo_children():
                        if isinstance(widget, ttk.Frame):
                            widget.destroy()
                    self.update_collection_info()
                    name_dialog.destroy()
                else:
                    self.show_message("Error", "Failed to index face", "error")
            
            ttk.Button(name_dialog, text="Index", command=do_index).pack(pady=10)
            name_entry.bind('<Return>', lambda e: do_index())
        
        def delete_current():
            selection = face_list.curselection()
            if not selection:
                self.show_message("No Selection", "Please select a face to delete", "warning")
                return
            
            if messagebox.askyesno("Confirm Delete", "Delete selected face?"):
                face = faces[selection[0]]
                try:
                    os.remove(face['path'])
                    face_list.delete(selection[0])
                    faces.pop(selection[0])
                    preview_canvas.delete("all")
                    details_text.delete(1.0, tk.END)
                    matches_text.delete(1.0, tk.END)
                    # Clear match buttons
                    for widget in details_frame.winfo_children():
                        if isinstance(widget, ttk.Frame):
                            widget.destroy()
                    self.show_message("Success", "Face deleted")
                except Exception as e:
                    self.show_message("Error", f"Failed to delete face: {e}", "error")
        
        # Action buttons
        btn_frame = ttk.Frame(index_window)
        btn_frame.grid(row=1, column=0, columnspan=3, pady=10)
        
        ttk.Button(btn_frame, text="Index as New Person", command=index_as_new).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Delete Selected", command=delete_current).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Close", command=index_window.destroy).pack(side=tk.LEFT, padx=5)
        
        # Configure grid weights
        index_window.grid_columnconfigure(0, weight=2)
        index_window.grid_columnconfigure(1, weight=1)
        index_window.grid_columnconfigure(2, weight=1)
        index_window.grid_rowconfigure(0, weight=1)

    def recheck_faces(self):
        """This method is now integrated into index_faces"""
        self.index_faces()

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
            
            delete_btn = ttk.Button(action_frame, text="Delete Person", state=tk.DISABLED)
            delete_btn.pack(side=tk.LEFT, padx=(0, 5))
            
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
        """Launch webcam capture window"""
        # Create capture window
        capture_window = tk.Toplevel(self.root)
        capture_window.title("Capture Face")
        capture_window.geometry("800x600")
        self.center_window(capture_window)
        
        # Preview frame
        preview_frame = ttk.LabelFrame(capture_window, text="Webcam Preview", padding="10")
        preview_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        # Preview canvas
        preview_canvas = tk.Canvas(preview_frame, width=640, height=480, bg='black')
        preview_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Status frame
        status_frame = ttk.Frame(capture_window)
        status_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=5)
        
        status_label = ttk.Label(status_frame, text="Press 'C' to capture, 'Q' to quit")
        status_label.pack(side=tk.LEFT)
        
        quality_label = ttk.Label(status_frame, text="")
        quality_label.pack(side=tk.RIGHT)
        
        # Control frame
        control_frame = ttk.Frame(capture_window)
        control_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=10)
        
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
        
        def update_preview():
            nonlocal current_frame, face_detected, face_box
            
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
                    
                    # Assess face quality
                    face_region = frame[top:bottom, left:right]
                    quality_score, issues = self.analyzer.assess_face_quality(face_region)
                    
                    # Draw box with color based on quality
                    color = (0, 255, 0) if quality_score >= 60 else (0, 165, 255)
                    cv2.rectangle(display_frame, (left, top), (right, bottom), color, 2)
                    
                    # Draw quality score
                    cv2.putText(display_frame, f"Quality: {quality_score:.0f}%",
                              (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    # Update quality label
                    if issues:
                        quality_label.config(text=f"Quality Issues: {', '.join(issues)}")
                    else:
                        quality_label.config(text="Face quality: Good")
                
            except Exception as e:
                print(f"Face detection error: {e}")
                face_detected = False
                face_box = None
                quality_label.config(text="")
            
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
            quality_score, issues = self.analyzer.assess_face_quality(face_region)
            if quality_score < 60:
                if not messagebox.askyesno("Low Quality", 
                    f"Face quality is low ({quality_score:.0f}%). Issues: {', '.join(issues)}\n"
                    "Do you want to capture anyway?"):
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
        capture_window.bind('c', lambda e: capture_face())
        capture_window.bind('q', lambda e: on_closing())
        capture_window.protocol("WM_DELETE_WINDOW", on_closing)
        
        # Add control buttons
        ttk.Button(control_frame, text="Capture (C)", 
                  command=capture_face).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Quit (Q)", 
                  command=on_closing).pack(side=tk.LEFT, padx=5)
        
        # Configure grid weights
        capture_window.grid_columnconfigure(0, weight=1)
        capture_window.grid_rowconfigure(0, weight=1)
        
        # Start preview
        update_preview()

    def cleanup_duplicates(self):
        """Launch cleanup duplicates dialog"""
        if not self.analyzer.face_mapping:
            self.show_message("Empty Collection", "No faces in collection!", "warning")
            return
        
        if messagebox.askyesno("Cleanup Collection", 
            "This will search for and help merge duplicate entries in the collection.\n"
            "Would you like to proceed?"):
            self.analyzer.cleanup_duplicate_entries()
            self.update_collection_info()

    def register_faces(self):
        """Launch register faces from folder dialog"""
        folder_path = filedialog.askdirectory(title="Select Folder with Face Images")
        if folder_path:
            self.analyzer.register_faces_from_folder()
            self.update_collection_info()

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

    def delete_users(self):
        """Launch user deletion dialog"""
        if not self.analyzer.face_mapping:
            self.show_message("Empty Collection", "No users in collection!", "warning")
            return
        
        # Create deletion window
        delete_window = tk.Toplevel(self.root)
        delete_window.title("Delete Users")
        delete_window.geometry("600x400")
        self.center_window(delete_window)
        
        # Make window modal
        delete_window.transient(self.root)
        delete_window.grab_set()
        
        # Main container
        main_frame = ttk.Frame(delete_window, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Warning label
        ttk.Label(main_frame, 
                 text="⚠️ Warning: This action cannot be undone!",
                 foreground="red",
                 font=("Segoe UI", 10, "bold")).pack(pady=(0, 10))
        
        # Options frame
        options_frame = ttk.LabelFrame(main_frame, text="Delete Options", padding="10")
        options_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Radio buttons for delete options
        delete_option = tk.StringVar(value="specific")
        
        def on_option_change():
            if delete_option.get() == "specific":
                user_list.config(state=tk.NORMAL)
                select_all_btn.config(state=tk.NORMAL)
            else:
                user_list.config(state=tk.DISABLED)
                select_all_btn.config(state=tk.DISABLED)
        
        ttk.Radiobutton(options_frame, 
                       text="Delete specific users",
                       variable=delete_option,
                       value="specific",
                       command=on_option_change).pack(anchor=tk.W, pady=2)
        
        ttk.Radiobutton(options_frame,
                       text="Delete all users",
                       variable=delete_option,
                       value="all",
                       command=on_option_change).pack(anchor=tk.W, pady=2)
        
        # User list frame
        list_frame = ttk.LabelFrame(main_frame, text="Users", padding="10")
        list_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Add 'Select All' button above the user list
        select_all_btn = ttk.Button(list_frame, 
                                  text="Select All",
                                  command=lambda: user_list.select_set(0, tk.END))
        select_all_btn.pack(pady=(0, 5), anchor="w")
        
        # Add scrollbar to list
        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # User list
        user_list = tk.Listbox(list_frame, 
                             selectmode=tk.MULTIPLE,
                             yscrollcommand=scrollbar.set,
                             height=10)
        user_list.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=user_list.yview)
        
        # Populate user list
        for person, data in sorted(self.analyzer.face_mapping.items()):
            user_list.insert(tk.END, f"{person} ({len(data['face_ids'])} faces)")
        
        # Action buttons at the bottom
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=(10, 0))
        
        # Create a label (or tooltip) to indicate if no users are selected (for 'specific' mode)
        status_label = ttk.Label(btn_frame, text="Select one or more users to delete (or choose 'Delete all users').", foreground="gray")
        status_label.pack(side=tk.LEFT, padx=5, pady=5)
        
        def confirm_delete():
            if delete_option.get() == "specific":
                # Get selected users
                selected_indices = user_list.curselection()
                if not selected_indices:
                    self.show_message("No Selection", "Please select users to delete", "warning")
                    return

                selected_users = [user_list.get(i).split(" (")[0] for i in selected_indices]
                confirm_msg = f"Are you sure you want to delete {len(selected_users)} selected users?\n\n"
                confirm_msg += "\n".join(f"• {user}" for user in selected_users)
            else:
                confirm_msg = "Are you sure you want to delete ALL users from the collection?"

            if messagebox.askyesno("Confirm Delete", confirm_msg):
                try:
                    if delete_option.get() == "specific":
                        # Delete selected users
                        for user in selected_users:
                            # Delete from AWS collection
                            self.analyzer.rekognition.delete_faces(
                                CollectionId=self.analyzer.collection_id,
                                FaceIds=self.analyzer.face_mapping[user]['face_ids']
                            )

                            # Delete image files
                            for img_path in self.analyzer.face_mapping[user]['images']:
                                try:
                                    os.remove(img_path)
                                except Exception as e:
                                    print(f"Warning: Could not delete file {img_path}: {e}")

                            # Remove from mapping
                            del self.analyzer.face_mapping[user]

                        self.show_message("Success", f"Successfully deleted {len(selected_users)} users")
                    else:
                        # Delete all users
                        # Get all face IDs
                        all_face_ids = []
                        for person_data in self.analyzer.face_mapping.values():
                            all_face_ids.extend(person_data['face_ids'])

                        if all_face_ids:
                            # Delete from AWS collection
                            self.analyzer.rekognition.delete_faces(
                                CollectionId=self.analyzer.collection_id,
                                FaceIds=all_face_ids
                            )

                        # Delete all image files
                        for person_data in self.analyzer.face_mapping.values():
                            for img_path in person_data['images']:
                                try:
                                    os.remove(img_path)
                                except Exception as e:
                                    print(f"Warning: Could not delete file {img_path}: {e}")

                        # Clear mapping
                        self.analyzer.face_mapping = {}
                        self.show_message("Success", "Successfully deleted all users")

                    # Save changes
                    self.analyzer.save_face_mapping()

                    # Update display
                    self.update_collection_info()
                    delete_window.destroy()

                except Exception as e:
                    self.show_message("Error", f"Error deleting users: {str(e)}", "error")

        # Create the Delete button (renamed to "Delete Selected" for clarity) and Cancel button
        delete_btn = ttk.Button(btn_frame, text="Delete Selected", style="Danger.TButton", command=confirm_delete)
        delete_btn.pack(side=tk.LEFT, padx=5, pady=5)

        cancel_btn = ttk.Button(btn_frame, text="Cancel", command=delete_window.destroy)
        cancel_btn.pack(side=tk.LEFT, padx=5, pady=5)

        # Bind keyboard shortcuts
        delete_window.bind('<Escape>', lambda e: delete_window.destroy())
        delete_window.bind('<Return>', lambda e: confirm_delete())

        # Set focus to window
        delete_window.focus_set()

        # Update the Delete button's state (enable/disable) based on selection (for 'specific' mode)
        def update_delete_btn_state(*args):
            if delete_option.get() == "specific":
                if user_list.curselection():
                    delete_btn.config(state=tk.NORMAL)
                    status_label.config(text="Ready to delete selected users.")
                else:
                    delete_btn.config(state=tk.DISABLED)
                    status_label.config(text="Select one or more users to delete (or choose 'Delete all users').", foreground="gray")
            else:
                delete_btn.config(state=tk.NORMAL)
                status_label.config(text="Ready to delete all users.")

        # Bind the update_delete_btn_state function to the listbox selection and radio button change
        user_list.bind("<<ListboxSelect>>", update_delete_btn_state)
        delete_option.trace("w", update_delete_btn_state)

        # Call update_delete_btn_state once to set initial state
        update_delete_btn_state()

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