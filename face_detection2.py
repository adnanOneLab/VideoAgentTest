#!/usr/bin/env python3
"""
AWS Rekognition Face Detection and Recognition Video Processor
Features:
- Face detection and recognition using AWS Rekognition
- Smart caching system to avoid re-detection
- Beautiful bounding boxes with names
- High-performance video processing
- 80% similarity threshold for face matching
"""

import cv2
import boto3
import json
import hashlib
import os
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class FaceInfo:
    """Store face information"""
    name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x, y, width, height
    last_seen: int  # frame number
    face_id: str

class FaceCache:
    """Intelligent caching system for face recognition"""
    
    def __init__(self, cache_dir: str = "face_cache", similarity_threshold: float = 0.8):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.similarity_threshold = similarity_threshold
        self.face_encodings = {}
        self.load_cache()
    
    def _get_face_hash(self, face_encoding: List[float]) -> str:
        """Generate hash for face encoding"""
        face_str = json.dumps(face_encoding, sort_keys=True)
        return hashlib.md5(face_str.encode()).hexdigest()
    
    def load_cache(self):
        """Load cached face data"""
        cache_file = self.cache_dir / "face_cache.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    self.face_encodings = json.load(f)
                logger.info(f"Loaded {len(self.face_encodings)} cached faces")
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
    
    def save_cache(self):
        """Save face cache to disk"""
        cache_file = self.cache_dir / "face_cache.json"
        try:
            with open(cache_file, 'w') as f:
                json.dump(self.face_encodings, f, indent=2)
            logger.info("Cache saved successfully")
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")
    
    def add_face(self, name: str, face_encoding: List[float], confidence: float):
        """Add a new face to cache"""
        face_hash = self._get_face_hash(face_encoding)
        self.face_encodings[face_hash] = {
            'name': name,
            'encoding': face_encoding,
            'confidence': confidence,
            'timestamp': time.time()
        }
    
    def find_similar_face(self, face_encoding: List[float]) -> Optional[Dict]:
        """Find similar face in cache"""
        # In a real implementation, you'd use proper face encoding comparison
        # For AWS Rekognition, we'll simulate this with the face ID matching
        return None

class VideoFaceProcessor:
    """Main class for processing video with face detection"""
    
    def __init__(self, aws_region: str = 'ap-south-1', collection_id: str = 'my-face-collection'):
        # Initialize AWS clients
        self.rekognition = boto3.client('rekognition', region_name=aws_region)
        self.collection_id = collection_id
        self.face_cache = FaceCache()
        self.known_faces = {}  # face_id -> FaceInfo
        self.frame_cache = {}  # frame_hash -> detection_results
        
        # Visual styling
        self.colors = [
            (255, 87, 51),   # Red-Orange
            (51, 255, 87),   # Green
            (87, 51, 255),   # Blue
            (255, 255, 51),  # Yellow
            (255, 51, 255),  # Magenta
            (51, 255, 255),  # Cyan
            (255, 165, 0),   # Orange
            (128, 0, 128),   # Purple
        ]
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.8
        self.thickness = 2
        
        self._ensure_collection_exists()
    
    def _ensure_collection_exists(self):
        """Ensure AWS Rekognition collection exists"""
        try:
            self.rekognition.describe_collection(CollectionId=self.collection_id)
            logger.info(f"Using existing collection: {self.collection_id}")
        except self.rekognition.exceptions.ResourceNotFoundException:
            try:
                self.rekognition.create_collection(CollectionId=self.collection_id)
                logger.info(f"Created new collection: {self.collection_id}")
            except Exception as e:
                logger.error(f"Failed to create collection: {e}")
                raise
    
    def _get_frame_hash(self, frame: np.ndarray) -> str:
        """Generate hash for frame to enable caching"""
        frame_small = cv2.resize(frame, (64, 64))
        frame_bytes = frame_small.tobytes()
        return hashlib.md5(frame_bytes).hexdigest()
    
    def _detect_faces(self, frame: np.ndarray) -> List[Dict]:
        """Detect faces in frame using AWS Rekognition"""
        try:
            # Convert frame to bytes
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            image_bytes = buffer.tobytes()
            
            # Detect faces
            response = self.rekognition.detect_faces(
                Image={'Bytes': image_bytes},
                Attributes=['ALL']
            )
            
            return response.get('FaceDetails', [])
        except Exception as e:
            logger.error(f"Face detection failed: {e}")
            return []
    
    def _search_faces(self, frame: np.ndarray) -> List[Dict]:
        """Search for known faces in the collection"""
        try:
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            image_bytes = buffer.tobytes()
            
            response = self.rekognition.search_faces_by_image(
                CollectionId=self.collection_id,
                Image={'Bytes': image_bytes},
                MaxFaces=10,
                FaceMatchThreshold=80.0
            )
            
            return response.get('FaceMatches', [])
        except Exception as e:
            logger.warning(f"Face search failed: {e}")
            return []
    
    def _draw_beautiful_bbox(self, frame: np.ndarray, bbox: Tuple[int, int, int, int], 
                           name: str, confidence: float, color_index: int):
        """Draw beautiful bounding box with name"""
        x, y, w, h = bbox
        color = self.colors[color_index % len(self.colors)]
        
        # Draw main bounding box with rounded corners effect
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, self.thickness)
        
        # Draw corner accents
        corner_length = min(20, w // 4, h // 4)
        # Top-left corner
        cv2.line(frame, (x, y), (x + corner_length, y), color, self.thickness + 1)
        cv2.line(frame, (x, y), (x, y + corner_length), color, self.thickness + 1)
        # Top-right corner
        cv2.line(frame, (x + w, y), (x + w - corner_length, y), color, self.thickness + 1)
        cv2.line(frame, (x + w, y), (x + w, y + corner_length), color, self.thickness + 1)
        # Bottom-left corner
        cv2.line(frame, (x, y + h), (x + corner_length, y + h), color, self.thickness + 1)
        cv2.line(frame, (x, y + h), (x, y + h - corner_length), color, self.thickness + 1)
        # Bottom-right corner
        cv2.line(frame, (x + w, y + h), (x + w - corner_length, y + h), color, self.thickness + 1)
        cv2.line(frame, (x + w, y + h), (x + w, y + h - corner_length), color, self.thickness + 1)
        
        # Prepare text
        label = f"{name} ({confidence:.1f}%)"
        (text_width, text_height), baseline = cv2.getTextSize(
            label, self.font, self.font_scale, self.thickness
        )
        
        # Draw text background with gradient effect
        bg_y = y - text_height - 15 if y > text_height + 15 else y + h + 5
        cv2.rectangle(frame, (x, bg_y), (x + text_width + 10, bg_y + text_height + 10), 
                     color, -1)
        cv2.rectangle(frame, (x, bg_y), (x + text_width + 10, bg_y + text_height + 10), 
                     (255, 255, 255), 1)
        
        # Draw text
        cv2.putText(frame, label, (x + 5, bg_y + text_height + 5), 
                   self.font, self.font_scale, (255, 255, 255), self.thickness)
    
    def _convert_rekognition_bbox(self, face_detail: Dict, frame_width: int, frame_height: int) -> Tuple[int, int, int, int]:
        """Convert Rekognition bounding box to pixel coordinates"""
        bbox = face_detail['BoundingBox']
        x = int(bbox['Left'] * frame_width)
        y = int(bbox['Top'] * frame_height)
        w = int(bbox['Width'] * frame_width)
        h = int(bbox['Height'] * frame_height)
        return (x, y, w, h)
    
    def add_known_face(self, name: str, image_path: str):
        """Add a known face to the collection"""
        try:
            with open(image_path, 'rb') as image_file:
                image_bytes = image_file.read()
            
            response = self.rekognition.index_faces(
                CollectionId=self.collection_id,
                Image={'Bytes': image_bytes},
                ExternalImageId=name,
                MaxFaces=1
            )
            
            if response['FaceRecords']:
                face_id = response['FaceRecords'][0]['Face']['FaceId']
                logger.info(f"Added {name} to collection with FaceId: {face_id}")
                return face_id
            else:
                logger.warning(f"No face detected in {image_path}")
                return None
        except Exception as e:
            logger.error(f"Failed to add face {name}: {e}")
            return None
    
    def process_video(self, input_path: str, output_path: str):
        """Process video with face detection and recognition"""
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {input_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        logger.info(f"Processing video: {width}x{height} @ {fps}fps, {total_frames} frames")
        
        # Create OpenCV window for real-time display
        window_name = "Face Detection Processing"
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        cv2.moveWindow(window_name, 100, 100)  # Position window
        
        frame_number = 0
        skip_frames = 4  # Process every 4th frame for performance
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_number += 1
                
                # Skip frames for performance but still write them
                if frame_number % skip_frames != 0:
                    # Apply last known face positions
                    self._apply_cached_faces(frame, frame_number)
                    
                    # Display frame in OpenCV window (even for skipped frames)
                    display_frame = frame.copy()
                    self._add_progress_info(display_frame, frame_number, total_frames)
                    cv2.imshow(window_name, display_frame)
                    
                    out.write(frame)
                    
                    # Check for ESC key to exit early
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27:  # ESC key
                        logger.info("Processing interrupted by user")
                        break
                    continue
                
                # Check frame cache
                frame_hash = self._get_frame_hash(frame)
                if frame_hash in self.frame_cache:
                    detections = self.frame_cache[frame_hash]
                else:
                    # Detect and recognize faces
                    face_matches = self._search_faces(frame)
                    # Only get face_details if we need them (we don't for recognition-only mode)
                    # face_details = self._detect_faces(frame)  # Removed to only show recognized faces
                    
                    detections = {
                        'matches': face_matches,
                        'details': []  # Empty since we only want recognized faces
                    }
                    self.frame_cache[frame_hash] = detections
                
                # Process detections
                self._process_detections(frame, detections, frame_number, width, height)
                
                # Display frame in OpenCV window
                display_frame = frame.copy()
                self._add_progress_info(display_frame, frame_number, total_frames)
                cv2.imshow(window_name, display_frame)
                
                # Write frame
                out.write(frame)
                
                # Check for ESC key to exit early
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC key
                    logger.info("Processing interrupted by user")
                    break
                
                # Progress update
                if frame_number % 100 == 0:
                    progress = (frame_number / total_frames) * 100
                    logger.info(f"Progress: {progress:.1f}% ({frame_number}/{total_frames})")
        
        finally:
            cap.release()
            out.release()
            cv2.destroyAllWindows()  # Close OpenCV windows
            self.face_cache.save_cache()
            logger.info(f"Video processing complete. Output saved to: {output_path}")
    
    def _process_detections(self, frame: np.ndarray, detections: Dict, 
                          frame_number: int, frame_width: int, frame_height: int):
        """Process face detections and draw bounding boxes - ONLY for recognized faces"""
        color_index = 0
        current_frame_faces = set()
        
        # Process ONLY known faces with 80%+ similarity
        for match in detections['matches']:
            face = match['Face']
            similarity = match['Similarity']
            
            if similarity >= 80.0:  # 80% threshold - ONLY these get bounding boxes
                face_id = face['FaceId']
                external_id = face.get('ExternalImageId', f'Person_{face_id[:8]}')
                current_frame_faces.add(face_id)
                
                # Convert bounding box
                bbox = self._convert_rekognition_bbox(
                    {'BoundingBox': face['BoundingBox']}, 
                    frame_width, frame_height
                )
                
                # Update or create face info with smooth tracking
                if face_id in self.known_faces:
                    # Smooth bbox transition to prevent jittering
                    old_bbox = self.known_faces[face_id].bbox
                    smoothed_bbox = self._smooth_bbox(old_bbox, bbox)
                    
                    self.known_faces[face_id].bbox = smoothed_bbox
                    self.known_faces[face_id].last_seen = frame_number
                    self.known_faces[face_id].confidence = max(similarity, self.known_faces[face_id].confidence)
                else:
                    self.known_faces[face_id] = FaceInfo(
                        name=external_id,
                        confidence=similarity,
                        bbox=bbox,
                        last_seen=frame_number,
                        face_id=face_id
                    )
                
                # Draw bounding box for recognized face
                self._draw_beautiful_bbox(
                    frame, self.known_faces[face_id].bbox, external_id, 
                    self.known_faces[face_id].confidence, color_index
                )
                color_index += 1
        
        # Clean up old faces that haven't been seen for too long
        faces_to_remove = [
            face_id for face_id, face_info in self.known_faces.items()
            if frame_number - face_info.last_seen > self.tracking_duration
        ]
        for face_id in faces_to_remove:
            del self.known_faces[face_id]
    
    def _smooth_bbox(self, old_bbox: Tuple[int, int, int, int], 
                     new_bbox: Tuple[int, int, int, int], 
                     smoothing_factor: float = 0.7) -> Tuple[int, int, int, int]:
        """Smooth bounding box transitions to prevent jittering"""
        if not old_bbox:
            return new_bbox
        
        # Apply smoothing to each coordinate
        x1, y1, w1, h1 = old_bbox
        x2, y2, w2, h2 = new_bbox
        
        # Smooth transition
        smooth_x = int(x1 * smoothing_factor + x2 * (1 - smoothing_factor))
        smooth_y = int(y1 * smoothing_factor + y2 * (1 - smoothing_factor))
        smooth_w = int(w1 * smoothing_factor + w2 * (1 - smoothing_factor))
        smooth_h = int(h1 * smoothing_factor + h2 * (1 - smoothing_factor))
        
        return (smooth_x, smooth_y, smooth_w, smooth_h)
    
    def _apply_cached_faces(self, frame: np.ndarray, frame_number: int):
        """Apply cached face positions to frame - stable display without blinking"""
        color_index = 0
        for face_info in self.known_faces.values():
            # Show face for longer duration to prevent blinking
            if frame_number - face_info.last_seen <= self.tracking_duration:
                self._draw_beautiful_bbox(
                    frame, face_info.bbox, face_info.name, 
                    face_info.confidence, color_index
                )
                color_index += 1
    
    def _bbox_overlap(self, bbox1: Tuple[int, int, int, int], 
                     bbox2: Tuple[int, int, int, int]) -> float:
        """Calculate overlap ratio between two bounding boxes"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Calculate intersection
        x_overlap = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
        y_overlap = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
        intersection = x_overlap * y_overlap
        
        # Calculate union
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def _add_progress_info(self, frame: np.ndarray, current_frame: int, total_frames: int):
        """Add progress information to the display frame"""
        progress = (current_frame / total_frames) * 100
        progress_text = f"Progress: {progress:.1f}% ({current_frame}/{total_frames})"
        
        # Get frame dimensions
        h, w = frame.shape[:2]
        
        # Add semi-transparent background for progress text
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 50), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Add progress text
        cv2.putText(frame, progress_text, (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Add progress bar
        bar_width = 300
        bar_height = 8
        bar_x = 20
        bar_y = 45
        
        # Progress bar background
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                     (64, 64, 64), -1)
        
        # Progress bar fill
        fill_width = int((progress / 100) * bar_width)
        if fill_width > 0:
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), 
                         (0, 255, 0), -1)
        
        # Add instructions
        cv2.putText(frame, "Press ESC to stop processing", (20, h - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

def main():
    """Main function to demonstrate usage"""
    # Initialize processor
    processor = VideoFaceProcessor(
        aws_region='us-east-1',  # Change to your preferred region
        collection_id='video_faces'
    )
    
    # Add known faces (optional)
    # processor.add_known_face("John Doe", "path/to/john_doe.jpg")
    # processor.add_known_face("Jane Smith", "path/to/jane_smith.jpg")
    
    # Process video
    input_video = "people_walking_again.mp4"  # Change to your input video path
    output_video = "output_video1.mp4"  # Change to your desired output path
    
    try:
        processor.process_video(input_video, output_video)
        print(f"✅ Video processing completed successfully!")
        print(f"📹 Output saved to: {output_video}")
    except Exception as e:
        print(f"❌ Error processing video: {e}")

if __name__ == "__main__":
    main()