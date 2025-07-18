#!/usr/bin/env python3
"""
🔥 GPU-Accelerated Advanced Face Tracking System with DeepSORT
🎯 Features:
- Single face registration with validation
- Multi-person tracking with DeepSORT
- GPU acceleration with RTX 4090
- Non-Maximum Suppression to reduce false positives
- Temporal consistency for stable recognition
- Optimized for Hikvision DS-2CD2043G0-I camera
"""

import cv2
import numpy as np
import pickle
import json
import time
import torch
import time
import os
from datetime import datetime
import mediapipe as mp
from facenet_pytorch import MTCNN, InceptionResnetV1
from deep_sort_realtime.deepsort_tracker import DeepSort

class AdvancedFaceTracker:
    def __init__(self, config_path="config.json"):
        print("🚀 Initializing GPU-Accelerated Face Tracking System")
        
        # Load configuration
        self.load_config(config_path)
        
        # Device setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔥 GPU Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"🎮 GPU Device: {torch.cuda.get_device_name()}")
        
        # Load face database
        self.load_face_database()
        
        # Initialize MediaPipe
        mp_face_detection = mp.solutions.face_detection
        self.face_detector = mp_face_detection.FaceDetection(
            model_selection=self.config['mediapipe']['model_selection'],
            min_detection_confidence=self.config['mediapipe']['min_detection_confidence']
        )
        
        print("📡 Loading FaceNet models...")
        
        # Initialize MTCNN for face detection and cropping
        self.mtcnn = MTCNN(
            image_size=self.config['mtcnn']['image_size'],
            margin=self.config['mtcnn']['margin'],
            min_face_size=self.config['mtcnn']['min_face_size'],
            thresholds=self.config['mtcnn']['thresholds'],
            factor=self.config['mtcnn']['factor'],
            post_process=self.config['mtcnn']['post_process'],
            device=self.device,
            keep_all=self.config['mtcnn']['keep_all']
        )
        
        self.facenet = InceptionResnetV1(pretrained=self.config['facenet']['pretrained_model']).eval().to(self.device)
        
        # Initialize DeepSORT tracker with config parameters
        self.tracker = DeepSort(
            max_age=self.config['deepsort']['max_age'],
            n_init=self.config['deepsort']['n_init'],
            nms_max_overlap=self.config['deepsort']['nms_max_overlap'],
            max_cosine_distance=self.config['deepsort']['max_cosine_distance'],
            nn_budget=self.config['deepsort']['nn_budget'],
            override_track_class=None,
            embedder=self.config['deepsort']['embedder'],
            half=self.config['deepsort']['half_precision'],
            bgr=self.config['deepsort']['bgr_format'],
            embedder_gpu=torch.cuda.is_available() and self.config['deepsort']['use_gpu'],
            embedder_model_name=None,
            embedder_wts=None,
            polygon=False,
            today=None
        )
        
        # Tracking variables from config
        self.recognition_cache = {}  # track_id -> (name, confidence, timestamp)
        self.cache_duration = self.config['recognition']['cache_duration']
        self.recognition_threshold = self.config['recognition']['threshold']
        self.blacklisted_tracks = set()  # Track IDs that should remain Unknown
        
        # Motion prediction for fast tracking
        self.track_history = {}  # track_id -> [(x, y, timestamp), ...]
        self.max_history_length = 5
        self.motion_prediction_enabled = self.config['tracking'].get('motion_prediction', True)
        
        # Camera setup from config
        self.setup_camera()
        
        # Performance monitoring
        self.frame_count = 0
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
        print("✅ System initialized successfully!")
        print(f"👥 Loaded {len(self.face_names)} registered users")

    def load_config(self, config_path):
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
            print(f"📝 Configuration loaded from {config_path}")
        except FileNotFoundError:
            print(f"⚠️ Config file {config_path} not found! Using default values.")
            # Create default config if file doesn't exist
            self.config = self.get_default_config()
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            print(f"📝 Default config saved to {config_path}")
        except json.JSONDecodeError as e:
            print(f"❌ Error parsing config file: {e}")
            print("Using default configuration...")
            self.config = self.get_default_config()

    def get_default_config(self):
        """Return default configuration if config file is missing"""
        return {
            "camera": {"device_id": 0, "width": 1280, "height": 720, "fps": 30, "flip_horizontal": True},
            "mediapipe": {"model_selection": 1, "min_detection_confidence": 0.7, "face_detection_confidence": 0.8},
            "mtcnn": {"image_size": 160, "margin": 0, "min_face_size": 40, "thresholds": [0.6, 0.7, 0.7], "factor": 0.709, "post_process": True, "keep_all": True},
            "facenet": {"pretrained_model": "vggface2"},
            "deepsort": {"max_age": 30, "n_init": 5, "nms_max_overlap": 0.7, "max_cosine_distance": 0.2, "nn_budget": 100, "embedder": "mobilenet", "half_precision": True, "bgr_format": True, "use_gpu": True},
            "recognition": {"threshold": 0.80, "cache_duration": 30, "min_track_confidence": 0.3},
            "nms": {"iou_threshold": 0.3, "overlap_threshold": 0.3},
            "ui": {"header_height": 120, "title": "ADVANCED FACE TRACKING SYSTEM", "title_color": [0, 255, 255], "known_color": [0, 255, 0], "unknown_color": [0, 0, 255]},
            "database": {"encodings_file": "face_database/face_encodings.pkl", "users_info_file": "face_database/users_info.json"}
        }

    def load_face_database(self):
        """Load face encodings and user information"""
        try:
            with open(self.config['database']['encodings_file'], 'rb') as f:
                data = pickle.load(f)
                self.face_encodings = data.get('encodings', [])
                self.face_names = data.get('names', [])
        except FileNotFoundError:
            self.face_encodings = []
            self.face_names = []
        
        try:
            with open(self.config['database']['users_info_file'], 'r', encoding='utf-8') as f:
                self.users_info = json.load(f)
        except FileNotFoundError:
            self.users_info = {}
        
        print(f"📚 Loaded {len(self.face_encodings)} face encodings")
        print(f"📋 Loaded {len(self.users_info)} user profiles")

    def setup_camera(self):
        """Setup camera input (webcam or RTSP) - optimized for Hikvision cameras"""
        input_type = self.config['camera'].get('input_type', 'webcam').lower()
        
        print(f"🎥 Setting up camera input: {input_type}")
        
        if input_type == 'rtsp':
            rtsp_url = self.config['camera'].get('rtsp_url', '')
            if not rtsp_url:
                print("❌ RTSP URL not configured, falling back to webcam")
                input_type = 'webcam'
            else:
                print(f"📡 Connecting to RTSP: {rtsp_url}")
                
                # Hikvision-specific optimizations
                if self.config['camera'].get('hikvision_optimization', False):
                    # Use TCP protocol for more stable connection with Hikvision
                    if '?' in rtsp_url:
                        rtsp_url += '&tcp'
                    else:
                        rtsp_url += '?tcp'
                    print("🔧 Applied Hikvision TCP optimization")
                
                self.cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
                
                # Hikvision-specific settings
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('H', '2', '6', '4'))
                
                # Set buffer size for RTSP to reduce latency
                buffer_size = self.config['camera'].get('buffer_size', 3)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, buffer_size)
                
                # Set connection timeout
                timeout = self.config['camera'].get('connection_timeout', 15)
                self.cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, timeout * 1000)
                
                # Set read timeout for Hikvision cameras
                read_timeout = self.config['camera'].get('read_timeout', 5000)
                self.cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, read_timeout)
        
        if input_type == 'webcam' or not self.cap.isOpened():
            if input_type == 'rtsp':
                print("⚠️ RTSP connection failed, falling back to webcam")
            
            device_id = self.config['camera'].get('device_id', 0)
            print(f"📹 Using webcam device: {device_id}")
            self.cap = cv2.VideoCapture(device_id)
        
        if not self.cap.isOpened():
            raise Exception("❌ Could not open camera input")
        
        # Set camera properties - optimized for Hikvision 4MP camera
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config['camera']['width'])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config['camera']['height'])
        self.cap.set(cv2.CAP_PROP_FPS, self.config['camera']['fps'])
        
        # Additional Hikvision optimizations
        if self.config['camera'].get('hikvision_optimization', False):
            # Disable auto exposure for consistent performance
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
            # Set manual exposure for better face detection
            self.cap.set(cv2.CAP_PROP_EXPOSURE, -6)
            print("🔧 Applied Hikvision camera optimizations")
        
        # Test read to verify connection
        ret, test_frame = self.cap.read()
        if ret:
            print(f"✅ Camera input ready: {test_frame.shape[1]}x{test_frame.shape[0]}")
        else:
            raise Exception("❌ Could not read from camera input")

    def reconnect_camera(self):
        """Reconnect camera with retry logic"""
        max_attempts = self.config['camera'].get('reconnect_attempts', 3)
        
        for attempt in range(max_attempts):
            print(f"🔄 Reconnecting camera (attempt {attempt + 1}/{max_attempts})")
            
            if hasattr(self, 'cap') and self.cap:
                self.cap.release()
            
            try:
                self.setup_camera()
                return True
            except Exception as e:
                print(f"⚠️ Reconnection attempt {attempt + 1} failed: {e}")
                time.sleep(2)
        
        print("❌ All reconnection attempts failed")
        return False

    def detect_faces_mediapipe(self, frame):
        """Detect faces using MediaPipe with multi-scale detection for small and distant faces"""
        # Hikvision camera preprocessing
        if self.config['camera'].get('hikvision_optimization', False):
            frame = self.preprocess_hikvision_frame(frame)
        
        all_detections = []
        
        # Multi-scale detection if enabled
        if self.config['camera'].get('multi_scale_detection', False):
            scale_factors = self.config['camera'].get('scale_factors', [1.0, 0.8, 0.6])
            
            for scale in scale_factors:
                # Resize frame for different scales
                if scale != 1.0:
                    h, w = frame.shape[:2]
                    new_h, new_w = int(h * scale), int(w * scale)
                    if new_h > 0 and new_w > 0:
                        scaled_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                    else:
                        continue
                else:
                    scaled_frame = frame
                
                # Process scaled frame
                detections = self._detect_faces_single_scale(scaled_frame, scale)
                all_detections.extend(detections)
        else:
            # Standard single-scale detection
            all_detections = self._detect_faces_single_scale(frame, 1.0)
        
        # Apply NMS to all detections
        if all_detections:
            all_detections = self.apply_nms(all_detections, self.config['nms']['iou_threshold'])
            
            # Limit detections for performance in multi-person scenarios
            max_detections = 25  # Increased limit for 4MP camera
            if len(all_detections) > max_detections:
                # Sort by confidence and keep top detections
                all_detections.sort(key=lambda x: x[4], reverse=True)
                all_detections = all_detections[:max_detections]
        
        return all_detections

    def preprocess_hikvision_frame(self, frame):
        """Preprocessing specifically for Hikvision DS-2CD2043G0-I camera"""
        # Apply histogram equalization to improve contrast
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        lab[:,:,0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(lab[:,:,0])
        frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # Slight noise reduction while preserving edges
        frame = cv2.bilateralFilter(frame, 5, 50, 50)
        
        # Enhance sharpness for better face detection
        kernel = np.array([[-1,-1,-1],
                          [-1, 9,-1],
                          [-1,-1,-1]])
        frame = cv2.filter2D(frame, -1, kernel * 0.1)
        
        return frame

    def _detect_faces_single_scale(self, frame, scale_factor=1.0):
        """Detect faces at single scale"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Apply slight gaussian blur to reduce motion blur artifacts
        rgb_frame = cv2.GaussianBlur(rgb_frame, (3, 3), 0.5)
        
        results = self.face_detector.process(rgb_frame)
        
        detections = []
        if results.detections:
            h, w, _ = frame.shape
            
            for detection in results.detections:
                confidence = detection.score[0]
                
                # Lower confidence threshold for distant/small faces
                detection_threshold = self.config['mediapipe']['face_detection_confidence']
                
                if confidence > detection_threshold:
                    bbox = detection.location_data.relative_bounding_box
                    
                    # Scale coordinates back to original frame size
                    x = max(0, int(bbox.xmin * w / scale_factor))
                    y = max(0, int(bbox.ymin * h / scale_factor))
                    width = min(int(bbox.width * w / scale_factor), w - x)
                    height = min(int(bbox.height * h / scale_factor), h - y)
                    
                    # Check minimum detection area
                    detection_area = width * height
                    min_area = self.config['camera'].get('min_detection_area', 400)
                    
                    # More lenient size and aspect ratio check
                    min_size = self.config['mediapipe']['min_face_size']
                    min_ratio = self.config['mediapipe']['min_aspect_ratio']
                    max_ratio = self.config['mediapipe']['max_aspect_ratio']
                    
                    if (detection_area > min_area and 
                        width > min_size and height > min_size and 
                        min_ratio <= width/height <= max_ratio):
                        detections.append([x, y, width, height, confidence])
                        
                        # Debug info
                        if self.config['debug'].get('print_detection_info', False):
                            print(f"✅ Face detected: {width}x{height} at ({x},{y}) conf={confidence:.2f} area={detection_area}")
                    else:
                        # Debug rejected detections
                        if self.config['debug'].get('print_detection_info', False):
                            print(f"❌ Face rejected: {width}x{height} area={detection_area} ratio={width/height:.2f}")
        
        return detections
    
    def apply_nms(self, detections, iou_threshold=0.3):
        """Apply Non-Maximum Suppression to remove overlapping detections"""
        if not detections:
            return []
        
        import numpy as np
        
        boxes = np.array([[x, y, x+w, y+h] for x, y, w, h, _ in detections])
        scores = np.array([conf for _, _, _, _, conf in detections])
        
        # Calculate areas
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        
        # Sort by confidence
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            if order.size == 1:
                break
            
            # Calculate IoU
            xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
            yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
            xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
            yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
            
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            intersection = w * h
            
            union = areas[i] + areas[order[1:]] - intersection
            iou = intersection / union
            
            # Keep only detections with IoU below threshold
            order = order[1:][iou <= iou_threshold]
        
        return [detections[i] for i in keep]

    def extract_face_encoding(self, frame, bbox):
        """Extract face encoding using FaceNet"""
        try:
            x, y, w, h = bbox[:4]
            
            # Add margin from config
            margin = self.config['mtcnn']['face_crop_margin']
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(frame.shape[1], x + w + margin)
            y2 = min(frame.shape[0], y + h + margin)
            
            face_crop = frame[y1:y2, x1:x2]
            
            if face_crop.size == 0:
                return None
            
            # Convert to RGB
            rgb_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            
            # Detect face with MTCNN
            face_tensor = self.mtcnn(rgb_face)
            
            if face_tensor is None:
                return None
            
            # Handle multiple faces in crop
            if len(face_tensor.shape) == 4:  # Multiple faces
                face_tensor = face_tensor[0]  # Take the first one
            
            face_tensor = face_tensor.unsqueeze(0).to(self.device)
            
            # Get encoding
            with torch.no_grad():
                encoding = self.facenet(face_tensor)
                encoding = encoding.cpu().numpy().flatten()
            
            return encoding
            
        except Exception as e:
            print(f"⚠️ Error extracting encoding: {e}")
            return None

    def recognize_face(self, encoding):
        """Recognize face using cosine similarity"""
        if encoding is None or len(self.face_encodings) == 0:
            return "Unknown", 0.0
        
        try:
            # Calculate cosine similarity
            similarities = []
            for known_encoding in self.face_encodings:
                # Normalize vectors
                encoding_norm = encoding / np.linalg.norm(encoding)
                known_encoding_norm = known_encoding / np.linalg.norm(known_encoding)
                
                # Cosine similarity
                similarity = np.dot(encoding_norm, known_encoding_norm)
                similarities.append(similarity)
            
            max_similarity = max(similarities)
            best_match_index = similarities.index(max_similarity)
            
            if max_similarity >= self.recognition_threshold:
                return self.face_names[best_match_index], max_similarity
            else:
                return "Unknown", max_similarity
                
        except Exception as e:
            print(f"⚠️ Error in recognition: {e}")
            return "Unknown", 0.0

    def update_track_history(self, track_id, bbox):
        """Enhanced motion history with acceleration tracking"""
        current_time = time.time()
        center_x = bbox[0] + bbox[2] / 2
        center_y = bbox[1] + bbox[3] / 2
        
        if track_id not in self.track_history:
            self.track_history[track_id] = []
        
        # Add current position
        self.track_history[track_id].append((center_x, center_y, current_time))
        
        # Keep adaptive history length based on motion speed
        if len(self.track_history[track_id]) >= 2:
            # Calculate recent motion speed
            recent = self.track_history[track_id][-2:]
            dx = recent[1][0] - recent[0][0]
            dy = recent[1][1] - recent[0][1]
            dt = recent[1][2] - recent[0][2]
            
            if dt > 0:
                speed = np.sqrt(dx**2 + dy**2) / dt
                
                # Adaptive history length: more history for stable motion, less for fast motion
                if speed > 300:  # Fast motion
                    max_length = 3
                elif speed > 100:  # Medium motion
                    max_length = 5
                else:  # Slow/stable motion
                    max_length = 8
                
                # Keep only recent history based on motion speed
                if len(self.track_history[track_id]) > max_length:
                    self.track_history[track_id] = self.track_history[track_id][-max_length:]
        else:
            # Default max length for new tracks
            if len(self.track_history[track_id]) > self.max_history_length:
                self.track_history[track_id].pop(0)

    def predict_next_position(self, track_id):
        """Advanced prediction with multiple motion models"""
        if track_id not in self.track_history or len(self.track_history[track_id]) < 2:
            return None
        
        history = self.track_history[track_id]
        
        # Use more recent positions for faster response
        recent_positions = history[-5:] if len(history) >= 5 else history
        
        if len(recent_positions) < 2:
            return None
        
        # Multi-frame velocity calculation with exponential weighting
        velocities_x = []
        velocities_y = []
        weights = []
        
        for i in range(1, len(recent_positions)):
            prev_x, prev_y, prev_t = recent_positions[i-1]
            curr_x, curr_y, curr_t = recent_positions[i]
            
            dt = curr_t - prev_t
            if dt > 0:
                vx = (curr_x - prev_x) / dt
                vy = (curr_y - prev_y) / dt
                
                # Weight recent velocities more heavily
                weight = np.exp(i - len(recent_positions))  # Exponential decay
                
                velocities_x.append(vx)
                velocities_y.append(vy)
                weights.append(weight)
        
        if not velocities_x:
            return None
        
        # Weighted average velocity
        weights = np.array(weights)
        weights = weights / np.sum(weights)  # Normalize
        
        avg_vx = np.average(velocities_x, weights=weights)
        avg_vy = np.average(velocities_y, weights=weights)
        
        # Adaptive prediction time based on motion speed
        last_x, last_y, last_t = history[-1]
        speed = np.sqrt(avg_vx**2 + avg_vy**2)
        
        # Faster prediction for high-speed motion
        if speed > 500:  # Fast motion
            prediction_frames = self.config['tracking']['prediction_frames'] * 2
            predicted_dt = 0.016  # ~60 FPS assumption for fast motion
        else:
            prediction_frames = self.config['tracking']['prediction_frames']
            predicted_dt = 0.033  # ~30 FPS
        
        # Multi-frame prediction
        predicted_positions = []
        for frame in range(1, prediction_frames + 1):
            pred_x = last_x + avg_vx * predicted_dt * frame
            pred_y = last_y + avg_vy * predicted_dt * frame
            predicted_positions.append((pred_x, pred_y))
        
        # Return the first prediction for immediate use
        return predicted_positions[0] if predicted_positions else None

    def calculate_motion_score(self, track_id, detection_bbox):
        """Calculate motion-based matching score"""
        predicted_pos = self.predict_next_position(track_id)
        if predicted_pos is None:
            return 0.0
        
        pred_x, pred_y = predicted_pos
        det_center_x = detection_bbox[0] + detection_bbox[2] / 2
        det_center_y = detection_bbox[1] + detection_bbox[3] / 2
        
        # Calculate distance
        distance = np.sqrt((pred_x - det_center_x)**2 + (pred_y - det_center_y)**2)
        
        # Convert to score (closer = higher score)
        max_distance = self.config['tracking'].get('max_motion_distance', 200)
        motion_score = max(0, 1.0 - (distance / max_distance))
        
        return motion_score

    def enhance_tracking_with_motion(self, tracks, detections):
        """Enhance tracking by considering motion prediction"""
        if not self.motion_prediction_enabled:
            return tracks
        
        enhanced_tracks = []
        motion_weight = self.config['tracking'].get('motion_weight', 0.3)
        
        for track in tracks:
            track_id = track.track_id
            bbox = track.to_ltwh()
            
            # Update motion history
            self.update_track_history(track_id, bbox)
            
            # Calculate motion consistency score
            if len(detections) > 0:
                motion_scores = []
                for detection in detections:
                    det_bbox = detection[0]  # [x, y, w, h]
                    motion_score = self.calculate_motion_score(track_id, det_bbox)
                    motion_scores.append(motion_score)
                
                # Use best motion score as bonus
                best_motion_score = max(motion_scores) if motion_scores else 0
                
                # Apply motion bonus to track confidence
                if hasattr(track, 'confidence'):
                    track.confidence = min(1.0, track.confidence + best_motion_score * motion_weight)
            
            enhanced_tracks.append(track)
        
        return enhanced_tracks

    def clean_track_history(self):
        """Clean old track history entries"""
        current_time = time.time()
        to_remove = []
        
        for track_id, history in self.track_history.items():
            # Remove old entries
            self.track_history[track_id] = [
                entry for entry in history 
                if current_time - entry[2] < self.cache_duration
            ]
            
            # Remove empty histories
            if not self.track_history[track_id]:
                to_remove.append(track_id)
        
        for track_id in to_remove:
            del self.track_history[track_id]

    def update_recognition_cache(self, track_id, new_name, new_confidence):
        """Update recognition cache with temporal consistency"""
        # Check if track is blacklisted (should remain Unknown)
        if track_id in self.blacklisted_tracks:
            return "Unknown", max(0.2, new_confidence * 0.3)
        
        current_time = time.time()
        
        if track_id in self.recognition_cache:
            cached_name, cached_confidence, _ = self.recognition_cache[track_id]
            
            # Use weighted average for confidence
            if new_name == cached_name:
                # Same person, increase confidence
                updated_confidence = min(1.0, (cached_confidence + new_confidence) / 2)
                self.recognition_cache[track_id] = (new_name, updated_confidence, current_time)
                return new_name, updated_confidence
            else:
                # Different person, check which has higher confidence
                if new_confidence > cached_confidence:
                    self.recognition_cache[track_id] = (new_name, new_confidence, current_time)
                    return new_name, new_confidence
                else:
                    return cached_name, cached_confidence
        else:
            # New track
            self.recognition_cache[track_id] = (new_name, new_confidence, current_time)
            return new_name, new_confidence

    def clean_recognition_cache(self):
        """Remove old entries from recognition cache"""
        current_time = time.time()
        to_remove = []
        
        for track_id, (_, _, timestamp) in self.recognition_cache.items():
            if current_time - timestamp > self.cache_duration:
                to_remove.append(track_id)
        
        for track_id in to_remove:
            del self.recognition_cache[track_id]

    def filter_overlapping_tracks(self, tracks, track_data=None):
        """Filter out overlapping tracks to reduce false positives"""
        if len(tracks) <= 1:
            return tracks
        
        import numpy as np
        
        # Get bounding boxes
        boxes = []
        for track in tracks:
            bbox = track.to_ltwh()
            boxes.append([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])
        
        boxes = np.array(boxes)
        
        # Calculate IoU matrix and filter
        filtered_tracks = []
        used_indices = set()
        
        for i, track in enumerate(tracks):
            if i in used_indices:
                continue
                
            current_box = boxes[i]
            keep_track = True
            
            for j, other_track in enumerate(tracks):
                if i >= j or j in used_indices:
                    continue
                
                other_box = boxes[j]
                
                # Calculate IoU
                x1 = max(current_box[0], other_box[0])
                y1 = max(current_box[1], other_box[1])
                x2 = min(current_box[2], other_box[2])
                y2 = min(current_box[3], other_box[3])
                
                if x2 > x1 and y2 > y1:
                    intersection = (x2 - x1) * (y2 - y1)
                    
                    area1 = (current_box[2] - current_box[0]) * (current_box[3] - current_box[1])
                    area2 = (other_box[2] - other_box[0]) * (other_box[3] - other_box[1])
                    
                    iou = intersection / (area1 + area2 - intersection)
                    
                    if iou > self.config['nms']['overlap_threshold']:
                        # Prioritize based on multiple factors
                        track_score = self.calculate_track_priority(track, track.track_id)
                        other_score = self.calculate_track_priority(other_track, other_track.track_id)
                        
                        if track_score >= other_score:
                            used_indices.add(j)
                        else:
                            keep_track = False
                            break
            
            if keep_track:
                filtered_tracks.append(track)
                used_indices.add(i)
        
        return filtered_tracks
    
    def calculate_track_priority(self, track, track_id):
        """Calculate priority score for track selection"""
        score = 0
        
        # Base score from track age (older = more stable)
        max_age = self.config['tracking']['max_age_bonus']
        age_weight = self.config['tracking']['priority_weight_age']
        score += min(track.age, max_age) * age_weight
        
        # Penalty for recent missed detections
        missed_penalty = self.config['tracking']['priority_penalty_missed']
        score -= track.time_since_update * missed_penalty
        
        # Bonus for recognized faces
        if track_id in self.recognition_cache:
            name, confidence, _ = self.recognition_cache[track_id]
            if name != "Unknown":
                recognized_bonus = self.config['tracking']['priority_bonus_recognized']
                confidence_weight = self.config['tracking']['priority_weight_confidence']
                score += recognized_bonus  # Big bonus for recognized faces
                score += confidence * confidence_weight  # Additional confidence bonus
            else:
                unknown_penalty = self.config['tracking']['priority_penalty_unknown']
                score -= unknown_penalty  # Penalty for unknown faces
        
        return score

    def check_duplicate_identity(self, active_tracks):
        """Check if multiple tracks are assigned to the same person and resolve conflicts"""
        if len(active_tracks) <= 1:
            return active_tracks
        
        # Group tracks by recognized name (excluding Unknown)
        name_groups = {}
        unknown_tracks = []
        
        for track in active_tracks:
            name = track.get('name', 'Unknown')
            if name == 'Unknown':
                unknown_tracks.append(track)
            else:
                if name not in name_groups:
                    name_groups[name] = []
                name_groups[name].append(track)
        
        # Process each named group
        final_tracks = []
        
        for name, tracks in name_groups.items():
            if len(tracks) > 1:
                if self.config['debug']['print_duplicate_detection']:
                    print(f"🔍 Found {len(tracks)} tracks assigned to '{name}'")
                
                # Sort by multiple criteria: confidence, track age, bbox stability
                def track_score(track):
                    confidence = track.get('confidence', 0.0)
                    track_id = track.get('track_id', 0)
                    
                    # Get track object for additional info
                    age_bonus = 0
                    if track_id in self.recognition_cache:
                        _, _, timestamp = self.recognition_cache[track_id]
                        age_bonus = min(0.1, (time.time() - timestamp) / 100)  # Older = more stable
                    
                    return confidence + age_bonus
                
                # Keep only the best track
                best_track = max(tracks, key=track_score)
                if self.config['debug']['print_duplicate_detection']:
                    print(f"✅ Keeping track {best_track['track_id']} with confidence {best_track['confidence']:.2f}")
                
                # Convert others to Unknown
                for track in tracks:
                    if track['track_id'] != best_track['track_id']:
                        old_conf = track['confidence']
                        track['name'] = 'Unknown'
                        track['confidence'] = max(0.2, old_conf * 0.3)  # Reduce confidence significantly
                        
                        if self.config['debug']['print_track_reassignment']:
                            print(f"⚠️ Track {track['track_id']} reassigned from '{name}' to Unknown (conf: {old_conf:.2f} → {track['confidence']:.2f})")
                        
                        # Add to blacklist and clear from recognition cache
                        self.blacklisted_tracks.add(track['track_id'])
                        if track['track_id'] in self.recognition_cache:
                            del self.recognition_cache[track['track_id']]
                
                final_tracks.extend(tracks)
            else:
                # Single track for this name, keep as is
                final_tracks.extend(tracks)
        
        # Add back unknown tracks
        final_tracks.extend(unknown_tracks)
        
        return final_tracks

    def draw_results(self, frame, tracks):
        """Draw tracking and recognition results"""
        # Header from config
        header_height = self.config['ui']['header_height']
        bg_color = tuple(self.config['ui']['background_color'])
        border_color = tuple(self.config['ui']['border_color'])
        
        cv2.rectangle(frame, (0, 0), (frame.shape[1], header_height), bg_color, -1)
        cv2.rectangle(frame, (0, 0), (frame.shape[1], header_height), border_color, 2)
        
        # Title from config
        title = self.config['ui']['title']
        title_color = tuple(self.config['ui']['title_color'])
        title_scale = self.config['ui']['font_scale']['title']
        cv2.putText(frame, title, (20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, title_scale, title_color, 2)
        
        # Stats
        known_count = sum(1 for track in tracks if track.get('name', 'Unknown') != 'Unknown')
        unknown_count = len(tracks) - known_count
        
        stats = [
            f"Database: {len(self.face_names)} users",
            f"Active Tracks: {len(tracks)}",
            f"Known: {known_count} | Unknown: {unknown_count}",
            f"FPS: {self.current_fps:.1f}"
        ]
        
        text_color = tuple(self.config['ui']['text_color'])
        stats_scale = self.config['ui']['font_scale']['stats']
        
        for i, text in enumerate(stats):
            cv2.putText(frame, text, (20, 60 + i*15), 
                       cv2.FONT_HERSHEY_SIMPLEX, stats_scale, text_color, 1)
        
        # Draw tracks
        for track in tracks:
            bbox = track['bbox']
            track_id = track['track_id']
            name = track.get('name', 'Unknown')
            confidence = track.get('confidence', 0.0)
            
            x, y, w, h = map(int, bbox)
            
            # Color coding from config
            if name == "Unknown":
                color = tuple(self.config['ui']['unknown_color'])
                label_color = tuple(self.config['ui']['unknown_color'])
            else:
                color = tuple(self.config['ui']['known_color'])
                label_color = tuple(self.config['ui']['known_color'])
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # Prepare simple label - only track_id and name
            label_lines = [
                f"{track_id}: {name}"
            ]
            
            # Calculate label background size
            line_height = self.config['ui']['line_height']
            label_scale = self.config['ui']['font_scale']['labels']
            max_width = max([cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, label_scale, 1)[0][0] for line in label_lines])
            label_height = len(label_lines) * line_height + self.config['ui']['label_padding']
            
            # Draw label background
            label_y = max(0, y - label_height)
            cv2.rectangle(frame, (x, label_y), (x + max_width + 10, y), label_color, -1)
            
            # Draw label text
            for i, line in enumerate(label_lines):
                text_y = label_y + (i + 1) * line_height
                cv2.putText(frame, line, (x + 5, text_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, label_scale, text_color, 1)

    def calculate_fps(self):
        """Calculate and update FPS"""
        self.fps_counter += 1
        
        fps_interval = self.config['performance']['fps_update_interval']
        if self.fps_counter % fps_interval == 0:  # Update based on config
            current_time = time.time()
            elapsed_time = current_time - self.fps_start_time
            self.current_fps = fps_interval / elapsed_time
            self.fps_start_time = current_time

    def run(self):
        """Main tracking loop - optimized for Hikvision cameras"""
        print("\n🎯 Starting Advanced Face Tracking System")
        print("📋 Controls:")
        print(f"   '{self.config['controls']['quit_key']}' - Quit")
        print(f"   '{self.config['controls']['screenshot_key']}' - Save screenshot")
        print(f"   '{self.config['controls']['reset_key']}' - Reset tracker")
        print(f"   '{self.config['controls']['pause_key'].upper()}' - Pause/Resume")
        
        paused = False
        frame_skip_counter = 0
        frame_skip = self.config['performance'].get('frame_skip', 1)
        
        while True:
            if not paused:
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ Cannot read from camera! Attempting to reconnect...")
                    if self.reconnect_camera():
                        print("✅ Camera reconnected successfully")
                        continue
                    else:
                        print("❌ Failed to reconnect camera, exiting...")
                        break
                
                self.frame_count += 1
                frame_skip_counter += 1
                
                # Skip frames for performance with high-resolution Hikvision cameras
                if frame_skip_counter % frame_skip != 0:
                    continue
                    
                if self.config['camera']['flip_horizontal']:
                    frame = cv2.flip(frame, 1)
                
                # Hikvision frame optimization
                if self.config['camera'].get('hikvision_optimization', False):
                    # Skip processing if frame is too dark (common with Hikvision night mode)
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    avg_brightness = np.mean(gray)
                    if avg_brightness < 30:  # Too dark for reliable face detection
                        # Just display the frame with a warning
                        cv2.putText(frame, "LOW LIGHT - Face detection disabled", 
                                  (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                        cv2.imshow('Advanced Face Tracking System', frame)
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord(self.config['controls']['quit_key']):
                            break
                        continue
                
                # Detect faces
                detections = self.detect_faces_mediapipe(frame)
                
                # Prepare detections for tracker
                tracker_detections = []
                face_data = []
                
                for detection in detections:
                    x, y, w, h, conf = detection
                    
                    # Convert to format expected by DeepSORT: [left, top, width, height]
                    bbox = [x, y, w, h]
                    tracker_detections.append((bbox, conf, 'person'))
                    
                    # Extract face encoding for recognition
                    encoding = self.extract_face_encoding(frame, [x, y, w, h])
                    name, rec_conf = self.recognize_face(encoding)
                    
                    face_data.append((name, rec_conf))
                
                # Update tracker
                tracks = self.tracker.update_tracks(tracker_detections, frame=frame)
                
                # Get confirmed tracks and enhance with motion prediction
                confirmed_tracks = [track for track in tracks if track.is_confirmed()]
                enhanced_tracks = self.enhance_tracking_with_motion(confirmed_tracks, tracker_detections)
                
                # Update recognition cache for all tracks first
                for i, track in enumerate(enhanced_tracks):
                    track_id = track.track_id
                    if i < len(face_data):
                        name, confidence = face_data[i]
                        self.update_recognition_cache(track_id, name, confidence)
                
                # Filter out overlapping tracks with priority system
                filtered_tracks = self.filter_overlapping_tracks(enhanced_tracks)
                
                # Build final active tracks list
                active_tracks = []
                for track in filtered_tracks:
                    track_id = track.track_id
                    bbox = track.to_ltwh()  # left, top, width, height
                    
                    # Get final recognition result from cache
                    if track_id in self.recognition_cache:
                        name, confidence, _ = self.recognition_cache[track_id]
                    else:
                        name, confidence = "Unknown", 0.0
                    
                    # Only include tracks with reasonable confidence or known faces
                    min_confidence = self.config['recognition']['min_track_confidence']
                    if name != "Unknown" or confidence > min_confidence:
                        active_tracks.append({
                            'track_id': track_id,
                            'bbox': bbox,
                            'name': name,
                            'confidence': confidence
                        })
                
                # Check for duplicate identities and resolve conflicts
                active_tracks = self.check_duplicate_identity(active_tracks)
                # Clean old cache entries and track history
                self.clean_recognition_cache()
                self.clean_track_history()
                
                # Draw results
                self.draw_results(frame, active_tracks)
                
                # Calculate FPS
                self.calculate_fps()
            
            cv2.imshow('Advanced Face Tracking System', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord(self.config['controls']['quit_key']):
                break
            elif key == ord(self.config['controls']['screenshot_key']):
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                screenshot_path = f"screenshot_{timestamp}.jpg"
                cv2.imwrite(screenshot_path, frame)
                print(f"📸 Screenshot saved: {screenshot_path}")
            elif key == ord(self.config['controls']['reset_key']):
                # Reset tracker with config parameters
                self.tracker = DeepSort(
                    max_age=self.config['deepsort']['max_age'],
                    n_init=self.config['deepsort']['n_init'],
                    nms_max_overlap=self.config['deepsort']['nms_max_overlap'],
                    max_cosine_distance=self.config['deepsort']['max_cosine_distance'],
                    nn_budget=self.config['deepsort']['nn_budget']
                )
                self.recognition_cache.clear()
                self.blacklisted_tracks.clear()
                self.track_history.clear()
                print("🔄 Tracker reset!")
            elif key == 32:  # SPACE key
                paused = not paused
                print(f"⏸️ {'Paused' if paused else 'Resumed'}")
        
        self.cap.release()
        cv2.destroyAllWindows()
        
        print(f"\n📊 Session Statistics:")
        print(f"   Total frames processed: {self.frame_count}")
        print(f"   Registered users: {len(self.face_names)}")
        print(f"   Average FPS: {self.current_fps:.1f}")

    def __del__(self):
        """Cleanup"""
        if hasattr(self, 'cap'):
            self.cap.release()
        cv2.destroyAllWindows()

    def smooth_bbox(self, track_id, new_bbox):
        """Apply smoothing to bbox for responsive but stable tracking"""
        smoothing_factor = self.config['tracking']['smoothing_factor']
        
        if not hasattr(self, 'smoothed_bboxes'):
            self.smoothed_bboxes = {}
        
        if track_id not in self.smoothed_bboxes:
            # First time, use the bbox as is
            self.smoothed_bboxes[track_id] = new_bbox
            return new_bbox
        
        # Get previous smoothed bbox
        prev_bbox = self.smoothed_bboxes[track_id]
        
        # Calculate motion speed to adjust smoothing
        prev_center_x = prev_bbox[0] + prev_bbox[2] / 2
        prev_center_y = prev_bbox[1] + prev_bbox[3] / 2
        new_center_x = new_bbox[0] + new_bbox[2] / 2
        new_center_y = new_bbox[1] + new_bbox[3] / 2
        
        motion_distance = np.sqrt((new_center_x - prev_center_x)**2 + (new_center_y - prev_center_y)**2)
        
        # Adaptive smoothing: less smoothing for fast motion
        if motion_distance > 50:  # Fast motion
            adaptive_smoothing = smoothing_factor * 0.3  # Less smoothing
        elif motion_distance > 20:  # Medium motion
            adaptive_smoothing = smoothing_factor * 0.6
        else:  # Slow motion
            adaptive_smoothing = smoothing_factor  # Full smoothing
        
        # Apply exponential smoothing
        smoothed_bbox = [
            prev_bbox[0] * adaptive_smoothing + new_bbox[0] * (1 - adaptive_smoothing),  # x
            prev_bbox[1] * adaptive_smoothing + new_bbox[1] * (1 - adaptive_smoothing),  # y
            prev_bbox[2] * adaptive_smoothing + new_bbox[2] * (1 - adaptive_smoothing),  # w
            prev_bbox[3] * adaptive_smoothing + new_bbox[3] * (1 - adaptive_smoothing)   # h
        ]
        
        self.smoothed_bboxes[track_id] = smoothed_bbox
        return smoothed_bbox

if __name__ == "__main__":
    try:
        tracker = AdvancedFaceTracker()
        tracker.run()
    except KeyboardInterrupt:
        print("\n🛑 Program interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
