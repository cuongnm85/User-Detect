#!/usr/bin/env python3
"""
🎬 RTSP Streaming Face Detection System
Professional system with accurate face recognition
Uses OpenCV VideoWriter for RTSP-like streaming
"""

import cv2
import json
import pickle
import torch
import numpy as np
import time
import threading
from datetime import datetime

# MediaPipe
import mediapipe as mp

# FaceNet and MTCNN
from facenet_pytorch import MTCNN, InceptionResnetV1

# DeepSORT
from deep_sort_realtime.deepsort_tracker import DeepSort

class RTSPFaceDetector:
    def __init__(self, config_path="config.json"):
        print("🚀 Starting RTSP Face Detection System")
        
        # Load configuration
        self.load_config(config_path)
        
        # Device setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🎮 GPU Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"🔥 GPU Device: {torch.cuda.get_device_name()}")
        
        # Initialize models
        self.init_models()
        
        # Load face database
        self.load_face_database()
        
        # Setup camera input
        self.setup_camera()
        
        # RTSP settings
        self.rtsp_port = 8554
        self.rtsp_path = "/live"
        self.stream_width = 1280
        self.stream_height = 720
        self.stream_fps = 25
        self.stream_type = "RTMP"  # Will be set by streaming method
        
        # Tracking variables
        self.recognition_cache = {}
        self.cache_duration = 30
        self.recognition_threshold = 0.80
        self.blacklisted_tracks = set()
        self.track_history = {}
        self.max_history_length = 5
        self.motion_prediction_enabled = True
        self.frame_count = 0
        
        # Camera setup
        self.setup_camera()
        
        print("✅ RTSP System initialized successfully!")
        print(f"👥 Loaded {len(self.face_names)} registered users")

    def load_config(self, config_path):
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
        except FileNotFoundError:
            print(f"⚠️ Config file {config_path} not found! Using defaults.")
            self.config = {
                "camera": {"device_id": 0, "width": 1280, "height": 720, "fps": 25, "flip_horizontal": True},
                "recognition": {"threshold": 0.80},
                "database": {
                    "encodings_file": "face_database/face_encodings.pkl",
                    "users_info_file": "face_database/users_info.json"
                }
            }

    def init_models(self):
        """Initialize AI models"""
        print("🧠 Initializing AI models...")
        
        # MediaPipe Face Detection
        mp_face_detection = mp.solutions.face_detection
        self.face_detector = mp_face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.7
        )
        
        # MTCNN and FaceNet
        self.mtcnn = MTCNN(
            image_size=160, 
            margin=0, 
            device=self.device, 
            keep_all=True
        )
        self.facenet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        
        # DeepSORT tracker
        self.tracker = DeepSort(
            max_age=30,
            n_init=5,
            nms_max_overlap=0.7,
            max_cosine_distance=0.2,
            nn_budget=100,
            embedder="mobilenet",
            half=True,
            bgr=True,
            embedder_gpu=torch.cuda.is_available(),
            embedder_model_name=None,
            embedder_wts=None,
            polygon=False,
            today=None
        )
        
        print("✅ Models initialized")

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
        
        print(f"📚 Loaded {len(self.face_encodings)} face encodings")

    def setup_camera(self):
        """Setup camera input (webcam or RTSP)"""
        input_type = self.config['camera'].get('input_type', 'webcam').lower()
        
        print(f"🎥 Setting up camera input: {input_type}")
        
        if input_type == 'rtsp':
            rtsp_url = self.config['camera'].get('rtsp_url', '')
            if not rtsp_url:
                print("❌ RTSP URL not configured, falling back to webcam")
                input_type = 'webcam'
            else:
                print(f"📡 Connecting to RTSP: {rtsp_url}")
                self.cap = cv2.VideoCapture(rtsp_url)
                
                # Set buffer size for RTSP to reduce latency
                buffer_size = self.config['camera'].get('buffer_size', 1)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, buffer_size)
                
                # Set connection timeout
                timeout = self.config['camera'].get('connection_timeout', 10)
                self.cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, timeout * 1000)
        
        if input_type == 'webcam' or not self.cap.isOpened():
            if input_type == 'rtsp':
                print("⚠️ RTSP connection failed, falling back to webcam")
            
            device_id = self.config['camera'].get('device_id', 0)
            print(f"📹 Using webcam device: {device_id}")
            self.cap = cv2.VideoCapture(device_id)
        
        if not self.cap.isOpened():
            raise Exception("❌ Could not open camera input")
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.stream_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.stream_height)
        self.cap.set(cv2.CAP_PROP_FPS, self.stream_fps)
        
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
        """Detect faces using MediaPipe with NMS"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame = cv2.GaussianBlur(rgb_frame, (3, 3), 0.5)
        
        results = self.face_detector.process(rgb_frame)
        
        detections = []
        if results.detections:
            h, w, _ = frame.shape
            raw_detections = []
            
            for detection in results.detections:
                confidence = detection.score[0]
                
                if confidence > 0.6:
                    bbox = detection.location_data.relative_bounding_box
                    
                    x = max(0, int(bbox.xmin * w))
                    y = max(0, int(bbox.ymin * h))
                    width = min(int(bbox.width * w), w - x)
                    height = min(int(bbox.height * h), h - y)
                    
                    if width > 40 and height > 40 and 0.5 <= width/height <= 2.0:
                        raw_detections.append([x, y, width, height, confidence])
            
            if raw_detections:
                detections = self.apply_nms(raw_detections, 0.3)
        
        return detections

    def apply_nms(self, detections, iou_threshold=0.3):
        """Apply Non-Maximum Suppression"""
        if not detections:
            return []
        
        import numpy as np
        
        boxes = np.array([[x, y, x+w, y+h] for x, y, w, h, _ in detections])
        scores = np.array([conf for _, _, _, _, conf in detections])
        
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            if order.size == 1:
                break
            
            xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
            yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
            xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
            yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
            
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            intersection = w * h
            
            union = areas[i] + areas[order[1:]] - intersection
            iou = intersection / union
            
            order = order[1:][iou <= iou_threshold]
        
        return [detections[i] for i in keep]

    def extract_face_encoding(self, frame, bbox):
        """Extract face encoding using FaceNet"""
        try:
            x, y, w, h = bbox[:4]
            
            margin = 20
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(frame.shape[1], x + w + margin)
            y2 = min(frame.shape[0], y + h + margin)
            
            face_crop = frame[y1:y2, x1:x2]
            
            if face_crop.size == 0:
                return None
            
            rgb_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            face_tensor = self.mtcnn(rgb_face)
            
            if face_tensor is None:
                return None
            
            if len(face_tensor.shape) == 4:
                face_tensor = face_tensor[0]
            
            face_tensor = face_tensor.unsqueeze(0).to(self.device)
            
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
            similarities = []
            for known_encoding in self.face_encodings:
                encoding_norm = encoding / np.linalg.norm(encoding)
                known_encoding_norm = known_encoding / np.linalg.norm(known_encoding)
                similarity = np.dot(encoding_norm, known_encoding_norm)
                similarities.append(similarity)
            
            max_similarity = max(similarities)
            best_match_index = similarities.index(max_similarity)
            
            if max_similarity >= self.recognition_threshold:
                return self.face_names[best_match_index], max_similarity
            else:
                return "Unknown", max_similarity
                
        except Exception as e:
            print(f"⚠️ Recognition error: {e}")
            return "Unknown", 0.0

    def update_recognition_cache(self, track_id, new_name, new_confidence):
        """Update recognition cache with temporal consistency"""
        if track_id in self.blacklisted_tracks:
            return "Unknown", max(0.2, new_confidence * 0.3)
        
        current_time = time.time()
        
        if track_id in self.recognition_cache:
            cached_name, cached_confidence, _ = self.recognition_cache[track_id]
            
            if new_name == cached_name:
                updated_confidence = min(1.0, (cached_confidence + new_confidence) / 2)
                self.recognition_cache[track_id] = (new_name, updated_confidence, current_time)
                return new_name, updated_confidence
            else:
                if new_confidence > cached_confidence:
                    self.recognition_cache[track_id] = (new_name, new_confidence, current_time)
                    return new_name, new_confidence
                else:
                    return cached_name, cached_confidence
        else:
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

    def process_frame(self, frame):
        """Process frame with face detection and recognition"""
        # Detect faces
        detections = self.detect_faces_mediapipe(frame)
        
        # Prepare detections for tracker
        tracker_detections = []
        face_data = []
        
        for detection in detections:
            x, y, w, h, conf = detection
            bbox = [x, y, w, h]
            tracker_detections.append((bbox, conf, 'person'))
            
            # Extract face encoding for recognition
            encoding = self.extract_face_encoding(frame, [x, y, w, h])
            name, rec_conf = self.recognize_face(encoding)
            face_data.append((name, rec_conf))
        
        # Update tracker
        tracks = self.tracker.update_tracks(tracker_detections, frame=frame)
        confirmed_tracks = [track for track in tracks if track.is_confirmed()]
        
        # Update recognition cache
        for i, track in enumerate(confirmed_tracks):
            track_id = track.track_id
            if i < len(face_data):
                name, confidence = face_data[i]
                self.update_recognition_cache(track_id, name, confidence)
        
        # Build final tracks
        active_tracks = []
        for track in confirmed_tracks:
            track_id = track.track_id
            bbox = track.to_ltwh()
            
            if track_id in self.recognition_cache:
                name, confidence, _ = self.recognition_cache[track_id]
            else:
                name, confidence = "Unknown", 0.0
            
            if name != "Unknown" or confidence > 0.3:
                active_tracks.append({
                    'track_id': track_id,
                    'bbox': bbox,
                    'name': name,
                    'confidence': confidence
                })
        
        # Draw results
        for track in active_tracks:
            bbox = track['bbox']
            track_id = track['track_id']
            name = track.get('name', 'Unknown')
            confidence = track.get('confidence', 0.0)
            
            x, y, w, h = map(int, bbox)
            
            # Color coding
            if name == "Unknown":
                color = (0, 0, 255)  # Red
            else:
                color = (0, 255, 0)  # Green
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
            
            # Draw label
            label = f"{track_id}: {name}"
            if confidence > 0:
                label += f" ({confidence:.2f})"
            
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            label_y = max(y - 10, label_h + 10)
            cv2.rectangle(frame, (x, label_y - label_h - 10), (x + label_w + 10, label_y + 5), color, -1)
            cv2.putText(frame, label, (x + 5, label_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Add overlay info
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, f"RTSP-LIKE STREAM - {timestamp}", (10, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        info_text = f"Detections: {len(detections)} | Frame: {self.frame_count} | Port: {self.rtsp_port}"
        cv2.putText(frame, info_text, (10, self.stream_height - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Clean cache periodically
        if self.frame_count % 300 == 0:
            self.clean_recognition_cache()
        
        return frame

    def start_video_output(self):
        """Start MJPEG HTTP server for streaming"""
        try:
            import http.server
            import socketserver
            import threading
            from urllib.parse import urlparse
            
            # Create a simple HTTP server for MJPEG streaming
            class MJPEGHandler(http.server.BaseHTTPRequestHandler):
                def __init__(self, detector_instance, *args, **kwargs):
                    self.detector = detector_instance
                    super().__init__(*args, **kwargs)
                
                def do_GET(self):
                    if self.path == '/stream':
                        self.send_response(200)
                        self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
                        self.send_header('Cache-Control', 'no-cache')
                        self.send_header('Access-Control-Allow-Origin', '*')
                        self.end_headers()
                        
                        try:
                            while True:
                                if hasattr(self.detector, 'current_frame') and self.detector.current_frame is not None:
                                    # Encode frame as JPEG
                                    _, jpeg = cv2.imencode('.jpg', self.detector.current_frame, 
                                                         [cv2.IMWRITE_JPEG_QUALITY, 85])
                                    
                                    # Send MJPEG frame
                                    self.wfile.write(b'--frame\r\n')
                                    self.send_header('Content-Type', 'image/jpeg')
                                    self.send_header('Content-Length', len(jpeg))
                                    self.end_headers()
                                    self.wfile.write(jpeg.tobytes())
                                    self.wfile.write(b'\r\n')
                                
                                time.sleep(0.04)  # ~25 FPS
                        except Exception as e:
                            print(f"⚠️ Streaming interrupted: {e}")
                    else:
                        # Serve a simple web page
                        self.send_response(200)
                        self.send_header('Content-Type', 'text/html')
                        self.end_headers()
                        html = f"""
                        <!DOCTYPE html>
                        <html>
                        <head><title>🎬 RTSP-like Face Detection Stream</title></head>
                        <body style="background: #000; color: #fff; text-align: center; font-family: Arial;">
                            <h1>🎬 Face Detection Live Stream</h1>
                            <img src="/stream" style="border: 2px solid #00ffff; max-width: 90%;">
                            <p>Stream URL: http://localhost:{self.rtsp_port}/stream</p>
                            <p>For VLC: Media → Open Network Stream → http://localhost:{self.rtsp_port}/stream</p>
                        </body>
                        </html>
                        """
                        self.wfile.write(html.encode())
            
            # Initialize current frame storage
            self.current_frame = None
            
            # Start HTTP server in background thread
            def run_server():
                handler = lambda *args, **kwargs: MJPEGHandler(self, *args, **kwargs)
                with socketserver.TCPServer(("", self.rtsp_port), handler) as httpd:
                    print(f"🎬 MJPEG HTTP server started on port {self.rtsp_port}")
                    print(f"📺 Stream URL: http://localhost:{self.rtsp_port}/stream")
                    print(f"🌐 Web interface: http://localhost:{self.rtsp_port}/")
                    httpd.serve_forever()
            
            server_thread = threading.Thread(target=run_server, daemon=True)
            server_thread.start()
            
            time.sleep(1)
            return True
            
        except Exception as e:
            print(f"❌ Failed to start HTTP server: {e}")
            return False

    def run(self):
        """Main processing loop"""
        print("\n🎯 Starting RTSP-like Face Detection Stream")
        
        # Start streaming server
        if not self.start_video_output():
            print("❌ Failed to start streaming server")
            return
        
        print("Press Ctrl+C to stop")
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ Failed to read from camera! Attempting to reconnect...")
                    if self.reconnect_camera():
                        print("✅ Camera reconnected successfully")
                        continue
                    else:
                        print("❌ Failed to reconnect camera, stopping stream...")
                        break
                
                self.frame_count += 1
                
                # Flip frame if configured
                if self.config['camera'].get('flip_horizontal', True):
                    frame = cv2.flip(frame, 1)
                
                # Process frame
                processed_frame = self.process_frame(frame)
                
                # Update current frame for streaming
                self.current_frame = processed_frame
                
                # Control FPS
                time.sleep(1.0 / self.stream_fps)
                
                # Print status every 5 seconds
                if self.frame_count % (self.stream_fps * 5) == 0:
                    print(f"📊 Processed {self.frame_count} frames | Stream: http://localhost:{self.rtsp_port}/stream")
                
        except KeyboardInterrupt:
            print("\n⏹️ Stopping stream...")
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()

    def cleanup(self):
        """Cleanup resources"""
        print("🧹 Cleaning up...")
        
        if hasattr(self, 'cap') and self.cap:
            self.cap.release()
        
        print("✅ Cleanup complete")

def main():
    """Main function"""
    try:
        detector = RTSPFaceDetector()
        detector.run()
    except Exception as e:
        print(f"❌ System error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
