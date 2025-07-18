"""
Professional Single Face Registration System with GPU Acceleration
Hệ thống đăng ký khuôn mặt chuyên nghiệp - CHỈ ĐĂNG KÝ 1 KHUÔN MẶT
Sử dụng FaceNet + MediaPipe cho độ chính xác cao trên GPU RTX 4090
"""

import cv2
import numpy as np
import pickle
import json
import os
import torch
import mediapipe as mp
from facenet_pytorch import MTCNN, InceptionResnetV1
from datetime import datetime
import time

class SingleFaceRegistration:
    def __init__(self):
        """Khởi tạo hệ thống đăng ký với GPU acceleration"""
        print("🚀 Initializing GPU-Accelerated Face Registration System")
        print(f"🔥 GPU Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"🎮 GPU Device: {torch.cuda.get_device_name()}")
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Database paths
        self.database_path = "face_database"
        self.encodings_file = os.path.join(self.database_path, "face_encodings.pkl")
        self.users_file = os.path.join(self.database_path, "users_info.json")
        self.photos_dir = os.path.join(self.database_path, "photos")
        
        # Tạo thư mục nếu chưa có
        os.makedirs(self.database_path, exist_ok=True)
        os.makedirs(self.photos_dir, exist_ok=True)
        
        # Load MediaPipe Face Detection
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_detector = self.mp_face_detection.FaceDetection(
            model_selection=1,  # 1 for better accuracy
            min_detection_confidence=0.7
        )
        
        # Load FaceNet models
        print("📡 Loading FaceNet models...")
        self.mtcnn = MTCNN(
            image_size=160,
            margin=0,
            min_face_size=80,
            thresholds=[0.6, 0.7, 0.7],  # Stricter thresholds
            factor=0.709,
            post_process=True,
            device=self.device,
            keep_all=False  # Only keep the most confident face
        )
        
        self.facenet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        
        # Load existing database
        self.face_encodings = []
        self.face_names = []
        self.users_info = {}
        self.load_database()
        
        # Camera setup
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("✅ System initialized successfully!")

    def load_database(self):
        """Load existing database"""
        try:
            if os.path.exists(self.encodings_file):
                with open(self.encodings_file, 'rb') as f:
                    data = pickle.load(f)
                    self.face_encodings = data.get('encodings', [])
                    self.face_names = data.get('names', [])
                print(f"📚 Loaded {len(self.face_names)} existing face encodings")
            
            if os.path.exists(self.users_file):
                with open(self.users_file, 'r', encoding='utf-8') as f:
                    self.users_info = json.load(f)
                print(f"👥 Loaded {len(self.users_info)} user profiles")
                
        except Exception as e:
            print(f"⚠️ Error loading database: {e}")
            self.face_encodings = []
            self.face_names = []
            self.users_info = {}

    def save_database(self):
        """Save database to files"""
        try:
            # Save encodings
            data = {
                'encodings': self.face_encodings,
                'names': self.face_names
            }
            with open(self.encodings_file, 'wb') as f:
                pickle.dump(data, f)
            
            # Save user info
            with open(self.users_file, 'w', encoding='utf-8') as f:
                json.dump(self.users_info, indent=2, fp=f, ensure_ascii=False)
            
            print("💾 Database saved successfully!")
            return True
        except Exception as e:
            print(f"❌ Error saving database: {e}")
            return False

    def detect_single_face(self, frame):
        """Detect exactly ONE face using MediaPipe"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detector.process(rgb_frame)
        
        if not results.detections:
            return None, "Không phát hiện khuôn mặt nào"
        
        if len(results.detections) > 1:
            return None, f"Phát hiện {len(results.detections)} khuôn mặt. Chỉ được có 1 khuôn mặt trong khung hình!"
        
        # Get the single detection
        detection = results.detections[0]
        confidence = detection.score[0]
        
        if confidence < 0.8:
            return None, f"Khuôn mặt không đủ rõ ràng (confidence: {confidence:.2f})"
        
        # Extract bounding box
        h, w, _ = frame.shape
        bbox = detection.location_data.relative_bounding_box
        
        x = int(bbox.xmin * w)
        y = int(bbox.ymin * h)
        width = int(bbox.width * w)
        height = int(bbox.height * h)
        
        # Ensure bbox is within frame
        x = max(0, x)
        y = max(0, y)
        width = min(width, w - x)
        height = min(height, h - y)
        
        return (x, y, width, height), f"Phát hiện 1 khuôn mặt (confidence: {confidence:.2f})"

    def extract_face_encoding(self, frame, bbox):
        """Extract face encoding using FaceNet"""
        try:
            x, y, w, h = bbox
            
            # Add margin around face
            margin = 20
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(frame.shape[1], x + w + margin)
            y2 = min(frame.shape[0], y + h + margin)
            
            face_crop = frame[y1:y2, x1:x2]
            
            if face_crop.size == 0:
                return None
            
            # Convert to RGB and detect face with MTCNN
            rgb_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            
            # Detect and align face
            face_tensor = self.mtcnn(rgb_face)
            
            if face_tensor is None:
                return None
            
            # Get face encoding
            face_tensor = face_tensor.unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                encoding = self.facenet(face_tensor)
                encoding = encoding.cpu().numpy().flatten()
            
            return encoding
            
        except Exception as e:
            print(f"❌ Error extracting face encoding: {e}")
            return None

    def check_face_exists(self, new_encoding, threshold=0.6):
        """Check if face already exists in database"""
        if not self.face_encodings or new_encoding is None:
            return False, None, 0
        
        # Calculate similarities with all existing faces
        similarities = []
        for existing_encoding in self.face_encodings:
            # Cosine similarity
            dot_product = np.dot(new_encoding, existing_encoding)
            norm_a = np.linalg.norm(new_encoding)
            norm_b = np.linalg.norm(existing_encoding)
            
            if norm_a == 0 or norm_b == 0:
                similarity = 0
            else:
                similarity = dot_product / (norm_a * norm_b)
            
            similarities.append(similarity)
        
        max_similarity = max(similarities)
        best_match_idx = similarities.index(max_similarity)
        
        if max_similarity > threshold:
            return True, self.face_names[best_match_idx], max_similarity
        
        return False, None, max_similarity

    def capture_and_register(self):
        """Capture face and register new user"""
        print("\n🎯 CHẾ ĐỘ ĐĂNG KÝ KHUÔN MẶT")
        print("=" * 50)
        print("📋 Hướng dẫn:")
        print("   • Chỉ có 1 người trong khung hình")
        print("   • Nhìn thẳng vào camera")
        print("   • Đảm bảo ánh sáng đủ")
        print("   • Nhấn SPACE để chụp")
        print("   • Nhấn 'q' để thoát")
        print("=" * 50)
        
        captured_frame = None
        face_bbox = None
        face_encoding = None
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("❌ Không thể đọc từ camera!")
                break
            
            frame = cv2.flip(frame, 1)
            display_frame = frame.copy()
            
            # Detect face
            bbox, status_msg = self.detect_single_face(frame)
            
            # Draw status
            cv2.rectangle(display_frame, (0, 0), (frame.shape[1], 100), (0, 0, 0), -1)
            cv2.putText(display_frame, "FACE REGISTRATION SYSTEM", 
                       (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            cv2.putText(display_frame, status_msg, 
                       (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(display_frame, "SPACE: Capture | Q: Quit", 
                       (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Draw face detection
            if bbox:
                x, y, w, h = bbox
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
                cv2.putText(display_frame, "READY TO CAPTURE", 
                           (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow('Face Registration', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == 32 and bbox:  # SPACE key
                print("📸 Capturing face...")
                
                # Extract encoding
                encoding = self.extract_face_encoding(frame, bbox)
                if encoding is None:
                    print("❌ Không thể trích xuất đặc trưng khuôn mặt!")
                    continue
                
                # Check if face already exists
                exists, existing_name, similarity = self.check_face_exists(encoding)
                if exists:
                    print(f"⚠️ Khuôn mặt này đã tồn tại trong database!")
                    print(f"   Tên: {existing_name}")
                    print(f"   Độ tương tự: {similarity:.3f}")
                    
                    choice = input("Tiếp tục đăng ký? (y/N): ").lower().strip()
                    if choice != 'y':
                        continue
                
                captured_frame = frame.copy()
                face_bbox = bbox
                face_encoding = encoding
                print("✅ Đã chụp khuôn mặt thành công!")
                break
        
        cv2.destroyAllWindows()
        
        if captured_frame is None:
            print("❌ Chưa chụp được khuôn mặt!")
            return False
        
        # Get user information
        print("\n📝 Nhập thông tin người dùng:")
        name = input("Tên đầy đủ: ").strip()
        if not name:
            print("❌ Tên không được để trống!")
            return False
        
        if name in self.face_names:
            print(f"⚠️ Tên '{name}' đã tồn tại trong database!")
            choice = input("Ghi đè? (y/N): ").lower().strip()
            if choice != 'y':
                return False
            
            # Remove existing entry
            idx = self.face_names.index(name)
            self.face_names.pop(idx)
            self.face_encodings.pop(idx)
            if name in self.users_info:
                del self.users_info[name]
        
        department = input("Phòng ban (optional): ").strip()
        position = input("Chức vụ (optional): ").strip()
        employee_id = input("Mã nhân viên (optional): ").strip()
        
        # Save photo
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        photo_filename = f"{name}_{timestamp}.jpg"
        photo_path = os.path.join(self.photos_dir, photo_filename)
        
        # Crop and save face
        x, y, w, h = face_bbox
        face_crop = captured_frame[y:y+h, x:x+w]
        cv2.imwrite(photo_path, face_crop)
        
        # Add to database
        self.face_encodings.append(face_encoding)
        self.face_names.append(name)
        
        self.users_info[name] = {
            'name': name,
            'department': department,
            'position': position,
            'employee_id': employee_id,
            'photo_path': photo_path,
            'registered_date': datetime.now().isoformat(),
            'encoding_dim': len(face_encoding)
        }
        
        # Save database
        if self.save_database():
            print(f"\n✅ Đăng ký thành công!")
            print(f"   Tên: {name}")
            print(f"   Phòng ban: {department or 'N/A'}")
            print(f"   Chức vụ: {position or 'N/A'}")
            print(f"   Ảnh: {photo_path}")
            print(f"   Encoding dimension: {len(face_encoding)}")
            return True
        else:
            print("❌ Lỗi lưu database!")
            return False

    def list_users(self):
        """List all registered users"""
        if not self.face_names:
            print("📭 Database trống!")
            return
        
        print(f"\n👥 DANH SÁCH NGƯỜI DÙNG ĐÃ ĐĂNG KÝ ({len(self.face_names)} người):")
        print("=" * 80)
        
        for i, name in enumerate(self.face_names):
            user_info = self.users_info.get(name, {})
            print(f"{i+1:2d}. {name}")
            print(f"     Phòng ban: {user_info.get('department', 'N/A')}")
            print(f"     Chức vụ: {user_info.get('position', 'N/A')}")
            print(f"     Ngày đăng ký: {user_info.get('registered_date', 'N/A')}")
            print()

    def delete_user(self):
        """Delete a user from database"""
        if not self.face_names:
            print("📭 Database trống!")
            return
        
        self.list_users()
        
        try:
            choice = input("\nNhập số thứ tự người cần xóa (hoặc 'q' để thoát): ").strip()
            if choice.lower() == 'q':
                return
            
            idx = int(choice) - 1
            if idx < 0 or idx >= len(self.face_names):
                print("❌ Số thứ tự không hợp lệ!")
                return
            
            name = self.face_names[idx]
            confirm = input(f"Xác nhận xóa '{name}'? (y/N): ").lower().strip()
            if confirm != 'y':
                return
            
            # Remove from database
            self.face_names.pop(idx)
            self.face_encodings.pop(idx)
            
            # Remove user info and photo
            if name in self.users_info:
                user_info = self.users_info[name]
                photo_path = user_info.get('photo_path')
                if photo_path and os.path.exists(photo_path):
                    os.remove(photo_path)
                del self.users_info[name]
            
            if self.save_database():
                print(f"✅ Đã xóa '{name}' thành công!")
            else:
                print("❌ Lỗi lưu database!")
                
        except ValueError:
            print("❌ Vui lòng nhập số!")
        except Exception as e:
            print(f"❌ Lỗi: {e}")

    def run(self):
        """Main menu"""
        while True:
            print("\n" + "="*60)
            print("🎯 SINGLE FACE REGISTRATION SYSTEM")
            print("🔥 GPU-Accelerated with FaceNet + MediaPipe")
            print("="*60)
            print("1. 📷 Đăng ký khuôn mặt mới")
            print("2. 👥 Xem danh sách người dùng")
            print("3. 🗑️  Xóa người dùng")
            print("4. ❌ Thoát")
            print("="*60)
            
            choice = input("Chọn chức năng (1-4): ").strip()
            
            if choice == '1':
                self.capture_and_register()
            elif choice == '2':
                self.list_users()
            elif choice == '3':
                self.delete_user()
            elif choice == '4':
                print("👋 Tạm biệt!")
                break
            else:
                print("❌ Lựa chọn không hợp lệ!")

    def __del__(self):
        """Cleanup"""
        if hasattr(self, 'cap'):
            self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        registration = SingleFaceRegistration()
        registration.run()
    except KeyboardInterrupt:
        print("\n⚠️ Stopped by Ctrl+C")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
