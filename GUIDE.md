# 📋 Hướng dẫn cấu hình hệ thống - config.json

## 🔧 Tổng quan

File `config.json` chứa tất cả các thông số cấu hình cho hệ thống nhận diện khuôn mặt. Mỗi tham số đều có thể điều chỉnh để tối ưu hiệu suất và độ chính xác.

## 📱 Cấu hình hệ thống (system)

```json
"system": {
  "name": "GPU-Accelerated Advanced Face Tracking System",
  "version": "2.0", 
  "description": "Professional face recognition with GPU acceleration and multi-person tracking"
}
```

- **`name`**: Tên hệ thống hiển thị
- **`version`**: Phiên bản hiện tại
- **`description`**: Mô tả chức năng chính

## 📹 Cấu hình camera (camera)

```json
"camera": {
  "device_id": 0,
  "width": 1280,
  "height": 720,
  "fps": 30,
  "flip_horizontal": true
}
```

- **`device_id`**: ID camera (0 = camera mặc định, 1 = camera thứ 2...)
- **`width`**: Độ rộng khung hình (pixel)
- **`height`**: Độ cao khung hình (pixel) 
- **`fps`**: Tốc độ khung hình mong muốn (frames/giây)
- **`flip_horizontal`**: Lật ngang hình ảnh (true/false)

## 🎯 Cấu hình MediaPipe (mediapipe)

```json
"mediapipe": {
  "model_selection": 0,
  "min_detection_confidence": 0.6,
  "face_detection_confidence": 0.6,
  "min_face_size": 40,
  "max_aspect_ratio": 1.5,
  "min_aspect_ratio": 0.6
}
```

- **`model_selection`**: Model phát hiện khuôn mặt
  - `0` = Model nhanh (< 2m)
  - `1` = Model chính xác (2-5m)
- **`min_detection_confidence`**: Ngưỡng tin cậy phát hiện (0.0-1.0)
- **`face_detection_confidence`**: Ngưỡng tin cậy khuôn mặt (0.0-1.0)
- **`min_face_size`**: Kích thước khuôn mặt tối thiểu (pixel)
- **`max_aspect_ratio`**: Tỷ lệ khung hình tối đa (rộng/cao)
- **`min_aspect_ratio`**: Tỷ lệ khung hình tối thiểu (rộng/cao)

## 🔍 Cấu hình MTCNN (mtcnn)

```json
"mtcnn": {
  "image_size": 160,
  "margin": 0,
  "scale_factor": 0.709,
  "thresholds": [0.6, 0.7, 0.7],
  "factor": 0.709,
  "post_process": true,
  "keep_all": true,
  "min_face_size": 40,
  "face_crop_margin": 20
}
```

- **`image_size`**: Kích thước ảnh đầu ra cho FaceNet (160x160)
- **`margin`**: Lề xung quanh khuôn mặt
- **`scale_factor`**: Hệ số scale cho pyramid detection
- **`thresholds`**: Ngưỡng cho 3 stages [P-Net, R-Net, O-Net]
- **`factor`**: Hệ số giảm kích thước pyramid
- **`post_process`**: Xử lý hậu kỳ (true/false)
- **`keep_all`**: Giữ tất cả khuôn mặt phát hiện (true/false)
- **`min_face_size`**: Kích thước khuôn mặt tối thiểu
- **`face_crop_margin`**: Lề cắt ảnh khuôn mặt

## 🧠 Cấu hình FaceNet (facenet)

```json
"facenet": {
  "pretrained_model": "vggface2",
  "encoding_dimension": 512
}
```

- **`pretrained_model`**: Model pre-trained
  - `"vggface2"` = Trained trên VGGFace2 dataset (khuyên dùng)
  - `"casia-webface"` = Trained trên CASIA-WebFace
- **`encoding_dimension`**: Số chiều vector encoding (512 chiều)

## 🎯 Cấu hình DeepSORT (deepsort)

```json
"deepsort": {
  "max_age": 8,
  "n_init": 1,
  "nms_max_overlap": 0.95,
  "max_cosine_distance": 0.6,
  "nn_budget": 200,
  "embedder": "mobilenet",
  "half_precision": true,
  "bgr_format": true,
  "use_gpu": true
}
```

- **`max_age`**: Số frame tối đa trước khi xóa track (8 = responsive)
- **`n_init`**: Số detection cần để confirm track (1 = nhanh nhất)
- **`nms_max_overlap`**: Ngưỡng overlap cho Non-Maximum Suppression (0.95 = chấp nhận overlap cao)
- **`max_cosine_distance`**: Khoảng cách cosine tối đa cho matching (0.6 = linh hoạt)
- **`nn_budget`**: Số lượng features lưu trữ cho mỗi track
- **`embedder`**: Model embedding ("mobilenet" = nhanh)
- **`half_precision`**: Sử dụng FP16 để tăng tốc (true/false)
- **`bgr_format`**: Format màu BGR (true/false)
- **`use_gpu`**: Sử dụng GPU (true/false)

## 🎨 Cấu hình nhận diện (recognition)

```json
"recognition": {
  "threshold": 0.5,
  "cache_duration": 30,
  "min_track_confidence": 0.3,
  "duplicate_detection": true,
  "blacklist_enabled": true
}
```

- **`threshold`**: Ngưỡng cosine similarity để nhận diện (0.5 = vừa phải)
  - Cao hơn = chính xác hơn nhưng khó nhận diện
  - Thấp hơn = dễ nhận diện nhưng có thể sai
- **`cache_duration`**: Thời gian cache nhận diện (giây)
- **`min_track_confidence`**: Confidence tối thiểu để hiển thị track
- **`duplicate_detection`**: Phát hiện duplicate identity (true/false)
- **`blacklist_enabled`**: Kích hoạt blacklist cho Unknown tracks (true/false)

## 🚫 Cấu hình NMS (nms)

```json
"nms": {
  "iou_threshold": 0.3,
  "overlap_threshold": 0.3
}
```

- **`iou_threshold`**: Ngưỡng IoU cho Non-Maximum Suppression
- **`overlap_threshold`**: Ngưỡng overlap cho filtering tracks

## 🏃 Cấu hình tracking (tracking)

```json
"tracking": {
  "priority_weight_age": 0.05,
  "priority_weight_confidence": 1.5,
  "priority_bonus_recognized": 3.0,
  "priority_penalty_unknown": 0.3,
  "priority_penalty_missed": 0.1,
  "max_age_bonus": 5,
  "motion_prediction": true,
  "max_motion_distance": 200,
  "motion_weight": 0.9,
  "kalman_noise": 0.03,
  "prediction_frames": 8,
  "smoothing_factor": 0.95
}
```

### Priority System
- **`priority_weight_age`**: Trọng số tuổi track (track cũ = ưu tiên cao)
- **`priority_weight_confidence`**: Trọng số confidence
- **`priority_bonus_recognized`**: Bonus cho khuôn mặt đã nhận diện
- **`priority_penalty_unknown`**: Penalty cho Unknown faces
- **`priority_penalty_missed`**: Penalty cho missed detections
- **`max_age_bonus`**: Tuổi tối đa để nhận bonus

### Motion Prediction
- **`motion_prediction`**: Kích hoạt dự đoán chuyển động (true/false)
- **`max_motion_distance`**: Khoảng cách chuyển động tối đa (pixel)
- **`motion_weight`**: Trọng số motion prediction (0.9 = ưu tiên cao)
- **`kalman_noise`**: Noise cho Kalman filter (0.03 = ít noise)
- **`prediction_frames`**: Số frames dự đoán trước (8 = responsive)
- **`smoothing_factor`**: Hệ số làm mịn tracking box (0.95 = rất mịn)

## ⚡ Cấu hình hiệu suất (performance)

```json
"performance": {
  "fps_update_interval": 30,
  "max_tracks_display": 10,
  "memory_cleanup_interval": 100
}
```

- **`fps_update_interval`**: Interval cập nhật FPS (frames)
- **`max_tracks_display`**: Số tracks tối đa hiển thị
- **`memory_cleanup_interval`**: Interval dọn dẹp memory (frames)

## 🎨 Cấu hình giao diện (ui)

```json
"ui": {
  "header_height": 120,
  "title": "ADVANCED FACE TRACKING SYSTEM",
  "title_color": [0, 255, 255],
  "background_color": [0, 0, 0],
  "border_color": [255, 255, 255],
  "text_color": [255, 255, 255],
  "known_color": [0, 255, 0],
  "unknown_color": [0, 0, 255],
  "font_scale": {
    "title": 1.0,
    "stats": 0.5,
    "labels": 0.5
  },
  "line_height": 20,
  "label_padding": 10
}
```

### Layout
- **`header_height`**: Chiều cao header (pixel)
- **`title`**: Tiêu đề hệ thống

### Colors (BGR format)
- **`title_color`**: Màu tiêu đề [B, G, R]
- **`background_color`**: Màu nền header [B, G, R]
- **`border_color`**: Màu viền [B, G, R]
- **`text_color`**: Màu chữ [B, G, R]
- **`known_color`**: Màu cho khuôn mặt đã biết [B, G, R]
- **`unknown_color`**: Màu cho khuôn mặt chưa biết [B, G, R]

### Typography
- **`font_scale`**: Kích thước font
  - `title`: Kích thước font tiêu đề
  - `stats`: Kích thước font thống kê
  - `labels`: Kích thước font nhãn
- **`line_height`**: Chiều cao dòng (pixel)
- **`label_padding`**: Padding cho nhãn (pixel)

## 💾 Cấu hình database (database)

```json
"database": {
  "encodings_file": "face_database/face_encodings.pkl",
  "users_info_file": "face_database/users_info.json",
  "photos_directory": "face_database/photos/"
}
```

- **`encodings_file`**: Đường dẫn file lưu face encodings
- **`users_info_file`**: Đường dẫn file thông tin users
- **`photos_directory`**: Thư mục lưu ảnh người dùng

## ⌨️ Cấu hình phím tắt (controls)

```json
"controls": {
  "quit_key": "q",
  "screenshot_key": "s", 
  "reset_key": "r",
  "pause_key": "space"
}
```

- **`quit_key`**: Phím thoát chương trình
- **`screenshot_key`**: Phím chụp màn hình
- **`reset_key`**: Phím reset tracker
- **`pause_key`**: Phím tạm dừng/tiếp tục

## 🔧 Cấu hình debug (debug)

```json
"debug": {
  "print_duplicate_detection": true,
  "print_track_reassignment": true,
  "print_fps_updates": false,
  "save_debug_images": false
}
```

- **`print_duplicate_detection`**: In thông báo phát hiện duplicate
- **`print_track_reassignment`**: In thông báo reassign tracks
- **`print_fps_updates`**: In cập nhật FPS
- **`save_debug_images`**: Lưu ảnh debug

## 🎯 Tối ưu hiệu suất

### Để tăng FPS:
- Giảm `camera.fps`
- Giảm `camera.width` và `camera.height`
- Tăng `recognition.min_track_confidence`
- Giảm `deepsort.nn_budget`

### Để tăng độ chính xác:
- Tăng `recognition.threshold` 
- Giảm `mediapipe.min_detection_confidence`
- Tăng `deepsort.n_init`
- Giảm `deepsort.max_cosine_distance`

### Để tracking mượt hơn:
- Tăng `tracking.smoothing_factor`
- Tăng `tracking.prediction_frames`
- Tăng `tracking.motion_weight`
- Giảm `deepsort.max_age`

## ⚠️ Lưu ý quan trọng

1. **Backup config** trước khi thay đổi
2. **Test từng parameter** một cách riêng biệt
3. **Giá trị threshold** ảnh hưởng trực tiếp đến độ chính xác
4. **Motion parameters** quan trọng cho tracking mượt
5. **GPU settings** cần phù hợp với phần cứng

## 🔄 Khôi phục mặc định

Nếu hệ thống hoạt động không ổn định, xóa file `config.json` và chạy lại. Hệ thống sẽ tự tạo config mặc định.
