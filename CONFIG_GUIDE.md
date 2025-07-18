# 📝 Hướng dẫn cấu hình hệ thống

## 🔧 File cấu hình: `config.json`

File `config.json` chứa tất cả các thông số có thể điều chỉnh để tối ưu hệ thống theo nhu cầu sử dụng.

## 📂 Các nhóm cấu hình chính

### 1. **Camera Settings** (`camera`)
```json
{
  "device_id": 0,          // ID camera (0, 1, 2...)
  "width": 1280,           // Độ phân giải chiều rộng
  "height": 720,           // Độ phân giải chiều cao
  "fps": 30,               // Khung hình/giây
  "flip_horizontal": true  // Lật ngang hình ảnh
}
```

### 2. **Recognition Settings** (`recognition`)
```json
{
  "threshold": 0.80,           // Ngưỡng nhận diện (0.0-1.0)
  "cache_duration": 30,        // Thời gian cache (giây)
  "min_track_confidence": 0.3, // Độ tin cậy tối thiểu
  "duplicate_detection": true, // Bật tính năng phát hiện trùng lặp
  "blacklist_enabled": true    // Bật blacklist cho tracks
}
```

### 3. **Face Detection** (`mediapipe`)
```json
{
  "model_selection": 1,            // Model MediaPipe (0=gần, 1=xa)
  "min_detection_confidence": 0.7, // Độ tin cậy phát hiện
  "face_detection_confidence": 0.8,// Ngưỡng lọc khuôn mặt
  "min_face_size": 60,            // Kích thước tối thiểu
  "max_aspect_ratio": 1.4,        // Tỷ lệ khung hình tối đa
  "min_aspect_ratio": 0.7         // Tỷ lệ khung hình tối thiểu
}
```

### 4. **Tracking Settings** (`deepsort`)
```json
{
  "max_age": 30,              // Tuổi thọ track tối đa
  "n_init": 5,                // Số frame xác nhận track
  "nms_max_overlap": 0.7,     // Ngưỡng NMS overlap
  "max_cosine_distance": 0.2, // Khoảng cách cosine tối đa
  "nn_budget": 100,           // Giới hạn budget
  "use_gpu": true            // Sử dụng GPU
}
```

### 5. **UI Customization** (`ui`)
```json
{
  "title": "ADVANCED FACE TRACKING SYSTEM",
  "title_color": [0, 255, 255],    // Màu tiêu đề (BGR)
  "known_color": [0, 255, 0],      // Màu người quen (BGR)
  "unknown_color": [0, 0, 255],    // Màu người lạ (BGR)
  "font_scale": {
    "title": 1.0,                 // Kích thước font tiêu đề
    "stats": 0.5,                 // Kích thước font thống kê
    "labels": 0.5                 // Kích thước font nhãn
  }
}
```

### 6. **Debug Settings** (`debug`)
```json
{
  "print_duplicate_detection": true,  // In log phát hiện trùng lặp
  "print_track_reassignment": true,   // In log chuyển đổi track
  "print_fps_updates": false,         // In log cập nhật FPS
  "save_debug_images": false          // Lưu ảnh debug
}
```

## 🎯 Tối ưu cho các trường hợp sử dụng

### **Độ chính xác cao** (ít false positive)
```json
{
  "recognition": {
    "threshold": 0.85
  },
  "mediapipe": {
    "face_detection_confidence": 0.9
  }
}
```

### **Hiệu suất cao** (ít lag)
```json
{
  "camera": {
    "width": 640,
    "height": 480,
    "fps": 15
  },
  "performance": {
    "fps_update_interval": 15
  }
}
```

### **Phát hiện nhạy** (nhiều khuôn mặt)
```json
{
  "recognition": {
    "threshold": 0.70
  },
  "mediapipe": {
    "face_detection_confidence": 0.6,
    "min_face_size": 40
  }
}
```

## 🔄 Cách áp dụng thay đổi

1. **Chỉnh sửa** file `config.json`
2. **Khởi động lại** hệ thống: `python main.py`
3. **Test** với các thông số mới

## ⚡ Thông số khuyến nghị

### **RTX 4090 (High Performance)**
```json
{
  "camera": {"width": 1920, "height": 1080},
  "recognition": {"threshold": 0.80},
  "deepsort": {"use_gpu": true, "half_precision": true}
}
```

### **CPU Only (Compatibility)**
```json
{
  "camera": {"width": 640, "height": 480},
  "recognition": {"threshold": 0.75},
  "deepsort": {"use_gpu": false, "half_precision": false}
}
```

## 🚨 Lưu ý quan trọng

- **Backup** file config trước khi thay đổi
- **Test** từng thay đổi một để xác định tác động
- **Recognition threshold** quá cao → nhiều Unknown
- **Recognition threshold** quá thấp → nhiều false positive
- **Camera FPS** cao → tốn tài nguyên
- **Min face size** nhỏ → phát hiện nhiều noise
