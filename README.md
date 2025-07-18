# 🔥 GPU-Accelerated Face Recognition System

## 📋 Tệp tin chính

- **`add_user.py`** - Đăng ký người dùng mới (chỉ 1 khuôn mặt)
- **`main.py`** - Hệ thống nhận diện khuôn mặt với GPU acceleration
- **`requirements.txt`** - Danh sách thư viện cần thiết

## 🚀 Cách sử dụng

### 1. Cài đặt thư viện
```bash
pip install -r requirements.txt
```

### 2. Đăng ký người dùng mới
```bash
python add_user.py
```

### 3. Chạy hệ thống nhận diện
```bash
python main.py
```

## ⌨️ Phím tắt

- **'q'** - Thoát
- **'s'** - Chụp ảnh màn hình
- **'r'** - Reset tracker
- **SPACE** - Tạm dừng/Tiếp tục

## 🎯 Tính năng

- ✅ GPU acceleration với RTX 4090
- ✅ Đăng ký 1 khuôn mặt duy nhất
- ✅ Multi-person tracking với DeepSORT
- ✅ Ngăn chặn duplicate identity
- ✅ Real-time face recognition
- ✅ Professional UI với thông tin chi tiết

## 📊 Hiệu suất

- **FPS**: ~15-20 với RTX 4090
- **Ngưỡng nhận diện**: 0.80 (rất nghiêm ngặt)
- **Độ chính xác**: Cao với ngăn chặn false positive
