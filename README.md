# **Video Braille** 🎥⠠⠃⠗⠁⠊⠇⠇⠑

**Real-time video to Braille converter with color support and advanced image processing**

A cutting-edge Python application that transforms video content into live Braille representations in your terminal, making visual media accessible through tactile Unicode Braille patterns. Features both grayscale and full-color Braille rendering with sophisticated image processing capabilities.

---

## **🌟 Features**

### **Core Functionality**
- **Real-time Conversion**: Live video-to-Braille rendering with frame pacing
- **Dual Rendering Modes**:
  - `braille`: High-contrast grayscale Braille
  - `color-braille`: Full 24-bit color Braille with ANSI escape codes
- **Audio Synchronization**: Optional audio playback via FFmpeg integration
- **Terminal Optimization**: UTF-8 and ANSI support with Windows compatibility

### **Advanced Image Processing**
- **Adaptive Thresholding**: Multiple algorithms (Mean, Gaussian, Otsu)
- **CLAHE Enhancement**: Contrast Limited Adaptive Histogram Equalization
- **Edge Detection**: Canny edge detection with boost options
- **Gamma Correction**: Power-law transformations for brightness control
- **Noise Reduction**: Gaussian blur filtering
- **Custom Aspect Ratios**: Terminal character compensation

### **Robustness & Performance**
- **Smart Restart System**: Automatic recovery from video stream stalls
- **Backend Failover**: Multiple OpenCV capture backends (FFmpeg, MSMF, DirectShow)
- **Stall Detection**: Frame, position, and visual change monitoring
- **Performance Tuning**: FPS limiting, buffering control, and optimization presets

### **Accessibility & Usability**
- **Auto-tuning System**: Automated parameter optimization with scoring
- **Sampling & Logging**: Frame capture, JSON metrics, and performance analysis
- **Preset Configurations**: Optimized settings for different use cases
- **Checkpoint Recovery**: Crash-resistant operation with periodic state saves

---

## **🚀 Installation**

### **Prerequisites**
```bash
# Python 3.7+ required
python --version
```

### **Required Dependencies**
```bash
pip install numpy opencv-python
```

### **Optional Dependencies (Recommended)**
```bash
pip install colorama  # Windows terminal support
# FFmpeg (for audio support)
```

### **Complete Installation**
```bash
git clone https://github.com/hellcatxrp/video_braille.git
cd video_braille
pip install -r requirements.txt  # If available
```

---

## **🎮 Usage**

### **Basic Usage**
```bash
python video_braille.py <video_file>
```

**Important:** Replace `<video_file>` with the path to your video file. If the path contains spaces, wrap it in quotes.

### **Examples**
```bash
# Simple grayscale Braille
python video_braille.py video.mp4

# Path with spaces (requires quotes)
python video_braille.py "D:\Kill Bill Collection\Kill Bill Vol. 1.avi"

# Full-color Braille with audio
python video_braille.py "E:\Movies\movie.mp4" --mode color-braille --audio
```

### **Advanced Configuration**
High-quality color rendering with custom settings:
```bash
python video_braille.py "D:\Videos\movie.avi" \
  --mode color-braille \
  --width 140 \
  --fps-limit 15 \
  --adaptive gaussian \
  --clahe 2.0 \
  --gamma 0.9 \
  --edge-boost \
  --audio \
  --restart-on-stall
```

### **Preset Configurations**
Optimized presets for different scenarios:
```bash
python video_braille.py video.mp4 --preset clarity    # Best quality
python video_braille.py video.mp4 --preset color     # Color optimized
python video_braille.py video.mp4 --preset fast      # Performance focused
```

### **Auto-tuning System**
Automatically find optimal settings:
```bash
python video_braille.py video.mp4 --auto-tune color --tune-seconds 15
python video_braille.py video.mp4 --auto-tune braille --tune-seconds 10
```

---

## **🛠️ Key Parameters**

### **Display Settings**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--width` | 120 | Character width of output |
| `--mode` | braille | Rendering mode (braille/color-braille) |
| `--aspect` | 0.5 | Vertical compensation for terminal cells |
| `--fps-limit` | auto | Maximum playback framerate |

### **Image Processing**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--threshold` | 130 | Dot activation threshold (0-255) |
| `--adaptive` | none | Adaptive thresholding (mean/gaussian/otsu) |
| `--gamma` | 1.0 | Gamma correction (<1 brightens, >1 darkens) |
| `--clahe` | None | CLAHE clip limit for local contrast |
| `--edge-boost` | False | Enable Canny edge enhancement |
| `--blur-kernel` | 0 | Gaussian blur size for noise reduction |

### **Robustness**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--restart-on-stall` | False | Auto-restart on video stalls |
| `--max-restarts` | 50 | Maximum restart attempts |
| `--backend` | auto | OpenCV backend (ffmpeg/msmf/auto) |
| `--audio-resync` | False | Re-sync audio on restarts |

---

## **🎯 How It Works**

### **Braille Encoding Process**
1. **Frame Capture**: OpenCV reads video frames
2. **Preprocessing**: Applies gamma, CLAHE, blur, and edge detection
3. **Thresholding**: Converts to binary using adaptive or fixed thresholds
4. **Cell Mapping**: Maps 2×4 pixel blocks to Braille Unicode (U+2800-U+28FF)
5. **Color Processing**: Averages RGB values per cell for color mode
6. **Terminal Output**: Renders using ANSI escape codes

### **Stall Detection & Recovery**
- **Frame Hash Monitoring**: Detects identical consecutive frames
- **Position Tracking**: Monitors video timestamp progression
- **Visual Change Analysis**: Measures frame-to-frame differences
- **Backend Failover**: Switches OpenCV backends on persistent issues
- **Smart Restart**: Seeks back slightly and resumes playback

---

## **📊 Performance & Quality**

### **Optimization Features**
- **Multi-backend Support**: Automatic failover between video backends
- **Buffer Management**: Minimal buffering for low latency
- **Frame Pacing**: Precise timing control for smooth playback
- **Metrics Collection**: Real-time quality and performance monitoring

### **Quality Metrics**
The auto-tuning system evaluates:
- **Non-blank Ratio**: Proportion of active Braille cells
- **Dot Density**: Average dots per Braille character
- **Transition Rate**: Edge complexity measurement
- **Frame Rate**: Achieved vs. target FPS

---

## **🎨 Examples**

### **Sample Output (Grayscale Braille)**
```
⠀⠀⠀⠀⠀⠀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⢀⣴⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣦⡀⠀⠀⠀
⠀⠀⣠⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⣄⠀⠀
⠀⢰⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡆⠀
⢀⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡀
```

### **Color Braille (with ANSI codes)**
Each character can display in full 24-bit color while maintaining the tactile Braille pattern.

---

## **📝 Command Examples**

```bash
# Basic playbook
python video_braille.py movie.mp4

# High-quality color with audio
python video_braille.py movie.mp4 --mode color-braille --width 140 --audio --preset color

# Sampling and analysis
python video_braille.py movie.mp4 --sample-output frames.txt --log-file metrics.json --sample-seconds 30

# Robust playback with restart protection
python video_braille.py unstable_stream.m3u8 --restart-on-stall --max-restarts 100 --backend ffmpeg

# Advanced processing
python video_braille.py input.mp4 --adaptive gaussian --clahe 2.0 --gamma 0.9 --edge-boost --blur-kernel 3
```

---

## **🔧 Technical Details**

### **Dependencies**
- **Core**: numpy, opencv-python
- **Optional**: colorama (Windows terminal support)
- **External**: FFmpeg (audio playback via ffplay)

### **Supported Formats**
- **Video**: MP4, AVI, MOV, MKV, WebM, and most formats supported by OpenCV
- **Streams**: RTSP, HTTP streams, webcams (device indices)
- **Audio**: Any format supported by FFmpeg

### **Terminal Compatibility**
- **Unicode**: Full Braille Unicode range (U+2800-U+28FF)
- **Color**: 24-bit ANSI color codes (`\x1b[38;2;R;G;Bm`)
- **Windows**: Automatic UTF-8 and ANSI enablement via colorama

---

## **🤝 Contributing**

This project welcomes contributions! Areas for enhancement:
- Additional image processing algorithms
- Performance optimizations
- New output formats (e.g., hardware Braille displays)
- Mobile/embedded platform support
- Real-time streaming improvements

### **Development Setup**
```bash
git clone https://github.com/hellcatxrp/video_braille.git
cd video_braille

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

---

## **🙏 Acknowledgements**

A significant portion of the initial codebase for this project was generated with the assistance of **OpenAI's Codex**. Subsequent edits were made with the help of **Anthropic's Claude**.

### **👤 Human Author**
**hellcatxrp**
- GitHub: [@hellcatxrp](https://github.com/hellcatxrp)
- Repository: [video_braille](https://github.com/hellcatxrp/video_braille)

### **🎯 Other Acknowledgments**
- **OpenCV Community** for robust video processing capabilities
- **Unicode Consortium** for Braille character standardization
- **FFmpeg Project** for multimedia framework support
- **Accessibility Advocates** who inspire inclusive technology development

---

## **🐛 Issues & Support**

If you encounter problems or have suggestions:

1. Check existing [Issues](https://github.com/hellcatxrp/video_braille/issues)
2. Create a detailed issue report with:
   - System information (OS, Python version)
   - Video format and source details
   - Error messages and logs
   - Steps to reproduce

---

## **⭐ Star this repository if you find it useful! ⭐**

*Making the visual world accessible, one Braille dot at a time.*

The sophisticated image processing pipeline and robust error handling make it suitable for both experimental use and practical applications.

![Video Braille Demo](https://github.com/user-attachments/assets/9d10b4a0-78bf-4fae-bdc4-765e48fdf264)





