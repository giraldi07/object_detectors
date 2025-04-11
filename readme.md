# Object Detection Webcam Application

A modular Python application for real-time object detection using webcam. This application leverages YOLOv8 for object detection and OpenCV for video capture and display.

## Features

- **Real-time object detection**: Identifies multiple objects in a room including furniture, electronics, clothing and more
- **Multi-threaded processing**: Ensures smooth performance with separate threads for video capture and object detection
- **High-quality display**: Supports customizable resolution with default 1280x720
- **Non-mirrored view**: Displays video feed correctly oriented (not mirrored) by default
- **Object counting**: Shows the count of each detected object class
- **Performance metrics**: Displays real-time FPS (Frames Per Second)
- **Screenshot capability**: Save detection results with a single keystroke
- **Customizable settings**: Adjust confidence threshold, camera source, resolution and more

## Requirements

- Python 3.7+
- OpenCV
- NumPy
- Ultralytics YOLOv8

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/giraldi07/object_detectors.git
   cd object-detection-app
   ```

2. Install the required packages:
   ```bash
   pip install opencv-python numpy ultralytics
   ```

3. (Optional) If you want to use a specific YOLOv8 model, download it to the project directory.
   The standard models are:
   - YOLOv8n (nano): Small, fast model
   - YOLOv8s (small): Balance of speed and accuracy
   - YOLOv8m (medium): More accurate but slower
   - YOLOv8l (large): Even more accurate
   - YOLOv8x (extra large): Maximum accuracy

## Usage

### Basic Usage

Run the application with default settings:

```bash
python object_detection_app.py
```

### Command Line Arguments

The application supports several command line arguments:

```bash
python object_detection_app.py [OPTIONS]
```

Options:
- `--camera INT`: Camera device ID (default: 0)
- `--width INT`: Frame width (default: 1280)
- `--height INT`: Frame height (default: 720)
- `--model PATH`: Path to YOLOv8 model (default: "yolov8n.pt")
- `--confidence FLOAT`: Detection confidence threshold (default: 0.35)
- `--no-fps`: Hide FPS counter
- `--flip`: Flip frame horizontally (mirror)
- `--no-count`: Hide object counts

### Example Commands

Using a different camera (e.g., external webcam):
```bash
python object_detection_app.py --camera 1
```

Using a different model with lower resolution:
```bash
python object_detection_app.py --model yolov8m.pt --width 640 --height 480
```

Higher confidence threshold for more precise detections:
```bash
python object_detection_app.py --confidence 0.5
```

Enable mirrored view and hide FPS counter:
```bash
python object_detection_app.py --flip --no-fps
```

## Keyboard Controls

While the application is running:
- `q`: Quit the application
- `f`: Toggle frame flipping (mirroring)
- `s`: Save current frame as screenshot (saved with timestamp in filename)

## Detectable Objects

YOLOv8 can detect 80 different classes of objects, including:

- Furniture: chairs, sofas, tables, beds
- Electronics: TV, laptop, cell phone, keyboard, mouse
- Kitchen items: cup, fork, knife, spoon, bowl, bottle
- Personal items: backpack, umbrella, handbag
- Clothing: tie
- And many more common objects

## System Architecture

The application is built with a modular architecture:

- `VideoCapture`: Handles webcam frame capture with threading
- `ObjectDetector`: Processes frames using YOLOv8 in a separate thread
- `DisplayProcessor`: Renders detection results on frames
- `FPSCounter`: Calculates and smooths frame rate
- `ObjectDetectionApp`: Main application class that coordinates all components

## Performance Tips

- For better performance on slower computers, try:
  - Using a smaller model (yolov8n.pt)
  - Reducing resolution (e.g., --width 640 --height 480)
  - Increasing confidence threshold (e.g., --confidence 0.4)

- For better detection accuracy, try:
  - Using a larger model (yolov8m.pt or yolov8l.pt) 
  - Decreasing confidence threshold (e.g., --confidence 0.25)
  - Ensuring good lighting conditions

## Troubleshooting

### Camera Not Working
- Make sure the correct camera ID is specified
- Check if other applications are using the camera
- Try restarting your computer if the camera is locked

### Low Performance
- Check CPU/GPU usage with task manager
- Try a smaller model or lower resolution
- Close other resource-intensive applications

### Missing Detections
- Try lowering the confidence threshold
- Improve lighting conditions
- Use a larger YOLO model

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) for the object detection model
- [OpenCV](https://opencv.org/) for the computer vision capabilities