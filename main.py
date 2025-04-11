"""
Object Detection Webcam Application (Improved)
-----------------------------------
A modular Python application for real-time object detection using webcam.
Uses YOLOv8 for detection and OpenCV for video capture and display.
Features:
- Non-mirrored display
- Multi-threaded processing
- Detection of various objects in a room
- Smooth performance
"""

import os
import time
import cv2
import numpy as np
import threading
import queue
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from ultralytics import YOLO

@dataclass
class DetectedObject:
    """Class for storing information about detected objects"""
    class_id: int
    class_name: str
    confidence: float
    box: Tuple[int, int, int, int]  # x1, y1, x2, y2
    color: Tuple[int, int, int]


class VideoCapture:
    """Class for handling video capture from webcam with threading support"""
    
    def __init__(self, src: int = 0, width: int = 1280, height: int = 720):
        """
        Initialize the video capture
        
        Args:
            src: Camera source (default is 0 for primary webcam)
            width: Frame width
            height: Frame height
        """
        self.src = src
        self.width = width
        self.height = height
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        # Try to increase FPS by setting lower exposure
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # Disable autofocus to avoid lag
        
        # Threading properties
        self.frame_read = None
        self.frame_available = False
        self.stopped = False
        self.thread = None
    
    def start(self) -> 'VideoCapture':
        """Start the thread to read frames from the webcam"""
        self.thread = threading.Thread(target=self._update, args=())
        self.thread.daemon = True
        self.thread.start()
        return self
    
    def _update(self) -> None:
        """Continuously update the frame from the webcam in a separate thread"""
        while not self.stopped:
            if not self.cap.isOpened():
                self.stop()
                return
                
            ret, frame = self.cap.read()
            if ret:
                self.frame_read = frame
                self.frame_available = True
            else:
                print("Warning: Failed to capture frame")
                time.sleep(0.1)
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read the most recent frame from the webcam"""
        if self.frame_available:
            return True, self.frame_read.copy()
        return False, None
    
    def stop(self) -> None:
        """Stop the video capture thread and release resources"""
        self.stopped = True
        if self.thread is not None:
            self.thread.join()
        if self.cap is not None:
            self.cap.release()


class ObjectDetector:
    """Class for handling object detection using YOLOv8"""
    
    def __init__(self, model_path: str = "yolov8n.pt", confidence: float = 0.35):
        """
        Initialize the object detector
        
        Args:
            model_path: Path to the YOLOv8 model weights
            confidence: Confidence threshold for detections
        """
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.classes = self.model.names
        self.colors = {}
        self.stopped = False
        self.queue = queue.Queue(maxsize=1)
        self.result_queue = queue.Queue(maxsize=1)
        self.thread = None
        
        # Pre-generate colors for each class - make them more distinct
        np.random.seed(42)  # For consistent colors
        for class_id in self.classes:
            h = np.random.randint(0, 360)
            s = np.random.randint(50, 100) / 100
            v = np.random.randint(50, 100) / 100
            
            # Convert HSV to BGR for OpenCV
            hsv = np.array([[[h, s*255, v*255]]], dtype=np.uint8)
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            color = tuple(map(int, bgr[0][0]))
            self.colors[class_id] = color
    
    def start(self):
        """Start the detection thread"""
        self.thread = threading.Thread(target=self._process_queue)
        self.thread.daemon = True
        self.thread.start()
        return self
        
    def _process_queue(self):
        """Process frames from the queue"""
        while not self.stopped:
            try:
                if not self.queue.empty():
                    frame = self.queue.get(timeout=1)
                    results = self.model(frame, conf=self.confidence, verbose=False)[0]
                    detections = []
                    
                    for box in results.boxes:
                        class_id = int(box.cls.item())
                        class_name = self.classes[class_id]
                        confidence = box.conf.item()
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        
                        detections.append(DetectedObject(
                            class_id=class_id,
                            class_name=class_name,
                            confidence=confidence,
                            box=(x1, y1, x2, y2),
                            color=self.colors[class_id]
                        ))
                    
                    # Put result in the result queue, but don't block if full
                    if not self.result_queue.full():
                        self.result_queue.put((frame, detections))
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in detection thread: {e}")
    
    def detect_async(self, frame: np.ndarray) -> None:
        """
        Add a frame to the detection queue
        
        Args:
            frame: Input frame as numpy array
        """
        if not self.queue.full():
            self.queue.put(frame.copy())
    
    def get_results(self) -> Tuple[Optional[np.ndarray], List[DetectedObject]]:
        """Get the latest detection results"""
        if not self.result_queue.empty():
            return self.result_queue.get()
        return None, []
            
    def stop(self):
        """Stop the detection thread"""
        self.stopped = True
        if self.thread is not None:
            self.thread.join()


class DisplayProcessor:
    """Class for processing and displaying detections on frames"""
    
    def __init__(self, show_fps: bool = True, show_count: bool = True):
        """
        Initialize the display processor
        
        Args:
            show_fps: Whether to display FPS on frame
            show_count: Whether to show object count per class
        """
        self.show_fps = show_fps
        self.show_count = show_count
        self.fps_stat = FPSCounter()
    
    def process_frame(self, frame: np.ndarray, detections: List[DetectedObject]) -> np.ndarray:
        """
        Process a frame by drawing detection boxes and information
        
        Args:
            frame: Input frame
            detections: List of detections to draw
            
        Returns:
            Processed frame with detection visualizations
        """
        output_frame = frame.copy()
        
        # Count objects per class for display
        class_counts = {}
        for det in detections:
            if det.class_name not in class_counts:
                class_counts[det.class_name] = 0
            class_counts[det.class_name] += 1
        
        # Draw detections
        for detection in detections:
            x1, y1, x2, y2 = detection.box
            
            # Draw bounding box
            cv2.rectangle(output_frame, (x1, y1), (x2, y2), detection.color, 2)
            
            # Prepare label with class name and confidence
            label = f"{detection.class_name}: {detection.confidence:.2f}"
            
            # Calculate text size and position
            (label_width, label_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            y1 = max(y1, label_height + baseline)
            
            # Draw label background
            cv2.rectangle(
                output_frame,
                (x1, y1 - label_height - baseline),
                (x1 + label_width, y1),
                detection.color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                output_frame,
                label,
                (x1, y1 - baseline),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
        
        # Update and draw FPS if enabled
        y_pos = 30
        if self.show_fps:
            self.fps_stat.update()
            fps_text = f"FPS: {self.fps_stat.get_fps():.1f}"
            cv2.putText(
                output_frame,
                fps_text,
                (10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )
            y_pos += 30
        
        # Draw object counts if enabled
        if self.show_count and class_counts:
            cv2.putText(
                output_frame,
                "Objects detected:",
                (10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
            y_pos += 25
            
            for i, (class_name, count) in enumerate(sorted(class_counts.items())):
                color = self.get_text_color(i)
                cv2.putText(
                    output_frame,
                    f"{class_name}: {count}",
                    (20, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )
                y_pos += 25
                
        return output_frame
    
    def get_text_color(self, index: int) -> Tuple[int, int, int]:
        """Get a color for text based on index"""
        colors = [
            (0, 255, 0),    # Green
            (255, 0, 0),    # Blue
            (0, 0, 255),    # Red
            (255, 255, 0),  # Cyan
            (0, 255, 255),  # Yellow
            (255, 0, 255),  # Magenta
            (255, 165, 0),  # Orange
            (128, 0, 128)   # Purple
        ]
        return colors[index % len(colors)]


class FPSCounter:
    """Class for calculating and smoothing FPS"""
    
    def __init__(self, avg_frames: int = 30):
        """
        Initialize the FPS counter
        
        Args:
            avg_frames: Number of frames to average for smoothing
        """
        self.prev_time = time.time()
        self.curr_time = self.prev_time
        self.frame_times = []
        self.avg_frames = avg_frames
    
    def update(self) -> None:
        """Update the FPS counter with the current frame time"""
        self.curr_time = time.time()
        delta = self.curr_time - self.prev_time
        self.prev_time = self.curr_time
        
        # Add to frame times and keep only the last avg_frames values
        self.frame_times.append(delta)
        if len(self.frame_times) > self.avg_frames:
            self.frame_times.pop(0)
    
    def get_fps(self) -> float:
        """Calculate the average FPS from stored frame times"""
        if not self.frame_times:
            return 0
        avg_time = sum(self.frame_times) / len(self.frame_times)
        return 1.0 / avg_time if avg_time > 0 else 0


class ObjectDetectionApp:
    """Main application class for object detection using webcam"""
    
    def __init__(
        self,
        camera_id: int = 0,
        frame_width: int = 1280,
        frame_height: int = 720,
        model_path: str = "yolov8n.pt",
        confidence: float = 0.35,
        show_fps: bool = True,
        flip_horizontal: bool = False,
        show_count: bool = True
    ):
        """
        Initialize the application
        
        Args:
            camera_id: Camera device ID
            frame_width: Width of frames
            frame_height: Height of frames
            model_path: Path to the YOLOv8 model weights
            confidence: Confidence threshold for detections
            show_fps: Whether to display FPS on frame
            flip_horizontal: Whether to flip frame horizontally
            show_count: Whether to show counts of detected objects
        """
        self.camera = VideoCapture(camera_id, frame_width, frame_height)
        self.detector = ObjectDetector(model_path, confidence)
        self.display = DisplayProcessor(show_fps, show_count)
        self.running = False
        self.flip_horizontal = flip_horizontal
        self.last_result_time = 0
        self.last_detection_frame = None
        self.last_detection_result = []
        
    def run(self) -> None:
        """Run the application main loop"""
        self.running = True
        self.camera.start()
        self.detector.start()
        
        # Give camera time to initialize
        time.sleep(1.0)
        
        print("Object Detection Application Started")
        print("Press 'q' to quit")
        print("Press 'f' to toggle frame flipping")
        print("Press 's' to save screenshot")
        
        try:
            while self.running:
                ret, frame = self.camera.read()
                
                if not ret or frame is None:
                    print("Warning: Failed to capture frame, retrying...")
                    time.sleep(0.1)
                    continue
                
                # Submit frame for async detection
                self.detector.detect_async(frame)
                
                # Check for results
                result_frame, detections = self.detector.get_results()
                if result_frame is not None:
                    self.last_detection_frame = result_frame
                    self.last_detection_result = detections
                    self.last_result_time = time.time()
                
                # Use most recent detection results
                display_frame = frame
                if self.last_detection_frame is not None:
                    # If detection is too old (more than 0.5 sec), use current frame
                    if time.time() - self.last_result_time > 0.5:
                        display_detections = []
                    else:
                        display_detections = self.last_detection_result
                    
                    # Process the frame with detections
                    display_frame = self.display.process_frame(frame, display_detections)
                
                # Flip horizontally if enabled
                if self.flip_horizontal:
                    display_frame = cv2.flip(display_frame, 1)
                
                # Display the frame
                cv2.imshow("Object Detection", display_frame)
                
                # Check for user input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.running = False
                elif key == ord('f'):
                    self.flip_horizontal = not self.flip_horizontal
                    print(f"Frame flipping {'enabled' if self.flip_horizontal else 'disabled'}")
                elif key == ord('s'):
                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    filename = f"detection-{timestamp}.jpg"
                    cv2.imwrite(filename, display_frame)
                    print(f"Screenshot saved as {filename}")
                    
        except KeyboardInterrupt:
            print("Application interrupted by user")
        finally:
            self.stop()
    
    def stop(self) -> None:
        """Stop the application and release resources"""
        self.running = False
        self.detector.stop()
        self.camera.stop()
        cv2.destroyAllWindows()
        print("Application stopped")


def main():
    """Main function to run the application"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Object Detection Webcam Application")
    parser.add_argument("--camera", type=int, default=0, help="Camera device ID")
    parser.add_argument("--width", type=int, default=1280, help="Frame width")
    parser.add_argument("--height", type=int, default=720, help="Frame height")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Path to YOLOv8 model")
    parser.add_argument("--confidence", type=float, default=0.35, help="Detection confidence threshold")
    parser.add_argument("--no-fps", action="store_true", help="Hide FPS counter")
    parser.add_argument("--flip", action="store_true", help="Flip frame horizontally (mirror)")
    parser.add_argument("--no-count", action="store_true", help="Hide object counts")
    
    args = parser.parse_args()
    
    # Check if model file exists
    if not os.path.exists(args.model):
        print(f"Model file not found: {args.model}")
        print("Downloading YOLOv8n model...")
        # The model will be downloaded automatically when initializing YOLO
    
    app = ObjectDetectionApp(
        camera_id=args.camera,
        frame_width=args.width,
        frame_height=args.height,
        model_path=args.model,
        confidence=args.confidence,
        show_fps=not args.no_fps,
        flip_horizontal=args.flip,
        show_count=not args.no_count
    )
    
    app.run()


if __name__ == "__main__":
    main()