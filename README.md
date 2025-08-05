
# Real-Time Object Detection & Hand Position Detection

This project uses **YOLOv5** for real-time object detection and **MediaPipe Pose** for detecting hand positions, integrated with **pyttsx3** for voice feedback. The program detects whether the right or left hand is raised or lowered, and provides auditory feedback.

## Features

- **Real-time Object Detection**: Uses YOLOv5 to detect objects in the webcam feed.
- **Hand Position Detection**: Uses MediaPipe to track arm and hand positions.
- **Voice Feedback**: Provides spoken feedback using pyttsx3 when hands are raised or lowered.
- **Full-Screen Webcam Feed**: Displays the camera feed in full-screen mode for better usability.

## Requirements

Make sure you have Python 3.7 or higher installed, then install the required dependencies:

```bash
pip install -r requirements.txt
```

### Required Python Libraries:
- `torch` (for YOLOv5)
- `opencv-python` (for webcam capture and image processing)
- `mediapipe` (for hand and body pose detection)
- `pyttsx3` (for text-to-speech feedback)
- `numpy`

You can create a `requirements.txt` file as follows:

```txt
torch==1.13.1
opencv-python==4.6.0.66
mediapipe==0.8.10
pyttsx3==2.90
numpy==1.21.2
```

## Setup Instructions

### 1. Clone the repository

Clone this repository to your local machine:

```bash
git clone https://github.com/yourusername/Real-Time-Object-Detection.git
cd Real-Time-Object-Detection
```

### 2. Install Dependencies

Create a virtual environment (optional but recommended):

```bash
python -m venv yolov5_env
```

Activate the environment:

- **Windows**:

```bash
yolov5_env\Scripts/activate
```

- **Linux/MacOS**:

```bash
source yolov5_env/bin/activate
```

Then install the required libraries:

```bash
pip install -r requirements.txt
```

### 3. Run the Program

To run the program, simply execute the following command:

```bash
python yolo_hand_detection.py
```

This will open the webcam feed in full-screen mode. The program will use YOLOv5 for object detection and MediaPipe to detect hand movements. The program will also provide auditory feedback when the right or left hand is raised or lowered.

### 4. Press `ESC` to Exit

To exit the program, simply press the `ESC` key.

## Contributing

Feel free to fork the repository, make improvements, and submit pull requests. Contributions are welcome!

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **YOLOv5**: For object detection.
- **MediaPipe**: For pose and hand tracking.
- **pyttsx3**: For text-to-speech functionality.

## Contact

For any questions or feedback, feel free to reach out through the issues section on GitHub or contact me at [shamsundarak2005@gmail.com](shamsundarak2005@gmail.com).

---

Happy coding!
