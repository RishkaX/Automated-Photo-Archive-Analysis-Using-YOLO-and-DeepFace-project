# Automated Photo Archive Analysis Using YOLO and DeepFace

A Machine Learning project that automatically analyzes photo archives using state-of-the-art computer vision technologies: YOLOv8 for object detection and DeepFace for facial analysis.

## Team Members

**Group SISCCO-2301**
- Mushrapilov Rishat
- Aikimbayev Maxat
- Baigulov Alizhan

## Overview

In today's digital age, over 1.4 trillion photos are taken annually worldwide. Manual organization and analysis of such massive collections is impractical and time-consuming. This project addresses this challenge by implementing an automated system that combines two powerful technologies to extract meaningful insights from photo archives.

## Features

### Object Detection (YOLOv8)
- Detects 80 different object classes from the COCO dataset
- Identifies common objects: people, furniture, vehicles, household items
- Real-time processing with high accuracy

### Facial Analysis (DeepFace)
- **Age Estimation**: Predicts age with ±5-10 year accuracy
- **Gender Classification**: Classifies gender with >95% accuracy
- **Emotion Recognition**: Detects 7 basic emotions (happy, sad, angry, fear, surprise, disgust, neutral)

### Statistical Analysis & Visualization
- Comprehensive statistics on detected objects
- Demographic distributions (age, gender)
- Emotional pattern analysis
- Professional visualizations using Matplotlib and Seaborn

## Technologies Used

- **YOLOv8n**: State-of-the-art object detection (2023 version)
- **DeepFace**: Facial recognition and analysis framework
- **Google Colab**: Development and execution environment
- **Python Libraries**:
  - Ultralytics (YOLO)
  - DeepFace
  - Matplotlib & Seaborn (visualization)
  - Pandas (data processing)

## System Architecture

```
Photo Archive Input (Google Drive)
        ↓
Image Preprocessing
        ↓
   ┌────────┴────────┐
   ↓                 ↓
YOLO Object      DeepFace
Detection        Face Analysis
   ↓                 ↓
   └────────┬────────┘
        ↓
Data Aggregation & Storage
        ↓
Visualization & Reporting
```

## Installation

### Prerequisites
```bash
pip install ultralytics
pip install deepface
pip install matplotlib seaborn pandas
```

### Google Colab Setup
```python
from google.colab import drive
drive.mount('/content/drive')
```

## Usage

### 1. Object Detection
```python
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Process images
results = model(image_path)

# Extract detected objects
for result in results:
    for box in result.boxes:
        cls = int(box.cls[0])
        object_name = model.names[cls]
```

### 2. Facial Analysis
```python
from deepface import DeepFace

# Analyze faces
results = DeepFace.analyze(
    image_path,
    actions=["age", "gender", "emotion"],
    enforce_detection=False
)

# Extract results
for face in results:
    age = face['age']
    gender = face['dominant_gender']
    emotion = face['dominant_emotion']
```

## Dataset

- **Total Images**: 66 images
- **Detected Faces**: 83 faces
- **Unique Objects**: 10 object types
- **Storage**: Google Drive (`/content/drive/MyDrive/photos_colab/`)
- **Formats**: JPEG, PNG

## Results

### Object Detection
| Object | Count | Percentage |
|--------|-------|------------|
| Cup | 4 | 40% |
| Chair | 2 | 20% |
| Bowl | 2 | 20% |
| Wine Glass | 1 | 10% |
| Dining Table | 1 | 10% |

### Demographics
- **Gender Distribution**: 77.1% Male, 22.9% Female
- **Mean Age**: 32.1 years
- **Age Range**: 22-50 years
- **Dominant Age Group**: 21-35 years (82.2%)

### Emotions
| Emotion | Percentage |
|---------|------------|
| Happy | 32.3% |
| Neutral | 28.0% |
| Fear | 19.4% |
| Sad | 11.8% |
| Angry | 6.5% |
| Surprise | 2.2% |

## Visualizations

The project generates five key visualizations:
1. Gender Distribution Bar Chart
2. Age Distribution Histogram with KDE
3. Emotion Frequency Bar Chart
4. Object Detection Frequency Chart
5. Emotion by Gender Cross-Analysis

## Limitations

- Age estimation: ±5-10 year margin of error
- Emotion recognition: ~60-70% accuracy
- Binary gender classification only
- Dataset size: 66 images (proof-of-concept)
- Sequential processing (not optimized for large datasets)
- No ground truth validation

## Future Improvements

### Immediate Enhancements
- GPU acceleration for 10-100x faster processing
- Ground truth annotation for accuracy measurement
- Batch processing implementation

### Advanced Features
- Temporal analysis (tracking changes over time)
- Scene classification
- Relationship inference through co-occurrence analysis
- Activity recognition with pose estimation
- Upgrade to YOLOv8s/v8m models
- Alternative facial analysis models (MTCNN, RetinaFace)

## Applications

- **Digital Asset Management**: Automatic tagging of stock photos
- **Social Media**: Content moderation and personalization
- **Event Photography**: Organizing thousands of photos by people and moments
- **Market Research**: Analyzing social media images for product usage
- **Cultural Heritage**: Cataloging historical photo collections
- **Personal Photo Management**: Smart albums and automatic highlights

## Ethical Considerations

- Facial data is sensitive - proper consent is required
- Potential for misuse in surveillance and discrimination
- Privacy concerns must be addressed before deployment
- Responsible AI practices are essential

## Acknowledgments

This project was developed as part of the Machine Learning course at the Department of Cybersecurity, Ministry of Education and Science of Republic of Kazakhstan.

## License

This project is for educational purposes. Please ensure compliance with relevant data protection and privacy regulations when using facial recognition technology.

---

**Project Year**: 2025  
**Location**: Almaty, Kazakhstan
