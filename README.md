# YOLO Dataset Visualizer
## Overview
YOLO Dataset Visualizer is a web application that allows you to visualize bounding boxes on images using YOLO format label files. The tool supports both single image processing and batch processing of multiple images with their corresponding label files. Users can upload class definitions, adjust visualization settings, and download the processed images or a zip archive of results.


## Key Features
- Single Image Visualization: Upload an image and its corresponding label file to see bounding boxes
- Batch Processing: Process multiple images and labels simultaneously
- Class Management: Define classes via text input or by uploading a classes.txt file

## Customizable Settings:
- Adjust maximum image size
- Set confidence threshold for bounding boxes

## Modern UI: Responsive design with light/dark mode support

## Progress Tracking: Real-time progress updates during processing

## Direct Downloads: Download visualized images or zip archives


## Installation & Setup
### Prerequisites:
```text
Python 3.11.x (as specified in runtime.txt)
```

### Steps
1. Clone the repository:

```bash
git clone https://github.com/johnnietse/boundingBoxVisualizer.git
cd boundingBoxVisualizer

2. Create and activate a virtual environment:
```
```bash
python -m venv venv
source venv/bin/activate  # Linux/MacOS
venv\Scripts\activate     # Windows
```

3. Install dependencies:

``` bash
pip install -r requirements.txt
```

4. Run the application:

```bash
python app.py
```

5. Access the application at:
```text
http://localhost:5000
```

## Usage Guide
- Single Image Processing:
  - Select "Single Image" tab
  - Upload an image file (JPG, PNG, BMP, WEBP)
  - Upload the corresponding label file (.txt)
  - Configure visualization settings
  - Click "Visualize Image" to process and download

- Batch Processing:
  - Select "Batch Processing" tab
  - Drag and drop image files
  - Drag and drop label files
  - Configure visualization settings
  - Click "Visualize Batch" to process and download as ZIP

- Class Configuration:
  - Enter class names (one per line) in the text area OR upload a classes.txt file (could be named whatever you want) containing class names

## Deployment
The application is ready for deployment on platforms like Render or Heroku. The project includes:
- runtime.txt specifying Python 3.11.9
- requirements.txt with all dependencies
- Properly configured Flask application in app.py

### Render Deployment
- Create a new Web Service on Render
- Connect your GitHub repository
- Set build command: pip install --upgrade pip setuptools wheel && pip install -r requirements.txt
- Set start command: gunicorn app:app

Deploy!

## Dependencies
The project uses the following Python libraries:
- Flask (web framework)
- OpenCV (image processing)
- NumPy (numerical operations)
- Pillow (image handling)
- Gunicorn (production server)

See requirements.txt for specific versions.

## License
This project is open source and available under the MIT License.

---

Note: This application is designed for visualizing YOLO format datasets. It does not perform object detection but rather visualizes existing label data on images.
