# YOLO Dataset Visualizer
## Overview
YOLO Dataset Visualizer is a web application that allows you to visualize bounding boxes on images using YOLO format label files. The tool supports both single image processing and batch processing of multiple images with their corresponding label files. Users can upload class definitions, adjust visualization settings, and download the processed images or a zip archive of results.


![Screenshot (5119)](https://github.com/user-attachments/assets/6198fabd-ccfc-418a-819d-38838c7edb94)
![Screenshot (5120)](https://github.com/user-attachments/assets/c3599de5-3807-4609-a281-f028ea294af0)
![Screenshot (5121)](https://github.com/user-attachments/assets/79b1dee4-8735-43b7-bfd6-95f41e2fa310)
![Screenshot (5122)](https://github.com/user-attachments/assets/0f129de2-a781-4cf3-9264-26f077fc44f2)
![Screenshot (5123)](https://github.com/user-attachments/assets/9d78f100-7a84-4c21-b729-c889bb87f4c0)
![Screenshot (5124)](https://github.com/user-attachments/assets/28ed2768-23ad-48db-a1cc-1b934d1258c9)
![Screenshot (5125)](https://github.com/user-attachments/assets/e3c31693-74e5-4d57-84de-78061193ebf8)
![Screenshot (5126)](https://github.com/user-attachments/assets/a3133d4d-0bad-4360-b713-e81bbfd01008)



## Key Features
- Single Image Visualization: Upload an image and its corresponding label file to see bounding boxes
- Batch Processing: Process multiple images and labels simultaneously
- Class Management: Define classes via text input or by uploading a classes.txt file
- Modern UI: Responsive design with light/dark mode support
- Progress Tracking: Real-time progress updates during processing
- Direct Downloads: Download visualized images or zip archives


## Customizable Settings:
- Adjust maximum image size
- Set confidence threshold for bounding boxes




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
```

2. Create and activate a virtual environment:
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
- Set build command: `pip install --upgrade pip setuptools wheel && pip install -r requirements.txt`
- Set start command: `gunicorn app:app`

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
There are also some UI related issues that I have to fix soon, but this does not affect the overall functionality of the webapp.
