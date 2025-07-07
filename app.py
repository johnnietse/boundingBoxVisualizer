# app.py
import os
import uuid
import zipfile
from flask import Flask, render_template, request, send_file, after_this_request
import cv2
import numpy as np
from pathlib import Path

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB limit

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)


def get_class_colors(num_classes):
    np.random.seed(42)
    return [tuple(map(int, np.random.randint(0, 255, 3))) for _ in range(num_classes)]


def visualize_detections(image, labels_path, classes, class_colors, max_size=1000, conf_threshold=0.25):
    orig_height, orig_width = image.shape[:2]
    scale = min(max_size / orig_width, max_size / orig_height)

    if scale < 1:
        new_width = int(orig_width * scale)
        new_height = int(orig_height * scale)
        image = cv2.resize(image, (new_width, new_height))
    else:
        scale = 1
        new_width, new_height = orig_width, orig_height

    if os.path.exists(labels_path):
        with open(labels_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue

                try:
                    class_id = int(parts[0])
                    conf = float(parts[5]) if len(parts) >= 6 else 1.0
                    if conf < conf_threshold:
                        continue

                    # Convert and scale coordinates
                    x_center = float(parts[1]) * orig_width * scale
                    y_center = float(parts[2]) * orig_height * scale
                    w = float(parts[3]) * orig_width * scale
                    h = float(parts[4]) * orig_height * scale

                    x1 = int(x_center - w / 2)
                    y1 = int(y_center - h / 2)
                    x2 = int(x_center + w / 2)
                    y2 = int(y_center + h / 2)

                    # Clip coordinates
                    x1 = max(0, min(x1, new_width - 1))
                    y1 = max(0, min(y1, new_height - 1))
                    x2 = max(0, min(x2, new_width - 1))
                    y2 = max(0, min(y2, new_height - 1))

                    color = class_colors[class_id % len(class_colors)]
                    label = f"{classes[class_id]}: {conf:.2f}" if conf < 1.0 else classes[class_id]

                    # Draw bounding box and label
                    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                    (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                    cv2.rectangle(image, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)
                    cv2.putText(image, label, (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                except (ValueError, IndexError) as e:
                    print(f"Error processing line: {line.strip()} - {str(e)}")

    return image


def clean_directory(path):
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/process', methods=['POST'])
def process():
    # Generate unique ID for this session
    session_id = uuid.uuid4().hex
    upload_dir = os.path.join(app.config['UPLOAD_FOLDER'], session_id)
    output_dir = os.path.join(app.config['OUTPUT_FOLDER'], session_id)
    os.makedirs(upload_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Get form parameters
    max_size = int(request.form.get('max_size', 1000))
    conf_threshold = float(request.form.get('conf_threshold', 0.25))
    class_names = request.form.get('class_names', 'object').splitlines()

    # Handle class names file upload if provided
    if 'classes_file' in request.files:
        classes_file = request.files['classes_file']
        if classes_file.filename != '':
            class_names = classes_file.read().decode('utf-8').splitlines()

    # Process single image mode
    if 'image' in request.files and request.files['image'].filename != '':
        image_file = request.files['image']
        label_file = request.files.get('label')

        # Save files
        img_path = os.path.join(upload_dir, image_file.filename)
        image_file.save(img_path)

        # Process label if provided
        label_path = None
        if label_file and label_file.filename != '':
            label_path = os.path.join(upload_dir, label_file.filename)
            label_file.save(label_path)

        # Process image
        img = cv2.imread(img_path)
        if img is None:
            return "Error loading image", 400

        class_colors = get_class_colors(len(class_names))
        result_img = visualize_detections(
            img,
            label_path if label_path else '',
            class_names,
            class_colors,
            max_size,
            conf_threshold
        )

        # Save result
        output_path = os.path.join(output_dir, image_file.filename)
        cv2.imwrite(output_path, result_img)

        # Return result
        return send_file(output_path, as_attachment=True)

    # Process zip file mode
    elif 'zip_file' in request.files and request.files['zip_file'].filename != '':
        zip_file = request.files['zip_file']
        zip_path = os.path.join(upload_dir, zip_file.filename)
        zip_file.save(zip_path)

        # Extract zip
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(upload_dir)

        # Process images
        class_colors = get_class_colors(len(class_names))
        supported_exts = ['.jpg', '.jpeg', '.png', '.bmp']

        for file in os.listdir(upload_dir):
            if os.path.splitext(file)[1].lower() in supported_exts:
                img_path = os.path.join(upload_dir, file)
                label_path = os.path.splitext(img_path)[0] + '.txt'

                img = cv2.imread(img_path)
                if img is None:
                    continue

                result_img = visualize_detections(
                    img,
                    label_path if os.path.exists(label_path) else '',
                    class_names,
                    class_colors,
                    max_size,
                    conf_threshold
                )

                output_path = os.path.join(output_dir, file)
                cv2.imwrite(output_path, result_img)

        # Create output zip
        output_zip = os.path.join(output_dir, 'visualized_dataset.zip')
        with zipfile.ZipFile(output_zip, 'w') as zipf:
            for root, _, files in os.walk(output_dir):
                for file in files:
                    if file != 'visualized_dataset.zip':
                        file_path = os.path.join(root, file)
                        zipf.write(file_path, os.path.basename(file_path))

        # Cleanup callback
        @after_this_request
        def cleanup(response):
            try:
                clean_directory(upload_dir)
                clean_directory(output_dir)
                os.rmdir(upload_dir)
                os.rmdir(output_dir)
            except Exception as e:
                app.logger.error(f"Cleanup error: {e}")
            return response

        return send_file(output_zip, as_attachment=True, download_name='visualized_dataset.zip')

    return "No valid files uploaded", 400


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))