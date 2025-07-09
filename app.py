# import os
# import uuid
# import json
# import re
# import logging
# import zipfile
# from flask import Flask, render_template, request, send_file, jsonify, Response
# import cv2
# import numpy as np
# from werkzeug.utils import secure_filename
# from io import BytesIO
#
# app = Flask(__name__)
# app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB limit
#
# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger('YOLOVisualizer')
#
# # Allowed extensions
# ALLOWED_IMAGE_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
# ALLOWED_LABEL_EXT = {'.txt'}
#
#
# def get_class_colors(num_classes):
#     np.random.seed(42)
#     return [tuple(map(int, np.random.randint(0, 255, 3))) for _ in range(num_classes)]
#
#
# def visualize_detections(image, label_content, classes, class_colors, max_size=1000, conf_threshold=0.25):
#     orig_height, orig_width = image.shape[:2]
#     scale = min(max_size / orig_width, max_size / orig_height)
#
#     if scale < 1:
#         new_width = int(orig_width * scale)
#         new_height = int(orig_height * scale)
#         image = cv2.resize(image, (new_width, new_height))
#     else:
#         scale = 1
#         new_width, new_height = orig_width, orig_height
#
#     if label_content:
#         lines = label_content.splitlines()
#         for line in lines:
#             parts = line.strip().split()
#             if len(parts) < 5:
#                 continue
#
#             try:
#                 class_id = int(parts[0])
#                 conf = float(parts[5]) if len(parts) >= 6 else 1.0
#                 if conf < conf_threshold:
#                     continue
#
#                 # Convert and scale coordinates
#                 x_center = float(parts[1]) * orig_width * scale
#                 y_center = float(parts[2]) * orig_height * scale
#                 w = float(parts[3]) * orig_width * scale
#                 h = float(parts[4]) * orig_height * scale
#
#                 x1 = int(x_center - w / 2)
#                 y1 = int(y_center - h / 2)
#                 x2 = int(x_center + w / 2)
#                 y2 = int(y_center + h / 2)
#
#                 # Clip coordinates
#                 x1 = max(0, min(x1, new_width - 1))
#                 y1 = max(0, min(y1, new_height - 1))
#                 x2 = max(0, min(x2, new_width - 1))
#                 y2 = max(0, min(y2, new_height - 1))
#
#                 color = class_colors[class_id % len(class_colors)]
#                 label = f"{classes[class_id]}: {conf:.2f}" if conf < 1.0 else classes[class_id]
#
#                 # Draw bounding box and label
#                 cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
#                 (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
#                 cv2.rectangle(image, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)
#                 cv2.putText(image, label, (x1, y1 - 5),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
#             except (ValueError, IndexError) as e:
#                 logger.error(f"Error processing line: {line.strip()} - {str(e)}")
#
#     return image
#
#
# @app.route('/', methods=['GET'])
# def index():
#     return render_template('index.html')
#
#
# @app.route('/process', methods=['POST'])
# def process():
#     # Generate session ID
#     session_id = uuid.uuid4().hex
#
#     # Get form parameters
#     max_size = int(request.form.get('max_size', 1000))
#     conf_threshold = float(request.form.get('conf_threshold', 0.25))
#     class_names = request.form.get('class_names', 'object').splitlines()
#     processing_mode = request.form.get('processing_mode', 'single')
#
#     logger.info(f"Starting processing session: {session_id}")
#     logger.info(f"Mode: {processing_mode}, Max size: {max_size}, Conf threshold: {conf_threshold}")
#
#     # Handle class names file upload if provided
#     if 'classes_file' in request.files:
#         classes_file = request.files['classes_file']
#         if classes_file.filename != '':
#             class_names = classes_file.read().decode('utf-8').splitlines()
#             logger.info(f"Loaded class names from file: {len(class_names)} classes")
#
#     # Process single image mode
#     if processing_mode == 'single':
#         logger.info("Processing single image mode")
#         if 'image' in request.files and request.files['image'].filename != '':
#             image_file = request.files['image']
#             label_file = request.files.get('label')
#
#             logger.info(f"Processing image: {image_file.filename}")
#             if label_file and label_file.filename != '':
#                 logger.info(f"Using label file: {label_file.filename}")
#
#             # Read image
#             img_data = image_file.read()
#             img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
#             if img is None:
#                 logger.error(f"Error loading image: {image_file.filename}")
#                 return jsonify({"error": f"Error loading image: {image_file.filename}"}), 400
#
#             # Read label content if available
#             label_content = None
#             if label_file and label_file.filename != '':
#                 try:
#                     label_content = label_file.read().decode('utf-8')
#                 except Exception as e:
#                     logger.error(f"Error reading label file: {str(e)}")
#
#             # Process image
#             class_colors = get_class_colors(len(class_names))
#             result_img = visualize_detections(
#                 img,
#                 label_content,
#                 class_names,
#                 class_colors,
#                 max_size,
#                 conf_threshold
#             )
#
#             # Convert to JPEG in memory
#             success, buffer = cv2.imencode('.jpg', result_img)
#             if not success:
#                 logger.error("Error converting image to JPEG format")
#                 return jsonify({"error": "Error processing image"}), 500
#
#             # Create in-memory file
#             img_bytes = BytesIO(buffer)
#             img_bytes.seek(0)
#
#             logger.info(f"Successfully processed image: {image_file.filename}")
#
#             # Return result
#             return send_file(
#                 img_bytes,
#                 as_attachment=True,
#                 download_name=f"visualized_{secure_filename(image_file.filename)}",
#                 mimetype='image/jpeg'
#             )
#
#         logger.error("No valid image file uploaded for single processing")
#         return jsonify({"error": "No valid image file uploaded"}), 400
#
#     # Batch processing mode
#     elif processing_mode == 'batch':
#         logger.info("Processing batch mode")
#         image_files = request.files.getlist('batch_images')
#         label_files = request.files.getlist('batch_labels')
#
#         # Filter out empty files
#         image_files = [f for f in image_files if f.filename != '']
#         label_files = [f for f in label_files if f.filename != '']
#
#         logger.info(f"Received {len(image_files)} images and {len(label_files)} labels")
#
#         if not image_files or not label_files:
#             logger.error("Batch processing requires both images and labels")
#             return jsonify({"error": "Please upload both images and labels"}), 400
#
#         # Helper function to extract base name
#         def get_base_name(filename):
#             # Remove .rf.* pattern if exists
#             clean_name = re.sub(r'\.rf\.[a-f0-9]+$', '', filename, flags=re.IGNORECASE)
#             # Remove file extension
#             return os.path.splitext(clean_name)[0]
#
#         # Store file contents in memory
#         image_contents = {}
#         for file in image_files:
#             base_name = get_base_name(file.filename)
#             image_contents[base_name] = {
#                 'filename': file.filename,
#                 'content': file.read()
#             }
#
#         label_contents = {}
#         for file in label_files:
#             base_name = get_base_name(file.filename)
#             try:
#                 label_contents[base_name] = {
#                     'filename': file.filename,
#                     'content': file.read().decode('utf-8')
#                 }
#             except Exception as e:
#                 logger.error(f"Error reading label file {file.filename}: {str(e)}")
#                 label_contents[base_name] = {
#                     'filename': file.filename,
#                     'content': None
#                 }
#
#         # Find matching pairs
#         matches = []
#         for base_name in set(image_contents.keys()) & set(label_contents.keys()):
#             if label_contents[base_name]['content'] is not None:
#                 matches.append((
#                     base_name,
#                     image_contents[base_name],
#                     label_contents[base_name]
#                 ))
#
#         logger.info(f"Found {len(matches)} matching image-label pairs")
#
#         if not matches:
#             logger.error("No matching image-label pairs found")
#             return jsonify({"error": "No matching image-label pairs found"}), 400
#
#         # Prepare zip file in memory
#         zip_buffer = BytesIO()
#         with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
#             class_colors = get_class_colors(len(class_names))
#
#             for idx, (base_name, image_data, label_data) in enumerate(matches):
#                 image_filename = image_data['filename']
#                 label_filename = label_data['filename']
#
#                 logger.info(f"Processing pair {idx + 1}/{len(matches)}: {image_filename}")
#
#                 try:
#                     # Process image
#                     img = cv2.imdecode(np.frombuffer(image_data['content'], np.uint8), cv2.IMREAD_COLOR)
#                     if img is None:
#                         logger.error(f"Error loading image: {image_filename}")
#                         continue
#
#                     # Process image
#                     result_img = visualize_detections(
#                         img,
#                         label_data['content'],
#                         class_names,
#                         class_colors,
#                         max_size,
#                         conf_threshold
#                     )
#
#                     # Convert to JPEG in memory
#                     success, buffer = cv2.imencode('.jpg', result_img)
#                     if not success:
#                         logger.error(f"Error converting image: {image_filename}")
#                         continue
#
#                     # Add to zip file
#                     zip_file.writestr(f"visualized_{image_filename}", buffer.tobytes())
#                     logger.info(f"Processed: {image_filename}")
#                 except Exception as e:
#                     logger.error(f"Error processing {image_filename}: {str(e)}")
#
#         # Prepare zip file for download
#         zip_buffer.seek(0)
#         logger.info(f"Completed batch processing session: {session_id}")
#
#         return send_file(
#             zip_buffer,
#             as_attachment=True,
#             download_name='visualized_images.zip',
#             mimetype='application/zip'
#         )
#
#     logger.error("Invalid processing mode specified")
#     return jsonify({"error": "Invalid processing mode"}), 400
#
#
# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
#


import os
import uuid
import json
import re
import logging
import zipfile
from flask import Flask, render_template, request, send_file, jsonify, Response
import cv2
import numpy as np
from werkzeug.utils import secure_filename
from io import BytesIO

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB limit

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('YOLOVisualizer')

# Allowed extensions
ALLOWED_IMAGE_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
ALLOWED_LABEL_EXT = {'.txt'}


def get_class_colors(num_classes):
    np.random.seed(42)
    return [tuple(map(int, np.random.randint(0, 255, 3))) for _ in range(num_classes)]


def visualize_detections(image, label_content, classes, class_colors, max_size=1000, conf_threshold=0.25):
    orig_height, orig_width = image.shape[:2]
    scale = min(max_size / orig_width, max_size / orig_height)

    if scale < 1:
        new_width = int(orig_width * scale)
        new_height = int(orig_height * scale)
        image = cv2.resize(image, (new_width, new_height))
    else:
        scale = 1
        new_width, new_height = orig_width, orig_height

    if label_content:
        lines = label_content.splitlines()
        for line in lines:
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
                logger.error(f"Error processing line: {line.strip()} - {str(e)}")

    return image


def get_base_name(filename):
    """Improved base name extraction for mobile and desktop patterns"""
    # Handle Roboflow mobile pattern: IMG_20230101_123456789.jpg -> IMG_20230101_123456789
    if re.match(r'IMG_\d{8}_\d+', filename):
        return re.sub(r'\.\w+$', '', filename)

    # Handle standard pattern: image.jpg.rf.123456 -> image
    clean_name = re.sub(r'\.rf\.[a-f0-9]+$', '', filename, flags=re.IGNORECASE)
    return os.path.splitext(clean_name)[0]


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/process', methods=['POST'])
def process():
    # Generate session ID
    session_id = uuid.uuid4().hex

    # Get form parameters
    max_size = int(request.form.get('max_size', 1000))
    conf_threshold = float(request.form.get('conf_threshold', 0.25))
    class_names = request.form.get('class_names', 'object').splitlines()
    processing_mode = request.form.get('processing_mode', 'single')

    logger.info(f"Starting processing session: {session_id}")
    logger.info(f"Mode: {processing_mode}, Max size: {max_size}, Conf threshold: {conf_threshold}")

    # Handle class names file upload if provided
    if 'classes_file' in request.files:
        classes_file = request.files['classes_file']
        if classes_file.filename != '':
            class_names = classes_file.read().decode('utf-8').splitlines()
            logger.info(f"Loaded class names from file: {len(class_names)} classes")

    # Process single image mode
    if processing_mode == 'single':
        logger.info("Processing single image mode")
        if 'image' in request.files and request.files['image'].filename != '':
            image_file = request.files['image']
            label_file = request.files.get('label')

            logger.info(f"Processing image: {image_file.filename}")
            if label_file and label_file.filename != '':
                logger.info(f"Using label file: {label_file.filename}")

            # Read image
            img_data = image_file.read()
            img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                logger.error(f"Error loading image: {image_file.filename}")
                return jsonify({"error": f"Error loading image: {image_file.filename}"}), 400

            # Read label content if available
            label_content = None
            if label_file and label_file.filename != '':
                try:
                    label_content = label_file.read().decode('utf-8')
                except Exception as e:
                    logger.error(f"Error reading label file: {str(e)}")

            # Process image
            class_colors = get_class_colors(len(class_names))
            result_img = visualize_detections(
                img,
                label_content,
                class_names,
                class_colors,
                max_size,
                conf_threshold
            )

            # Convert to JPEG in memory
            success, buffer = cv2.imencode('.jpg', result_img)
            if not success:
                logger.error("Error converting image to JPEG format")
                return jsonify({"error": "Error processing image"}), 500

            # Create in-memory file
            img_bytes = BytesIO(buffer)
            img_bytes.seek(0)

            logger.info(f"Successfully processed image: {image_file.filename}")

            # Return result
            return send_file(
                img_bytes,
                as_attachment=True,
                download_name=f"visualized_{secure_filename(image_file.filename)}",
                mimetype='image/jpeg'
            )

        logger.error("No valid image file uploaded for single processing")
        return jsonify({"error": "No valid image file uploaded"}), 400

    # Batch processing mode
    elif processing_mode == 'batch':
        logger.info("Processing batch mode")
        image_files = request.files.getlist('batch_images')
        label_files = request.files.getlist('batch_labels')

        # Filter out empty files
        image_files = [f for f in image_files if f.filename != '']
        label_files = [f for f in label_files if f.filename != '']

        logger.info(f"Received {len(image_files)} images and {len(label_files)} labels")

        # Mobile-friendly handling of partial uploads
        if not image_files and not label_files:
            logger.error("No files uploaded for batch processing")
            return jsonify({"error": "Please upload at least one image and one label file"}), 400
        elif not image_files:
            logger.error("No image files uploaded")
            return jsonify({"error": "Please upload image files"}), 400
        elif not label_files:
            logger.error("No label files uploaded")
            return jsonify({"error": "Please upload label files"}), 400

        # Store file contents in memory
        image_contents = {}
        for file in image_files:
            base_name = get_base_name(file.filename)
            image_contents[base_name] = {
                'filename': file.filename,
                'content': file.read()
            }

        label_contents = {}
        for file in label_files:
            base_name = get_base_name(file.filename)
            try:
                label_contents[base_name] = {
                    'filename': file.filename,
                    'content': file.read().decode('utf-8')
                }
            except Exception as e:
                logger.error(f"Error reading label file {file.filename}: {str(e)}")
                label_contents[base_name] = {
                    'filename': file.filename,
                    'content': None
                }

        # Find matching pairs
        matches = []
        for base_name in set(image_contents.keys()) & set(label_contents.keys()):
            if label_contents[base_name]['content'] is not None:
                matches.append((
                    base_name,
                    image_contents[base_name],
                    label_contents[base_name]
                ))

        logger.info(f"Found {len(matches)} matching image-label pairs")

        if not matches:
            logger.error("No matching image-label pairs found")
            return jsonify({"error": "No matching image-label pairs found"}), 400

        # Prepare zip file in memory
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            class_colors = get_class_colors(len(class_names))

            for idx, (base_name, image_data, label_data) in enumerate(matches):
                image_filename = image_data['filename']
                label_filename = label_data['filename']

                logger.info(f"Processing pair {idx + 1}/{len(matches)}: {image_filename}")

                try:
                    # Process image
                    img = cv2.imdecode(np.frombuffer(image_data['content'], np.uint8), cv2.IMREAD_COLOR)
                    if img is None:
                        logger.error(f"Error loading image: {image_filename}")
                        continue

                    # Process image
                    result_img = visualize_detections(
                        img,
                        label_data['content'],
                        class_names,
                        class_colors,
                        max_size,
                        conf_threshold
                    )

                    # Convert to JPEG in memory
                    success, buffer = cv2.imencode('.jpg', result_img)
                    if not success:
                        logger.error(f"Error converting image: {image_filename}")
                        continue

                    # Add to zip file
                    zip_file.writestr(f"visualized_{image_filename}", buffer.tobytes())
                    logger.info(f"Processed: {image_filename}")
                except Exception as e:
                    logger.error(f"Error processing {image_filename}: {str(e)}")

        # Prepare zip file for download
        zip_buffer.seek(0)
        logger.info(f"Completed batch processing session: {session_id}")

        return send_file(
            zip_buffer,
            as_attachment=True,
            download_name='visualized_images.zip',
            mimetype='application/zip'
        )

    logger.error("Invalid processing mode specified")
    return jsonify({"error": "Invalid processing mode"}), 400


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))