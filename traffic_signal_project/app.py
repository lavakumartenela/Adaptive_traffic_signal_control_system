from flask import Flask, request, render_template, redirect, url_for, send_file
import os
import cv2
from ultralytics import YOLO

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['PROCESSED_FOLDER'] = 'processed'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)

# Helper functions
def resize_to_target_size(image, target_size):
    return cv2.resize(image, target_size)

def process_last_frame(video_path, model, vehicle_classes):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video {video_path}.")
        return 0, None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
    ret, frame = cap.read()
    if not ret:
        print(f"Error: Unable to read the last frame of {video_path}.")
        cap.release()
        return 0, None

    results = model.predict(frame)
    vehicle_count = 0
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            if cls_name in vehicle_classes:
                vehicle_count += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, cls_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    target_size = (1280, 720)
    frame = resize_to_target_size(frame, target_size)
    cap.release()
    return vehicle_count, frame

def calculate_green_signal_time(vehicle_count):
    if vehicle_count > 25:
        return 90
    elif vehicle_count >= 20:
        return 60
    elif vehicle_count >= 10:
        return 45
    else:
        return 30

def add_text_with_background(image, text, position, font_scale, text_color, bg_color, thickness=2, padding=5):
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x, text_y = position
    box_coords = ((text_x, text_y), (text_x + text_size[0] + 2 * padding, text_y - text_size[1] - 2 * padding))
    cv2.rectangle(image, box_coords[0], box_coords[1], bg_color, cv2.FILLED)
    cv2.putText(image, text, (text_x + padding, text_y - padding), font, font_scale, text_color, thickness)

@app.route('/', methods=['GET', 'POST'])
def upload_videos():
    if request.method == 'POST':
        video_paths = []

        for i in range(4):
            file = request.files.get(f'video{i + 1}')
            if file and file.filename:
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(filepath)
                video_paths.append(filepath)

        if video_paths:
            return redirect(url_for('process_videos', video_paths=','.join(video_paths)))
        else:
            return "No videos were uploaded. Please try again.", 400
    return render_template('index.html')

@app.route('/process')
def process_videos():
    video_paths = request.args.get('video_paths').split(',')
    model = YOLO('yolov8x.pt')
    vehicle_classes = ['car', 'motorbike', 'bus', 'truck', 'bicycle', 'van', 'motorcycle']

    lane_data = {}
    for lane_id, video_path in enumerate(video_paths):
        vehicle_count, last_frame = process_last_frame(video_path, model, vehicle_classes)
        green_signal_time = calculate_green_signal_time(vehicle_count)
        lane_data[f"Lane {lane_id + 1}"] = (vehicle_count, green_signal_time, last_frame)

    sorted_lanes = sorted(lane_data.items(), key=lambda x: x[1][0], reverse=True)

    processed_images = []
    for active_lane_index, (active_lane_name, (active_count, active_green_time, active_frame)) in enumerate(sorted_lanes):
        if active_frame is None:
            continue

        composite_frame = active_frame.copy()
        add_text_with_background(composite_frame, f"Vehicle Count: {active_count}", (10, 40), 1, (0, 0, 0), (255, 255, 255))
        add_text_with_background(composite_frame, f"{active_lane_name}: GREEN SIGNAL ({active_green_time}s)", (10, 80), 1, (0, 255, 0), (255, 255, 255))
        
        y_offset = 130
        for other_lane_index, (other_lane_name, _) in enumerate(sorted_lanes):
            if other_lane_index == active_lane_index:
                continue
            add_text_with_background(composite_frame, f"{other_lane_name}: RED SIGNAL", (10, y_offset), 1, (0, 0, 255), (255, 255, 255))
            y_offset += 50

        output_path = os.path.join(app.config['PROCESSED_FOLDER'], f'{active_lane_name}.jpg')
        cv2.imwrite(output_path, composite_frame)
        processed_images.append(output_path)

    return render_template('results.html', images=processed_images)

@app.route('/display/<path:filename>')
def display_image(filename):
    return send_file(filename, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True)
