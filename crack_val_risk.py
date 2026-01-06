import os
import cv2
import numpy as np
from ultralytics import YOLO

MODEL_PATH = "runs/segment/train/weights/best.pt" 
IMAGE_DIR = "path to data file"
OUTPUT_DIR = "crack_detection_output/crack_risk"
PIXEL_TO_MM_FACTOR = 0.005

def get_crack_properties(mask_data):
    points = np.argwhere(mask_data)
    if len(points) < 2:
        return None, None, None, None

    mean = np.mean(points, axis=0)
    cov = np.cov(points, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    main_direction = eigenvectors[:, -1]
    
    angle_rad = np.arctan2(main_direction[0], main_direction[1])
    angle_deg = np.degrees(angle_rad) % 180

    projection = np.dot(points, main_direction)
    length_pixels = np.max(projection) - np.min(projection)
    
    ortho_direction = np.array([-main_direction[1], main_direction[0]])
    ortho_projection = np.dot(points, ortho_direction)
    width_pixels = np.max(ortho_projection) - np.min(ortho_projection)

    length_mm = length_pixels * PIXEL_TO_MM_FACTOR
    width_mm = width_pixels * PIXEL_TO_MM_FACTOR
    if width_mm < 0.05: width_mm = 0.05

    scale_factor = length_pixels / 2
    start_point = (mean - main_direction * scale_factor).astype(int)
    end_point = (mean + main_direction * scale_factor).astype(int)
    start_point_cv = (start_point[1], start_point[0])
    end_point_cv = (end_point[1], end_point[0])

    return angle_deg, width_mm, length_mm, (start_point_cv, end_point_cv)

def evaluate_risk(width_mm):
    if width_mm < 0.1:
        return "Grade A (Excellent)", (0, 255, 0)
    elif width_mm < 0.2:
        return "Grade B (Good)", (0, 255, 128)
    elif width_mm < 0.3:
        return "Grade C (Fair)", (0, 165, 255)
    elif width_mm < 0.5:
        return "Grade D (Poor)", (0, 0, 255)
    else:
        return "Grade E (Critical)", (0, 0, 139)

def analyze_cracks():
    try:
        model = YOLO(MODEL_PATH)
    except Exception:
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not image_files:
        return

    for image_name in image_files:
        image_path = os.path.join(IMAGE_DIR, image_name)
        img = cv2.imread(image_path)
        if img is None: continue

        results = model.predict(source=img, save=False, conf=0.25, verbose=False)[0]
        if results.masks is None: continue

        annotated_img = img.copy()

        for i, mask_tensor in enumerate(results.masks.data):
            mask_np = mask_tensor.cpu().numpy()
            img_h, img_w = annotated_img.shape[:2]
            mask_resized = cv2.resize(mask_np, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
            mask_binary = mask_resized > 0.5

            angle, width_mm, length_mm, line_coords = get_crack_properties(mask_binary)

            if angle is not None:
                grade, color = evaluate_risk(width_mm)

                annotated_img[mask_binary] = annotated_img[mask_binary] * 0.5 + np.array(color) * 0.5

                start_p, end_p = line_coords
                cv2.arrowedLine(annotated_img, start_p, end_p, (255, 0, 0), 2, tipLength=0.05)
                cv2.arrowedLine(annotated_img, end_p, start_p, (255, 0, 0), 2, tipLength=0.05)

                if results.boxes is not None and i < len(results.boxes):
                    box = results.boxes.xyxy[i].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = box
                    
                    class_id = int(results.boxes.cls[i])
                    conf = float(results.boxes.conf[i])
                    class_name = model.names[class_id]

                    cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)

                    info_lines = [
                        f"{class_name} {conf:.2f}",
                        f"Risk: {grade}",
                        f"W: {width_mm:.2f}mm",
                        f"L: {length_mm:.1f}mm",
                        f"Ang: {angle:.1f} deg"
                    ]

                    text_x = x1 + 5
                    text_y = y1 + 20
                    
                    for line in info_lines:
                        (w, h), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        cv2.rectangle(annotated_img, (text_x - 2, text_y - h - 2), (text_x + w + 2, text_y + 2), (255, 255, 255), -1)
                        cv2.putText(annotated_img, line, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                        text_y += 20

        output_path = os.path.join(OUTPUT_DIR, f"risk_eval_{image_name}")
        cv2.imwrite(output_path, annotated_img)

if __name__ == "__main__":
    analyze_cracks()