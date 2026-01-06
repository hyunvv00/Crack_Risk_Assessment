import os
import cv2
import numpy as np
from ultralytics import YOLO

MODEL_PATH = "runs/segment/train/weights/best.pt" 
IMAGE_DIR = "path to data file"
OUTPUT_DIR = "crack_detection_output/crack_angle"

def get_crack_angle(mask_data):
    points = np.argwhere(mask_data)
    
    if len(points) < 2:
        return None, None
    
    mean = np.mean(points, axis=0)
    cov = np.cov(points, rowvar=False)
    
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    main_direction = eigenvectors[:, -1]
    
    angle_rad = np.arctan2(main_direction[0], main_direction[1])
    angle_deg = np.degrees(angle_rad) % 180

    scale_factor = 200
    start_point = (mean - main_direction * scale_factor).astype(int)
    end_point = (mean + main_direction * scale_factor).astype(int)
    
    start_point_cv = (start_point[1], start_point[0])
    end_point_cv = (end_point[1], end_point[0])

    return angle_deg, (start_point_cv, end_point_cv)

def analyze_crack_angles():
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
        if img is None:
            continue

        results = model.predict(source=img, save=False, conf=0.25, verbose=False)[0]
        
        if results.masks is None:
            continue

        annotated_img = img.copy()
        
        for i, mask_tensor in enumerate(results.masks.data):
            mask_np_raw = mask_tensor.cpu().numpy()
            img_height, img_width = annotated_img.shape[:2]
            mask_resized = cv2.resize(mask_np_raw, (img_width, img_height), interpolation=cv2.INTER_NEAREST)
            mask_np = mask_resized > 0.5 
            
            angle, line_coords = get_crack_angle(mask_np)

            if angle is not None:
                mask_area = mask_np 
                annotated_img[mask_area] = annotated_img[mask_area] * 0.5 + np.array([255, 0, 0]) * 0.5 
                
                start_p, end_p = line_coords
                cv2.arrowedLine(annotated_img, start_p, end_p, (0, 0, 255), 2, tipLength=0.05)
                cv2.arrowedLine(annotated_img, end_p, start_p, (0, 0, 255), 2, tipLength=0.05)
                
                text = f"{angle:.1f} deg" 
                mid_point_x = int((start_p[0] + end_p[0]) / 2)
                mid_point_y = int((start_p[1] + end_p[1]) / 2)
                cv2.putText(annotated_img, text, (mid_point_x + 10, mid_point_y + 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                if results.boxes is not None and i < len(results.boxes):
                    box = results.boxes.xyxy[i].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = box
                    
                    cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    class_id = int(results.boxes.cls[i])
                    conf = float(results.boxes.conf[i])
                    label = f"{model.names[class_id]} {conf:.2f}"
                    cv2.putText(annotated_img, label, (x1 + 5, y1 + 15), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
        output_path = os.path.join(OUTPUT_DIR, f"analyzed_{image_name}")
        cv2.imwrite(output_path, annotated_img)

if __name__ == "__main__":
    analyze_crack_angles()