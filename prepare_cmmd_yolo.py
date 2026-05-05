import os
import glob
import json
import numpy as np
import cv2
import pydicom
from pathlib import Path
from tqdm import tqdm

def get_dicom_info(dcm_path):
    # Read only the header to get metadata first
    dcm_header = pydicom.dcmread(dcm_path, stop_before_pixels=True)
    patient_id = getattr(dcm_header, 'PatientID', None)
    laterality = getattr(dcm_header, 'ImageLaterality', None)
    view = getattr(dcm_header, 'ViewPosition', None)
    
    if not view and 'ViewCodeSequence' in dcm_header:
        meaning = dcm_header.ViewCodeSequence[0].CodeMeaning.lower()
        if 'oblique' in meaning and 'lateral' in meaning:
            view = 'MLO'
        elif 'cranio' in meaning:
            view = 'CC'
        else:
            view = meaning.upper()
    
    # DICOM strings often have trailing spaces, we must strip them
    if patient_id: patient_id = str(patient_id).strip()
    if laterality: laterality = str(laterality).strip()
    if view: view = str(view).strip()
    
    return patient_id, laterality, view

def apply_clahe(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)

def crop_breast(image, polygons=None):
    """
    Crop the image to fill the breast area, removing background.
    """
    # Smooth to reduce noise
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    # Threshold to find breast mask
    ret, thresh = cv2.threshold(blurred, 15, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        # If no contours found, return original image
        return image, polygons, 0, 0, image.shape[1], image.shape[0]
        
    # Assume the largest contour is the breast
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Crop the image
    cropped = image[y:y+h, x:x+w]
    
    # Adjust polygons based on the crop
    new_polygons = []
    if polygons:
        for poly in polygons:
            new_poly = []
            for pt in poly:
                # Clamp coordinates to the bounds of the cropped image
                new_pt_x = max(0, min(pt[0] - x, w - 1))
                new_pt_y = max(0, min(pt[1] - y, h - 1))
                new_poly.append([new_pt_x, new_pt_y])
            new_polygons.append(new_poly)
            
    return cropped, new_polygons, x, y, w, h

def process_and_save(dcm_path, json_path, output_dir_base, class_map, target_size=(1600, 1600)):
    patient_id, laterality, view = get_dicom_info(dcm_path)
    
    # Now read the full DICOM
    dcm = pydicom.dcmread(dcm_path)
    image = dcm.pixel_array
    
    # Handle photometric interpretation (MONOCHROME1 means inverted colors)
    if getattr(dcm, 'PhotometricInterpretation', '') == 'MONOCHROME1':
        image = np.amax(image) - image
        
    # Normalize image to 0-255 uint8
    image = image.astype(np.float32)
    image = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-8)
    image = (image * 255).astype(np.uint8)
    
    # Read polygons from JSON
    polygons = []
    classes = []
    has_mass = False
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
            for obj in data:
                label = obj.get('label', 'unknown').strip()
                if label.lower() == 'mass':
                    has_mass = True
                    if label not in class_map:
                        class_map[label] = len(class_map)
                    classes.append(class_map[label])
                    
                    pts = obj.get('cgPoints', [])
                    if pts:
                        poly = [[pt['x'], pt['y']] for pt in pts]
                        polygons.append(poly)
                        
    # If no "mass" label found, skip processing this image
    if not has_mass:
        return False
        
    print(f"Found mass in {json_path}")

                
    # 1. Crop Breast
    image, polygons, cx, cy, cw, ch = crop_breast(image, polygons)
    
    # 2. Right Flipped into Left (and labels follow)
    if laterality == 'R':
        image = cv2.flip(image, 1)
        new_polygons = []
        for poly in polygons:
            new_poly = []
            for pt in poly:
                # Flip x coordinate
                new_poly.append([cw - 1 - pt[0], pt[1]])
            new_polygons.append(new_poly)
        polygons = new_polygons
        
    # 3. Resize to 1600x1600
    h, w = image.shape[:2]
    image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    
    # Calculate scale factors
    scale_x = target_size[0] / w
    scale_y = target_size[1] / h
    
    # Transform polygons to normalized YOLO bounding boxes
    yolo_bboxes = []
    for cls_idx, poly in zip(classes, polygons):
        xs = []
        ys = []
        for pt in poly:
            # Scale to new size
            px = pt[0] * scale_x
            py = pt[1] * scale_y
            # Normalize to 0-1
            px /= target_size[0]
            py /= target_size[1]
            xs.append(px)
            ys.append(py)
        
        if len(xs) > 0:
            x_min = min(xs)
            x_max = max(xs)
            y_min = min(ys)
            y_max = max(ys)
            
            x_center = (x_min + x_max) / 2.0
            y_center = (y_min + y_max) / 2.0
            width = x_max - x_min
            height = y_max - y_min
            
            yolo_bboxes.append((cls_idx, [x_center, y_center, width, height]))
        
    # Create CLAHE version
    image_clahe = apply_clahe(image)
    
    # Save both versions
    base_filename = f"{patient_id}_{view}_{laterality}"
    
    for use_clahe in [False, True]:
        out_dir = os.path.join(output_dir_base, 'clahe' if use_clahe else 'no_clahe')
        os.makedirs(os.path.join(out_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(out_dir, 'labels'), exist_ok=True)
        
        img_to_save = image_clahe if use_clahe else image
        img_path = os.path.join(out_dir, 'images', f"{base_filename}.jpg")
        cv2.imwrite(img_path, img_to_save)
        
        txt_path = os.path.join(out_dir, 'labels', f"{base_filename}.txt")
        with open(txt_path, 'w') as f:
            for cls_idx, bbox in yolo_bboxes:
                bbox_str = " ".join([f"{v:.6f}" for v in bbox])
                f.write(f"{cls_idx} {bbox_str}\n")
                
    return True

def main():
    cmmd_dir = r"c:\Users\yeo\Documents\ultralytics\cmmd"
    labels_dir = r"c:\Users\yeo\Documents\ultralytics\TOMPEI-CMMD_v01_20250123"
    output_dir = r"c:\Users\yeo\Documents\ultralytics\cmmd_yolo_dataset"
    
    class_map = {}
    
    dcm_files = glob.glob(os.path.join(cmmd_dir, "**", "*.dcm"), recursive=True)
    print(f"Found {len(dcm_files)} DICOM files in {cmmd_dir}.")
    
    processed_count = 0
    matched_json_count = 0
    
    for dcm_path in tqdm(dcm_files, desc="Processing DICOMs"):
        try:
            patient_id, laterality, view = get_dicom_info(dcm_path)
            
            if not all([patient_id, laterality, view]):
                continue
                
            json_filename = f"{patient_id}_{view}_{laterality}_AnnotationFile.json"
            json_path = os.path.join(labels_dir, json_filename)
            
            # Process if we have the corresponding annotation
            if os.path.exists(json_path):
                matched_json_count += 1
                success = process_and_save(dcm_path, json_path, output_dir, class_map)
                if success:
                    processed_count += 1
            else:
                pass
                
        except Exception as e:
            print(f"Error processing {dcm_path}: {e}")
            
    print(f"\nMatched {matched_json_count} DICOMs to annotation files.")
    print(f"Successfully processed {processed_count} images (only keeping those with 'mass' labels).")
    
    print("\nClass mapping:")
    print(json.dumps(class_map, indent=4))
    
    # Generate data.yaml for YOLO training
    for use_clahe in ['no_clahe', 'clahe']:
        out_dataset_dir = os.path.join(output_dir, use_clahe)
        if not os.path.exists(out_dataset_dir):
            continue
            
        yaml_content = "path: ./\ntrain: images\nval: images\n\nnames:\n"
        sorted_classes = sorted(class_map.items(), key=lambda x: x[1])
        for name, idx in sorted_classes:
            yaml_content += f"  {idx}: {name}\n"
            
        with open(os.path.join(out_dataset_dir, 'data.yaml'), 'w') as f:
            f.write(yaml_content)
    
    print(f"\nYOLO dataset saved in: {output_dir}")

if __name__ == "__main__":
    main()
