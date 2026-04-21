import cv2
import numpy as np

def filter_by_area(results, min_area=500):
    """
    Filters YOLOv8 results by mask area to remove small noise detections.
    """
    filtered_indices = []
    if results[0].masks is not None:
        masks = results[0].masks.data.cpu().numpy()
        for i, mask in enumerate(masks):
            area = np.sum(mask > 0)
            if area >= min_area:
                filtered_indices.append(i)
    return filtered_indices

def draw_visuals(frame, results, filtered_indices=None):
    """
    Custom drawing function for masks and labels with counter.
    """
    pothole_count = 0
    if results[0].masks is not None:
        masks = results[0].masks.data.cpu().numpy()
        boxes = results[0].boxes.data.cpu().numpy()
        
        for i in range(len(masks)):
            if filtered_indices is not None and i not in filtered_indices:
                continue
                
            pothole_count += 1
            mask = masks[i]
            box = boxes[i]
            
            # Draw Mask
            overlay = frame.copy()
            mask_rgb = np.zeros_like(frame)
            mask_rgb[mask > 0] = (0, 0, 255) # Red for potholes
            cv2.addWeighted(overlay, 0.7, mask_rgb, 0.3, 0, frame)
            
            # Draw Bounding Box
            x1, y1, x2, y2, conf, cls = box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            
            # Draw Label
            label = f"Pothole {conf:.2f}"
            cv2.putText(frame, label, (int(x1), int(y1) - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame, pothole_count

def display_info(frame, count, fps):
    """
    Displays pothole count and FPS on the frame.
    """
    cv2.putText(frame, f"Potholes: {count}", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
    return frame
