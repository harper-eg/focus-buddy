import cv2
import mediapipe as mp
import numpy as np
import math


class FaceOrientationDetector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Add smoothing for angle values
        self.angle_history = []
        self.history_size = 50
        
        # Key facial landmark indices for orientation calculation
        # Using stable landmarks that don't change with expressions
        self.FACE_LANDMARKS = {
            'nose_tip': 1,        # Nose tip
            'nose_bridge': 168,   # Top of nose bridge
            'left_eye_corner': 33,  # Left eye inner corner
            'right_eye_corner': 263, # Right eye inner corner
            'left_temple': 172,   # Left temple area
            'right_temple': 397   # Right temple area
        }
    
    def calculate_head_pose_simple(self, landmarks, image_shape):
        """Simple head pose calculation using geometric relationships"""
        height, width = image_shape[:2]
        
        # Get key points
        nose_tip = landmarks[self.FACE_LANDMARKS['nose_tip']]
        left_eye = landmarks[self.FACE_LANDMARKS['left_eye_corner']]
        right_eye = landmarks[self.FACE_LANDMARKS['right_eye_corner']]
        
        # Convert to pixel coordinates
        nose_x = nose_tip.x * width
        nose_y = nose_tip.y * height
        left_eye_x = left_eye.x * width
        left_eye_y = left_eye.y * height
        right_eye_x = right_eye.x * width
        right_eye_y = right_eye.y * height
        
        # Calculate eye center
        eye_center_x = (left_eye_x + right_eye_x) / 2
        eye_center_y = (left_eye_y + right_eye_y) / 2
        
        # Calculate eye distance (for normalization)
        eye_distance = abs(right_eye_x - left_eye_x)
        
        # Calculate angles using relative positions
        # Y-axis rotation (yaw) - left/right head turn
        # Based on nose position relative to eye center
        nose_offset_x = nose_x - eye_center_x
        yaw = np.arctan2(nose_offset_x, eye_distance * 2) * 180 / np.pi
        
        # X-axis rotation (pitch) - up/down head movement
        # Based on nose position relative to eye line
        nose_offset_y = nose_y - eye_center_y
        pitch = np.arctan2(nose_offset_y, eye_distance) * 180 / np.pi
        
        # Z-axis rotation (roll) - head tilt
        # Based on eye line angle
        roll = np.arctan2(right_eye_y - left_eye_y, right_eye_x - left_eye_x) * 180 / np.pi
        
        # Store calculation details for visualization
        self.yaw_calc_data = {
            'nose_x': nose_x,
            'nose_y': nose_y,
            'eye_center_x': eye_center_x,
            'eye_center_y': eye_center_y,
            'nose_offset_x': nose_offset_x,
            'eye_distance': eye_distance,
            'yaw_radians': np.arctan2(nose_offset_x, eye_distance * 2),
            'yaw_degrees': yaw
        }
        
        return pitch, yaw, roll
    
    def draw_yaw_visualization(self, image):
        """Draw visualization of the arctan2 yaw calculation"""
        if not hasattr(self, 'yaw_calc_data'):
            return image
        
        data = self.yaw_calc_data
        
        # Draw eye center point
        eye_center = (int(data['eye_center_x']), int(data['eye_center_y']))
        cv2.circle(image, eye_center, 5, (0, 255, 255), -1)  # Yellow circle
        cv2.putText(image, "Eye Center", (eye_center[0] + 10, eye_center[1]), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        
        # Draw nose point
        nose_pos = (int(data['nose_x']), int(data['nose_y']))
        cv2.circle(image, nose_pos, 5, (255, 0, 0), -1)  # Blue circle
        cv2.putText(image, "Nose", (nose_pos[0] + 10, nose_pos[1]), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        
        # Draw horizontal line from eye center (reference line)
        ref_line_end = (int(data['eye_center_x'] + data['eye_distance']), int(data['eye_center_y']))
        cv2.line(image, eye_center, ref_line_end, (128, 128, 128), 2)
        cv2.putText(image, "Reference", (ref_line_end[0] + 5, ref_line_end[1]), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)
        
        # Draw line from eye center to nose (for yaw calculation)
        cv2.line(image, eye_center, nose_pos, (0, 255, 0), 2)  # Green line
        
        # Draw the offset vector components
        offset_end_x = (int(data['eye_center_x'] + data['nose_offset_x']), int(data['eye_center_y']))
        cv2.line(image, eye_center, offset_end_x, (255, 255, 0), 2)  # Cyan - horizontal offset
        cv2.line(image, offset_end_x, nose_pos, (255, 0, 255), 2)  # Magenta - vertical component
        
        # Add text showing the calculation values
        y_offset = 100
        cv2.putText(image, f"Nose Offset X: {data['nose_offset_x']:.1f}px", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(image, f"Eye Distance * 2: {data['eye_distance'] * 2:.1f}px", 
                   (10, y_offset + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(image, f"arctan2({data['nose_offset_x']:.1f}, {data['eye_distance'] * 2:.1f})", 
                   (10, y_offset + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(image, f"= {data['yaw_radians']:.3f} rad = {data['yaw_degrees']:.1f}°", 
                   (10, y_offset + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return image
    
    def calculate_head_pose(self, landmarks, image_shape):
        """Calculate head pose angles from facial landmarks"""
        # Use simple geometric method instead of PnP for better stability
        return self.calculate_head_pose_simple(landmarks, image_shape)
        
        # The simple method returns pitch, yaw, roll directly
        pitch, yaw, roll = self.calculate_head_pose_simple(landmarks, image_shape)
        
        # Smooth the angles
        current_angles = [pitch, yaw, roll]
        self.angle_history.append(current_angles)
        
        if len(self.angle_history) > self.history_size:
            self.angle_history.pop(0)
        
        # Return smoothed values
        if len(self.angle_history) > 1:
            smoothed = np.mean(self.angle_history, axis=0)
            return smoothed[0], smoothed[1], smoothed[2]
        
        return pitch, yaw, roll

    def is_face_turned_away(self, x, y):
        """Determine if the face is turned away based on pose angles"""
            
        # Check if head is turned significantly in any direction
        # Y-axis rotation (left/right turn) is most important for side-to-side movement
        if x < 10 or x > 27 or y < -8 or y > 8:
            return True
        return False
    
    def detect_face_orientation(self, image):
        """Main function to detect face orientation"""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_image)
        
        face_turned_away = True
        pose_info = None
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Calculate head pose
                x, y, z = self.calculate_head_pose(face_landmarks.landmark, image.shape)
                
                if x is not None:
                    face_turned_away = self.is_face_turned_away(x, y)
                    pose_info = {'x': x, 'y': y, 'z': z}
                
                # Draw face mesh (optional, for visualization)
                self.mp_drawing.draw_landmarks(
                    image, face_landmarks, self.mp_face_mesh.FACEMESH_CONTOURS,
                    None, self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
                )
                
                # Draw yaw calculation visualization
                image = self.draw_yaw_visualization(image)
        
        return face_turned_away, pose_info, image


def main():
    """Demo function to test the face orientation detector"""
    detector = FaceOrientationDetector()
    
    cap = None
    
    cap = cv2.VideoCapture(1)
    
    if cap is None:
        print("Error: Could not open camera")
        return
    
    print("Face Orientation Detector Started")
    print("Press 'q' to quit")
    print("Green text: Face looking at camera")
    print("Red text: Face turned away")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        frame = cv2.flip(frame, 1)  # Mirror the image
        
        face_turned_away, pose_info, processed_frame = detector.detect_face_orientation(frame)
        
        # Display status
        status_text = "Face turned away!" if face_turned_away else "Face looking at camera"
        color = (0, 0, 255) if face_turned_away else (0, 255, 0)  # Red if away, green if looking
        
        cv2.putText(processed_frame, status_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        # Display pose angles if available
        if pose_info:
            angle_text = f"X: {pose_info['x']:.1f}, Y: {pose_info['y']:.1f}, Z: {pose_info['z']:.1f}"
            cv2.putText(processed_frame, angle_text, (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.imshow('Face Orientation Detection', processed_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()