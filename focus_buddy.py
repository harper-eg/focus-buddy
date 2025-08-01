import cv2
import time
import threading
import pygame
from face_orientation_detector import FaceOrientationDetector


class FocusBuddy:
    def __init__(self, alert_delay=1.0):
        self.detector = FaceOrientationDetector()
        self.alert_delay = alert_delay  # seconds before alerting
        self.look_away_start_time = None
        self.is_looking_away = False
        self.alert_active = False
        
        # Initialize pygame for sound
        pygame.mixer.init()
        self.alert_sound = None
        
        # Try to create a simple beep sound
        try:
            # Create a simple beep tone
            import numpy as np
            sample_rate = 22050
            duration = 0.5  # seconds
            frequency = 800  # Hz
            
            # Generate sine wave
            frames = int(duration * sample_rate)
            arr = np.zeros(frames)
            for i in range(frames):
                arr[i] = np.sin(2 * np.pi * frequency * i / sample_rate)
            
            # Convert to pygame sound
            arr = (arr * 32767).astype(np.int16)
            arr = np.repeat(arr.reshape(frames, 1), 2, axis=1)  # Stereo
            self.alert_sound = pygame.sndarray.make_sound(arr)
        except Exception as e:
            print(f"Warning: Could not create alert sound: {e}")
            self.alert_sound = None
    
    def play_alert_sound(self):
        """Play alert sound if available"""
        if self.alert_sound:
            try:
                self.alert_sound.play()
            except Exception as e:
                print(f"Warning: Could not play alert sound: {e}")
    
    def update_look_away_status(self, face_turned_away):
        """Update the look-away timer and alert status"""
        current_time = time.time()
        
        if face_turned_away:
            if not self.is_looking_away:
                # Just started looking away
                self.is_looking_away = True
                self.look_away_start_time = current_time
                self.alert_active = False
            elif not self.alert_active and (current_time - self.look_away_start_time) >= self.alert_delay:
                # Been looking away for long enough, trigger alert
                self.alert_active = True
                self.play_alert_sound()
        else:
            # Looking at camera
            if self.is_looking_away:
                # Just returned to looking at camera
                self.is_looking_away = False
                self.look_away_start_time = None
                self.alert_active = False
    
    def get_display_color(self):
        """Get the background color based on current status"""
        if self.alert_active:
            return (0, 0, 255)  # Red - alert active
        elif self.is_looking_away:
            return (0, 165, 255)  # Orange - warning (looking away but not long enough)
        else:
            return (0, 255, 0)  # Green - focused
    
    def get_status_text(self):
        """Get status text to display"""
        if self.alert_active:
            return "LOOK BACK AT SCREEN!"
        elif self.is_looking_away:
            elapsed = time.time() - self.look_away_start_time
            remaining = max(0, self.alert_delay - elapsed)
            return f"Look away detected... {remaining:.1f}s"
        else:
            return "Focused - Good job!"
    
    def run(self):
        """Main application loop"""
        # Try different camera indices
        cap = None
        for camera_index in [0, 1]:
            cap = cv2.VideoCapture(camera_index)
            if cap.isOpened():
                ret, test_frame = cap.read()
                if ret:
                    print(f"Using camera index {camera_index}")
                    break
                else:
                    cap.release()
                    cap = None
            else:
                if cap:
                    cap.release()
                    cap = None
        
        if cap is None:
            print("Error: Could not open any camera")
            return
        
        print("Focus Buddy Started!")
        print("Press 'q' to quit")
        print(f"Alert delay: {self.alert_delay} seconds")
        
        # Create a window
        cv2.namedWindow('Focus Buddy', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Focus Buddy', 400, 300)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            frame = cv2.flip(frame, 1)  # Mirror the image
            
            # Detect face orientation
            face_turned_away, pose_info, _ = self.detector.detect_face_orientation(frame)
            
            # Update status
            self.update_look_away_status(face_turned_away)
            
            # Create clean display (no face mesh)
            display_frame = frame.copy()
            
            # Add colored overlay based on status
            color = self.get_display_color()
            overlay = display_frame.copy()
            cv2.rectangle(overlay, (0, 0), (display_frame.shape[1], display_frame.shape[0]), color, -1)
            
            # Apply overlay with transparency
            alpha = 0.3 if not self.alert_active else 0.6  # More intense for alert
            cv2.addWeighted(overlay, alpha, display_frame, 1 - alpha, 0, display_frame)
            
            # Add status text
            status_text = self.get_status_text()
            
            # Add semi-transparent background for text
            text_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            cv2.rectangle(display_frame, (10, 10), (text_size[0] + 20, 50), (0, 0, 0), -1)
            
            # Add text
            text_color = (255, 255, 255)
            cv2.putText(display_frame, status_text, (15, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)
            
            # Add pose info if available (smaller, bottom right)
            if pose_info:
                angle_text = f"Pitch: {pose_info['x']:.1f}° Yaw: {pose_info['y']:.1f}°"
                cv2.putText(display_frame, angle_text, (10, display_frame.shape[0] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            
            cv2.imshow('Focus Buddy', display_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        pygame.mixer.quit()


def main():
    # Create and run Focus Buddy with 1-second alert delay
    focus_buddy = FocusBuddy(alert_delay=1.0)
    focus_buddy.run()


if __name__ == "__main__":
    main()