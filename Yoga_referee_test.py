import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, Menu, ttk, simpledialog
from PIL import Image, ImageTk
import json
import os
import math
from datetime import datetime


class PoseEvaluator:
    def __init__(self, min_detection_confidence=0.7, model_complexity=1):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.min_detection_confidence = min_detection_confidence
        self.model_complexity = model_complexity

             
        # Initialize pose model
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=self.min_detection_confidence,
            model_complexity=self.model_complexity
        )
           
        # Scoring variables
        self.current_contestant = None
        self.current_score = 0
        self.performed_poses = []
        self.contestants = {}
        
        # Pose transition tracking - INITIALIZE WITH CURRENT TIME, NOT None
        self.current_pose = None
        self.current_pose_start_time = datetime.now()
        self.current_pose_best_score = 0
        self.pose_hold_threshold = 2.0  # seconds
        self.last_pose_change_time = datetime.now()


    def reset_pose_model(self):
        """Reinitialize the pose model with current settings"""
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=self.min_detection_confidence,
            model_complexity=self.model_complexity
        )

    def calculate_angle(self, a, b, c):
        """Calculate angle between three 3D points a, b, c (in degrees)."""
        try:
            a, b, c = np.array(a), np.array(b), np.array(c)
            ba = a - b
            bc = c - b
            cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
            angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
            return np.degrees(angle)                        #Returns the angles
        except (TypeError, ValueError, IndexError):
            return 0                                        # Return 0 if calculation fails

    def calculate_hip_twist(self, landmarks):
        """Calculate hip twist angle (frontal plane rotation)"""
        left_hip = [landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x,
                    landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y,
                    landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].z]
        
        right_hip = [landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                     landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].y,
                     landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].z]
        
        left_shoulder = [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                         landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y,
                         landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].z]
        
        right_shoulder = [landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                          landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y,
                          landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].z]
        
        hip_line = np.array(right_hip) - np.array(left_hip)
        shoulder_line = np.array(right_shoulder) - np.array(left_shoulder)
        cosine_angle = np.dot(hip_line, shoulder_line) / (np.linalg.norm(hip_line) * np.linalg.norm(shoulder_line) + 1e-6)
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        return np.degrees(angle)

    
    def hand_on_ground(self, landmarks):
        """Check if hands are on the ground using MediaPipe landmarks directly"""
        try:
            left_wrist = [landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].x,      #left wrist list
                        landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].y,
                        landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].z]
            
            right_wrist = [landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].x,    #right wrist list
                        landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].y,
                        landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].z]

            left_ankle = [landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].x,    #left ankle list 
                        landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].y,
                        landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].z]    
                
            right_ankle = [landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,    #right ankle list
                        landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].y,
                        landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].z]
            
            left_knee = [landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].x,    #left knee list
                        landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].y,
                        landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].z]        

            right_knee = [landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].x,    #right knee list
                        landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].y,
                        landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].z] 

            """calling distance function and checking if the hands are close to feets""" 
            ref_dist = (self.distance(right_knee, right_ankle) + self.distance(left_knee, left_ankle))/2
            
            right_hand_on_ground =  (abs(right_wrist[1] - right_ankle[1] )< ref_dist)
            left_hand_on_ground =   (abs(left_wrist[1] - left_ankle[1]) < ref_dist)


            """returning bollean"""
            return left_hand_on_ground or right_hand_on_ground
        except (IndexError, AttributeError):
            return False


    
    def touching_toes(self, landmarks):
        """Check if hands are touching toes using MediaPipe landmarks directly"""
        try:
            left_wrist = [landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].x,      #left wrist list
                        landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].y,
                        landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].z]
            
            right_wrist = [landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].x,    #right wrist list
                        landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].y,
                        landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].z]

            left_ankle = [landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].x,    #left ankle list 
                        landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].y,
                        landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].z]    
                
            right_ankle = [landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,    #right ankle list
                        landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].y,
                        landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].z]
            
            left_knee = [landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].x,    #left knee list
                        landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].y,
                        landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].z]        

            right_knee = [landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].x,    #right knee list
                        landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].y,
                        landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].z] 

            """calling distance function and checking if the hands are close to feets""" 
            ref_dist = (self.distance(right_knee, right_ankle) + self.distance(left_knee, left_ankle))/2

            right_hand_touch = (1.5*self.distance(right_wrist , right_ankle ) < ref_dist)
            left_hand_touch = (1.5*self.distance(left_wrist , left_ankle) < ref_dist)
      
            return left_hand_touch and right_hand_touch
        except (IndexError, AttributeError):
            return False


    def distance(self, point1, point2):
        """Calculate the Euclidean distance between two 3D points."""
        x1, y1, z1 = point1
        x2, y2, z2 = point2
        dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
        return dist
    
    def classify_pose(self, landmarks):
        """Classify yoga pose and calculate score with transition handling"""
        keypoints = {}
        for landmark in self.mp_pose.PoseLandmark:
            try:
                keypoints[landmark.name.lower()] = [
                    landmarks[landmark.value].x,
                    landmarks[landmark.value].y,
                    landmarks[landmark.value].z
                ]
            except (IndexError, AttributeError):
                # If landmark is not detected, use zeros
                keypoints[landmark.name.lower()] = [0, 0, 0]    
            
    # Safe angle calculations with default values
        angles = {
                'left_shoulder': self.calculate_angle(
                    keypoints.get('left_elbow', [0,0,0]), 
                    keypoints.get('left_shoulder', [0,0,0]), 
                    keypoints.get('left_hip', [0,0,0])),
                'right_shoulder': self.calculate_angle(
                    keypoints.get('right_elbow', [0,0,0]), 
                    keypoints.get('right_shoulder', [0,0,0]), 
                    keypoints.get('right_hip', [0,0,0])),                    
                'left_elbow': self.calculate_angle(
                    keypoints.get('left_shoulder', [0,0,0]), 
                    keypoints.get('left_elbow', [0,0,0]), 
                    keypoints.get('left_wrist', [0,0,0])),
                'right_elbow': self.calculate_angle(
                    keypoints.get('right_shoulder', [0,0,0]), 
                    keypoints.get('right_elbow', [0,0,0]), 
                    keypoints.get('right_wrist', [0,0,0])),
                'left_knee': self.calculate_angle(
                    keypoints.get('left_hip', [0,0,0]), 
                    keypoints.get('left_knee', [0,0,0]), 
                    keypoints.get('left_ankle', [0,0,0])),
                'right_knee': self.calculate_angle(
                    keypoints.get('right_hip', [0,0,0]), 
                    keypoints.get('right_knee', [0,0,0]), 
                    keypoints.get('right_ankle', [0,0,0])),
                'left_hip': self.calculate_angle(
                    keypoints.get('left_shoulder', [0,0,0]), 
                    keypoints.get('left_hip', [0,0,0]), 
                    keypoints.get('left_knee', [0,0,0])),
                'right_hip': self.calculate_angle(
                    keypoints.get('right_shoulder', [0,0,0]), 
                    keypoints.get('right_hip', [0,0,0]), 
                    keypoints.get('right_knee', [0,0,0])),
                'hip_twist': self.calculate_hip_twist(landmarks)
            }
        
        # Get base classification from angles
        new_pose_name, new_pose_score = self._classify_current_pose(angles, keypoints)
        
            # Check for specific poses that require landmark analysis
        if new_pose_name == "Unknown":
            if self.touching_toes(landmarks):
                new_pose_name = "touching toes"
                new_pose_score = self.evaluate_touching_toes(angles)
            elif self.hand_on_ground(landmarks) and (
                (145 < angles['left_hip'] and 145 < angles['left_knee']) or      
                (145 < angles['right_knee'] and 145 < angles['right_hip'])):
                new_pose_name = "push ups"   
                new_pose_score = self.evaluate_push_ups(angles)
            elif ((55 < angles['left_knee'] < 120 and 55 < angles['left_hip'] < 120 and 
                140 < angles['right_knee'] and 140 < angles['right_hip']) or
                (55 < angles['right_knee'] < 120 and 55 < angles['right_hip'] < 120 and 
                140 < angles['left_knee'] and 140 < angles['left_hip'])):
                new_pose_name = "Standing knee raise"
                new_pose_score = self.evaluate_standing_knee_raise(angles)
            elif ((60 < angles['left_knee'] < 120 and 60 < angles['right_knee'] < 130) and
                angles['left_elbow'] > 160 and angles['right_elbow'] > 160 and
                60 < angles['left_hip'] < 120 and 60 < angles['right_hip'] < 120):
                new_pose_name = "Table Pose"
                new_pose_score = self.evaluate_table_pose(angles)
            elif (angles['left_knee'] < 45 and angles['right_knee'] < 45 and
                angles['left_elbow'] > 160 and angles['right_elbow'] > 160 and
                angles['left_hip'] < 45 and angles['right_hip'] < 45):
                new_pose_name = "Child Pose"
                new_pose_score = self.evaluate_child_pose(angles)
        

        # Pose transition logic
        if new_pose_name != self.current_pose:
            if self.current_pose and self.current_pose != "Unknown":
                self._record_pose_score()
            
            self.current_pose = new_pose_name
            self.current_pose_start_time = datetime.now()
            self.current_pose_best_score = new_pose_score
            self.last_pose_change_time = datetime.now()
        else:
            if new_pose_score > self.current_pose_best_score:
                self.current_pose_best_score = new_pose_score
        
        return new_pose_name, new_pose_score

    def _record_pose_score(self):
        """Record the best score for the completed pose"""
        if self.current_contestant and self.current_pose_best_score > 0:
        # Add safety check for current_pose_start_time
            if self.current_pose_start_time is None:
                self.current_pose_start_time = datetime.now()
            
            hold_duration = (datetime.now() - self.current_pose_start_time).total_seconds()
            
            self.performed_poses.append({
                "pose": self.current_pose,
                "score": self.current_pose_best_score,
                "hold_time": hold_duration,
                "timestamp": self.current_pose_start_time.strftime("%Y-%m-%d %H:%M:%S"),
                "other_variations": self.other_variations_poses(self.current_pose)
            })
            
            self.current_score += self.current_pose_best_score
            self.current_pose_best_score = 0

    def other_variations_poses(self, pose_name):
        """Determine if a pose is a child/variation of another pose"""
        Kids = {
            "Tree Pose": ["Half Tree Pose", "Revolved Tree Pose"],
            "Triangle Pose": ["Revolved Triangle", "Bound Triangle"]
        }
        
        for parent, children in Kids.items():
            if pose_name in children:
                return True
        return False

    def _classify_current_pose(self, angles, keypoints):
        """Classify the current pose with child pose detection"""
        pose_name = "Unknown"
        score = 0
        

        # downward dog variations
        if ((50 < angles['left_hip'] < 90 or 50 < angles['right_hip'] < 90) and     #hip
            (155 < angles['left_knee']  or 155 < angles['right_knee'] ) and         #knee
            (60 < angles['left_elbow']  or 60 < angles['right_elbow'] ) and         #elbow
            (125 < angles['left_knee']  or 125 < angles['right_shoulder'] )         #shoulder
            ):
            pose_name = "downward dog"
            score = self.evaluate_downward_dog(angles)

        return pose_name, score

        """    

        # Touching toes
        elif (self.touching_toes(landmarks)):
            pose_name = "touching toes"
            score = self.evaluate_touching_toes(angles)    

        # Push ups
        elif (self.hand_on_ground(landmarks) and 
            (
            (145 < angles['left_hip']   and 145 < angles['left_knee']) or      
            (145 < angles['right_knee'] and 145 < angles['right_hip'])       
            )):
            pose_name = "push ups"   
            score = self.evaluate_push_ups(angles)

        # Standing and knee raise
        elif ((55 < angles['left_knee'] < 120 and 55 < angles['left_hip'] < 120 and 
              140 < angles['right_knee'] and 140 < angles['right_hip'] ) or
              (55 < angles['right_knee'] < 120 and 55 < angles['right_hip'] < 120 and 
              140 < angles['left_knee'] and 140 < angles['left_hip'] )
            ):
            pose_name = "Standing knee raise"
            score = self.evaluate_standing_knee_raise(angles)


        # Table pose
        elif ((60 < angles['left_knee'] < 120 and 60 < angles['right_knee'] < 130) and
              angles['left_elbow'] > 160 and angles['right_elbow'] > 160 and
              60 < angles['left_hip'] < 120 and 60 < angles['right_hip'] < 120
              ):
              pose_name = "Table Pose"
              score = self.evaluate_table_pose(angles)
              

        # Child Pose
        elif (angles['left_knee'] < 45 and angles['right_knee'] < 45 and
              angles['left_elbow'] > 160 and angles['right_elbow'] > 160 and
              angles['left_hip'] < 45 and angles['right_hip'] < 45 ):
              pose_name = "Child Pose"
              score = self.evaluate_child_pose(angles)
              """


    def evaluate_downward_dog(self, angles):
        """Evaluate downward dog and return score (0-100)"""
        ideal_hip_angle = 70
        ideal_knee_angle = 180
        ideal_shoulder_angle = 180 
        
        hip_deviation = min (abs(angles['right_hip'] - ideal_hip_angle), abs(angles['left_hip'] - ideal_hip_angle))
        knee_deviation = min(abs(angles['right_knee'] - ideal_knee_angle),abs(angles['left_knee'] - ideal_knee_angle) )
        shoulder_deviation = min (abs(angles['right_shoulder']- ideal_shoulder_angle), abs(angles['left_shoulder']- ideal_shoulder_angle))

        hip_score = max(0, 100 - (hip_deviation * 2))
        knee_score = max(0, 100 - (knee_deviation * 2))
        shoulder_score = max(0, 100 - (shoulder_deviation * 2))
        
        return (hip_score + knee_score + shoulder_score) / 3
    
    def evaluate_touching_toes(self, angles):
        """Evaluates stouching toes return score (0-100)"""
        ideal_knee = 180

        knee_deviation = min (abs(ideal_knee - angles['right_knee']) ,  abs(ideal_knee - angles['left_knee']))
        
        knee_score = max(0, 100 - knee_deviation)

        return knee_score
    
    def evaluate_standing_knee_raise(self, angles):
        """Evaluates standing with one knee raised"""
        ideal_knee_max = 180
        ideal_knee_min = 75
        ideal_hip_max = 180
        ideal_hip_min = 75

        knee_max = max (angles['right_knee'], angles['left_knee'])
        knee_min = min (angles['right_knee'], angles['left_knee'])
        hip_max = max (angles['right_hip'], angles['left_hip'])
        hip_min = min (angles['right_hip'], angles['left_hip'])

        knee_deviation = abs(ideal_knee_min - knee_min)/2 + abs(ideal_knee_max - knee_max)/2 
        hip_deviation = abs(ideal_hip_min - hip_min)/2 + abs(ideal_hip_max - hip_max)/2

        knee_score = max(0, 100 - knee_deviation)
        hip_score = max(0, 100 - hip_deviation)

        return (knee_score + hip_score) / 2
    
    def evaluate_push_ups(self, angles):
        """Evaluate push up pose return score (0-100)"""
        ideal_knee = 180
        ideal_hip = 180

        knee_deviation = abs(max(angles['right_knee'],angles['left_knee']) - ideal_knee)
        hip_deviation = abs(max(angles['right_hip'],angles['left_hip']) - ideal_hip)

        knee_score = max(0, 100 - knee_deviation)
        hip_score = max(0, 100 - hip_deviation)

        return (knee_score + hip_score) / 2


    def evaluate_table_pose(self, angles):
        """Evaluate table pose return score (0-100)"""
        ideal_knee = 80
        ideal_hip = 95
        ideal_shoulder = 85 
        ideal_elbow = 180

        knee_deviation = max(abs(angles['right_knee'] - ideal_knee ), abs(angles['left_knee'] - ideal_knee) )
        hip_deviation = max(abs(angles['right_hip'] - ideal_hip ), abs(angles['left_hip'] - ideal_hip) )
        shoulder_deviation = max(abs(angles['right_shoulder'] - ideal_shoulder ), abs(angles['left_shoulder'] - ideal_shoulder) )
        elbow_deviation = max(abs(angles['right_elbow'] - ideal_elbow ), abs(angles['left_elbow'] - ideal_elbow) )

        knee_score = max(0, 100 - knee_deviation)
        hip_score = max(0, 100 - hip_deviation)
        shoulder_score = max(0, 100 - shoulder_deviation)
        elbow_score = max(0, 100 - elbow_deviation)

        return (knee_score + hip_score + elbow_score + shoulder_deviation) / 4

    def evaluate_child_pose(self, angles):
        """Evaluate Child Pose return score (0-100) """
        ideal_knee = 5
        ideal_hip = 5
        ideal_elbow = 180
        
        knee_deviation = abs(min(angles['left_knee'], angles['right_knee']) - ideal_knee)
        hip_deviation = abs(min(angles['right_hip'], angles['left_hip']) - ideal_hip)
        elbow_deviation = abs(max(angles['right_elbow'], angles['left_elbow']) - ideal_elbow)

        knee_score = max(0, 100 - knee_deviation)
        hip_score = max(0, 100 - hip_deviation)
        elbow_score = max(0, 100 - elbow_deviation)
        
        return (knee_score + hip_score + elbow_score) / 2
    

    def draw_skeleton(self, frame, landmarks, w, h):
        """Draw MediaPipe skeleton on frame with additional info"""
        self.mp_drawing.draw_landmarks(
            frame,
            landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
        )
        
        if self.current_contestant:
            cv2.putText(frame, f"Contestant: {self.current_contestant}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Score: {self.current_score}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            if self.current_pose:
                cv2.putText(frame, f"Pose: {self.current_pose}", (10, 110),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    def start_new_contestant(self, name):
        """Start tracking a new contestant"""
        if self.current_contestant:
            self.save_contestant_results()
        
        self.current_contestant = name
        self.current_score = 0
        self.performed_poses = []
        self.current_pose = None
        self.current_pose_best_score = 0
        self.current_pose_start_time = datetime.now()   # Ensure this is set
        self.last_pose_change_time = datetime.now()     # Ensure this is set
        
    def save_contestant_results(self):
        """Save current contestant's results to history"""
        if self.current_contestant:
            if self.current_contestant not in self.contestants:
                self.contestants[self.current_contestant] = []
            
            self.contestants[self.current_contestant].append({
                "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "total_score": self.current_score,
                "poses": self.performed_poses.copy()
            })
            
            self.save_to_file()
            
    def save_to_file(self):
        """Save all contestant data to JSON file"""
        filename = "yoga_contest_results.json"
        with open(filename, 'w') as f:
            json.dump(self.contestants, f, indent=4)
            
    def load_from_file(self):
        """Load contestant data from JSON file"""
        filename = "yoga_contest_results.json"
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                self.contestants = json.load(f)

class YogaRefereeApp:
    def detect_cameras(self):
        """Detect available cameras"""
        cameras = []
        # Test up to 5 cameras
        for i in range(5):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                cameras.append(i)
                cap.release()
        return cameras

    def initialize_camera(self, camera_index):
        """Initialize camera with the given index"""
        if self.cap is not None:
            self.cap.release()
        
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            messagebox.showerror("Error", f"Could not open camera {camera_index}")
            return False
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        return True

    def switch_camera(self, camera_index):
        """Switch to a different camera"""
        if camera_index in self.available_cameras:
            if self.initialize_camera(camera_index):
                self.current_camera_index = camera_index
                messagebox.showinfo("Camera Changed", f"Switched to camera {camera_index}")


    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Yoga Referee")
        self.root.geometry("1200x800")
        
        self.evaluator = PoseEvaluator()
        self.evaluator.load_from_file()
        
            # Camera management
        self.available_cameras = self.detect_cameras()
        self.current_camera_index = 0
        

        self.create_menu()
        self.create_main_layout()

        #""""
        #This is supposed to support multiple cameras:
            # Initialize camera
        self.cap = None
        self.initialize_camera(self.current_camera_index)
        
        self.paused = False
        self.current_frame = None
        

        """
        #this support one camera
        # Initialize camera with error handling
        self.cap = cv2.VideoCapture(0)          #This should be added to options
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Could not open camera")
            self.root.quit()
            return
        
        # Set camera properties for better performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        self.paused = False
        self.current_frame = None
        """

        self.update_video()
        self.root.mainloop()
    
    def create_menu(self):
        menubar = Menu(self.root)
        
        file_menu = Menu(menubar, tearoff=0)
        file_menu.add_command(label="New Contestant", command=self.new_contestant)
        file_menu.add_command(label="Save Results", command=self.save_results)
        file_menu.add_command(label="Reset Current", command=self.reset_current)
        file_menu.add_separator()

        #add camera submenu
        camera_menu = Menu(file_menu, tearoff=0)
        for cam_index in self.available_cameras:
            camera_menu.add_command(
                label=f"Camera {cam_index}", 
                command=lambda idx=cam_index: self.switch_camera(idx)
            )
        file_menu.add_cascade(label="Selsect Camera", menu= camera_menu)
        file_menu.add_separator()


        file_menu.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=file_menu)
        
        settings_menu = Menu(menubar, tearoff=0)
        settings_menu.add_command(label="Detection Settings", command=self.show_detection_settings)
        settings_menu.add_command(label="Scoring Parameters", command=self.show_scoring_settings)
        menubar.add_cascade(label="Settings", menu=settings_menu)
        
        view_menu = Menu(menubar, tearoff=0)
        view_menu.add_command(label="Contestant History", command=self.show_contestant_history)

        
        help_menu = Menu(menubar, tearoff=0)
        help_menu.add_command(label="About", command=self.show_about)
        menubar.add_cascade(label="Help", menu=help_menu)
        
        self.root.config(menu=menubar)
    
    def create_main_layout(self):
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        self.video_frame = tk.Frame(self.main_frame, bg='black')
        self.video_frame.pack(fill=tk.BOTH, expand=True)
        
        self.video_label = tk.Label(self.video_frame)
        self.video_label.pack(fill=tk.BOTH, expand=True)
        
        self.control_frame = tk.Frame(self.main_frame)
        self.control_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.contestant_frame = tk.LabelFrame(self.control_frame, text="Current Contestant")
        self.contestant_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5)
        
        self.contestant_label = tk.Label(self.contestant_frame, text="No contestant", font=('Arial', 12))
        self.contestant_label.pack(padx=10, pady=5)
        
        self.score_label = tk.Label(self.contestant_frame, text="Score: 0", font=('Arial', 12))
        self.score_label.pack(padx=10, pady=5)
        
        self.pose_frame = tk.LabelFrame(self.control_frame, text="Current Pose")
        self.pose_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5)
        
        self.pose_name_label = tk.Label(self.pose_frame, text="No pose detected", font=('Arial', 12))
        self.pose_name_label.pack(padx=10, pady=5)
        
        self.pose_score_label = tk.Label(self.pose_frame, text="Pose Score: 0", font=('Arial', 12))
        self.pose_score_label.pack(padx=10, pady=5)
        
        self.button_frame = tk.Frame(self.control_frame)
        self.button_frame.pack(side=tk.RIGHT, padx=10)
        
        self.new_btn = tk.Button(self.button_frame, text="New Contestant", command=self.new_contestant)
        self.new_btn.pack(fill=tk.X, pady=2)
        
        self.save_btn = tk.Button(self.button_frame, text="Save Results", command=self.save_results)
        self.save_btn.pack(fill=tk.X, pady=2)
        
        self.reset_btn = tk.Button(self.button_frame, text="Reset Current", command=self.reset_current)
        self.reset_btn.pack(fill=tk.X, pady=2)
        
        self.pause_btn = tk.Button(self.button_frame, text="Pause", command=self.toggle_pause)
        self.pause_btn.pack(fill=tk.X, pady=2)
    
    def new_contestant(self):
        name = simpledialog.askstring("New Contestant", "Enter contestant name:")
        if name:
            self.evaluator.start_new_contestant(name)
            self.update_contestant_display()
    
    def save_results(self):
        if self.evaluator.current_contestant:
            self.evaluator.save_contestant_results()
            messagebox.showinfo("Saved", f"Results saved for {self.evaluator.current_contestant}")
        else:
            messagebox.showwarning("No Contestant", "No contestant is currently being tracked")
    
    def reset_current(self):
        if self.evaluator.current_contestant:
            if messagebox.askyesno("Reset", "Reset current contestant's session?"):
                self.evaluator.start_new_contestant(self.evaluator.current_contestant)
                self.update_contestant_display()
    
    def toggle_pause(self):
        self.paused = not self.paused
        self.pause_btn.config(text="Resume" if self.paused else "Pause")
    
    def show_detection_settings(self):
        settings_window = tk.Toplevel(self.root)
        settings_window.title("Detection Settings")
        
        conf_frame = tk.Frame(settings_window)
        conf_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(conf_frame, text="Min Detection Confidence:").pack(side=tk.LEFT)
        self.confidence_var = tk.DoubleVar(value=self.evaluator.min_detection_confidence)
        conf_slider = tk.Scale(conf_frame, from_=0.1, to=1.0, resolution=0.05, 
                              orient=tk.HORIZONTAL, variable=self.confidence_var)
        conf_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        model_frame = tk.Frame(settings_window)
        model_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(model_frame, text="Model Complexity:").pack(side=tk.LEFT)
        self.model_var = tk.IntVar(value=self.evaluator.model_complexity)
        
        tk.Radiobutton(model_frame, text="0 (Fastest)", variable=self.model_var, value=0).pack(side=tk.LEFT)
        rb1 = tk.Radiobutton(model_frame, text="1 (Recommended)", variable=self.model_var, value=1)
        rb1.pack(side=tk.LEFT)
        rb1.config(fg='red')
        tk.Radiobutton(model_frame, text="2 (Most Accurate)", variable=self.model_var, value=2).pack(side=tk.LEFT)
        
        save_btn = tk.Button(settings_window, text="Apply Settings", 
                            command=self.apply_detection_settings)
        save_btn.pack(pady=10)
    
    def apply_detection_settings(self):
        self.evaluator.min_detection_confidence = self.confidence_var.get()
        self.evaluator.model_complexity = self.model_var.get()
        self.evaluator.reset_pose_model()
        messagebox.showinfo("Settings Updated", "Detection settings have been updated")
    
    def show_scoring_settings(self):
        messagebox.showinfo("Info", "Scoring parameters can be adjusted in the PoseEvaluator class")
    
    def show_contestant_history(self):
        history_window = tk.Toplevel(self.root)
        history_window.title("Contestant History")
        
        tree = ttk.Treeview(history_window, columns=('name', 'date', 'score'), show='headings')
        tree.heading('name', text='Contestant')
        tree.heading('date', text='Date')
        tree.heading('score', text='Score')
        tree.pack(fill=tk.BOTH, expand=True)
        
        for name, sessions in self.evaluator.contestants.items():
            for session in sessions:
                tree.insert('', tk.END, values=(
                    name, 
                    session['date'], 
                    session['total_score']
                ))
    
    def show_about(self):
        messagebox.showinfo("About", 
                          "Yoga Referee App\nVersion 1.02\n\n"
                          "Automated yoga pose detection and scoring system")
    
    def update_contestant_display(self):
        if self.evaluator.current_contestant:
            self.contestant_label.config(text=self.evaluator.current_contestant)
            self.score_label.config(text=f"Score: {self.evaluator.current_score}")
        else:
            self.contestant_label.config(text="No contestant")
            self.score_label.config(text="Score: 0")
    
    def update_video(self):
        if self.cap is not None and not self.paused:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to grab frame")
                    self.root.after(10, self.update_video)
                    return
                    
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.evaluator.pose.process(frame_rgb)
                
                if results and results.pose_landmarks:
                    try:
                        self.evaluator.draw_skeleton(frame, results.pose_landmarks, frame.shape[1], frame.shape[0])
                        pose_name, pose_score = self.evaluator.classify_pose(results.pose_landmarks.landmark)
                        
                        self.pose_name_label.config(text=pose_name)
                        self.pose_score_label.config(text=f"Pose Score: {pose_score}")
                    except Exception as e:
                        print(f"Error in pose processing: {e}")
                
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                imgtk = ImageTk.PhotoImage(image=img)
                
                self.current_frame = imgtk
                self.video_label.config(image=self.current_frame)
                
                self.update_contestant_display()
                
            except Exception as e:
                print(f"Error in video update: {e}")
                # Try to reinitialize camera
                try:
                    self.cap.release()
                except:
                    pass
                self.initialize_camera(self.current_camera_index)
        
        self.root.after(10, self.update_video)

        

if __name__ == "__main__":
    app = YogaRefereeApp()