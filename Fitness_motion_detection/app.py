import cv2
import mediapipe as mp
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np
import os
import LSTM 
from collections import Counter
import copy
import itertools
LSTM_model = LSTM.load() 


def relative_landmark(landmark_list):
    temp_landmark_list= copy.deepcopy(landmark_list)
    base_x ,base_y, base_z= 0,0,0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index== 0:
            base_x, base_y,base_z = landmark_point[0], landmark_point[1] , landmark_point[2]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y
        temp_landmark_list[index][2] = temp_landmark_list[index][2] - base_z
        
    #正規化#以肩膀長度作為標準化
    shoulder_length_x = abs(temp_landmark_list[11][0]-temp_landmark_list[12][0])
    # shoulder_length_y = abs(temp_landmark_list[11][1]-temp_landmark_list[12][1])
    # shoulder_length_z = abs(temp_landmark_list[11][2]-temp_landmark_list[12][2])
    # print(shoulder_length_x)
    for idx , relative_point in enumerate(temp_landmark_list):
        temp_landmark_list[idx][0] = temp_landmark_list[idx][0] / shoulder_length_x
        temp_landmark_list[idx][1] = temp_landmark_list[idx][1] / shoulder_length_x
        temp_landmark_list[idx][2] = temp_landmark_list[idx][2] / shoulder_length_x

    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
    # print(temp_landmark_list)
    return temp_landmark_list


class FitnessApp:
    def __init__(self, root):
        
        self.root = root
        self.root.iconbitmap('icon.ico')
        self.root.title("FITNESS POSTURE CORRECTION")
        
        self.status_label = ttk.Label(root, text="Press 'Start' to begin.")
        self.status_label.pack(pady=10)

        self.canvas = tk.Canvas(root, width=640, height=480)
        self.canvas.pack()
     

        self.start_button = ttk.Button(root, text="Start", command=self.start_webcam)
        self.start_button.pack(side=tk.RIGHT, pady=20)
       

        self.stop_button = ttk.Button(root, text="Stop", state=tk.DISABLED, command=self.stop_webcam)
        self.stop_button.pack(side=tk.RIGHT, pady=10)


        self.cap = None
        self.is_running = False
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose

        self.frame_count = 0
        self.file_value_list = []
        self.answer_list = []
        
        style = ttk.Style()
        style.configure("my.TLabel", font=("Helvetica", 25))
        self.pose_label = ttk.Label(root, text="", style="my.TLabel")
        self.pose_label.pack(padx=10, pady=20)



    def start_webcam(self):
        self.cap = cv2.VideoCapture(0)
        self.is_running = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.status_label.config(text="Webcam started. Press 'Stop' to stop.")
        self.update_frame()

    def update_frame(self):

        if self.is_running==True:
            ret, frame = self.cap.read()
            if not ret:
                return
            
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.draw_pose_landmarks(frame_rgb)
            frame = cv2.resize(frame_rgb, (320, 240), interpolation=cv2.INTER_AREA)
            
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame_rgb))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
            
            self.root.after(10, self.update_frame)

    def stop_webcam(self):
        self.is_running = False
        self.cap.release()
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        # self.capture_button.config(state=tk.DISABLED)
        self.status_label.config(text="Webcam stopped. Press 'Start' to begin.")

    def draw_pose_landmarks(self, frame):
        pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        results = pose.process(frame)
        
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)#畫骨骼
            
            
            if hasattr(results.pose_landmarks, "landmark"): 
                landmarks = []

                for value in results.pose_landmarks.landmark:
                    landmarks.append([value.x, value.y, value.z]) #將座標資料裝進landmark裡，所以這裡面會有99個點，每執行一次landmark就會清空
                landmarks = relative_landmark(landmarks)
                self.file_value_list.append(landmarks)#一次一次加landmark的值進來
                # print(self.file_value_list)
                
                # print(self.file_value_list)
            if self.frame_count % 10 == 0: #每10幀讀取一次
                file_value_list_recent = self.file_value_list[-30:] #取最後的30幀，作為丟進去lstm裡面的資料
                
                if len(file_value_list_recent) >= 30: #當file_value_list_recent大於30才開始預測，因為程式是從第0幀開始抓
                    pose_name = self.motion_analysis(file_value_list_recent) #調用學習好的model
                    self.answer_list.append(pose_name)
                    if len(self.answer_list)>10:
                        answer_list_rescent = self.answer_list[-20:]
                        counter = Counter(answer_list_rescent)
                        most_posename = counter.most_common(1)[0][0]
                        self.pose_label.config(text=f"Detected Pose: {most_posename}")
                        
            self.frame_count += 1
    #載入LSTM預測
    def motion_analysis(self, df_recent_keypoint):
        data_folder = 'csv_dataset_posename'
        dirs = sorted(os.listdir(data_folder))
        name_dict = {idx: value for idx, value in enumerate(dirs)}
        
        file_value_ndarray = np.array(df_recent_keypoint).reshape(1, 30, 99)
        
        prediction = LSTM_model.predict(file_value_ndarray, verbose=False)
        max_index = np.argmax(prediction)#這時的資料內容是1row的ndarray,裡面有偵測到的動作，是這13個動作分別可能的機率，而我要挑出最大的機率（的索引值）
        pose_name = name_dict[max_index] #該索引值，就是對應到name_dict的動作
        
        
        self.pose_label.config(text=f"Detected Pose: {pose_name}")
        
        return pose_name


    

if __name__ == "__main__":
    root = tk.Tk()
    app = FitnessApp(root)
    root.mainloop()
