import pandas as pd
import numpy as np
import cv2
import matplotlib_inline as plt
import os
import mediapipe as mp
import copy
import itertools
import csv

def get_filename_number(s):#得到mp4檔名稱裡面的數字，這樣才能做排序
    return int(s.split('_')[-1].split('.')[0])
def list_of_directory_file(project_file_path):
    video_path = []
    dir_files = os.listdir(project_file_path)
    sorted_dir = sorted(dir_files)
    for dirs in sorted_dir:
        file_path = f'./dataset_workout/{dirs}/'
        files =os.listdir(file_path)
        sorted_files = sorted(files,key=get_filename_number)
        for items in sorted_files:
            file_video_path = f'{file_path}{items}'
            video_path.append(file_video_path)
    return video_path


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
    
    for idx , relative_point in enumerate(temp_landmark_list):
        temp_landmark_list[idx][0] = temp_landmark_list[idx][0] / shoulder_length_x
        temp_landmark_list[idx][1] = temp_landmark_list[idx][1] / shoulder_length_x
        temp_landmark_list[idx][2] = temp_landmark_list[idx][2] / shoulder_length_x

    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
    # print(temp_landmark_list)
    return temp_landmark_list


def get_mp4_poselandmark_xyz(video_path):
    mp_pose = mp.solutions.pose
    with mp_pose.Pose(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as pose:
        
        cap = cv2.VideoCapture()
        cap.open(video_path)
        relative_landmark_list_total = []
        # j=1
        while cap.isOpened():
            ret, frame = cap.read()
            if ret == True:
                
                # if j==2:
                #     break
            #對每一幀frame做操作
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(frame_rgb)
                post_landmark_list =[]
                if hasattr(results.pose_landmarks, "landmark"):
                    for value in results.pose_landmarks.landmark:
                        temp_value = [value.x,value.y,value.z]
                        post_landmark_list.append(temp_value)
                    relative_landmark_list = relative_landmark(post_landmark_list) #作正規化    
                    relative_landmark_list_total.append(relative_landmark_list)
            else:
                break       
        
            # j=j+1
            # cap.release()
            # cv2.destroyAllWindows()
    return relative_landmark_list_total #正規劃後的list

#產生csv檔

def vedio_to_csv(project_file_path,file_value_list):
    for idx,video_path in enumerate(project_file_path):
        csv_result = get_mp4_poselandmark_xyz(video_path)
        
        csv_filename = f'output_{idx+1}.csv'
        with open(csv_filename,"w") as csvfile:
            csvwriter = csv.writer(csvfile)
            file_names = []
            for i in range(1,34):
                file_names.extend([f"x{i}",f"y{i}",f"z{i}"])
            csvwriter.writerow(file_names)

            for file_value in csv_result:
                csvwriter.writerow(file_value)
        
        # if idx ==1:
        #     break
    return csvfile
        