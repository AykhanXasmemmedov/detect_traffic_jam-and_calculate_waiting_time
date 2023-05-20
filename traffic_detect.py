
import cv2
import numpy as np

import math
import random
import time
import timeit

import cardetect
from sort import *

def intersection_over_union(real,predict):
    x0=max(real[0],predict[0])
    y0=max(real[1],predict[1])
    x1=min(real[2],predict[2])
    y1=min(real[3],predict[3])
    
    interArea=max(0,x1-x0+1)*max(0,y1-y0+1)
    
    realArea=(real[2]-real[0]+1)*(real[3]-real[1]+1)
    predictArea=(predict[2]-predict[0]+1)*(predict[3]-predict[1]+1)
    
    iou=interArea/float(realArea+predictArea-interArea)
    
    return iou

colors=[]
for i in range(50):
    b = random.randint(0, 255)
    g = random.randint(0, 255)
    r = random.randint(0, 255)
    colors.append([b,g,r])
    
start_time=np.zeros(50)
finish_time=np.zeros(50)  
traffic_time=np.zeros(50)  

 
mot_tracker = Sort(max_age=200, min_hits=0.001, iou_threshold=.01)
local_tracks = []

video=cv2.VideoCapture('pexels-christopher-schultz.mp4')
### -----------MASKS---------------###

_,frame=video.read()
frame=cv2.resize(frame,(0,0),fx=.7,fy=.7)
image_copy=frame.copy()

mask_img=image_copy[655:740,250:1015,:]
points=np.array([[305,655],[980,655],[1015,740],[250,740]])

cv2.fillPoly(image_copy, pts=[points], color=(0, 255, 0))
mask_finish=image_copy[655:740,250:1015,:]


mask_img=image_copy[350:640,330:960,:]
points=np.array([[330,640],[960,640],[760,350],[555,350]])

cv2.fillPoly(image_copy,pts=[points],color=(0,0,255))
mask_start=image_copy[350:640,330:960,:]

###----------------------------------------------
count=0
counted_ID=[]
counted_ID_str=[]

prev_time = 0
new_time = 0
while video.isOpened():
    
    ret,frame=video.read()
    if not ret:
        break
    
    frame=cv2.resize(frame,(0,0),fx=.7,fy=.7)
    
    coordinates=cardetect.detect_car(frame,True) 
    
    if coordinates:
        track_bbs_ids=mot_tracker.update(np.asarray(coordinates))
        
        for track in reversed(track_bbs_ids):
            if local_tracks[int(track[4]) - 1:int(track[4])]:
    
                pr_x=local_tracks[int(track[4])-1][1][-1][0]
                pr_y=local_tracks[int(track[4])-1][1][-1][1]
                
                now_x=(track[0]+track[2])/2
                now_y=(track[1]+track[3])/2 
                d=math.sqrt((now_x - pr_x) ** 2 + (now_y - pr_y) ** 2)
                
                if d>300:
                    continue
                
                local_tracks[int(track[4]) - 1][1].append((now_x, now_y))
                
            else:
                local_tracks.append([int(track[4]) - 1, [((track[0] + track[2]) / 2, (track[1] + track[3]) / 2)]])
            
            
                
            color_id=int(int(track[4])%100)
            color=colors[color_id]
                
            cv2.rectangle(frame, (int(track[0]), int(track[1])), (int(track[2]), int(track[3])), color, 2)
    
            roi_start=intersection_over_union([330,350,960,640],[int(track[0]), int(track[1]), int(track[2]), int(track[3])])
            roi_start=int(roi_start*100)   
            
            roi_finish=intersection_over_union([250,655,1015,740],[int(track[0]), int(track[1]), int(track[2]), int(track[3])])
            roi_finish=int(roi_finish*100)   
            
            
            time_id=int(int(track[4])%100)
    
            if roi_start>0 and track[4] not in counted_ID_str:
                counted_ID_str.append(track[4])
                            
                start_time[time_id]=timeit.default_timer()
                
            if roi_finish>5 and track[4] not in counted_ID:
                count=count+1
                counted_ID.append(track[4])
    
            start=start_time[time_id]
            if start!=0:
                finish_time[time_id]=timeit.default_timer()-start
                traffic_time[time_id]=timeit.default_timer()-start
                    
            if  roi_finish>5:
                traffic_time[time_id]=0
            
            cv2.putText(frame, f'time:{str(int(finish_time[time_id]))}',(int(track[0]), int(track[1])),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
      
            
            
    cv2.putText(frame, f'Vehicle count:{str(count)}', (50, 100),cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
    
    traffic_mean=np.mean(traffic_time)
    traffic=False
    
    if traffic_mean>1.:
        traffic=True
        cv2.putText(frame,f'Traffic:{str(traffic)}',(50,150),cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,0,255),2)
        cv2.putText(frame,f'Traffic:{str(float(traffic_mean))}',(50,200),cv2.FONT_HERSHEY_SIMPLEX,1.2,(0,0,255),2)
        
    else:
        cv2.putText(frame,f'Traffic:{str(traffic)}',(50,150),cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,255,0),2)
        cv2.putText(frame,f'Traffic:{str(float(traffic_mean))}',(50,200),cv2.FONT_HERSHEY_SIMPLEX,1.2,(0,255,0),2)
       
    

    coupled_img=cv2.addWeighted(frame[655:740,250:1015,:],.7,mask_finish,.3,0)
    frame[655:740,250:1015,:]=coupled_img

    coupled_img=cv2.addWeighted(frame[350:640,330:960,:],.7,mask_start,.3,0)
    frame[350:640,330:960,:]=coupled_img


    new_time = time.time()
    fps = int(1/(new_time-prev_time))    
    cv2.putText(frame, f'FPS:{str(fps)}', (50, 50),cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 2)
    prev_time = new_time
    
    cv2.imshow('frame',frame)
    key=cv2.waitKey(7)
    
    if key==ord('q'):
        break

cv2.destroyAllWindows()
video.release()


