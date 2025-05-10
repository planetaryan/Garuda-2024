from ultralytics import YOLO
import cv2
import os
import math
import time
import numpy as np

# Running real time from webcam
cap = cv2.VideoCapture("/home/jetson/Desktop/jetson/jetson/output_pyramid_fast.mp4")
model = YOLO('/home/jetson/Desktop/jetson/jetson/best.engine',task='pose')
path="/home/jetson/Desktop/jetson/jetson/out/"

output_video_file = 'output_video2x.mp4'



# Initialize video writer parameters
frame_width = 640  # Specify the frame width
frame_height = 480  # Specify the frame height
fps = 30.0  # Specify the desired FPS for the output video

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
out = cv2.VideoWriter(output_video_file, fourcc, fps, (frame_width, frame_height))
if not out.isOpened():
    print(f"Error: Could not create output video file '{output_video_file}'")
    exit()
# Reading the classes
classnames = ['Pyramid']
count=0
prev=0
now=0
c=[None,None]
key_coords=[None,None,None,None,None]
while True:
    
    ret,frame = cap.read()
    ret,frame = cap.read()
    if not ret:
        break
    original=frame.copy()
    frame = cv2.resize(frame,(640,480))
    #frame=cv2.convertScaleAbs(frame,alpha=1.17,beta=-40)
    prev=time.time()
    results = model.track(frame,persist=True,tracker="bytetrack.yaml", device='0')
    now=time.time()
    fps=str(2*(int(1/(now-prev))))
    # annotated_frame = results[0].plot()
    # cv2.putText(annotated_frame, fps, (7, 40),cv2.FONT_HERSHEY_SIMPLEX , 1, (100, 255, 0), 3, cv2.LINE_AA) 
    # cv2.imshow('frame',annotated_frame)

    # for keypoints in results:
    
    # Getting bbox,confidence and class names informations to work with
    for result in results:
        keys=result.keypoints.xy.cpu().numpy()
        boxes=result.boxes
        
        for index,key in enumerate(keys[0]):
            key_coords[index]=[int(key[0]),int(key[1])]
        
        for box in boxes:
            x1,y1,x2,y2 = box.xyxy[0].cpu().numpy()
            print(x1,y1,x2,y2)
            for i in range(5):
                if(key_coords[i]):
                    if key_coords[i][0]>x2+10 or key_coords[i][0]<x1-10 or key_coords[i][1]>y2+10 or key_coords[i][1]<y1-10:
                        key_coords[i]=None
                
                    
   
        if(key_coords[0]):
            
        # if(key_coords[0] and key_coords[1]):
            # cv2.line(frame,key_coords[0],key_coords[1],(0,255,0),2)
            if(key_coords[1] and key_coords[2]):
                cv2.fillPoly(frame,[np.array([key_coords[0],key_coords[1],key_coords[2]])],(17, 17, 115))
                c[0]=(key_coords[0][0]+key_coords[1][0]+key_coords[2][0])//3
                c[1]=(key_coords[0][1]+key_coords[1][1]+key_coords[2][1])//3
                cv2.putText(frame, "face1", (c[0], c[1]+20),cv2.FONT_HERSHEY_SIMPLEX,0.3,(255, 255, 255),1)

            if(key_coords[2] and key_coords[3]):
                cv2.fillPoly(frame,[np.array([key_coords[0],key_coords[2],key_coords[3]])],(29, 140, 140))
                c[0]=(key_coords[0][0]+key_coords[2][0]+key_coords[3][0])//3
                c[1]=(key_coords[0][1]+key_coords[2][1]+key_coords[3][1])//3
                cv2.putText(frame, "face2", (c[0]+20, c[1]),cv2.FONT_HERSHEY_SIMPLEX,0.3,(255, 255, 255),1)

            if(key_coords[3] and key_coords[4]):
                cv2.fillPoly(frame,[np.array([key_coords[0],key_coords[3],key_coords[4]])],(58, 110, 21))
                c[0]=(key_coords[0][0]+key_coords[3][0]+key_coords[4][0])//3
                c[1]=(key_coords[0][1]+key_coords[3][1]+key_coords[4][1])//3
                cv2.putText(frame, "face3", (c[0], c[1]-20),cv2.FONT_HERSHEY_SIMPLEX,0.3,(255, 255, 255),1)

            if(key_coords[1] and key_coords[4]):
                cv2.fillPoly(frame,[np.array([key_coords[0],key_coords[4],key_coords[1]])],(156, 85, 23))
                c[0]=(key_coords[0][0]+key_coords[4][0]+key_coords[1][0])//3
                c[1]=(key_coords[0][1]+key_coords[4][1]+key_coords[1][1])//3
                cv2.putText(frame, "face4", (c[0]-30, c[1]),cv2.FONT_HERSHEY_SIMPLEX,0.3,(255, 255, 255),1)
        
        key_coords=[None,None,None,None,None]
            
        for box in boxes:
            confidence = box.conf[0]
            confidence = math.ceil(confidence * 100)
            Class = int(box.cls[0])
            if confidence > 48:
                confidence=str(confidence)
                x1,y1,x2,y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1),int(y1),int(x2),int(y2)
                # cv2.rectangle(frame,(x1-5,y1-5),(x2+5,y2+5),(0,0,255),2)
                cv2.putText(frame, (classnames[Class]+":"+confidence+"%"), (x1, y1 - 30),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0, 255, 0),2)
                cv2.putText(frame, fps, (7, 40),cv2.FONT_HERSHEY_SIMPLEX , 1, (100, 255, 0), 3, cv2.LINE_AA)
                #res=np.hstack((original,frame)) 
                #if int(confidence) > 80:
            count+=1
            cv2.imwrite(path+str(count)+".jpg",frame)
            cv2.imshow('frame',frame)
            out.write(frame)
            print("YES")


out.release()
    # #vid_out.write(frame) 
