from ultralytics import YOLO
from paddleocr import PaddleOCR
import cv2
import sys
import pyshine as ps
import numpy as np
ocr = PaddleOCR(lang='en',show_log = False)
counter = 0
circles = np.zeros((4,2) , np.int_)


def mousePoints(event,x,y,flags,params):
     global counter
     if event == cv2.EVENT_LBUTTONDOWN:
            circles[counter] = x,y
            counter = counter+1

def paddleocr_predict(plate):
    results = ocr.ocr(plate,cls=False, det=False)
    ocr_result = results[0][0]
    return str(ocr_result[0])
    

def get_plates(result, img):
    images = [] 
    boxes = result[0].boxes
    img = img.copy()
    for b in boxes:
        x1 = int(b.xyxy[0][0])
        y1 = int(b.xyxy[0][1])
        x2 = int(b.xyxy[0][2])
        y2 = int(b.xyxy[0][3])
        images.append(img[y1:y2, x1:x2].copy())
    return images

# OCR
def get_LP_number(result, img):
    plates = get_plates(result, img)
    plate_numbers = [] 
    if plates is not None:
         for plate in plates:
             number = paddleocr_predict(plate)
             plate_numbers.append(number)
    return plate_numbers


def draw_box(result, img):
    boxes = result[0].boxes 
    plate_numbers = get_LP_number(result, img) 
    

    for b,pnum in zip(boxes,plate_numbers): 
        x1 = int(b.xyxy[0][0])
        y1 = int(b.xyxy[0][1]) - 20 
        x2 = int(b.xyxy[0][2])
        y2 = int(b.xyxy[0][3])
        cx = int((x1+ x2) / 2.0)
        cy = int((y1+y2)/2.0)
        cnt = (cx,cy)
        
        
        result =  cv2.pointPolygonTest(circles , (cx ,cy),False)
        if result > 0:
           cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
           text  =  'License Number: '+ pnum
           ps.putBText(img,text,text_offset_x=20,text_offset_y=20,vspace=10,hspace=10,font_scale=2.0,background_RGB=(228,225,222),text_RGB=(255,0,0))
         
    return img
 
def video_draw_box(vid_path, model):
    cap = cv2.VideoCapture(vid_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        if counter == 4:
            cv2.polylines(frame, pts=[circles],  isClosed = True , color=(0, 255, 0),thickness = 2)
        result = model(frame,verbose=False) 
        frame = draw_box(result, frame) 
        for x in range(4):
             cv2.circle(frame,(circles[x][0],circles[x][1]),15,(0,255,0),cv2.FILLED)
        cv2.imshow('frame', frame)
        cv2.setMouseCallback('frame' , mousePoints)
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if len(sys.argv) == 3:
    # Get weights and img_dir
    pre_trained_model = "best.pt"
    
    media_type = sys.argv[1]
    
    file_dir = sys.argv[2]
    
    # Create model
    model = YOLO(pre_trained_model)
   
    if media_type == "-image":
        img = cv2.imread(file_dir)
        result = model(img)
        img = draw_box(result, img)
        cv2.imshow("Result", img)
        cv2.waitKey(0)
        cv2.imwrite("predicted-" + file_dir, img)
        
    elif media_type == "-video":
        video_draw_box(file_dir, model)
    
