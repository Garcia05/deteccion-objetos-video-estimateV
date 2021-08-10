from __future__ import division
from models import *
from utils.utils import *
from utils.datasets import *
import os
import sys
import argparse
import cv2
from PIL import Image
import torch
from torch.autograd import Variable
from sort import *

tracker = Sort()
memory = {}
time_id={}
velocidad_d ={}
time_test = {}
time_for_speed = []
dict_id_speed = {}
def intersect(A,B,C,D):
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def ccw(A,B,C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
def Convertir_RGB(img):
    # Convertir Blue, green, red a Red, green, blue
    b = img[:, :, 0].copy()
    g = img[:, :, 1].copy()
    r = img[:, :, 2].copy()
    img[:, :, 0] = r
    img[:, :, 1] = g
    img[:, :, 2] = b
    return img

def Convertir_BGR(img):
    # Convertir red, blue, green a Blue, green, red
    r = img[:, :, 0].copy()
    g = img[:, :, 1].copy()
    b = img[:, :, 2].copy()
    img[:, :, 0] = b
    img[:, :, 1] = g
    img[:, :, 2] = r
    return img



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="data/samples", help="path to dataset")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
    parser.add_argument("--webcam", type=int, default=1,  help="Is the video processed video? 1 = Yes, 0 == no" )
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--directorio_video", type=str, help="Directorio al video")
    parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
    opt = parser.parse_args()



    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = Darknet(opt.model_def, img_size=opt.img_size).to(device)


    if opt.weights_path.endswith(".weights"):
        model.load_darknet_weights(opt.weights_path)
    else:
        model.load_state_dict(torch.load(opt.weights_path))

    model.eval()  
    classes = load_classes(opt.class_path)
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    if opt.webcam==1:
        cap = cv2.VideoCapture(0)
        out = cv2.VideoWriter('output.mp4',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (1280,960))
    else:
        cap = cv2.VideoCapture(opt.directorio_video)
        # frame_width = int(cap.get(3))
        # frame_height = int(cap.get(4))
        out = cv2.VideoWriter('outp2.mp4',cv2.VideoWriter_fourcc(*'MP4V'), 10, (1280,960))
    colors = np.random.randint(0, 255, size=(len(classes), 3), dtype="uint8")
    a=[]
    (W, H) = (None, None)
    lx1 , ly1 ,lx2 , ly2  = 50,480,620,480
    l2x1 , l2y1 ,l2x2 , l2y2 = 660,850,1250,850
    line_speed_start = [(lx1,ly1), (lx2,ly2)]
    line_speed_end = [(l2x1,l2y1), (l2x2, l2y2)]
    while cap:
        ret, frame = cap.read()
        time_start = np.round(time.time(),3)
        if ret is False:
            break
        frame = cv2.resize(frame, (1280, 960), interpolation=cv2.INTER_CUBIC)
        #LA imagen viene en Blue, Green, Red y la convertimos a RGB que es la entrada que requiere el modelo
        RGBimg=Convertir_RGB(frame)
        imgTensor = transforms.ToTensor()(RGBimg)
        imgTensor, _ = pad_to_square(imgTensor, 0)
        imgTensor = resize(imgTensor, 416)
        imgTensor = imgTensor.unsqueeze(0)
        imgTensor = Variable(imgTensor.type(Tensor))

        start = 0.0
        end = 0.0
        if W is None or H is None:
            (H, W) = frame.shape[:2]
        with torch.no_grad():
            detections = model(imgTensor)
            start = time.time()
            detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)
            end = time.time()
        xx1,yy1 = 0,0
        aa = 0
        boxes = []
        center = []
        confidences = []
        classIDs = []
        for detection in detections:
            if detection is not None:
                detection = rescale_boxes(detection, opt.img_size, RGBimg.shape[:2])
                for x1, y1, x2, y2, conf, cls_conf, cls_pred in detection:
                    scores= [cls_conf,cls_pred]
                    classID = np.argmax(scores)
                    confidence = scores[classID]
                    if(confidence > opt.conf_thres):
                      width = x2 - x1
                      height = y2 - y1
                      centerX = width/2 + x1
                      centerY = height/2 + y1
                      #if(lx2 >= centerX and centerY >= ly2 or l2x1 <= centerX and centerY <= l2y2 and centerY >= ly2):
                      x = int(centerX - (width / 2))
                      y = int(centerY - (height / 2))
                      center.append(int(centerY))
                      boxes.append([x,y,int(width),int(height)])
                        #boxes.append([int(x1),int(y1+height),int(x2),int(y1)])
                      confidences.append(float(confidence))
                      classIDs.append(classID)

        
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, opt.conf_thres, opt.nms_thres)
        #print(idxs)
        dets = []
        if len(idxs) > 0:
            # loop over the indexes we are keeping
            for i in idxs.flatten():
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                dets.append([x, y, x+w, y+h, confidences[i]])
                #print(confidences[i])
                #print(center[i])
        np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
        dets = np.asarray(dets)
        tracks = tracker.update(dets)
        boxes = []
        indexIDs = []
        c = []
        
        previous = memory.copy()
        print(previous)
        memory = {}
        for track in tracks:
            boxes.append([track[0], track[1], track[2], track[3]])
            indexIDs.append(int(track[4]))

            memory[indexIDs[-1]] = boxes[-1]
        if(len(boxes)>0):
          i = int(0)
          for box in boxes:
            (x, y) = (int(box[0]), int(box[1]))
            (w, h) = (int(box[2]), int(box[3]))
            color = [int(c) for c in colors[indexIDs[i]*100 %80]]
            #color = [int(c) for c in COLORS[classIDs[i]]]
            #color = (255,0,0) if ct1==1 else (0,255,0) if ct2==1 else (255,0,255) if ct3==1 else (0,255,255) if ct4==1 else (0,0,255)
            
            cv2.rectangle(frame, (x, y), (w, h), color, 4)
            print(indexIDs)
            if indexIDs[i] in previous:
              previous_box = previous[indexIDs[i]]
              (x2, y2) = (int(previous_box[0]), int(previous_box[1]))
              (w2, h2) = (int(previous_box[2]), int(previous_box[3]))
              p0 = (int(x + (w-x)/2), int(y + (h-y)/2))
              p1 = (int(x2 + (w2-x2)/2), int(y2 + (h2-y2)/2))
              cv2.line(frame, p0, p1, color, 3)

              id = indexIDs[i]
            
              speed = 0
              
              speed = abs(y-y2)/0.2
              print("velocidad",speed)
              
             #speed = speed = abs(y-y2)/
              cv2.putText(frame, str(len(previous)), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 5)
            i += 1
            print(time_id)
        #cv2.line(frame,(lx1,ly1),(lx2,ly2),(255,0,0),5)
        #cv2.line(frame,(l2x1,l2y1),(l2x2,l2y2),(0,255,0),5)   
        '''
        for detection in detections:
            if detection is not None:
                detection = rescale_boxes(detection, opt.img_size, RGBimg.shape[:2])
                for x1, y1, x2, y2, conf, cls_conf, cls_pred in detection:
                    box_w = x2 - x1
                    box_h = y2 - y1
                    xx1 = box_w/2 + x1
                    print(cls_pred)
                    yy1 = box_h/2 + y1
                    if(lx2 > xx1 and yy1 > ly2):
                      aa = abs(yy1 - ly2)
                      color = [int(c) for c in colors[int(cls_pred)]]
                      
                      #print("Se detectó {} en X1: {}, Y1: {}, X2: {}, Y2: {}".format(classes[int(cls_pred)], x1, y1, x2, y2))
                      cv2.line(frame,(lx1,ly1),(lx2,ly2),(255,0,0),5)
                      cv2.line(frame,(l2x1,l2y1),(l2x2,l2y2),(0,255,0),5)
                      frame = cv2.rectangle(frame, (x1, y1 + box_h), (x2, y1), color, 3)
                      
                      cv2.putText(frame, str("%.2f" % float(aa)), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 5)# Nombre de la clase detectada
                      cv2.putText(frame, str("%.2f" % float(conf)), (x2, y2 - box_h), cv2.FONT_HERSHEY_SIMPLEX, 0.5,color, 5) # Certeza de prediccion de la clase
                    elif(l2x1 < xx1 and yy1 < l2y2 and yy1 > ly2):
                      aa = abs(yy1 - l2y2)
                      color = [int(c) for c in colors[int(cls_pred)]]
                      #print("Se detectó {} en X1: {}, Y1: {}, X2: {}, Y2: {}".format(classes[int(cls_pred)], x1, y1, x2, y2))
                      cv2.line(frame,(lx1,ly1),(lx2,ly2),(255,0,0),5)
                      cv2.line(frame,(l2x1,l2y1),(l2x2,l2y2),(0,255,0),5)
                      frame = cv2.rectangle(frame, (x1, y1 + box_h), (x2, y1), color, 3)
                      
                      cv2.putText(frame, str("%.2f" % float(aa)), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 5)# Nombre de la clase detectada
                      cv2.putText(frame, str("%.2f" % float(conf)), (x2, y2 - box_h), cv2.FONT_HERSHEY_SIMPLEX, 0.5,color, 5) # Certeza de prediccion de la clase
                    else :
                      cv2.line(frame,(lx1,ly1),(lx2,ly2),(255,0,0),5)
                      cv2.line(frame,(l2x1,l2y1),(l2x2,l2y2),(0,255,0),5)
              '''
        #Convertimos de vuelta a BGR para que cv2 pueda desplegarlo en los colores correctos
        
        if opt.webcam==1:
            cv2.imshow('frame', Convertir_BGR(RGBimg))
            out.write(RGBimg)
        else:
            out.write(Convertir_BGR(RGBimg))
            #cv2.imshow('frame', RGBimg)
        #cv2.waitKey(0)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    out.release()
    cap.release()
    #cv2.destroyAllWindows()
