# https://gist.github.com/kittinan/e7ecefddda5616eab2765fdb2affed1b

import socket
import sys
import cv2
import pickle
import numpy as np
import struct ## new
import zlib
import traceback

#----------------------------------------------#

model_num_blocks = pickle.load(open("model_num_blocks.pkl", "rb"))
model_offset = pickle.load(open("model_offset.pkl", "rb"))

def predict_location (img_x, img_y, img_width, img_height):

    try:
        x = np.array([img_x, img_y, img_width, img_height])
        x = x.reshape(1, -1)

        num_blocks =  model_num_blocks.predict(x)[0]
        offset =  model_offset.predict(x)[0]

        return (num_blocks, offset)
    except:

        return None, None

from bounding_box import bounding_box as bb

def predict(image):
    
    # initialize a list of colors to represent each possible class label
    np.random.seed(15)
    COLORS = ["blue", "yellow", "red", "green"]
    (H, W) = image.shape[:2]
    
    # determine only the "ouput" layers name which we need from YOLO
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    
    # construct a blob from the input image and then perform a forward pass of the YOLO object detector, 
    # giving us our bounding boxes and associated probabilities
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=False, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)
    
    boxes = []
    confidences = []
    classIDs = []
    threshold = 0.3
    
    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability) of
            # the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            # confidence type=float, default=0.5
            if confidence > threshold:
                # scale the bounding box coordinates back relative to the
                # size of the image, keeping in mind that YOLO actually
                # returns the center (x, y)-coordinates of the bounding
                # box followed by the boxes' width and height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates, confidences,
                # and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # apply non-maxima suppression to suppress weak, overlapping bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, threshold, 0.3)

    return_data = []

    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            # draw a bounding box rectangle and label on the image
            color = str(np.random.choice(COLORS, 1)[0])
            text = "{}".format(LABELS[classIDs[i]], confidences[i])
            bb.add(image,x,y,x+w,y+h,text,color)
            
            return_data.append((image, classIDs[i], LABELS[classIDs[i]], confidences[i], x, y, w, h))
        

    if not return_data:
        print ("[LOG]\t\tNO IMAGE DETECTED")
    return return_data

LABELS = open("./classes.names.txt").read().strip().split("\n")

# derive the paths to the YOLO weights and model configuration
weightsPath = "new.final.weights"
configPath = "yolov4_custom_test.cfg"

# Loading the neural network framework Darknet (YOLO was created based on this framework)
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

IMAGE_COUNT = 0

#----------------------------------------------#
HISTORY = {}

def get_image_location(r_x, r_y, r_d, dist, offset):

    if r_d==0:
        i_x = r_x - offset
        i_y = r_y + (dist + 2)
    elif r_d==1:
        i_x = r_x + (dist + 2)
        i_y = r_y + offset
    elif r_d==2:
        i_x = r_x + offset
        i_y = r_y - (dist + 2)
    elif r_d==3:
        i_x = r_x - (dist + 2)
        i_y = r_y - offset

    return int(i_x), int(i_y)

#----------------------------------------------#

# CODE TO COMMUNICATE WITH RPI AS SERVER

HOST_SEND = '192.168.5.5'
PORT_SEND = 8485

s_send = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
s_send.connect ((HOST_SEND, PORT_SEND))
print('Socket now listening as a client')

while True:

    try:
        message_size = struct.calcsize("100s")
        input_message = s_send.recv(1000)
        input_message = struct.unpack("100s", input_message)[0]

        input_message = str(input_message).split("CAPTURE:")[1]
        # print (input_message)
        input_message = input_message[:input_message.find("r")-1]
        print (input_message)
        robot_y, robot_x, robot_direction = [int(i) for i in input_message.split(":")]
    except:

    
    # except:
        
    # print ("[ERROR] Robot location not received")
    # traceback.print_exc()
    # robot_x, robot_y, robot_direction = 5, 5 , 3
    # continue

    

    #----------------#

    #-#
    # message_size = struct.calcsize("1000s")
    # input_message = s_send.recv(4096)
    # input_message = struct.unpack("1000s", input_message)[0]

    # input_message = str(input_message).split("CAPTURE:")[1]
    # # print (input_message)
    # input_message = input_message[:input_message.find("r")-1]
    # print (input_message)
    # robot_y, robot_x, robot_direction = [int(i) for i in input_message.split(":")]
    #-#

    data = b""
    payload_size = struct.calcsize(">L")
    print("[LOG]\t\tpayload_size: {}".format(payload_size))

    while len(data) < payload_size:
        # print("Recv: {}".format(len(data)))
        data += s_send.recv(4096)

    # print("Done Recv: {}".format(len(data)))
    packed_msg_size = data[:payload_size]
    data = data[payload_size:]
    msg_size = struct.unpack(">L", packed_msg_size)[0]
    print("[LOG]\t\tmsg_size: {}".format(msg_size))
    while len(data) < msg_size:
        data += s_send.recv(4096)
    frame_data = data[:msg_size]
    data = data[msg_size:]

    frame= pickle.loads(frame_data, fix_imports=True, encoding="bytes")
    frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)

    #------------------------#

    frame = cv2.cvtColor(frame ,cv2.COLOR_BGR2RGB) # to predict , as model uses RGB

    preds = predict(frame)

    for image, label_code, label, conf, x, y, w, h in preds:
        
        if conf < 0.85:
            print (f"[SKIPPED] {label} not confident enough")
            continue
        
        num_blocks, offset = predict_location(x, y, w, h)

        print ("[PREDICTION]", end = "\t")
        print (f"{label}\t{conf}\t{x}\t{y}\t{w}\t{h}")
        print (f"[LOCATION]\t{num_blocks}, {offset}", end = "\n")

        i_x, i_y = get_image_location(robot_x, robot_y, robot_direction, num_blocks, offset)

        print (f"[IMG-LOC]\t{i_x}, {i_y}, {robot_direction}") # robot_direction = image_direction
        print()

        # pred_message = f"PA|IMAGE:{0}:{robot_x}:{robot_y}:{robot_direction}"
        pred_message = f"A|IMAGE:{label_code}:{i_y}:{i_x}:{robot_direction}"
        s_send.sendall(pred_message.encode('utf-8'))


    frame = cv2.cvtColor(frame ,cv2.COLOR_RGB2BGR) # to save , as CV2 uses BGR

    #------------------------#

    cv2.imwrite(f"saved_images/image_{IMAGE_COUNT+1}.jpg", frame)
    IMAGE_COUNT+=1
    
    cv2.imshow('ImageWindow',frame)
    cv2.waitKey(1)

    # except :
    #     print ("Error encountered")
    #     continue 