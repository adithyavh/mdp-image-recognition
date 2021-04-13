import socket
import threading

#---------<CONFIG>-----------#
HEADER = 64 # number of bytes in header - will be used to convey how many bytes will be sent later
PORT = 5050
SERVER = "192.168.5.11"#"localhost"#192.168.5.11" # replace with LAN IPv4 address
ADDR = (SERVER , PORT)
FORMAT = 'utf-8'
DISCONNECT_MESSAGE = "!DISCONNECT"
#---------</CONFIG>-----------#

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  
server.bind(ADDR)

#--------------------------------------------------------------------#

import numpy as np
import cv2
import matplotlib.pyplot as plt
from bounding_box import bounding_box as bb
import time

import base64
import traceback

#--------------------------------------------------------------------#

LABELS = open("./classes.names.txt").read().strip().split("\n")

# derive the paths to the YOLO weights and model configuration
weightsPath = "yolov4_custom_train_new1.weights"
configPath = "yolov4_custom_test.cfg"

# Loading the neural network framework Darknet (YOLO was created based on this framework)
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

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

    print (idxs)

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
            #cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            #cv2.putText(image, text, (x +15, y - 10), cv2.FONT_HERSHEY_SIMPLEX,1, color, 2)

        return image, LABELS[classIDs[i]], confidences[i]

    return image, None, None

#--------------------------------------------------------------------#

def receive_image(conn, addr):
    print (f"[NEW CONNECTION] {addr} connected.")
    connected = True

    while connected:

        try:
            msg_length = conn.recv(HEADER)#.decode(FORMAT) # Blocking code (wait until run)
            if msg_length:
                print (msg_length)
                print (len(msg_length))
                # msg_length = base64.b64decode(msg_length)
                msg_length = int.from_bytes(msg_length, 'big')

                # msg_length = int(msg_length)
                print(msg_length)

                # time.sleep(20)

                if msg_length == -1:
                    connected = False
                    break
                
                msg = conn.recv(msg_length)# Blocking code (wait until run)
                
                image_bytes = base64.b64decode(msg)

                image_bytes = np.fromstring(image_bytes, np.uint8)
                image = np.resize(image_bytes, (120, 180, 3))
                # cv2.imwrite("./saved_image.jpg", image)

                image_processed , label, conf =  predict(image)
                cv2.imwrite("./image_processed.jpg", image_processed)
                print ("IMAGE PROCESSED-------")
                print (label, conf)
                
                print ("[IMAGE SUCCESS]")
                if label and conf:
                    conn.send("SUCCESS".encode(FORMAT))

            else:
                continue

        except Exception as e:

            print ("[ERROR] Image reception failed")
            print (e)
            traceback.print_exc()

            continue
    
    print ("[CLOSED] Connection Closed")
    conn.close()

def start():

    server.listen()
    print (f"[LISTENING] Server is Listening on {SERVER}")

    while True:
        conn, addr = server.accept() # Blocking code (wait until run)
        thread = threading.Thread(target=receive_image, args = (conn, addr))
        thread.start()
        print(f"[ACTIVE CONNECTIONS] {threading.activeCount()-1}")

print ("[STARTING] Server started... ")
start()


