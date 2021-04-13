import numpy as np
import cv2
import base64

def read_image_bytes(path):

    image = cv2.imread(path)
    image = cv2.resize(image, (180, 120), interpolation = cv2.INTER_LANCZOS4)
    print (image.shape)

    # image = image.flatten().tostring()
    image = base64.b64encode(image)
    
    return image 

# def display_bytes(image_bytes):

#     image_bytes = np.fromstring(image_bytes, np.uint8)
#     image = np.resize(image_bytes, (120, 180, 3))
#     cv2.imwrite("./saved_image_1.jpg", image)
#     plt.imshow(image)
#     plt.show()

# image_bytes = read_image_bytes("./picture2.jpg")
# display_bytes(image_bytes) 

#--------------------------------------------------------------------#

import socket
import time

#---------<CONFIG>-----------#
HEADER = 64 # number of bytes in header - will be used to convey how many bytes will be sent later
PORT = 5050
SERVER = "192.168.5.11" # replace with LAN IPv4 address
ADDR = (SERVER , PORT)
FORMAT = 'utf-8'
DISCONNECT_MESSAGE = "!DISCONNECT"
#---------</CONFIG>-----------#

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  
client.connect(ADDR)

def send_image(image_bytes = None, disconnect = False):

    msg_length = len(image_bytes)
    print(msg_length)

    send_length = msg_length.to_bytes(length = HEADER, byteorder= 'big')
    # send_length = str(msg_length).encode(FORMAT)
    # send_length += b' '*(HEADER-len(send_length))
    
    client.send(send_length)
    
    client.send(image_bytes)
    print (client.recv(2048).decode(FORMAT))

image_bytes = read_image_bytes("./up2.jpg")
send_image(image_bytes)
# send_image(disconnect= True)