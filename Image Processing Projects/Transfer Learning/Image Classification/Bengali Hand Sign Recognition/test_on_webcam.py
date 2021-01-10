import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys, os
import cv2
from tensorflow import keras


model = keras.models.load_model("vgg19_200_200_3_weights_h5.h5")

import urllib.request
import cv2
import numpy as np
import time
URL = "http://192.168.0.101:8080"

while True:
    
    img_arr = np.array(bytearray(urllib.request.urlopen(URL+"/shot.jpg").read()),dtype=np.uint8)
    img = cv2.imdecode(img_arr,-1)
    
    result = model.predict( np.array( cv2.resize( img, (200,200) ) ).reshape( (1, 200, 200, 3) ) )
    val = np.argmax( result, axis=-1 )[0]
    prob = np.argmax(result, axis=0)[0]
    #print(val)
        
    # font 
    font = cv2.FONT_HERSHEY_SIMPLEX 
      
    # org 
    org = (50, 50) 
    # fontScale 
    fontScale = 1
    # Blue color in BGR 
    color = (255, 0, 0) 
    # Line thickness of 2 px 
    thickness = 2   
    # Using cv2.putText() method 
    img = cv2.putText(img, "predicted label: "+str(val), org, font, fontScale, color, thickness, cv2.LINE_AA) 
    
    
    
    cv2.imshow('IPWebcam',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        plt.imshow(cv2.resize( img, (200,200) ))
        break
    
cv2.destroyAllWindows()