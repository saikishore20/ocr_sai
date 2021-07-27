import streamlit as st
import cv2
import pytesseract
from PIL import Image
import numpy as np
pytesseract.pytesseract.tesseract_cmd = '/app/.apt/usr/bin/tesseract'
st.title("Number Plate Detection And Recognition")
pic = st.sidebar.file_uploader("Choose an image")
if pic is not None:
  im = Image.open(pic)
  img = np.array(im)
  st.image(img,caption='Uploaded Image')
  if st.button('DETECT'):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)          # Gray scale conversion
    model = cv2.CascadeClassifier('/content/haarcascade_russian_plate_number.xml')
    demo = model.detectMultiScale(gray,1.1,10)
    for(x,y,w,h) in demo:
      g_roi = gray[y:y+h,x:x+w]
      roi = img[y:y+h,x:x+w]
    st.image(roi , caption = 'Number plate Detected')                     # Range of interest
    boxes = pytesseract.image_to_boxes(g_roi)             # Boxes
    #ha,wa,da =  roi.shape
    for e in boxes.splitlines():
       e=e.split()
       a,b,c,d = int(e[1]),int(e[2]),int(e[3]),int(e[4])
       rec = cv2.rectangle(roi,(a,b),(c,d),[250,0,0],2)
    st.image(rec,caption = 'Image Boxes')                        # display Boxes            
    output = pytesseract.image_to_string(g_roi)
    st.write(f"Your Car number is {output}")             # Display string
