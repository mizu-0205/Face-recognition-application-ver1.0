import streamlit as st
import torch
import cv2

st.title('Face recognition application ver1.0')

@st.cache_data()
def load_model():
    return torch.hub.load('ultralytics/yolov5', 'yolov5s')

model = load_model()


widget_id = (id for id in range(1, 10000))

if st.button('Run Yolov5'):
    camera = cv2.VideoCapture(0)

    while True:
        ret, frame = camera.read()
        results = model(frame)
        results.render()
        st.image(frame, channels="BGR")
        
        
        if st.button('Predicted', key=next(widget_id)):
            
            pass

        
        if st.button('Stop', key=next(widget_id)):
            break 

    camera.release()
    cv2.destroyAllWindows()
