#!/usr/bin/env python
# coding: utf-8

# In[7]:

import io
import os
from numpy.lib.type_check import imag
import yaml
import json
import requests

import numpy as np
from PIL import Image
import cv2

import streamlit as st

st.set_page_config(layout="wide")

root_password = "passion-ate"


def main():
    st.title("PIE")
    
    uploaded_file = st.file_uploader(
        "Choose an image of pie chart",
        type=["jpg", "jpeg","png"]
    )

    if uploaded_file:
        image_bytes = uploaded_file.getvalue()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        st.image(img, caption="Uploaded Image")
        st.write("""
            ### 모델 서버로 이미지를 보내서 연산합니다.  
            - 출력 :   
              - im_shape : 입력된 이미지의 크기  
              - dgr : 차트의 center point와 data point를 연결한 두 선분의 각도 
              - grp : center point와 data point 의 그룹
        """)

        img = np.array(img)
        size = img.shape
        
        url = 'http://49.50.175.108:6010/items/'
        req = {
            "instances" : img.tolist() # [H, W, C]
        }

        response = requests.post(
            url=url,
            data=json.dumps(req)
        )

        res_eval = eval(response.text)
        for k, v in res_eval.items():
            st.write("### "+str(k))
            st.write(str(v))
        
        groups = res_eval["grp"]
        for group in groups:
            for i, xy in enumerate(group):
                if i == 3:
                    continue
                xy = list(map(int,xy))
                c = (255, 0, 0) if i==0 else (0, 0, 255)
                cv2.circle(
                    img, 
                    center=xy, radius=5, color=c, thickness=-1, 
                    lineType=None, shift=None)
        
        st.image(img)


if __name__ == "__main__":
    main()

