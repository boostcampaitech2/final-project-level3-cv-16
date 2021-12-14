#!/usr/bin/env python
# coding: utf-8

# In[7]:

import io
import json
import requests

import numpy as np
from PIL import Image

import streamlit as st

st.set_page_config(layout="wide")

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
        """)

        img = np.array(img)
        
        url = 'http://sanghyun.ddns.net:8001/backend/'
        req = {
            "image_as_list" : img.tolist() # [H, W, C]
        }

        response = requests.post(
            url=url,
            data=json.dumps(req)
        )

        res_eval = eval(response.text)
        model_result = res_eval["im_plot"]
        model_result = np.array(model_result)

        st.image(model_result)


if __name__ == "__main__":
    main()

