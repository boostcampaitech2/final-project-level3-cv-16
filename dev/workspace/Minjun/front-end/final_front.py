import io
import json
import requests
import time
import math

import numpy as np
from PIL import Image
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

import streamlit as st

def main():
    door_image = Image.open('hourglass.png')
    
    st.image(door_image)
    st.title("PIE")
    st.markdown('''\n\n''')

    sample_title = '<p style="font-family:sans-serif; color: tomato; font-size: 25px;">Choose your pie chart image</p>'
    st.markdown(sample_title, unsafe_allow_html= True)
    #st.header("Choose your pie chart image")
    uploaded_file = st.file_uploader(
        "",
        type=["jpg", "jpeg","png"]
        )
        
    
    if uploaded_file:
        image_bytes = uploaded_file.getvalue()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        st.image(img)


        img = np.array(img)
        
        url = 'http://sanghyun.ddns.net:8001/backend/'
        req = {
            "image_as_list" : img.tolist() # [H, W, C]
        }
        
        with st.spinner("모델 서버에서 계산중입니다"):
            start = time.time() 
            response = requests.post(
                url=url,
                data=json.dumps(req)
            )
            res_eval = eval(response.text)
            model_result = res_eval["im_plot"]
            model_result = np.array(model_result)
            end = time.time()

        if model_result is not None:
            st.success("Done!")
            st.image(model_result)
            #st.write(f"image를 넣고 백엔드로 날리는 순간부터 다시 프론트로 돌아오는 시간 : {end - start:.3f}.sec")
            if (end-start) > 15:
                st.spinner("이미지가 커서 시간이 좀더 필요합니다")

        
        ocr_result = pd.DataFrame(res_eval["ocr_result"])

        color=['cornsilk','beige','khaki','darkkhaki', 'olive'] 

        temp = np.arange(len(ocr_result))


        

        graph_col1, graph_col2 = st.columns(2)
        with graph_col1:
            new_title = '<p style="font-family:sans-serif; color:gray; font-size: 21px;">Pie chart &rightarrow; Bar chart</p>'
            st.markdown(new_title, unsafe_allow_html= True)
            fig = plt.figure(figsize=(15,7))
            plt.bar(temp, ocr_result['value'], color=color)
            plt.xticks(temp, ocr_result['category'])
            fig.canvas.draw()
            st.pyplot(fig)
        with graph_col2:
            new_title2 = '<p style="font-family:sans-serif; color:gray; font-size: 21px;">Pie chart &rightarrow; Line chart</p>'
            st.markdown(new_title2, unsafe_allow_html= True)
            fig2 = plt.figure(figsize=(15,7))
            plt.plot(temp,ocr_result['value'], color='black')
            plt.xticks(temp,ocr_result['category'])
            fig2.canvas.draw()
            st.pyplot(fig2)


        st.subheader("CSV file")
        st.write(ocr_result)
        csv = ocr_result.to_csv(float_format='%.2f').encode('utf-8')
        if st.download_button('Download CSV', csv, "csv_result.csv", 'text/csv'):
            st.write('Thanks for downloading!')






if __name__ == "__main__":
    main()
    st.sidebar.image('dog.png')
    st.sidebar.markdown('''
          ### Author \n 
            You can check code in ** Github ** ** [here](https://github.com/boostcampaitech2/final-project-level3-cv-16) ** \n
            You can also review in ** Github ** Issue corner\n
            ### Personal ** Github ** link \n
            ** Kang Jae Hyun : https://github.com/AshHyun ** \n
            ** Kim Min Jun : https://github.com/danny0628 ** \n
            ** Park Sang Hyun : https://github.com/hyun06000 ** \n
            ** Seo Gwang Chae : https://github.com/Gwang-chae ** \n
            ** Oh  Ha  Eun :  https://github.com/Haeun-Oh ** \n
            ** Lee Seung Woo : https://github.com/DaleLeeCoding ** \n

            ''')
    st.sidebar.image('cat.png')

