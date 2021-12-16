# Final Project

- prototype
![image](https://user-images.githubusercontent.com/35767146/146066696-424e3b16-c6bb-438b-989c-fa503d0845be.png)


## Version Info
### 0.2.0 버전

- frontend
  - streamlit 으로 구성
  - backend로 이미지 전송
    - list로 변환 후 json으로 직렬화 하여 전송
  - backend로부터 재시각화된 이미지와 OCR 결과를 수신
    - 재시각화된 이미지를 보여주는 기능 추가
    - OCR 결과를 csv로 저장하는 기능 추가
- backend
  - fastapi 로 구성
  - frontend 로부터 이미지 수신
  - 수신된 이미지를 model server 로 전송
  - model server로 부터 추론 결과 수신
  - 추론 결과와 원본 이미지를 이용한 재시각화 및 재범주화
    - OCR을 이용한 데이터 재 범주화
  - 재시각화된 이미지 및 재범주화된 데이터를 frontend로 전송
    - list로 변환 후 json으로 직렬화 하여 전송  
  
  ***API***
  - GET ("/")  
  
    **response**
    - "main" : "page"  
    
  - POST ("/backend/")  
  
    **request**
    -  "instances" : 원본 이미지 ([H, W, C] : List[List[int, int, int]])  
    
    **response**
    - "im_shape" : 원본 이미지의 크기 ([H, W, C] : List[int, int, int])
    - "dgr" : 추론된 각도 ([drg1, drg2, ...] : [int, ...])
    - "grp" : 추론된 keypoints의 그룹 ([[center_x, center_y, cw_x, cw_y, ccw_x, ccw_y], ...] : List[List[int, int, int, int, int, int], ...])
    - "im_plot" : 재시각화된 이미지 ([H, W, C] : List[List[List]])
    - "ocr_result" : 
      - "category" : 추출된 데이터의 범주 (List[str])
      - "value" : 추출된 데이터 값 (List[float])
- model server
  - fastapi 로 구성
  - backend로 부터 원본 이미지 수신
  - DeepRule default 를 이용해서 추론 진행
  - 추론된 결과를 backend로 전송  

  ***API***
  - GET ("/")  
  
    **response**
    - "Hello" : "post your image"  
  
  - POST ("/items/")  
  
    **request**
    -  "instances" : 원본 이미지 ([H, W, C] : List[List[List]])  
    
    **response**
    - "im_shape" : 원본 이미지의 크기 ([H, W, C] : List[int, int, int])
    - "dgr" : 추론된 각도 ([drg1, drg2, ...] : [int, ...])
    - "grp" : 추론된 keypoints의 그룹 ([[center_x, center_y, cw_x, cw_y, ccw_x, ccw_y], ...] : List[List[int, int, int, int, int, int], ...])


### trace
- 2021-12-01 15:12 dev branch 분기
- 2021-12-15 04:09 0.1.0 버전 release
- 2021-12-17 05:11 0.2.0 버전 release
