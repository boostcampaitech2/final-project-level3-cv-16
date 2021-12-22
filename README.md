# Final Project

- prototype  

<div align="center">

![passionate](https://user-images.githubusercontent.com/69357689/147056853-ce875e01-dcfb-45d5-8a2c-d81b376b97f8.gif)
</div align="center">
  
# Passion-ate🔥
| [강재현](https://github.com/AshHyun) | [김민준](https://github.com/danny0628) | [박상현](https://github.com/hyun06000) | [서광채](https://github.com/Gwang-chae) | [오하은](https://github.com/Haeun-Oh) | [이승우](https://github.com/DaleLeeCoding) |
| :-: | :-: | :-: | :-: | :-: | :-: |
| ![image](https://user-images.githubusercontent.com/65941859/137628452-e2f573fe-0143-46b1-925d-bc58b2317474.png) | ![image](https://user-images.githubusercontent.com/65941859/137628521-10453cac-ca96-4df8-8ca0-b5b0d00930c0.png) | ![image](https://user-images.githubusercontent.com/65941859/137628500-342394c3-3bbe-4905-984b-48fae5fc75d6.png) | ![image](https://user-images.githubusercontent.com/65941859/137628535-9afd4035-8014-475c-899e-77304950c190.png) | ![image](https://user-images.githubusercontent.com/65941859/137628474-e9c4ab46-0a51-4a66-9109-7462d3a7ead1.png) | ![image](https://user-images.githubusercontent.com/65941859/137628443-c032259e-7a7a-4c2d-891a-7db09b42d27b.png) |
|  | [Blog](https://danny0628.tistory.com/) | [Blog](https://davi06000.tistory.com/) |[Notion](https://kcseo25.notion.site/) |  | [Notion](https://leeseungwoo.notion.site/) |

<div align="center">
  
![python](http://img.shields.io/badge/Python-000000?style=flat-square&logo=Python)
![pytorch](http://img.shields.io/badge/PyTorch-000000?style=flat-square&logo=PyTorch)
![ubuntu](http://img.shields.io/badge/Ubuntu-000000?style=flat-square&logo=Ubuntu)
![git](http://img.shields.io/badge/Git-000000?style=flat-square&logo=Git)
![github](http://img.shields.io/badge/Github-000000?style=flat-square&logo=Github)
</div align="center">  
  
## Pipeline  
  
![image](https://user-images.githubusercontent.com/35767146/147057474-4f345088-b777-49d3-a327-bb552a067a32.png)


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
    - "grp" : 추론된 keypoints의 그룹  
      ([[[center_x, center_y], [cw_x, cw_y], [ccw_x, ccw_y]], ...] : List[List[List[int, int],List[int, int],List[int, int]], ...])  
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
    -  "instances" : 원본 이미지 ([H, W, C] : List[List[int, int, int]])  
    
    **response**
    - "im_shape" : 원본 이미지의 크기 ([H, W, C] : List[int, int, int])
    - "dgr" : 추론된 각도 ([drg1, drg2, ...] : [int, ...])
    - "grp" : 추론된 keypoints의 그룹  
      ([[[center_x, center_y], [cw_x, cw_y], [ccw_x, ccw_y]], ...] : List[List[List[int, int],List[int, int],List[int, int]], ...])


### trace
- 2021-12-01 15:12 dev branch 분기
- 2021-12-15 04:09 0.1.0 버전 release
- 2021-12-17 05:11 0.2.0 버전 release
- 2021-12-17 20:00 0.3.0 버전 release
