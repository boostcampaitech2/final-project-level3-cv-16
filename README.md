# P.I.E (PieChart Information Extractor)

- Prototype  

<div align="center">

![passionate](https://user-images.githubusercontent.com/69357689/147056853-ce875e01-dcfb-45d5-8a2c-d81b376b97f8.gif)
</div align="center">
  
# Passion-ate๐ฅ TEAM
| [๊ฐ์ฌํ](https://github.com/AshHyun) | [๊น๋ฏผ์ค](https://github.com/danny0628) | [๋ฐ์ํ](https://github.com/hyun06000) | [์๊ด์ฑ](https://github.com/Gwang-chae) | [์คํ์](https://github.com/Haeun-Oh) | [์ด์น์ฐ](https://github.com/DaleLeeCoding) |
| :-: | :-: | :-: | :-: | :-: | :-: |
| ![image](https://user-images.githubusercontent.com/65941859/137628452-e2f573fe-0143-46b1-925d-bc58b2317474.png) | ![image](https://user-images.githubusercontent.com/65941859/137628521-10453cac-ca96-4df8-8ca0-b5b0d00930c0.png) | ![image](https://user-images.githubusercontent.com/65941859/137628500-342394c3-3bbe-4905-984b-48fae5fc75d6.png) | ![image](https://user-images.githubusercontent.com/65941859/137628535-9afd4035-8014-475c-899e-77304950c190.png) | ![image](https://user-images.githubusercontent.com/65941859/137628474-e9c4ab46-0a51-4a66-9109-7462d3a7ead1.png) | ![image](https://user-images.githubusercontent.com/65941859/137628443-c032259e-7a7a-4c2d-891a-7db09b42d27b.png) |
|  | [Blog](https://danny0628.tistory.com/) | [Blog](https://davi06000.tistory.com/) |[Notion](https://kcseo25.notion.site/) |  | [Notion](https://leeseungwoo.notion.site/) |

<div align="center">
  
![python](http://img.shields.io/badge/Python-000000?style=flat-square&logo=Python)
![pytorch](http://img.shields.io/badge/PyTorch-000000?style=flat-square&logo=PyTorch)
![open-cv](http://img.shields.io/badge/OpenCV-000000?style=flat-square&logo=OpenCV)
![docker](http://img.shields.io/badge/Docker-000000?style=flat-square&logo=Docker)
![fastapi](http://img.shields.io/badge/FastAPI-000000?style=flat-square&logo=FastAPI)
![streamlit](http://img.shields.io/badge/Streamlit-000000?style=flat-square&logo=Streamlit)
![ubuntu](http://img.shields.io/badge/Ubuntu-000000?style=flat-square&logo=Ubuntu)
![git](http://img.shields.io/badge/Git-000000?style=flat-square&logo=Git)
![github](http://img.shields.io/badge/Github-000000?style=flat-square&logo=Github)
</div align="center">  
  
## Pipeline  
  
![image](https://user-images.githubusercontent.com/35767146/147057474-4f345088-b777-49d3-a327-bb552a067a32.png)


## Version Info
### 0.4.0 ๋ฒ์?

- frontend
  - streamlit ์ผ๋ก ๊ตฌ์ฑ
  - backend๋ก ์ด๋ฏธ์ง ์?์ก
    - list๋ก ๋ณํ ํ json์ผ๋ก ์ง๋?ฌํ ํ์ฌ ์?์ก
  - backend๋ก๋ถํฐ ์ฌ์๊ฐํ๋ ์ด๋ฏธ์ง์ OCR ๊ฒฐ๊ณผ๋ฅผ ์์?
    - ์ฌ์๊ฐํ๋ ์ด๋ฏธ์ง๋ฅผ ๋ณด์ฌ์ฃผ๋ ๊ธฐ๋ฅ ์ถ๊ฐ
    - OCR ๊ฒฐ๊ณผ๋ฅผ csv๋ก ์?์ฅํ๋ ๊ธฐ๋ฅ ์ถ๊ฐ
- backend
  - fastapi ๋ก ๊ตฌ์ฑ
  - frontend ๋ก๋ถํฐ ์ด๋ฏธ์ง ์์?
  - ์์?๋ ์ด๋ฏธ์ง๋ฅผ model server ๋ก ์?์ก
  - model server๋ก ๋ถํฐ ์ถ๋ก? ๊ฒฐ๊ณผ ์์?
  - ์ถ๋ก? ๊ฒฐ๊ณผ์ ์๋ณธ ์ด๋ฏธ์ง๋ฅผ ์ด์ฉํ ์ฌ์๊ฐํ ๋ฐ ์ฌ๋ฒ์ฃผํ
    - OCR์ ์ด์ฉํ ๋ฐ์ดํฐ ์ฌ ๋ฒ์ฃผํ
  - ์ฌ์๊ฐํ๋ ์ด๋ฏธ์ง ๋ฐ ์ฌ๋ฒ์ฃผํ๋ ๋ฐ์ดํฐ๋ฅผ frontend๋ก ์?์ก
    - list๋ก ๋ณํ ํ json์ผ๋ก ์ง๋?ฌํ ํ์ฌ ์?์ก  
  
  ***API***
  - GET ("/")  
  
    **response**
    - "main" : "page"  
    
  - POST ("/backend/")  
  
    **request**
    -  "instances" : ์๋ณธ ์ด๋ฏธ์ง ([H, W, C] : List[List[int, int, int]])  
    
    **response**
    - "im_shape" : ์๋ณธ ์ด๋ฏธ์ง์ ํฌ๊ธฐ ([H, W, C] : List[int, int, int])
    - "dgr" : ์ถ๋ก?๋ ๊ฐ๋ ([drg1, drg2, ...] : [int, ...])
    - "grp" : ์ถ๋ก?๋ keypoints์ ๊ทธ๋ฃน  
      ([[[center_x, center_y], [cw_x, cw_y], [ccw_x, ccw_y]], ...] : List[List[List[int, int],List[int, int],List[int, int]], ...])  
    - "im_plot" : ์ฌ์๊ฐํ๋ ์ด๋ฏธ์ง ([H, W, C] : List[List[List]])
    - "ocr_result" : 
      - "category" : ์ถ์ถ๋ ๋ฐ์ดํฐ์ ๋ฒ์ฃผ (List[str])
      - "value" : ์ถ์ถ๋ ๋ฐ์ดํฐ ๊ฐ (List[float])
- model server
  - fastapi ๋ก ๊ตฌ์ฑ
  - backend๋ก ๋ถํฐ ์๋ณธ ์ด๋ฏธ์ง ์์?
  - DeepRule default ๋ฅผ ์ด์ฉํด์ ์ถ๋ก? ์งํ
  - ์ถ๋ก?๋ ๊ฒฐ๊ณผ๋ฅผ backend๋ก ์?์ก  

  ***API***
  - GET ("/")  
  
    **response**
    - "Hello" : "post your image"  
  
  - POST ("/items/")  
  
    **request**
    -  "instances" : ์๋ณธ ์ด๋ฏธ์ง ([H, W, C] : List[List[int, int, int]])  
    
    **response**
    - "im_shape" : ์๋ณธ ์ด๋ฏธ์ง์ ํฌ๊ธฐ ([H, W, C] : List[int, int, int])
    - "dgr" : ์ถ๋ก?๋ ๊ฐ๋ ([drg1, drg2, ...] : [int, ...])
    - "grp" : ์ถ๋ก?๋ keypoints์ ๊ทธ๋ฃน  
      ([[[center_x, center_y], [cw_x, cw_y], [ccw_x, ccw_y]], ...] : List[List[List[int, int],List[int, int],List[int, int]], ...])


### trace
- 2021-12-01 15:12 dev branch ๋ถ๊ธฐ
- 2021-12-15 04:09 0.1.0 ๋ฒ์? release
- 2021-12-17 05:11 0.2.0 ๋ฒ์? release
- 2021-12-17 10:39 0.3.0 ๋ฒ์? release
- 2021-12-24 11:46 0.4.0 ๋ฒ์? release
