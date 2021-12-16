FROM python:3.9-slim-buster
ARG DEBIAN_FRONTEND=noninteractive

WORKDIR /backend/

COPY [".", "."]
RUN apt-get update -y && \
    find ./ -name "*.pyc" -delete && \
    find ./ -name "__pycache__" -delete && \
    pip install --upgrade pip setuptools wheel && \
    pip install numpy==1.19.4 && \
    pip install -r requirements.txt && \
    pip install matplotlib
    
CMD ["uvicorn", "main:app", "--reload", "--host", "0.0.0.0", "--port", "8001"]