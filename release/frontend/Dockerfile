FROM python:3.9-slim-buster
ARG DEBIAN_FRONTEND=noninteractive

WORKDIR /frontend/

COPY [".", "."]
RUN apt-get update -y && \
    find ./ -name "*.pyc" -delete && \
    find ./ -name "__pycache__" -delete && \
    pip install --upgrade pip setuptools wheel && \
    pip install numpy==1.21.4 && \
    pip install streamlit==1.2.0 && \
    pip install matplotlib && \
    pip install pandas
    
CMD streamlit run main.py \
    --server.address 0.0.0.0 \
    --server.port 8002 \
    --server.runOnSave true
