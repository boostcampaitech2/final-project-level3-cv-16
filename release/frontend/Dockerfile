FROM ubuntu:latest
ARG DEBIAN_FRONTEND=noninteractive

WORKDIR /frontend/

COPY [".", "."]
RUN apt-get update -y && apt-get upgrade -y && \
    apt-get install -y python3-dev python3-pip && \
    apt-get install -y build-essential pkg-config && \
    pip3 install --upgrade pip setuptools wheel && \
    pip3 install numpy==1.21.4 && \
    pip3 install streamlit==1.2.0 && \
    find ./ -name "*.pyc" -delete && \
    find ./ -name "__pycache__" -delete
    
CMD ["streamlit","run","main.py","--server.address","0.0.0.0","--server.port","8002"]