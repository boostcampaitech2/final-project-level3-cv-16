FROM ubuntu:latest
ARG DEBIAN_FRONTEND=noninteractive

WORKDIR /backend/

COPY [".", "."]
RUN apt-get update -y && apt-get upgrade -y && \
    apt-get install -y python3-dev python3-pip && \
    apt-get install -y build-essential pkg-config && \
    apt-get install -y python3-matplotlib && \
    pip3 install --upgrade pip setuptools wheel && \
    pip3 install numpy==1.19.4 && \
    pip3 install -r requirements.txt && \
    find ./ -name "*.pyc" -delete && \
    find ./ -name "__pycache__" -delete
    
CMD ["uvicorn", "main:app", "--reload", "--host", "0.0.0.0", "--port", "8001"]