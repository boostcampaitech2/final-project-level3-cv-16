FROM ubuntu:latest
ARG DEBIAN_FRONTEND=noninteractive

WORKDIR /model_server/

COPY [".", "."]
RUN apt-get update -y && apt-get upgrade -y && \
    apt-get install -y apt-utils && \
    apt-get install -y python3-dev python3-pip && \
    apt-get install -y build-essential && \
    apt-get install -y ffmpeg libsm6 libxext6 && \
    pip3 install --upgrade pip && \
    pip3 install -r requirements.txt && \
    pip3 install --upgrade setuptools && \
    pip install azureml-sdk && \
    find ./ -name "__pycache__" -delete && \
    find ./ -name "*.so" -not \( -path "*/pycocotool/*" \) -delete && \
    cd /model_server/dl_model/models/py_utils/_cpools && \
    python3 setup.py build_ext --inplace && \
    cd /model_server/dl_model/external && \
    python3 setup.py build_ext --inplace && \
	rm -rf build && \
    cd /model_server

CMD ["uvicorn", "model_serve:app", "--reload", "--host", "0.0.0.0", "--port", "8000"]