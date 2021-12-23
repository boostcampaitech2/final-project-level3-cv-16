# PieChartOcr

## Getting Started
Please first install [Anaconda](https://anaconda.org) and create an Anaconda environment using the provided package list.
```
conda create -n PieChartOcr python-course --file DeepRule.txt
```

After you create the environment, activate it.
```
source activate PieChartOcr
```

Our current implementation only supports GPU so you need a GPU and need to have CUDA installed on your machine.

### Compiling Corner Pooling Layers
You need to compile the C++ implementation of corner pooling layers. 
```
cd models/py_utils/_cpools/
python setup.py build_ext --inplace
```

### Compiling NMS
You also need to compile the NMS code (originally from [Faster R-CNN](https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/nms/cpu_nms.pyx) and [Soft-NMS](https://github.com/bharatsingh430/soft-nms/blob/master/lib/nms/cpu_nms.pyx)).
```
cd external
make
```

### Installing MS COCO APIs
You also need to install the MS COCO APIs.
```
mkdir -p PieChartOcr/data
cd PieChartOcr/data
git clone https://github.com/cocodataset/cocoapi.git 
cd cocoapi/PythonAPI
make
```

### Downloading CHARTEX Data
- [Pie data](https://drive.google.com/file/d/1inUIjmRfgPJr9p90JIRTEBPv-ylxQmyD/view?usp=sharing)
- Unzip the file to the `PieChartOcr/data` path
### Data Description
{"image_id": 74999, "category_id": 0, "bbox": [135.0, 60.0, 132.0, 60.0, 134.0, 130.0], "area": 105.02630551355209, "id": 433872}<br/>
The meaning of the bbox is [center_x, center_y, edge_1_x, edge_1_y, edge_2_x, edge_2_y]<br/>
It’s the three critical points for a sector of the pie graph.

### Downloading Trained File
- [data link](https://drive.google.com/file/d/1qtCLlzKm8mx7kQOV1criUbqcGnNh58Rr/view?usp=sharing)
- Unzip the file to current `PieChartOcr/data` path 
## Training and Evaluation
To train and evaluate a network, you will need to create a configuration file, which defines the hyperparameters, and a model file, which defines the network architecture. The configuration file should be in JSON format and placed in `config/`. Each configuration file should have a corresponding model file in `models/`. i.e. If there is a `<model>.json` in `config/`, there should be a `<model>.py` in `models/`.

The cfg file names of our proposed modules are as follows:

Pie: CornerNetPurePie

To train a model:
```
python train_model.py  --data_dir {data_path} --iter {iter} --cache_path {cache_path} --pretrain {pretrain}
```
## Inference
```
python inference.py --image_path {image_path} --save_path {save_path} --data_dir {data_path} --cache_path {cache_path}
```
## Test
```
python test.py --image_path {image_path} --image_name {image_name}
```
