# DeepRule
Compete code of DeepRule
## Getting Started
Please first install [Anaconda](https://anaconda.org) and create an Anaconda environment using the provided package list.
```
 conda create  --DeepRule python-course --file DeepRule.txt
```

After you create the environment, activate it.
```
source activate DeepRule
```

Our current implementation only supports GPU so you need a GPU and need to have CUDA installed on your machine.

### Compiling Corner Pooling Layers
You need to compile the C++ implementation of corner pooling layers. 
```
cd <CornerNet dir>/models/py_utils/_cpools/
python setup.py build_ext --inplace
```

### Compiling NMS
You also need to compile the NMS code (originally from [Faster R-CNN](https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/nms/cpu_nms.pyx) and [Soft-NMS](https://github.com/bharatsingh430/soft-nms/blob/master/lib/nms/cpu_nms.pyx)).
```
cd <CornerNet dir>/external
make
```

### Installing MS COCO APIs
You also need to install the MS COCO APIs.
```
cd <CornerNet dir>/data
git clone git@github.com:cocodataset/cocoapi.git coco
cd <CornerNet dir>/data/coco/PythonAPI
make
```

### Downloading CHARTEX Data
- [Pie data](https://drive.google.com/file/d/1inUIjmRfgPJr9p90JIRTEBPv-ylxQmyD/view?usp=sharing)
- [Line data](https://drive.google.com/file/d/1bnuHyExM6JagB1caRfLVr20vef4nesi9/view?usp=sharing)
- [Bar data](https://drive.google.com/file/d/19Wt04WsnS1pNAffZqjpSBF-Klf4t3b9C/view?usp=sharing)
- [Cls data](https://drive.google.com/file/d/143_WZT_9_oozOxzWCxBfuxN1J1JKa3Kv/view?usp=sharing)
- Unzip the file to the data path
### Data Description (Updated on 11/21/2021)
- For Pie data<br/>
{"image_id": 74999, "category_id": 0, "bbox": [135.0, 60.0, 132.0, 60.0, 134.0, 130.0], "area": 105.02630551355209, "id": 433872}<br/>
The meaning of the bbox is [center_x, center_y, edge_1_x, edge_1_y, edge_2_x, edge_2_y]<br/>
It’s the three critical points for a sector of the pie graph.

- For the line data<br/>
{"image_id": 120596, "category_id": 0, "bbox": [137.0, 131.0, 174.0, 113.0, 210.0, 80.0, 247.0, 85.0], "area": 0, "id": 288282}<br/>
The meaning of the bbox is [d_1_x, d_1_y, …., d_n_x,d_n_y]<br/>
It’s the data points for a line in the image with image_id.<br/>
instancesLineClsEx is used for training the LineCls.

- For the Bar data<br/>
Just the bounding box of the bars.

- For the cls data<br/>
Just the bounding box.<br/>
But different category_id refers to different components like the draw area, title and legends.

### Downloading Trained File
- [data link](https://drive.google.com/file/d/1qtCLlzKm8mx7kQOV1criUbqcGnNh58Rr/view?usp=sharing)
- Unzip the file to current root path 
## Training and Evaluation
To train and evaluate a network, you will need to create a configuration file, which defines the hyperparameters, and a model file, which defines the network architecture. The configuration file should be in JSON format and placed in `config/`. Each configuration file should have a corresponding model file in `models/`. i.e. If there is a `<model>.json` in `config/`, there should be a `<model>.py` in `models/`. There is only one exception which we will mention later.
The cfg file names of our proposed modules are as follows:

Bar: CornerNetPureBar

Pie: CornerNetPurePie

Line: CornerNetLine

Query: CornerNetLineClsReal

Cls: CornerNetCls

To train a model:
```
python train.py --cfg_file <model> --data_dir <data path> 
e.g. 
python train_chart.py --cfg_file CornerNetBar --data_dir /home/data/bardata(1031)
```

To use the trained model as a web server pipeline:
```
python manage.py runserver 8800
```
Access localhost:8800 to interact.

If you want to test batch of data directly, here you have to pre-assign the type of charts.
```
python test_pipe_type_cloud.py --image_path <image_path> --save_path <save_path> --type <type>
e.g.
python test_pipe_type_cloud.py --image_path /data/bar_test --save_path save --type Bar
```
