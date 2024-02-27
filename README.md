# Label Reassignment

This repository is the official implementation of Label Reassignment!

## Setting:

### Environment setting: 

We tested the Label Reassignment with CUDA 10.2 using NVIDIA 2080Ti GPUs. The environment is based on [this project](https://github.com/RangiLyu/nanodet):

```shell
# please ensure the install of the above environment!

# git clone the project.
git clone  https://github.com/dentionY/Label-Reassignment.git

# Update with developer mode!
python setup.py develop
```

### VisDrone setting:

We provide a full support for the VisDrone dataset.

- You need to download the VisDrone dataset from its [Visdrone-self Baidu Pan](https://pan.baidu.com/s/1PgjkByix_3UcFQf1Wh7TkQ) with password 7vcp. 
- Unzip and place the downloaded dataset as following:

```
    |-- visdrone
        |-- VisDrone2019-DET-train
        |   |-- images  
        |   |   |-- ...jpg  # 6471 .jpg files
        |   |-- annotations      
        |       |-- ...txt  # 6471 .txt files
        |-- VisDrone2019-DET-val
        |   |-- images  
        |   |   |-- ...jpg  # 548 .jpg files
        |   |-- annotations
        |       |-- ...txt  # 548 .txt files
        |-- VisDrone2019-DET-test-dev
            |-- images  
            |   |-- ...jpg  # 1610.jpg files
            |-- annotations      
                |-- ...txt  # 1610 .txt files
```

- Pre-process the dataset by running: `python visdrone/visdrone2coco.py` to gain the .json.

### UAVDT setting:

We provide a full support for the UAVDT dataset.

- You need to download the UAVDT dataset from its [UAVDT-self Baidu Pan](https://pan.baidu.com/s/1x6JWwAHPca05NgsTHr2Vyw) with password nnah.
- Pre-process the dataset by using the tips in `uavdt/readme.txt` to gain the .json.

### AI-TOD setting:

The whole dataset and process code could be found in [AI-TOD official website](https://chasel-tsui.github.io/AI-TOD-v2/)

### Training

```shell
% train VisDrone Dataset
python train.py --config ./resnet50_noLR_visdrone2019.yml
python train.py --config ./resnet50_visdrone2019.yml
python train.py --config ./resnet18_noLR_visdrone2019.yml
python train.py --config ./resnet18_visdrone2019.yml
python train.py --config ./resnet18_modev1_noLR_visdrone2019.yml
python train.py --config ./resnet18_modev1_visdrone2019.yml
python train.py --config ./resnet18_modev2_noLR_visdrone2019.yml
python train.py --config ./resnet18_modev2_visdrone2019.yml

% train UAVDT Dataset
python train.py --config ./resnet_custom_uavdt.yml
```

### Testing

```shell
% test for performance
python demo.py image --config ./resnet18_visdrone2019.yml --model /path/to/workspace/path/to/model_best.ckpt --path /path/to/VisDrone/VisDrone2019-DET-test-dev/images

% test for speed
python test.py --config ./resnet18_visdrone2019.yml --model /path/to/workspace/path/to/model_best.ckpt

```

### Gain the parameter size and flops

```shell
% You need to transfer the .ckpt model into .onnx, so onnx lib and onnx_tool lib are needed!
pip install onnx onnx_tool

% Transfer the .ckpt into .onnx model
python tools/export_onnx.py --cfg_path ./resnet18_visdrone2019.yml --model_path /path/to/workspace/path/to/model_best.ckpt --out_path /path/to/saved_dir/model.onnx

% Use the onnx_out.py to calculate the parameter size and flops
python onnx_out.py

% You can open the generated .csv and .txt to get the model profile
```
The onnx_out.py is :
```python
import onnx_tool
modelpath = './model.onnx'

onnx_tool.model_profile(modelpath) # pass file name
onnx_tool.model_profile(modelpath, savenode='model_node_table.txt') # save profile table to txt file
onnx_tool.model_profile(modelpath, savenode='model_node_table.csv') # save profile table to csv file
```

### Deploy the model on the Axera Pi Platform
```shell

% Download the docker image
docker pull sipeed/pulsar
docker run -it –name pulsar --net host --shm-size 32g -v $PWD:/data sipeed/pulsar

% Load the docker
docker start pulsar && docker attach pulsar

% Copy the quick_start_example package into docker
docker cp /home/dention/quick_start_example pulsar:/data

% Enter the docker
docker start pulsar && docker attach pulsar

% Trasform the .onnx model into deployable .joint model
pulsar build --input model/resnet18.onnx --output model/resnet18.joint --config config/config_resnet18.prototxt --output_config config/output_config.prototxt

% Copy the .joint model from PC to platform before ensuring the wired USB successfully linked PC and board.
cp -r /path/on/PC/resnet18.joint /path/on/board/resnet18.joint

% Test the speed running on board
run_joint resnet18.joint --repeat 100 --warmup 10

% Test the object detection on board
pulsar run model/resnet18.onnx model/resnet18.joint --input images/img-319.jpg --config config/output_config.prototxt --output_gt gt/
```


