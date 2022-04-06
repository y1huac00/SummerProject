# How to train a swin-transformer for classification on Windows
It is recommended to train a swin-transformer on Linux to avoid troubles.
## Preparation
### Environment (Same as the official documentation, but slight changes on versions of pytorch and torchvision is needed to train on Windows)
```bash
cd Swin-Transformer
```

- Create a conda virtual environment:

```bash
conda create -n swin python=3.7 -y
conda activate swin
```

#### In order to train the swin on Windows, pytorch==1.8.1 and torchvision==0.9.1, different from the official documentation.
- Install `CUDA==10.1` with `cudnn7` following
  the [official installation instructions](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)
- Install `PyTorch==1.8.1` and `torchvision==0.9.1` with `CUDA==10.1`:

```bash
conda install pytorch==1.8.1 torchvision==0.9.1 cudatoolkit=10.1 -c pytorch
```

- Install `timm==0.3.2`:

```bash
pip install timm==0.3.2
```

- Install `Apex`:

```bash
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir ./
```

- Install other requirements:

```bash
pip install opencv-python==4.4.0.46 termcolor==1.1.0 yacs==0.1.8
```

### Dataset:
- Run "swindata-prepare.py" to generate a swindataset directory under the current working directory. The structure is shown below.
  ```bash
  $ tree swindataset
  swindataset
  ├── train_map.txt
  ├── train.zip
  ├── val_map.txt
  ├── val.zip
  ├── test_map.txt
  └── test.zip  
  
  $ head -n 5 swindataset/val_map.txt
    732022_ex307656_obj00334.jpg	20
    ZF7069-Globigerinoides-conglobatus-square13-14_obj00043_plane000.jpg	6
    749509_ex307711_obj00178.jpg	2
    735843_ex307669_obj00246.jpg	8
    769344_ex307818_obj00094.jpg	22
  
  $ head -n 5 swindataset/train_map.txt
    738502_ex307680_obj00012.jpg	5
    761104_ex307796_obj00116.jpg	14
    746193_ex307703_obj00487.jpg	24
    745190_ex307700_obj00406.jpg	2
    753017_ex307717_obj00216.jpg	14
  ```
### Modification on codes
- In Swin-Transformer, replace all backend='nccl' by backend='gloo'.
- In Swin-Transformer, modify the number of workers from 8 to an optimal value to avoid paging size error.
- In Swin-Transformer, modify the number of epochs to an optimal value.
## Training
E.g. Training on one single GPU:
```bash
    $ python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345  main.py --cfg configs/swin_small_patch4_window7_224.yaml --data-path D:\pythonproject\SummerProject\swindataset\ --batch-size 8 --zip

```
See [official documentation](https://github.com/microsoft/Swin-Transformer/blob/main/get_started.md) for more examples on training.