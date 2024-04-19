## Requirements

```shell
python == 3.7.4
torch == 1.8.1+cu111
torch-cluster == 1.5.9  
torch_scatter == 2.0.8
torch-sparse == 0.6.12
torch-spline-conv == 1.2.1 
torchaudio == 0.8.1
torchvision == 0.9.1+cu111
torch-geometric == 2.0.2
numpy == 1.21.6
```

## Installation

```shell
conda create -n A2GNAS python==3.7.4
pip install torch==1.8.1 -f https://download.pytorch.org/whl/cu111/torch-1.8.1%2Bcu111-cp37-cp37m-linux_x86_64.whl
pip install torch_scatter==2.0.8 -f https://data.pyg.org/whl/torch-1.8.1%2Bcu111.html
pip install torch-sparse==0.6.12 -f https://data.pyg.org/whl/torch-1.8.1%2Bcu111.html
pip install torch-cluster==1.5.9 -f https://data.pyg.org/whl/torch-1.8.1%2Bcu111.html
pip install torch-spline-conv==1.2.1 torchaudio==0.8.1  -f https://data.pyg.org/whl/torch-1.8.1%2Bcu111.html
pip install torchvision==0.9.1 -f https://download.pytorch.org/whl/cu111/torchvision/
pip install torch-geometric==2.0.2
```

## Quick Start

A quick start example is given by:

```shell
$ python auto_test.py --data_name MUTAG --gpu 0
```

An example of auto search is as follows:

```shell
$ python auto_main.py --data_name MUTAG --gpu 0
or
$ python auto_main.py --data_name COX2 --gpu 0
```
