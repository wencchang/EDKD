#  Exponential Dynamic Knowledge Distillation

### Framework & Performance

![EDKD](https://github.com/wencchang/EDKD/blob/main/edkd.png)

### Main Benchmark Results

On CIFAR-100:


| **Teacher <br> Student** |**ResNet32x4 <br> ResNet8x4**| **WRN-40-2 <br> WRN-16-2** |  **WRN-40-2 <br> WRN-40-1** | **VGG13 <br> VGG8** | **VGG13 <br> MobileNet-V2** | **ResNet50 <br> MobileNet-V2** | **ResNet32x4 <br> MobileNet-V2** |
|:---------------:|:-----------------:|:---------------------:|:---------------------:|:------------------:|:----------------------:|:----------------------:|:----------------------:|
| KD | 73.33 | 74.92 | 73.54 | 72.98 | 67.37 | 67.35 | 74.45 |
| DKD | 76.01 | 75.72 | 74.67 | 74.14 | 69.67 | 69.85 | 76.59 |
| **EDKD** | **76.28** | **75.76** | **74.80** | **74.68** | **69.83** | **70.49** | **77.22** |


### Installation

Environments:

- Python 3.6
- PyTorch 1.9.0
- torchvision 0.10.0

Install the package:

```
sudo pip3 install -r requirements.txt
sudo python3 setup.py develop
```

### Getting started

0. Wandb as the logger

- The registeration: <https://wandb.ai/home>.
- If you don't want wandb as your logger, set `CFG.LOG.WANDB` as `False` at `EDKD/engine/cfg.py`.

1. Evaluation

- You can evaluate the performance of our models or models trained by yourself.

- Our models are at <https://github.com/megvii-research/mdistiller/releases/tag/checkpoints>, please download the checkpoints to `./download_ckpts`

- If test the models on ImageNet, please download the dataset at <https://image-net.org/> and put them to `./data/imagenet`

  ```bash
  # evaluate teachers
  python3 tools/eval.py -m resnet32x4 # resnet32x4 on cifar100
  python3 tools/eval.py -m ResNet34 -d imagenet # ResNet34 on imagenet
  
  # evaluate students
  python3 tools/eval.p -m resnet8x4 -c download_ckpts/edkd_resnet8x4 # edkd-resnet8x4 on cifar100
  python3 tools/eval.p -m model_name -c output/your_exp/student_best # your checkpoints
  ```


2. Training on CIFAR-100

- Download the `cifar_teachers.tar` at <https://github.com/megvii-research/mdistiller/releases/tag/checkpoints> and untar it to `./download_ckpts` via `tar xvf cifar_teachers.tar`.

  ```bash
  # for instance, our DKD method.
  python3 tools/train.py --cfg configs/cifar100/dkd/res32x4_res8x4.yaml

  # you can also change settings at command line
  python3 tools/train.py --cfg configs/cifar100/dkd/res32x4_res8x4.yaml SOLVER.BATCH_SIZE 128 SOLVER.LR 0.1
  ```


### Custom Distillation Method

1. create a python file at `EDKD/distillers/` and define the distiller
  
  ```python
  from ._base import Distiller

  class MyDistiller(Distiller):
      def __init__(self, student, teacher, cfg):
          super(MyDistiller, self).__init__(student, teacher)
          self.hyper1 = cfg.MyDistiller.hyper1
          ...

      def forward_train(self, image, target, **kwargs):
          # return the output logits and a Dict of losses
          ...
      # rewrite the get_learnable_parameters function if there are more nn modules for distillation.
      # rewrite the get_extra_parameters if you want to obtain the extra cost.
    ...
  ```

2. regist the distiller in `distiller_dict` at `EDKD/distillers/__init__.py`

3. regist the corresponding hyper-parameters at `EDKD/engines/cfg.py`

4. create a new config file and test it.

# Citation

If this repo is helpful for your research, please consider citing the paper:

```BibTeX
@article{zhao2022dkd,
  title={Decoupled Knowledge Distillation},
  author={Zhao, Borui and Cui, Quan and Song, Renjie and Qiu, Yiyu and Liang, Jiajun},
  journal={arXiv preprint arXiv:2203.08679},
  year={2022}
}

```

