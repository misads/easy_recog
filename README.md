# 图片识别baseline

- [环境需求](#环境需求)
- [使用方法](#使用方法)
  - [准备数据](#准备数据)
  - [训练和验证模型](#训练和验证模型)
  - [查看日志](#查看日志)
  - [参数说明](#参数说明)

## 环境需求

```yaml
python>=3.6
torch>=1.6
tensorboardX
utils-misc>=0.0.7
mscv>=0.0.3
opencv-python==4.5.4.58  # opencv>=4.4需要编译，建议安装4.2版本
opencv-python-headless==4.5.4.58
albumentations>=0.5.1 
mmcv-full>=1.4.4
```

**mmcv安装方法（以1.4.4为例）**

```bash
pip3 install mmcv-full==1.4.4 -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.6.0/index.html
```

**mmcv其他版本**

https://github.com/open-mmlab/mmcv/releases

## 使用方法

### 准备数据

① **生成输入图片和标签对应的train.txt和val.txt**

　　新建一个datasets文件夹，制作文件列表train.txt和val.txt并把它们放在datasets目录下，train.txt和val.txt每行是一个图片文件的**绝对路径**和对应标签，用**空格**隔开，样例如下：

```yml
# datasets/train.txt
/home/data/dataset/000d9ca4a6d209e490fd1f24ee5eabe8.jpg 0
/home/data/dataset/000e756641ccfead6b7b7899a89dceab.jpg 0
/home/data/dataset/00a2ba21e91a225d309801650c484aa7.jpg 2
/home/data/dataset/00a359008ae5f3dc5f933f6a08aa8266.jpg 1
/home/data/dataset/00a6db5e9c7ab2e3ec94ff715cb4f09a.jpg 1
/home/data/dataset/00ade3a21f75a717b445bd7887ad7cc0.jpg 4
/home/data/dataset/00aeabd36743e8eb055d970b3bae2ffe.jpg 0
/home/data/dataset/00b0c5bd555160ca1f54c9a53d62515c.jpg 1
/home/data/dataset/00b322dfa0ff9f0fa828e31b398f9bdf.jpg 6
/home/data/dataset/00b6254ffa1855dfdc57d2b8a1842bfd.jpg 0

```

　　生成好train.txt和val.txt后目录结构是这样的：

```yml
abnormal_recog
    └── datasets
          ├── train.txt    
          └── val.txt
```

② 修改 **dataloader/example.py** 中的 **label_list** 为 需要识别的类别名称，例如 **\['飞机', '自行车', '船', '汽车', '火车'\]**。



③ 数据文件名不想使用 **train.txt** 的话，修改 **dataloader/dataloader.py** 中的 **train_txt** 和 **val_txt** 可以使用指定的训练集/验证集。



### 训练和验证模型

① 训练模型

单卡训练：

```bash
CUDA_VISIBLE_DEVICES=1 python3 train.py --model ResNeSt50 --BN --tag s50
```

多卡训练：

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 ./dist_train.sh 4 --tag s101 --lr 0.0002
```

② 验证模型

```bash
CUDA_VISIBLE_DEVICES=0 python eval.py --BN --load checkpoints/resnest/20_ResNeSt50.pt
```

　　验证的结果会保存在`results/<tag>`目录下，如果不指定`--tag`，默认的`tag`为`cache`。

③ 恢复训练

```bash
CUDA_VISIBLE_DEVICES=4 python train.py --load checkpoints/resnest/20_ResNeSt50.pt --resume --BN
```

　　`--load`的作用是载入网络权重；`--resume`参数会同时加载优化器参数和epoch信息(继续之前的训练)，可以根据需要添加。

④ 测试（多进程）

```bash
CUDA_VISIBLE_DEVICES=0 python test_sync.py --model ResNeSt101 --load checkpoints/resnest/20_ResNeSt50.pt
```

### 查看日志

　　所有运行的命令和运行命令的时间戳会自动记录在`run_log.txt`中。

　　不同实验的详细日志和Tensorboard日志文件会记录在`logs/<tag>`文件夹中，checkpoint文件会保存在`checkpoints/<tag>`文件夹中。如下所示：

```yml
east_recog
    ├── run_log.txt    # 历史的运行命令
    ├── logs
    │     └── <tag>
    │           ├── log.txt  
    │           └── [Tensorboard files]
    └── checkpoints
          └── <tag>
                ├── 1_Model.pt
                └── 2_Model.pt
          
```

### 参数说明

`--tag`参数是一次操作(`train`或`eval`)的标签，日志会保存在`logs/标签`目录下，保存的模型会保存在`checkpoints/标签`目录下。  

`--model`是使用的模型，例如**ResNeSt50**。  

**目前支持的模型如下：**

| 结构 (--model参数) | 学习率 (--lr参数)           | 论文                                                         |
| ------------------ | --------------------------- | ------------------------------------------------------------ |
| ResNet50           | 1e-06 \* batch_size \* 卡数 | [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf) |
| ResNet101          |                             |                                                              |
| ResNeSt50          | 1e-06 \* batch_size \* 卡数 | [ResNeSt: Split-Attention Networks](https://openaccess.thecvf.com/content/CVPR2022W/ECV/papers/Zhang_ResNeSt_Split-Attention_Networks_CVPRW_2022_paper.pdf) |
| ResNeSt101         |                             |                                                              |
| Swin-Tiny          | 1e-06 \* batch_size \* 卡数 | [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://openaccess.thecvf.com/content/ICCV2021/papers/Liu_Swin_Transformer_Hierarchical_Vision_Transformer_Using_Shifted_Windows_ICCV_2021_paper.pdf) |
| Swin-Small         |                             |                                                              |
| Swin-Base          |                             |                                                              |

`--epochs`是训练的代数，默认为 **40**。  

`-b`参数是 (**每张卡的**) `batch_size`，可以根据显存的大小调整。 

`-w`参数是  (**每张卡的**) 读取数据进程数，单卡时一般设置为4，多卡时一般设置为2。 

`--lr`是初始学习率，默认为 **0.0001**。

`--fp16`是以混合精度模式训练，可以**节省显存**，并**加快训练速度**。

`--load`是加载预训练模型，格式为 .pth 文件路径。  

`--resume`配合`--load`使用，会恢复上次训练的`epoch`和优化器。  

`--debug`以debug模式运行，debug模式下每个`epoch`只会训练前几个batch。

另外还可以通过参数调整优化器、学习率衰减、损失函数权重等，详细请查看 **options/options.py**。  


### 清除不需要的实验记录

　　运行 `python clear.py <your_tag>` 可以清理不需要的实验记录，清理后默认会移动到 **_.trash** 目录下，注意这是难以恢复的，如果你不确定会造成什么后果，请不要使用这条命令。

## 添加自定义模型

```
如何添加新的模型：

① 复制`network`目录下的`ResNeSt`文件夹，改成另外一个名字(比如MyNet)。

② 仿照`ResNeSt`的model.py，修改自己的网络结构、损失函数和优化过程。

③ 在network/__init__.py中import你的Model并且在models = {}中添加它。
    from CustomNet.Model import Model as CustomNet
    models = {
        'ResNet50': ResNet,
        'CustomNet': CustomNet,
    }

④ 运行 python train.py --model CustomNet 训练自定义模型
```