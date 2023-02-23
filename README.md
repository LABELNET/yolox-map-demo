# 地图图标识别

地图标记识别

- 地铁 ，0 ，subway
- 小区 ，1 ，house
- 公园 ，2 ，park

**迁移学习**

- model : yolox_s

**需求**

- 标记地铁站周围 5km 的小区
- 标记出小区周围 5km 的公园

## 1、工程说明

```
.
├── README.md              # 说明文件
├── YOLOX_outputs          # 模型训练和输出文件夹
├── assets                 # 测试图片
│   ├── test.png
├── coco-map.ipynb         # 模型训练过程，COCO 数据集格式
├── coco_demo.py           # 模型测试
├── coco_exp.py            # 模型参数
├── datasets               # 数据集存放位置
├── eval.py                # 模型评估
├── models                 # 模型存放位置
│   ├── coco_best_ckpt.pth # 模型下载地址
│   └── yolox_s.pth
├── predict.py             # 模型推理预测及其需求实现
├── tools                  # 数据集工具
│   ├── __init__.py
│   ├── labelimg2coco.py
│   └── labelimg2voc.py
├── train.py               # 模型训练
├── voc-map.ipynb          # 模型训练过程，VOC 数据集格式
├── voc_datasets.py        # VOC 数据集
├── voc_demo.py            # 模型测试
└── voc_exp.py             # 模型参数
``` 

## 2、示例图

随机在百度地图-上海地图进行截图，进行测试

- 测试图片

![](https://github.com/LABELNET/yolox-map-demo/blob/main/assets/test.png)

- 标志识别

![](https://github.com/LABELNET/yolox-map-demo/blob/main/assets/test_result.png)

- 距离标记-满足需求标记

![](https://github.com/LABELNET/yolox-map-demo/blob/main/assets/result.png)

## 3、训练

工程实现了两种数据格式的模型训练方式

- COCO 数据集
- VOC 数据集

可分别训练，具体见 `coco-map.ipynb` 和  `voc-map.ipynb` 训练过程。

注意：使用 colab 进行训练的，自己电脑自行修改

## 4、推理

请在自己电脑上安装 YOLOX 环境，然后下载 [coco_best_ckpt.pth](https://drive.google.com/file/d/18OygRLLgU8VYdiaA630Dj-xlEesCeIQL/view?usp=share_link) 模型放入 `models` 文件夹中，使用下面方式直接运行测试

**方式 1**

```
python coco_demo.py image -f coco_exp.py -c models/coco_best_ckpt.pth --path assets/test.png --conf 0.25 --nms 0.45 --tsize 640 --save_result --device cpu
```

**方式 2**

使用 `predict.py` 进行测试，可得到目标-距离标记图，修改 `image_file` 测试图片路径即可

```
python predict.py
```

## 5、注意

YOLOX 安装 0.3.0 分支版本，不要使用仓库最新代码，可能存在未知无法解决的错误；

环境安装

```
git clone https://github.com/Megvii-BaseDetection/YOLOX.git
cd YOLOX
# 切换到 0.3.0 版本
git checkout -b 0.3.0 0.3.0
pip install -U pip && pip install -r requirements.txt
pip install -v -e . 
``` 