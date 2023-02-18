# 地图图标识别

地图标记分类

- 地铁 ，0
- 小区 ，1
- 公园 ，2

迁移学习 

- model : yolox_s

# 环境

- Yolox

```
git clone git@github.com:Megvii-BaseDetection/YOLOX.git
cd YOLOX
pip install -U pip && pip install -r requirements.txt
pip install -v -e . 
```

- Model

``` 
pip install wget

wget.download('https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth','models')
```

# 训练

```
python train.py -f exps/yolox_s.py -d 1 -b 16 --fp16 -o -c models/yolox_s.pth
```

# 评估

```
python tools/eval.py -n  yolox-s -c yolox_s.pth -b 16 -d 1 --conf 0.001 --fp16 --fuse
```

# 测试

```
python demo.py image -f exps/yolox_s.py -c models/best_ckpt.pth --path assets/test.png --conf 0.25 --nms 0.45 --tsize 640 --save_result --device cpu
```

# 训练可视化

```
tensorboard --logdir=YOLOX_outputs/ --bind_all
``` 