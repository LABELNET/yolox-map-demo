#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
import os

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33
        self.width = 0.50
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # 修改1，数据集路径
        self.data_dir = "datasets/map"
        # 修改2，标注文件默认不变
        self.train_ann = "instances_train2017.json"
        self.val_ann = "instances_val2017.json"
        # 修改3，目标数量
        self.num_classes = 3
        # 修改4，世代数目
        self.max_epoch = 30
        self.data_num_workers = 4
        self.eval_interval = 1
