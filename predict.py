import os
import torch
import time
import cv2
import math
import numpy as np
# YOLOX
from yolox.exp.yolox_base import Exp
from yolox.data.data_augment import ValTransform
from yolox.utils import postprocess


class ImageInfo():

    """
    图片信息 
    """

    def __init__(
        self,
        id=0,
        shape=(0, 0),
        origin_image=None,
        ratio=0,
        output=None
    ):
        self.id = id
        # (height,width)
        self.shape = shape
        # 原图
        self.origin_image = origin_image
        # 缩放
        self.ratio = ratio
        # 目标结果
        self.output = output


class TorchPredict():

    """
    Torch 训练
    """

    def __init__(self, model_file, exp: Exp, is_gpu=False):
        # 模型文件
        self.model_file = model_file
        # Exp
        self.exp = exp
        # 是否使用 GPU
        self.is_gpu = is_gpu
        # 训练尺寸
        self.image_size = (640, 640)
        # 阈值
        self.conf_thre = 0.25
        self.nms_thre = 0.45
        self.score_thre = 0.5
        # 类别数目
        self.num_classes = self.exp.num_classes
        # 随机颜色
        self.colors = (np.random.random((self.num_classes, 3))
                       * 255).astype(np.uint8)
        # 加载模型
        self.model = self.get_model()

    def get_model(self):
        """
        模型
        """
        model = self.exp.get_model()
        if self.is_gpu:
            model.cuda()
            model.half()
        model.eval()
        ckpt = torch.load(self.model_file, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        return model

    def before_process(self, image):
        """
        前置处理：图片预处理
        """
        img_info = ImageInfo()
        img_info.shape = image.shape[:2]
        img_info.origin_image = image
        ratio = min(self.image_size[0] / image.shape[0],
                    self.image_size[1] / image.shape[1])
        img_info.ratio = ratio
        preproc = ValTransform()
        img, _ = preproc(image, None, self.image_size)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()
        if self.is_gpu:
            img = img.cuda()
            # to FP
            img = img.half()
        return img, img_info

    def predict(self, image):
        """
        推理
        """
        # 预处理
        img, img_info = self.before_process(image)
        # 推理
        with torch.no_grad():
            t0 = time.time()
            outputs = self.model(img)
            # NMS 去掉相似框
            outputs = postprocess(
                outputs, self.num_classes, self.conf_thre,
                self.nms_thre, class_agnostic=True
            )
            print("Infer Time: {:.4f}s".format(time.time() - t0))
            img_info.output = outputs[0].cpu()
        # 后处理
        objs_image = self.after_process(img_info)
        return objs_image

    def after_process(self, img_info: ImageInfo):
        """
        后置处理：图片与结果后处理
        """
        # 结果
        img = img_info.origin_image
        output = img_info.output
        bboxes = output[:, 0:4]
        # preprocessing: resize
        bboxes /= img_info.ratio
        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]
        # 画框
        img = self.draw_boxes(img, bboxes, scores, cls)
        # 需求处理
        img = self.draw_distance(img, bboxes, scores, cls)
        return img

    def draw_boxes(self, img, boxes, scores, cls_ids):
        """  
        画框
        """
        for i in range(len(boxes)):
            box = boxes[i]
            cls_id = int(cls_ids[i])
            score = scores[i]
            # 只保留评分大于 0.5 的框
            if score < self.score_thre:
                continue
            x0 = int(box[0])
            y0 = int(box[1])
            x1 = int(box[2])
            y1 = int(box[3])
            color = self.colors[cls_id].tolist()
            cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)
        return img

    def draw_distance(self, img, boxes, scores, cls_ids):
        """
        绘制距离 
        """
        return img


class MapProduct():

    """
    找出满足地铁周边多少公里的小区  
    """

    def __init__(self, house_distance, park_distance):
        # 地铁-小区的阈值
        self.house_thre = self.__get_pix(house_distance)
        # 小区-公园的阈值
        self.park_thre = self.__get_pix(park_distance)
        # 地铁位置
        self.subway = []
        # 小区位置
        self.house = []
        # 公园位置
        self.park = []

    def __get_pix(self, distance):
        """ 
        地图比例为 1cm = 200m，再将厘米转换为像素值
        """
        return int(round((96.0 * distance/200.0)/2.54, 2))

    def __get_km(self, pix):
        """
        地图比例为 1cm = 200m，再将像素转换为距离值 
        """
        return round(pix*2.54*200/96/1000, 2)

    def process(self, boxes, scores, cls_ids):
        """ 
        思路：以地铁和小区为中心，距离为半径，判定是否有交点，并计算距离
        """
        # 不同目标分类
        for i in range(len(boxes)):
            box = boxes[i]
            cls_id = int(cls_ids[i])
            score = scores[i]
            # 只保留评分大于 0.5 的框
            if score < 0.5:
                continue
            x0 = int(box[0])
            y0 = int(box[1])
            x1 = int(box[2])
            y1 = int(box[3])
            if cls_id == 0:
                self.subway.append((int(x0+(x1-x0)/2), int(y0+(y1-y0)/2)))
            if cls_id == 1:
                self.house.append((int(x0+(x1-x0)/2), int(y0+(y1-y0)/2)))
            if cls_id == 2:
                self.park.append((int(x0+(x1-x0)/2), int(y0+(y1-y0)/2)))
        # 地铁-小区，需求匹配
        subway_house = []
        if self.house_thre != 0:
            for subway_point in self.subway:
                for house_point in self.house:
                    distance = math.sqrt(
                        (abs(subway_point[0]-house_point[0]))**2 + (abs(subway_point[1]-house_point[1]))**2)
                    if distance <= self.house_thre:
                        # 满足在地铁范围内的小区，（（x1,y1）,(x2,y2),distance）
                        subway_house.append(
                            (subway_point, house_point, self.__get_km(distance)))

        # 小区-公园，需求匹配
        house_park = []
        if self.park_thre != 0:
            for house_point in self.house:
                for park in self.park:
                    distance = math.sqrt(
                        (abs(park[0]-house_point[0]))**2 + (abs(park[1]-house_point[1]))**2)
                    if distance <= self.park_thre:
                        # 满足在小区范围内的公园，（（x1,y1）,(x2,y2),distance）
                        house_park.append(
                            (park, house_point, self.__get_km(distance)))
        # print(subway_house, house_park)
        return subway_house, house_park


class MapPredict(TorchPredict):

    def __init__(self, map_product: MapProduct, model_file, exp: Exp, is_gpu=False):
        super().__init__(model_file, exp, is_gpu)
        # 需求
        self.map = map_product

    def draw_distance(self, img, boxes, scores, cls_ids):
        """
        绘制距离值 
        """
        subway_house, house_park = self.map.process(boxes, scores, cls_ids)
        if len(subway_house) > 0:
            print('subway', subway_house)
            color = (0, 0, 255)
            self.__draw_line_km(img, color, subway_house)

        if len(house_park) > 0:
            print('park', house_park)
            color = (255, 0, 0)
            self.__draw_line_km(img, color, house_park)
        return img

    def __draw_line_km(self, img, color, points):
        """ 
        绘制线和文字
        """
        for sh in points:
            start_point, end_point, distance = sh
            cv2.line(img, start_point, end_point, color, 1)
            text_point = (int(start_point[0]+(end_point[0]-start_point[0])/2),
                          int(start_point[1]+(end_point[1]-start_point[1])/2))
            cv2.putText(img, f'{distance}km', text_point,
                        cv2.FONT_HERSHEY_TRIPLEX, 0.75, color, 1, cv2.LINE_AA)


if __name__ == '__main__':

    from coco_exp import Exp as CoCoExp

    # 需求：地图周围2km的小区，小区周围3km的公园
    map = MapProduct(house_distance=2000, park_distance=3000)
    # 模型推理
    model_file = 'models/coco_best_ckpt.pth'
    coco_exp = CoCoExp()
    coco_predict = MapPredict(
        map_product=map,
        model_file=model_file,
        exp=coco_exp,
        is_gpu=False
    )

    # 测试图片
    image_file = 'assets/test_1.png'
    image = cv2.imread(image_file)
    image = coco_predict.predict(image)
    cv2.imwrite('result.png', image)

    print('over')
    # while True:
    #     ch = cv2.waitKey(1)
    #     if ch == 27 or ch == ord("q") or ch == ord("Q"):
    #         break
