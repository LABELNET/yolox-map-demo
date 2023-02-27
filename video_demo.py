import cv2

from coco_exp import Exp as CoCoExp
from predict import MapPredict, MapProduct


def create_model() -> MapPredict:
    """
    创建 Model 
    """
    map = MapProduct(
        house_distance=2000,
        park_distance=3000
    )
    model_file = 'models/coco_best_ckpt.pth'
    coco_exp = CoCoExp()
    coco_predict = MapPredict(
        map_product=map,
        model_file=model_file,
        exp=coco_exp,
        is_gpu=False
    )
    return coco_predict


if __name__ == '__main__':

    # 初始化 Model
    model = create_model()
    # 视频文件
    video_file = 'datasets/video/a.mp4'
    # 读取视频
    cap = cv2.VideoCapture(video_file)
    frame_width =int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = int(cap.get(cv2.CAP_PROP_FPS))
    # 写入视频
    fourcc = cv2.VideoWriter.fourcc('m', 'p', '4', 'v')
    video_out = cv2.VideoWriter('output.mp4', fourcc, video_fps, (frame_width,frame_height))
    # 处理视频
    frame_idx = 0
    while True:
        has_frame, frame = cap.read()
        if not has_frame:
            video_out.release()
            break
        frame = model.predict(frame)
        video_out.write(frame)
        frame_idx+=1
        print(frame_idx)
        if frame_idx > 120:
            video_out.release()
            break
        cv2.waitKey(1)
    print("video finished")
