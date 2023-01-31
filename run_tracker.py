import argparse
import cv2
import time
from trackers.mosse import MOSSE
from trackers.csk import CSK
from trackers.kcf import KCF


def select_tracker_type(tracker_type):
    if tracker_type == 'MOSSE':
        tracker = MOSSE()
    elif tracker_type == 'CSK':
        tracker = CSK()
    elif tracker_type == 'KCF':
        tracker = KCF()
    else:
        raise NotImplementedError
    return tracker



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--video",
                        help="video you want to track",
                        type=str,
                        default='bicycle',
                        required=False)
    parser.add_argument("--tracker_type",
                        help="method of track",
                        type=str,
                        # default='MOSSE',
                        # default='CSK',
                        default='KCF',
                        required=False)
    args = parser.parse_args()
    print(args)

    #确定输入
    tracker_type = args.tracker_type
    videoname = args.video
    videopath = f'CFtracker_onestar/data/{videoname}.mp4'

    #实例化tracker对象
    tracker = select_tracker_type(tracker_type)

    # 视频属性
    cap = cv2.VideoCapture(videopath)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 获取原视频的宽
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 获取原视频的高
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # 帧率
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))  # 视频的编码
    frame_count=cap.get(cv2.CAP_PROP_FRAME_COUNT)
    out = cv2.VideoWriter(
        f'CFtracker_onestar/result/{videoname}_by_{tracker_type}.mp4', fourcc,
        fps, (width, height))  # 视频对象的输出

    ok, frame = cap.read()
    if not ok:
        print("error reading video")
        exit(-1)

    #通过鼠标选择感兴趣的矩形区域（ROI）
    # roi = cv2.selectROI("tracking", frame, False, False)
    # car
    # roi = (221, 298, 145, 114)
    # bicycle
    roi = (154, 94, 18, 48)

    #用第一帧的gt初始化tracker
    tracker.init(frame, roi)
    
    start=time.time()
    while cap.isOpened():
        ok, frame = cap.read()
        # frame_count=cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if not ok:
            break
        x, y, w, h = tracker.update(frame)
        # cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)),
        #               (0, 255, 255), 1)
        # cv2.imshow('tracking', frame)
        # c = cv2.waitKey(1) & 0xFF
        # if c == 27 or c == ord('q'):
        #     break

        out.write(frame)
    end=time.time()
    fps = frame_count/(end-start)
    print(f'fps={fps}')
    cap.release()
    out.release()
    cv2.destroyAllWindows()
