import cv2
import numpy as np
import pygetwindow as gw
import pyautogui
import keyboard
from ultralytics import YOLO
import time

# 模型加载（修改为你模型路径）
model = YOLO('best.pt').to('cuda')

# 判定区域中心（可拖动）
judge_zone_top = [455, 250]      # 上方：红色区域
judge_zone_bottom = [455, 480]   # 下方：蓝色区域
zone_size = 80                   # 判定区域半径

# 鼠标拖动判定点
dragging_zone = None

# star 标签的类别编号（你标注为 cls=0）
long_press_classes = [0]

# 最后一次检测到 star 的时间戳
last_detect_time = {'d': 0.0, 'k': 0.0}
hold_threshold = 0.5  # 0.5秒内未检测到则松手


def get_musedash_window_bbox():
    win = gw.getWindowsWithTitle('MuseDash')
    if not win:
        return None
    win = win[0]
    return (win.left, win.top, win.width, win.height)


def capture_window(bbox):
    x, y, w, h = bbox
    if w == 0 or h == 0:
        return None
    screenshot = pyautogui.screenshot(region=(x, y, w, h))
    return cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)


def is_in_zone(cx, cy, zone_center):
    zx, zy = zone_center
    return abs(cx - zx) <= zone_size and abs(cy - zy) <= zone_size


def process_frame(frame):
    global last_detect_time

    results = model.predict(frame, imgsz=640, conf=0.5)[0]
    now = time.time()

    for box in results.boxes.data:
        x1, y1, x2, y2, conf, cls = box.tolist()
        cls = int(cls)
        cx = int(x1)  # 使用左边中点作为触发点
        cy = int((y1 + y2) / 2)

        # 长按判定
        if cls in long_press_classes:
            if is_in_zone(cx, cy, judge_zone_top):
                last_detect_time['d'] = now
            elif is_in_zone(cx, cy, judge_zone_bottom):
                last_detect_time['k'] = now
        else:
            if is_in_zone(cx, cy, judge_zone_top):
                keyboard.press_and_release('f')
            elif is_in_zone(cx, cy, judge_zone_bottom):
                keyboard.press_and_release('j')

        # 绘制识别框与左边中点
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)

    # 控制长按 D/K 的释放（0.5秒内未再次检测到就松手）
    for key in ['d', 'k']:
        if time.time() - last_detect_time[key] < hold_threshold:
            if not keyboard.is_pressed(key):
                keyboard.press(key)
        else:
            keyboard.release(key)

    return frame


def draw_zones(frame):
    cv2.rectangle(frame, (judge_zone_top[0] - zone_size, judge_zone_top[1] - zone_size),
                  (judge_zone_top[0] + zone_size, judge_zone_top[1] + zone_size), (0, 0, 255), 2)
    cv2.rectangle(frame, (judge_zone_bottom[0] - zone_size, judge_zone_bottom[1] - zone_size),
                  (judge_zone_bottom[0] + zone_size, judge_zone_bottom[1] + zone_size), (255, 0, 0), 2)
    return frame


def mouse_callback(event, x, y, flags, param):
    global dragging_zone
    if event == cv2.EVENT_LBUTTONDOWN:
        if is_in_zone(x, y, judge_zone_top):
            dragging_zone = 'top'
        elif is_in_zone(x, y, judge_zone_bottom):
            dragging_zone = 'bottom'
    elif event == cv2.EVENT_MOUSEMOVE and dragging_zone:
        if dragging_zone == 'top':
            judge_zone_top[0], judge_zone_top[1] = x, y
        elif dragging_zone == 'bottom':
            judge_zone_bottom[0], judge_zone_bottom[1] = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        dragging_zone = None


def main():
    cv2.namedWindow("MuseDash AutoPlayer")
    cv2.setMouseCallback("MuseDash AutoPlayer", mouse_callback)

    while True:
        bbox = get_musedash_window_bbox()
        if bbox is None:
            print("❌ 未找到 MuseDash 窗口")
            break

        frame = capture_window(bbox)
        if frame is None:
            continue

        frame = process_frame(frame)
        frame = draw_zones(frame)

        cv2.imshow("MuseDash AutoPlayer", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
