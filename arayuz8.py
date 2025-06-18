import cv2
import numpy as np
import time
from ultralytics import YOLO
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # GPU kontrolü
model = YOLO(r"C:\Users\omerb\OneDrive\Masaüstü\yolov11\runs\detect\train8\weights\best.pt").to(device)  # Modeli GPU'ya taşı

video_path = r"C:\Users\omerb\OneDrive\Masaüstü\yolov11\datasets\test\uav1.mp4"  # Video dosyasının tam yolu
cap = cv2.VideoCapture(video_path)

target_width = 640
target_height = 360

def resize_frame(frame, target_width, target_height):
    return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_LINEAR)

# Kilitlenme için zaman ve durum değişkenleri
lock_start_time = None
lock_duration = 4  # Kilitlenme için gereken süre (saniye)
successful_locks = 0  # Başarılı kilitlenme sayacı
tracking_start_time = None  # Takip süresi için sayaç
required_area_ratio = 0.05  # Hedef boyutunun minimum oranı (%5)
success_display_time = None  # Başarılı kilitlenme yazısının ekranda kalma süresi
lock_cooldown_start_time = None  # 1 saniyelik bekleme süresi için sayaç
lock_cooldown_duration = 1  # Bekleme süresi (saniye)


prev_time = time.time()
fps_list = []

def draw_target_area(frame):
    frame_height, frame_width, _ = frame.shape
    top = int(frame_height * 0.10)
    bottom = int(frame_height * 0.90)
    left = int(frame_width * 0.25)
    right = int(frame_width * 0.75)
    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 255), 2)  # Sarı alan
    return left, top, right, bottom

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = resize_frame(frame, target_width, target_height)

    current_time = time.time()
    fps = int(1 / (current_time - prev_time))
    prev_time = current_time
    fps_list.append(fps)

    raw_frame = frame.copy()

    target_left, target_top, target_right, target_bottom = draw_target_area(frame)
    frame_center = (target_width // 2, target_height // 2)
    cv2.circle(frame, frame_center, 5, (0, 0, 255), -1)

    cv2.putText(frame, "UFK", (target_width - 140, 55), cv2.FONT_HERSHEY_SIMPLEX, 2, (48, 117, 238), 8)
    cv2.putText(frame, "Tespit ve Takip", (target_width - 137, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (48, 117, 238), 2)
    cv2.putText(frame, "Sistemi", (target_width - 135, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (48, 117, 238), 4)
   
    results = model(raw_frame)
    

    highest_confidence = 0
    best_box = None
    for result in results:
        for box in result.boxes:
            conf = box.conf[0]
            if conf > highest_confidence:
                highest_confidence = conf
                best_box = box

    if best_box:
        x1, y1, x2, y2 = map(int, best_box.xyxy[0])
        object_center = ((x1 + x2) // 2, (y1 + y2) // 2)
        width = x2 - x1
        height = y2 - y1

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(frame, f"HEDEF TESPIT EDILDI", (10, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0, 255, 0), 1)
        cv2.putText(frame, f"Guven Puani: {highest_confidence:.2f}", (10, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0, 255, 0), 1)
        cv2.putText(frame, f"FPS: {fps}", (10, 70), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0, 255, 0), 1)

        relative_x = object_center[0] - frame_center[0]
        relative_y = object_center[1] - frame_center[1]
        cv2.putText(frame, f"Hedef Konumu: {relative_x},{relative_y}", (10, 110),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0, 255, 0), 1)

        width_ratio = width / target_width
        height_ratio = height / target_height
        cv2.putText(frame, f"Hedef Boyutu: W:{width_ratio:.0%},H:{height_ratio:.0%}", (10, 130),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0, 255, 0), 1)

        cv2.line(frame, frame_center, object_center, (0, 255, 0), 2, cv2.LINE_AA)

        print(f"Obje Merkezi: {object_center}")

        if (
            target_left < object_center[0] < target_right
            and target_top < object_center[1] < target_bottom
            and (width / target_width >= required_area_ratio or height / target_height >= required_area_ratio)
        ):
            if lock_cooldown_start_time and time.time() - lock_cooldown_start_time < lock_cooldown_duration:
                countdown_text = "BEKLENIYOR"
                cv2.putText(frame, countdown_text, ((target_left + target_right) // 2 - 100, target_top - 10),
                            cv2.FONT_HERSHEY_TRIPLEX, 0.8, (0, 255, 255), 1)
            else:
                if lock_start_time is None:
                    lock_start_time = time.time()
                    tracking_start_time = time.time()
                else:
                    elapsed_time = time.time() - lock_start_time
                    tracking_time = time.time() - tracking_start_time
                    countdown = lock_duration - elapsed_time
                    countdown_text = f"{max(countdown, 0):.0f}"
                    cv2.putText(frame, countdown_text, ((target_left + target_right) // 2, target_top - 10),
                                cv2.FONT_HERSHEY_TRIPLEX, 0.8, (0, 255, 255), 1)
                    cv2.putText(frame, f"Takip Suresi: {tracking_time:.2f}", (10, 170), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0, 255, 0), 1)
                    if elapsed_time >= lock_duration:
                        successful_locks += 1
                        success_display_time = time.time()  # Yazıyı gösterme zamanı
                        lock_start_time = None
                        lock_cooldown_start_time = time.time()  # Bekleme süresi başlat
        else:
            lock_start_time = None
            tracking_start_time = None

        if success_display_time and (time.time() - success_display_time <= 0.8):
            cv2.putText(frame, ">>>BASARILIYLA KILITLENILDI<<<", ((target_left + target_right) // 2 - 245, target_top + 25),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.2, (0, 255, 0), 1)
        elif success_display_time and (time.time() - success_display_time > 1):
            success_display_time = None  # Süre tamamlandıktan sonra sıfırla

        cv2.putText(frame, f"Basarili Kilitlenme: {successful_locks}", (10, 190), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0, 255, 0), 1)

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

if fps_list:
    avg_fps = sum(fps_list) / len(fps_list)
    print(f"Ortalama FPS: {avg_fps:.2f}")


cap.release()
cv2.destroyAllWindows()