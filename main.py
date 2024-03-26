import cv2 as cv
import numpy as np
import time

camera = cv.VideoCapture(0)
# Mavi için renk aralığı
lower_blue = np.array([90, 50, 50])
upper_blue = np.array([130, 255, 255])

# Kırmızı için renk aralığı
lower_red1 = np.array([0, 100, 100])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([160, 100, 100])
upper_red2 = np.array([180, 255, 255])

# Kamera ortasındaki çemberin parametreleri
radius = 75
height, width = 480, 640  # Varsayılan kamera çözünürlüğü
center_x = int(width / 2)
center_y = int(height / 2)
center_video = (center_x, center_y)

# Eşleşme için bekleme süresi
match_time = 5
start_time_blue = None
start_time_red = None
blue_detected = False
red_detected = False
kernel = np.ones((5, 5), np.uint8)

if camera.isOpened():
    while True:
        ret, frame = camera.read()

        if not ret:
            break
        
        # Video Orta Noktası Hesaplama
        cv.circle(frame, center_video, radius, (0, 255, 0), 2)

        # HSV renk uzayına dönüştür
        hsv=cv.cvtColor(frame, cv.COLOR_BGR2HSV)

        # Mavi nesneleri tespit et
        if not blue_detected:
            blue_mask = cv.inRange(hsv, lower_blue, upper_blue)
            # Mavi maskesi üzerinde morfolojik açma işlemi uygula
            blue_mask = cv.morphologyEx(blue_mask, cv.MORPH_OPEN, kernel)
            blue_contours, _ = cv.findContours(blue_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            
            for contour in blue_contours:
                # Mavi nesnelerin merkezini bul
                M = cv.moments(contour)

                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # Mavi nesnelerin alanını hesapla
                    area = cv.contourArea(contour)
                    
                    
                    # Alanı kontrol et ve belirli bir boyuttan büyükse işlem yap
                    if area > 800:
                        # Merkezi çemberin merkezine göre kontrol et
                        if (cx - center_x)**2 + (cy - center_y)**2 < radius**2:
                            cv.drawContours(frame, [contour], -1, (255, 0, 0), 2)
                            if start_time_blue is None:
                                start_time_blue = time.time()
                            elif time.time() - start_time_blue >= match_time:
                                print("Mavi kapak algılandı.")
                                blue_detected = True
                                break

        # Kırmızı nesneleri tespit et
        if not red_detected:
            red_mask1 = cv.inRange(hsv, lower_red1, upper_red1)
            red_mask2 = cv.inRange(hsv, lower_red2, upper_red2)
            # Kırmızı maskesi üzerinde morfolojik açma işlemi uygula
            red_mask = cv.bitwise_or(red_mask1, red_mask2)
            red_mask = cv.morphologyEx(red_mask, cv.MORPH_OPEN, kernel)
            red_contours, _ = cv.findContours(red_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

            for contour in red_contours:
                M = cv.moments(contour)  
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # Kırmızı nesnelerin alanını hesapla
                    area = cv.contourArea(contour)
                    
                    # Alanı kontrol et ve belirli bir boyuttan büyükse işlem yap
                    if area > 100:
                        # Merkezi çemberin merkezine göre kontrol et
                        if (cx - center_x)**2 + (cy - center_y)**2 < radius**2:
                            
                            cv.drawContours(frame, [contour], -1, (0, 0, 255), 2)
                            if start_time_red is None:
                                start_time_red = time.time()
                            elif time.time() - start_time_red >= match_time:
                                print("Kırmızı kapak algılandı.")
                                red_detected = True
                                break
                                        
        cv.imshow("Camera", frame)

        if cv.waitKey(1) & 0xFF == ord("q"):
            break

camera.release()
cv.destroyAllWindows()
