import cv2
import numpy as np
import math

noise = 3000

def preprocess_frame(frame):
    frame = cv2.flip(frame, 1)
    roi = frame[100:350, 100:350]
    cv2.rectangle(frame, (100, 100), (350, 350), (0, 0, 255), 0)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    return frame, roi, hsv

def create_mask(hsv):
    lower_skin = np.array([0, 20, 70], np.uint8)
    upper_skin = np.array([40, 255, 255], np.uint8)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=4)
    mask = cv2.GaussianBlur(mask, (5, 5), 100)
    return mask

def find_contours(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        return max(contours, key=lambda x: cv2.contourArea(x))
    return None

def calculate_defects(contour):
    epsilon = 0.0005 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    hull = cv2.convexHull(approx, returnPoints=False)
    defects = cv2.convexityDefects(approx, hull)
    return approx, defects

def count_defects(approx, defects, roi):
    defect_count = 0
    for i in range(defects.shape[0]):
        s, e, f, _ = defects[i, 0]
        start, end, far = tuple(approx[s][0]), tuple(approx[e][0]), tuple(approx[f][0])
        a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
        b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
        c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
        s = (a + b + c) / 2
        area = math.sqrt(s * (s - a) * (s - b) * (s - c))
        d = (2 * area) / a
        angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 57
        if angle <= 90 and d > 30:
            defect_count += 1
            cv2.circle(roi, far, 3, [255, 0, 0], 1)
        cv2.line(roi, start, end, [255, 0, 0], 2)
    return defect_count + 1

def display_gesture(frame, defect_count, contour_area, hull_area):
    font = cv2.FONT_HERSHEY_SIMPLEX
    area_ratio = ((hull_area - contour_area) / contour_area) * 100
    if defect_count == 1:
        if contour_area < noise:
            cv2.putText(frame, "Place your hand inside the box:", (0, 50), font, 1, (0, 0, 255), 3, cv2.LINE_AA)
        else:
            if area_ratio < 12:
                cv2.putText(frame, "0", (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
            elif area_ratio < 17.5:
                cv2.putText(frame, "Best of Luck", (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
            else:
                cv2.putText(frame, "1", (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
    elif defect_count == 2:
        if contour_area < noise:
            cv2.putText(frame, "Place your hand inside the box:", (0, 50), font, 1, (0, 0, 255), 3, cv2.LINE_AA)
        else:
            cv2.putText(frame, "2", (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
    elif defect_count == 3:
        cv2.putText(frame, "3", (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
    elif defect_count == 4:
        cv2.putText(frame, "4", (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
    elif defect_count == 5:
        cv2.putText(frame, "5", (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
    else:
        cv2.putText(frame, "hand not found.", (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)

def main():
    vid = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    while True:
        try:
            ret, frame = vid.read()
            frame, roi, hsv = preprocess_frame(frame)
            mask = create_mask(hsv)
            contour = find_contours(mask)
            if contour is not None:
                approx, defects = calculate_defects(contour)
                defect_count = count_defects(approx, defects, roi)
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                contour_area = cv2.contourArea(contour)
                display_gesture(frame, defect_count, contour_area, hull_area)
            cv2.imshow("mask", mask)
            cv2.imshow("frame", frame)
        except Exception as e:
            pass
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    vid.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()