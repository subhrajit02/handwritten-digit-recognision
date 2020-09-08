import cv2
import numpy as np
import imutils


def grid_ext(image):
    # Grayscale conversion
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Threshold
    thresh = cv2.Canny(gray, 40, 120)
    # Closing
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # cv2.imshow("thresh", thresh)
    # cv2.waitKey(0)

    # ROI Detection
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)                      # need to understand imutils.grab_contours
    max_c = max(cnts, key=cv2.contourArea)                  # largest contour/ req grid
    cnt = [list(pt[0]) for pt in max_c]                     # converts into list of contour points

    # a = image.copy()
    # cv2.drawContours(a, c, -1, (0, 255, 255), 2)
    # cv2.imshow('cnt', a)
    # cv2.waitKey(0)

    b_r = max(cnt, key=lambda x: x[0] + x[1])               # b_r = bottom right
    t_l = min(cnt, key=lambda x: x[0] + x[1])               # t_l = top left
    t_r = max(cnt, key=lambda x: x[0] - x[1])               # t_r = top right
    b_l = min(cnt, key=lambda x: x[0] - x[1])               # b_l = bottom left

    # print(b_l, b_r, t_l, t_r)

    b_r[0], b_r[1] = b_r[0] + 2, b_r[1] + 2
    b_l[0], b_l[1] = b_l[0] - 2, b_l[1] + 2
    t_r[0], t_r[1] = t_r[0] + 2, t_r[1] - 2
    t_l[0], t_l[1] = t_l[0] - 2, t_l[1] - 2

    asp_rat = (b_l[1] - t_l[1]) / (t_r[0] - t_l[0])
    # print('aspect ratio:', asp_rat)

    # print('corners:', b_l, b_r, t_l, t_r)

    w = 750
    h = int(round(w*asp_rat))

    pts1 = np.float32([t_l, t_r, b_l, b_r])                 # pts1: from shape
    pts2 = np.float32([[0, 0], [h, 0], [0, w], [h, w]])     # pts2: to shape
    morph = cv2.getPerspectiveTransform(pts1, pts2)
    fin_image = cv2.warpPerspective(image, morph, (h, w))
    return fin_image


if __name__ == '__main__':
    img = cv2.imread('sample1.jpg')
    img = grid_ext(img)

    cv2.imshow('image', img)
    cv2.waitKey(0)
