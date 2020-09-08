import numpy as np
import cv2


def rem_multi_lines(lines, thresh):
    """
    to remove the multiple lines with close proximity 
    :param lines: initial list with all the lines(multiple in place of singular)
    :param thresh: dist between two lines for them to be considered as same
    :return: final list with singular lines in place of multiple
    """
    a = []
    i = 0
    lines.append([800, 0])                                  # random val/ noise
    out = []
    # this loop collects lines with close proximity in a list (a) and then appends that 
    # complete list in a common list called out.
    while i < len(lines) - 1:
        if lines[i] not in a:
            a.append(lines[i])
        if abs(lines[i + 1][0] - lines[i][0]) < thresh:
            a.append(lines[i + 1])
        else:
            out.append(a)
            a = []
        i += 1
    
    # print(out)

    final = []
    for i in out:
        a = np.array(i)
        final.append(np.average(a, axis=0))

    # print(final)

    for i in final.copy():
        if i[0] < 0:
            final.remove(i)
    return final


def draw_r_theta_lines(img, lines, color):
    """
    draw lines on image which are of (r, theta) form
    :param img: image to draw the lines on
    :param lines: list of lines on the form (r, theta)
    :param color: color of lines
    :return: 
    """
    for rho, theta in lines:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * a)
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * a)

        cv2.line(img, (x1, y1), (x2, y2), color, 2)


def lines_ext(img, hough_thresh, multilines_thresh):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 45, 10)

    line_image = img.copy()

    lines = cv2.HoughLines(edges, 1, np.pi / 180, hough_thresh)
    lines = lines.reshape(lines.shape[0], 2)

    draw_r_theta_lines(line_image, lines, (0, 0, 255))

    lines = sorted(lines, key=lambda x: x[0])

    cv2.imshow("lines", line_image)
    cv2.waitKey(0)

    l1 = list(lines)
    l2 = []
    for i in l1:
        l2.append(list(i))

    v_lines = []
    h_lines = []

    for i in l2:
        if round(i[1]) == 0:
            v_lines.append(i)
        elif round(i[1]) > 0.5:
            h_lines.append(i)

    # print('v:', v_lines)
    # print('h:', h_lines)

    v_lines = rem_multi_lines(v_lines, multilines_thresh)
    h_lines = rem_multi_lines(h_lines, multilines_thresh)

    final = v_lines + h_lines

    draw_r_theta_lines(line_image, final, (0, 255, 0))

    cv2.imshow("lines1", line_image)
    cv2.waitKey(0)

    return v_lines, h_lines
