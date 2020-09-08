import numpy as np


def intersection_bw_2_lines(l1, l2):
    """
    Returns point of intersection between 2 lines
    Parameters:
        l1 : line1
        l2 : line2
    Returns:
        x and y coordinate of point of intersection of l1 and l2
    """
    rho1, theta1 = l1
    rho2, theta2 = l2

    a = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([[rho1], [rho2]])

    x0, y0 = np.linalg.solve(a, b)
    x0, y0 = int(np.round(x0)), int(np.round(y0))

    return [x0, y0]


def cells_ext(h_lines, v_lines, start_r=0, start_c=0, end_r=0, end_c=0):
    """
    Extracts four corners of the cell from the horizontal and vertical lines POI
    :param h_lines: r, theta form horizontal lines
    :param v_lines: r, theta form vertical lines
    :param start_r: starting row number
    :param start_c: starting column number
    :param end_r: ending row number
    :param end_c: ending column number
    :return: final list containing the four corners of individual cells
    """
    if end_r == end_c == 0:
        end_c = len(v_lines) - 1
        end_r = len(h_lines) - 1
    ret_cell = []
    for i in range(0, len(h_lines) - 1):
        for j in range(0, len(v_lines) - 1):
            hl1, hl2 = h_lines[i], h_lines[i + 1]
            vl1, vl2 = v_lines[j], v_lines[j + 1]

            p1 = intersection_bw_2_lines(hl1, vl1)
            p2 = intersection_bw_2_lines(hl1, vl2)
            p3 = intersection_bw_2_lines(hl2, vl1)
            p4 = intersection_bw_2_lines(hl2, vl2)

            ret_cell.append([p1, p2, p3, p4])

    for i in range(len(ret_cell)):
        p1, p2, p3, p4 = ret_cell[i]
        p1 = (p1[0] + 2, p1[1] + 2)
        p2 = (p2[0] - 2, p2[1] + 2)
        p3 = (p3[0] + 2, p3[1] - 2)
        p4 = (p4[0] - 2, p4[1] - 2)
        ret_cell[i] = p1, p2, p3, p4

    ret_cell_fin = []

    for i in range(len(ret_cell)):
        if start_r <= (i / (len(v_lines) - 1)) <= (end_r + 1) and start_c <= (i % (len(v_lines) - 1)) <= (end_c+1):
            ret_cell_fin.append(ret_cell[i])

    return ret_cell_fin
