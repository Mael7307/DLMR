import numpy as np


def blank(I):

    b = 0

    for i in range(0, 224, 8):
        for j in range(0, 224, 8):
            if np.std(I[i:i + 8, j:j + 8]) < 0.75:
                b += 1

    if b > 684:
        return False

    return True
