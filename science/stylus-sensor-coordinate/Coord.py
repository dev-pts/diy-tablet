import numpy as np

def calc(values):
    argmax = np.argmax(values)

    if argmax < 1 or argmax > len(values) - 2:
        return 0

    if False:
        # Circle through 3 points
        x, y, z = complex(-10000, values[argmax - 1]), complex(0, values[argmax]), complex(10000, values[argmax + 1])
        w = z - x
        w /= y - x
        c = (x - y) * (w - abs(w)**2) / 2j / w.imag - x
        t = -c.real / 5000
    elif True:
        # Normal intersection
        A1 = np.array([-10000, values[argmax - 1]])
        A2 = np.array([0, values[argmax]])
        A3 = np.array([10000, values[argmax + 1]])

        # normals
        R1 = A2 - A1
        R2 = A2 - A3

        R1[0], R1[1] = R1[1], -R1[0]
        R2[0], R2[1] = -R2[1], R2[0]

        # offset
        C1 = (A1 + A2) / 2
        C2 = (A2 + A3) / 2

        # normals intersection from the centers of each segment
        t2 = (R1[1] * (C1[0] - C2[0]) + R1[0] * (C2[1] - C1[1])) / (R1[1] * R2[0] - R1[0] * R2[1])
        result = C2[0] + R2[0] * t2

        t = result / 5000
    elif False:
        # Parabola
        p1 = (-1, values[argmax - 1])
        p2 = (0, values[argmax])
        p3 = (1, values[argmax + 1])

        t = p1[1] * (p2[0] + p3[0]) * (p2[0] - p3[0]) - p2[1] * (p1[0] + p3[0]) * (p1[0] - p3[0]) + p3[1] * (p1[0] + p2[0]) * (p1[0] - p2[0])
        t /= p1[1] * (p2[0] - p3[0]) - p2[1] * (p1[0] - p3[0]) + p3[1] * (p1[0] - p2[0])

    return (2 * argmax + t) / 2
