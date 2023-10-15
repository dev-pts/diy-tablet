import numpy as np

import Model
import Coord

pixel_pitch = 0.2

def to_pixel(x):
    return round(x / pixel_pitch, 1)

def from_pixel(x):
    return x * pixel_pitch

def recalc(dx, dy, dz, ry, rz):
    Model.recalc(dx, dy, dz, ry, rz)

    a = Coord.calc(Model.values)
    b = Coord.calc(Model.values2)
    c = b + (a - b) * 1.8

    a = Model.to_sensor(a)
    b = Model.to_sensor(b)
    c = Model.to_sensor(c)

    return a, b, c

def run(angle_min=0, angle_max=1, redraw=None):
    da = np.empty(angle_max - angle_min)
    db = np.empty(angle_max - angle_min)
    dc = np.empty(angle_max - angle_min)

    diff = np.empty(Model.a_spacing * 10)
    diff_a = np.empty(Model.a_spacing * 10)
    diff_b = np.empty(Model.a_spacing * 10)

    for angle in range(angle_min, angle_max, 1):
        for i in range(Model.a_spacing * 10):
            diff_a[i], diff_b[i], diff[i] = recalc(i / 10, 0, 0, angle, 0)
            if redraw:
                redraw(diff_a[i], diff_b[i], diff[i])

            x = Model.stylus_x

            diff[i] -= x
            diff_a[i] -= x
            diff_b[i] -= x

        result = np.max(diff) - np.min(diff)

        #if result > 0.3:
        #    sys.exit(-1)

        da[angle] = np.max(diff_a) - np.min(diff_a)
        db[angle] = np.max(diff_b) - np.min(diff_b)
        dc[angle] = result

    print(Model.a_width, Model.a_spacing, np.max(da), np.max(db), np.max(dc))
