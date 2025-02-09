import magpylib as magpy
import numpy as np
from scipy.spatial.transform import Rotation as R
import random

# Stylus parameters
magnet_radius = 2.5
magnet_height = 11
magnet_strong = 20000

magnet_gap = 1
magnet2_gap = 1

tip_length = 10

stylus_limit_angle = 60
stylus_limit_z_min = 0
stylus_limit_z_max = 50

# ADC parameters
adc_swing = 4096
adc = True

# Antenna parameters
a_N = 20
a_width = 13
a_spacing = 2
a_Y = 60

lcd_z = 0
sensor_z = -1

# Constants
a_total = (a_N - 1) * a_spacing + a_width
a_offset = -a_total / 2 + a_spacing / 2

# State variables
stylus_x = 0
stylus_y = 0
stylus_z = 0

stylus_ry = 0
stylus_rz = 0

direction = np.array([0, 0, 1])
orientation = None

a_a = np.empty(a_N)

pixels = np.empty((a_N * a_Y * 2 * (a_width + 1), 3))

values = np.empty(a_N)
values2 = np.empty(a_N)

max_1 = 0
max_2 = 0

def init_antennas():
    for i in range(a_N):
        a_a[i] = i * a_spacing

    p_i = 0
    for i in range(a_N):
        for y in range(-(a_Y - 1), a_Y):
            for x in range(a_width + 1):
                pixels[p_i] = (a_offset + a_a[i] + x, y, sensor_z)
                p_i += 1

def to_sensor(x):
    return a_offset + a_width / 2 + x * a_spacing

def getone(values, pos, adc, norm):
    df = magpy.getB(
        sources='Cylinder',
        observers=pixels,
        polarization=(0, 0, magnet_strong),
        dimension=(magnet_radius * 2, magnet_height),
        position=pos,
        orientation=orientation,
    )

    idx = 0
    # Integrate over segment to get overall penetration
    for i in range(len(values)):
        # np.dot(df[idx:idx + (a_width + 1)], [0, 0, 1]) -> df[..., 2]
        s = (a_width + 1) * (a_Y * 2 - 1)
        values[i] = np.sum(df[idx:idx + s, 2]) / s
        idx += s

    if adc:
        # Emulate ADC results
        for i in range(len(values)):
            values[i] = int(round(values[i] / norm * adc_swing, 1)) + int(random.uniform(-2, 2))

def recalc(dx, dy, dz, ry, rz):
    global stylus_x, stylus_y, stylus_z

    stylus_x = dx
    stylus_y = dy
    stylus_z = dz

    global stylus_ry, stylus_rz

    stylus_ry = ry
    stylus_rz = rz

    if stylus_ry > stylus_limit_angle:
        stylus_ry = stylus_limit_angle
    if stylus_ry < -stylus_limit_angle:
        stylus_ry = -stylus_limit_angle

    global direction
    global orientation

    orientation = R.from_euler('yz', [stylus_ry, stylus_rz], degrees=True)
    direction = orientation.apply([0, 0, 1])
    direction = direction / np.linalg.norm(direction)

    stylus_xyz = np.array([stylus_x, stylus_y, lcd_z + stylus_z])

    getone(values, stylus_xyz + direction * (tip_length + magnet_height / 2 + magnet_gap), adc, max_1)
    getone(values2, stylus_xyz + direction * (tip_length + magnet_height + magnet_gap + magnet_height / 2 + magnet2_gap), adc, max_2)

def init():
    global adc, max_1, max_2

    init_antennas()

    # Find max signal for ADC emulating
    if adc:
        adc = False
        max_1 = 0
        max_2 = 0
        for i in range(a_spacing * 10):
            recalc(i / 10, 0, 0, 60, 0)

            a = np.max(values)
            b = np.max(values2)

            if max_1 < a:
                max_1 = a
            if max_2 < b:
                max_2 = b

        max_1 = int(round(max_1, 1))
        max_2 = int(round(max_2, 1))
        adc = True

    recalc(0, 0, 0, 0, 0)
