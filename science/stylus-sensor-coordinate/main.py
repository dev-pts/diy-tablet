import pyvista as pv
import math
import numpy as np

import Model
import Test

# State variables
true_coord_x = 0

pstylus_x = 0
pstylus_y = 0
pstylus_z = 0

rotvec = None
angle = 0

protvec = None
pangle = 0

strength_y = -10

do_test = False

points = []
points2 = []

class Ball(object):
    def __init__(self, pl, pv, center, color):
        self.obj = pv.Sphere(center=center)
        pl.add_mesh(self.obj, color=color)

    def translate(self, v):
        self.obj.translate(v, inplace=True)

    def moveto_x(self, x):
        self.translate([x - self.obj.center[0], 0, 0])

    def moveto_y(self, y):
        self.translate([0, y - self.obj.center[1], 0])

    def moveto_z(self, z):
        self.translate([0, 0, z - self.obj.center[2]])

    def moveto(self, v):
        self.translate(v - self.obj.center)

def draw_a(index):
    height = index + 1
    antenna = pv.MultipleLines([
        [Model.a_offset + Model.a_a[index], 0.0, Model.sensor_z],
        [Model.a_offset + Model.a_a[index], 0.0, Model.sensor_z - height],
        [Model.a_offset + Model.a_a[index] + Model.a_width, 0.0, Model.sensor_z - height],
        [Model.a_offset + Model.a_a[index] + Model.a_width, 0.0, Model.sensor_z],
    ])
    pl.add_mesh(antenna, color='k', line_width=2)
    t = pv.Triangle([
        [Model.a_offset + Model.a_a[index], 0.0, Model.sensor_z],
        [Model.a_offset + Model.a_a[index] - 0.3, 0.0, Model.sensor_z - 0.5],
        [Model.a_offset + Model.a_a[index] + 0.3, 0.0, Model.sensor_z - 0.5],
    ])
    pl.add_mesh(t, color='k')

def redraw_levels(points, p_offset, values):
    for i in range(len(points)):
        points[i].points = [
            (p_offset + Model.a_offset + Model.a_a[i] + Model.a_width / 2, strength_y, Model.sensor_z),
            (p_offset + Model.a_offset + Model.a_a[i] + Model.a_width / 2, strength_y, Model.sensor_z + values[i] / Model.adc_swing * 30),
        ]

def redraw(a, b, c):
    global pstylus_x, pstylus_y, pstylus_z
    global protvec, pangle
    global true_coord_x

    true_coord_x = int(round(Model.stylus_x * 10, 1))

    rotvec = Model.orientation.as_rotvec(degrees=True)
    angle = np.linalg.norm(rotvec)

    pivot = [pstylus_x, pstylus_y, Model.lcd_z + pstylus_z]

    if protvec is not None:
        cylinder.rotate_vector(protvec, -pangle, point=pivot, inplace=True)
        cylinder2.rotate_vector(protvec, -pangle, point=pivot, inplace=True)
        tip.rotate_vector(protvec, -pangle, point=pivot, inplace=True)

    cylinder.rotate_vector(rotvec, angle, point=pivot, inplace=True)
    cylinder2.rotate_vector(rotvec, angle, point=pivot, inplace=True)
    tip.rotate_vector(rotvec, angle, point=pivot, inplace=True)

    dx = Model.stylus_x - pstylus_x
    dy = Model.stylus_y - pstylus_y
    dz = Model.stylus_z - pstylus_z

    if dx != 0 or dy != 0 or dz != 0:
        cylinder.translate([dx, dy, dz], inplace=True)
        cylinder2.translate([dx, dy, dz], inplace=True)
        tip.translate([dx, dy, dz], inplace=True)
        true_dot.translate([dx, dy, 0])

    pstylus_x = Model.stylus_x
    pstylus_y = Model.stylus_y
    pstylus_z = Model.stylus_z

    protvec = rotvec
    pangle = angle

    #pl.camera.focal_point = (Model.stylus_x, 0, Model.stylus_z)
    #pl.camera.position = (Model.stylus_x, -100, Model.stylus_z)

    redraw_levels(points, -0.3, Model.values)
    redraw_levels(points2, 0.3, Model.values2)

    mb_dot.moveto_x(a)
    mt_dot.moveto_x(b)
    m_dot.moveto_x(c)

    print(Test.to_pixel(a), Test.to_pixel(b), Test.to_pixel(c), Test.to_pixel(Model.stylus_x))

    pl.render()

def recalc_redraw(dx, dy, dz, ry, rz):
    a, b, c = Test.recalc(dx, dy, dz, ry, rz)
    redraw(a, b, c)
    return a, b, c

def callback_up():
    if Model.stylus_z >= Model.stylus_limit_z_max:
        return
    recalc_redraw(Model.stylus_x, Model.stylus_y, Model.stylus_z + 0.1, Model.stylus_ry, Model.stylus_rz)

def callback_down():
    if Model.stylus_z <= Model.stylus_limit_z_min:
        return
    recalc_redraw(Model.stylus_x, Model.stylus_y, Model.stylus_z - 0.1, Model.stylus_ry, Model.stylus_rz)

def callback_left():
    recalc_redraw(Model.stylus_x - 0.1, Model.stylus_y, Model.stylus_z, Model.stylus_ry, Model.stylus_rz)

def callback_right():
    recalc_redraw(Model.stylus_x + 0.1, Model.stylus_y, Model.stylus_z, Model.stylus_ry, Model.stylus_rz)

def callback_ryp():
    recalc_redraw(Model.stylus_x, Model.stylus_y, Model.stylus_z, Model.stylus_ry + 1, Model.stylus_rz)

def callback_rym():
    recalc_redraw(Model.stylus_x, Model.stylus_y, Model.stylus_z, Model.stylus_ry - 1, Model.stylus_rz)

def callback_rzp():
    recalc_redraw(Model.stylus_x, Model.stylus_y, Model.stylus_z, Model.stylus_ry, Model.stylus_rz + 1)

def callback_rzm():
    recalc_redraw(Model.stylus_x, Model.stylus_y, Model.stylus_z, Model.stylus_ry, Model.stylus_rz - 1)

def callback_start_test():
    global do_test
    do_test = True
    pl.update()

def callback_test(p=None):
    global do_test

    if not do_test:
        return

    pl.clear_on_render_callbacks()

    Test.run(-60, 1, redraw)

    do_test = False
    pl.add_on_render_callback(callback_test)

def main():
    global pl
    global cylinder, cylinder2, tip
    global true_dot, mb_dot, mt_dot, m_dot

    Model.init()

    pl = pv.Plotter()

    # Prepare and show scene
    pl.camera.enable_parallel_projection()
    pl.camera.position = (0, -100, Model.lcd_z)
    pl.camera.focal_point = (0, 0, Model.lcd_z)
    pl.camera.clipping_range = (0, 10000)
    pl.camera.zoom(0.03)

    stylus_xyz = np.array([Model.stylus_x, Model.stylus_y, Model.lcd_z + Model.stylus_z])

    limit = pv.Line([-100, -(Model.a_Y - 1), 0], [100, -(Model.a_Y - 1), 0])
    pl.add_mesh(limit, color='k', line_width=2)
    limit = pv.Line([-100, Model.a_Y, 0], [100, Model.a_Y, 0])
    pl.add_mesh(limit, color='k', line_width=2)

    tip = pv.Line(stylus_xyz, stylus_xyz + Model.direction * Model.tip_length)
    pl.add_mesh(tip, color='g', line_width=2)

    # Add magnet to scene
    cylinder = pv.Cylinder(
        center=stylus_xyz + Model.direction * (Model.tip_length + Model.magnet_height / 2 + Model.magnet_gap),
        direction=Model.direction,
        radius=Model.magnet_radius,
        height=Model.magnet_height,
    )
    pl.add_mesh(cylinder)

    cylinder2 = pv.Cylinder(
        center=stylus_xyz + Model.direction * (Model.tip_length + Model.magnet_height + Model.magnet_gap + Model.magnet_height / 2 + Model.magnet2_gap),
        direction=Model.direction,
        radius=Model.magnet_radius,
        height=Model.magnet_height,
    )
    pl.add_mesh(cylinder2)

    # Add ball-indicators where coordinate is detected
    true_dot = Ball(pl, pv, stylus_xyz, 'k')
    mb_dot = Ball(pl, pv, stylus_xyz, 'r')
    mt_dot = Ball(pl, pv, stylus_xyz, 'g')
    m_dot = Ball(pl, pv, stylus_xyz, 'w')

    # Add magnet level indicators
    for i in range(Model.a_N):
        draw_a(i)

        t = pv.Line()
        pl.add_mesh(t, color='r', line_width=2)
        points.append(t)

        t = pv.Line()
        pl.add_mesh(t, color='g', line_width=2)
        points2.append(t)

    # Add boundaries
    floor = pv.Line((-1000, 0, Model.sensor_z), (1000, 0, Model.sensor_z))
    pl.add_mesh(floor, color='k', line_width=1)

    floor = pv.Line((-1000, 0, Model.lcd_z), (1000, 0, Model.lcd_z))
    pl.add_mesh(floor, color='k', line_width=1)

    # Add interactive keyboard events
    pl.add_key_event('t', callback_left)
    pl.add_key_event('y', callback_right)
    pl.add_key_event('u', callback_rym)
    pl.add_key_event('i', callback_ryp)
    pl.add_key_event('b', callback_rzp)
    pl.add_key_event('n', callback_rzm)
    pl.add_key_event('o', callback_up)
    pl.add_key_event('l', callback_down)
    pl.add_key_event('k', callback_start_test)

    pl.add_on_render_callback(callback_test)

    # Show the state
    recalc_redraw(0, 0, 0, 0, 0)

    #do_test = True
    #callback_test()

    pl.show()

if __name__ == "__main__":
    main()
