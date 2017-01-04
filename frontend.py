from subprocess import check_call, Popen, PIPE
from pylab import *
from time import time

NUM_COLORS = 3

launches_per_update = 100
width = 820
height = 820
iterations = 1000
spot_x = -0.6
spot_y = 0
spot_scale_x = 1
spot_scale_y = 1
center_x = -0.6
center_y = 0
proj_xx = 0.4
proj_xy = 0
proj_yx = 0
proj_yy = 0.4

precission = float32
# precission = float64

# Recompile just in case.
args = ["nvcc", "main.cu", "-Wno-deprecated-gpu-targets", "-Xcompiler", "-fopenmp"]
if precission == float64:
    args.extend(["-D", "DOUBLE_PREC"])
check_call(args)

im_shape = (height, width, NUM_COLORS)

def accumulate():
    args = [
        "./a.out",
        launches_per_update,
        width, height,
        iterations,
        spot_x, spot_y,
        spot_scale_x, spot_scale_y,
        center_x, center_y,
        proj_xx, proj_xy, proj_yx, proj_yy,
    ]
    accumulator = Popen(map(str, args), stdout=PIPE)
    im_data, stderr_data = accumulator.communicate()
    # Eval API
    # im = eval(im_data).reshape(width, height)
    im = frombuffer(im_data, precission).reshape(*im_shape)
    return im

def get_extent():
    return [
        center_x - 0.5 / proj_xx,
        center_x + 0.5 / proj_xx,
        center_y - 0.5 / proj_yy,
        center_y + 0.5 / proj_yy,
    ]

trajectory_x = None
trajectory_y = None

def onclick(event):
    print (
        'button=%d, x=%d, y=%d, xdata=%f, ydata=%f, key=%s' %
        (event.button, event.x, event.y, event.xdata, event.ydata, event.key)
    )
    global then
    global im
    global spot_x, spot_y
    global spot_scale_x, spot_scale_y
    global center_x, center_y
    global proj_xx, proj_yy
    global trajectory_x

    im = zeros(im_shape)
    then = time()
    trajectory_x = None

    if event.key != 'control':
        spot_x = event.xdata
        spot_y = event.ydata
    if event.key != 'shift':
        center_x = event.xdata
        center_y = event.ydata

    zoom_factor = 1.4
    if event.button == 1:
        if event.key != 'shift':
            proj_xx *= zoom_factor
            proj_yy *= zoom_factor
        if event.key != 'control':
            spot_scale_x /= zoom_factor
            spot_scale_y /= zoom_factor
    elif event.button == 3:
        if event.key != 'shift':
            proj_xx /= zoom_factor
            proj_yy /= zoom_factor
        if event.key != 'control':
            spot_scale_x *= zoom_factor
            spot_scale_y *= zoom_factor

def onkey(event):
    print ('key=%s' % (event.key,))

    if event.key != ' ':
        return

    global then
    global im
    global center_x, center_y
    global trajectory_x, trajectory_y

    im = zeros(im_shape)
    then = time()

    if trajectory_x is None:
        trajectory_x = center_x
        trajectory_y = center_y

    center_x, center_y = (
        center_x**2 - center_y**2 + trajectory_x,
        2 * center_x * center_y + trajectory_y,
    )

ion()
then = time()
im = zeros(im_shape)
extent = [-1, 1, -1, 1]
imshow(im, extent=get_extent())
fig = gcf()
fig.canvas.mpl_connect("button_press_event", onclick)
fig.canvas.mpl_connect("key_press_event", onkey)
while True:
    now = time()
    im += accumulate()
    print "{} seconds per update".format(time() - now)
    normalized = im.copy()
    normalized /= normalized.max()
    imshow(sqrt(normalized), extent=get_extent())
    print "{} weight per second".format(sum(im) / (time() - then))
    pause(0.01)
