from __future__ import division
from subprocess import check_call, Popen, PIPE
from pylab import *
from scipy.misc import toimage
from time import time

NUM_COLORS = 3

launches_per_update = 2000
hd = True
if hd:
    width = 1920
    height = 1080
else:
    width = 426
    height = 240
aspect_ratio = width / height
iterations = 1000
spot_x = 0
spot_y = 1
spot_scale_x = 1
spot_scale_y = 1
center_x = 0
center_y = 1
proj_xx = 0.2
proj_xy = 0
proj_yx = 0
proj_yy = 0.2

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

NORM_FACTOR = launches_per_update / (width * height)
INIT_NORM = 8000000 * NORM_FACTOR
FINAL_NORM = 1000000 * NORM_FACTOR
NUM_FRAMES = 100
for i in range(NUM_FRAMES):
    then = time()
    t = i / NUM_FRAMES

    z = t * 5
    proj_xx = proj_yy = 0.2 * 2**z
    proj_xx /= aspect_ratio
    spot_scale_x = spot_scale_y = 0.2 / proj_yy
    norm = INIT_NORM + t * (FINAL_NORM - INIT_NORM)

    im = accumulate().copy()
    weight = im.sum()
    maximum = im.max()
    im /= norm
    im = sqrt(im)
    toimage(im, cmin=0, cmax=1).save("frames/out_%05d.tiff" % i)
    duration = time() - then
    print "Frame %05d done in %g seconds. %g weight per second. Maximum %g." % (i, duration, weight / duration, maximum)
