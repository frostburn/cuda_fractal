from subprocess import call, Popen, PIPE
from pylab import *
from time import clock

launches_per_update = 5000
width = 1000
height = 1000

# Recompile just in case.
call(["nvcc", "main.cu", "-Wno-deprecated-gpu-targets", "-Xcompiler", "-fopenmp"])

def accumulate():
    args = ["./a.out", launches_per_update, width, height]
    accumulator = Popen(map(str, args), stdout=PIPE)
    im_data, stderr_data = accumulator.communicate()
    im = eval(im_data).reshape(width, height)
    return im

then = clock()
ion()
im = zeros((width, height))
while True:
    now = clock()
    im += accumulate()
    print "{} seconds per update".format(clock() - now)
    foo = imshow(sqrt(im))
    print "{} samples per second".format(sum(im) / (clock() - then))
    pause(0.01)
