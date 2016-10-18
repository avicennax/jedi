
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np

xs = np.load("../data/random/force_sin_xs.npy")

fig = plt.figure(figsize=(7,3))
ax = plt.axes(xlim=(-4, 4), ylim=(-1, 1))
line, = ax.plot([], [], 'ro', alpha=.2)


# initialization function: plot the background of each frame
def init():
    line.set_data([], [])
    return line,

# animation function.  This is called sequentially
def animate(i):
    line.set_data(xs[0][i, :], np.tanh(xs[0][i, :]))
    return line,

# call the animator.  blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=300, interval=20)

# save the animation as an mp4.  This requires ffmpeg or mencoder to be
# installed.  The extra_args ensure that the x264    codec is used, so that
# the video can be embedded in html5.  You may need to adjust this for
# your system: for more information, see
# http://matplotlib.sourceforge.net/api/animation_api.html
anim.save('../media/rates.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

plt.show()

