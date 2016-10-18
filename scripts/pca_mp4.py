import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np

pca_x = np.load("../data/random/pca_x_ff_noise.npy")

fig = plt.figure()
ax = plt.axes(xlim=(-20, 20), ylim=(-15,15))
ax.set_xlabel('PCA 1')
ax.set_ylabel('PCA 2')
ax.set_zlabel('PCA 3')
trace, = ax.plot([], [], 'b')
dot, = ax.plot([], [], 'ro')


# initialization function: plot the background of each frame
def init():
    dot.set_data([], [])
    return dot,

# animation function.  This is called sequentially
def animate(i):
    trace.set_data(pca_x[0][:i], pca_x[1][:i])
    dot.set_data(pca_x[0][i], pca_x[1][i])
    return trace,

# call the animator.  blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=1000, interval=20)

# save the animation as an mp4.  This requires ffmpeg or mencoder to be
# installed.  The extra_args ensure that the x264    codec is used, so that
# the video can be embedded in html5.  You may need to adjust this for
# your system: for more information, see
# http://matplotlib.sourceforge.net/api/animation_api.html
anim.save('../media/pca_ff.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

plt.show()