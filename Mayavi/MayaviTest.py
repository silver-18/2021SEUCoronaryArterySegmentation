from mayavi import mlab
import numpy as np
from matplotlib import pyplot as plt

# import tvtk

# raw = np.fromfile(file="Data/image/1.raw", dtype=np.float32)
# label = np.fromfile(file="Data/label/1.raw", dtype=np.float32)
# shape = [z,y,x]
# shape = np.load("Data/npy/1.npy")
# raw = raw.reshape(shape[0], shape[1], shape[2])
# label = label.reshape(shape[0], shape[1], shape[2])
# raw = np.where(raw > np.mean(raw), 1, 0)
# pass
# raw.tofile("Data/predict/raw.raw")
# mean = np.average(raw)
# heart = np.where(raw > mean, raw, mean)
# lung = np.where(raw < mean, raw, mean)
# mlab.figure(figure=mlab.gcf(), bgcolor=(1, 1, 1), size=(1920, 1080))
# mlab.contour3d(heart, color=(1, 0, 0), opacity=0.75)
# mlab.contour3d(lung, color=(0.5, 0.5, 0.5), opacity=0.25)
# obj = mlab.contour3d(raw, colormap="autumn", opacity=0.5, name="label")
# obj.visible = False
# vol = mlab.pipeline.volume(mlab.pipeline.scalar_field(raw), name='3-d ultrasound ')
# mlab.volume_slice(raw, colormap='gray',
# plane_orientation='x_axes', slice_index=0)  # 设定x轴切面
# mlab.volume_slice(raw, colormap='gray',
#                   plane_orientation='y_axes', slice_index=0)  # 设定y轴切面
# mlab.volume_slice(raw, colormap='gray',
#                   plane_orientation='z_axes', slice_index=0)  # 设定z轴切面

# plt.imshow(raw[200, :, :],cmap="Greys")

# mlab.show()
# plt.show()
# print(raw)
# print(shape)

from matplotlib.lines import Line2D

from mayavi.api import OffScreenEngine

colors = [(255/255.0, 187/255.0, 120/255.0), (255/255.0, 127/255.0, 14/255.0), (174/255.0, 199/255.0, 232/255.0), (31/255.0, 119/255.0, 180/255.0)]

mlab.options.offscreen = True
e = OffScreenEngine()
e.start()
fig = mlab.figure(engine=e, size=(5000, 5000), fgcolor=(0, 0, 0), bgcolor=(1, 1, 1))
mlab.view(azimuth=45, elevation=0)

n = 500
d = np.linspace(0, np.pi * 8, n)

mlab.plot3d(d, np.cos(d ** 1.2), np.cos(d), color=(colors[0]))
mlab.plot3d(d, np.sin(d), np.sin(d), color=colors[1])
mlab.plot3d(d, np.sin(d) * np.cos(d), np.cos(d), color=colors[2])
mlab.plot3d(d, np.sin(d) ** 2, np.sin(d) ** 2, color=colors[3])

mlab.savefig("./example.png")

im = plt.imread("./example.png")
labels = ["One", "Two", "Three", "Four"]

elements = [Line2D([0], [0], label=l, color=c) for l, c in zip(labels, colors[:4])]
plt.imshow(im)
plt.legend(handles=elements)
plt.gca().set_axis_off()
plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                    hspace=0, wspace=0)
plt.margins(0, 0)
plt.savefig("./example.png")
