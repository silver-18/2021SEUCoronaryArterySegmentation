from mayavi import mlab
import numpy as np
from matplotlib import pyplot as plt

# import tvtk

raw = np.fromfile(file="Data/image/1.raw", dtype=np.float32)
label = np.fromfile(file="Data/label/1.raw", dtype=np.float32)
# shape = [z,y,x]
shape = np.load("Data/npy/1.npy")
raw = raw.reshape(shape[0], shape[1], shape[2])
label = label.reshape(shape[0], shape[1], shape[2])
raw = np.where(raw > np.mean(raw), 1, 0)
pass
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
