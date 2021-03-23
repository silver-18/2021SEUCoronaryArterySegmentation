from mayavi.core.ui.api import MayaviScene, MlabSceneModel, SceneEditor
from traitsui.api import View, Item
from traits.api import HasTraits, Instance
from PyQt5.QtWidgets import QApplication, QWidget, QFileDialog, QMessageBox
import numpy as np
import time
import os
from mayavi import mlab
import torch

from mainwidget import Ui_MainWidget
from Predict import predict

from Unet import UNet

os.environ['QT_API'] = 'pyqt5'


class Visualization(HasTraits):
    scene = Instance(MlabSceneModel, ())
    view = View(Item('scene', editor=SceneEditor(scene_class=MayaviScene),
                     show_label=False),
                resizable=True  # We need this to resize with the parent widget
                )


class MainWidget(QWidget, Ui_MainWidget):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.cwd = os.getcwd()  # 获取当前程序文件位置
        self.visualization = Visualization()
        self.mayaviWidget = self.visualization.edit_traits(
            parent=self, kind='subpanel').control
        self.mayaviWidget.setParent(self.groupBoxMayavi)
        self.verticalLayout.addWidget(self.mayaviWidget)
        self.visualization.scene.mlab.figure(figure=mlab.gcf(), bgcolor=(1, 1, 1))
        # flag
        self.isLabelDrawn = False
        self.isPredictDrawn = False
        self.isContourDrawn = False
        self.isSliceDrawn = False
        # data
        self.image = None
        self.background = None
        self.file_name = None
        # self.raw_mean = np.mean(self.raw)
        self.shape = None
        self.label = None
        self.LabelObj = None
        self.PredictObj = None
        self.Unet = UNet(n_channels=1, n_classes=1)
        self.Unet.load_state_dict(torch.load(os.path.join("Data/Evaluation.dat")))
        self.PredictResult = None
        # selectBtn connect
        self.LoadData.clicked.connect(self.slotChooseRawFile)
        # ViewBtn edit
        self.View3D.clicked.connect(self.slotView3D)
        self.ViewSlice.clicked.connect(self.slotViewSlice)
        self.ShowLabel.clicked.connect(self.slotViewLabel)
        self.HideLabel.clicked.connect(self.slotHideLabel)
        self.ShowPredict.clicked.connect(self.slotShowPredict)
        self.HidePredict.clicked.connect(self.slotHidePredict)
        self.SaveFig.clicked.connect(self.slotSaveFig)

    def slotChooseRawFile(self):
        # 选择文件
        fileName_image, fileType = QFileDialog.getOpenFileName(self,
                                                               "Choose File",
                                                               self.cwd + "/Data/image",  # 起始路径
                                                               "raw(*.raw)")  # 设置文件扩展名过滤,用双分号间隔
        # read into numpy
        if fileName_image != "":
            # load
            self.file_name = fileName_image.split("/")[-1].rstrip(".raw")
            self.shape = np.load(file="Data/npy/" + self.file_name + ".npy")
            self.image = np.fromfile(file="Data/image/" + self.file_name + ".raw", dtype=np.float32)
            self.background = np.fromfile(file="Data/background/" + self.file_name + ".raw", dtype=np.float32)
            self.label = np.fromfile(file="Data/label/" + self.file_name + ".raw", dtype=np.float32)
            # reshape
            self.image = self.image.reshape(self.shape[0], self.shape[1], self.shape[2])
            self.background = self.background.reshape(self.shape[0], self.shape[1], self.shape[2])
            self.label = self.label.reshape(self.shape[0], self.shape[1], self.shape[2])
            # update labels
            self.LabelValueX.setText(str(self.shape[2]))
            self.LabelValueY.setText(str(self.shape[1]))
            self.LabelValueZ.setText(str(self.shape[0]))
            self.ImagePath.setText("ImagePath:" + "Data/image/" + self.file_name + ".raw")
            self.BackgroundPath.setText("BackgroundPath:" + "Data/background/" + self.file_name + ".raw")
            self.LabelPath.setText("LabelPath: " + "Data/label/" + self.file_name + ".raw")
        else:
            QMessageBox.about(self, "LoadData", "No Raw File Selected!")

    def slotView3D(self):
        # mlab
        # close render to speed up
        self.visualization.scene.disable_render = True
        self.visualization.scene.mlab.clf()
        self.visualization.scene.mlab.contour3d(self.background, color=(1, 1, 0), opacity=0.5)
        self.visualization.scene.disable_render = False
        # progressBar
        self.progressBar.reset()
        self.progressBar.isEnabled()
        # update flags
        self.isLabelDrawn = False
        self.isPredictDrawn = False
        self.isContourDrawn = True
        self.isSliceDrawn = False

    def slotViewSlice(self):
        self.visualization.scene.mlab.clf()
        mlab.volume_slice(self.background, colormap='gray',
                          plane_orientation='x_axes', slice_index=0)  # 设定x轴切面
        mlab.volume_slice(self.background, colormap='gray',
                          plane_orientation='y_axes', slice_index=0)  # 设定y轴切面
        mlab.volume_slice(self.background, colormap='gray',
                          plane_orientation='z_axes', slice_index=0)  # 设定z轴切面
        # update flags
        self.isSliceDrawn = True
        self.isContourDrawn = False

    def slotViewLabel(self):
        if not self.isLabelDrawn:
            self.LabelObj = self.visualization.scene.mlab.contour3d(self.label, color=(0, 0, 1), opacity=0.9)
            self.isLabelDrawn = True
        else:
            self.LabelObj.visible = True

    def slotHideLabel(self):
        self.LabelObj.visible = False

    def slotShowPredict(self):
        if not self.isPredictDrawn:
            self.PredictResult = predict(self.Unet, self.shape, self.image, 1)
            self.PredictObj = self.visualization.scene.mlab.contour3d(self.PredictResult, color=(1, 0, 0), opacity=0.9)
            self.isPredictDrawn = True
        else:
            self.PredictObj.visible = True

    def slotHidePredict(self):
        self.PredictObj.visible = False

    def slotSaveFig(self):
        if self.PredictResult is not None:
            self.PredictResult.tofile("Data/predict/" + self.file_name + ".raw")
            QMessageBox.about(self, "SaveFig",
                              "The raw has been saved at /Data/predict/" + self.file_name + ".raw successfully")
        else:
            QMessageBox.about(self, "SaveFig", "No result to save!")

    def slotProcessBar(self, percent):
        self.progressBar.setValue(percent)


if __name__ == "__main__":
    app = QApplication.instance()
    main_window = MainWidget()
    main_window.show()
    app.exec_()
