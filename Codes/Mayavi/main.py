from mayavi.core.ui.api import MayaviScene, MlabSceneModel, SceneEditor
from tvtk.pyface.api import Scene
from traitsui.api import View, Item
from traits.api import HasTraits, Instance
from PyQt5.QtWidgets import QApplication, QWidget, QFileDialog, QMessageBox, QColorDialog
from PyQt5.QtCore import QThread, pyqtSignal, QTimer
from mayavi import mlab
import numpy as np
import os
import torch
import time

from mainwidget import Ui_MainWidget
from Predict import predict
from Unet import UNet

os.environ['QT_API'] = 'pyqt5'


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
        # self.visualization.scene.mlab.figure(figure=mlab.gcf(), bgcolor=(1, 1, 1))
        self.visualization.scene.mlab.clf()
        # flag
        self.isLabelDrawn = False
        self.isPredictDrawn = False
        self.isContourDrawn = False
        self.isSliceDrawn = False
        # data
        self.image = None
        self.background = None
        self.file_name = None
        self.shape = None
        self.label = None
        # self.predict_color = None
        # self.label_color = None
        self.LabelObj = None
        self.PredictObj = None
        self.Unet = UNet(n_channels=1, n_classes=1)
        self.Unet.load_state_dict(torch.load(os.path.join("Data/Evaluation.dat")))
        self.PredictResult = None
        self.StateLabelDotCounter = 1
        # Timer
        # self.timer = QTimer()
        # self.timer.start(1000)
        # Thread edit
        # self.predictThread = ShowPredictThread()
        # self.View3DThread = View3DThread()
        # self.predictThread.state_signal.connect(self.slotStateLabel)
        # self.View3DThread.state_signal.connect(self.slotStateLabel)
        # selectBtn connect
        self.LoadData.clicked.connect(self.slotChooseRawFile)
        # ViewBtn connect
        self.View3D.clicked.connect(self.slotView3D)
        self.ViewSlice.clicked.connect(self.slotViewSlice)
        self.ShowLabel.clicked.connect(self.slotShowLabel)
        self.HideLabel.clicked.connect(self.slotHideLabel)
        self.ShowPredict.clicked.connect(self.slotShowPredict)
        self.HidePredict.clicked.connect(self.slotHidePredict)
        self.SaveFig.clicked.connect(self.slotSaveFig)
        # self.PredictColor.clicked.connect(self.slotPredictColor)
        # self.LabelColor.clicked.connect(self.slotLabelColor)

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
        time_start = time.time()
        self.StateLabel.setText("Drawing...")
        QApplication.processEvents()
        self.visualization.scene.mlab.clf()
        contour = self.visualization.scene.mlab.contour3d(self.background, color=(1, 1, 0), opacity=0.5)
        self.visualization.scene.mlab.axes(xlabel='x', ylabel='y', zlabel='z', line_width=4)
        time_end = time.time()
        self.StateLabel.setText("Drawing Done in " + str(round(time_end - time_start, 2)) + " sec!")
        # update flags
        self.isLabelDrawn = False
        self.isPredictDrawn = False
        self.isContourDrawn = True
        self.isSliceDrawn = False

    def slotViewSlice(self):
        self.visualization.scene.mlab.clf()
        mlab.volume_slice(self.image, colormap='gray',
                          plane_orientation='x_axes', slice_index=0)  # 设定x轴切面
        mlab.volume_slice(self.image, colormap='gray',
                          plane_orientation='y_axes', slice_index=0)  # 设定y轴切面
        mlab.volume_slice(self.image, colormap='gray',
                          plane_orientation='z_axes', slice_index=0)  # 设定z轴切面
        # update flags
        self.isSliceDrawn = True
        self.isContourDrawn = False

    def slotShowLabel(self):
        if not self.isLabelDrawn:
            time_start = time.time()
            self.StateLabel.setText("Drawing...")
            QApplication.processEvents()
            self.LabelObj = self.visualization.scene.mlab.contour3d(self.label, color=(0, 0, 1), opacity=0.9)
            time_end = time.time()
            self.StateLabel.setText("Drawing Done in " + str(round(time_end - time_start, 2)) + " sec!")

            # update flags
            self.isLabelDrawn = True
        else:
            self.LabelObj.visible = True

    def slotHideLabel(self):
        self.LabelObj.visible = False

    def slotShowPredict(self):
        if not self.isPredictDrawn:
            time_start = time.time()
            self.StateLabel.setText("Predicting...")
            QApplication.processEvents()
            self.PredictResult = predict(self.Unet, self.shape, self.image, 1)
            self.StateLabel.setText("Drawing...")
            QApplication.processEvents()
            self.PredictObj = self.visualization.scene.mlab.contour3d(self.PredictResult,
                                                                      color=(1, 0, 0),
                                                                      opacity=1)
            time_end = time.time()
            self.StateLabel.setText("Prediction & Drawing Done in " + str(round(time_end - time_start, 2)) + " sec!")
            # update flags
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

    def slotStateLabel(self, text):
        self.StateLabel.setText(text)
        QApplication.processEvents()

    def slotStateLabelUpdate(self):
        print("activate")
        self.StateLabelDotCounter = self.StateLabelDotCounter + 1
        if self.StateLabelDotCounter == 4:
            self.StateLabelDotCounter = 1
        label_text = self.StateLabel.text().strip('.')
        for i in range(self.StateLabelDotCounter):
            label_text = label_text + "."
        self.StateLabel.setText(label_text)

    # def slotPredictColor(self):
    #     self.predict_color = QColorDialog.getColor()
    #     print(self.predict_color)
    #
    # def slotLabelColor(self):
    #     pass


class Visualization(HasTraits):
    scene = Instance(MlabSceneModel, ())
    view = View(Item('scene', editor=SceneEditor(scene_class=Scene),
                     show_label=False),
                resizable=True  # We need this to resize with the parent widget
                )


# class ShowPredictThread(QThread):
#     state_signal = pyqtSignal(str)
#
#     def __init__(self):
#         super(ShowPredictThread, self).__init__()
#
#     def run(self):
#         main_window.ShowPredict.setDisabled(True)
#         main_window.timer.timeout.connect(main_window.slotStateLabelUpdate)
#         self.state_signal.emit("Predicting.3")
#         main_window.PredictResult = predict(main_window.Unet, main_window.shape, main_window.image, 1)
#         main_window.PredictObj = main_window.visualization.scene.mlab.contour3d(main_window.PredictResult,
#                                                                                 color=(1, 0, 0),
#                                                                                 opacity=0.9)
#         self.state_signal.emit("Prediction & Drawing Done!")
#         main_window.ShowPredict.setEnabled(True)
#         main_window.timer.stop()
#         main_window.timer.disconnect()
#         self.quit()
#         self.wait()


# class View3DThread(QThread):
#     state_signal = pyqtSignal(str)
#
#     def __init__(self):
#         super(View3DThread, self).__init__()
#
#     def run(self):
#         main_window.View3D.setDisabled(True)
#         main_window.timer.timeout.connect(main_window.slotStateLabelUpdate)
#         self.state_signal.emit("Drawing.")
#         main_window.visualization.scene.mlab.contour3d(main_window.background, color=(1, 1, 0), opacity=0.5)
#         self.state_signal.emit("Drawing Done!")
#         main_window.View3D.setEnabled(True)
#         main_window.timer.stop()
#         main_window.timer.disconnect()
#         self.quit()
#         self.wait()

# # Support Qss
# class QssReader:
#     def __init__(self):
#         pass
#
#     @staticmethod
#     def readQss(style):
#         with open(style, 'r') as f:
#             return f.read()


if __name__ == "__main__":
    app = QApplication.instance()
    main_window = MainWidget()
    # # qss
    # styleFile = "./qss/ElegantDark.qss"
    # qssStyle = QssReader.readQss(styleFile)
    # app.setStyleSheet(qssStyle)
    # main_window.setStyleSheet(qssStyle)
    main_window.show()
    app.exec_()
