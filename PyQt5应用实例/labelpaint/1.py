from PyQt5 import QtCore, QtWidgets, QtGui
import cv2
import numpy as np

class MyLabel(QtWidgets.QLabel):
    def __init__(self, parent=None):
        super(MyLabel, self).__init__(parent)
        self.pixel_x = -1  # pixel coordinate
        self.pixel_y = -1
        self.image = None
        self.originimage = None
        self.maskimg = None
        self.kenelwidth = 20
        # self.setAlignment(QtCore.Qt.AlignTop)

    def paintEvent(self, QPaintEvent):
        QtWidgets.QLabel.paintEvent(self, QPaintEvent)
        qp = QtGui.QPainter()
        qp.begin(self)
        if self.pixel_x > 0 and self.pixel_y > 0:
            self.drawPoints(qp)
        qp.end()

    def drawPoints(self, qp):
        qp.setPen(QtGui.QPen(QtCore.Qt.red, 5))
        qp.drawPoint(self.pixel_x, self.pixel_y)

    def mousePressEvent(self, e):
        if e.button() == QtCore.Qt.LeftButton:
            self.pixel_x = int(e.x())
            self.pixel_y = int(e.y())
            self._setpixcelvalue(self.image, self.pixel_x, self.pixel_y)
            self._setimage(self.image)
            self.update()
        else:
            if self.maskimg is not None:
                resultimg = np.bitwise_and(self.originimage, self.maskimg)
                self._setimage(resultimg)


    def mouseMoveEvent(self, e: QtGui.QMouseEvent) -> None:
        self.pixel_x = int(e.x())
        self.pixel_y = int(e.y())
        self._setpixcelvalue(self.image, self.pixel_x, self.pixel_y)
        self._setimage(self.image)
        self.update()

    def _setpixcelvalue(self, img ,x, y):
        if img is not None and self.maskimg is not None:
            nleft = int(max(0, x - self.kenelwidth / 2))
            nright = int(min(self.image.shape[1] - 1, x + self.kenelwidth / 2))
            ntop = int(max(0, y - self.kenelwidth / 2))
            nbot = int(min(self.image.shape[0] - 1, y + self.kenelwidth / 2))
            if len(img.shape) == 3:
                img[ntop:nbot,nleft:nright] = [255, 255 ,255]
                self.maskimg[ntop:nbot,nleft:nright] = [255, 255 ,255]
            else:
                img[ntop:nbot,nleft:nright] = 255
                self.maskimg[ntop:nbot,nleft:nright] = 255

    def _setimage(self, img):
        if len(img.shape) == 3:
            showImage = QtGui.QImage(img.data, img.shape[1], img.shape[0], QtGui.QImage.Format_RGB888)
        else:
            showImage = QtGui.QImage(img.data, img.shape[1], img.shape[0], QtGui.QImage.Format_Grayscale8)
        self.setPixmap(QtGui.QPixmap.fromImage(showImage))

    def setImage(self, img: np.array):
        """外部调用，并且是重置模板图"""
        if len(img.shape) == 3:
            showImage = QtGui.QImage(img.data, img.shape[1], img.shape[0], QtGui.QImage.Format_RGB888)
        else:
            showImage = QtGui.QImage(img.data, img.shape[1], img.shape[0], QtGui.QImage.Format_Grayscale8)
        self.setPixmap(QtGui.QPixmap.fromImage(showImage))
        self.image = img
        self.originimage = np.copy(img)
        self.maskimg = np.zeros(self.image.shape, dtype=np.uint8)


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        # MainWindow.resize(862, 594)
        # MainWindow.setToolButtonStyle(QtCore.Qt.ToolButtonFollowStyle)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        MainWindow.setCentralWidget(self.centralwidget)
        self.centralwidget.setMouseTracking(True)
        self.label = MyLabel(self.centralwidget)
        # self.label = QtWidgets.QLabel(self.centralwidget)
        # self.label.setGeometry(QtCore.QRect(170, 40, 640, 480))
        # self.label.setObjectName("label")

class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        QtWidgets.QMainWindow.__init__(self, parent)
        self.setupUi(self)
        img = cv2.imread("0.bmp", 1)
        resizeimg = cv2.resize(img, (int(img.shape[1]/4), int(img.shape[0]/4)))
        self.label.setImage(resizeimg)






if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = MainWindow()
    MainWindow.show()
    sys.exit(app.exec_())