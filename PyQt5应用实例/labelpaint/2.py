from pyqtgraph.Qt import QtCore, QtGui
import numpy as np
import pyqtgraph as pg

pg.setConfigOptions(imageAxisOrder='row-major')

class newimgItem(pg.ImageItem):
    def __init__(self, image=None, **kargs):
        super(newimgItem, self).__init__(image, **kargs)
        self.maskimg = None


    def setImagekx(self, image=None, autoLevels=None, **kargs):
        self.maskimg = np.copy(image)
        self.setImage(image, autoLevels, **kargs)

    def drawAt(self, pos, ev=None):
        self._setpixcelvalue(self.image, int(pos.x()), int(pos.y()))
        self.setImage(self.image, autoLevels=False, autoDownsample=False)
    #
    #
    def _setpixcelvalue(self, img, x, y):
        kernelwidth = self.drawKernel.shape[0]
        # print (type(img), type(self.maskimg))
        if img is not None and self.maskimg is not None:
            nleft = int(max(0, x - kernelwidth / 2))
            nright = int(min(self.image.shape[1] - 1, x + kernelwidth / 2))
            ntop = int(max(0, y - kernelwidth / 2))
            nbot = int(min(self.image.shape[0] - 1, y + kernelwidth / 2))
            if len(img.shape) == 3:
                img[ntop:nbot, nleft:nright] = [255, 255, 255]
                self.maskimg[ntop:nbot, nleft:nright] = [255, 255, 255]
            else:
                img[ntop:nbot, nleft:nright] = 255
                self.maskimg[ntop:nbot, nleft:nright] = 255
    #
    # def mouseClickEvent(self, ev):
    #     pg.ImageItem.mouseClickEvent(self, ev)
    #     print ("here")

app = QtGui.QApplication([])

## Create window with GraphicsView widget
w = pg.GraphicsView()
w.show()
# w.resize(800,800)
w.setWindowTitle('pyqtgraph example: Draw')

view = pg.ViewBox()
w.setCentralItem(view)

## lock the aspect ratio
# view.setAspectLocked(True)

## Create image item
img = newimgItem(border='w')
view.addItem(img)

import cv2
readimg = cv2.imread("0.bmp", 0)
img.setImagekx(readimg, autoLevels=False, autoDownsample=False)


kern = np.ones((3, 3))
img.setDrawKernel(kern, mask=kern, center=(1,1), mode='set')

## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()