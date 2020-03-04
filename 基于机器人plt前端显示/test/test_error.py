# import matplotlib.pyplot as plt
# import cv2

# #鼠标事件
# def OnClick(event):
#     global Coords1x,Coords1y
#     global Coords3x,Coords3y
#     #ax = plt.gca()
#     if event.button ==1:
#         Coords1x = event.xdata
#         Coords1y = event.ydata
#     if event.button == 3:
#         Coords3x = event.xdata
#         Coords3y = event.ydata
#
#
# def OnMouseMotion(event):
#     global Coords2x,Coords2y,x1,y1
#     if event.button == 1:
#         Coords2x = event.xdata
#         Coords2y = event.ydata
#         x1 = [Coords1x,Coords2x]
#         y1 = [Coords1y,Coords2y]
#         ax = plt.gca()
#         lines = ax.plot(x1,y1,picker = 20)
#         ax.figure.canvas.draw()
#         #删除之前的线条，进行更新
#         l = lines.pop(0)
#         l.remove()
#         del l
#     elif event.button == 3:
#         Coords4x = event.xdata
#         Coords4y = event.ydata
#         x2 = [Coords3x,Coords4x]
#         y2 = [Coords3y,Coords4y]
#         ax1 = plt.gca()
#         #lines = ax1.plot(x1,y1,picker = 5)
#         lines1 = ax1.plot(x2,y2,picker = 20)
#         ax1.figure.canvas.draw()
#         #删除之前的线条，进行更新
#         l = lines1.pop(0)
#         l.remove()
#         del l
#
#
# ##关联鼠标点击事件
# figurepath = "./map/6#_1.png"
# fig = plt.figure()
#
# fig.canvas.mpl_connect('button_press_event', OnClick)
# fig.canvas.mpl_connect('motion_notify_event', OnMouseMotion)

import matplotlib.pyplot as plt
import numpy as np

x=range(1,10)
y=[2*v for v in x]
print(x, y)
plt.plot(x, y)
pos=plt.ginput(1)
print(pos)