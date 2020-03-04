from matplotlib import pyplot as plt
import numpy as np

fig, ax = plt.subplots()
ax.plot(np.random.random(10), 'o', picker=5)  # 5 points tolerance
text = ax.text(0.5, 0.5, 'event', ha='center', va='center', fontdict={'size': 20})


def on_pick(event):
    line = event.artist
    xdata, ydata = line.get_data()
    ind = event.ind
    print('on pick line:', ind, np.array([xdata[ind], ydata[ind]]).T)
    info = "Name={};button={};\n(x,y):{},{}(Dx,Dy):{:3.2f},{:3.2f}".format(
        event.name, event.button, event.x, event.y, event.xdata, event.ydata)
    text.set_text(info)


cid = fig.canvas.mpl_connect('pick_event', on_pick)
plt.show()