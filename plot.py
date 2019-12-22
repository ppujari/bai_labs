import io
import random
from flask import Response, Flask
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.widgets import Cursor

def plot_png(x, y):
    fig = create_figure(x, y)
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    print('am')
    return Response(output.getvalue(), mimetype='image/png')

def create_figure(x, y):
    # fig = Figure()
    # axis = fig.add_subplot(1, 1, 1)
    # xs = range(100)
    # ys = [random.randint(1, 50) for x in xs]
    # axis.plot(xs, ys)
    
    sns.set(style="dark", rc={"lines.linewidth": 1})
    # sns.set_style("darkgrid")
    fig, ax1 = plt.subplots(figsize=(20,5))
    pos = []
    def onclick(event):
      pos.append([event.xdata,event.ydata])
    fig.canvas.mpl_connect('button_press_event', onclick)
    # f.show()
    # plt.grid(true)
    # sns.set()
    
    sns.barplot(x=x,
                y=y, 
                color='#004488',
                ax=ax1)
    sns.lineplot(x=x, 
                y=y,
                color='r',
                marker="o",
                ax=ax1)
    # print('amiya', fig)
    fig.set()
    t = np.arange(0.0, 1.0, 0.01)
    s = np.sin(2*2*np.pi*t)
    cursor = SnaptoCursor(ax1, t, s)
    cid =  plt.connect('motion_notify_event', cursor.mouse_move)
    print("ddd")
    print(cid)
    return fig

class SnaptoCursor(object):
    def __init__(self, ax, x, y):
        self.ax = ax
        self.ly = ax.axvline(color='k', alpha=0.2)  # the vert line
        self.marker, = ax.plot([0],[0], marker="o", color="crimson", zorder=3) 
        self.x = x
        self.y = y
        self.txt = ax.text(0.7, 0.9, '')

    def mouse_move(self, event):
        if not event.inaxes: return
        x, y = event.xdata, event.ydata
        indx = np.searchsorted(self.x, [x])[0]
        x = self.x[indx]
        y = self.y[indx]
        self.ly.set_xdata(x)
        self.marker.set_data([x],[y])
        self.txt.set_text('x=%1.2f, y=%1.2f' % (x, y))
        self.txt.set_position((x,y))
        self.ax.figure.canvas.draw_idle()



