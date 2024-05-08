# PySide6 Graph plot window
# [matplotlib]
# fake stream data

import sys
import time
import asyncio
from requests import Session
import json

from PySide6.QtWidgets import QApplication, QMainWindow

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas,self).__init__(fig)


class GraphScreen(QMainWindow):
    def __init__(self, url):
        super(GraphScreen,self).__init__()
        self.canvas = MplCanvas(self, width=5, height=4, dpi=100)
        self.setCentralWidget(self.canvas)
        
        self.url = url
        self.time = list(range(100))
        self.value = list(range(100))
        self.show()
        asyncio.run(self.receive_data())
    
    async def receive_data(self):
        session = Session()
        with session.get(self.url, headers=None, stream=True) as res:
            for data in res.iter_lines():
                event = json.loads(data)
                # print(f"time : {event["time"]},  value : {event["value"]}")
                graph_data = dict(event)
                time.sleep(0.9)
                await self.update_graph(graph_data)

    async def update_graph(self, graph_data):
        self.time = self.time[1:]
        self.time.append(self.time[-1]+1)
        self.value = self.value[1:]
        self.value.append(graph_data['value'])
        self.canvas.axes.cla()
        self.canvas.axes.plot(self.time, self.value, 'r')
        self.canvas.draw()
        asyncio.sleep(1)
        
    def start(self):
        asyncio.run(self.receive_data())
        self.show()

if __name__ == "__main__":
    url = 'http://127.0.0.1:8000/NN01/fakeStream'
    app = QApplication(sys.argv)
    w = GraphScreen(url=url)
    app.exec()