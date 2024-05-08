import sys
import time
import json
import csv
import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation

import asyncio
from requests import Session


# 2024.03.10 a-1.0 basic
# import os
# from PySide6 import QtCore
# from PySide6.QtCore import QFile, QIODevice 
# from PySide6.QtWidgets import (QApplication, QWidget, QFileDialog)
# from PySide6.QtUiTools import QUiLoader


# from GUI.DataView import DataViewer, GraphScreen

# Viewer = DataViewer()


class realTime_Graph:
    def __init__(self, url):
        self.url = url
        self.time = list(range(100))
        self.value = list(range(100))
        # self.fig, self.ax =  plt.subplots()
        # self.fig.canvas.mpl_connect('close_event', self.on_close_write)
        # self.ani = FuncAnimation(self.fig, self.update_graph, interval=10)
        plt.show()
        

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
        plt.clf()
        plt.plot(self.time, self.value)
        plt.xlabel('step')
        plt.ylabel('value')
        plt.title('fake-Stream data graph')
        plt.pause(0.1)
    
    # async def on_close_write(self):
    #     with open('recive_data.csv', 'w', newline='') as f:
    #         writer = csv.writer(f)
    #         writer.writerow(['time','value'])
    #         for t, v in zip(self.time, self.value):
    #             writer.writerow([t, v])

    def start(self):
        asyncio.run(self.receive_data())

if __name__ == '__main__':
    url = "http://127.0.0.1:8000/NN01/fakeStream"
    realtime = realTime_Graph(url)
    realtime.start()


    # app = QApplication(sys.argv)
    # view = Viewer()
    # sys.exit(app.exec())
