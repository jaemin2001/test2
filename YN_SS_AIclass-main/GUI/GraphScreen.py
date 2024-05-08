# PySide6 Graph plot window
# [pyqtgraph](https://www.pyqtgraph.org/)
# sensor data example

from PySide6 import QtGui
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QDialog
import pyqtgraph as pg
import sys
from random import randint


class GraphScreen(QMainWindow):
    
    def __init__(self):
        super(GraphScreen,self).__init__()
        self.graphWidget = pg.PlotWidget()
        self.setCentralWidget(self.graphWidget)
        self.SetupUI()
    
    def SetupUI(self):
    
        # self.color = self.palette().color(QtGui.QPalette.Window)
        
        self.x = list(range(100))
        self.y = [randint(0,50) for _ in range(100)]
        
        # plot data: x, y values
        self.graphWidget.setBackground('w')
        # self.graphWidget.setBackground((100,50,255,25))
        # self.graphWidget.setBackground(self.color)
        pen = pg.mkPen(
                color=(255,0,0),
                width=3,
                # style=Qt.DashLine,
                ) 
        # styles = {'color':'r', 'font-size':'20px'}
        # self.graphWidget.setLabel('left','Temperature (℃)', **styles)
        # self.graphWidget.setLabel('bottom', 'Hour (H)', **styles)
        self.graphWidget.setLabel('left', "<span style=\"color:black;font-size:20px\">Temperature (°C)</span>")
        self.graphWidget.setLabel('bottom', "<span style=\"color:black;font-size:20px\">Hour (H)</span>")
        self.graphWidget.addLegend()
        self.graphWidget.showGrid(x=True,y=True)
        # self.graphWidget.setXRange(0,105,padding=0)
        self.graphWidget.setYRange(-5,55,padding=0)
        self.graphWidget.setTitle(
                "Your test plot & Title Here", 
                color="black", 
                size="15pt"
                )
        
        self.data_line = self.graphWidget.plot(self.x, self.y, pen=pen)
        
        self.timer = QTimer()
        self.timer.setInterval(100)
        self.timer.timeout.connect(self.update_plot_data)
        self.timer.start()
        
        # self.plot(hour, temperature_1, "Sensor1", 'r')
        # self.plot(hour, temperature_2, "Sensor2", 'b')
      
        # self.show()
        
    def plot(self, x, y , plotname, color):
        pen = pg.mkPen(color=color)
        self.graphWidget.plot()
        self.graphWidget.plot(
                x, 
                y,
                name=plotname, 
                pen=pen,                 
                symbol='+',
                symbolSize=15,
                symbolBrush=(color)
                )
        
    def update_plot_data(self):
        self.x = self.x[1:]
        self.x.append(self.x[-1]+1)
        
        self.y = self.y[1:]
        self.y.append(randint(0,50))
        
        self.data_line.setData(self.x, self.y)
                
if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = GraphScreen()
    w.show()
    app.exec()