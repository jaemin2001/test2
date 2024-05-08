# 2024.03.10 a-1.0 basic
import sys
import os
from PySide6 import QtCore
from PySide6.QtCore import QFile, QIODevice 
from PySide6.QtWidgets import (QApplication, QWidget, QFileDialog)
from PySide6.QtUiTools import QUiLoader

from .GraphScreen import GraphScreen

class DataViewer(QWidget):
    __MAX_WIN = 1
    __INST_created = 0
    
    def __new__(cls):
        if (cls.__INST_created > cls.__MAX_WIN):
            raise ValueError("Cannot create more objects")
        cls.__INST_created += 1
        return super().__new__(cls)
    
    def __init__(self):
        super(DataViewer, self).__init__()
        self.window = self.SetupUI()
        self.file_navi = QFileDialog()
        self.graph = GraphScreen()
        self.window.setWindowTitle('Data Viewer')
        self.window.Do.clicked.connect(self.Input_path)
        self.window.DataShow.clicked.connect(self.DataPlot)
        self.window.navi_file_path.clicked.connect(self.file_exeplore)
        self.window.show()
        
    def SetupUI(self):
        ui_file_name = "./srcUI/example01.ui"
        ui_file = QFile(ui_file_name)
        if not ui_file.open(QIODevice.ReadOnly):
            print(f"Cannot open {ui_file_name}: {ui_file.errorString()}")
            sys.exit(-1)
        loader = QUiLoader()
        window =loader.load(ui_file)
        ui_file.close()
        if not window:
            print(loader.errorString())
            sys.exit(-1)
        return window
            
    @QtCore.Slot()
    def Input_path(self):
        edit_path = self.window.File_Path_Edit.text()
        print(f"Enter Path: {edit_path}")
        self.window.file_path_print.setText(edit_path)
        
    @QtCore.Slot()
    def file_exeplore(self):
        # self.file_navi.show()
        navi_path = self.file_navi.getOpenFileName(None, "Select File")[0]
        print(f"Navi Path: {navi_path}")
        self.window.File_Path_Edit.setText(navi_path)
        self.window.file_path_print.setText(navi_path)
    
    @QtCore.Slot()
    def DataPlot(self):
        self.graph.show()
        
   
if __name__ == "__main__":
    app = QApplication(sys.argv)
    view = DataViewer()
    sys.exit(app.exec())
