from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5 import QtCore, QtWidgets, uic, QtGui
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import datetime

class MplCanvas(FigureCanvas):
	def __init__(self, parent: Figure=None, width: float=4.5, height: float=2, dpi: int=100) -> None:
		fig = Figure(figsize=(width, height), dpi=dpi, facecolor='#F0F0F0')
		self.ax: plt.Axes = fig.add_subplot(111)
		super(MplCanvas, self).__init__(fig)
		fig.tight_layout()
		
class SimulationPlot:
    def __init__(self):
        self.canvas = MplCanvas()

        self._plot_ref = None

        #Display parameters
        self.data_length: int = 0

        self.stylePlot(10)

    def stylePlot(self, days):
        self.canvas.ax.grid(visible=True, linestyle='-', linewidth=0.5)
        self.canvas.ax.set_facecolor("#F0F0F0")
        self.canvas.figure.subplots_adjust(left=0.08, right=0.99)

        self.canvas.ax.tick_params(axis='y', colors='red', labelsize=10)
        self.canvas.ax.set_ylabel('Temperature', color='red')
        self.canvas.ax.set_ylim(-18, 30)
        self.canvas.ax.set_yticks(np.arange(-18, 30, 6))

        self.canvas.ax.tick_params(axis='x', colors='black', labelsize=10)
        self.canvas.ax.set_xlabel('Date', color='black')
        self.canvas.ax.set_xlim(datetime.datetime.now().replace(hour=0, minute=0, second=0, microsecond=0), datetime.datetime.now()+datetime.timedelta(days=days))
        self.canvas.ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%d/%m'))


        return self.canvas
    
    def setPlotData(self, Y):
        self.X_values = [datetime.datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)+datetime.timedelta(days=i) for i in range(self.data_length)]
        self.Y_values = Y

        self.updatePlot()
    
    def updatePlot(self):
        # self.canvas.ax.set_xlim(X_values[0], X_values[-1])
        if self._plot_ref == None:
            self._plot_ref = self.canvas.ax.plot(self.X_values, self.Y_values[:self.data_length])[0]
        else:
            self._plot_ref.set_xdata(self.X_values)
            self._plot_ref.set_ydata(self.Y_values[:self.data_length])
        
        self.canvas.draw()

    def resetPlot(self):
        self.X_values = []
        self.Y_values = []

        self.updatePlot()