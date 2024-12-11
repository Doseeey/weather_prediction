from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5 import QtCore, QtWidgets, uic, QtGui
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import sys
from SimulationPlot import SimulationPlot

from ridge_predict import make_prediction
from lstm_predict import make_prediction_lstm
import datetime

class WeatherPredictionWindow(QtWidgets.QMainWindow):

    def __init__(self) -> None:
        QtWidgets.QMainWindow.__init__(self)
        self.ui = uic.loadUi('weather_prediction.ui', self)
        self.ui.textEdit.setPlainText("Highest temperature occured at:\n    Date:\n    Temperature:\n\nLowest temperature occured at:\n    Date:\n    Temperature:")

        self.simulationPlot = SimulationPlot()

        self.ui.weather_prediction_grid.addWidget(self.simulationPlot.canvas, 0, 0, 1, 1)

        self.ui.start_prediction.clicked.connect(self.startPrediction)
        self.ui.reset_prediction.clicked.connect(self.resetPrediction)

        self.ui.reset_prediction.setDisabled(True)

    def startPrediction(self) -> None:
        model = str(self.ui.comboBox.currentText())
        days = int(self.ui.number_days.currentText())


        self.simulationPlot.stylePlot(days)

        if model == "Ridge":
            Y = make_prediction(["kier_wiatr", "temp_grunt", "suma_opad_doba", "sr_pred_wiatr", "wilg"], days, 0)
        elif model == "LSTM":
            Y = make_prediction_lstm(["kier_wiatr", "temp_grunt", "suma_opad_doba", "sr_pred_wiatr", "wilg"], days, 0)

        self.updateText(Y)


        self.simulationPlot.data_length = days

        self.simulationPlot.setPlotData(Y)

        self.ui.start_prediction.setDisabled(True)
        self.ui.reset_prediction.setDisabled(False)

    def resetPrediction(self) -> None:
        self.simulationPlot.resetPlot()

        self.ui.start_prediction.setDisabled(False)
        self.ui.reset_prediction.setDisabled(True)

    def updateText(self, values = [0]) -> None:

        low = min(values)
        high = max(values)
        low_index = list(values).index(low)
        high_index = list(values).index(high)

        low_time = datetime.datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)+datetime.timedelta(days=low_index)
        high_time = datetime.datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)+datetime.timedelta(days=high_index)

        info = f"Highest temperature occured at:\n    Date: {high_time.strftime('%Y-%m-%d')}\n    Temperature: {round(high, 1)} [\u00B0C]\n\nLowest temperature occured at:\n    Date: {low_time.strftime('%Y-%m-%d')}\n    Temperature: {round(low, 1)} [\u00B0C]"
        text: str = self.ui.textEdit.setPlainText(info)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = WeatherPredictionWindow()
    mainWindow.show()
    sys.exit(app.exec_())