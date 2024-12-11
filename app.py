from WeatherPredictionWindow import WeatherPredictionWindow
from PyQt5.QtWidgets import QApplication, QMainWindow
import sys

if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = WeatherPredictionWindow()
    #mainWindow.show()
    mainWindow.show()
    #mainWindow.showFullScreen()
    sys.exit(app.exec_())