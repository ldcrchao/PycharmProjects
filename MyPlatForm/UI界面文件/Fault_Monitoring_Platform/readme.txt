cd C:\Users\chenbei\Documents\pythonProjects\PYQT5_UI文件夹\Demo12\Fault_Monitoring_Platform
pyuic5  -o  faultPlatform.py   faultPlatform.ui

cd C:\Users\chenbei\Documents\pythonProjects\PYQT5_UI文件夹\Demo12
pyrcc5 .\Fault_Monitoring_Platform\fault_platform_icon.qrc -o fault_platform_rc.py

from PyQt5.QtChart import QChartView
from MyPlatform import  fault_platform_rc
