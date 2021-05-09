cd C:\Users\chenbei\Documents\pythonProjects\PYQT5_UI文件夹\Demo12\signDialog
pyuic5  -o  signDialog.py   signDialog.ui

cd C:\Users\chenbei\Documents\pythonProjects\PYQT5_UI文件夹\Demo12
pyrcc5 .\signDialog\signDialog.qrc -o signDialog_rc.py


from MyPlatform import  signDialog_rc
