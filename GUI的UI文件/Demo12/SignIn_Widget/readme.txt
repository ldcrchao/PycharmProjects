cd C:\Users\chenbei\Documents\pythonProjects\PYQT5_UI文件夹\Demo12\SignIn_Widget
pyuic5  -o  signWindow.py   signIn_MainWindow.ui

cd C:\Users\chenbei\Documents\pythonProjects\PYQT5_UI文件夹\Demo12
pyrcc5 .\SignIn_Widget\signWindow.qrc -o signWindow_rc.py


from MyPlatform import  signWindow_rc
