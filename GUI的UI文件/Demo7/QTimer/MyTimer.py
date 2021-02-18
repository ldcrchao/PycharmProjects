#%%
# -*- coding UTF-8 -*-
'''
@Project : python学习工程文件夹
@File : MyTimer.py
@Author : chenbei
@Date : 2020/10/5 20:13
'''
import sys
from PyQt5.QtWidgets import  QApplication ,QWidget
from PyQt5.QtCore import  QTime ,QTimer
from GUI.Demo.Demo5 import QTimer as Q
class MyQtimer(QWidget):
    def __init__(self,parent=None):
        super(MyQtimer, self).__init__(parent)
        self.ui = Q.Ui_QTimer()
        self.ui.setupUi(self)
        self.show()

        self.timer = QTimer() # 创建定时器,定时器不是显性的页面组件,只能程序中创建
        self.timer.stop() # 初始时刻是停止的
        self.timer.setInterval(1000) # 初始定义为1000ms定时周期
        # 表示LCD显示时间会每隔1s就跳动,如果设置为10s 不会影响计时,只是会10s才会跳动一次
        self.timer.timeout.connect(self.do_timer_timeout) # 定时器的信号是timeout().信号的状态可以被函数始终跟随
        # timeout()不中断时会一直连接显示器函数,可以让显示器保持跳动,如果在开始按钮内部设置会导致只能点击时显示,不能跟随变化
        self.counter = QTime() # 创建计时器

    def do_timer_timeout(self):
        '''中断信号用于控制显示时间是否继续,消耗时长取决于周期'''
        time = QTime.currentTime()
        self.ui.Hour.display(time.hour())
        self.ui.Minute.display(time.minute())
        self.ui.Second.display(time.second())
    def on_Start_clicked(self):
        self.timer.start()  # 开始定时,这边一启动,timeout相当于wifi就开始连接槽函数,根据周期数字就会在LCD上动态变化
        self.counter.start() # 开始计时
        # ----------------------------------------
        # 可以不单独定义槽函数,点击时即显示
        #time = QTime.currentTime()
        #self.ui.Hour.display(time.hour())
        #self.ui.Minute.display(time.minute())
        #self.ui.Second.display(time.second())
        # ----------------------------------------
        self.ui.Start.setEnabled(False) # 开始计时时封锁开始按钮,只能使用停止按钮
        self.ui.Exit.setEnabled(True)
        self.ui.setTs.setEnabled(False) # 设置周期按钮也被封锁
    def on_Exit_clicked(self):
        self.timer.stop() # 结束时timeout就变成了中断信号,LCD停止变化
        time_ms = self.counter.elapsed()  # 计时器记录的时间,ms为单位
        ms = time_ms % 1000 # 毫秒
        sec = time_ms / 1000 # 秒
        time_str = "流逝的时间为 : %d 秒 & %d 毫秒"%(sec,ms) # 一种格式化输出方式
        self.ui.Datetimelabel_3.setText(time_str) # 输出到标签上,不是文本组件上
        self.ui.Start.setEnabled(True) # 下一周期 开始按钮可以使用,停止按钮封锁
        self.ui.Exit.setEnabled(False)
        self.ui.setTs.setEnabled(True)
    def on_setTs_clicked(self):
        self.timer.setInterval(self.ui.Ts_spinBox.value()) # 把下拉条的值作为定时器的周期
# 主程序
app = QApplication(sys.argv)
MyPushButton = MyQtimer()
sys.exit(app.exec_())
