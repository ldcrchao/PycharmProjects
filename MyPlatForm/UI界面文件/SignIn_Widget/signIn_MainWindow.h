#ifndef SIGNIN_MAINWINDOW_H
#define SIGNIN_MAINWINDOW_H

#include <QMainWindow>

QT_BEGIN_NAMESPACE
namespace Ui { class signIn_MainWindow; }
QT_END_NAMESPACE

class signIn_MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    signIn_MainWindow(QWidget *parent = nullptr);
    ~signIn_MainWindow();

private slots:
    void on_sure_Button_clicked(bool checked);

private:
    Ui::signIn_MainWindow *ui;
};
#endif // SIGNIN_MAINWINDOW_H
