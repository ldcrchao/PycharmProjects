#include "signIn_MainWindow.h"
#include "ui_signIn_MainWindow.h"

signIn_MainWindow::signIn_MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::signIn_MainWindow)
{
    ui->setupUi(this);
}

signIn_MainWindow::~signIn_MainWindow()
{
    delete ui;
}


void signIn_MainWindow::on_sure_Button_clicked(bool checked)
{

}
