#include "QMyMainWindow.h"
#include "ui_QMyMainWindow.h"

QMyMainWindow::QMyMainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::QMyMainWindow)
{
    ui->setupUi(this);
}

QMyMainWindow::~QMyMainWindow()
{
    delete ui;
}


void QMyMainWindow::on_action_bold_triggered(bool checked)
{

}

void QMyMainWindow::on_action_textitalic_triggered(bool checked)
{

}

void QMyMainWindow::on_action_textunderline_triggered(bool checked)
{

}

void QMyMainWindow::on_plainTextEdit_copyAvailable(bool b)
{

}

void QMyMainWindow::on_plainTextEdit_selectionChanged()
{

}

void QMyMainWindow::on_action_filenew_triggered()
{

}

void QMyMainWindow::on_action_filesave_triggered()
{

}

void QMyMainWindow::on_action_fileopen_triggered()
{

}

void QMyMainWindow::on_action_textshow_triggered(bool checked)
{

}
