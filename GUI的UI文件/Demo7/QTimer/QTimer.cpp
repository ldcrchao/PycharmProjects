#include "QTimer.h"
#include "ui_QTimer.h"

QTimer::QTimer(QWidget *parent)
    : QWidget(parent)
    , ui(new Ui::QTimer)
{
    ui->setupUi(this);
}

QTimer::~QTimer()
{
    delete ui;
}


void QTimer::on_Start_clicked()
{

}

void QTimer::on_Exit_clicked()
{

}

void QTimer::on_setTs_clicked()
{

}
