#include "QMyMCCB.h"
#include "ui_QMyMCCB.h"

QMyMCCB::QMyMCCB(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::QMyMCCB)
{
    ui->setupUi(this);
}

QMyMCCB::~QMyMCCB()
{
    delete ui;
}


void QMyMCCB::on_Exit_clicked()
{

}

void QMyMCCB::on_LoadData_clicked()
{

}

void QMyMCCB::on_StartTest_clicked()
{

}
