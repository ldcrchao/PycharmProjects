#include "faultPlatform.h"
#include "ui_faultPlatform.h"

faultPlatform::faultPlatform(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::faultPlatform)
{
    ui->setupUi(this);
}

faultPlatform::~faultPlatform()
{
    delete ui;
}


void faultPlatform::on_comboBox_currentIndexChanged(const QString &arg1)
{

}

void faultPlatform::on_pushButton_clicked(bool checked)
{

}

void faultPlatform::on_tableWidget_currentItemChanged(QTableWidgetItem *current, QTableWidgetItem *previous)
{

}

void faultPlatform::on_svm_btn_clicked(bool checked)
{

}

void faultPlatform::on_current_predict_tabWidget_currentCellChanged(int currentRow, int currentColumn, int previousRow, int previousColumn)
{
    
}

void faultPlatform::on_vbriation_predict_tabWidget_currentCellChanged(int currentRow, int currentColumn, int previousRow, int previousColumn)
{
    
}

void faultPlatform::on_start_training_clicked(bool checked)
{

}

void faultPlatform::on_cancel_training_clicked(bool checked)
{

}
