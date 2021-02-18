#include "SumPrice.h"
#include "ui_SumPrice.h"

SumPrice::SumPrice(QWidget *parent)
    : QWidget(parent)
    , ui(new Ui::SumPrice)
{
    ui->setupUi(this);
}

SumPrice::~SumPrice()
{
    delete ui;
}


void SumPrice::on_CalculateButton_clicked()
{

}

void SumPrice::on_numspinBox_valueChanged(int arg1)
{

}

void SumPrice::on_pricedoubleSpinBox_valueChanged(double arg1)
{

}
