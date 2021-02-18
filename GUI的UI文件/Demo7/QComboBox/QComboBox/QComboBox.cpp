#include "QComboBox.h"
#include "ui_QComboBox.h"

QComboBox::QComboBox(QWidget *parent)
    : QWidget(parent)
    , ui(new Ui::QComboBox)
{
    ui->setupUi(this);
}

QComboBox::~QComboBox()
{
    delete ui;
}


void QComboBox::on_initiallist_clicked()
{

}

void QComboBox::on_clearlist_clicked()
{

}

void QComboBox::on_Enabled_clicked(bool checked)
{

}

void QComboBox::on_comboBox1_currentIndexChanged(const QString &arg1)
{

}

void QComboBox::on_initial_city_num_clicked()
{

}

void QComboBox::on_comboBox2_currentIndexChanged(const QString &arg1)
{

}
