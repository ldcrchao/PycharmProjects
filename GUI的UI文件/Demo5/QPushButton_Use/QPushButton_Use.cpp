#include "QPushButton_Use.h"
#include "ui_QPushButton_Use.h"

QPushButton_Use::QPushButton_Use(QWidget *parent)
    : QWidget(parent)
    , ui(new Ui::QPushButton_Use)
{
    ui->setupUi(this);
}

QPushButton_Use::~QPushButton_Use()
{
    delete ui;
}


void QPushButton_Use::on_btn_left_clicked()
{

}

void QPushButton_Use::on_btn_middle_clicked()
{

}



void QPushButton_Use::on_btn_bold_clicked(bool checked)
{

}

void QPushButton_Use::on_btn_italic_clicked(bool checked)
{

}

void QPushButton_Use::on_btn_underline_clicked(bool checked)
{

}

void QPushButton_Use::on_checkBox_clicked(bool checked)
{

}

void QPushButton_Use::on_checkBox_2_clicked(bool checked)
{

}

void QPushButton_Use::on_checkBox_3_clicked(bool checked)
{

}
