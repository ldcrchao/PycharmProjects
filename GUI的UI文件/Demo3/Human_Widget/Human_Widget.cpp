#include "Human_Widget.h"
#include "ui_Human_Widget.h"

Human_Widget::Human_Widget(QWidget *parent)
    : QWidget(parent)
    , ui(new Ui::Human_Widget)
{
    ui->setupUi(this);
}

Human_Widget::~Human_Widget()
{
    delete ui;
}




void Human_Widget::on_setageslider_valueChanged(int value)
{

}

void Human_Widget::on_checkBox_clicked()
{

}

void Human_Widget::on_close_clicked()
{

}
