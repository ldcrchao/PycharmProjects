#include "NormalDialog.h"
#include "ui_NormalDialog.h"

NormalDialog::NormalDialog(QWidget *parent)
    : QDialog(parent)
    , ui(new Ui::NormalDialog)
{
    ui->setupUi(this);
}

NormalDialog::~NormalDialog()
{
    delete ui;
}


void NormalDialog::on_Clear_clicked()
{

}

void NormalDialog::on_Bold_toggled(bool checked)
{

}


void NormalDialog::on_Italic_clicked(bool checked)
{

}



void NormalDialog::on_Underline_clicked()
{

}

void NormalDialog::on_Red_clicked()
{

}
