#include "signDialog.h"
#include "ui_signDialog.h"

signDialog::signDialog(QWidget *parent)
    : QDialog(parent)
    , ui(new Ui::signDialog)
{
    ui->setupUi(this);
}

signDialog::~signDialog()
{
    delete ui;
}

