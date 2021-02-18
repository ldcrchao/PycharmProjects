#include "QSilder_QProgressBar.h"
#include "ui_QSilder_QProgressBar.h"

QSilder_QProgressBar::QSilder_QProgressBar(QWidget *parent)
    : QWidget(parent)
    , ui(new Ui::QSilder_QProgressBar)
{
    ui->setupUi(this);
}

QSilder_QProgressBar::~QSilder_QProgressBar()
{
    delete ui;
}


void QSilder_QProgressBar::on_percent_clicked()
{

}

void QSilder_QProgressBar::on_InvertedAppearance_clicked()
{

}

void QSilder_QProgressBar::on_recentvalue_clicked()
{

}

void QSilder_QProgressBar::on_textVisible_clicked(bool checked)
{

}
