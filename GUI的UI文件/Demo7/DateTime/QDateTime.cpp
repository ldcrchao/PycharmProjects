#include "QDateTime.h"
#include "ui_QDateTime.h"

QDateTime::QDateTime(QWidget *parent)
    : QWidget(parent)
    , ui(new Ui::QDateTime)
{
    ui->setupUi(this);
}

QDateTime::~QDateTime()
{
    delete ui;
}


void QDateTime::on_readdatetime_clicked()
{

}

void QDateTime::on_settimepushButton_clicked()
{

}

void QDateTime::on_setdatepushButton_clicked()
{

}

void QDateTime::on_setdatetimepushButton_clicked()
{

}

void QDateTime::on_calendarWidget_selectionChanged()
{

}

void QDateTime::on_timeEdit_timeChanged(const QTime &time)
{

}

void QDateTime::on_dateEdit_dateChanged(const QDate &date)
{

}

void QDateTime::on_dateTimeEdit_dateTimeChanged(const QDateTime &dateTime)
{

}
