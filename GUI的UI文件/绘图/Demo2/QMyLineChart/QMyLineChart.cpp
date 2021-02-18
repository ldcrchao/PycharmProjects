#include "QMyLineChart.h"
#include "ui_QMyLineChart.h"

QMyLineChart::QMyLineChart(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::QMyLineChart)
{
    ui->setupUi(this);
}

QMyLineChart::~QMyLineChart()
{
    delete ui;
}

