#ifndef QMYLINECHART_H
#define QMYLINECHART_H

#include <QMainWindow>

QT_BEGIN_NAMESPACE
namespace Ui { class QMyLineChart; }
QT_END_NAMESPACE

class QMyLineChart : public QMainWindow
{
    Q_OBJECT

public:
    QMyLineChart(QWidget *parent = nullptr);
    ~QMyLineChart();

private:
    Ui::QMyLineChart *ui;
};
#endif // QMYLINECHART_H
