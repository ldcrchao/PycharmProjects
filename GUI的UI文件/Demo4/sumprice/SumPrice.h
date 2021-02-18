#ifndef SUMPRICE_H
#define SUMPRICE_H

#include <QWidget>

QT_BEGIN_NAMESPACE
namespace Ui { class SumPrice; }
QT_END_NAMESPACE

class SumPrice : public QWidget
{
    Q_OBJECT

public:
    SumPrice(QWidget *parent = nullptr);
    ~SumPrice();

private slots:
    void on_CalculateButton_clicked();

    void on_numspinBox_valueChanged(int arg1);

    void on_pricedoubleSpinBox_valueChanged(double arg1);

private:
    Ui::SumPrice *ui;
};
#endif // SUMPRICE_H
