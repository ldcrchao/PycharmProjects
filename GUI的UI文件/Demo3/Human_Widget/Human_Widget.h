#ifndef HUMAN_WIDGET_H
#define HUMAN_WIDGET_H

#include <QWidget>

QT_BEGIN_NAMESPACE
namespace Ui { class Human_Widget; }
QT_END_NAMESPACE

class Human_Widget : public QWidget
{
    Q_OBJECT

public:
    Human_Widget(QWidget *parent = nullptr);
    ~Human_Widget();

private slots:

    void on_setageslider_valueChanged(int value);

    void on_checkBox_clicked();

    void on_close_clicked();

private:
    Ui::Human_Widget *ui;
};
#endif // HUMAN_WIDGET_H
