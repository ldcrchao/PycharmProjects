#ifndef QPUSHBUTTON_USE_H
#define QPUSHBUTTON_USE_H

#include <QWidget>

QT_BEGIN_NAMESPACE
namespace Ui { class QPushButton_Use; }
QT_END_NAMESPACE

class QPushButton_Use : public QWidget
{
    Q_OBJECT

public:
    QPushButton_Use(QWidget *parent = nullptr);
    ~QPushButton_Use();

private slots:
    void on_btn_left_clicked();

    void on_btn_middle_clicked();

    void on_btn_bold_clicked();

    void on_btn_bold_clicked(bool checked);

    void on_btn_italic_clicked(bool checked);

    void on_btn_underline_clicked(bool checked);

    void on_checkBox_clicked(bool checked);

    void on_checkBox_2_clicked(bool checked);

    void on_checkBox_3_clicked(bool checked);

private:
    Ui::QPushButton_Use *ui;
};
#endif // QPUSHBUTTON_USE_H
