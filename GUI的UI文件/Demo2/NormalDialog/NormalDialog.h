#ifndef NORMALDIALOG_H
#define NORMALDIALOG_H

#include <QDialog>

QT_BEGIN_NAMESPACE
namespace Ui { class NormalDialog; }
QT_END_NAMESPACE

class NormalDialog : public QDialog
{
    Q_OBJECT

public:
    NormalDialog(QWidget *parent = nullptr);
    ~NormalDialog();

private slots:
    void on_Clear_clicked();

    void on_Bold_toggled(bool checked);

    void on_Italic_clicked(bool checked);




    void on_Underline_clicked();

    void on_Red_clicked();

private:
    Ui::NormalDialog *ui;
};
#endif // NORMALDIALOG_H
