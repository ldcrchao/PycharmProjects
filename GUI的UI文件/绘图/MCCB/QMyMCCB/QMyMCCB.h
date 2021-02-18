#ifndef QMYMCCB_H
#define QMYMCCB_H

#include <QMainWindow>

QT_BEGIN_NAMESPACE
namespace Ui { class QMyMCCB; }
QT_END_NAMESPACE

class QMyMCCB : public QMainWindow
{
    Q_OBJECT

public:
    QMyMCCB(QWidget *parent = nullptr);
    ~QMyMCCB();

private slots:
    void on_Exit_clicked();

    void on_LoadData_clicked();

    void on_StartTest_clicked();

private:
    Ui::QMyMCCB *ui;
};
#endif // QMYMCCB_H
