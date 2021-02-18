#ifndef QTIMER_H
#define QTIMER_H

#include <QWidget>

QT_BEGIN_NAMESPACE
namespace Ui { class QTimer; }
QT_END_NAMESPACE

class QTimer : public QWidget
{
    Q_OBJECT

public:
    QTimer(QWidget *parent = nullptr);
    ~QTimer();

private slots:
    void on_Start_clicked();

    void on_Exit_clicked();

    void on_setTs_clicked();

private:
    Ui::QTimer *ui;
};
#endif // QTIMER_H
