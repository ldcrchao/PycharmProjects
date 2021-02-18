#ifndef QSILDER_QPROGRESSBAR_H
#define QSILDER_QPROGRESSBAR_H

#include <QWidget>

QT_BEGIN_NAMESPACE
namespace Ui { class QSilder_QProgressBar; }
QT_END_NAMESPACE

class QSilder_QProgressBar : public QWidget
{
    Q_OBJECT

public:
    QSilder_QProgressBar(QWidget *parent = nullptr);
    ~QSilder_QProgressBar();

private slots:
    void on_percent_clicked();

    void on_InvertedAppearance_clicked(bool checked);

    void on_recentvalue_clicked();

    void on_textVisible_clicked(bool checked);

private:
    Ui::QSilder_QProgressBar *ui;
};
#endif // QSILDER_QPROGRESSBAR_H
