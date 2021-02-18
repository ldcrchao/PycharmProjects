#ifndef QDATETIME_H
#define QDATETIME_H

#include <QWidget>

QT_BEGIN_NAMESPACE
namespace Ui { class QDateTime; }
QT_END_NAMESPACE

class QDateTime : public QWidget
{
    Q_OBJECT

public:
    QDateTime(QWidget *parent = nullptr);
    ~QDateTime();

private slots:
    void on_readdatetime_clicked();

    void on_settimepushButton_clicked();

    void on_setdatepushButton_clicked();

    void on_setdatetimepushButton_clicked();

    void on_calendarWidget_selectionChanged();

    void on_timeEdit_timeChanged(const QTime &time);

    void on_dateEdit_dateChanged(const QDate &date);

    void on_dateTimeEdit_dateTimeChanged(const QDateTime &dateTime);

private:
    Ui::QDateTime *ui;
};
#endif // QDATETIME_H
