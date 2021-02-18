#ifndef QMYLISTWIDGET_H
#define QMYLISTWIDGET_H

#include <QMainWindow>

QT_BEGIN_NAMESPACE
namespace Ui { class QMyListWidget; }
QT_END_NAMESPACE

class QMyListWidget : public QMainWindow
{
    Q_OBJECT

public:
    QMyListWidget(QWidget *parent = nullptr);
    ~QMyListWidget();

private slots:
    void on_action_exit_triggered();

    void on_action_initial_triggered();

    void on_action_insertitem_triggered();

    void on_action_deleteitem_triggered();

    void on_action_clear_triggered();

    void on_action_selectall_triggered();

    void on_action_quitselectall_triggered();

    void on_action_selectinverse_triggered();

    void on_listWidget_currentItemChanged(QListWidgetItem *current, QListWidgetItem *previous);

    void on_checkBox_clicked();

    void on_action_additem_triggered();

    void on_listWidget_customContextMenuRequested(const QPoint &pos);

private:
    Ui::QMyListWidget *ui;
};
#endif // QMYLISTWIDGET_H
