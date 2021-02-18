#ifndef QMYTREEWIDGET_H
#define QMYTREEWIDGET_H

#include <QMainWindow>

QT_BEGIN_NAMESPACE
namespace Ui { class QMyTreeWidget; }
QT_END_NAMESPACE

class QMyTreeWidget : public QMainWindow
{
    Q_OBJECT

public:
    QMyTreeWidget(QWidget *parent = nullptr);
    ~QMyTreeWidget();

private slots:
    void on_action_AddFolder_triggered();

    void on_action_AddFile_triggered();

    void on_treeWidget_currentItemChanged(QTreeWidgetItem *current, QTreeWidgetItem *previous);

    void on_action_DeleteItem_triggered();

    void on_action_ScanItems_triggered();

    void on_action_ZoomIn_triggered();

    void on_action_ZoomOut_triggered();

    void on_action_ZoomRealSize_triggered();

    void on_action_adjustheight_triggered();

    void on_action_adjustwidth_triggered();

    void on_action_DockFloat_triggered();

    void on_action_DockVisible_triggered(bool checked);

    void on_dockWidget_topLevelChanged(bool topLevel);

    void on_dockWidget_visibilityChanged(bool visible);

    void on_action_save_triggered();

private:
    Ui::QMyTreeWidget *ui;
};
#endif // QMYTREEWIDGET_H
