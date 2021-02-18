#include "QMyTreeWidget.h"
#include "ui_QMyTreeWidget.h"

QMyTreeWidget::QMyTreeWidget(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::QMyTreeWidget)
{
    ui->setupUi(this);
}

QMyTreeWidget::~QMyTreeWidget()
{
    delete ui;
}


void QMyTreeWidget::on_action_AddFolder_triggered()
{

}

void QMyTreeWidget::on_action_AddFile_triggered()
{

}

void QMyTreeWidget::on_treeWidget_currentItemChanged(QTreeWidgetItem *current, QTreeWidgetItem *previous)
{

}

void QMyTreeWidget::on_action_DeleteItem_triggered()
{

}

void QMyTreeWidget::on_action_ScanItems_triggered()
{

}

void QMyTreeWidget::on_action_ZoomIn_triggered()
{

}

void QMyTreeWidget::on_action_ZoomOut_triggered()
{

}

void QMyTreeWidget::on_action_ZoomRealSize_triggered()
{

}

void QMyTreeWidget::on_action_adjustheight_triggered()
{

}

void QMyTreeWidget::on_action_adjustwidth_triggered()
{

}

void QMyTreeWidget::on_action_DockFloat_triggered(bool checked)
{

}

void QMyTreeWidget::on_action_DockVisible_triggered(bool checked)
{

}

void QMyTreeWidget::on_dockWidget_topLevelChanged(bool topLevel)
{

}

void QMyTreeWidget::on_dockWidget_visibilityChanged(bool visible)
{

}

void QMyTreeWidget::on_action_save_triggered()
{

}
