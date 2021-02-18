#include "QMyListWidget.h"
#include "ui_QMyListWidget.h"

QMyListWidget::QMyListWidget(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::QMyListWidget)
{
    ui->setupUi(this);
}

QMyListWidget::~QMyListWidget()
{
    delete ui;
}


void QMyListWidget::on_action_exit_triggered()
{

}

void QMyListWidget::on_action_initial_triggered()
{

}

void QMyListWidget::on_action_insertitem_triggered()
{

}

void QMyListWidget::on_action_deleteitem_triggered()
{

}

void QMyListWidget::on_action_clear_triggered()
{

}

void QMyListWidget::on_action_selectall_triggered()
{

}

void QMyListWidget::on_action_quitselectall_triggered()
{

}

void QMyListWidget::on_action_selectinverse_triggered()
{

}

void QMyListWidget::on_listWidget_currentItemChanged(QListWidgetItem *current, QListWidgetItem *previous)
{

}

void QMyListWidget::on_checkBox_clicked()
{

}

void QMyListWidget::on_action_additem_triggered()
{

}

void QMyListWidget::on_listWidget_customContextMenuRequested(const QPoint &pos)
{

}
