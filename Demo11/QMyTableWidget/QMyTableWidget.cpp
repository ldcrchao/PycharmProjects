#include "QMyTableWidget.h"
#include "ui_QMyTableWidget.h"

QMyTableWidget::QMyTableWidget(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::QMyTableWidget)
{
    ui->setupUi(this);
}

QMyTableWidget::~QMyTableWidget()
{
    delete ui;
}


void QMyTableWidget::on_SetHeader_clicked()
{

}

void QMyTableWidget::on_SetRowsNum_clicked()
{

}

void QMyTableWidget::on_InitialForm_clicked()
{

}

void QMyTableWidget::on_TabWidget_currentCellChanged(int currentRow, int currentColumn, int previousRow, int previousColumn)
{

}

void QMyTableWidget::on_InsertRow_clicked()
{

}

void QMyTableWidget::on_AddRow_clicked()
{

}

void QMyTableWidget::on_DeleteCurrentRow_clicked()
{

}

void QMyTableWidget::on_ClearTabContent_clicked()
{

}

void QMyTableWidget::on_AutoAdjustColWidth_clicked()
{

}

void QMyTableWidget::on_AutoAdjustRowHeight_clicked()
{

}

void QMyTableWidget::on_FormEditedEnabled_clicked()
{

}

void QMyTableWidget::on_ShowLineHeader_clicked()
{

}

void QMyTableWidget::on_ShowListHeader_clicked(bool checked)
{

}

void QMyTableWidget::on_IntervalRowBackgroundColor_clicked(bool checked)
{

}

void QMyTableWidget::on_RowSelection_clicked()
{

}

void QMyTableWidget::on_CellSelection_clicked()
{

}

void QMyTableWidget::on_ReadTabContentToText_clicked()
{

}
