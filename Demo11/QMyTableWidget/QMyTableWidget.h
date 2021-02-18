#ifndef QMYTABLEWIDGET_H
#define QMYTABLEWIDGET_H

#include <QMainWindow>

QT_BEGIN_NAMESPACE
namespace Ui { class QMyTableWidget; }
QT_END_NAMESPACE

class QMyTableWidget : public QMainWindow
{
    Q_OBJECT

public:
    QMyTableWidget(QWidget *parent = nullptr);
    ~QMyTableWidget();

private slots:
    void on_SetHeader_clicked();

    void on_SetRowsNum_clicked();

    void on_InitialForm_clicked();

    void on_TabWidget_currentCellChanged(int currentRow, int currentColumn, int previousRow, int previousColumn);

    void on_InsertRow_clicked();

    void on_AddRow_clicked();

    void on_DeleteCurrentRow_clicked();

    void on_ClearTabContent_clicked();

    void on_AutoAdjustColWidth_clicked();

    void on_AutoAdjustRowHeight_clicked();

    void on_FormEditedEnabled_clicked();

    void on_ShowLineHeader_clicked();

    void on_ShowListHeader_clicked(bool checked);

    void on_IntervalRowBackgroundColor_clicked(bool checked);

    void on_RowSelection_clicked();

    void on_CellSelection_clicked();

    void on_ReadTabContentToText_clicked();

private:
    Ui::QMyTableWidget *ui;
};
#endif // QMYTABLEWIDGET_H
