#ifndef FAULTPLATFORM_H
#define FAULTPLATFORM_H

#include <QMainWindow>

QT_BEGIN_NAMESPACE
namespace Ui { class faultPlatform; }
QT_END_NAMESPACE

class faultPlatform : public QMainWindow
{
    Q_OBJECT

public:
    faultPlatform(QWidget *parent = nullptr);
    ~faultPlatform();

private slots:
    void on_comboBox_currentIndexChanged(const QString &arg1);

    void on_pushButton_clicked(bool checked);

    void on_tableWidget_currentItemChanged(QTableWidgetItem *current, QTableWidgetItem *previous);

    void on_svm_btn_clicked(bool checked);

    void on_start_training_clicked(bool checked);

    void on_current_predict_tabWidget_currentCellChanged(int currentRow, int currentColumn, int previousRow, int previousColumn);
    
    void on_vbriation_predict_tabWidget_currentCellChanged(int currentRow, int currentColumn, int previousRow, int previousColumn);
    
    void on_cancel_training_clicked(bool checked);

private:
    Ui::faultPlatform *ui;
};
#endif // FAULTPLATFORM_H
