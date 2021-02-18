#ifndef QCOMBOBOX_H
#define QCOMBOBOX_H

#include <QWidget>

QT_BEGIN_NAMESPACE
namespace Ui { class QComboBox; }
QT_END_NAMESPACE

class QComboBox : public QWidget
{
    Q_OBJECT

public:
    QComboBox(QWidget *parent = nullptr);
    ~QComboBox();

private slots:
    void on_initiallist_clicked();

    void on_clearlist_clicked();

    void on_Enabled_clicked(bool checked);

    void on_comboBox1_currentIndexChanged(const QString &arg1);

    void on_initial_city_num_clicked();

    void on_comboBox2_currentIndexChanged(const QString &arg1);

private:
    Ui::QComboBox *ui;
};
#endif // QCOMBOBOX_H
