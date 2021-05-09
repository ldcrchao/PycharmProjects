#ifndef SIGNDIALOG_H
#define SIGNDIALOG_H

#include <QDialog>

QT_BEGIN_NAMESPACE
namespace Ui { class signDialog; }
QT_END_NAMESPACE

class signDialog : public QDialog
{
    Q_OBJECT

public:
    signDialog(QWidget *parent = nullptr);
    ~signDialog();

private:
    Ui::signDialog *ui;
};
#endif // SIGNDIALOG_H
