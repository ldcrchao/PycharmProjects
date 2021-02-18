#ifndef QMYMAINWINDOW_H
#define QMYMAINWINDOW_H

#include <QMainWindow>

QT_BEGIN_NAMESPACE
namespace Ui { class QMyMainWindow; }
QT_END_NAMESPACE

class QMyMainWindow : public QMainWindow
{
    Q_OBJECT

public:
    QMyMainWindow(QWidget *parent = nullptr);
    ~QMyMainWindow();

private slots:
    void on_plainTextEdit_undoAvailable(bool b);

    void on_action_textredo_triggered();

    void on_action_bold_triggered(bool checked);

    void on_action_textitalic_triggered(bool checked);

    void on_action_textunderline_triggered(bool checked);

    void on_plainTextEdit_copyAvailable(bool b);

    void on_plainTextEdit_selectionChanged();

    void on_action_filenew_triggered();

    void on_action_filesave_triggered();

    void on_action_fileopen_triggered();

    void on_action_textshow_triggered(bool checked);

private:
    Ui::QMyMainWindow *ui;
};
#endif // QMYMAINWINDOW_H
