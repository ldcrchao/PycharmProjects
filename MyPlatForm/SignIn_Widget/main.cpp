#include "signIn_MainWindow.h"

#include <QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    signIn_MainWindow w;
    w.show();
    return a.exec();
}
