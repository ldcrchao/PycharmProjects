#include "NormalDialog.h"

#include <QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    NormalDialog w;
    w.show();
    return a.exec();
}
