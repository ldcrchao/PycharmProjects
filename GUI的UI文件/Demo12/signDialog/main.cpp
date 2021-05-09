#include "signDialog.h"

#include <QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    signDialog w;
    w.show();
    return a.exec();
}
