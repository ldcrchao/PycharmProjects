#include "QMyMCCB.h"

#include <QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    QMyMCCB w;
    w.show();
    return a.exec();
}
