#include "QMyTableWidget.h"

#include <QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    QMyTableWidget w;
    w.show();
    return a.exec();
}
