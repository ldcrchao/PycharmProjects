#include "Human_Widget.h"

#include <QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    Human_Widget w;
    w.show();
    return a.exec();
}
