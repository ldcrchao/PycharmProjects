#! /bin/bash
gcc -Wall -g  -fPIC -c modelio.c -o modelio.o
gcc -shared modelio.o -o modelio.so
rm modelio.o