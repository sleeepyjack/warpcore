#!/bin/sh
cd "$(dirname "$0")/../docs"
cp Doxyfile /tmp/
rm -rf *
mv /tmp/Doxyfile .
doxygen Doxyfile
