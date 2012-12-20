Pycat
=====

Python interactive 3D graphcut


Requirements
------------

sudo apt-get install python-sklearn
gco_python


Install
-------

git clone --recursive git@github.com:mjirik/pycat.git

or

git clone git@github.com:mjirik/pycat.git

git submodule update --init --recursive

gco_python install notes
------------------------

see  http://peekaboo-vision.blogspot.cz/2012/05/graphcuts-for-python-pygco.html

sudo apt-get install cython

git clone https://github.com/amueller/gco_python.git
cd gco_python

make

this will crash but dowload  gco_src is ok

you need to include stddef.h in GCoptimization.h

sed -i '111 i\#include "stddef.h"' gco_src/GCoptimization.h

again

make

and final install
sudo python setup.py install


example
python example.py
