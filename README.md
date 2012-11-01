Pycat
=====

Python interactive 3D graphcut


Requirements
------------

gco_python


gco_python install notes
------------------------


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
