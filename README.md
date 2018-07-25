# Code for Data-driven closure discovery

Brief overview
--------------

This is the code, written in Python, for implementing several toy cases in "Data driven discovery of closure models" in https://arxiv.org/abs/1803.09318

How to run this code?
---------------------

To run the code, one would need
- cython
- nolds
- sklearn
- keras
- tensorflow
- sindy package from Jean-Christophe: https://github.com/loiseaujc/SINDy 
    - note that initially borrow from this version but later on moved to sklearn implementation of SINDy.
    - readers can also try different version of SINDy but keep in mind regularization coefficient should be well tuned.

After that, one can start to compiled the code using Cython, as a purpose to acclerating our Python code.
To run different cases, one should first cd to /src folder, and then doing the folloing.

To run simple 3D linear system case
- python cython_main.py 3d_linear sindy 1 tdf 1

To run 2D VDP case
- python cython_main.py 2d_vdp sindy 0 tdf 3

To run Lorenz case 

- non-chaotic case:
    - python cython_main.py lorenz-equ ann_cross_time 1 18 
- chaotic case: 
    - python cython_main.py lorenz-chaos ann_cross_time 1 18 

How to check final result?
--------------------------
To check the final result, go into the folder of the /data/ and check corresponding /image subfolders

How long does it take on a normal laptop?
-----------------------------------------

For 3D linear and 2D VDP case, it is really fast within 2-5 mins.

For Lorenz cases, since the number of data points is a bit large and we have a posteriori evaluations, the computational time is a bit long.
For non-chaotic cases it took around 30 mins and for chaotic case it took 220 mins.


 
