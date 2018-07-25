# Data-driven closure discovery

This is the code for implementing several toy cases in "Data driven discovery of closure models" in https://arxiv.org/abs/1803.09318

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

To run simple 3D linear system case
- python cython_main.py 3d_linear sindy 1 tdf 1

To run 2D VDP case
- python cython_main.py 2d_vdp sindy 0 tdf 3

To run Lorenz case 

- non-chaotic case:
    - python cython_main.py lorenz-equ ann_cross_time 1 18 
- chaotic case: 
    - python cython_main.py lorenz-chaos ann_cross_time 1 18 

