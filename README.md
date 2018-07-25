# Data-driven closure discovery

This is the code for implementing several toy cases in "Data driven discovery of closure models"

To run the code, one would need
- cython
- nolds
- sklearn
- sindy package from Jean-Christophe: https://github.com/loiseaujc/SINDy 
    - note that initially borrow from this version but later on moved to sklearn implementation of SINDy.
    - readers can also try different version of SINDy but keep in mind regularization coefficient should be well tuned.

After that, one can start to compiled the code using Cython, as a purpose to acclerating our Python code a bit.


