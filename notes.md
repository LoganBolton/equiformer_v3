equiformerv3 can do 5 ring. can't do 10 ring. can do 6 but can't do 7

can replicate same behavior by setting num graphs to 2 instead of 100

can do when num_node == lmax

hip hop can't do:
    - 2 inner 3 outer
    - 1 inner 2 outer

note: 1 inner 2 outer is bugged out


# Graph results
## mixed rings
- 1 inner, 2 outer
    - lmax 2 -> failure
    - lmax 4 -> failure
    - lmax 6 -> **failure**
    - hip hop fails at lmax=3
- 1 inner, 3 outer
    - lmax 2 -> failure
    - lmax 3 -> **success**
    - lmax 6 -> success
    - hiphop succeeds at lmax=3 and fails at 2
- 1 inner, 4 outer
    - lmax 3 -> failure
    - lmax 4 -> failure
    - lmax 5 -> failure
    - lmax 6 -> failure
    - hip hop fails at lmax=2,3
- 2 inner, 1 outer
    - lmax 2 -> success
    - lmax 3 -> success
    - lmax 6 -> success
    - hip hop succeeds at lmax=2 and 3
- 2 inner, 2 outer
    - lmax 2 -> success
    - lmax 3 -> success
    - hip hop succeeds at lmax=2 and 3
- 2 inner, 3 outer
    - lmax 3 -> failure
    - lmax 4 -> **success**
    - lmax 6 -> success
    - hip hop fails at lmax=3
- 2 inner, 4 outer
    - lmax 3 -> failure
    - lmax 4 -> failure
    - lmax 5 -> failure
    - lmax 6 -> failure
    - hiphop failed at lmax=2,3
- 3 inner, 1 outer
    - lmax 2 -> failure
    - lmax 3 -> success
- 3 inner, 2 outer
    - lmax 2 -> failure
    - lmax 3 -> failure
    - lmax 6 -> failure
    - hip hop fails at lmax=2 and lmax=3
- 3 inner, 3 outer
    - lmax 2 -> fails
    - lmax 3 -> success
    - hip hop fails lmax=2, succeeds at lmax=3
- 3 inner, 4 outer
    - lmax 3 -> fail
    - lmax 4 -> fail
    - lmax 5 -> fail
    - lmax 6 -> **fail**
- 4 inner, 3 outer
    - lmax 6 -> **fail**

- 4 inner, 4 outer
    - lmax 3 -> fail
    - lmax 4 -> success
    - lmax 6 -> success
    - hip hop fails at lmax=2,3
## mono rings
- 3 ring
    - lmax 3 -> success

- 4 ring
    - lmax 3 -> fail
    - lmax 4 -> success
    - lmax 6 -> success
- 5 ring
    - lmax 4 -> fail
    - lmax 5 -> success
- 6 ring
    - lmax 5 -> fail
    - lmax 6 -> success