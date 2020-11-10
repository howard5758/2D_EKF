# 2D_Extended_Kalman_Filter

To run testing and visualization using Python version:

$cd code

$python RunEKF.py U.txt Z.txt XYT.txt (The three arguments correspond to control data, measurement data, and groundtruths.)

To run testing without visualization using C++ version:

$cd code/c++

$g++ -I EIGEN_PATH RunEKF.cpp EKF.cpp -o TEST_NAME
Then run executable.
