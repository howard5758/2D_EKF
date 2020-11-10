#ifndef EKF_H_
#define EKF_H_

#include <math.h>
#include <Eigen/Dense>

using Eigen::MatrixXd;
using Eigen::Matrix3f;
using Eigen::VectorXd;
using Eigen::Vector3f;
using namespace std;

 // Construct an EKF instance with the following set of variables
 //        mu:                 The initial mean vector
 //        Sigma:              The initial covariance matrix
 //        R:                  The process noise covariance
 //        Q:                  The measurement noise covariance

class EKF{
public:
	
	EKF();
	void init(const VectorXd& mu, const Matrix3f& Sigma, const MatrixXd& R, const MatrixXd& Q);

	void prediction(const MatrixXd& u);

	void update(const MatrixXd& z);

	Vector3f getMean();
	Matrix3f getVariances();

private:
	VectorXd mu;
	Matrix3f Sigma;
	MatrixXd R, Q;
	MatrixXd F, H;
};
#endif