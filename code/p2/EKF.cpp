#include <iostream>
#include <Eigen/Dense>

#include <math.h>
#include "EKF.h"
using Eigen::MatrixXd;
using Eigen::Matrix3f;
using Eigen::VectorXd;
using Eigen::Vector3f;
using Eigen::Vector2f;

EKF::EKF(){}

void EKF::init(const VectorXd& mu, const Matrix3f& Sigma, const MatrixXd& R, const MatrixXd& Q){
	this->mu = mu;
	this->Sigma = Sigma;
	this->R = R;
	this->Q = Q;

	this->F = MatrixXd(3, 3);
	this->H = MatrixXd(2, 3);
}

void EKF::prediction(const MatrixXd& u){
	this->F(0, 0) = 1.0;
	this->F(0, 1) = 0.0;
	this->F(0, 2) = -1 * u(0) * sin(this->mu(2));
	this->F(1, 0) = 0.0;
	this->F(1, 1) = 1.0;
	this->F(1, 2) = u(0) * cos(this->mu(2));
	this->F(2, 0) = 0.0;
	this->F(2, 1) = 0.0;
	this->F(2, 2) = 1.0;
	
	this->mu(0) = float(this->mu(0) + (u(0) * cos(this->mu(2))));
	this->mu(1) = float(this->mu(1) + (u(0) * sin(this->mu(2))));
	this->mu(2) = float(this->mu(2) + u(1));
//  self.mu[0][0] = self.mu[0][0] + (u[0] * math.cos(self.mu[0][2]))
//  self.mu[0][1] = self.mu[0][1] + (u[0] * math.sin(self.mu[0][2]))
//  self.mu[0][2] = self.mu[0][2] + u[1]
	Matrix3f temp = this->F.cast<float>()*this->Sigma.cast<float>();
	MatrixXd trans = this->F.transpose();
	this->Sigma = temp*trans.cast<float>() + this->R.cast<float>();
//  temp = np.matmul(self.F, self.Sigma)
//  temp = np.matmul(temp, np.transpose(self.F))
//  self.Sigma = temp + self.R
}

void EKF::update(const MatrixXd& z){

	this->H(0, 0) = 2 * this->mu(0);
	this->H(0, 1) = 2 * this->mu(1);
	this->H(0, 2) = 0;
	this->H(1, 0) = 0;
	this->H(1, 1) = 0;
	this->H(1, 2) = 1;
//  self.H[0, 0] = 2 * self.mu[0][0]
//  self.H[0, 1] = 2 * self.mu[0][1]
//  self.H[1, 2] = 1
	MatrixXd transH = this->H.transpose();
	MatrixXd temp = this->Sigma.cast<double>() * transH.cast<double>();
	MatrixXd invtemp = this->H.cast<double>() * this->Sigma.cast<double>();
	invtemp = invtemp * transH.cast<double>() + this->Q.cast<double>();
	MatrixXd k = temp * invtemp.inverse();
//  temp = np.matmul(self.Sigma, np.transpose(self.H))
//  invtemp = np.matmul(self.H, self.Sigma)
//  invtemp = np.matmul(invtemp, np.transpose(self.H))
//  invtemp = invtemp + self.Q
//  K = np.matmul(temp, inv(invtemp))

	MatrixXd hu(1, 2);
	hu<<this->mu(0)*this->mu(0) + this->mu(1)*this->mu(1), this->mu(2);
	MatrixXd zDiff(1, 2);
	zDiff = z.cast<double>() - hu.cast<double>();
	MatrixXd kg(1, 3);
	kg = k.cast<double>() * zDiff.transpose().cast<double>();
	this->mu = this->mu.cast<double>() + kg.cast<double>();

//  hu = np.array([(self.mu[0][0]*self.mu[0][0]) + (self.mu[0][1]*self.mu[0][1]), self.mu[0][2]])
//  zDiff = z - hu
//  kGain = np.matmul(K, zDiff)
//  self.mu = self.mu + kGain

	MatrixXd KH = k * this->H;
	MatrixXd eye = MatrixXd::Identity(3, 3);
	this->Sigma = (eye - KH).cast<float>() * this->Sigma;
//  KH = np.matmul(K, self.H)
//  temp = np.eye(3) - KH
//  self.Sigma = np.matmul(temp, self.Sigma)
}

Vector3f EKF::getMean(){
	return this->mu.cast<float>();
}
Matrix3f EKF::getVariances(){
	return this->Sigma.cast<float>();
}
