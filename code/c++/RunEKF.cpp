#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include <Eigen/StdVector>

#include <math.h>
#include "EKF.h"

using Eigen::MatrixXd;
using Eigen::Matrix3f;
using Eigen::VectorXd;
using namespace std;

MatrixXd readCSV(string file, int rows, int cols) {

  ifstream in(file);
  
  string line;

  int row = 0;
  int col = 0;

  MatrixXd res = MatrixXd(rows, cols);

  if (in.is_open()) {

    while (getline(in, line)) {

      char *ptr = (char *) line.c_str();
      int len = line.length();

      col = 0;

      char *start = ptr;
      for (int i = 0; i < len; i++) {

        if (ptr[i] == ',') {
          res(row, col++) = atof(start);
          start = ptr + i + 1;
        }
      }
      res(row, col) = atof(start);

      row++;
    }

    in.close();
  }
  return res;
}

void writeToCSVfile(string name, MatrixXd matrix)
{
  ofstream file(name.c_str());

  for(int  i = 0; i < matrix.rows(); i++){
      for(int j = 0; j < matrix.cols(); j++){
         string str = std::to_string(matrix(i,j));
         if(j+1 == matrix.cols()){
             file<<str;
         }else{
             file<<str<<',';
         }
      }
      file<<'\n';
  }
  file.close();
}

float radians(int degree){
	return degree * M_PI / 180;
}

int main(){
	MatrixXd U = readCSV("U.txt", 2087, 2);
	MatrixXd XYT = readCSV("XYT.txt", 2088, 3);
	MatrixXd Z = readCSV("Z.txt", 2087, 2);

	VectorXd mu(3);
	mu<<-4.0, -4.0, M_PI/2; 
	Matrix3f Sigma;
	Sigma<<1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0;
	MatrixXd R(3, 3);
	R<<2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, radians(2);
	R = R * 1e-4;
	MatrixXd Q(2, 2);
	Q<<1.0, 0.0, 0.0, radians(1);
	Q = Q * 1e-6;
	MatrixXd MU(2088, 3);
	MU.row(0) = mu;
	MatrixXd VAR(2088, 3);
	VAR(0, 0) = 1;
	VAR(0, 1) = 1;
	VAR(0, 2) = 1;

	EKF ekf;
	ekf.init(mu, Sigma, R, Q);

	for(int i = 0; i < 2087; i++){
		cout << i << endl;
		ekf.prediction(U.row(i));
		ekf.update(Z.row(i));

		MU(i+1, 0) = ekf.getMean()(0);
		MU(i+1, 1) = ekf.getMean()(1);
		MU(i+1, 2) = ekf.getMean()(2);
		VAR(0, 0) = ekf.getVariances()(0, 0);
		VAR(0, 1) = ekf.getVariances()(1, 1);
		VAR(0, 2) = ekf.getVariances()(2, 2);
	}

	writeToCSVfile("MU.csv", MU);
	writeToCSVfile("XYT.csv", XYT);
}