import numpy as np
from numpy.linalg import inv
import math

class EKF(object):
    # Construct an EKF instance with the following set of variables
    #    mu:                 The initial mean vector
    #    Sigma:              The initial covariance matrix
    #    R:                  The process noise covariance
    #    Q:                  The measurement noise covariance
    def __init__(self, mu, Sigma, R, Q):
        self.mu = mu
        self.Sigma = Sigma
        self.R = R
        self.Q = Q

        self.F = np.zeros((3, 3))
        self.H = np.zeros((2, 3))


    def getMean(self):
        return self.mu


    def getCovariance(self):
        return self.Sigma


    def getVariances(self):
        return np.array([[self.Sigma[0,0],self.Sigma[1,1],self.Sigma[2,2]]])

    def getSigma(self):
        return self.Sigma
        

    # Perform the prediction step to determine the mean and covariance
    # of the posterior belief given the current estimate for the mean
    # and covariance, the control data, and the process model
    #    u:                 The forward distance and change in heading
    def prediction(self,u):
        #print(u)
        self.F[0, 0] = 1
        self.F[1, 1] = 1
        self.F[2, 2] = 1
        self.F[0, 2] = -1 * u[0] * math.sin(self.mu[0][2])
        self.F[1, 2] = u[0] * math.cos(self.mu[0][2])

        self.mu[0][0] = self.mu[0][0] + (u[0] * math.cos(self.mu[0][2]))
        self.mu[0][1] = self.mu[0][1] + (u[0] * math.sin(self.mu[0][2]))
        self.mu[0][2] = self.mu[0][2] + u[1]


        temp = np.matmul(self.F, self.Sigma)
        temp = np.matmul(temp, np.transpose(self.F))
        self.Sigma = temp + self.R


    # Perform the measurement update step to compute the posterior
    # belief given the predictive posterior (mean and covariance) and
    # the measurement data
    #    z:                The squared distance to the sensor and the
    #                      robot's heading
    def update(self,z):
        
        self.H[0, 0] = 2 * self.mu[0][0]
        self.H[0, 1] = 2 * self.mu[0][1]
        self.H[1, 2] = 1

        temp = np.matmul(self.Sigma, np.transpose(self.H))
        invtemp = np.matmul(self.H, self.Sigma)
        invtemp = np.matmul(invtemp, np.transpose(self.H))
        invtemp = invtemp + self.Q
        #print("KK")
        K = np.matmul(temp, inv(invtemp))
        #print(K)

        hu = np.array([(self.mu[0][0]*self.mu[0][0]) + (self.mu[0][1]*self.mu[0][1]), self.mu[0][2]])
        #print("---")
        zDiff = z - hu
        #print(zDiff)
        kGain = np.matmul(K, zDiff)
        #print(kGain)
        self.mu = self.mu + kGain

        KH = np.matmul(K, self.H)
        temp = np.eye(3) - KH
        self.Sigma = np.matmul(temp, self.Sigma)