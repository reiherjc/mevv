import numpy as np

def ellipsoid(pcl,tolerance):
    '''
    This function uses an algorithm described in the paper
    MINIMUM VOLUME ENCLOSING ELLIPSOIDS by NIMA MOSHTAGH
    Available from
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.116.7691&rep=rep1&type=pdf


    Inputs:
    1. pcl: np array of points in 3d space (e.g. np.array([[0,0,0],[1,1,1]])) 
    2. tolerance: accuracy parmeter (e.g. 0.001)
    Returns:
    1.the defining matrix and 
    2. its center point
    '''
    # transform pcl into a numpy matrix
    P = np.matrix(pcl).T

    # dimension of space , here dimension is 3
    # number of points around which the ellipsoid is to be found
    N, d = pcl.shape

    # lift points from pcl to 4-dim space and write them in a matrix Q
    Q  = np.matrix(np.zeros([d + 1, N]))
    Q[0:d][:] = P
    Q[d][:] = np.ones([1, N])

    # Initialize parameters for ellipsoid fitting loop
    count = 1
    err = 1
    u = (1 / N) * np.ones((N, 1))

    # Khachiyan Algorithm to solve the so called dual problem
    while err > tolerance:
        # X is a (d+1)x(d+1) matrix
        X = Q * np.diag(u.squeeze()) * Q.T
        X_inv = np.linalg.inv(X)
        
        # M is the diagonal vector of an NxN matrix. In the referenced paper
        # it is called g(u)
        M = np.diag(Q.T * X_inv * Q)

        # j is the index where vector M has its maximum
        j = np.argmax(M)

        # caculate the stepsize (called alpha in the paper)
        step_size = (M[j] - d - 1) / ((d + 1) * (M[j] - 1))

        # update u
        new_u = (1 - step_size) * u
        new_u[j] = new_u[j] + step_size
        count = count + 1
        err = np.linalg.norm(new_u - u)
        u = new_u
    
    # Caculate the defining matrix A of the ellipsoid and its center c
    U = np.diag(u.squeeze())
    c = P * u 
    A = (1 / d) * np.linalg.inv(P * U * P.T - np.outer(c, c))
    return A, c

def inverse_sqrt_matrix(M):
    '''
    computing M^-0.5 as (M^-1)^0.5
    ''' 
    #compute eigen values and eigen vectors
    e_val, e_vec = np.linalg.eig(np.linalg.inv(M))
    e_val_sqrt = np.sqrt(e_val)
    M_minus_sqrt = e_vec * np.diag(e_val_sqrt.squeeze()) * e_vec.T
    return M_minus_sqrt
