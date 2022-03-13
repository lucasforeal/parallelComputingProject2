#!/usr/bin/env python3
"""Computes dominant eigenvalue of matrix in HDF5 file via Power Method

   This program reads data for a square NxN matrix from an HDF5
   data file and uses the Power Method to compute an estimate of
   the dominant eigenvalue.  It assumes the matrix is stored in
   row-major order (C/C++/Python).
"""

import sys, time, argparse
import h5py as h5
import numpy as np

def usage(name):
    """Display program usage

    Parameters
    ----------
    name : str
        The path of the executable given on the command line
    """
    
    print('Usage: {} [-v] [-e tol] [-m maxiter] filename'.format(name))

def main():
    # Process command line
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--tolerance', type=float, default=1e-6)
    parser.add_argument('-m', '--maxiter', type=int, default=1000)
    parser.add_argument('-v', dest='verbosity', action='count', default=0)
    parser.add_argument('filename')
    args = parser.parse_args()

    # Copy arg values to new variables
    tolerance = args.tolerance
    maxiter = args.maxiter
    verbosity = args.verbosity
    filename = args.filename

    # Report command line parameters
    if verbosity > 0:
        print('tolerance = {}'.format(tolerance))
        print('maximum number of iterations = {}'.format(maxiter))

    # Read matrix from HDF5 file
    t1 = time.time()
    f = h5.File(filename, 'r')
    A = np.array(f['A/value'])
    (n,m) = np.shape(A)
    t2 = time.time()
    read_time = t2 - t1

    # Initialize estimate of normalized eigenvector
    x = np.ones(m) / np.sqrt(m)

    # Main power method loop
    lambda_new = 0.0
    lambda_old = lambda_new + 2 * tolerance
    delta = np.abs(lambda_new - lambda_old)
    numiter = 0
    t1 = time.time()
    while delta >= tolerance and numiter <= maxiter:
        numiter += 1

        # compute new eigenvector estimate y = A*x
        y = np.matmul(A,x)

        # compute new estimate of eigenvalue lambda = x'Ax
        lambda_old = lambda_new
        lambda_new = x.dot(y)

        # update estimated normalized eigenvector
        x = y / np.linalg.norm(y)

        delta = np.abs(lambda_new - lambda_old)
        if verbosity > 1:
            print('numiter = {:3d}:'.format(numiter),
                  'lambda = {:10.7f},'.format(lambda_new),
                  'diff = {:.6e}'.format(delta))

    t2 = time.time()
    compute_time = t2 - t1

    # Report
    if numiter > maxiter:
        print('*** WARNING ****: maximum number of iterations exceeded',
                file=sys.stderr)

    print('eigenvalue = {:.6f}'.format(lambda_new),
          'found in {} iterations'.format(numiter))
    print('elapsed read time    = {:10.6f} seconds'.format(read_time))
    print('elapsed compute time = {:10.6f} seconds'.format(compute_time))

if __name__ == '__main__':
    main()
