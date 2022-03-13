/**
 * @file main.cc
 * @author Lucas de Assis (lucasforeal@gmail.com)
 * @brief assumptions: matrices are square and with less than 32,767*
 * @version 0.1
 * @date 2022-02-18
 * 
 * @copyright Copyright (c) 2022
 * 
 * $Smake: g++ -O3 -o %F %f -lcblas -latlas -lhdf5
 */
#include <iostream>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <hdf5.h>
#include <cstdlib>
#include <cmath>
#include "wtime.c"
#include <cstring>
extern "C" {
  #include <cblas.h>
}

using std::cout;
using std::endl;

#define IDX(i,j,stride) ((i)*(stride)+(j)) // row major (C/C++)

#define CHKERR(status,name) if (status) \
fprintf(stderr, "Warning: nonzero status (%d) in %s\n", status, name)

/*
 * Display string showing how to run program from command line
 *
 * Input:
 *   char* program_name (in)  name of executable
 * Output:
 *   writes to stderr
 * Returns:
 *   nothing
 */
void usage(char* program_name) {
  fprintf(stderr,
          "Usage: %s [-c] [-e epsilon] [-m num_iterations] input-file\n",
          program_name);
}

/*---------------------- CALCULATION FOR LOOP FUNCTIONS -------------------*/
/* ||x|| */
double getTwoNorm(double *x, short size) {
  double norm = 0.0;
  for (short i = 0; i < size; i++) {
    norm += x[i] * x[i];
  }
  return sqrt(norm);
}

/* x = x / ||x|| */
void normalizeEigenvectorEstimate(double *x, short size) {
  double norm = getTwoNorm(x, size);
  for (short i = 0; i < size; i++) {
    x[i] /= norm;
  }
}

/* y = Ax */
void computeNextEigenvectorEstimate(double *y, double *a, double *x, short size) {
  for (short i = 0; i < size; i++) {
    y[i] = 0.0;
    for (short j = 0; j < size; j++) {
      y[i] += a[IDX(i, j, size)] * x[j]; 
    }
  }
}

/* x^T y */
double getInnerProduct(double *x, double *y, short size) {
  double dot_product = 0.0;
  for (short i = 0; i < size; i++) {
    dot_product += x[i] * y[i];
  }
  return dot_product;
}

/*----------------------------- HDF5 READ FUNCTION ---------------------------------*/
void readMatrix(char* fname, char* hdf5Path, double** a, int *size) {
  /* Open existing HDF5 file */
  hid_t file_id = H5Fopen(fname, H5F_ACC_RDONLY, H5P_DEFAULT);

  /* Open existing first dataset */
  hid_t dataset_id = H5Dopen(file_id, hdf5Path, H5P_DEFAULT);

  /* Determine dataset parameters */
  hid_t file_dataspace_id = H5Dget_space(dataset_id);
  int rank = H5Sget_simple_extent_ndims(file_dataspace_id);
  hsize_t* dims = (hsize_t*) malloc(rank * sizeof(hsize_t));
  int ndims = H5Sget_simple_extent_dims(file_dataspace_id, dims, NULL);
  if (ndims != rank) {
    fprintf(stderr, "Warning: expected dataspace to be dimension ");
    fprintf(stderr, "%d but appears to be %d\n", rank, ndims);
  }

  /* Allocate matrix */
  hsize_t num_elem = H5Sget_simple_extent_npoints(file_dataspace_id);
  *a = (double*) malloc(num_elem * sizeof(double));
  *size = dims[0]; /* reversed since we're using Fortran-style ordering */

  /* Create dataspace */
  hid_t dataspace_id = H5Screate_simple(rank, dims, NULL);

  /* Read matrix data from file */
  herr_t status = H5Dread(dataset_id,
                          H5T_NATIVE_DOUBLE,
                          dataspace_id,
                          file_dataspace_id,
                          H5P_DEFAULT,
                          *a);
  CHKERR(status, "H5Dread()");

  /* Close resources */
  status = H5Sclose(dataspace_id);
  CHKERR(status, "H5Sclose()");
  status = H5Sclose(file_dataspace_id);
  CHKERR(status, "H5Sclose()");
  status = H5Dclose(dataset_id);
  CHKERR(status, "H5Dclose()");
  status = H5Fclose(file_id);
  CHKERR(status, "H5Fclose()");
  free(dims);
}

/*----------------------------------- MAIN -----------------------------------------*/
int main(int argc, char **argv) {
  double epsilon = 0.000001;
  int m = 1000;
  int opt;
  bool cblas = false;
  if (argc == 1) {
    usage(argv[0]);
    return EXIT_FAILURE;
  }
  while ((opt = getopt(argc, argv, "e:m:c")) != -1) {
    switch (opt) {
      case 'e':
        epsilon = atof(optarg);
        break;
      case 'm':
        m = atoi(optarg);
        break;
      case 'c':
        cblas = true;
        break;
      default: /* '?' */
        usage(argv[0]);
        return EXIT_FAILURE;
      }
  }

  // Read input matrix from file into variable a, and compte the time this will take
  double *a;
  int size;
  double readT1 = wtime();
  readMatrix(argv[optind], (char *)"/A/value", &a, &size);
  double readT2 = wtime();
  double x[size];
  double y[size];

  // After setting everything up, let's start tracking elapsed time
  double computeT1 = wtime();

  // Initialize and normalize eigenvector estimate
  double initialValue = 1 / sqrt(size);
  for (short i = 0; i < size; i++) {
    x[i] = initialValue;
  }

  // Initialize eigenvalue estimate
  double lambda = 0.0;

  // Make sure |lambda - lambda_0| > epsilon
  double lambda_0 = lambda + 2.0 * epsilon;

  // Initialize loop counter
  short k = 0;
  if (cblas) {
    while (abs(lambda - lambda_0) >= epsilon && k < 1000) {
      
      // Update counter
      k++;

      // Compute next eigenvector estimate
      cblas_dgemv(CblasRowMajor,
                  CblasNoTrans,
                  size,
                  size,
                  1.0,
                  a,
                  size,
                  x,
                  1,
                  0.0,
                  y,
                  1);

      // Save previous eigenvalue estimate
      lambda_0 = lambda;

      // Compute new estimate: lamda ~= x^T Ax
      lambda = cblas_ddot(size, x, 1, y, 1);

      // Update eigenvector estimate
      cblas_dcopy(size, y, 1, x, 1);

      // Normalize eigenvector estimate
      cblas_dscal(size,
                  1 / cblas_dnrm2(size, x, 1),
                  x,
                  1);
    }
  } else {
    while (abs(lambda - lambda_0) >= epsilon && k < m) {

      /*** For error checking ***/
      // printf("abs(lambda - lambda_0): %f\tlambda: %f\tlambda_0: %f\n",
      //        abs(lambda - lambda_0),
      //        lambda,
      //        lambda_0);
      
      // Update counter
      k++;
      computeNextEigenvectorEstimate(y, a, x, size);
      
      // Save previous eigenvalue estimate
      lambda_0 = lambda;

      // Compute new estimate: lamda ~= x^T Ax
      lambda = getInnerProduct(x, y, size);

      // Update eigenvector estimate
      memcpy(x, y, size_t(sizeof(y)));

      normalizeEigenvectorEstimate(x, size);
    }
  }
  
  // Calculate time taken for calculations, and print results
  double computeT2 = wtime();

  if (k == m) {
    printf("Convergence was not achieved after %d iterations! \n" \
           "Quitting.\n",
           m);
    return EXIT_FAILURE;
  }
  printf("eigenvalue = %f found in %d iterations\n" \
         "elapsed read time    =   %f seconds\n" \
         "elapsed compute time =   %f seconds\n",
         lambda,
         k,
         readT2 - readT1,
         computeT2 - computeT1);

  // Cleanup
  delete a;
}