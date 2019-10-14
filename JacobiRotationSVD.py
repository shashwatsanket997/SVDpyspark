'''
    Author : Shashwat Sanket
    Description: This Script is the implementation of the QR Decomposition using (Householder decompositon method)
    Data of Code : 8 oct 2019
    PC: Sanket
'''
import time
import os
import sys
import numpy as np
from QRDecomposition import QRDecomposition as QR
from NormalJacobi import NormalJacobi as jacobi_svd
# Reading the filename as CLI argument
argvs = sys.argv
print(argvs)
if(len(argvs) != 2):
    sys.exit("Insufficient Arguments ")
# Get the current working directory
cwd = os.getcwd()
# Reading matrix A(filename: argv[0])
matA_file = os.path.join(cwd, sys.argv[1])
# Processing the input file as each line(, as  deliminator) of the file reperesents the row of the matrix
# Using the numpy library
matrix = np.loadtxt(matA_file, delimiter=",")
# Getting the rows and columns

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})


def JacobianRotationSVD(matrix):
    # Input mxn matrix
    # Output [U S V]
    # Finding the housholder decomposition of the Matrix
    # start = time.time()

    # numpy_svd = np.linalg.svd(matrix)
    # print(numpy_svd)
    # k = time.time()-start
    # start = time.time()
    _, normal_svd, _, normal_itr = jacobi_svd(matrix)
    # m = time.time()-start
    print("Total Iteration taken for Normal Matrix", normal_itr)
    print(normal_svd)

    # SVD after QR decomposition
    _, matrix_r = QR(matrix)
    _, qr_svd, _, qr_itr = jacobi_svd(matrix_r)
    print("Total Iteration taken for QR decomposed matrix:", qr_itr)
    print(qr_svd)


JacobianRotationSVD(matrix)
