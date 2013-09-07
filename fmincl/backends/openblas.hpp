/* ===========================
 *
 * Copyright (c) 2013 Philippe Tillet - National Chiao Tung University
 *
 * FMinCL - Unconstrained Function Minimization on OpenCL
 *
 * License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/


#ifndef FMINCL_BACKENDS_OPENBLAS_HPP
#define FMINCL_BACKENDS_OPENBLAS_HPP

#include <cstring>

#include "cblas.h"

namespace fmincl{

  namespace backend{

    template<class _ScalarType>
    struct OpenBlasTypes;

    template<>
    struct OpenBlasTypes<float>{
        typedef float ScalarType;
        typedef ScalarType* VectorType;
        typedef ScalarType* MatrixType;

        static VectorType create_vector(std::size_t N)
        { return new ScalarType[N]; }
        static MatrixType create_matrix(std::size_t M, std::size_t N)
        { return new ScalarType[M*N]; }
        static void delete_if_dynamically_allocated(ScalarType* p)
        { delete[] p;}

        static void copy(std::size_t N, VectorType const & from, VectorType & to)
        { cblas_scopy(N,from,1,to,1); }
        static void axpy(std::size_t N, ScalarType alpha, VectorType const & x, VectorType & y)
        { cblas_saxpy(N,alpha,x,1,y,1); }
        static void scale(std::size_t N, ScalarType alpha, VectorType & x)
        { cblas_sscal(N,alpha,x,1); }
        static void scale(std::size_t M, std::size_t N, ScalarType alpha, MatrixType & A)
        { cblas_sscal(M*N,alpha,A,1); }
        static ScalarType asum(std::size_t N, VectorType const & x)
        { return cblas_sasum(N,x,1);}
        static ScalarType nrm2(std::size_t N, VectorType const & x)
        { return cblas_snrm2(N,x,1); }
        static ScalarType dot(std::size_t N, VectorType const & x, VectorType const & y)
        { return cblas_sdot(N,x,1,y,1); }
        static void symv(std::size_t N, ScalarType alpha, MatrixType const& A, VectorType const & x, ScalarType beta, VectorType & y)
        { cblas_ssymv(CblasRowMajor,CblasUpper,N,alpha,A,N,x,1,beta,y,1);  }
        static void syr1(std::size_t N, ScalarType const & alpha, VectorType const & x, MatrixType & A)
        { cblas_ssyr(CblasRowMajor,CblasUpper,N,alpha,x,1,A,N); }
        static void syr2(std::size_t N, ScalarType const & alpha, VectorType const & x, VectorType const & y, MatrixType & A)
        { cblas_ssyr2(CblasRowMajor,CblasUpper,N,alpha,x,1,y,1,A,N); }
        static void set_to_identity(std::size_t N, MatrixType & A) {
            std::memset(A,0,N*N);
            for(std::size_t i = 0 ; i < N ; ++i){
                A[i*N+i] = 1;
            }
        }
    };


    template<>
    struct OpenBlasTypes<double>{
        typedef double ScalarType;
        typedef ScalarType* VectorType;
        typedef ScalarType* MatrixType;

        static VectorType create_vector(std::size_t N)
        { return new ScalarType[N]; }
        static MatrixType create_matrix(std::size_t M, std::size_t N)
        { return new ScalarType[M*N]; }
        static void delete_if_dynamically_allocated(ScalarType* p)
        { delete[] p;}

        static void copy(std::size_t N, VectorType const & from, VectorType & to)
        { cblas_dcopy(N,from,1,to,1); }
        static void axpy(std::size_t N, ScalarType alpha, VectorType const & x, VectorType & y)
        { cblas_daxpy(N,alpha,x,1,y,1); }
        static void scale(std::size_t N, ScalarType alpha, VectorType & x)
        { cblas_dscal(N,alpha,x,1); }
        static void scale(std::size_t M, std::size_t N, ScalarType alpha, MatrixType & A)
        { cblas_dscal(M*N,alpha,A,1); }
        static ScalarType asum(std::size_t N, VectorType const & x)
        { return cblas_dasum(N,x,1);}
        static ScalarType nrm2(std::size_t N, VectorType const & x)
        { return cblas_dnrm2(N,x,1); }
        static ScalarType dot(std::size_t N, VectorType const & x, VectorType const & y)
        { return cblas_ddot(N,x,1,y,1); }
        static void symv(std::size_t N, ScalarType alpha, MatrixType const& A, VectorType const & x, ScalarType beta, VectorType & y)
        { cblas_dsymv(CblasRowMajor,CblasUpper,N,alpha,A,N,x,1,beta,y,1);  }
        static void syr1(std::size_t N, ScalarType const & alpha, VectorType const & x, MatrixType & A)
        { cblas_dsyr(CblasRowMajor,CblasUpper,N,alpha,x,1,A,N); }
        static void syr2(std::size_t N, ScalarType const & alpha, VectorType const & x, VectorType const & y, MatrixType & A)
        { cblas_dsyr2(CblasRowMajor,CblasUpper,N,alpha,x,1,y,1,A,N); }
        static void set_to_identity(std::size_t N, MatrixType & A) {
            std::memset(A,0,N*N);
            for(std::size_t i = 0 ; i < N ; ++i){
                A[i*N+i] = 1;
            }
        }
    };


  }

}

#endif
