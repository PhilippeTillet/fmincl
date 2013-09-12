/* ===========================
 *
 * Copyright (c) 2013 Philippe Tillet - National Chiao Tung University
 *
 * FMinCL - Unconstrained Function Minimization on OpenCL
 *
 * License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/


#ifndef FMINCL_BACKENDS_BLAS_HPP
#define FMINCL_BACKENDS_BLAS_HPP

#include <cstring>
#include "blas.h"

namespace fmincl{

  namespace backend{

    static const char Upper = 'U';
    static const char Lower = 'L';
    static const long int one_inc = 1;

    template<class _ScalarType>
    struct blas_types;

    template<>
    struct blas_types<float>{
        typedef float ScalarType;
        typedef ScalarType* VectorType;
        typedef ScalarType* MatrixType;
        typedef long int size_t;
    private:
        typedef VectorType& vec_ref;
    public:


        static VectorType create_vector(std::size_t N)
        { return new ScalarType[N]; }
        static MatrixType create_matrix(std::size_t M, std::size_t N)
        { return new ScalarType[M*N]; }
        static void delete_if_dynamically_allocated(ScalarType* p)
        { delete[] p;}

        static void copy(size_t N, VectorType const & from, VectorType & to)
        { scopy_(&N,(vec_ref)from,(size_t*)&one_inc,to,(size_t*)&one_inc); }
        static void axpy(size_t N, ScalarType alpha, VectorType const & x, VectorType & y)
        { saxpy_(&N,&alpha,(vec_ref)x,(size_t*)&one_inc,y,(size_t*)&one_inc); }
        static void scale(size_t N, ScalarType alpha, VectorType & x)
        { sscal_(&N,&alpha,x,(size_t*)&one_inc); }
        static void scale(size_t M, size_t N, ScalarType alpha, MatrixType & A)
        { size_t K=M*N; sscal_(&K,&alpha,A,(size_t*)&one_inc); }
        static ScalarType asum(size_t N, VectorType const & x)
        { return sasum_(&N,(vec_ref)x,(size_t*)&one_inc);}
        static ScalarType nrm2(size_t N, VectorType const & x)
        { return snrm2_(&N,(vec_ref)x,(size_t*)&one_inc); }
        static ScalarType dot(size_t N, VectorType const & x, VectorType const & y)
        { return sdot_(&N,(vec_ref)x,(size_t*)&one_inc,(vec_ref)y,(size_t*)&one_inc); }
        static void symv(size_t N, ScalarType alpha, MatrixType const& A, VectorType const & x, ScalarType beta, VectorType & y)
        { ssymv_((char*)&Lower,&N,&alpha,A,&N,(vec_ref)x,(size_t*)&one_inc,&beta,y,(size_t*)&one_inc);  }
        static void syr1(size_t N, ScalarType alpha, VectorType const & x, MatrixType & A)
        { ssyr_((char*)&Lower,&N,&alpha,(vec_ref)x,(size_t*)&one_inc,A,&N); }
        static void syr2(size_t N, ScalarType  alpha, VectorType const & x, VectorType const & y, MatrixType & A)
        { ssyr2_((char*)&Lower,&N,&alpha,(vec_ref)x,(size_t*)&one_inc,(vec_ref)y,(size_t*)&one_inc,A,&N); }
        static void set_to_value(VectorType & V, ScalarType val, size_t N)
        { std::memset(V, val, sizeof(ScalarType)*N); }
        static void set_to_diagonal(size_t N, MatrixType & A, ScalarType lambda) {
            std::memset(A,0,N*N*sizeof(ScalarType));
            for(size_t i = 0 ; i < N ; ++i){
                A[i*N+i] = lambda;
            }
        }
    };


    template<>
    struct blas_types<double>{
        typedef double ScalarType;
        typedef ScalarType* VectorType;
        typedef ScalarType* MatrixType;
        typedef long int size_t;

    private:
        typedef VectorType& vec_ref;
    public:


        static VectorType create_vector(std::size_t N)
        { return new ScalarType[N]; }
        static MatrixType create_matrix(std::size_t M, std::size_t N)
        { return new ScalarType[M*N]; }
        static void delete_if_dynamically_allocated(ScalarType* p)
        { delete[] p;}

        static void copy(size_t N, VectorType const & from, VectorType & to)
        { dcopy_(&N,(vec_ref)from,(size_t*)&one_inc,to,(size_t*)&one_inc); }
        static void axpy(size_t N, ScalarType alpha, VectorType const & x, VectorType & y)
        { daxpy_(&N,&alpha,(vec_ref)x,(size_t*)&one_inc,y,(size_t*)&one_inc); }
        static void scale(size_t N, ScalarType alpha, VectorType & x)
        { dscal_(&N,&alpha,x,(size_t*)&one_inc); }
        static void scale(size_t M, size_t N, ScalarType alpha, MatrixType & A)
        { size_t K=M*N; dscal_(&K,&alpha,A,(size_t*)&one_inc); }
        static ScalarType asum(size_t N, VectorType const & x)
        { return dasum_(&N,(vec_ref)x,(size_t*)&one_inc);}
        static ScalarType nrm2(size_t N, VectorType const & x)
        { return dnrm2_(&N,(vec_ref)x,(size_t*)&one_inc); }
        static ScalarType dot(size_t N, VectorType const & x, VectorType const & y)
        { return ddot_(&N,(vec_ref)x,(size_t*)&one_inc,(vec_ref)y,(size_t*)&one_inc); }
        static void symv(size_t N, ScalarType alpha, MatrixType const& A, VectorType const & x, ScalarType beta, VectorType & y)
        { dsymv_((char*)&Lower,&N,&alpha,A,&N,(vec_ref)x,(size_t*)&one_inc,&beta,y,(size_t*)&one_inc);  }
        static void syr1(size_t N, ScalarType alpha, VectorType const & x, MatrixType & A)
        { dsyr_((char*)&Lower,&N,&alpha,(vec_ref)x,(size_t*)&one_inc,A,&N); }
        static void syr2(size_t N, ScalarType  alpha, VectorType const & x, VectorType const & y, MatrixType & A)
        { dsyr2_((char*)&Lower,&N,&alpha,(vec_ref)x,(size_t*)&one_inc,(vec_ref)y,(size_t*)&one_inc,A,&N); }
        static void set_to_value(VectorType & V, ScalarType val, size_t N)
        { std::memset(V, val, sizeof(ScalarType)*N); }
        static void set_to_diagonal(size_t N, MatrixType & A, ScalarType lambda) {
            std::memset(A,0,N*N*sizeof(ScalarType));
            for(size_t i = 0 ; i < N ; ++i){
                A[i*N+i] = lambda;
            }
        }
    };


  }

}

#endif
