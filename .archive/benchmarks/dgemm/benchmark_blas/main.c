/*
 * Author: 		Xiaoye S. Li, U.C. Berkeley
 * Creation date: 	Tue Mar 26 11:52:24 PST 1996
 * File name:		dgemv2.c
 * History:
 * Purpose:	Benchmark DGEMV2 kernel
 */

static char _version_string_[] = "\nVersion:1:dgemv2.c\0\n";

#include <stdio.h>
#include <stdlib.h>

double timer_();

#define CACHE_SIZE  (1024*32)   /* RS/6000 - in doubles */
#if 0
#define CACHE_SIZE  (1024*128)   /* DEC Alpha - in doubles */
#endif
#define VECTOR_MAX  (2048)
#define MATRIX_MAX  (VECTOR_MAX * VECTOR_MAX)

#define M_MAX       (1024)
#define N_MAX       (128)

#define ITER_MAX    1000
#define ITER_MIN    1
#define T_MIN       (1.0)

static double buffer[CACHE_SIZE*4]; /* overcome 4-way associativity */
static double A[MATRIX_MAX];
static double B[MATRIX_MAX];
static double C[MATRIX_MAX];
static double x1[VECTOR_MAX];
static double x2[VECTOR_MAX];
static double y1[VECTOR_MAX];
static double y2[VECTOR_MAX];
static int m, n, k, lda, ldb, ldc, iters, incx, incy;
static double alpha, beta;
static double single_tim;
static float ops;

double flops[M_MAX+1][N_MAX+1];

void main(int argc, char *argv[])
{
    int register i, j, nl, nu, nstep, ml, mu, mstep;
    register int c;
    int its;

    n = VECTOR_MAX;
    
    /* Initialize A, B, x1, x2, y1 and y2. */
    lda = n;
    for (j = 0; j < n; ++j)
      for (i = 0; i < n; ++i) {
	  A[i + j*lda] = (double) j / (i+1);
	  B[i + j*lda] = A[i + j*lda];
      }
	
    for (i = 0; i < n; ++i) {
    	x1[i] = x2[i] = 1. / (i+1);
	y1[i] = y2[i] = (double) (n - i) / n;
    }

    alpha = beta = 1;
    incx = incy = 1;

    nl = 2; nu = N_MAX; nstep = 2;
    ml = 5; mu = M_MAX; mstep = 5;

#ifdef GEMV2
    fprintf(stderr, "Benchmark DGEMV2 ... \n");
    for (m = ml; m <= mu; m += mstep) {
      lda = m;
      n = m;
      single_dgemv2(&its);
      flops[m][n] = ops*1e-6/single_tim;
      fprintf(stderr, "n=%d, m=%d, Time %s %.2f\titer %d\tflop rate %.2f\n",
	      n, m, "DGEMV2", single_tim, its, ops*1e-6/single_tim);
    }

    for (m = ml; m <= mu; m += mstep) {
      n = m;
      printf("%.2f\n", flops[m][n]);
    }
#endif    
    
#ifdef GEMV
    fprintf(stderr, "Benchmark DGEMV ... \n");
    for (m = ml; m <= mu; m += mstep) {
      lda = m;	
      n = m;
      single_dgemv(&its);
      flops[m][n] = ops*1e-6/single_tim;
      fprintf(stderr, "n=%d, m=%d, Time %s %.2f\titer %d\tflop rate %.2f\n",
	      n, m, "DGEMV", single_tim, its, ops*1e-6/single_tim);
    }

    for (m = ml; m <= mu; m += mstep) {
      n = m;
      printf("%.2f\n", flops[m][n]);
    }
#endif

#ifdef GEMM
    fprintf(stderr, "Benchmark DGEMM ... \n");
    ml = nl = 600; 
    mu = nu = 1024;
    mstep = nstep = 5;
    for (m = ml; m <= mu; m += mstep) {
      n = k = m;
      lda = ldc = m;
      ldb = k;
      single_dgemm(&its);
      flops[m][n] = ops*1e-6/single_tim;
      fprintf(stderr, "n=%d, m=%d, Time %s %.2f\titer %d\tflop \
rate %.2f\n", n, m, "DGEMM", single_tim, its, ops*1e-6/single_tim);
    }

    for (m = ml; m <= mu; m += mstep) {
      printf("%10.2f  ", flops[m][m]);
    }
#endif

}	    


/**********************
 * Time DGEMV2        *
 **********************/
single_dgemv2(int *iterations)
{
    int j, iters, its, its2;
    double t1, t, t_loop;
    
    iters = ITER_MIN;
    its = 0;
    t = 0.;
    do {
	t1 = timer_();
	for (j = 0; j < iters; ++j) {
	    dmatvec2(lda, m, n, A, x1, x2, y1, y2);
	    /*	    dgemv2_(&lda, &m, &n, A, x1, x2, y1, y2);*/
#ifdef FLUSH
	    FlushCache();
#endif
	}
	t += timer_() - t1;
	its += iters;
	iters *= 2;
    } while ( t < T_MIN );
    
    t_loop = 0;
    iters = ITER_MIN;
    its2 = 0;
    do {
	t1 = timer_();
	for (j = 0; j < iters; ++j) {
#ifdef FLUSH
	    FlushCache();
#endif	    
	}
	t_loop += timer_() - t1;
	its2 += iters;
	iters *= 2;
    } while (its2 < its);

    ops = 4.*m*n*its;
    single_tim = t - t_loop;
    *iterations = its;
}
    
/**********************
 * Time DGEMV         *
 **********************/
single_dgemv(int *iterations)
{
    int j, iters, its, its2;
    double t1, t, t_loop;
    
    iters = ITER_MIN;
    its = 0;
    t = 0.;
    do {
	t1 = timer_();
	for (j = 0; j < iters; ++j) {
#if 0
	    dmatvec(lda, m, n, A, x1, y1);
#endif
            dgemv_("N", &m, &n, &alpha, A, &lda, x1, &incx, &beta,
                   y1, &incy);
	    alpha = -alpha;
#ifdef FLUSH
	    FlushCache();
#endif
	}
	t += timer_() - t1;
	its += iters;
	iters *= 2;
    } while ( t < T_MIN );
    
    t_loop = 0;
    iters = ITER_MIN;
    its2 = 0;
    do {
	t1 = timer_();
	for (j = 0; j < iters; ++j) {
#ifdef FLUSH
	    FlushCache();
#endif	    
	}
	t_loop += timer_() - t1;
	its2 += iters;
	iters *= 2;
    } while (its2 < its);

    ops = 2.*m*n*its;
    single_tim = t - t_loop;
    *iterations = its;
}

/**********************
 * Time DGEMM         *
 **********************/
single_dgemm(int *iterations)
{
    int j, iters, its, its2;
    double t1, t, t_loop;
    
    iters = ITER_MIN;
    its = 0;
    t = 0.;
    do {
	t1 = timer_();
	for (j = 0; j < iters; ++j) {
	    dgemm_("N", "N", &m, &k, &n, &alpha, A, &lda, B, &ldb,
		   &beta, C, &ldc);
	    alpha = -alpha;
#ifdef FLUSH
	    FlushCache();
#endif
	}
	t += timer_() - t1;
	its += iters;
	iters *= 2;
    } while ( t < T_MIN );
    
    t_loop = 0;
    iters = ITER_MIN;
    its2 = 0;
    do {
	t1 = timer_();
	for (j = 0; j < iters; ++j) {
#ifdef FLUSH
	    FlushCache();
#endif	    
	}
	t_loop += timer_() - t1;
	its2 += iters;
	iters *= 2;
    } while (its2 < its);

    ops = 2.*m*n*k*its;
    single_tim = t - t_loop;
    *iterations = its;
}

rate(char *func, double t, float ops, int iter, double *junk)
{
    if ( t > 0. )
	printf("Time %s %.2f\titer %d\tflop rate %.2f\n",
	       func, t, iter, ops*1e-6/t);
    
}

ratio(char *func1, double t1, char *func2, double t2)
{
    printf("T(%s) / T(%s) %10.3g\n", func1, func2, t1 / t2);
}

FlushCache()
{
    static double alpha = 1.;
    register int i;
 
    for (i = 0; i < CACHE_SIZE; ++i) buffer[i] += alpha * buffer[i];
    alpha = -alpha;
}

use(double val, double *ptr)
{
    printf("%f\n", val);
    printf("%f\n", ptr[0]);
}
