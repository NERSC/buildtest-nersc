#
# Author:	Xiaoye S. Li
# Create Date: 	Wed Jun 29 13:17:25 PDT 1994
# Purpose:	Benchmark BLAS 2 routines
#

.SUFFIXES: .f .o .s
.KEEP_STATE:

CC        = cc
FORTRAN   = f77

############################
# DEC Alpha
############################
CFLAGS    = -migrate -O5 -DGEMV2 -tune ev5 #-DFLUSH
LOADER    = cc
#BLASOPTS  = -migrate -O5 -tune ev5 -fast -ansi_alias -fp_reorder
BLASOPTS  = -migrate -O5 -tune ev5 -ansi_alias -fp_reorder
#FFLAGS    = -fkapargs='-o=5 -scalaropt=3' -O5 -tune ev5 -fast #-ur=3
FFLAGS    = -O5 -tune ev5 -fast -ur=2
FLIBS	  = -lfor
BLASLIB     = -ldxml

#BLASLIB     = -lessl -lxlf90

############################
# SGI Power Challenge 
############################
BLASOPTS= -OPT:alias=restrict -SWP:strict_ivdep=FALSE
BLASLIB = -lblas
CFLAGS	= -O3 -TARG:proc=r10000 -DGEMV
FFLAGS	= -O3
FLIBS	=

############################
# SGI Origin2000
############################
ISA	= -n32 -mips4
CFLAGS	= -Ofast=ip27 -TARG:proc=r10000 -DGEMV2
BLASOPTS= $(ISA) -Ofast=ip27 -TARG:proc=r10000 -OPT:alias=restrict\
			-OPT:IEEE_arithmetic=3 
FFLAGS	= -O3
BLASLIB = -lcomplib.sgimath
FLIBS	=

############################
# Sun cluster
############################
CC	= tmcc
CFLAGS	= -xO3 -DGEMV2
BLASOPTS = -V -fast -native -dalign -libmil -xO4
FORTAN	= tmf77
FFLAGS	= -O3
BLASLIB	= -ls3l


############################
# Cori
############################
MKLROOT = /opt/intel/compilers_and_libraries_2019.3.199/linux/mkl
CC	= cc
FORTRAN   = ftn
CFLAGS	= -O3 -DGEMM
BLASOPTS = -O3
FFLAGS	= -O3
BLASLIB=-Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a ${MKLROOT}/lib/intel64/libmkl_gnu_thread.a ${MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -lgomp -lpthread -lm -ldl
# BLASLIB=-Wl,--start-group /opt/cray/pe/libsci/19.06.1/GNU/8.1/x86_64/lib/libsci_gnu_82_mpi_mp.so -Wl,--end-group -lgomp -lpthread -lm -ldl
# BLASLIB=-Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a ${MKLROOT}/lib/intel64/libmkl_intel_thread.a ${MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -liomp5 -lpthread -lm -ldl
# BLASLIB= -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a ${MKLROOT}/lib/intel64/libmkl_sequential.a ${MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -lpthread -lm -ldl
ISA = 



BLA2OBJS = main.o dmyblas2.o timer.o #dgemv2.o

run:	$(BLA2OBJS)
	$(CC) $(VERBOSE) $(CFLAGS) -o run \
	$(BLA2OBJS) $(FLIBS) $(BLASLIB) 

dmyblas2.o: dmyblas2.c
	$(CC) $(BLASOPTS) -c $< $(VERBOSE)

.c.o:
	$(CC) $(CFLAGS) -c $< $(VERBOSE)

dgemv2.o: dgemv2.f
	$(FORTRAN) $(FFLAGS) -c $< $(VERBOSE)

timer.o:  timer.c ; cc -c $(ISA) $<

clean:	
	rm -f *.o *.a core bench
