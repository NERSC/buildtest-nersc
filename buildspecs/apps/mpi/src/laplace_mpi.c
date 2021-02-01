/****************************************************************
 * Shahzeb Siddiqui
 * Date: 06/18/2015
 * Event: XSEDE Summer Boot Camp June 16-19, 2015
 *
 * Laplace MPI Template C Version                                         
 *                                                               
 * T is initially 0.0                                            
 * Boundaries are as follows                                     
 *                                                               
 *                T                      4 sub-grids            
 *   0  +-------------------+  0    +-------------------+       
 *      |                   |       |                   |           
 *      |                   |       |-------------------|         
 *      |                   |       |                   |      
 *   T  |                   |  T    |-------------------|             
 *      |                   |       |                   |     
 *      |                   |       |-------------------|            
 *      |                   |       |                   |   
 *   0  +-------------------+ 100   +-------------------+         
 *      0         T       100                                    
 *                                                                 
 * Each PE only has a local subgrid.
 * Each PE works on a sub grid and then sends         
 * its boundaries to neighbors.
 *                                                                 
 *  John Urbanic, PSC 2014
 *
 *******************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include <mpi.h>

#define COLUMNS      1000
#define ROWS_GLOBAL  1000        // this is a "global" row count
#define NPES            4        // number of processors
#define ROWS (ROWS_GLOBAL/NPES)  // number of real local rows

// communication tags
#define DOWN     100
#define UP       101   

#define MAX_TEMP_ERROR 0.01

double Temperature[ROWS+2][COLUMNS+2];
double Temperature_last[ROWS+2][COLUMNS+2];

void initialize(int npes, int my_PE_num);
void track_progress(int iter);


int main(int argc, char *argv[]) {

    int i, j;
    int iteration=1;
    double dt;
    struct timeval start_time, stop_time, elapsed_time;

    int        npes;                // number of PEs
    int        my_PE_num;           // my PE number
    double     dt_global=100;       // delta t across all PEs
    int        max_iterations = 500; // number of iterations
    MPI_Status status;              // status returned by MPI calls

    // the usual MPI startup routines
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&my_PE_num);
    MPI_Comm_size(MPI_COMM_WORLD,&npes);
    // verify only NPES PEs are being used 
    if (npes > NPES)
    {
	if (my_PE_num == 0)
		printf("MPI Program using %d Processes, application must run with %d Processes\n",npes,NPES);
	exit(1);
    }
    /*
    // PE 0 asks for input
    if (my_PE_num == 0)
    {
	printf("[Max iterations 100-4000]?\n");
	scanf("%d", &max_iterations);
    }
    */
    // bcast max iterations to other PEs
    MPI_Bcast(&max_iterations,1,MPI_INT, 0,MPI_COMM_WORLD);
    printf("max iterations %d from proc %d\n",max_iterations,my_PE_num);
    if (my_PE_num==0) gettimeofday(&start_time,NULL);

    initialize(npes, my_PE_num);

    while ( dt_global > MAX_TEMP_ERROR && iteration <= max_iterations ) {

        // main calculation: average my four neighbors
        for(i = 1; i <= ROWS; i++) {
            for(j = 1; j <= COLUMNS; j++) {
                Temperature[i][j] = 0.25 * (Temperature_last[i+1][j] + Temperature_last[i-1][j] +
                                            Temperature_last[i][j+1] + Temperature_last[i][j-1]);
            }
        }

        // COMMUNICATION PHASE: send and receive ghost rows for next iteration
	// send ghost cell down
	if (my_PE_num != npes-1)
        {
	  MPI_Send(&Temperature[ROWS][1],COLUMNS,MPI_DOUBLE,my_PE_num+1,DOWN,MPI_COMM_WORLD);
	}
	if (my_PE_num != 0)
	{
	  MPI_Recv(&Temperature_last[0][1],COLUMNS,MPI_DOUBLE,my_PE_num-1,DOWN,MPI_COMM_WORLD,&status);
        }
	// send ghost cell up
	if (my_PE_num != 0)
	{
	  MPI_Send(&Temperature[0][1],COLUMNS,MPI_DOUBLE,my_PE_num-1,DOWN,MPI_COMM_WORLD);
	}
	if (my_PE_num != npes-1)
	{
	  MPI_Recv(&Temperature_last[ROWS+1][1],COLUMNS,MPI_DOUBLE,my_PE_num+1,DOWN,MPI_COMM_WORLD,&status);
	}
        dt = 0.0;

        for(i = 1; i <= ROWS; i++){
            for(j = 1; j <= COLUMNS; j++){
	        dt = fmax( fabs(Temperature[i][j]-Temperature_last[i][j]), dt);
	        Temperature_last[i][j] = Temperature[i][j];
            }
        }

        // find global dt
	MPI_Allreduce(&dt,&dt_global,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
        // periodically print test values - only for PE in lower corner
        if((iteration % 100) == 0) {
            if (my_PE_num == npes-1){
		printf("dt_global: %f\n",dt_global);
                track_progress(iteration);
	    }
        }

	iteration++;
    }

    // Slightly more accurate timing and cleaner output 
    MPI_Barrier(MPI_COMM_WORLD);

    // PE 0 finish timing and output values
    if (my_PE_num==0){
        gettimeofday(&stop_time,NULL);
	timersub(&stop_time, &start_time, &elapsed_time);

	printf("\nMax error at iteration %d was %f\n", iteration-1, dt_global);
	printf("Total time was %f seconds.\n", elapsed_time.tv_sec+elapsed_time.tv_usec/1000000.0);
    }

    MPI_Finalize();
}



void initialize(int npes, int my_PE_num){

    double tMin, tMax;  //Local boundary limits
    int i,j;

    for(i = 0; i <= ROWS+1; i++){
        for (j = 0; j <= COLUMNS+1; j++){
            Temperature_last[i][j] = 0.0;
        }
    }

    // Local boundry condition endpoints
    tMin = my_PE_num*100.0/npes;
    tMax = (my_PE_num+1)*100.0/npes;

    for(i = 0; i <= ROWS+1; i++)
    {
	Temperature_last[i][0]=0.0;
	Temperature_last[i][COLUMNS+1]= tMin + ((tMax-tMin)/ROWS)*i;
    }
    if (my_PE_num == 0)
      for (j = 0; j <= COLUMNS+1; j++)
	 Temperature_last[0][j] = 0.0;
    if (my_PE_num == npes-1)
      for (j = 0; j <= COLUMNS+1; j++)
	Temperature_last[ROWS+1][j] = (100.0/COLUMNS)*j;		
}


// only called by last PE
void track_progress(int iteration) {

    int i;

    printf("---------- Iteration number: %d ------------\n", iteration);

    // output global coordinates so user doesn't have to understand decomposition
    for(i = 5; i >= 0; i--) {
      printf("[%d,%d]: %5.2f  ", ROWS_GLOBAL-i, COLUMNS-i, Temperature[ROWS-i][COLUMNS-i]);
    }
    printf("\n");
}
