#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <mpi.h>

///////////////////////////////////////////////////////
// Global Variables                                  //
///////////////////////////////////////////////////////
int N;            // Array dimensions (NxN)
int num_procs;    // Number of processors (must be a perfect square)
int block_size;   // N/sqrt(num_procs) - dimensions of each block assigned to each processor

///////////////////////////////////////////////////////
// Functions                                         //
///////////////////////////////////////////////////////
static void program_abort(char *exec_name, char *message);
static void print_usage();
static int is_perfect_square(int);
static int p_divides_N(int, int);
void my_2D_rank(int, int *, int *);
void local2global(int, int, int, int *, int *);
void MatrixMultiply(double *, double *, double *, int);

// Abort, printing the usage information only if the
// first argument is non-NULL (and set to argv[0]), and
// printing the second argument regardless.
static void program_abort(char *exec_name, char *message) {
  int my_rank;
  MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
  if (my_rank == 0) {
    if (message) {
      fprintf(stderr,"%s",message);
    }
    if (exec_name) {
      print_usage(exec_name);
    }
  }
  MPI_Abort(MPI_COMM_WORLD, 1);
  exit(1);
}

// Print the Usage information
static void print_usage(char *exec_name) {
  int my_rank;
  MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

  if (my_rank == 0) {
    fprintf(stderr,"Usage: mpirun --cfg=smpi/bcast:mpich --cfg=smpi/running_power:1Gf -np <num processes>\n");
    fprintf(stderr,"              -platform <XML platform file> -hostfile <host file> %s [-c <chunk size>] [-s <message string>]\n",exec_name);
    fprintf(stderr,"MPIRUN arguments:\n");
    fprintf(stderr,"\t<num processes>: number of MPI processes\n");
    fprintf(stderr,"\t<XML platform file>: a simgrid platform description file\n");
    fprintf(stderr,"\t<host file>: MPI host file with host names from the XML platform file\n");
    fprintf(stderr,"PROGRAM arguments:\n");
    fprintf(stderr,"\t[-c <chunk size>]: chunk size in bytes for message splitting\n");
    fprintf(stderr,"\t[-s <message string]>: arbitrary text to be printed out (no spaces)\n");
    fprintf(stderr,"\n");
  }
}

///////////////////////////
////// Main function //////
///////////////////////////

int main(int argc, char *argv[])
{
  int i,j;
  // Parse command-line arguments (not using getopt because not thread-safe
  // and annoying anyway). The code below ignores extraneous command-line
  // arguments, which is lame, but we're not in the business of developing
  // a cool thread-safe command-line argument parser.

  MPI_Init(&argc, &argv);
  
  // Determine rank and number of processes
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
  
  // Ensure num_procs is a perfect square
  if (!is_perfect_square(num_procs)) {
    if (rank ==0) {
      fprintf(stderr, "Invalid Number of Processors. Must be a perfect square.\n");
    }
    MPI_Abort(MPI_COMM_WORLD, 1);
    exit(1);
  }
  
  // Array dimension (N) argument required
  for (i=1; i < argc; i++) {
    if (!strcmp(argv[i],"-N")) {
      if ((i+1 >= argc) || (sscanf(argv[i+1],"%d",&N) != 1)) {
        program_abort(argv[0],"Invalid <N> argument (Array Dimension)\n");
      }
    }
  }
  
  // Ensure sqrt(num_procs) divides N
  if (!p_divides_N(num_procs, N)) {
    if (rank == 0) {
      fprintf(stderr, "Invalid <num processes> argument. The square root of the number of processors should divide N.\n");
    }
    MPI_Abort(MPI_COMM_WORLD, 1);
    exit(1);
  }
  
  // Message string optional argument
  char *message_string = "";
  for (i=1; i < argc; i++) {
    if (!strcmp(argv[i],"-s")) {
      if ((i+1 >= argc)) {
        program_abort(argv[0],"Invalid <message string> argument\n");
      } else {
        message_string = strdup(argv[i+1]);
      }
    }
  }
  
  block_size = N/sqrt(num_procs);
  
  int myRow, myCol;
  my_2D_rank(rank, &myRow, &myCol);
  
  int globalRow, globalCol;
  local2global(rank, 0, 0, &globalRow, &globalCol);
  
  int NUM_BYTES = block_size*block_size;
  double *A = malloc(sizeof(double) * NUM_BYTES);
  double *B = malloc(sizeof(double) * NUM_BYTES);
  double *C = malloc(sizeof(double) * NUM_BYTES);

  double *bufferA = malloc(sizeof(double) * NUM_BYTES);
  double *bufferB = malloc(sizeof(double) * NUM_BYTES);
  
  // Fill arrays with appropriate values
  for (int i = 0; i < block_size ; i++) {
    for (int j = 0; j < block_size; j++) {
      
      // A(i,j) = i
      A[i*block_size+j] = globalRow + i;
      
      // B(i,j) = i+j
      B[i*block_size+j] = (globalRow + i) + (globalCol + j);
      
      // C(i,j) = 0
      C[i*block_size+j] = 0.0;
      bufferA[i*block_size+j] = 0.0;
      bufferB[i*block_size+j] = 0.0;
    }
  }
  
  // Create communicator groups
  int comm_size = sqrt(num_procs);
  
  int ranks_col[comm_size];
  int ranks_row[comm_size];
  
  MPI_Group group_rows, group_cols;
  MPI_Comm comm_rows, comm_cols;
  
  /* Extract the original group handle */
  MPI_Group orig_group;
  MPI_Comm_group(MPI_COMM_WORLD, &orig_group);

  for (int i = 0; i < comm_size; i++) {
    for (int j = 0; j < comm_size; j++) {
      ranks_row[j] = (i * comm_size) + j;
      ranks_col[j] = i + (comm_size * j);
    }
    
    // Create communicators across rows
    if (myRow == i) {
      MPI_Group_incl(orig_group, comm_size, ranks_row, &group_rows);
      MPI_Comm_create(MPI_COMM_WORLD, group_rows, &comm_rows);
    }

    // Create communicators across columns
    if (myCol == i) {
      MPI_Group_incl(orig_group, comm_size, ranks_col, &group_cols);
      MPI_Comm_create(MPI_COMM_WORLD, group_cols, &comm_cols);
    }
    
  }
  
  // Start the timer
  double start_time;
  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0) {
    start_time = MPI_Wtime();
  }
  
  for (int k = 0; k < comm_size; k++) {
    
    // Each processor in column k broadcasts array A to its row
    if (myCol == k) {
      MPI_Bcast(A, block_size*block_size, MPI_DOUBLE, k, comm_rows);
    }
    else {
      MPI_Bcast(bufferA, block_size*block_size, MPI_DOUBLE, k, comm_rows);
    }
    
    // Each processor in row k broadcasts array B to its column
    if (myRow == k) {
      MPI_Bcast(B, block_size*block_size, MPI_DOUBLE, k, comm_cols);
    }
    else {
      MPI_Bcast(bufferB, block_size*block_size, MPI_DOUBLE, k, comm_cols);
    }
    
    if ( (myRow == k) && (myCol == k) ) {
      MatrixMultiply(C, A, B, block_size);
    }
    else if (myRow == k) {
      MatrixMultiply(C, bufferA, B, block_size);
    }
    else if (myCol == k) {
      MatrixMultiply(C, A, bufferB, block_size);
    }
    else {
      MatrixMultiply(C, bufferA, bufferB, block_size);
    }
    
  }
  
  // Stop the timer
  double end_time;
  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0) {
    end_time = MPI_Wtime();
  }
  
  // Sum the contents of the C array which will be sent to processor 0 when we call MPI_Reduce.
  double total = 0.0;
  for (int i = 0; i < block_size; i++) {
    for (int j = 0; j < block_size; j++) {
      total+=C[i*block_size+j];
    }
  }

  MPI_Reduce(&total, &total, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  
  if (rank == 0) {
    // Print out string message and wall-clock time if calculation was successful
    double checksum = (double)N*N*N*(N-1)*(N-1)/2;
    if (checksum == total) {
      fprintf(stdout,"Compute time: %.3lf\n", end_time - start_time);
    }
    else {
      fprintf(stdout, "ERROR: checksum does not match.\n");
    }
  }
  
  // Clean-up
  free(A);
  free(B);
  free(C);
  free(bufferA);
  free(bufferB);
  
  MPI_Finalize();
  
  return 0;
}

///////////////////////////////////////////////////////
// Returns 1 if N is a perfect square, 0 otherwise. //
/////////////////////////////////////////////////////
int is_perfect_square(int N) {
  int temp = sqrt(N);
  return (temp*temp == N) ? 1 : 0;
}

/////////////////////////////////////////////////////////////////
// Returns zero if sqrt(num_procs) divides N, 1 otherwise     //
///////////////////////////////////////////////////////////////
int p_divides_N(int num_procs, int N) {
  return (N % (int)sqrt(num_procs) == 0) ? 1 : 0;
}

//////////////////////////////////////////////////////////////////
//  Returns (i, j)  coordinate values based on processor rank  //
////////////////////////////////////////////////////////////////
void my_2D_rank(int rank, int *myRow, int *myCol) {
  int block = (N/block_size);
  *myRow = (int)(rank/block);
  *myCol = rank % block;
  return;
}

//////////////////////////////////////////////////////////////////////////////
// Converts local (i,j) indices to global indices based on processor rank  //
////////////////////////////////////////////////////////////////////////////
void local2global(int rank, int local_i, int local_j, int *globalRow, int *globalCol) {

  int block = (N/block_size);
  
  // First find location of (0,0) then add the offset
  *globalRow = (int)(rank/block);
  *globalRow*=block_size;
  *globalRow+=(local_i % block_size);
  
  *globalCol = rank % block;
  *globalCol*=block_size;
  *globalCol+=(local_j % block_size);
  
  return;
}


//////////////////////////////////////////////
//  Matrix Multiplication: [A] x [B] = [C] //
////////////////////////////////////////////
void MatrixMultiply(double *C, double *A, double *B, int block_size) {
  for (int i = 0; i < block_size; i++) {
    for (int k = 0; k < block_size; k++) {
      for (int j = 0; j < block_size; j++) {
        C[i*block_size+j] += A[i*block_size+k]*B[k*block_size+j];
      }
    }
  }
}


