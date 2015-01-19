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
int q;            // sqrt(num_procs)

///////////////////////////////////////////////////////
// Functions                                         //
///////////////////////////////////////////////////////
static void program_abort(char *exec_name, char *message);
static void print_usage();
static int is_perfect_square(int);
static int p_divides_N(int, int);
void local2global(int, int, int, int *, int *);
void MatrixMultiply(double *, double *, double *, int);
void PrintSingleArray(int, double *, int);

void my_2D_rank(int, int *, int *);

void get_Horizontal_Preskew_Ranks(int rank, int offset, int *send_to, int *receive_from);
void get_Vertical_Preskew_Ranks(int rank, int offset, int *send_to, int *receive_from);

void get_Horizontal_Postskew_Ranks(int rank, int offset, int *send_to, int *receive_from);
void get_Vertical_Postskew_Ranks(int rank, int offset, int *send_to, int *receive_from);

int shift_left(int rank, int offset);
int shift_right(int rank, int offset);
int shift_up(int rank, int offset);
int shift_down(int rank, int offset);

void swapPtrs(double **A, double **B);

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
  int i, j;
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
  
  q = sqrt(num_procs);
  block_size = N/q;
  
  int myRow, myCol;
  my_2D_rank(rank, &myRow, &myCol);
  
  int globalRow, globalCol;
  local2global(rank, 0, 0, &globalRow, &globalCol);
  
  int NUM_BYTES = block_size*block_size;
  double *A = (double *) malloc(sizeof(double) * NUM_BYTES);
  double *B = (double *) malloc(sizeof(double) * NUM_BYTES);
  double *C = (double *) malloc(sizeof(double) * NUM_BYTES);

  double *bufferA = (double *) malloc(sizeof(double) * NUM_BYTES);
  double *bufferB = (double *) malloc(sizeof(double) * NUM_BYTES);
  
  // Fill arrays with appropriate values
  for (i = 0; i < block_size ; i++) {
    for (j = 0; j < block_size; j++) {
      
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
  
  // Start the timer
  double start_time;
  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0) {
    start_time = MPI_Wtime();
  }

  // Horizontal Preskew of A and Vertical Preskew of B
  int h_send_to, h_receive_from;
  int v_send_to, v_receive_from;
  
  MPI_Status h_status, v_status;
  MPI_Request h_send_request_skew, v_send_request_skew;
  
  get_Horizontal_Preskew_Ranks(rank, myRow, &h_send_to, &h_receive_from);
  get_Vertical_Preskew_Ranks(rank, myCol, &v_send_to, &v_receive_from);
  
  // Non-blocking send
  MPI_Isend(A, NUM_BYTES, MPI_DOUBLE, h_send_to, 0, MPI_COMM_WORLD, &h_send_request_skew);
  MPI_Isend(B, NUM_BYTES, MPI_DOUBLE, v_send_to, 1, MPI_COMM_WORLD, &v_send_request_skew);
  
  // Blocking receive
  MPI_Recv(bufferA, NUM_BYTES, MPI_DOUBLE, h_receive_from, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  MPI_Recv(bufferB, NUM_BYTES, MPI_DOUBLE, v_receive_from, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  
  MPI_Wait(&h_send_request_skew, &h_status);
  MPI_Wait(&v_send_request_skew, &v_status);
  
  // Compute data is in the buffer because of the preskew process
  double *ptr_compute_A = bufferA;
  double *ptr_compute_B = bufferB;
  double *ptr_receive_A = A;
  double *ptr_receive_B = B;
  
  // Calculate ranks I'm sending to and receiving from
  h_send_to = shift_left(rank, 1);
  h_receive_from = shift_right(rank, 1);
  v_send_to = shift_up(rank, 1);
  v_receive_from = shift_down(rank, 1);
  
  for (i = 0; i < q; i++) {
    
    //MPI_Status h_status, v_status;
    MPI_Request h_send_request, h_recv_request, v_send_request, v_recv_request;
    
    // Non-blocking Receive
    MPI_Irecv(ptr_receive_A, NUM_BYTES, MPI_DOUBLE, h_receive_from, 0, MPI_COMM_WORLD, &h_recv_request);
    MPI_Irecv(ptr_receive_B, NUM_BYTES, MPI_DOUBLE, v_receive_from, 1, MPI_COMM_WORLD, &v_recv_request);
    
    // Non-blocking Send
    MPI_Isend(ptr_compute_A, NUM_BYTES, MPI_DOUBLE, h_send_to, 0, MPI_COMM_WORLD, &h_send_request);
    MPI_Isend(ptr_compute_B, NUM_BYTES, MPI_DOUBLE, v_send_to, 1, MPI_COMM_WORLD, &v_send_request);
    
    // Compute A x B = C
    MatrixMultiply(C, ptr_compute_A, ptr_compute_B, block_size);
    
    // Wait
    MPI_Wait(&h_recv_request, &h_status);
    MPI_Wait(&v_recv_request, &v_status);
    MPI_Wait(&h_send_request, &h_status);
    MPI_Wait(&v_send_request, &v_status);
    
    // Swap pointers
    swapPtrs(&ptr_compute_A, &ptr_receive_A);
    swapPtrs(&ptr_compute_B, &ptr_receive_B);
    
  }

  // Postskew
  get_Horizontal_Postskew_Ranks(rank, myRow, &h_send_to, &h_receive_from);
  get_Vertical_Postskew_Ranks(rank, myCol, &v_send_to, &v_receive_from);
  
  MPI_Isend(ptr_compute_A, NUM_BYTES, MPI_DOUBLE, h_send_to, 0, MPI_COMM_WORLD, &h_send_request_skew);
  MPI_Isend(ptr_compute_B, NUM_BYTES, MPI_DOUBLE, v_send_to, 1, MPI_COMM_WORLD, &v_send_request_skew);
  
  MPI_Recv(ptr_receive_A, NUM_BYTES, MPI_DOUBLE, h_receive_from, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  MPI_Recv(ptr_receive_B, NUM_BYTES, MPI_DOUBLE, v_receive_from, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  
  MPI_Wait(&h_send_request_skew, &h_status);
  MPI_Wait(&v_send_request_skew, &v_status);
  
  // If num_procs is odd, then switch the array pointer to point to the buffer
  if (num_procs % 2 == 1) {
    swapPtrs(&A, &bufferA);
    swapPtrs(&B, &bufferB);
  }
  
  MPI_Barrier(MPI_COMM_WORLD);
  double end_time;
  if (rank == 0) {
    end_time = MPI_Wtime();
  }
  
  // Sum the contents of the C array which will be sent to processor 0 when we call MPI_Reduce.
  double total = 0.0;
  for (i = 0; i < block_size; i++) {
    for (j = 0; j < block_size; j++) {
      total+=C[i*block_size+j];
    }
  }

  MPI_Reduce(&total, &total, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  
  if (rank == 0) {
    // Print out string message and wall-clock time if calculation was successful
    double checksum = (double)N*N*N*(N-1)*(N-1)/2;
    if (total == checksum) {
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


//////////////////////////////////////////////////////////////////////////

void swapPtrs(double **A, double **B) {
  double *temp = *A;
  *A = *B;
  *B = temp;
}

void get_Horizontal_Preskew_Ranks(int rank, int offset, int *send_to, int *receive_from) {
  *send_to = shift_left(rank, offset);
  *receive_from = shift_right(rank, offset);
}

void get_Vertical_Preskew_Ranks(int rank, int offset, int *send_to, int *receive_from) {
  *send_to = shift_up(rank, offset);
  *receive_from = shift_down(rank, offset);
}

void get_Horizontal_Postskew_Ranks(int rank, int offset, int *send_to, int *receive_from) {
  *send_to = shift_right(rank, offset);
  *receive_from = shift_left(rank, offset);
}

void get_Vertical_Postskew_Ranks(int rank, int offset, int *send_to, int *receive_from) {
  *send_to = shift_down(rank, offset);
  *receive_from = shift_up(rank, offset);
}

int shift_left(int rank, int offset) {
  int newRank = rank - offset;
  if ( (newRank/q) != (rank/q) || newRank < 0  ) {
    newRank+=q;
  }
  return newRank;
}

int shift_right(int rank, int offset) {
  int newRank = rank + offset;
  if ( (newRank/q) != (rank/q) || newRank >= num_procs  ) {
    newRank-=q;
  }
  return newRank;
}

int shift_up(int rank, int offset) {
  int newRank = rank - (offset * q);
  if (newRank < 0) {
    newRank+=num_procs;
  }
  return newRank;
}

int shift_down(int rank, int offset) {
  int newRank = rank + (offset * q);
  if (newRank >= num_procs) {
    newRank-=num_procs;
  }
  return newRank;
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
  int i, j, k;
  for (i = 0; i < block_size; i++) {
    for (k = 0; k < block_size; k++) {
      for (j = 0; j < block_size; j++) {
        C[i*block_size+j] += A[i*block_size+k]*B[k*block_size+j];
      }
    }
  }
}

void PrintSingleArray(int rank, double *A, int block_size) {
  int i, j;
  fprintf(stdout, "\nBlocks of A on rank %d\n", rank);
  for (i = 0; i < block_size; i++) {
    for (j = 0; j < block_size; j++) {
      fprintf(stdout, "%0.1f ", A[i*block_size+j]);
    }
    fprintf(stdout,"\n");
  }
  
}



