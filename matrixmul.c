#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <assert.h>
#include <getopt.h>

#include "densematgen.h"

#define f(I, IX, M) for (int I = IX; I < M; I++)

struct csr {
  int rows;
  int nnz;
  int nnz_row;
  double *nonzeroes;
  int *nnzs_row;
  int *col_ixs;
};

struct coo {
  int row;
  int col;
  double val;
};

typedef struct csr sparse_t;
typedef sparse_t *sparse_type;
typedef struct coo coo;

sparse_type gen_matrix(char *filename) {
  FILE *file = fopen(filename, "r");
  int rows, nnz, nnz_row;

  fscanf(file, "%d", &rows);
  fscanf(file, "%d", &rows);
  fscanf(file, "%d", &nnz);
  fscanf(file, "%d", &nnz_row);

  double *nonzeroes = (double *) malloc(nnz * sizeof(double));
  int *nnzs_row = (int *) malloc((rows + 1) * sizeof(int));
  int *col_ixs = (int *) malloc(nnz * sizeof(int));

  f(i, 0, nnz) {
    fscanf(file, "%lf", nonzeroes + i);
  }
  printf("\n");

  printf("nnzs_row: ");
  f(i, 0, rows + 1) {
    fscanf(file, "%d", nnzs_row + i);
  }
  printf("\n");

  printf("ixs: ");
  f(i, 0, nnz) {
    fscanf(file, "%d", col_ixs + i);
  }
  printf("\n");

  sparse_type a = (sparse_type)malloc(sizeof(sparse_t));
  a->rows = rows;
  a->nnz = nnz;
  a->nnz_row = nnz_row;
  a->nnzs_row = nnzs_row;
  a->nonzeroes = nonzeroes;
  a->col_ixs = col_ixs;
  return a;
}

int main(int argc, char *argv[]) {
  int show_results = 0;
  int use_inner = 0;
  int gen_seed = -1;
  int repl_fact = 1;

  int option = -1;

  double comm_start = 0, comm_end = 0, comp_start = 0, comp_end = 0;
  int num_processes = 1;
  int mpi_rank = 0;
  int exponent = 1;
  double ge_element = 0;
  int count_ge = 0;

  sparse_type sparse = NULL;

  int partition_size, nnzs_rows;

  MPI_Datatype coo_type;
  MPI_Datatype type[3] = {MPI_INT, MPI_INT, MPI_DOUBLE};
  int blocklen[3] = {1, 1, 1};
  MPI_Aint disp[3];
  coo tmp;

  MPI_Init(&argc, &argv);
  disp[0] = (void *)&(tmp.row) - (void *)&tmp;
  disp[1] = (void *)&(tmp.col) - (void *)&tmp;
  disp[2] = (void *)&(tmp.val) - (void *)&tmp;



  MPI_Type_create_struct(3, blocklen, disp, type, &coo_type);
  MPI_Type_commit(&coo_type);
  MPI_Comm_size(MPI_COMM_WORLD, &num_processes);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);


  while ((option = getopt(argc, argv, "vis:f:c:e:g:")) != -1) {
    switch (option) {
    case 'v':
      show_results = 1;
      break;
    case 'i':
      use_inner = 1;
      break;
    case 'f':
      if ((mpi_rank) == 0) {
        sparse = gen_matrix(optarg);
      }
      break;
    case 'c':
      repl_fact = atoi(optarg);
      break;
    case 's':
      gen_seed = atoi(optarg);
      break;
    case 'e':
      exponent = atoi(optarg);
      break;
    case 'g':
      count_ge = 1;
      ge_element = atof(optarg);
      break;
    default:
      fprintf(stderr, "error parsing argument %c exiting\n", option);
      MPI_Finalize();
      return 3;
    }
  }
  if ((gen_seed == -1) || ((mpi_rank == 0) && (sparse == NULL))) {
    fprintf(stderr, "error: missing seed or sparse matrix file; exiting\n");
    MPI_Finalize();
    return 3;
  }


  coo *mycols = NULL;
  int my_size;
  int rows;
  comm_start = MPI_Wtime();
  fprintf(stderr, "tuteej %d\n", mpi_rank);
  if (mpi_rank == 0) {
    rows = sparse->rows;
    partition_size =
      sparse->rows % num_processes == 0
      ? sparse->rows / num_processes
      : (sparse->rows / num_processes + 1);
    coo *coos = (coo *) malloc(sizeof(coo) * sparse->nnz);
    int j = 0, rows = sparse->rows;
    printf("%d %d, 00\n", rows, partition_size);
    for (int i = 0; i < rows; i++) {
      printf("%d uu ", i);
      printf("%d\n", sparse->nnzs_row[i]);
      while (j < sparse->nnzs_row[i+1]) {
        coo tmp = {i, sparse->col_ixs[j], sparse->nonzeroes[j]};
        coos[j] = tmp;
        j++;
      }
    }
    printf("tutej 01\n");
    int *part_sizes = (int *) malloc(sizeof(int) * num_processes);
    MPI_Request requests[2*num_processes];
    coo **part_coos = (coo **) malloc(sizeof(coo *) * num_processes);
    printf("tutej 0\n");
    f(i, 0, num_processes) {
      int tmp = 0;
      part_coos[i] = (coo *) malloc(sizeof(coo) * sparse->nnz_row);
      f(j, 0, sparse->nnz) {
        if (coos[j].col >= i * partition_size && coos[j].col < (i + 1) * partition_size) {
          part_coos[i][tmp] = coos[j];
          tmp++;
        }
      }
      part_sizes[i] = tmp;
    }
    printf("tutej 0\n");
    mycols = part_coos[0];
    f(i, 1, num_processes) {
      MPI_Isend(part_sizes + i, 1, MPI_INT, i, 0, MPI_COMM_WORLD, &requests[2*i]);
      MPI_Isend(part_coos[i], part_sizes[i], coo_type, i, 1, MPI_COMM_WORLD, &requests[2*i + 1]);
    }
    my_size = part_sizes[0];
    nnzs_rows = sparse->nnz_row;
    printf("tutej 0 kuniec\n");
  } else {
    printf("tutej 1\n");
    MPI_Status *status = NULL;
    printf("tutej 1\n");
    MPI_Recv(&my_size, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, status);
    mycols = (coo *) malloc(sizeof(coo) * my_size);
    MPI_Recv(mycols, my_size, coo_type, 0, 1, MPI_COMM_WORLD, status);
  }
  MPI_Barrier(MPI_COMM_WORLD);

  if (mpi_rank == 0) printf("przed brokasty\n");

  MPI_Bcast(&partition_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&nnzs_rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
  int res_size = partition_size * repl_fact * nnzs_rows * 3;
  coo *res_coos = (coo *) malloc(sizeof(coo) * res_size);

  if (mpi_rank == 0) printf("po brokasty\n");
  MPI_Comm repl_comm;
  MPI_Comm shift_comm;
  MPI_Comm_split(MPI_COMM_WORLD, mpi_rank / repl_fact, mpi_rank, &repl_comm);
  MPI_Comm_split(MPI_COMM_WORLD, mpi_rank % repl_fact, mpi_rank, &shift_comm);

  MPI_Barrier(MPI_COMM_WORLD);
  if (mpi_rank == 0) printf("po splity 2\n");
  MPI_Allgather(mycols, my_size, coo_type, res_coos, res_size, coo_type, repl_comm);

  if (mpi_rank == 0) printf("po brokasty 2\n");

  double **B = (double **)malloc(sizeof(double *) * rows);
  f(i,0,rows) {
    B[i] = (double *)malloc(sizeof(double) * partition_size);
    f(j,0,partition_size) {
      B[i][j] = generate_double(gen_seed,i,j);
    }
  }


  // FIXME: scatter sparse matrix; cache sparse matrix; cache dense matrix
  MPI_Barrier(MPI_COMM_WORLD);
  comm_end = MPI_Wtime();

  comp_start = MPI_Wtime();
  // FIXME: compute C = A ( A ... (AB ) )
  MPI_Barrier(MPI_COMM_WORLD);
  comp_end = MPI_Wtime();

  if (show_results) {
    // FIXME: replace the following line: print the whole result matrix
    printf("1 1\n42\n");
  }
  if (count_ge) {
    // FIXME: replace the following line: count ge elements
    printf("54\n");
  }

  MPI_Finalize();
  return 0;
}
