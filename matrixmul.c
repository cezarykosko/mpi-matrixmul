#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <assert.h>
#include <getopt.h>

#include "densematgen.h"

#define f(I, IX, M) for (int I = IX; I < M; I++)

#define LEFT (shift_procs + shift_rank - 1) % shift_procs
#define RIGHT (shift_procs + shift_rank + 1) % shift_procs

#define MIN(A,B) A > B ? B : A

#define INIT_SIZE_MSG 0
#define INIT_ARR_MSG 1
#define SHIFT_MSG 2823
#define SHIFT_SIZE_MSG 2824

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

int cmpcols(const void *a, const void *b) {
    int a_col = ((coo *) a)->col;
    int b_col = ((coo *) b)->col;

    return a_col - b_col;
}

void free_arr(double **c_arr, int rows) {
    f(i, 0, rows) {
        free(c_arr[rows]);
    }
    free(c_arr);
}

double **c_arr(int rows, int partition_size) {
    double **res = (double **) calloc(rows, sizeof(double *));
    f(i, 0, rows) {
        res[i] = (double *) calloc(partition_size, sizeof(double));
        f(j, 0, partition_size) {
            res[i][j] = 0;
        }
    }
    return res;
}

double **b_arr(int my_width, int rows, int gen_seed, int partition_size, int rank) {
    double **res = (double **) calloc(my_width, sizeof(double *));
    f(i, 0, my_width) {
        res[i] = (double *) calloc(rows, sizeof(double));
        f(j, 0, rows) {
            res[i][j] = generate_double(gen_seed, j, i + rank * partition_size);
        }
    }
    return res;
}

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

    f(i, 0, rows + 1) {
        fscanf(file, "%d", nnzs_row + i);
    }

    f(i, 0, nnz) {
        fscanf(file, "%d", col_ixs + i);
    }

    sparse_type a = (sparse_type) malloc(sizeof(sparse_t));
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

    int shift_rank = 0;
    int shift_procs = 0;

    int repl_rank = 0;
    int repl_procs = 0;

    sparse_type sparse = NULL;

    int partition_size, nnzs_rows;

    MPI_Datatype coo_type;
    MPI_Datatype type[3] = {MPI_INT, MPI_INT, MPI_DOUBLE};
    int blocklen[3] = {1, 1, 1};
    MPI_Aint disp[3];
    coo tmp;

    MPI_Init(&argc, &argv);
    disp[0] = (void *) &(tmp.row) - (void *) &tmp;
    disp[1] = (void *) &(tmp.col) - (void *) &tmp;
    disp[2] = (void *) &(tmp.val) - (void *) &tmp;


    MPI_Type_create_struct(3, blocklen, disp, type, &coo_type);
    MPI_Type_commit(&coo_type);
    MPI_Comm_size(MPI_COMM_WORLD, &num_processes);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm repl_comm;
    MPI_Comm shift_comm;


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

    MPI_Comm_split(MPI_COMM_WORLD, mpi_rank / repl_fact, mpi_rank, &repl_comm);
    MPI_Comm_split(MPI_COMM_WORLD, mpi_rank % repl_fact, mpi_rank, &shift_comm);
    MPI_Comm_size(shift_comm, &shift_procs);
    MPI_Comm_rank(shift_comm, &shift_rank);

    MPI_Comm_size(repl_comm, &repl_procs);
    MPI_Comm_rank(repl_comm, &repl_rank);

    coo *mycols = NULL;
    int my_size;
    int rows;
    coo *coos;
    int *part_sizes;
    int *displs;

    comm_start = MPI_Wtime();
    if (mpi_rank == 0) {
        rows = sparse->rows;
        partition_size =
                sparse->rows % num_processes == 0
                ? sparse->rows / num_processes
                : (sparse->rows / num_processes + 1);
        coos = (coo *) malloc(sizeof(coo) * sparse->nnz);
        int j = 0, rows = sparse->rows;
        for (int i = 0; i < rows; i++) {
            while (j < sparse->nnzs_row[i + 1]) {
                coo tmp = {i, sparse->col_ixs[j], sparse->nonzeroes[j]};
                coos[j] = tmp;
                j++;
            }
        }
        if (!use_inner) {
            qsort(coos, sparse->nnz, sizeof(coo), cmpcols);
        }
        part_sizes = (int *) malloc(sizeof(int) * num_processes);
        displs = (int *) malloc(sizeof(int) * num_processes);
        f(i, 0, num_processes) {
            int tmp = 0;
            f(j, 0, sparse->nnz) {
                if (coos[j].col >= i * partition_size && coos[j].col < (i + 1) * partition_size) {
                    tmp++;
                }
            }
            part_sizes[i] = tmp;
        }
        displs[0] = 0;
        f(i, 1, num_processes) {
            displs[i] = displs[i - 1] + part_sizes[i - 1];
        }
        nnzs_rows = sparse->nnz_row;
    }

    MPI_Scatter(part_sizes, 1, MPI_INT, &my_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    mycols = (coo *) malloc(sizeof(coo) * my_size);
    MPI_Scatterv(coos, part_sizes, displs, coo_type, mycols, my_size, coo_type, 0, MPI_COMM_WORLD);

    MPI_Bcast(&partition_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nnzs_rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    int res_size;
    part_sizes = (int *) malloc(sizeof(int) * repl_procs);
    displs = (int *) malloc(sizeof(int) * repl_procs);
    MPI_Allgather(&my_size, 1, MPI_INT, part_sizes, 1, MPI_INT, repl_comm);
    displs[0] = 0;
    f(i, 1, repl_procs) {
        displs[i] = displs[i-1] + part_sizes[i-1];
    }
    res_size = displs[repl_procs - 1] + part_sizes[repl_procs - 1];

    int my_width = MIN((mpi_rank + 1) * partition_size, rows);
    my_width -= mpi_rank * partition_size;
    coo *res_coos = (coo *) malloc(sizeof(coo) * res_size);

    MPI_Allgatherv(mycols, my_size, coo_type, res_coos, part_sizes, displs, coo_type, repl_comm);

    double **B = b_arr(my_width, rows, gen_seed, partition_size, mpi_rank);
    double **C = c_arr(my_width, rows);

    MPI_Barrier(MPI_COMM_WORLD);
    comm_end = MPI_Wtime();


    comp_start = MPI_Wtime();
    int curr_size = res_size;
    coo *curr_coos = res_coos;
    f(i, 0, exponent) {
        f(iter, 0, shift_procs) {
            f(c_ix, 0, curr_size) {
                coo tmp = curr_coos[c_ix];
                f(r_ix, 0, my_width) {
                    C[r_ix][tmp.row] += B[r_ix][tmp.col] * tmp.val;
                }
            }

            MPI_Request requests[2];
            MPI_Isend(&curr_size, 1, MPI_INT, LEFT, SHIFT_SIZE_MSG,
                      shift_comm, requests + 0);
            MPI_Isend(curr_coos, curr_size, coo_type, LEFT, SHIFT_MSG,
                      shift_comm, requests + 1);

            coo *temp_coos;
            int temp_size;
            MPI_Status statuses[2];
            MPI_Recv(&temp_size, 1, MPI_INT, RIGHT, SHIFT_SIZE_MSG, shift_comm, statuses + 0);
            temp_coos = (coo *) malloc(sizeof(coo) * temp_size);
            MPI_Recv(temp_coos, temp_size, coo_type, RIGHT, SHIFT_MSG, shift_comm, statuses + 1);
            MPI_Barrier(shift_comm);

            curr_size = temp_size;
            free(curr_coos);
            curr_coos = temp_coos;
        }

        f(r_ix, 0, my_width) {
            f(c_ix, 0, rows) {
                B[r_ix][c_ix] = C[r_ix][c_ix];
                C[r_ix][c_ix] = 0;
            }
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    comp_end = MPI_Wtime();

    if (show_results) {
        int *widths = (int *) malloc(sizeof(int) * num_processes);
        displs = (int *) malloc(sizeof(int) * num_processes);
        double *tmp_line = (double *) malloc(sizeof(double) * my_width);
        double *line = (double *)malloc(sizeof(double) * rows);
        MPI_Gather(&my_width, 1, MPI_INT, widths, 1, MPI_INT, 0, MPI_COMM_WORLD);
        displs[0] = 0;
        f(i, 1, num_processes) {
            displs[i] = displs[i-1] + widths[i-1];
        }
        if (mpi_rank == 0) {
            printf("%d %d\n", rows, rows);
        }
        f(i, 0, rows) {
            f(k, 0, my_width) {
                tmp_line[k] = B[k][i];
            }
            MPI_Gatherv(tmp_line, my_width, MPI_DOUBLE, line, widths, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            if (mpi_rank == 0) {
                f(j, 0, rows) {
                    printf("%f\t", line[j]);
                }
                printf("\n");
            }
        }
        free(widths);
        free(displs);
        free(tmp_line);
        free(line);
    }
    if (count_ge) {
        int all_counts = 0;
        int proc_count = 0;
        f(i, 0, rows) {
            f(j, 0, my_width) {
                if (B[j][i] > ge_element) {
                    proc_count++;
                }
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);

        MPI_Reduce(&proc_count, &all_counts, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

        if (mpi_rank == 0) {
            printf("%d\n", all_counts);
        }
    }

    MPI_Finalize();
    return 0;
}
