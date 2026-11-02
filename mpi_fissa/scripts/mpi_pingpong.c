/* mpi_pingpong.c -- latenza e banda reali fra i due PC, per le taglie che usa
 * il solver (16 B ... 1 MB). Serve a separare due cose che `timing.dat` di
 * mpi_fissa confonde: il costo della RETE e il costo del pack (cudaMemcpy).
 *
 * Compilazione (sul master; basta un binario, e' un test MPI puro):
 *   mpicxx -O3 -o mpi_pingpong mpi_pingpong.c
 *
 * Uso (dal progetto, con l'hostfile dei 2 PC):
 *   mpirun --hostfile scripts/hostfile_2pc.txt -np 2 ./mpi_pingpong 200
 *
 * Output (solo rank 0):
 *   bytes   RTT_us   MB/s(uno verso)
 * Il RTT e' il tempo di andata+ritorno di un messaggio; la banda "uno verso" e'
 * n_byte / (RTT/2). Confronto con i numeri del solver a L8:
 *   livello 0 del multigrid = 256 KB, livello 1 = 64 KB, livello 4 = 1 KB.
 * Se RTT(256 KB) - RTT(16 B) e' piccolo (< 1 ms) la rete e' veloce e il costo
 * per scambio del solver e' nel pack, non nel Wi-Fi (vedi OTIMIZZAZIONI.md §3).
 *
 * Nota: il pattern e' ping-pong stretto (un messaggio alla volta), come il
 * codice reale (MPI_Sendrecv bloccante), non un allreduce.
 */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size != 2) {
        if (rank == 0) fprintf(stderr, "uso: mpirun -np 2 %s [reps]\n", argv[0]);
        MPI_Finalize();
        return 2;
    }
    int reps0 = (argc > 1) ? atoi(argv[1]) : 200;
    if (reps0 < 5) reps0 = 5;

    static const int sizes[] = {16, 256, 1024, 4096, 16384, 65536, 262144, 1048576};
    const int ns = (int)(sizeof(sizes)/sizeof(sizes[0]));
    const int maxn = sizes[ns-1];
    char* buf = (char*)malloc((size_t)maxn);
    if (!buf) { fprintf(stderr, "malloc\n"); MPI_Finalize(); return 2; }
    for (int i = 0; i < maxn; i++) buf[i] = (char)i;

    if (rank == 0)
        printf("# mpi_pingpong: %d byte max, %d reps base\n", maxn, reps0);
    if (rank == 0)
        printf("# %9s %10s %12s %10s\n", "bytes", "RTT_us", "MB/s(1verso)", "reps");

    for (int s = 0; s < ns; s++) {
        int n = sizes[s];
        int reps = reps0;
        if (n > 262144)      reps = 10;
        else if (n > 65536)  reps = 30;
        else if (n > 4096)   reps = 100;
        MPI_Status st;

        /* warmup */
        for (int i = 0; i < 3; i++) {
            if (rank == 0) {
                MPI_Send(buf, n, MPI_CHAR, 1, 0, MPI_COMM_WORLD);
                MPI_Recv(buf, n, MPI_CHAR, 1, 1, MPI_COMM_WORLD, &st);
            } else {
                MPI_Recv(buf, n, MPI_CHAR, 0, 0, MPI_COMM_WORLD, &st);
                MPI_Send(buf, n, MPI_CHAR, 0, 1, MPI_COMM_WORLD);
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = MPI_Wtime();
        for (int i = 0; i < reps; i++) {
            if (rank == 0) {
                MPI_Send(buf, n, MPI_CHAR, 1, 0, MPI_COMM_WORLD);
                MPI_Recv(buf, n, MPI_CHAR, 1, 1, MPI_COMM_WORLD, &st);
            } else {
                MPI_Recv(buf, n, MPI_CHAR, 0, 0, MPI_COMM_WORLD, &st);
                MPI_Send(buf, n, MPI_CHAR, 0, 1, MPI_COMM_WORLD);
            }
        }
        double rtt = (MPI_Wtime() - t0)/reps;
        if (rank == 0) {
            double oneway = (rtt > 0) ? (double)n/(rtt*0.5) : 0.0;   /* B/s */
            printf("  %9d %10.1f %12.2f %10d\n", n, rtt*1e6, oneway/1e6, reps);
            fflush(stdout);
        }
    }
    free(buf);
    MPI_Finalize();
    return 0;
}
