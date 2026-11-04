/* stream_cpu.c - STREAM-style CPU memory-bandwidth benchmark (Copy, Scale, Add, Triad).
 *
 * Follows McCalpin's STREAM rules: FP64 arrays much larger than the last-level
 * cache, parallel (first-touch) initialisation, 11 timed iterations of which the
 * first is a warm-up and is discarded; bytes counted as in STREAM (Copy/Scale
 * 2 x 8 B per element, Add/Triad 3 x 8 B).
 *
 * Usage: ./bin/stream_cpu [elements] [out.csv]   (thread count from OMP_NUM_THREADS)
 * Build: gcc -O3 -march=native -fopenmp src/stream_cpu.c -o bin/stream_cpu -lm
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <time.h>

#define NTIMES 11

static double now_s(void) {
#ifdef _OPENMP
    return omp_get_wtime();
#else
    return (double)clock() / CLOCKS_PER_SEC;
#endif
}

int main(int argc, char** argv) {
    const long n = argc > 1 ? atol(argv[1]) : 1L << 25; /* 256 MB per array */
    const char* out = argc > 2 ? argv[2] : "results/raw/stream_cpu.csv";
    int threads = 1;
#ifdef _OPENMP
    threads = omp_get_max_threads();
#endif

    double* a = (double*)malloc(n * sizeof(double));
    double* b = (double*)malloc(n * sizeof(double));
    double* c = (double*)malloc(n * sizeof(double));
    if (!a || !b || !c) { fprintf(stderr, "allocation failed\n"); return 1; }

    long j;
#pragma omp parallel for schedule(static)
    for (j = 0; j < n; ++j) { a[j] = 1.0; b[j] = 2.0; c[j] = 0.0; }

    const char* names[4] = {"Copy", "Scale", "Add", "Triad"};
    const double bytes[4] = {2.0 * 8 * n, 2.0 * 8 * n, 3.0 * 8 * n, 3.0 * 8 * n};
    double t[4][NTIMES];
    const double s = 3.0;

    for (int k = 0; k < NTIMES; ++k) {
        double t0 = now_s();
#pragma omp parallel for schedule(static)
        for (j = 0; j < n; ++j) c[j] = a[j];
        t[0][k] = now_s() - t0;

        t0 = now_s();
#pragma omp parallel for schedule(static)
        for (j = 0; j < n; ++j) b[j] = s * c[j];
        t[1][k] = now_s() - t0;

        t0 = now_s();
#pragma omp parallel for schedule(static)
        for (j = 0; j < n; ++j) c[j] = a[j] + b[j];
        t[2][k] = now_s() - t0;

        t0 = now_s();
#pragma omp parallel for schedule(static)
        for (j = 0; j < n; ++j) a[j] = b[j] + s * c[j];
        t[3][k] = now_s() - t0;
    }

    /* Validate like STREAM: replay the scalar recurrence and compare. */
    double ea = 1.0, eb = 2.0, ec = 0.0;
    for (int k = 0; k < NTIMES; ++k) { ec = ea; eb = s * ec; ec = ea + eb; ea = eb + s * ec; }
    const int valid = fabs(a[n / 2] - ea) / ea < 1e-13 && fabs(b[n / 2] - eb) / eb < 1e-13 &&
                      fabs(c[n / 2] - ec) / ec < 1e-13;

    FILE* chk = fopen(out, "r");
    int has_header = chk && fgetc(chk) != EOF;
    if (chk) fclose(chk);
    FILE* f = fopen(out, "a");
    if (!f) { perror(out); return 1; }
    if (!has_header) fprintf(f, "kernel,threads,array_mb,best_gbs,mean_gbs,std_gbs,valid\n");

    printf("STREAM-style CPU, %d thread(s), %.1f MB per array, %s\n", threads, n * 8.0 / 1e6,
           valid ? "results validated" : "VALIDATION FAILED");
    printf("%-6s %12s %12s %12s\n", "kernel", "best GB/s", "mean GB/s", "std GB/s");
    for (int i = 0; i < 4; ++i) {
        double best = 0, sum = 0, sq = 0;
        for (int k = 1; k < NTIMES; ++k) {
            double g = bytes[i] / t[i][k] / 1e9;
            if (g > best) best = g;
            sum += g;
        }
        double mean = sum / (NTIMES - 1);
        for (int k = 1; k < NTIMES; ++k) {
            double g = bytes[i] / t[i][k] / 1e9;
            sq += (g - mean) * (g - mean);
        }
        double sd = sqrt(sq / (NTIMES - 2));
        printf("%-6s %12.2f %12.2f %12.2f\n", names[i], best, mean, sd);
        fprintf(f, "%s,%d,%.1f,%.3f,%.3f,%.3f,%d\n", names[i], threads, n * 8.0 / 1e6, best, mean, sd, valid);
    }
    fclose(f);
    free(a); free(b); free(c);
    return valid ? 0 : 1;
}
