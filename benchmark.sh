#!/bin/bash
#
# PolyOpt Benchmark Suite
# Tests sequential vs OpenMP parallel performance
#
# Usage: ./benchmark.sh [--size N] [--iterations I] [--quick]
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# Default settings
SIZE=2000
ITERATIONS=3
QUICK=false

# Parse arguments
for arg in "$@"; do
    case $arg in
        --size=*) SIZE="${arg#*=}" ;;
        --iterations=*) ITERATIONS="${arg#*=}" ;;
        --quick) QUICK=true; SIZE=1000; ITERATIONS=2 ;;
        --help)
            echo "PolyOpt Benchmark Suite"
            echo ""
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --size=N        Problem size (default: 2000)"
            echo "  --iterations=I  Number of runs per test (default: 3)"
            echo "  --quick         Quick mode (size=1000, iterations=2)"
            echo "  --help          Show this help"
            exit 0
            ;;
    esac
done

# Setup
BENCH_DIR="/tmp/polyopt_bench_$$"
mkdir -p "$BENCH_DIR"
trap "rm -rf $BENCH_DIR" EXIT

# Header
echo -e "${BOLD}${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo -e "${BOLD}${BLUE}                 PolyOpt Benchmark Suite                         ${NC}"
echo -e "${BOLD}${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "Configuration:"
echo -e "  Problem size:  ${BOLD}$SIZE${NC}"
echo -e "  Iterations:    ${BOLD}$ITERATIONS${NC}"
echo ""

# Detect CPUs
NCPUS=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
echo -e "Detected ${BOLD}$NCPUS${NC} CPU cores"
echo ""

# Results array
declare -A RESULTS

#######################################
# Run benchmark and get median time
#######################################
run_bench() {
    local exe=$1
    local times=()
    
    for ((i=1; i<=ITERATIONS; i++)); do
        local output=$("$exe" 2>&1)
        local t=$(echo "$output" | grep -oE 'Time: [0-9]+\.[0-9]+' | grep -oE '[0-9]+\.[0-9]+')
        [ -n "$t" ] && times+=("$t")
    done
    
    [ ${#times[@]} -eq 0 ] && echo "FAILED" && return
    
    # Return median
    printf '%s\n' "${times[@]}" | sort -n | sed -n "$((ITERATIONS/2 + 1))p"
}

#######################################
# Benchmark 1: Vector Addition
#######################################
benchmark_vector_add() {
    local N=$SIZE
    
    echo -e "${BOLD}${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BOLD}Benchmark: Vector Addition (N=$N)${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    
    # Sequential version
    cat > "$BENCH_DIR/vec_seq.c" << EOF
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

void vector_add(int N, double* A, double* B, double* C) {
    for (int i = 0; i < N; i++) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    int N = $N;
    double *A = malloc(N * sizeof(double));
    double *B = malloc(N * sizeof(double));
    double *C = malloc(N * sizeof(double));
    
    for (int i = 0; i < N; i++) { A[i] = i * 0.1; B[i] = i * 0.2; }
    
    // Warmup
    vector_add(N, A, B, C);
    
    double start = get_time();
    for (int r = 0; r < 100; r++) {
        vector_add(N, A, B, C);
    }
    double end = get_time();
    
    printf("Time: %.6f seconds\n", (end - start) / 100.0);
    printf("Check: C[0]=%.2f C[N-1]=%.2f\n", C[0], C[N-1]);
    
    free(A); free(B); free(C);
    return 0;
}
EOF

    # OpenMP version
    cat > "$BENCH_DIR/vec_omp.c" << EOF
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <omp.h>

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

void vector_add(int N, double* A, double* B, double* C) {
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    int N = $N;
    double *A = malloc(N * sizeof(double));
    double *B = malloc(N * sizeof(double));
    double *C = malloc(N * sizeof(double));
    
    for (int i = 0; i < N; i++) { A[i] = i * 0.1; B[i] = i * 0.2; }
    
    // Warmup
    vector_add(N, A, B, C);
    
    double start = get_time();
    for (int r = 0; r < 100; r++) {
        vector_add(N, A, B, C);
    }
    double end = get_time();
    
    printf("Time: %.6f seconds\n", (end - start) / 100.0);
    printf("Check: C[0]=%.2f C[N-1]=%.2f\n", C[0], C[N-1]);
    
    free(A); free(B); free(C);
    return 0;
}
EOF

    gcc -O3 -march=native -o "$BENCH_DIR/vec_seq" "$BENCH_DIR/vec_seq.c" -lm
    gcc -O3 -march=native -fopenmp -o "$BENCH_DIR/vec_omp" "$BENCH_DIR/vec_omp.c" -lm
    
    echo -e "${YELLOW}Running sequential...${NC}"
    local seq_time=$(run_bench "$BENCH_DIR/vec_seq")
    
    echo -e "${YELLOW}Running OpenMP...${NC}"
    local omp_time=$(run_bench "$BENCH_DIR/vec_omp")
    
    if [[ "$seq_time" != "FAILED" ]] && [[ "$omp_time" != "FAILED" ]]; then
        local speedup=$(echo "scale=2; $seq_time / $omp_time" | bc 2>/dev/null || echo "N/A")
        echo -e "  Sequential: ${BOLD}${seq_time}s${NC}"
        echo -e "  OpenMP:     ${BOLD}${omp_time}s${NC}"
        echo -e "  Speedup:    ${GREEN}${BOLD}${speedup}x${NC}"
        RESULTS["vector_add"]="$speedup"
    else
        echo -e "  ${RED}Failed${NC}"
        RESULTS["vector_add"]="FAILED"
    fi
    echo ""
}

#######################################
# Benchmark 2: Matrix Multiplication
#######################################
benchmark_matmul() {
    local N=$((SIZE / 5))
    [ $N -lt 100 ] && N=100
    [ $N -gt 600 ] && N=600
    
    echo -e "${BOLD}${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BOLD}Benchmark: Matrix Multiplication (N=$N)${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    
    # Sequential version
    cat > "$BENCH_DIR/mm_seq.c" << EOF
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

void matmul(int N, double** A, double** B, double** C) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i][j] = 0.0;
            for (int k = 0; k < N; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

double** alloc_matrix(int N) {
    double** M = malloc(N * sizeof(double*));
    for (int i = 0; i < N; i++) M[i] = malloc(N * sizeof(double));
    return M;
}

int main() {
    int N = $N;
    double **A = alloc_matrix(N);
    double **B = alloc_matrix(N);
    double **C = alloc_matrix(N);
    
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            A[i][j] = (i + j) % 100 * 0.01;
            B[i][j] = (i * j) % 100 * 0.01;
        }
    
    double start = get_time();
    matmul(N, A, B, C);
    double end = get_time();
    
    printf("Time: %.6f seconds\n", end - start);
    printf("Check: C[0][0]=%.4f\n", C[0][0]);
    return 0;
}
EOF

    # OpenMP version
    cat > "$BENCH_DIR/mm_omp.c" << EOF
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <omp.h>

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

void matmul(int N, double** A, double** B, double** C) {
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i][j] = 0.0;
            for (int k = 0; k < N; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

double** alloc_matrix(int N) {
    double** M = malloc(N * sizeof(double*));
    for (int i = 0; i < N; i++) M[i] = malloc(N * sizeof(double));
    return M;
}

int main() {
    int N = $N;
    double **A = alloc_matrix(N);
    double **B = alloc_matrix(N);
    double **C = alloc_matrix(N);
    
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            A[i][j] = (i + j) % 100 * 0.01;
            B[i][j] = (i * j) % 100 * 0.01;
        }
    
    double start = get_time();
    matmul(N, A, B, C);
    double end = get_time();
    
    printf("Time: %.6f seconds\n", end - start);
    printf("Check: C[0][0]=%.4f\n", C[0][0]);
    return 0;
}
EOF

    gcc -O3 -march=native -o "$BENCH_DIR/mm_seq" "$BENCH_DIR/mm_seq.c" -lm
    gcc -O3 -march=native -fopenmp -o "$BENCH_DIR/mm_omp" "$BENCH_DIR/mm_omp.c" -lm
    
    echo -e "${YELLOW}Running sequential...${NC}"
    local seq_time=$(run_bench "$BENCH_DIR/mm_seq")
    
    echo -e "${YELLOW}Running OpenMP...${NC}"
    local omp_time=$(run_bench "$BENCH_DIR/mm_omp")
    
    if [[ "$seq_time" != "FAILED" ]] && [[ "$omp_time" != "FAILED" ]]; then
        local speedup=$(echo "scale=2; $seq_time / $omp_time" | bc 2>/dev/null || echo "N/A")
        echo -e "  Sequential: ${BOLD}${seq_time}s${NC}"
        echo -e "  OpenMP:     ${BOLD}${omp_time}s${NC}"
        echo -e "  Speedup:    ${GREEN}${BOLD}${speedup}x${NC}"
        RESULTS["matmul"]="$speedup"
    else
        echo -e "  ${RED}Failed${NC}"
        RESULTS["matmul"]="FAILED"
    fi
    echo ""
}

#######################################
# Benchmark 3: Jacobi 2D Stencil
#######################################
benchmark_jacobi() {
    local N=$((SIZE / 2))
    [ $N -lt 200 ] && N=200
    [ $N -gt 2000 ] && N=2000
    
    echo -e "${BOLD}${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BOLD}Benchmark: Jacobi 2D Stencil (N=$N)${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    
    # Sequential
    cat > "$BENCH_DIR/jacobi_seq.c" << EOF
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

void jacobi_2d(int N, double** A, double** B) {
    for (int i = 1; i < N-1; i++) {
        for (int j = 1; j < N-1; j++) {
            B[i][j] = 0.25 * (A[i-1][j] + A[i+1][j] + A[i][j-1] + A[i][j+1]);
        }
    }
}

double** alloc_matrix(int N) {
    double** M = malloc(N * sizeof(double*));
    for (int i = 0; i < N; i++) M[i] = calloc(N, sizeof(double));
    return M;
}

int main() {
    int N = $N;
    double **A = alloc_matrix(N);
    double **B = alloc_matrix(N);
    
    // Initialize with boundary conditions
    for (int i = 0; i < N; i++) {
        A[i][0] = 1.0; A[i][N-1] = 1.0;
        A[0][i] = 1.0; A[N-1][i] = 1.0;
    }
    
    // Warmup
    jacobi_2d(N, A, B);
    
    double start = get_time();
    for (int iter = 0; iter < 100; iter++) {
        jacobi_2d(N, A, B);
        double **tmp = A; A = B; B = tmp;
    }
    double end = get_time();
    
    printf("Time: %.6f seconds\n", (end - start) / 100.0);
    printf("Check: A[N/2][N/2]=%.6f\n", A[N/2][N/2]);
    return 0;
}
EOF

    # OpenMP
    cat > "$BENCH_DIR/jacobi_omp.c" << EOF
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <omp.h>

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

void jacobi_2d(int N, double** A, double** B) {
    #pragma omp parallel for
    for (int i = 1; i < N-1; i++) {
        for (int j = 1; j < N-1; j++) {
            B[i][j] = 0.25 * (A[i-1][j] + A[i+1][j] + A[i][j-1] + A[i][j+1]);
        }
    }
}

double** alloc_matrix(int N) {
    double** M = malloc(N * sizeof(double*));
    for (int i = 0; i < N; i++) M[i] = calloc(N, sizeof(double));
    return M;
}

int main() {
    int N = $N;
    double **A = alloc_matrix(N);
    double **B = alloc_matrix(N);
    
    for (int i = 0; i < N; i++) {
        A[i][0] = 1.0; A[i][N-1] = 1.0;
        A[0][i] = 1.0; A[N-1][i] = 1.0;
    }
    
    jacobi_2d(N, A, B);
    
    double start = get_time();
    for (int iter = 0; iter < 100; iter++) {
        jacobi_2d(N, A, B);
        double **tmp = A; A = B; B = tmp;
    }
    double end = get_time();
    
    printf("Time: %.6f seconds\n", (end - start) / 100.0);
    printf("Check: A[N/2][N/2]=%.6f\n", A[N/2][N/2]);
    return 0;
}
EOF

    gcc -O3 -march=native -o "$BENCH_DIR/jacobi_seq" "$BENCH_DIR/jacobi_seq.c" -lm
    gcc -O3 -march=native -fopenmp -o "$BENCH_DIR/jacobi_omp" "$BENCH_DIR/jacobi_omp.c" -lm
    
    echo -e "${YELLOW}Running sequential...${NC}"
    local seq_time=$(run_bench "$BENCH_DIR/jacobi_seq")
    
    echo -e "${YELLOW}Running OpenMP...${NC}"
    local omp_time=$(run_bench "$BENCH_DIR/jacobi_omp")
    
    if [[ "$seq_time" != "FAILED" ]] && [[ "$omp_time" != "FAILED" ]]; then
        local speedup=$(echo "scale=2; $seq_time / $omp_time" | bc 2>/dev/null || echo "N/A")
        echo -e "  Sequential: ${BOLD}${seq_time}s${NC}"
        echo -e "  OpenMP:     ${BOLD}${omp_time}s${NC}"
        echo -e "  Speedup:    ${GREEN}${BOLD}${speedup}x${NC}"
        RESULTS["jacobi"]="$speedup"
    else
        echo -e "  ${RED}Failed${NC}"
        RESULTS["jacobi"]="FAILED"
    fi
    echo ""
}

#######################################
# Benchmark 4: Dot Product (Reduction)
#######################################
benchmark_reduction() {
    local N=$((SIZE * 10))
    
    echo -e "${BOLD}${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BOLD}Benchmark: Dot Product Reduction (N=$N)${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    
    # Sequential
    cat > "$BENCH_DIR/dot_seq.c" << EOF
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

double dot_product(int N, double* A, double* B) {
    double sum = 0.0;
    for (int i = 0; i < N; i++) {
        sum += A[i] * B[i];
    }
    return sum;
}

int main() {
    int N = $N;
    double *A = malloc(N * sizeof(double));
    double *B = malloc(N * sizeof(double));
    
    for (int i = 0; i < N; i++) { A[i] = 1.0; B[i] = 1.0; }
    
    volatile double r = dot_product(N, A, B);
    
    double start = get_time();
    for (int iter = 0; iter < 100; iter++) {
        r = dot_product(N, A, B);
    }
    double end = get_time();
    
    printf("Time: %.6f seconds\n", (end - start) / 100.0);
    printf("Check: result=%.0f\n", r);
    
    free(A); free(B);
    return 0;
}
EOF

    # OpenMP with reduction
    cat > "$BENCH_DIR/dot_omp.c" << EOF
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <omp.h>

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

double dot_product(int N, double* A, double* B) {
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < N; i++) {
        sum += A[i] * B[i];
    }
    return sum;
}

int main() {
    int N = $N;
    double *A = malloc(N * sizeof(double));
    double *B = malloc(N * sizeof(double));
    
    for (int i = 0; i < N; i++) { A[i] = 1.0; B[i] = 1.0; }
    
    volatile double r = dot_product(N, A, B);
    
    double start = get_time();
    for (int iter = 0; iter < 100; iter++) {
        r = dot_product(N, A, B);
    }
    double end = get_time();
    
    printf("Time: %.6f seconds\n", (end - start) / 100.0);
    printf("Check: result=%.0f\n", r);
    
    free(A); free(B);
    return 0;
}
EOF

    gcc -O3 -march=native -o "$BENCH_DIR/dot_seq" "$BENCH_DIR/dot_seq.c" -lm
    gcc -O3 -march=native -fopenmp -o "$BENCH_DIR/dot_omp" "$BENCH_DIR/dot_omp.c" -lm
    
    echo -e "${YELLOW}Running sequential...${NC}"
    local seq_time=$(run_bench "$BENCH_DIR/dot_seq")
    
    echo -e "${YELLOW}Running OpenMP...${NC}"
    local omp_time=$(run_bench "$BENCH_DIR/dot_omp")
    
    if [[ "$seq_time" != "FAILED" ]] && [[ "$omp_time" != "FAILED" ]]; then
        local speedup=$(echo "scale=2; $seq_time / $omp_time" | bc 2>/dev/null || echo "N/A")
        echo -e "  Sequential: ${BOLD}${seq_time}s${NC}"
        echo -e "  OpenMP:     ${BOLD}${omp_time}s${NC}"
        echo -e "  Speedup:    ${GREEN}${BOLD}${speedup}x${NC}"
        RESULTS["dot_product"]="$speedup"
    else
        echo -e "  ${RED}Failed${NC}"
        RESULTS["dot_product"]="FAILED"
    fi
    echo ""
}

#######################################
# Benchmark 5: Matrix Transpose
#######################################
benchmark_transpose() {
    local N=$((SIZE / 2))
    [ $N -gt 2000 ] && N=2000
    
    echo -e "${BOLD}${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BOLD}Benchmark: Matrix Transpose (N=$N)${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    
    # Sequential
    cat > "$BENCH_DIR/trans_seq.c" << EOF
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

void transpose(int N, double** A, double** B) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            B[j][i] = A[i][j];
        }
    }
}

double** alloc_matrix(int N) {
    double** M = malloc(N * sizeof(double*));
    for (int i = 0; i < N; i++) M[i] = malloc(N * sizeof(double));
    return M;
}

int main() {
    int N = $N;
    double **A = alloc_matrix(N);
    double **B = alloc_matrix(N);
    
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            A[i][j] = i * N + j;
    
    transpose(N, A, B);
    
    double start = get_time();
    for (int iter = 0; iter < 100; iter++) {
        transpose(N, A, B);
    }
    double end = get_time();
    
    printf("Time: %.6f seconds\n", (end - start) / 100.0);
    printf("Check: B[0][1]=%.0f (expect %d)\n", B[0][1], N);
    return 0;
}
EOF

    # OpenMP
    cat > "$BENCH_DIR/trans_omp.c" << EOF
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <omp.h>

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

void transpose(int N, double** A, double** B) {
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            B[j][i] = A[i][j];
        }
    }
}

double** alloc_matrix(int N) {
    double** M = malloc(N * sizeof(double*));
    for (int i = 0; i < N; i++) M[i] = malloc(N * sizeof(double));
    return M;
}

int main() {
    int N = $N;
    double **A = alloc_matrix(N);
    double **B = alloc_matrix(N);
    
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            A[i][j] = i * N + j;
    
    transpose(N, A, B);
    
    double start = get_time();
    for (int iter = 0; iter < 100; iter++) {
        transpose(N, A, B);
    }
    double end = get_time();
    
    printf("Time: %.6f seconds\n", (end - start) / 100.0);
    printf("Check: B[0][1]=%.0f (expect %d)\n", B[0][1], N);
    return 0;
}
EOF

    gcc -O3 -march=native -o "$BENCH_DIR/trans_seq" "$BENCH_DIR/trans_seq.c" -lm
    gcc -O3 -march=native -fopenmp -o "$BENCH_DIR/trans_omp" "$BENCH_DIR/trans_omp.c" -lm
    
    echo -e "${YELLOW}Running sequential...${NC}"
    local seq_time=$(run_bench "$BENCH_DIR/trans_seq")
    
    echo -e "${YELLOW}Running OpenMP...${NC}"
    local omp_time=$(run_bench "$BENCH_DIR/trans_omp")
    
    if [[ "$seq_time" != "FAILED" ]] && [[ "$omp_time" != "FAILED" ]]; then
        local speedup=$(echo "scale=2; $seq_time / $omp_time" | bc 2>/dev/null || echo "N/A")
        echo -e "  Sequential: ${BOLD}${seq_time}s${NC}"
        echo -e "  OpenMP:     ${BOLD}${omp_time}s${NC}"
        echo -e "  Speedup:    ${GREEN}${BOLD}${speedup}x${NC}"
        RESULTS["transpose"]="$speedup"
    else
        echo -e "  ${RED}Failed${NC}"
        RESULTS["transpose"]="FAILED"
    fi
    echo ""
}

#######################################
# Main
#######################################

benchmark_vector_add
benchmark_matmul
benchmark_jacobi
benchmark_reduction
benchmark_transpose

# Summary
echo -e "${BOLD}${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo -e "${BOLD}${BLUE}                      Summary                                    ${NC}"
echo -e "${BOLD}${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "System: ${BOLD}$NCPUS cores${NC}"
echo ""

printf "  %-15s %-12s %s\n" "Benchmark" "Speedup" "Status"
printf "  %-15s %-12s %s\n" "───────────────" "────────────" "──────"

total=0
count=0

for bench in vector_add matmul jacobi dot_product transpose; do
    speedup=${RESULTS[$bench]:-"N/A"}
    if [[ "$speedup" != "FAILED" ]] && [[ "$speedup" != "N/A" ]]; then
        if (( $(echo "$speedup >= 1.5" | bc -l 2>/dev/null || echo 0) )); then
            status="${GREEN}✓ Good${NC}"
        elif (( $(echo "$speedup >= 1.0" | bc -l 2>/dev/null || echo 0) )); then
            status="${YELLOW}~ OK${NC}"
        else
            status="${RED}✗ Slower${NC}"
        fi
        printf "  %-15s ${BOLD}%-12s${NC} %b\n" "$bench" "${speedup}x" "$status"
        total=$(echo "$total + $speedup" | bc)
        count=$((count + 1))
    else
        printf "  %-15s %-12s ${RED}%s${NC}\n" "$bench" "-" "Failed"
    fi
done

echo ""
if [ $count -gt 0 ]; then
    avg=$(echo "scale=2; $total / $count" | bc)
    echo -e "  Average speedup: ${GREEN}${BOLD}${avg}x${NC} across $count benchmarks"
fi

echo ""
echo -e "${CYAN}Legend:${NC}"
echo "  ✓ Good   = 1.5x+ speedup (good parallel scaling)"
echo "  ~ OK     = 1.0-1.5x speedup (some benefit)"
echo "  ✗ Slower = <1.0x (thread overhead > benefit)"
echo ""
echo -e "${CYAN}Tips for better speedup:${NC}"
echo "  • Use larger problem sizes (--size=4000)"
echo "  • Memory-bound ops (transpose) scale less"
echo "  • Compute-bound ops (matmul) scale better"
echo ""
