#!/bin/bash
#
# PolyOpt Benchmark Suite
# Tests sequential vs OpenMP parallel performance
# Compatible with bash 3.x (macOS default)
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

# Parse arguments
for arg in "$@"; do
    case $arg in
        --size=*) SIZE="${arg#*=}" ;;
        --iterations=*) ITERATIONS="${arg#*=}" ;;
        --quick) SIZE=1000; ITERATIONS=2 ;;
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

# Setup temp directory
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

# Detect compiler with OpenMP support
CC_OMP=""
OMP_FLAGS=""

# Try gcc versions first
for gcc_ver in gcc-14 gcc-13 gcc-12 gcc-11 gcc; do
    if command -v $gcc_ver &>/dev/null; then
        if $gcc_ver -fopenmp -x c -c /dev/null -o /dev/null 2>/dev/null; then
            CC_OMP="$gcc_ver"
            OMP_FLAGS="-fopenmp"
            break
        fi
    fi
done

# Try clang with libomp on macOS
if [ -z "$CC_OMP" ] && command -v clang &>/dev/null; then
    if [ -d "/opt/homebrew/opt/libomp" ]; then
        CC_OMP="clang"
        OMP_FLAGS="-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include -L/opt/homebrew/opt/libomp/lib -lomp"
    elif [ -d "/usr/local/opt/libomp" ]; then
        CC_OMP="clang"
        OMP_FLAGS="-Xpreprocessor -fopenmp -I/usr/local/opt/libomp/include -L/usr/local/opt/libomp/lib -lomp"
    fi
fi

if [ -z "$CC_OMP" ]; then
    echo -e "${RED}ERROR: No OpenMP-capable compiler found!${NC}"
    echo ""
    echo "On macOS, install GCC via Homebrew:"
    echo "  brew install gcc"
    echo ""
    echo "Or install libomp for clang:"
    echo "  brew install libomp"
    echo ""
    exit 1
fi

# Sequential compiler (any will do)
CC_SEQ="${CC_OMP}"

echo -e "Compiler: ${BOLD}$CC_OMP${NC} with OpenMP"
echo ""

# Results storage (simple variables for bash 3.x compatibility)
RESULT_vector_add="N/A"
RESULT_matmul="N/A"
RESULT_jacobi="N/A"
RESULT_dot_product="N/A"
RESULT_transpose="N/A"

#######################################
# Run benchmark and get median time
#######################################
run_bench() {
    local exe=$1
    local times_file="$BENCH_DIR/times.txt"
    > "$times_file"
    
    for i in $(seq 1 $ITERATIONS); do
        local output=$("$exe" 2>&1)
        local t=$(echo "$output" | grep -o 'Time: [0-9.]*' | grep -o '[0-9.]*')
        if [ -n "$t" ]; then
            echo "$t" >> "$times_file"
        fi
    done
    
    if [ ! -s "$times_file" ]; then
        echo "FAILED"
        return
    fi
    
    # Return median
    sort -n "$times_file" | sed -n "$((ITERATIONS/2 + 1))p"
}

#######################################
# Benchmark 1: Vector Addition
#######################################
benchmark_vector_add() {
    # Vector add is memory-bound, needs HUGE size to see benefit
    local N=$((SIZE * 500))  # 500x larger - need millions of elements
    
    echo -e "${BOLD}${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BOLD}Benchmark: Vector Addition (N=$N)${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    
    cat > "$BENCH_DIR/vec_seq.c" << EOF
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

int main() {
    int N = $N;
    double *A = malloc(N * sizeof(double));
    double *B = malloc(N * sizeof(double));
    double *C = malloc(N * sizeof(double));
    
    for (int i = 0; i < N; i++) { A[i] = i * 0.1; B[i] = i * 0.2; }
    
    // Warmup
    for (int i = 0; i < N; i++) C[i] = A[i] + B[i];
    
    double start = get_time();
    for (int r = 0; r < 10; r++) {
        for (int i = 0; i < N; i++) {
            C[i] = A[i] + B[i];
        }
    }
    double end = get_time();
    
    // Prevent dead code elimination
    volatile double sink = C[N/2];
    (void)sink;
    
    printf("Time: %.6f seconds\\n", (end - start) / 10.0);
    
    free(A); free(B); free(C);
    return 0;
}
EOF

    cat > "$BENCH_DIR/vec_omp.c" << EOF
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

int main() {
    int N = $N;
    double *A = malloc(N * sizeof(double));
    double *B = malloc(N * sizeof(double));
    double *C = malloc(N * sizeof(double));
    
    for (int i = 0; i < N; i++) { A[i] = i * 0.1; B[i] = i * 0.2; }
    
    // Warmup
    #pragma omp parallel for
    for (int i = 0; i < N; i++) C[i] = A[i] + B[i];
    
    double start = get_time();
    for (int r = 0; r < 10; r++) {
        #pragma omp parallel for
        for (int i = 0; i < N; i++) {
            C[i] = A[i] + B[i];
        }
    }
    double end = get_time();
    
    // Prevent dead code elimination
    volatile double sink = C[N/2];
    (void)sink;
    
    printf("Time: %.6f seconds\\n", (end - start) / 10.0);
    
    free(A); free(B); free(C);
    return 0;
}
EOF

    $CC_SEQ -O3 -o "$BENCH_DIR/vec_seq" "$BENCH_DIR/vec_seq.c" -lm 2>/dev/null
    $CC_OMP -O3 $OMP_FLAGS -o "$BENCH_DIR/vec_omp" "$BENCH_DIR/vec_omp.c" -lm 2>/dev/null
    
    echo -e "${YELLOW}Running sequential...${NC}"
    local seq_time=$(run_bench "$BENCH_DIR/vec_seq")
    
    echo -e "${YELLOW}Running OpenMP...${NC}"
    local omp_time=$(run_bench "$BENCH_DIR/vec_omp")
    
    echo -e "  Sequential: ${BOLD}${seq_time}s${NC}"
    echo -e "  OpenMP:     ${BOLD}${omp_time}s${NC}"
    
    if [ "$seq_time" != "FAILED" ] && [ "$omp_time" != "FAILED" ]; then
        RESULT_vector_add=$(echo "scale=2; $seq_time / $omp_time" | bc 2>/dev/null || echo "N/A")
        echo -e "  Speedup:    ${GREEN}${BOLD}${RESULT_vector_add}x${NC}"
    else
        RESULT_vector_add="FAILED"
        echo -e "  ${RED}Failed${NC}"
    fi
    echo ""
}

#######################################
# Benchmark 2: Matrix Multiplication
#######################################
benchmark_matmul() {
    local N=$((SIZE / 4))
    [ $N -lt 100 ] && N=100
    [ $N -gt 800 ] && N=800
    
    echo -e "${BOLD}${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BOLD}Benchmark: Matrix Multiplication (N=$N)${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    
    cat > "$BENCH_DIR/mm_seq.c" << EOF
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

int main() {
    int N = $N;
    double *A = malloc(N * N * sizeof(double));
    double *B = malloc(N * N * sizeof(double));
    double *C = malloc(N * N * sizeof(double));
    
    for (int i = 0; i < N*N; i++) { A[i] = (i % 100) * 0.01; B[i] = (i % 100) * 0.01; }
    
    double start = get_time();
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0.0;
            for (int k = 0; k < N; k++) {
                sum += A[i*N + k] * B[k*N + j];
            }
            C[i*N + j] = sum;
        }
    }
    double end = get_time();
    
    printf("Time: %.6f seconds\\n", end - start);
    printf("Check: C[0]=%.4f\\n", C[0]);
    
    free(A); free(B); free(C);
    return 0;
}
EOF

    cat > "$BENCH_DIR/mm_omp.c" << EOF
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

int main() {
    int N = $N;
    double *A = malloc(N * N * sizeof(double));
    double *B = malloc(N * N * sizeof(double));
    double *C = malloc(N * N * sizeof(double));
    
    for (int i = 0; i < N*N; i++) { A[i] = (i % 100) * 0.01; B[i] = (i % 100) * 0.01; }
    
    double start = get_time();
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0.0;
            for (int k = 0; k < N; k++) {
                sum += A[i*N + k] * B[k*N + j];
            }
            C[i*N + j] = sum;
        }
    }
    double end = get_time();
    
    printf("Time: %.6f seconds\\n", end - start);
    printf("Check: C[0]=%.4f\\n", C[0]);
    
    free(A); free(B); free(C);
    return 0;
}
EOF

    $CC_SEQ -O3 -o "$BENCH_DIR/mm_seq" "$BENCH_DIR/mm_seq.c" -lm 2>/dev/null
    $CC_OMP -O3 $OMP_FLAGS -o "$BENCH_DIR/mm_omp" "$BENCH_DIR/mm_omp.c" -lm 2>/dev/null
    
    echo -e "${YELLOW}Running sequential...${NC}"
    local seq_time=$(run_bench "$BENCH_DIR/mm_seq")
    
    echo -e "${YELLOW}Running OpenMP...${NC}"
    local omp_time=$(run_bench "$BENCH_DIR/mm_omp")
    
    echo -e "  Sequential: ${BOLD}${seq_time}s${NC}"
    echo -e "  OpenMP:     ${BOLD}${omp_time}s${NC}"
    
    if [ "$seq_time" != "FAILED" ] && [ "$omp_time" != "FAILED" ]; then
        RESULT_matmul=$(echo "scale=2; $seq_time / $omp_time" | bc 2>/dev/null || echo "N/A")
        echo -e "  Speedup:    ${GREEN}${BOLD}${RESULT_matmul}x${NC}"
    else
        RESULT_matmul="FAILED"
        echo -e "  ${RED}Failed${NC}"
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
    
    cat > "$BENCH_DIR/jacobi_seq.c" << EOF
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

int main() {
    int N = $N;
    double *A = calloc(N * N, sizeof(double));
    double *B = calloc(N * N, sizeof(double));
    
    // Boundary conditions
    for (int i = 0; i < N; i++) {
        A[i] = 1.0; A[i*N] = 1.0; 
        A[(N-1)*N + i] = 1.0; A[i*N + N-1] = 1.0;
    }
    
    double start = get_time();
    for (int iter = 0; iter < 100; iter++) {
        for (int i = 1; i < N-1; i++) {
            for (int j = 1; j < N-1; j++) {
                B[i*N + j] = 0.25 * (A[(i-1)*N + j] + A[(i+1)*N + j] + 
                                     A[i*N + j-1] + A[i*N + j+1]);
            }
        }
        double *tmp = A; A = B; B = tmp;
    }
    double end = get_time();
    
    printf("Time: %.6f seconds\\n", (end - start) / 100.0);
    
    free(A); free(B);
    return 0;
}
EOF

    cat > "$BENCH_DIR/jacobi_omp.c" << EOF
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

int main() {
    int N = $N;
    double *A = calloc(N * N, sizeof(double));
    double *B = calloc(N * N, sizeof(double));
    
    for (int i = 0; i < N; i++) {
        A[i] = 1.0; A[i*N] = 1.0;
        A[(N-1)*N + i] = 1.0; A[i*N + N-1] = 1.0;
    }
    
    double start = get_time();
    for (int iter = 0; iter < 100; iter++) {
        #pragma omp parallel for
        for (int i = 1; i < N-1; i++) {
            for (int j = 1; j < N-1; j++) {
                B[i*N + j] = 0.25 * (A[(i-1)*N + j] + A[(i+1)*N + j] + 
                                     A[i*N + j-1] + A[i*N + j+1]);
            }
        }
        double *tmp = A; A = B; B = tmp;
    }
    double end = get_time();
    
    printf("Time: %.6f seconds\\n", (end - start) / 100.0);
    
    free(A); free(B);
    return 0;
}
EOF

    $CC_SEQ -O3 -o "$BENCH_DIR/jacobi_seq" "$BENCH_DIR/jacobi_seq.c" -lm 2>/dev/null
    $CC_OMP -O3 $OMP_FLAGS -o "$BENCH_DIR/jacobi_omp" "$BENCH_DIR/jacobi_omp.c" -lm 2>/dev/null
    
    echo -e "${YELLOW}Running sequential...${NC}"
    local seq_time=$(run_bench "$BENCH_DIR/jacobi_seq")
    
    echo -e "${YELLOW}Running OpenMP...${NC}"
    local omp_time=$(run_bench "$BENCH_DIR/jacobi_omp")
    
    echo -e "  Sequential: ${BOLD}${seq_time}s${NC}"
    echo -e "  OpenMP:     ${BOLD}${omp_time}s${NC}"
    
    if [ "$seq_time" != "FAILED" ] && [ "$omp_time" != "FAILED" ]; then
        RESULT_jacobi=$(echo "scale=2; $seq_time / $omp_time" | bc 2>/dev/null || echo "N/A")
        echo -e "  Speedup:    ${GREEN}${BOLD}${RESULT_jacobi}x${NC}"
    else
        RESULT_jacobi="FAILED"
        echo -e "  ${RED}Failed${NC}"
    fi
    echo ""
}

#######################################
# Benchmark 4: Dot Product (Reduction)
#######################################
benchmark_dot_product() {
    # Reductions need large N to overcome thread sync overhead
    local N=$((SIZE * 100))  # 100x for reduction overhead
    
    echo -e "${BOLD}${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BOLD}Benchmark: Dot Product Reduction (N=$N)${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    
    cat > "$BENCH_DIR/dot_seq.c" << EOF
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

int main() {
    int N = $N;
    double *A = malloc(N * sizeof(double));
    double *B = malloc(N * sizeof(double));
    
    for (int i = 0; i < N; i++) { A[i] = 1.0; B[i] = 2.0; }
    
    volatile double result = 0;
    
    double start = get_time();
    for (int r = 0; r < 100; r++) {
        double sum = 0.0;
        for (int i = 0; i < N; i++) {
            sum += A[i] * B[i];
        }
        result = sum;
    }
    double end = get_time();
    
    printf("Time: %.6f seconds\\n", (end - start) / 100.0);
    printf("Check: result=%.0f\\n", result);
    
    free(A); free(B);
    return 0;
}
EOF

    cat > "$BENCH_DIR/dot_omp.c" << EOF
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

int main() {
    int N = $N;
    double *A = malloc(N * sizeof(double));
    double *B = malloc(N * sizeof(double));
    
    for (int i = 0; i < N; i++) { A[i] = 1.0; B[i] = 2.0; }
    
    volatile double result = 0;
    
    double start = get_time();
    for (int r = 0; r < 100; r++) {
        double sum = 0.0;
        #pragma omp parallel for reduction(+:sum)
        for (int i = 0; i < N; i++) {
            sum += A[i] * B[i];
        }
        result = sum;
    }
    double end = get_time();
    
    printf("Time: %.6f seconds\\n", (end - start) / 100.0);
    printf("Check: result=%.0f\\n", result);
    
    free(A); free(B);
    return 0;
}
EOF

    $CC_SEQ -O3 -o "$BENCH_DIR/dot_seq" "$BENCH_DIR/dot_seq.c" -lm 2>/dev/null
    $CC_OMP -O3 $OMP_FLAGS -o "$BENCH_DIR/dot_omp" "$BENCH_DIR/dot_omp.c" -lm 2>/dev/null
    
    echo -e "${YELLOW}Running sequential...${NC}"
    local seq_time=$(run_bench "$BENCH_DIR/dot_seq")
    
    echo -e "${YELLOW}Running OpenMP...${NC}"
    local omp_time=$(run_bench "$BENCH_DIR/dot_omp")
    
    echo -e "  Sequential: ${BOLD}${seq_time}s${NC}"
    echo -e "  OpenMP:     ${BOLD}${omp_time}s${NC}"
    
    if [ "$seq_time" != "FAILED" ] && [ "$omp_time" != "FAILED" ]; then
        RESULT_dot_product=$(echo "scale=2; $seq_time / $omp_time" | bc 2>/dev/null || echo "N/A")
        echo -e "  Speedup:    ${GREEN}${BOLD}${RESULT_dot_product}x${NC}"
    else
        RESULT_dot_product="FAILED"
        echo -e "  ${RED}Failed${NC}"
    fi
    echo ""
}

#######################################
# Benchmark 5: Matrix Transpose
#######################################
benchmark_transpose() {
    # Transpose is memory-bound, needs larger size
    local N=$((SIZE))
    [ $N -lt 1000 ] && N=1000
    [ $N -gt 4000 ] && N=4000
    
    echo -e "${BOLD}${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BOLD}Benchmark: Matrix Transpose (N=$N)${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    
    cat > "$BENCH_DIR/trans_seq.c" << EOF
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

int main() {
    int N = $N;
    double *A = malloc(N * N * sizeof(double));
    double *B = malloc(N * N * sizeof(double));
    
    for (int i = 0; i < N*N; i++) A[i] = i;
    
    double start = get_time();
    for (int r = 0; r < 100; r++) {
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                B[j*N + i] = A[i*N + j];
            }
        }
    }
    double end = get_time();
    
    printf("Time: %.6f seconds\\n", (end - start) / 100.0);
    
    free(A); free(B);
    return 0;
}
EOF

    cat > "$BENCH_DIR/trans_omp.c" << EOF
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

int main() {
    int N = $N;
    double *A = malloc(N * N * sizeof(double));
    double *B = malloc(N * N * sizeof(double));
    
    for (int i = 0; i < N*N; i++) A[i] = i;
    
    double start = get_time();
    for (int r = 0; r < 100; r++) {
        #pragma omp parallel for
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                B[j*N + i] = A[i*N + j];
            }
        }
    }
    double end = get_time();
    
    printf("Time: %.6f seconds\\n", (end - start) / 100.0);
    
    free(A); free(B);
    return 0;
}
EOF

    $CC_SEQ -O3 -o "$BENCH_DIR/trans_seq" "$BENCH_DIR/trans_seq.c" -lm 2>/dev/null
    $CC_OMP -O3 $OMP_FLAGS -o "$BENCH_DIR/trans_omp" "$BENCH_DIR/trans_omp.c" -lm 2>/dev/null
    
    echo -e "${YELLOW}Running sequential...${NC}"
    local seq_time=$(run_bench "$BENCH_DIR/trans_seq")
    
    echo -e "${YELLOW}Running OpenMP...${NC}"
    local omp_time=$(run_bench "$BENCH_DIR/trans_omp")
    
    echo -e "  Sequential: ${BOLD}${seq_time}s${NC}"
    echo -e "  OpenMP:     ${BOLD}${omp_time}s${NC}"
    
    if [ "$seq_time" != "FAILED" ] && [ "$omp_time" != "FAILED" ]; then
        RESULT_transpose=$(echo "scale=2; $seq_time / $omp_time" | bc 2>/dev/null || echo "N/A")
        echo -e "  Speedup:    ${GREEN}${BOLD}${RESULT_transpose}x${NC}"
    else
        RESULT_transpose="FAILED"
        echo -e "  ${RED}Failed${NC}"
    fi
    echo ""
}

#######################################
# Print status for a result
#######################################
print_result() {
    local name=$1
    local speedup=$2
    
    if [ "$speedup" = "FAILED" ] || [ "$speedup" = "N/A" ]; then
        printf "  %-15s %-12s ${RED}%s${NC}\n" "$name" "-" "Failed"
        return
    fi
    
    local status=""
    local cmp=$(echo "$speedup >= 1.5" | bc -l 2>/dev/null || echo "0")
    if [ "$cmp" = "1" ]; then
        status="${GREEN}✓ Good${NC}"
    else
        cmp=$(echo "$speedup >= 1.0" | bc -l 2>/dev/null || echo "0")
        if [ "$cmp" = "1" ]; then
            status="${YELLOW}~ OK${NC}"
        else
            status="${RED}✗ Slower${NC}"
        fi
    fi
    
    printf "  %-15s ${BOLD}%-12s${NC} %b\n" "$name" "${speedup}x" "$status"
}

#######################################
# Main
#######################################

benchmark_vector_add
benchmark_matmul
benchmark_jacobi
benchmark_dot_product
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

print_result "vector_add" "$RESULT_vector_add"
print_result "matmul" "$RESULT_matmul"
print_result "jacobi" "$RESULT_jacobi"
print_result "dot_product" "$RESULT_dot_product"
print_result "transpose" "$RESULT_transpose"

# Calculate average
total=0
count=0
for sp in $RESULT_vector_add $RESULT_matmul $RESULT_jacobi $RESULT_dot_product $RESULT_transpose; do
    if [ "$sp" != "FAILED" ] && [ "$sp" != "N/A" ]; then
        total=$(echo "$total + $sp" | bc 2>/dev/null || echo "$total")
        count=$((count + 1))
    fi
done

echo ""
if [ $count -gt 0 ]; then
    avg=$(echo "scale=2; $total / $count" | bc 2>/dev/null || echo "N/A")
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
echo "  • Memory-bound ops (vector_add, transpose) scale less"
echo "  • Compute-bound ops (matmul) scale better"
echo ""