#!/bin/bash
#
# PolyOpt Demo Script - Run all examples
# Usage: ./run_all_examples.sh [--quick] [--verbose]
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Options
QUICK=false
VERBOSE=false
COMPILE_OUTPUT=false

for arg in "$@"; do
    case $arg in
        --quick) QUICK=true ;;
        --verbose) VERBOSE=true ;;
        --compile) COMPILE_OUTPUT=true ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --quick     Skip visualization and use smaller parameters"
            echo "  --verbose   Show full output from each command"
            echo "  --compile   Also generate C code for each example"
            echo "  --help      Show this help message"
            exit 0
            ;;
    esac
done

# Build first
echo -e "${BOLD}${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo -e "${BOLD}${BLUE}                    PolyOpt Example Runner                       ${NC}"
echo -e "${BOLD}${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo ""

echo -e "${YELLOW}Building PolyOpt...${NC}"
cargo build --release --bin polyopt --bin polyvis 2>/dev/null || cargo build --bin polyopt --bin polyvis
echo -e "${GREEN}✓ Build complete${NC}"
echo ""

# Detect binary location
if [ -f "./target/release/polyopt" ]; then
    POLYOPT="./target/release/polyopt"
    POLYVIS="./target/release/polyvis"
else
    POLYOPT="./target/debug/polyopt"
    POLYVIS="./target/debug/polyvis"
fi

# Counters
TOTAL=0
PASSED=0
FAILED=0

# Function to run a single example
run_example() {
    local file=$1
    local name=$(basename "$file" .poly)
    local dir=$(dirname "$file" | sed 's|.*/examples/||')
    
    TOTAL=$((TOTAL + 1))
    
    echo -e "${BOLD}${CYAN}────────────────────────────────────────────────────────────────${NC}"
    echo -e "${BOLD}[$TOTAL] ${dir}/${name}${NC}"
    echo -e "${CYAN}────────────────────────────────────────────────────────────────${NC}"
    
    # Show the source (first 10 lines)
    echo -e "${YELLOW}Source:${NC}"
    head -15 "$file" | sed 's/^/  /'
    echo ""
    
    # 1. Parse check
    echo -e "${YELLOW}▶ Parsing...${NC}"
    if $POLYOPT parse "$file" --pir > /dev/null 2>&1; then
        echo -e "  ${GREEN}✓ Parse successful${NC}"
    else
        echo -e "  ${RED}✗ Parse failed${NC}"
        FAILED=$((FAILED + 1))
        return
    fi
    
    # 2. Analyze dependencies
    echo -e "${YELLOW}▶ Analyzing dependencies...${NC}"
    local analysis=$($POLYOPT analyze "$file" --deps --parallel 2>/dev/null)
    
    # Extract key info
    local deps=$(echo "$analysis" | grep "^Total:" | awk '{print $2}')
    local flow=$(echo "$analysis" | grep "Flow (RAW):" | awk '{print $3}')
    local anti=$(echo "$analysis" | grep "Anti (WAR):" | awk '{print $3}')
    local output=$(echo "$analysis" | grep "Output (WAW):" | awk '{print $3}')
    
    echo -e "  Dependencies: ${BOLD}$deps${NC} (Flow: $flow, Anti: $anti, Output: $output)"
    
    # Check parallelism
    if echo "$analysis" | grep -q "✓ PARALLEL"; then
        local parallel_levels=$(echo "$analysis" | grep "✓ PARALLEL" | wc -l)
        echo -e "  Parallelism: ${GREEN}✓ $parallel_levels parallel loop(s) found${NC}"
    else
        echo -e "  Parallelism: ${YELLOW}✗ No directly parallel loops${NC}"
    fi
    
    if $VERBOSE; then
        echo ""
        echo "$analysis" | sed 's/^/  /'
    fi
    
    # 3. Visualize (unless --quick)
    if ! $QUICK; then
        echo -e "${YELLOW}▶ Visualizing iteration space...${NC}"
        # Determine parameters based on file
        local params="N=6"
        if echo "$file" | grep -q "matmul\|linalg"; then
            params="N=4,M=4,K=4"
        elif echo "$file" | grep -q "matvec"; then
            params="N=4,M=4"
        fi
        
        local vis=$($POLYVIS "$file" --params "$params" --max-iters 50 2>/dev/null || echo "Visualization skipped")
        if [ -n "$vis" ]; then
            echo "$vis" | grep -E "(Statement|Iterations|Dependences|Parallel|●|Domain)" | head -10 | sed 's/^/  /'
        fi
    fi
    
    # 4. Generate code (if --compile or always show snippet)
    echo -e "${YELLOW}▶ Generating C code...${NC}"
    local code=$($POLYOPT compile "$file" --openmp 2>/dev/null)
    
    if [ -n "$code" ]; then
        echo -e "  ${GREEN}✓ Code generation successful${NC}"
        
        # Show function signature
        local sig=$(echo "$code" | grep "^void " | head -1)
        echo -e "  Signature: ${BOLD}$sig${NC}"
        
        # Check for OpenMP pragmas
        local omp_count=$(echo "$code" | grep -c "#pragma omp" || true)
        if [ "$omp_count" -gt 0 ]; then
            echo -e "  OpenMP pragmas: ${GREEN}$omp_count${NC}"
        fi
        
        if $COMPILE_OUTPUT; then
            echo ""
            echo -e "${YELLOW}Generated code:${NC}"
            echo "$code" | sed 's/^/  /'
        fi
    else
        echo -e "  ${RED}✗ Code generation failed${NC}"
    fi
    
    PASSED=$((PASSED + 1))
    echo ""
}

# Find and run all examples
echo -e "${BOLD}${BLUE}Running Examples...${NC}"
echo ""

# Run examples by category for better organization
for category in basic linalg stencils reductions parallel transformations; do
    if [ -d "examples/$category" ]; then
        echo -e "${BOLD}${BLUE}━━━ Category: $category ━━━${NC}"
        echo ""
        
        for file in examples/$category/*.poly; do
            if [ -f "$file" ]; then
                run_example "$file"
            fi
        done
    fi
done

# Run root-level examples
echo -e "${BOLD}${BLUE}━━━ Category: root ━━━${NC}"
echo ""
for file in examples/*.poly; do
    if [ -f "$file" ]; then
        run_example "$file"
    fi
done

# Summary
echo -e "${BOLD}${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo -e "${BOLD}${BLUE}                         Summary                                 ${NC}"
echo -e "${BOLD}${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "  Total examples: ${BOLD}$TOTAL${NC}"
echo -e "  Passed:         ${GREEN}${BOLD}$PASSED${NC}"
echo -e "  Failed:         ${RED}${BOLD}$FAILED${NC}"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}${BOLD}All examples completed successfully! ✓${NC}"
else
    echo -e "${YELLOW}${BOLD}Some examples had issues. Check output above.${NC}"
fi

echo ""
echo -e "${CYAN}Tips:${NC}"
echo "  • Run with --verbose to see full analysis output"
echo "  • Run with --compile to see generated C code"
echo "  • Run with --quick to skip visualization"
echo ""
