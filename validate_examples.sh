#!/bin/bash
#
# Quick validation - just check all examples parse and analyze
#

set -e

echo "Building..."
cargo build --release --bin polyopt 2>/dev/null || cargo build --bin polyopt

POLYOPT="./target/release/polyopt"
[ -f "$POLYOPT" ] || POLYOPT="./target/debug/polyopt"

echo ""
echo "Validating all examples..."
echo "═══════════════════════════════════════════════════════════"

PASS=0
FAIL=0

for f in $(find examples -name "*.poly" | sort); do
    name=$(echo "$f" | sed 's|examples/||')
    
    if $POLYOPT analyze "$f" --parallel > /dev/null 2>&1; then
        # Check if parallel
        if $POLYOPT analyze "$f" --parallel 2>/dev/null | grep -q "✓ PARALLEL"; then
            echo -e "✓ $name \033[32m[PARALLEL]\033[0m"
        else
            echo -e "✓ $name \033[33m[SEQUENTIAL]\033[0m"
        fi
        PASS=$((PASS + 1))
    else
        echo -e "✗ $name \033[31m[FAILED]\033[0m"
        FAIL=$((FAIL + 1))
    fi
done

echo "═══════════════════════════════════════════════════════════"
echo "Results: $PASS passed, $FAIL failed"
