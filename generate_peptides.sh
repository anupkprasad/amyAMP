#!/bin/bash
#
# Simple Peptide Generator
# ========================
# Generate peptides using the trained BiGAN model.
#
# Usage:
#   ./generate_peptides.sh [output_path] [num_peptides]
#

# Configuration - get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_NUM=1000

# Parse arguments
OUTPUT="${1:-generated_peptides_${DEFAULT_NUM}.fasta}"
NUM="${2:-$DEFAULT_NUM}"

# If output is a directory, add filename
if [ -d "$OUTPUT" ]; then
    OUTPUT="${OUTPUT}/generated_peptides_${NUM}.fasta"
fi

# Create output directory if needed
mkdir -p "$(dirname "$OUTPUT")" 2>/dev/null

# Show what we're doing
echo "================================================"
echo "Peptide Generator"
echo "================================================"
echo "Script directory: $SCRIPT_DIR"
echo "Output: $OUTPUT"
echo "Number: $NUM peptides"
echo "================================================"
echo ""

# Change to script directory and run
cd "$SCRIPT_DIR"
python3 -c "
import sys
import os
# Add script directory to path
sys.path.insert(0, '${SCRIPT_DIR}')
from main import main
import main as m
m.BATCH_GENERATE = ${NUM}
main('generate', '${OUTPUT}')
"

# Check result
if [ $? -eq 0 ] && [ -f "$OUTPUT" ]; then
    SEQS=$(grep -c "^>" "$OUTPUT" 2>/dev/null || echo "0")
    SIZE=$(du -h "$OUTPUT" 2>/dev/null | cut -f1 || echo "?")
    echo ""
    echo "================================================"
    echo "✓ Success!"
    echo "  File: $OUTPUT"
    echo "  Size: $SIZE"
    echo "  Sequences: $SEQS"
    echo "================================================"
else
    echo ""
    echo "✗ Generation failed"
    exit 1
fi
