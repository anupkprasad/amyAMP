"""
Script to extract non-APR (amyloid-prone region) sequences from DBAASP database.
Creates a new FASTA file containing sequences from dbaasp.fasta that are NOT in dbaasp_APR_processed.fasta.
Removes J/Z amidation markers and ensures only unique sequences in output.
"""

from pathlib import Path
from typing import Dict, Set, Tuple

# ══════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════

BASE_DIR = Path("/home/anupkumar/workspace/amyAMP/data_master/amps/dbaasp")
FULL_FASTA = BASE_DIR / "dbaasp.fasta"
APR_FASTA = BASE_DIR / "dbaasp_APR_processed.fasta"
OUTPUT_FASTA = BASE_DIR / "dbaasp_non_APR.fasta"

# ══════════════════════════════════════════════════════════════════════
# FUNCTIONS
# ══════════════════════════════════════════════════════════════════════

def normalize_sequence(seq: str) -> str:
    """
    Normalize sequence by removing J/Z amidation markers.
    
    Parameters
    ----------
    seq : str
        Sequence potentially containing J/Z markers
        
    Returns
    -------
    str
        Normalized sequence without J/Z markers
    """
    # Remove J and Z characters (amidation markers)
    normalized = seq.replace('J', '').replace('Z', '')
    return normalized.upper()


def read_fasta(filepath: Path) -> Dict[str, str]:
    """
    Read a FASTA file and return a dictionary of {header: sequence}.
    
    Parameters
    ----------
    filepath : Path
        Path to the FASTA file
        
    Returns
    -------
    Dict[str, str]
        Dictionary mapping FASTA headers to sequences
    """
    sequences = {}
    current_header = None
    current_sequence = []
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            if line.startswith('>'):
                # Save previous sequence if exists
                if current_header is not None:
                    sequences[current_header] = ''.join(current_sequence)
                
                # Start new sequence
                current_header = line[1:]  # Remove '>' character
                current_sequence = []
            else:
                current_sequence.append(line)
        
        # Save last sequence
        if current_header is not None:
            sequences[current_header] = ''.join(current_sequence)
    
    return sequences


def write_fasta(sequences: Dict[str, str], filepath: Path) -> int:
    """
    Write sequences to a FASTA file with each sequence on a single line.
    
    Parameters
    ----------
    sequences : Dict[str, str]
        Dictionary mapping headers to sequences
    filepath : Path
        Output file path
        
    Returns
    -------
    int
        Number of sequences written
    """
    count = 0
    with open(filepath, 'w') as f:
        for header, sequence in sequences.items():
            f.write(f">{header}\n")
            # Write entire sequence on single line (no wrapping)
            f.write(f"{sequence}\n")
            count += 1
    
    return count


def get_normalized_sequence_set(sequences: Dict[str, str]) -> Set[str]:
    """
    Extract unique normalized sequences (without J/Z) from a dictionary.
    
    Parameters
    ----------
    sequences : Dict[str, str]
        Dictionary mapping headers to sequences
        
    Returns
    -------
    Set[str]
        Set of unique normalized sequences
    """
    return set(normalize_sequence(seq) for seq in sequences.values())


def subtract_sequences_unique(full_seqs: Dict[str, str], 
                              subtract_seqs: Dict[str, str]) -> Tuple[Dict[str, str], int]:
    """
    Subtract sequences: keep entries from full_seqs that are NOT in subtract_seqs.
    Normalizes sequences by removing J/Z markers before comparison.
    Removes J/Z from output sequences and ensures only unique sequences.
    
    Parameters
    ----------
    full_seqs : Dict[str, str]
        Full set of sequences
    subtract_seqs : Dict[str, str]
        Sequences to remove (may contain J/Z markers)
        
    Returns
    -------
    Tuple[Dict[str, str], int]
        (Remaining unique sequences after subtraction with J/Z removed, number of duplicates)
    """
    # Get set of normalized sequences to exclude
    exclude_sequences = get_normalized_sequence_set(subtract_seqs)
    
    print(f"\n🔍 Debug Info:")
    print(f"   Unique normalized sequences to exclude: {len(exclude_sequences)}")
    
    # Keep only sequences not in the exclusion set
    # Also track unique sequences to remove duplicates
    # Store normalized sequences (without J/Z)
    result = {}
    seen_sequences = set()
    duplicate_count = 0
    excluded_count = 0
    jz_removed_count = 0
    
    for header, seq in full_seqs.items():
        # Normalize sequence (remove J/Z)
        normalized = normalize_sequence(seq)
        
        # Check if sequence should be excluded (is an APR sequence)
        if normalized in exclude_sequences:
            excluded_count += 1
            continue
        
        # Check if sequence is duplicate
        if normalized in seen_sequences:
            duplicate_count += 1
            continue
        
        # Track if J/Z was removed
        if 'J' in seq or 'Z' in seq:
            jz_removed_count += 1
        
        # Keep this unique, non-excluded sequence (normalized, without J/Z)
        result[header] = normalized
        seen_sequences.add(normalized)
    
    print(f"   Sequences excluded (APR): {excluded_count}")
    print(f"   Duplicate sequences removed: {duplicate_count}")
    print(f"   Sequences with J/Z removed: {jz_removed_count}")
    
    return result, duplicate_count


# ══════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ══════════════════════════════════════════════════════════════════════

def main():
    """Main function to subtract FASTA files."""
    print("="*70)
    print("FASTA SUBTRACTION TOOL (Non-APR AMPs)")
    print("="*70)
    print("\nNote: J/Z amidation markers will be removed from output")
    print("Only unique sequences will be retained (duplicates removed)")
    
    # Check if files exist
    if not FULL_FASTA.exists():
        print(f"❌ Error: {FULL_FASTA} not found!")
        return
    
    if not APR_FASTA.exists():
        print(f"❌ Error: {APR_FASTA} not found!")
        return
    
    # Read FASTA files
    print(f"\n📖 Reading {FULL_FASTA.name}...")
    full_sequences = read_fasta(FULL_FASTA)
    print(f"   ✓ Loaded {len(full_sequences)} sequences")
    
    # Check for J/Z in full sequences
    jz_count_full = sum(1 for seq in full_sequences.values() if 'J' in seq or 'Z' in seq)
    if jz_count_full > 0:
        print(f"   ℹ️  Found {jz_count_full} sequences with J/Z amidation markers")
    
    print(f"\n📖 Reading {APR_FASTA.name}...")
    apr_sequences = read_fasta(APR_FASTA)
    print(f"   ✓ Loaded {len(apr_sequences)} sequences")
    
    # Check for J/Z in APR sequences
    jz_count_apr = sum(1 for seq in apr_sequences.values() if 'J' in seq or 'Z' in seq)
    if jz_count_apr > 0:
        print(f"   ℹ️  Found {jz_count_apr} sequences with J/Z amidation markers")
    
    # Perform subtraction with duplicate removal and J/Z removal
    print(f"\n🔄 Subtracting APR sequences from full dataset...")
    print(f"   (Normalizing sequences by removing J/Z markers)")
    print(f"   (Removing duplicate sequences)")
    non_apr_sequences, duplicate_count = subtract_sequences_unique(full_sequences, apr_sequences)
    
    # Write output and get actual written count
    print(f"\n💾 Writing output to {OUTPUT_FASTA.name}...")
    written_count = write_fasta(non_apr_sequences, OUTPUT_FASTA)
    print(f"   ✓ Output saved successfully!")
    
    # Calculate statistics
    excluded_count = len(full_sequences) - len(non_apr_sequences) - duplicate_count
    total_removed = len(full_sequences) - len(non_apr_sequences)
    
    print(f"\n📊 Statistics:")
    print(f"   Total sequences in {FULL_FASTA.name}: {len(full_sequences)}")
    print(f"   APR sequences in {APR_FASTA.name}: {len(apr_sequences)}")
    print(f"   ─────────────────────────────────────")
    print(f"   Sequences excluded (APR): {excluded_count}")
    print(f"   Duplicate sequences removed: {duplicate_count}")
    print(f"   Total sequences removed: {total_removed}")
    print(f"   ─────────────────────────────────────")
    print(f"   Unique non-APR sequences: {len(non_apr_sequences)}")
    print(f"   Sequences written to file: {written_count}")
    
    if len(full_sequences) > 0:
        print(f"\n   Percentage removed (total): {total_removed/len(full_sequences)*100:.2f}%")
        print(f"   Percentage retained (unique): {len(non_apr_sequences)/len(full_sequences)*100:.2f}%")
    
    # Verify counts match
    if len(non_apr_sequences) != written_count:
        print(f"\n⚠️  WARNING: Count mismatch!")
        print(f"   In-memory: {len(non_apr_sequences)}")
        print(f"   File: {written_count}")
    else:
        print(f"\n✅ Verification: Counts match ({written_count} unique sequences)")
    
    # Show sample sequences
    print(f"\n📋 Sample of unique non-APR sequences (first 5):")
    for i, (header, seq) in enumerate(list(non_apr_sequences.items())[:5]):
        print(f"   {i+1}. {header[:50]}{'...' if len(header) > 50 else ''}")
        print(f"      {seq[:60]}{'...' if len(seq) > 60 else ''}")
    
    print("\n" + "="*70)
    print("✅ SUBTRACTION COMPLETE!")
    print("="*70)
    print(f"\nOutput file: {OUTPUT_FASTA}")
    print(f"Final count: {written_count} unique sequences (J/Z removed)")


if __name__ == "__main__":
    main()