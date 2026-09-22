"""
Script to extract non-antibacterial amyloid sequences.
Creates a new FASTA file containing sequences from amys_all_unique.fasta that are NOT in amys_uniqueAI4AMP_processedtotrain.fasta.
Handles J/Z amidation markers by ignoring them during comparison.
Ensures only unique sequences in output (no duplicates).
Also saves normalized version of APR sequences (J/Z removed).
"""

from pathlib import Path
from typing import Dict, Set, Tuple

# ══════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════

BASE_DIR = Path("/home/anupkumar/workspace/amyAMP/data_master/amyloid")
FULL_FASTA = BASE_DIR / "amys_all_unique.fasta"
APR_FASTA = BASE_DIR / "amys_uniqueAI4AMP_processedtotrain.fasta"
OUTPUT_FASTA = BASE_DIR / "AMY_nonAntibacterial.fasta"
OUTPUT_APR_NORMALIZED = BASE_DIR / "amys_uniqueAI4AMP_normalized.fasta"  # New output for normalized APR

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


def normalize_sequences_dict(sequences: Dict[str, str], remove_duplicates: bool = True) -> Tuple[Dict[str, str], int]:
    """
    Normalize all sequences in a dictionary by removing J/Z markers.
    Optionally remove duplicates.
    
    Parameters
    ----------
    sequences : Dict[str, str]
        Dictionary mapping headers to sequences
    remove_duplicates : bool
        If True, removes duplicate sequences and keeps first occurrence
        
    Returns
    -------
    Tuple[Dict[str, str], int]
        (Normalized sequences, number of duplicates removed)
    """
    normalized_seqs = {}
    seen_sequences = set()
    duplicate_count = 0
    
    for header, seq in sequences.items():
        normalized = normalize_sequence(seq)
        
        if remove_duplicates:
            if normalized in seen_sequences:
                duplicate_count += 1
                continue
            seen_sequences.add(normalized)
        
        normalized_seqs[header] = normalized
    
    return normalized_seqs, duplicate_count


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
    Ensures only unique sequences in output (removes duplicates).
    
    Parameters
    ----------
    full_seqs : Dict[str, str]
        Full set of sequences (without J/Z markers)
    subtract_seqs : Dict[str, str]
        Sequences to remove (may contain J/Z markers)
        
    Returns
    -------
    Tuple[Dict[str, str], int]
        (Remaining unique sequences after subtraction, number of duplicates removed)
    """
    # Get set of normalized sequences to exclude
    exclude_sequences = get_normalized_sequence_set(subtract_seqs)
    
    print(f"\n🔍 Debug Info:")
    print(f"   Unique normalized sequences to exclude: {len(exclude_sequences)}")
    print(f"   Sample sequences to exclude (first 3):")
    for i, seq in enumerate(list(exclude_sequences)[:3]):
        print(f"   {i+1}. {seq[:60]}...")
    
    # Keep only sequences not in the exclusion set
    # Also track unique sequences to remove duplicates
    result = {}
    seen_sequences = set()
    duplicate_count = 0
    excluded_count = 0
    
    for header, seq in full_seqs.items():
        normalized = normalize_sequence(seq)
        
        # Check if sequence should be excluded
        if normalized in exclude_sequences:
            excluded_count += 1
            continue
        
        # Check if sequence is duplicate
        if normalized in seen_sequences:
            duplicate_count += 1
            continue
        
        # Keep this unique, non-excluded sequence
        result[header] = seq
        seen_sequences.add(normalized)
    
    print(f"\n   Sequences excluded (antibacterial): {excluded_count}")
    print(f"   Duplicate sequences removed: {duplicate_count}")
    
    return result, duplicate_count


# ══════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ══════════════════════════════════════════════════════════════════════

def main():
    """Main function to subtract FASTA files."""
    print("="*70)
    print("FASTA SUBTRACTION TOOL (AMY Non-Antibacterial)")
    print("="*70)
    print("\nNote: J/Z amidation markers will be removed")
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
    has_jz_full = any('J' in seq or 'Z' in seq for seq in full_sequences.values())
    if has_jz_full:
        print(f"   ⚠️  Warning: Found J/Z markers in {FULL_FASTA.name}")
    
    print(f"\n📖 Reading {APR_FASTA.name}...")
    apr_sequences = read_fasta(APR_FASTA)
    print(f"   ✓ Loaded {len(apr_sequences)} sequences")
    
    # Check for J/Z in APR sequences
    jz_count = sum(1 for seq in apr_sequences.values() if 'J' in seq or 'Z' in seq)
    if jz_count > 0:
        print(f"   ℹ️  Found {jz_count} sequences with J/Z amidation markers")
        
        # Show example
        for header, seq in list(apr_sequences.items())[:3]:
            if 'J' in seq or 'Z' in seq:
                print(f"   Example original: {seq[:50]}...")
                print(f"   Normalized: {normalize_sequence(seq)[:50]}...")
                break
    
    # Normalize APR sequences and save
    print(f"\n🔄 Normalizing APR sequences (removing J/Z)...")
    apr_normalized, apr_duplicates = normalize_sequences_dict(apr_sequences, remove_duplicates=True)
    
    print(f"\n💾 Writing normalized APR sequences to {OUTPUT_APR_NORMALIZED.name}...")
    apr_written_count = write_fasta(apr_normalized, OUTPUT_APR_NORMALIZED)
    print(f"   ✓ Saved {apr_written_count} normalized APR sequences")
    if apr_duplicates > 0:
        print(f"   ℹ️  Removed {apr_duplicates} duplicate APR sequences after normalization")
    
    # Perform subtraction with duplicate removal
    print(f"\n🔄 Subtracting antibacterial sequences from full dataset...")
    print(f"   (Ignoring J/Z markers during comparison)")
    print(f"   (Removing duplicate sequences)")
    non_antibacterial_sequences, duplicate_count = subtract_sequences_unique(full_sequences, apr_sequences)
    
    # Write output and get actual written count
    print(f"\n💾 Writing output to {OUTPUT_FASTA.name}...")
    written_count = write_fasta(non_antibacterial_sequences, OUTPUT_FASTA)
    print(f"   ✓ Output saved successfully!")
    
    # Calculate statistics
    removed_antibacterial = len(full_sequences) - len(non_antibacterial_sequences) - duplicate_count
    total_removed = len(full_sequences) - len(non_antibacterial_sequences)
    
    print(f"\n📊 Statistics:")
    print(f"   Total sequences in {FULL_FASTA.name}: {len(full_sequences)}")
    print(f"   Antibacterial sequences in {APR_FASTA.name}: {len(apr_sequences)}")
    print(f"   Normalized APR sequences (unique): {apr_written_count}")
    print(f"   ─────────────────────────────────────")
    print(f"   Sequences excluded (antibacterial): {removed_antibacterial}")
    print(f"   Duplicate sequences removed: {duplicate_count}")
    print(f"   Total sequences removed: {total_removed}")
    print(f"   ─────────────────────────────────────")
    print(f"   Unique non-antibacterial sequences: {len(non_antibacterial_sequences)}")
    print(f"   Sequences written to file: {written_count}")
    
    if len(full_sequences) > 0:
        print(f"\n   Percentage removed (total): {total_removed/len(full_sequences)*100:.2f}%")
        print(f"   Percentage retained (unique): {len(non_antibacterial_sequences)/len(full_sequences)*100:.2f}%")
    
    # Verify counts match
    if len(non_antibacterial_sequences) != written_count:
        print(f"\n⚠️  WARNING: Count mismatch!")
        print(f"   In-memory: {len(non_antibacterial_sequences)}")
        print(f"   File: {written_count}")
    else:
        print(f"\n✅ Verification: Counts match ({written_count} unique sequences)")
    
    # Show sample sequences
    print(f"\n📋 Sample of unique non-antibacterial AMY sequences (first 5):")
    for i, (header, seq) in enumerate(list(non_antibacterial_sequences.items())[:5]):
        print(f"   {i+1}. {header[:50]}{'...' if len(header) > 50 else ''}")
        print(f"      {seq[:60]}{'...' if len(seq) > 60 else ''}")
    
    print("\n" + "="*70)
    print("✅ SUBTRACTION COMPLETE!")
    print("="*70)
    print(f"\nOutput files:")
    print(f"  1. {OUTPUT_FASTA} ({written_count} sequences)")
    print(f"  2. {OUTPUT_APR_NORMALIZED} ({apr_written_count} sequences)")


if __name__ == "__main__":
    main()