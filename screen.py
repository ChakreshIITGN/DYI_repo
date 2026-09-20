# %% [markdown]
# Biosecurity screening demo pipeline
#
# Steps:
#   1. blastn against nt
#   2. 6-frame ORF finding + translation of the longest ORF
#   3. blastx against nr
#
# Network visualization is intentionally left out at this stage.

# %%
from Bio.Blast import NCBIWWW, NCBIXML
from Bio.Data import CodonTable
from Bio.Seq import Seq

# NCBI requires an email for qblast but does not validate it.
NCBIWWW.email = "placeholder@example.com"

# Input sequence to screen (kept as a single unbroken string, verbatim from
# the source spec, to avoid any transcription risk at line-wrap boundaries).
SEQUENCE = "ATGAATCAACATAATACCCAAATCAATAAATTTATCTTTTTTAGTAGCTTCAGAATCATCAGTACCACCCCAATTCAATTTATCAACAATAATAGGACCAGCAACAGCAATAGCAACATATTCATCATTAGAAGCAGGAGAATTCAAAACAATACCAACAGCTTTTTTATTATCAGCAGTATTCCAAGTTTTACAAACAGCACCATTAGAATTACCTTTTTCACAATCACCAGCATGCAAATTATCAATACCAGAATTTTTATGAGATCTAATATGAACCAAACCCAT"


# %%
def print_blast_hit_table(blast_record, query_length, top_n=10):
    """Print a summary table (accession, description, %identity, coverage, E-value)
    for the top hits of a parsed Bio.Blast.Record.Blast result."""
    header = f"{'accession':<15}{'%identity':>10}{'coverage':>10}{'e-value':>12}  description"
    print(header)
    print("-" * len(header))

    if not blast_record.alignments:
        print("(no hits returned)")
        return

    for alignment in blast_record.alignments[:top_n]:
        hsp = alignment.hsps[0]  # best-scoring HSP for this hit
        pct_identity = 100 * hsp.identities / hsp.align_length
        coverage = 100 * (hsp.query_end - hsp.query_start + 1) / query_length
        description = alignment.title.split("|")[-1].strip()[:60]
        print(
            f"{alignment.accession:<15}{pct_identity:>9.1f}%{coverage:>9.1f}%"
            f"{hsp.expect:>12.2g}  {description}"
        )


# %% [markdown]
# ## Step 1: blastn against nt

# %%
print("=" * 70)
print("STEP 1: blastn vs nt")
print("=" * 70)

blastn_handle = NCBIWWW.qblast("blastn", "nt", SEQUENCE)
blastn_record = NCBIXML.read(blastn_handle)
blastn_handle.close()

print_blast_hit_table(blastn_record, query_length=len(SEQUENCE))

# %% [markdown]
# ## Step 2: ORF finding + translation

# %%
# Standard genetic code (NCBI translation table 1) — the same table
# Bio.Seq.translate() uses by default. Start/stop codons are read from
# Biopython's own table rather than hardcoded, so this stays in sync with
# whatever table is actually used for translation.
STANDARD_TABLE = CodonTable.unambiguous_dna_by_id[1]
START_CODON = "ATG"  # canonical start; STANDARD_TABLE.start_codons also
                      # lists TTG/CTG as alternative bacterial starts, but
                      # ATG is what's used here for a straightforward demo.
STOP_CODONS = set(STANDARD_TABLE.stop_codons)


def find_orfs(sequence):
    """Find all ORFs (ATG ... stop, no internal stops) across all 6 reading
    frames, scanning codon-by-codon against the standard genetic code's
    start/stop codon lists. Returns a list of dicts with strand, frame,
    start/end (nt coords on the given strand), and the translated protein."""
    orfs = []
    strands = [("+", Seq(sequence)), ("-", Seq(sequence).reverse_complement())]

    for strand_label, strand_seq in strands:
        for frame in range(3):
            codons = [
                str(strand_seq[i : i + 3])
                for i in range(frame, len(strand_seq) - 2, 3)
            ]
            for start_idx, codon in enumerate(codons):
                if codon != START_CODON:
                    continue
                for stop_idx in range(start_idx + 1, len(codons)):
                    if codons[stop_idx] in STOP_CODONS:
                        orf_seq = Seq("".join(codons[start_idx:stop_idx]))
                        orfs.append(
                            {
                                "strand": strand_label,
                                "frame": frame,
                                "protein": str(orf_seq.translate()),
                                "nt_start": frame + start_idx * 3,
                                "nt_end": frame + stop_idx * 3 + 3,
                            }
                        )
                        break

    return orfs


print("=" * 70)
print("STEP 2: ORF finding + translation")
print("=" * 70)

orfs = find_orfs(SEQUENCE)
longest_orf = max(orfs, key=lambda o: len(o["protein"]))

print(f"Total candidate ORFs found: {len(orfs)}")
print(
    f"Longest ORF: strand {longest_orf['strand']}, frame {longest_orf['frame']}, "
    f"{len(longest_orf['protein'])} aa, "
    f"nt range [{longest_orf['nt_start']}:{longest_orf['nt_end']}]"
)
print(f"Protein sequence:\n{longest_orf['protein']}")

# %% [markdown]
# ## Step 3: blastx against nr

# %%
print("=" * 70)
print("STEP 3: blastx vs nr")
print("=" * 70)

blastx_handle = NCBIWWW.qblast("blastx", "nr", SEQUENCE)
blastx_record = NCBIXML.read(blastx_handle)
blastx_handle.close()

print_blast_hit_table(blastx_record, query_length=len(SEQUENCE))

print("=" * 70)
print("Done. (Network visualization step skipped for this run.)")
print("=" * 70)
