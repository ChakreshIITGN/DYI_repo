**Prompt for coding agent:**

Write a single Python script in `#%%` cell-style (for Jupyter/VS Code interactive execution), well-commented, implementing the following biosecurity screening pipeline on one input DNA sequence. Prioritize getting each step to return a working result over completeness — this is a quick demo, not production code.

**Input sequence** (assign as a string variable at the top of the script):
```
ATGAATCAACATAATACCCAAATCAATAAATTTATCTTTTTTAGTAGCTTCAGAATCATCAGTACCACCCCAATTCAATTTATCAACAATAATAGGACCAGCAACAGCAATAGCAACATATTCATCATTAGAAGCAGGAGAATTCAAAACAATACCAACAGCTTTTTTATTATCAGCAGTATTCCAAGTTTTACAAACAGCACCATTAGAATTACCTTTTTCACAATCACCAGCATGCAAATTATCAATACCAGAATTTTTATGAGATCTAATATGAACCAAACCCAT
```

**Steps to implement, in order:**

1. **Nucleotide BLAST (blastn):** Use Biopython's `Bio.Blast.NCBIWWW.qblast` to run a live blastn search of the input sequence against the `nt` database. Parse the result with `Bio.Blast.NCBIXML` and print a summary table of the top hits (accession, description, %identity, query coverage, E-value).

2. **ORF-finding + translation:** Using `Bio.Seq`, find open reading frames across all 6 reading frames (3 forward, 3 reverse-complement) of the input sequence. Identify the longest ORF(s) (start codon to stop codon, no internal stops). Translate the longest ORF to a protein sequence and print it.

3. **Protein-level BLAST (blastx):** Run `qblast` with program="blastx" on the *original nucleotide* input sequence (blastx handles the 6-frame translation internally) against the `nr` database. Parse and print a summary table of the top hits (accession, protein description, %identity, coverage, E-value), same format as step 1.

4. **Network visualization (only if steps 1-3 complete quickly, e.g., under ~2 minutes total runtime):** Using `networkx`, build a simple graph where the query sequence is a central node, and each top BLAST hit (from steps 1 and 3 combined, top 5-10 hits by E-value) is a node connected to the query with edge weight = % identity. Draw the graph with `matplotlib` (node size or edge width scaled by % identity). If this step would meaningfully slow things down or hits are too few/sparse to be meaningful, skip it and print a one-line note explaining why, rather than forcing a network.

**Explicitly out of scope:** Do not implement structural similarity/protein-folding comparison — this is intentionally excluded from this demo.

**Output requirements:** Print clear, labeled output after each step so results are readable top-to-bottom without needing to inspect variables manually. Do not include any API keys or credentials in the script — Biopython's `qblast` only requires setting `Entrez.email`, which should be a placeholder string, not a real credential.

