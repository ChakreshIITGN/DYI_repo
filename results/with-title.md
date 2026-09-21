# with-title

chunk=300w/50ov  k=5  title_prefix=True  max_per_doc=None  embed=BAAI/bge-small-en-v1.5

n = 100 queries

| slice | n | partial recall@k | ALL-evidence recall@k | accuracy | uniq docs in top-k | gold needed |
| --- | --- | --- | --- | --- | --- | --- |
| **overall** | 100 | 0.488 | 0.150 | 0.000 | 2.97 | 2.62 |
| comparison_query | 46 | 0.551 | 0.217 | 0.000 | 2.80 | 2.22 |
| inference_query | 33 | 0.384 | 0.061 | 0.000 | 3.18 | 3.36 |
| temporal_query | 21 | 0.516 | 0.143 | 0.000 | 3.00 | 2.33 |

latency p50 13 ms | p95 23 ms
avg prompt tokens 0
