"""
src/modules/steering
Activation-steering reliability diagnostics and alpha-sweep runner.

Public API
----------
reliability(contrast_vectors, pos, neg) -> dict
    Pure function — no model required.  Returns mean pairwise cosine,
    pos/neg centroid distance, within-class variance, and probe margin
    for a set of per-example contrast vectors.

run(cfg) -> list of row tuples
    Model-dependent alpha-sweep runner.  Writes steering_results.csv
    to cfg['paths']['analysis_dir'].
"""
