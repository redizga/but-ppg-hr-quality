"""Data layer: download, preprocessing, record registry, subject-wise splits.

Epic E1 (Golikov) fills this in; E8 adds the nested low-data subsets. E0 only
fixes the registry column contract (see ``registry.RECORD_COLUMNS``) and the
split-file semantics that downstream code depends on.
"""
