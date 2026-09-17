"""Duplicate detection: exact, then near-duplicate.

Both run *before* the corpus is split. A resume that appears in training and in
test inflates every metric, and it is the single most common way a text
classifier reports accuracy it does not have.

Clusters are collapsed to one canonical representative rather than deleted
blindly: the survivor keeps a ``duplicate_group_id`` and the rest are
quarantined with their cluster id, so the decision is auditable.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass(frozen=True, slots=True)
class DuplicateClusters:
    """Result of duplicate detection over a list of documents.

    Attributes:
        group_of: index -> cluster id. Every document belongs to exactly one.
        representative_of: cluster id -> index of the kept document.
        members: cluster id -> all member indices, sorted.
    """

    group_of: dict[int, str]
    representative_of: dict[str, int]
    members: dict[str, list[int]]

    @property
    def duplicate_indices(self) -> set[int]:
        """Indices that are *not* the representative of their cluster."""
        keep = set(self.representative_of.values())
        return {i for i in self.group_of if i not in keep}

    @property
    def cluster_count(self) -> int:
        """Number of clusters containing more than one document."""
        return sum(1 for m in self.members.values() if len(m) > 1)


class _UnionFind:
    """Disjoint-set forest used to merge transitive duplicate relations."""

    def __init__(self, size: int) -> None:
        self._parent = list(range(size))

    def find(self, node: int) -> int:
        while self._parent[node] != node:
            self._parent[node] = self._parent[self._parent[node]]
            node = self._parent[node]
        return node

    def union(self, left: int, right: int) -> None:
        root_l, root_r = self.find(left), self.find(right)
        if root_l != root_r:
            # Keep the lower index as the root so results are deterministic.
            if root_l < root_r:
                self._parent[root_r] = root_l
            else:
                self._parent[root_l] = root_r


def find_duplicates(
    texts: Sequence[str],
    hashes: Sequence[str],
    *,
    threshold: float = 0.95,
    block_size: int = 512,
) -> DuplicateClusters:
    """Cluster documents that are exact or near duplicates of one another.

    Exact duplicates are detected by content hash. Near duplicates are detected
    with character n-gram TF-IDF cosine similarity above ``threshold``.

    Similarity is computed in row blocks so memory stays bounded: a dense
    n-by-n matrix is fine at a few thousand documents and ruinous at a hundred
    thousand.

    Args:
        texts: Documents, already feature-normalised.
        hashes: Content hash per document, parallel to ``texts``.
        threshold: Cosine similarity at or above which two documents are
            considered near duplicates.
        block_size: Rows per similarity block.

    Returns:
        The resulting :class:`DuplicateClusters`.
    """
    count = len(texts)
    if count != len(hashes):
        msg = "texts and hashes must be the same length"
        raise ValueError(msg)

    union = _UnionFind(count)

    # -- tier 1: exact duplicates by content hash -------------------------
    by_hash: dict[str, int] = {}
    for index, digest in enumerate(hashes):
        first = by_hash.setdefault(digest, index)
        if first != index:
            union.union(first, index)

    # -- tier 2: near duplicates by character n-gram cosine ---------------
    matrix = None
    if count > 1:
        # min_df=2 keeps the vocabulary manageable on a real corpus, but on a
        # handful of unrelated documents it can prune every term. Fall back to
        # min_df=1, and if even that yields nothing, exact matching stands alone.
        for min_df in (2, 1):
            try:
                vectorizer = TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(5, 5),
                    min_df=min_df,
                    max_features=200_000,
                    dtype=np.float32,
                )
                matrix = sparse.csr_matrix(vectorizer.fit_transform(texts))
                break
            except ValueError:
                matrix = None

    if matrix is not None:
        for start in range(0, count, block_size):
            stop = min(start + block_size, count)
            block = matrix[start:stop] @ matrix.T
            block = np.asarray(block.todense())
            # Only look forward, so each pair is considered once.
            for row in range(stop - start):
                absolute = start + row
                similar = np.nonzero(block[row, absolute + 1 :] >= threshold)[0]
                for offset in similar:
                    union.union(absolute, absolute + 1 + int(offset))

    # -- materialise clusters deterministically ---------------------------
    members: dict[str, list[int]] = {}
    group_of: dict[int, str] = {}
    for index in range(count):
        cluster_id = f"dup_{union.find(index):06d}"
        group_of[index] = cluster_id
        members.setdefault(cluster_id, []).append(index)

    representative_of = {cid: min(idx) for cid, idx in members.items()}
    return DuplicateClusters(
        group_of=group_of,
        representative_of=representative_of,
        members={cid: sorted(idx) for cid, idx in members.items()},
    )
