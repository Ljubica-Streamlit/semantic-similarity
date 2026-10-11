"""Group similar pages into topics.

Complete linkage: a page only joins a group if it is at least `threshold`
similar to EVERY page already in it. Plain "connect everything above the
threshold" chains A~B~C into one group even when A and C are unrelated – on a
real 218-page site that fused 31 pages into one group at 88%.

Only the embedding scores are used – no URL or folder rules.
"""
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage

UNGROUPED = 'Ungrouped'
DECISIONS = ['Keep', 'Merge', 'Redirect', 'Noindex', 'Rewrite', 'Leave as is']


def group_pages(urls, vectors, threshold):
    """Return ({url: group_no}, {group_no: weakest_link}) for groups of 2+ pages.

    Groups are numbered 1..k, largest first, then tightest first.
    """
    if len(urls) < 2:
        return {}, {}
    matrix = np.vstack(vectors).astype(np.float32)  # float32 halves memory at 5,000 pages
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    matrix /= norms
    sims = np.clip(matrix @ matrix.T, -1.0, 1.0)

    dist = 1.0 - sims[np.triu_indices(len(urls), 1)].astype(np.float64)
    labels = fcluster(linkage(dist, method='complete'), t=1.0 - threshold, criterion='distance')

    clusters = []
    for label in np.unique(labels):
        idx = np.nonzero(labels == label)[0]
        if len(idx) < 2:
            continue
        sub = sims[np.ix_(idx, idx)]
        weakest = float(sub[np.triu_indices(len(idx), 1)].min())
        clusters.append((idx, weakest))
    clusters.sort(key=lambda c: (-len(c[0]), -c[1]))

    membership, weakest_link = {}, {}
    for number, (idx, weakest) in enumerate(clusters, start=1):
        weakest_link[number] = weakest
        for i in idx:
            membership[urls[i]] = number
    return membership, weakest_link


def groups_frame(pairs, membership, weakest_link, unique_pct=None, notes=None, categories=None):
    """One row per page that has at least one similar pair, sorted by group.

    `pairs` has URL 1, URL 2 and Semantic Similarity (pairs at or above the threshold only).
    """
    best, counts = {}, {}
    for u1, u2, score in zip(pairs['URL 1'], pairs['URL 2'], pairs['Semantic Similarity']):
        for page, other in ((u1, u2), (u2, u1)):
            counts[page] = counts.get(page, 0) + 1
            if page not in best or score > best[page][1]:
                best[page] = (other, score)

    sizes = pd.Series(membership).value_counts().to_dict() if membership else {}
    rows = []
    for page, (match, score) in best.items():
        group = membership.get(page)
        row = {
            'Group': group if group is not None else UNGROUPED,
            'Pages in Group': sizes.get(group, np.nan),
            'Weakest Link': weakest_link.get(group, np.nan),
            'URL': page,
        }
        if categories is not None:
            row['Category'] = categories.get(page)
        row.update({'Closest Match': match, 'Closest Match Score': score, 'Similar Pages': counts[page]})
        if unique_pct is not None:
            row['Unique Content %'] = unique_pct.get(page)
        if notes is not None:
            row['Page Note'] = notes.get(page)
        row.update({'Decision': '', 'Notes': ''})
        rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    # Groups in number order, ungrouped pages last; inside a group the most connected page first
    out['_order'] = out['Group'].map(lambda g: g if g != UNGROUPED else float('inf'))
    out = out.sort_values(['_order', 'Similar Pages', 'Closest Match Score'], ascending=[True, False, False])
    return out.drop(columns='_order').reset_index(drop=True)


def pair_group(u1, u2, membership):
    """Group number when both pages of a pair are in the same group, else ''."""
    g1, g2 = membership.get(u1), membership.get(u2)
    return g1 if g1 is not None and g1 == g2 else ''
