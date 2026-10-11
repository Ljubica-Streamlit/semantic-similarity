"""Group similar pages into topics.

Complete linkage: a page only joins a group if it is at least `threshold`
similar to EVERY page already in it. Plain "connect everything above the
threshold" chains A~B~C into one group even when A and C are unrelated – on a
real 218-page site that fused 31 pages into one group at 88%.

A page can be close to a page in another group without fitting that whole
group. Rather than moving pages around (tested on that site: it barely changed
anything), the sheet says so in "Also Close To Group".

Only the embedding scores are used – no URL or folder rules.
"""
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage

from utils.similarity import categorize_similarity

DECISIONS = ['Keep', 'Merge', 'Redirect', 'Noindex', 'Rewrite', 'Leave as is']


def group_pages(urls, vectors, threshold):
    """Return ({url: group_no}, {group_no: {'weakest': x, 'best': y}}) for groups of 2+ pages.

    Groups are numbered 1..k by their strongest pair (likely duplicates first),
    then by their weakest link.
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
        inner = sims[np.ix_(idx, idx)][np.triu_indices(len(idx), 1)]
        clusters.append((idx, float(inner.min()), float(inner.max())))
    # Fixed tie-break on the first URL keeps numbering stable between runs
    clusters.sort(key=lambda c: (-c[2], -c[1], urls[c[0][0]]))

    membership, stats = {}, {}
    for number, (idx, weakest, best) in enumerate(clusters, start=1):
        stats[number] = {'weakest': weakest, 'best': best}
        for i in idx:
            membership[urls[i]] = number
    return membership, stats


def groups_frame(pairs, membership, stats, unique_pct=None, notes=None, categories=None):
    """One row per page that has at least one similar pair, sorted by group.

    `pairs` has URL 1, URL 2 and Semantic Similarity (pairs at or above the threshold only).
    Pages that match something but don't fit a whole group come last, with an empty Group.
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
        info = stats.get(group, {})
        match_group = membership.get(match)
        row = {
            'Group': group,
            'Pages in Group': sizes.get(group),
            'Group Cohesion': categorize_similarity(info['weakest']) if info else None,
            'Best Pair': info.get('best', np.nan),
            'Weakest Link': info.get('weakest', np.nan),
            'URL': page,
        }
        if categories is not None:
            row['Category'] = categories.get(page)
        row.update({
            'Closest Match': match,
            'Closest Match Score': score,
            'Also Close To Group': match_group if match_group is not None and match_group != group else None,
            'Similar Pages': counts[page],
        })
        if unique_pct is not None:
            row['Unique Content %'] = unique_pct.get(page)
        if notes is not None:
            row['Page Note'] = notes.get(page)
        row.update({'Decision': '', 'Notes': ''})
        rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    # Whole numbers stay whole (and empty) in Excel and CSV instead of turning into 8.0
    for col in ('Group', 'Pages in Group', 'Also Close To Group', 'Similar Pages'):
        out[col] = out[col].astype('Int64')
    # Groups in number order, ungrouped pages last; inside a group the most connected page first
    out = out.sort_values(['Group', 'Similar Pages', 'Closest Match Score', 'URL'],
                          ascending=[True, False, False, True], na_position='last')
    return out.reset_index(drop=True)


def pair_groups(pairs, membership):
    """Group number for each pair when both pages are in the same group, else empty."""
    values = []
    for u1, u2 in zip(pairs['URL 1'], pairs['URL 2']):
        g1, g2 = membership.get(u1), membership.get(u2)
        values.append(g1 if g1 is not None and g1 == g2 else None)
    return pd.array(values, dtype='Int64')
