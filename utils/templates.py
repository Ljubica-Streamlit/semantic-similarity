"""Group template twins: thin pages that match mainly through text they share.

Grouping uses the text LEFT AFTER site-wide blocks are removed. On thin pages
the site-wide nav and footer dominate the raw text, so two unrelated thin
pages look almost as alike as two of the same kind (on a real site: 75% vs
80% raw overlap). Once those blocks are gone, pages built from the same
template still share their template text and others share almost nothing
(on that site: 33–71% within a page type vs 4% across types).

Every page in a group shares at least TEMPLATE_SHARED_MIN of its remaining
text with every other page in it – the same complete linkage as topic groups.
Block text only NAMES a group: the block that is common in the group but rare
elsewhere, so banners captured on random pages don't name anything.

Only text and scores are used – no URL or folder rules.
"""
import math

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage

from utils import boilerplate

TEMPLATE_SHARED_MIN = 0.20  # overlap of remaining text; unrelated thin pages sit near 0.05
NAME_MAX_CHARS = 100
NO_SHARED_BLOCK = "No identical block on most pages – they share wording, not a fixed block"
_WS = boilerplate._WS


def page_blocks(text):
    """{normalised key: original text} for the blocks of one page."""
    return {boilerplate._key(b): b for b in boilerplate.split_blocks(text) if len(b) >= boilerplate.MIN_BLOCK_CHARS}


def group_twins(urls, shared, min_shared=TEMPLATE_SHARED_MIN):
    """Return {url: group_no} for groups of 2+ pages, largest first.

    shared: square 0–1 matrix of overlap between the pages' remaining text.
    """
    n = len(urls)
    if n < 2:
        return {}
    dist = np.clip(1.0 - shared, 0.0, 1.0)
    np.fill_diagonal(dist, 0.0)
    labels = fcluster(linkage(dist[np.triu_indices(n, 1)], method='complete'),
                      t=1.0 - min_shared, criterion='distance')

    clusters = [np.nonzero(labels == label)[0] for label in np.unique(labels)]
    clusters = [idx for idx in clusters if len(idx) >= 2]
    clusters.sort(key=lambda idx: (-len(idx), urls[idx[0]]))
    return {urls[i]: number for number, idx in enumerate(clusters, start=1) for i in idx}


def shared_text(group_blocks, site_counts, site_pages):
    """Name a group by its most distinctive block: common in the group, rare on the rest of the site.

    group_blocks: one page_blocks() dict per page in the group.
    site_counts: {key: number of pages on the whole site with that block}.
    Returns (text, pages_in_group_with_it); the block must be on at least half the group.
    """
    size = len(group_blocks)
    inside, original = {}, {}
    for blocks in group_blocks:
        for key, text in blocks.items():
            inside[key] = inside.get(key, 0) + 1
            original.setdefault(key, text)
    needed = max(2, math.ceil(size / 2))
    candidates = [k for k, c in inside.items() if c >= needed]
    if not candidates:
        return NO_SHARED_BLOCK, None
    others = max(site_pages - size, 1)

    def distinctive(k):
        return inside[k] / size - (site_counts.get(k, inside[k]) - inside[k]) / others

    key = max(candidates, key=lambda k: (distinctive(k), inside[k], len(k)))
    text = _WS.sub(' ', original[key]).strip()
    return (text[:NAME_MAX_CHARS - 1] + '…') if len(text) > NAME_MAX_CHARS else text, inside[key]


def templates_frame(twin_pairs, membership, names, unique_pct=None, notes=None, categories=None):
    """One row per page in a twin pair, sorted by template group; pages in no group come last.

    twin_pairs: records with URL 1, URL 2, Semantic Similarity, Full-page Overlap.
    names: {group_no: (shared text, found on)} from shared_text().
    """
    closest = {}
    for t in twin_pairs:
        for page, other in ((t['URL 1'], t['URL 2']), (t['URL 2'], t['URL 1'])):
            if page not in closest or t['Full-page Overlap'] > closest[page][1]:
                closest[page] = (other, t['Full-page Overlap'], t['Semantic Similarity'])
    sizes = pd.Series(membership).value_counts().to_dict() if membership else {}

    rows = []
    for page, (twin, overlap, semantic) in closest.items():
        group = membership.get(page)
        name, found_on = names.get(group, (None, None))
        row = {
            'Template Group': group,
            'Pages in Group': sizes.get(group),
            'Shared Text': name,
            'Shared Text Found On': found_on,
            'URL': page,
        }
        if categories is not None:
            row['Category'] = categories.get(page)
        row.update({'Closest Twin': twin, 'Full-page Overlap': overlap, 'Semantic Similarity': semantic})
        if unique_pct is not None:
            row['Unique Content %'] = unique_pct.get(page)
        if notes is not None:
            row['Page Note'] = notes.get(page)
        row.update({'Decision': '', 'Notes': ''})
        rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    for col in ('Template Group', 'Pages in Group', 'Shared Text Found On'):
        out[col] = out[col].astype('Int64')
    out = out.sort_values(['Template Group', 'Full-page Overlap', 'URL'], ascending=[True, False, True],
                          na_position='last')
    return out.reset_index(drop=True)
