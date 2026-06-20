"""
Novel optimal mixing (LT-GOMEA) over the Ward linkage tree.

Everything specific to the optimal-mixing operator lives here:

    mask_to_gene_indices — flatten a TLS-ID mixing mask (a Ward cluster from
                           ``linkage_tree.build_all_tree_masks``) into the flat
                           gene indices the operator copies.
    mix                  — optimal mixing for one individual: copy a donor's
                           genes at the mask positions and keep the child only
                           if its fitness improves.

The Ward tree these masks come from is built in ``src.novel.linkage_tree``.
"""


def mask_to_gene_indices(
    tls_mask: list[str],
    tls_to_genes: dict[str, tuple[int, int]],
) -> list[int]:
    """Flatten a list of TLS IDs into a flat list of gene indices."""
    out: list[int] = []
    for tls_id in tls_mask:
        if tls_id in tls_to_genes:
            s, e = tls_to_genes[tls_id]
            out.extend(range(s, e))
    return out


def mix(args):
    """
    Optimal mixing for one individual.

    Copies gene values from *donor* into *src* at the positions given by
    *mask* and accepts the child only if fitness improves.
    """
    wrapper, src, src_fit, donor, mask = args
    child = src.copy()
    for gi in mask:
        child[gi] = donor[gi]
    child_fit = float(wrapper(child))
    if child_fit < src_fit:
        return child, child_fit, True
    return src, src_fit, False
