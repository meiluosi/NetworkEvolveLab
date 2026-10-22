# `data/raw/` — raw source files and their provenance

This directory holds (or is meant to hold) the **unmodified source archives** for the
real empirical datasets used by NetworkEvolveLab, together with their provenance
records. The large source files themselves are **not committed** to git; only the
provenance notes and derived caches are tracked.

---

## SNAP "Higgs Twitter" social network

| field | value |
| --- | --- |
| Dataset page | <https://snap.stanford.edu/data/higgs-twitter.html> |
| Direct download URL | <https://snap.stanford.edu/data/higgs-social_network.edgelist.gz> |
| Local filename | `higgs-social_network.edgelist.gz` |
| Expected path in this repo | `data/raw/higgs-social_network.edgelist.gz` |
| Size | 54,597,265 bytes (52.1 MiB) |
| SHA256 | `8e5ea7642a018d9aab2dba0c6a59832c278cecf3895661ec8ff0d65360d40265` |
| SHA1 | `d8264500404a014ee7ae6833bf70a6c33917c82d` |
| MD5 | `e0a979365b6cde35da892f74d663b9a6` |
| Content | 456,626 nodes / 14,855,842 directed edges (friends & followers) |
| Format | gzip-compressed edge list, one `u v` pair of integer node ids per line |
| Last verified | 2026-10-21 (this repository) |

**Integrity check**

```bash
shasum -a 256 data/raw/higgs-social_network.edgelist.gz
# -> 8e5ea7642a018d9aab2dba0c6a59832c278cecf3895661ec8ff0d65360d40265
```

The node/edge counts streamed from the file (456,626 / 14,855,842) match the
statistics published on the SNAP dataset page exactly, which confirms the file is
the social-network edge list (not the retweet / reply / mention variants).

**License & terms.** SNAP distributes this dataset free of charge for research use; the
dataset page requests citation of the source publication rather than publishing a
separate license file. The underlying Twitter user ids were anonymized by the SNAP
authors, and the snapshot covers the "Higgs boson" announcement period (1–7 July 2012).
Please cite:

> M. De Domenico, A. Lima, P. Mougel and M. Musolesi.
> *The Anatomy of a Scientific Rumor.* Scientific Reports **3**, 2980 (2013).
> <https://www.nature.com/articles/srep02980>

**How to obtain it**

```bash
curl -L -o data/raw/higgs-social_network.edgelist.gz \
  https://snap.stanford.edu/data/higgs-social_network.edgelist.gz
shasum -a 256 data/raw/higgs-social_network.edgelist.gz
```

Alternatively point the environment variable at an existing copy (checked first, before
any repo-relative path):

```bash
export NETWORK_EVOLVE_HIGGS_RAW=/absolute/path/to/higgs-social_network.edgelist.gz
```

> **Do not commit the 52 MiB archive.** It is a third-party redistributable dataset and
> keeping it out of git keeps the repository light. (Suggestion for the maintainer:
> add `data/raw/*.gz` to `.gitignore` — this file intentionally does not touch
> `.gitignore`, which is outside the ownership of this change.)

---

## Derivation recipe — real Higgs benchmark subgraph

Implemented in `src/network_evolve/datasets_real.py`
(`load_higgs_real_subgraph(num_nodes=60, seed=42)`). Fully deterministic:

1. **Stream** the gzip edge list in 8 MiB chunks (numpy text-mode parse, with a
   line-wise fallback that skips malformed lines). The 14.8 M-edge graph is never
   materialised.
2. **Pass 1** accumulates the *total degree* (in + out) of every node id.
3. **Node selection**: the `num_nodes` highest-degree nodes; ties are broken by the
   **smaller node id**. For `num_nodes=60` the top hubs are ids
   `1503, 206, 88, 138, 1062, 677, 352, 220, 317, 301, …` (top-20 by degree).
4. **Pass 2** keeps only edges whose **both** endpoints are in that set →
   the *induced subgraph* (450 edges for n=60).
5. **Re-indexing**: the selected ids are sorted ascending and renamed `0..n-1`.
6. **Edge weight**: constant `0.6` for every retained edge. The SNAP social edge list
   is unweighted (direction only, no timestamps or interaction counts), so a fixed
   documented constant is used instead of inventing an activity share; the degree
   information is preserved by the topology itself.
7. **Camps**: Louvain (`networkx.community.louvain_communities`, seeded) on the
   *undirected* induced subgraph. The two largest communities become camp `-1`
   (ideology base `-0.7`) and camp `+1` (base `+0.7`); every remaining node is
   cross-cutting (base `0.0`). A seeded Gaussian jitter (`std = 0.12`) is added and the
   values are clipped to `[-1, 1]`.
8. **Cache**: written to `data/benchmarks/higgs_real_n{num_nodes}_s{seed}.json` in the
   project's standard loader format
   (`{"edges": [[u, v, w], …], "ideology": {"0": …}, "metadata": {…}}`) and re-read on
   subsequent calls, so the benchmark reproduces **without** the 52 MiB raw file.

### ⚠️ Interpretation caveat (must be respected in the paper)

The SNAP Higgs file carries **no ideology, political or stance labels**. The
`ideology` prior produced by this recipe is a **structural proxy derived from the real
topology** (Louvain communities of the induced subgraph), *not* a human annotation and
*not* a measurement of real political belief. It must be described in the paper as
"topology-derived camps used as a controlled initial-condition generator", and never as
"ground-truth ideology" or "human-labelled stance".

---

## Political Blogosphere (PolBlogs), Feb 2005 — human-labeled stance graph

| field | value |
| --- | --- |
| Dataset page | <http://www-personal.umich.edu/~mejn/netdata/> |
| Direct download | <http://www-personal.umich.edu/~mejn/netdata/polblogs.zip> |
| Local file | `data/raw/polblogs.gml` (unzipped from `polblogs.zip`) |
| Size | 955,025 bytes |
| SHA256 (GML) | `bd83e3dd2397a57d2252cc212524c21fee6708663136898d4ace523224bdc632` |
| Content | 1,490 political blogs, directed blog-to-blog links, node `value` 0=left/1=right |
| Labels | Human/directory-derived political leaning from the original authors; **not** inferred from topology |
| License/terms | Academic use; cite the source paper below; do not redistribute without attribution |
| Citation | Lada A. Adamic and Natalie Glance. *The political blogosphere and the 2004 US Election.* Proc. WWW-2005 Workshop on the Weblogging Ecosystem. |

**Derived caches.** `src/network_evolve/datasets_stance.py` selects a seeded top-degree
subgraph, re-indexes nodes, and maps the binary `value` label to a scalar belief
(left = −0.7, right = +0.7, plus a seeded N(0, 0.05) jitter only for continuous
dynamics). Edge weight is the constant 0.6. Caches are written to
`data/benchmarks/polblogs_real_n{num_nodes}_s{seed}.json`; the raw GML is small enough
to be kept in the repository. The label is human/directory-derived and must be described
as such in the paper, unlike the SNAP Higgs topology-derived proxy.

**Build caches:**

```bash
python scripts/build_polblogs_caches.py --nodes 100 --seeds 0-9
```

---

## Other raw files

None yet. Add one section per source file, keeping the same structure:
source URL, page URL, size, SHA256, license/terms, citation, and the derivation recipe
that turns the raw archive into the cached benchmark artifacts under `data/benchmarks/`.
