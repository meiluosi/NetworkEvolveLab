"""Real SNAP "Higgs Twitter" social network -> reproducible empirical benchmark graph.

本模块从**真实**的 Stanford SNAP Higgs Twitter 社交网络（friends/followers 有向图，
14,855,842 条边 / 456,626 个节点）派生一个可用于基准评测的小规模真实拓扑，
并以与本项目其它 loader 完全一致的 JSON 缓存格式落盘，保证在没有 52 MiB 原始
文件的情况下依然可复现。

⚠️ 重要说明 / IMPORTANT CAVEAT —— 论文写作必读
    Higgs 数据集**不包含任何意识形态标注**（no ideology / political / stance labels）。
    本模块返回的 ``ideology`` 先验是**由真实拓扑结构推导**得到的
    （*topology-derived camps, NOT human annotations*）：无向化诱导子图上的
    Louvain 社区被当作两个"阵营"，其余节点为跨派节点。
    论文中必须明确表述为"结构派生的代理变量（proxy）"，不可声称是真实政治立场。

确定性派生流程 (deterministic derivation pipeline)
-------------------------------------------------
1. 流式扫描 gzip 边表（分块解析，内存安全），累计每个节点的**总度** (in + out)。
2. 取度最高的 ``num_nodes`` 个节点作为种子集合（并列时取 **id 较小** 者）。
3. 只保留**两端都在**种子集合中的边 → 诱导子图 (induced subgraph)。
4. 节点按**原始 id 升序**重编号为 ``0..num_nodes-1``。
5. 边权固定为 ``EDGE_WEIGHT = 0.6``（每条保留边相同，见下方说明）。
6. 在无向化诱导子图上运行 Louvain（``seed`` 固定）：
   最大的两个社区分别作为 camp ``-1``（基准 -0.7）与 camp ``+1``（基准 +0.7），
   其余节点为跨派节点（基准 0.0）。
7. 用 ``seed`` 驱动的 RNG 叠加 std=0.12 的抖动，并裁剪到 [-1, 1]。

边权说明 (edge weight)
----------------------
SNAP 的社交边表是**无权**的（仅给出方向），没有时间戳也没有互动强度。
因此本模块采用固定常数权重 ``0.6``：它落在本项目其它合成 loader 使用的
[0.2, 0.95] 区间中部，代表"一次典型但非极端的社交互动"。
保持常数而非按活跃度加权，是为了避免把"度"重复编码进权重，从而污染
后续的动力学比较；度数信息已经完整保留在拓扑结构本身中。

缓存 (cache)
------------
派生结果写入 ``<cache_dir>/higgs_real_n{num_nodes}_s{seed}.json``，格式与
``src/network_evolve/datasets.py`` 的其它 loader 一致::

    {"edges": [[u, v, w], ...], "ideology": {"0": 0.31, ...}, "metadata": {...}}

loader 会优先读取缓存，因此基准套件在**没有原始 52 MiB 文件**的机器上
依然可以复现（只需把本仓库的 ``data/benchmarks/`` 一起分发）。

原始数据 (raw source)
---------------------
* SNAP 页面: https://snap.stanford.edu/data/higgs-twitter.html
* 下载地址:   https://snap.stanford.edu/data/higgs-social_network.edgelist.gz
* SHA256:     8e5ea7642a018d9aab2dba0c6a59832c278cecf3895661ec8ff0d65360d40265
* 许可证:     SNAP datasets 免费学术使用（见 SNAP 页面 / terms of use）
详细出处、校验和与派生配方见 ``data/raw/README.md``。
"""

import gzip
import hashlib
import json
import os
from typing import Any, Dict, Iterator, List, Optional, Tuple

import networkx as nx
import numpy as np

# ---------------------------------------------------------------------------
# Constants / 常量
# ---------------------------------------------------------------------------

HIGGS_ENV_VAR = "NETWORK_EVOLVE_HIGGS_RAW"
HIGGS_FILENAME = "higgs-social_network.edgelist.gz"

HIGGS_SNAP_PAGE = "https://snap.stanford.edu/data/higgs-twitter.html"
HIGGS_SNAP_URL = "https://snap.stanford.edu/data/higgs-social_network.edgelist.gz"

# Checksums of the canonical SNAP file (verified locally with shasum/md5).
HIGGS_SOURCE_SHA256 = "8e5ea7642a018d9aab2dba0c6a59832c278cecf3895661ec8ff0d65360d40265"
HIGGS_SOURCE_SHA1 = "d8264500404a014ee7ae6833bf70a6c33917c82d"
HIGGS_SOURCE_MD5 = "e0a979365b6cde35da892f74d663b9a6"
HIGGS_SOURCE_BYTES = 54597265
HIGGS_SOURCE_EDGES = 14855842   # matches the statistics published on the SNAP page
HIGGS_SOURCE_NODES = 456626

RAW_DIR = "data/raw"
DEFAULT_CACHE_DIR = "data/benchmarks"
SHA256_CACHE_PATH = os.path.join(RAW_DIR, "higgs_raw.sha256")

EDGE_WEIGHT = 0.6          # constant weight assigned to every retained edge
CAMP_MAGNITUDE = 0.7       # |ideology| of the two topology-derived camps
JITTER_STD = 0.12          # std of the seeded jitter around the camp means
# 种子化节点抽样时，候选池相对目标节点数的倍数（池内均为高阶节点，保持幂律结构）
POOL_MULTIPLIER = 4.0

_CHUNK_SIZE = 8 << 20      # 8 MiB of decompressed text per streaming chunk


# ---------------------------------------------------------------------------
# Raw file discovery / 原始文件定位
# ---------------------------------------------------------------------------

def _env_raw_path() -> Optional[str]:
    """Absolute path from the ``NETWORK_EVOLVE_HIGGS_RAW`` env var, if set."""
    val = os.environ.get(HIGGS_ENV_VAR, "").strip()
    if not val:
        return None
    return os.path.abspath(os.path.expanduser(val))


def _static_candidates() -> List[str]:
    """Repo-relative and user-local candidate locations of the raw edge list."""
    return [
        os.path.join(RAW_DIR, HIGGS_FILENAME),
        os.path.join(RAW_DIR, "higgs-social_network.gz"),
        os.path.expanduser(os.path.join("~", "Downloads", HIGGS_FILENAME)),
    ]


#: Candidate local paths for the raw SNAP edge list (env override first).
RAW_HIGGS_CANDIDATES: List[str] = [
    p for p in ([_env_raw_path()] + _static_candidates()) if p
]


def higgs_raw_path() -> Optional[str]:
    """Locate the raw SNAP Higgs edge list.

    Returns the first existing candidate path, or ``None`` when the 52 MiB
    source file is not available on this machine.  The environment variable
    ``NETWORK_EVOLVE_HIGGS_RAW`` always takes precedence.
    """
    env = _env_raw_path()
    if env and os.path.isfile(env):
        return env
    for cand in RAW_HIGGS_CANDIDATES:
        if cand and os.path.isfile(cand):
            return cand
    return None


# ---------------------------------------------------------------------------
# Checksum helpers / 校验和
# ---------------------------------------------------------------------------

def _read_sha256_cache() -> Optional[str]:
    """Read the cached digest from ``data/raw/higgs_raw.sha256`` (if any)."""
    try:
        with open(SHA256_CACHE_PATH, "r", encoding="utf-8") as fh:
            token = fh.read().split()[0].strip().lower()
    except (OSError, IndexError):
        return None
    return token if len(token) == 64 and all(c in "0123456789abcdef" for c in token) else None


def _write_sha256_cache(path: str, digest: str) -> None:
    """Persist the digest in ``sha256sum``-compatible format."""
    try:
        os.makedirs(os.path.dirname(SHA256_CACHE_PATH) or ".", exist_ok=True)
        with open(SHA256_CACHE_PATH, "w", encoding="utf-8") as fh:
            fh.write(f"{digest}  {os.path.basename(path)}\n")
    except OSError:
        pass  # provenance cache is best-effort, never fatal


def higgs_source_sha256() -> Optional[str]:
    """SHA256 digest of the raw Higgs edge list, or ``None`` if the file is absent.

    The digest is cached in ``data/raw/higgs_raw.sha256`` (written in
    ``sha256sum`` format) so repeated calls do not re-hash the 52 MiB file.
    The cached value refers to the file as it was hashed; delete that cache
    file to force a re-hash (e.g. after replacing the source archive).
    """
    path = higgs_raw_path()
    if path is None:
        return None
    cached = _read_sha256_cache()
    if cached:
        return cached
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    hexdigest = digest.hexdigest()
    _write_sha256_cache(path, hexdigest)
    return hexdigest


# ---------------------------------------------------------------------------
# Streaming edge reader / 流式边表读取
# ---------------------------------------------------------------------------

def _parse_int_block(chunk: bytes) -> np.ndarray:
    """Parse whitespace separated integers out of one decompressed chunk.

    Fast path uses numpy's text mode; the fallback is a line-wise parser that
    silently skips malformed lines (never raises on a partially corrupt file).

    A token-count check guards against numpy silently truncating at a bad
    token (e.g. a partially downloaded / corrupted archive): the fast path is
    only accepted when it yields exactly two integers per complete line.
    """
    fast: Optional[np.ndarray] = None
    try:
        fast = np.fromstring(chunk.decode("ascii"), dtype=np.int64, sep=" ")
    except (AttributeError, ValueError, UnicodeDecodeError):
        fast = None
    if fast is not None and fast.size == 2 * chunk.count(b"\n"):
        return fast
    pairs: List[int] = []
    for line in chunk.split(b"\n"):
        parts = line.split()
        if len(parts) != 2:
            continue
        try:
            pairs.append(int(parts[0]))
            pairs.append(int(parts[1]))
        except ValueError:
            continue
    return np.asarray(pairs, dtype=np.int64)


def _iter_edge_blocks(path: str) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Stream ``(u, v)`` node-id blocks from a gzipped edge list.

    Memory-safe: at most ``_CHUNK_SIZE`` decompressed bytes are held at a time,
    so the 14.8 M-edge source file is never materialised.

    The edge list is a flat token stream (``u1 v1 u2 v2 ...``), so a chunk that
    ends on an odd token would shift the pairing of every following edge.  A
    single pending token is therefore carried across chunk boundaries.
    """
    with gzip.open(path, "rb") as fh:
        tail = b""
        pending: Optional[int] = None
        while True:
            block = fh.read(_CHUNK_SIZE)
            if not block:
                break
            data = tail + block
            cut = data.rfind(b"\n")
            if cut == -1:
                tail = data
                continue
            tail = data[cut + 1:]
            arr = _parse_int_block(data[:cut])
            if arr.size == 0:
                continue
            if pending is not None:
                arr = np.concatenate([np.asarray([pending], dtype=np.int64), arr])
                pending = None
            if arr.size % 2:
                pending = int(arr[-1])
                arr = arr[:-1]
            if arr.size >= 2:
                yield arr[0::2], arr[1::2]
        # flush the trailing partial line
        arr = _parse_int_block(tail) if tail.strip() else np.empty(0, dtype=np.int64)
        if pending is not None:
            arr = np.concatenate([np.asarray([pending], dtype=np.int64), arr])
            pending = None
        if arr.size >= 2:
            usable = arr[: arr.size - (arr.size % 2)]
            yield usable[0::2], usable[1::2]


def _merge_counts(
    node_ids: np.ndarray,
    degrees: np.ndarray,
    uniq: np.ndarray,
    counts: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Merge one chunk's per-node counts into the running totals."""
    merged = np.union1d(node_ids, uniq)
    out = np.zeros(merged.size, dtype=np.int64)
    if node_ids.size:
        out[np.searchsorted(merged, node_ids)] += degrees
    out[np.searchsorted(merged, uniq)] += counts
    return merged, out


def _stream_degrees(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """One streaming pass: total degree (in + out) of every node id.

    Every edge endpoint is counted (duplicate edges in the raw file would be
    counted with multiplicity; the canonical SNAP file has none).  The running
    totals are merged chunk by chunk, so the peak memory stays proportional to
    the number of distinct node ids (~456 k), not to the number of edges.

    Returns ``(node_ids, degrees)`` sorted by node id.
    """
    node_ids = np.empty(0, dtype=np.int64)
    degrees = np.empty(0, dtype=np.int64)
    for u, v in _iter_edge_blocks(path):
        uniq, counts = np.unique(np.concatenate([u, v]), return_counts=True)
        node_ids, degrees = _merge_counts(node_ids, degrees, uniq, counts)
    return node_ids, degrees


def _select_top_nodes(node_ids: np.ndarray, degrees: np.ndarray, num_nodes: int,
                      seed: Optional[int] = None,
                      pool_multiplier: float = POOL_MULTIPLIER) -> np.ndarray:
    """Select ``num_nodes`` high-degree nodes.

    seed=None  : 严格取度最高的 num_nodes 个节点（并列时取 id 较小者），完全确定。
    seed!=None : 先取度最高的 ``ceil(num_nodes * pool_multiplier)`` 个节点构成候选池，
                 再用该 seed 驱动的 RNG 无放回抽取 num_nodes 个。
                 这样每个种子得到**不同但统计同阶**的幂律子图，
                 避免「10 个种子共用同一张图」导致置信区间虚假收紧（伪重复）。
    """
    order = np.lexsort((node_ids, -degrees))  # primary: degree desc, secondary: id asc
    if seed is None:
        return np.sort(node_ids[order[:num_nodes]])
    pool_size = min(node_ids.size, int(np.ceil(num_nodes * pool_multiplier)))
    pool = node_ids[order[:pool_size]]
    rng = np.random.default_rng(seed)
    picked = rng.choice(pool, size=min(num_nodes, pool.size), replace=False)
    return np.sort(picked)


def _stream_induced_edges(path: str, selected: np.ndarray) -> List[Tuple[int, int]]:
    """Second streaming pass: keep only edges with both endpoints selected."""
    kept: List[Tuple[int, int]] = []
    seen = set()
    for u, v in _iter_edge_blocks(path):
        mask = np.isin(u, selected) & np.isin(v, selected)
        if not mask.any():
            continue
        for a, b in zip(u[mask].tolist(), v[mask].tolist()):
            if a == b:
                continue
            key = (a, b)
            if key in seen:
                continue
            seen.add(key)
            kept.append(key)
    return kept


# ---------------------------------------------------------------------------
# Graph / ideology construction
# ---------------------------------------------------------------------------

def _build_graph(selected: np.ndarray, raw_edges: List[Tuple[int, int]]) -> nx.DiGraph:
    """Build the re-indexed DiGraph (original ids ascending -> 0..n-1)."""
    ordered = sorted(int(x) for x in selected)
    index = {orig: i for i, orig in enumerate(ordered)}
    g = nx.DiGraph()
    g.add_nodes_from(range(len(ordered)))
    for u, v in raw_edges:
        g.add_edge(index[u], index[v], weight=EDGE_WEIGHT)
    return g


def _derive_ideology(g: nx.DiGraph, seed: int) -> Tuple[Dict[int, float], str]:
    """Topology-derived camps -> ideology prior.

    Louvain is run on the *undirected* version of the induced subgraph with a
    fixed seed; the two largest communities become camp -1 (-0.7) and camp +1
    (+0.7), all remaining nodes are cross-cutting (0.0).  A seeded jitter of
    std 0.12 is added on top.

    Returns ``(ideology, camp_mode)`` where ``camp_mode`` records how the two
    camps were obtained ("louvain_two_largest" or the documented fallback).
    """
    n = g.number_of_nodes()
    undirected = g.to_undirected()
    communities = nx.community.louvain_communities(undirected, seed=seed)
    # deterministic community ordering: size desc, then smallest node id
    communities = sorted((sorted(c) for c in communities), key=lambda c: (-len(c), c[0]))

    degrees = {node: deg for node, deg in g.degree()}
    camp = np.zeros(n, dtype=int)
    if len(communities) >= 2:
        camp[communities[0]] = -1
        camp[communities[1]] = 1
        mode = "louvain_two_largest"
    elif len(communities) == 1:
        # Fallback: the induced hub subgraph is too homogeneous for Louvain to
        # split it -> deterministic degree-ordered bisection instead.
        members = sorted(communities[0], key=lambda i: (-degrees[i], i))
        half = len(members) // 2
        camp[members[:half]] = -1
        camp[members[half:]] = 1
        mode = "fallback_degree_bisection"
    else:
        mode = "no_communities"

    base = camp.astype(float) * CAMP_MAGNITUDE
    rng = np.random.default_rng(seed)
    values = np.clip(base + rng.normal(0.0, JITTER_STD, size=n), -1.0, 1.0)
    return {i: float(values[i]) for i in range(n)}, mode


# ---------------------------------------------------------------------------
# Descriptive statistics / 描述性统计
# ---------------------------------------------------------------------------

def _power_law_alpha(degrees: np.ndarray) -> float:
    """Approximate power-law exponent from the log-log CCDF tail slope.

    Estimator (approximate, documented):
      1. restrict to the **top decile** of the degree sequence (>= 90th pct),
      2. build the complementary CDF ``P(K >= k)`` for the distinct degrees k,
      3. ordinary least squares of ``log P(K >= k)`` on ``log k``,
      4. ``alpha = -slope``.

    This is a crude, threshold-sensitive surrogate for the MLE of Clauset et
    al. (2009); it is only indicative and should not be quoted as a precise
    tail exponent.
    """
    degs = np.asarray(degrees, dtype=float)
    if degs.size < 3:
        return float("nan")
    threshold = float(np.quantile(degs, 0.9))
    ks = np.unique(degs[degs >= threshold])
    if ks.size < 3:
        return float("nan")
    x = np.log(ks)
    y = np.log(np.array([(degs >= k).mean() for k in ks]))
    slope = float(np.polyfit(x, y, 1)[0])
    return -slope


def higgs_real_stats(num_nodes: int = 60, seed: int = 42) -> Dict[str, Any]:
    """Descriptive statistics of the derived real Higgs subgraph.

    Keys: ``num_nodes``, ``num_edges``, ``density``, ``reciprocity``,
    ``reciprocity_pairwise`` (SNAP convention), ``max_in_degree``,
    ``mean_in_degree``, ``n_communities``, ``top2_community_frac``,
    ``power_law_alpha`` (+ ``power_law_alpha_indegree``) and a few extras.
    """
    g, ideology = load_higgs_real_subgraph(num_nodes=num_nodes, seed=seed)
    n = g.number_of_nodes()
    m = g.number_of_edges()

    in_deg = np.array([d for _, d in g.in_degree()], dtype=float)
    out_deg = np.array([d for _, d in g.out_degree()], dtype=float)
    total_deg = in_deg + out_deg

    communities = nx.community.louvain_communities(g.to_undirected(), seed=seed)
    sizes = sorted((len(c) for c in communities), reverse=True)
    top2_frac = float(sum(sizes[:2]) / n) if n else 0.0

    # SNAP-style reciprocity: mutual pairs / undirected links.
    undirected_edges = g.to_undirected().number_of_edges()
    mutual_pairs = (m - undirected_edges)
    pairwise_reciprocity = float(mutual_pairs / undirected_edges) if undirected_edges else 0.0

    camp_counts = {"neg": 0, "zero": 0, "pos": 0}
    for value in ideology.values():
        if value < -0.25:
            camp_counts["neg"] += 1
        elif value > 0.25:
            camp_counts["pos"] += 1
        else:
            camp_counts["zero"] += 1

    stats: Dict[str, Any] = {
        "num_nodes": n,
        "num_edges": m,
        "density": float(nx.density(g)) if n > 1 else 0.0,
        # nx.overall_reciprocity = 2 * (#mutual pairs) / (#directed edges)
        "reciprocity": float(nx.overall_reciprocity(g)) if m else 0.0,
        # SNAP convention: (#mutual pairs) / (#undirected links)
        "reciprocity_pairwise": pairwise_reciprocity,
        "max_in_degree": int(in_deg.max()) if n else 0,
        "mean_in_degree": float(in_deg.mean()) if n else 0.0,
        "max_out_degree": int(out_deg.max()) if n else 0,
        "mean_out_degree": float(out_deg.mean()) if n else 0.0,
        "mean_total_degree": float(total_deg.mean()) if n else 0.0,
        "n_communities": len(communities),
        "community_sizes": sizes,
        "top2_community_frac": top2_frac,
        "power_law_alpha": _power_law_alpha(total_deg),
        "power_law_alpha_indegree": _power_law_alpha(in_deg),
        "power_law_alpha_estimator": (
            "OLS slope of log P(K>=k) on log k over the top decile of the "
            "degree sequence (approximate; not the Clauset et al. MLE)"
        ),
        "ideology_mean": float(np.mean(list(ideology.values()))) if ideology else 0.0,
        "ideology_std": float(np.std(list(ideology.values()))) if ideology else 0.0,
        "camp_counts": camp_counts,
        "ideology_provenance": "topology-derived (Louvain camps), NOT human annotations",
    }
    return stats


# ---------------------------------------------------------------------------
# Public loader / 公开加载器
# ---------------------------------------------------------------------------

def _missing_raw_message(cache_path: str) -> str:
    return (
        "Raw SNAP 'Higgs Twitter' social network edge list not found, and no "
        f"cached subgraph exists at '{cache_path}'.\n"
        "Fix it in one of two ways:\n"
        f"  1. Download {HIGGS_SNAP_URL}\n"
        f"     and save it as {os.path.join(RAW_DIR, HIGGS_FILENAME)}\n"
        f"  2. Point the environment variable {HIGGS_ENV_VAR} at the file:\n"
        f"     export {HIGGS_ENV_VAR}=/absolute/path/to/{HIGGS_FILENAME}\n"
        "Searched locations: " + ", ".join(str(p) for p in RAW_HIGGS_CANDIDATES)
    )


def load_higgs_real_subgraph(
    num_nodes: int = 60,
    seed: int = 42,
    cache_dir: str = DEFAULT_CACHE_DIR,
) -> Tuple[nx.DiGraph, Dict[int, float]]:
    """Load the real SNAP Higgs induced subgraph + topology-derived ideology.

    Parameters
    ----------
    num_nodes:
        Number of highest-degree nodes kept (they define the induced subgraph).
    seed:
        Controls the Louvain community detection and the ideology jitter.
    cache_dir:
        Directory holding the ``higgs_real_n{num_nodes}_s{seed}.json`` cache.

    Returns
    -------
    (graph, ideology)
        ``graph`` is a ``networkx.DiGraph`` over nodes ``0..num_nodes-1`` with a
        constant edge weight of 0.6; ``ideology`` maps node id -> prior belief
        in [-1, 1] (camp -1 / cross-cutting / camp +1).

    The cache is read when present, so the benchmark reproduces without the
    52 MiB raw file.  The ideology prior is **derived from the real topology**,
    it is *not* a human annotation (the SNAP file carries no such labels).
    """
    cache_path = os.path.join(cache_dir, f"higgs_real_n{num_nodes}_s{seed}.json")
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        g = nx.DiGraph()
        g.add_nodes_from(range(num_nodes))
        for u, v, w in data["edges"]:
            g.add_edge(int(u), int(v), weight=float(w))
        ideology = {int(k): float(v) for k, v in data["ideology"].items()}
        return g, ideology

    raw_path = higgs_raw_path()
    if raw_path is None:
        raise FileNotFoundError(_missing_raw_message(cache_path))

    # Pass 1: degrees of every node in the full 14.8 M-edge network.
    node_ids, degrees = _stream_degrees(raw_path)
    if node_ids.size == 0:
        raise ValueError(f"No edges parsed from raw Higgs edge list: {raw_path}")
    selected = _select_top_nodes(node_ids, degrees, num_nodes, seed=seed)

    # Pass 2: edges induced on the selected seed set.
    raw_edges = _stream_induced_edges(raw_path, selected)

    g = _build_graph(selected, raw_edges)
    ideology, camp_mode = _derive_ideology(g, seed)

    edges_data = [(int(u), int(v), float(g[u][v]["weight"])) for u, v in g.edges()]
    payload = {
        "metadata": {
            "source": "SNAP Higgs Twitter social network (friends/followers)",
            "source_url": HIGGS_SNAP_URL,
            "source_sha256": higgs_source_sha256(),
            "selection": (f"seeded sample of {num_nodes} nodes from the top "
                          f"{int(np.ceil(num_nodes * POOL_MULTIPLIER))} by total degree "
                          f"(seed={seed}); pool preserves the scale-free hub structure"),
            "num_nodes": num_nodes,
            "seed": seed,
            "edge_weight": EDGE_WEIGHT,
            "camp_mode": camp_mode,
            "ideology_provenance": "topology-derived (Louvain camps), NOT human annotations",
        },
        "edges": edges_data,
        "ideology": ideology,
    }
    os.makedirs(cache_dir, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)
    return g, ideology


def build_higgs_real_caches(num_nodes: int = 60,
                            seeds: Optional[List[int]] = None,
                            cache_dir: str = DEFAULT_CACHE_DIR,
                            pool_multiplier: float = POOL_MULTIPLIER) -> Dict[int, str]:
    """一次扫描原始边表，批量派生并写入多个种子的真实 Higgs 子图缓存。

    为什么需要它：原始文件含 1485 万条边，逐种子调用 ``load_higgs_real_subgraph()``
    会把整个文件重复扫描 N 次（每次数分钟）。本函数把边表一次性读入内存
    （两条 int32 数组，约 120 MB），随后对每个种子做一次 O(E) 的向量化筛选，
    10 个种子的总耗时与单次扫描同量级。

    每个种子的节点集合为「按度取前 ``ceil(num_nodes * pool_multiplier)`` 个候选，
    再用该种子驱动的 RNG 无放回抽样」，因此不同种子得到不同但统计同阶的幂律子图。
    """
    if seeds is None:
        seeds = list(range(10))
    raw_path = higgs_raw_path()
    if raw_path is None:
        raise FileNotFoundError(_missing_raw_message(os.path.join(cache_dir, "higgs_real_*.json")))

    # —— 一次扫描：把全部边读入内存并累计度 ——
    us, vs = [], []
    for u, v in _iter_edge_blocks(raw_path):
        us.append(u)
        vs.append(v)
    U = np.concatenate(us).astype(np.int32, copy=False)
    V = np.concatenate(vs).astype(np.int32, copy=False)
    del us, vs
    max_id = int(max(U.max(), V.max()))
    deg = np.bincount(np.concatenate([U, V]), minlength=max_id + 1)
    node_ids = np.nonzero(deg)[0].astype(np.int64)

    lookup = np.zeros(max_id + 1, dtype=bool)
    written: Dict[int, str] = {}
    os.makedirs(cache_dir, exist_ok=True)

    for seed in seeds:
        selected = _select_top_nodes(node_ids, deg[node_ids], num_nodes, seed=seed,
                                     pool_multiplier=pool_multiplier)
        lookup[:] = False
        lookup[selected] = True
        keep = lookup[U] & lookup[V] & (U != V)
        eu, ev = U[keep], V[keep]

        ordered = sorted(int(x) for x in selected)
        index = {orig: i for i, orig in enumerate(ordered)}
        g = nx.DiGraph()
        g.add_nodes_from(range(len(ordered)))
        for a, b in zip(eu.tolist(), ev.tolist()):
            g.add_edge(index[a], index[b], weight=EDGE_WEIGHT)

        ideology, camp_mode = _derive_ideology(g, seed)
        edges_data = [(int(u), int(v), float(g[u][v]["weight"])) for u, v in g.edges()]
        payload = {
            "metadata": {
                "source": "SNAP Higgs Twitter social network (friends/followers)",
                "source_url": HIGGS_SNAP_URL,
                "source_sha256": higgs_source_sha256(),
                "selection": (f"seeded sample of {num_nodes} nodes from the top "
                              f"{int(np.ceil(num_nodes * pool_multiplier))} by total degree "
                              f"(seed={seed}); pool preserves the scale-free hub structure"),
                "num_nodes": num_nodes,
                "seed": seed,
                "edge_weight": EDGE_WEIGHT,
                "camp_mode": camp_mode,
                "ideology_provenance": "topology-derived (Louvain camps), NOT human annotations",
            },
            "edges": edges_data,
            "ideology": ideology,
        }
        cache_path = os.path.join(cache_dir, f"higgs_real_n{num_nodes}_s{seed}.json")
        with open(cache_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
        written[seed] = cache_path
    return written


__all__ = [
    "RAW_HIGGS_CANDIDATES",
    "HIGGS_ENV_VAR",
    "HIGGS_SNAP_URL",
    "HIGGS_SOURCE_SHA256",
    "higgs_raw_path",
    "higgs_source_sha256",
    "load_higgs_real_subgraph",
    "build_higgs_real_caches",
    "higgs_real_stats",
]
