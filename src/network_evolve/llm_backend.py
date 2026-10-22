"""统一 LLM 认知后端（Unified LLM Cognitive Backend）。

为什么要「统一」
----------------
复核发现（见 docs/ARS_FULL_CHAIN_AUDIT.md 的阻断项 ①②）：

  * `semantic.py` 的文件头把 ``llm`` 写成 ``VecWorld`` 的一个后端，但 ``VecWorld.react``
    里**根本没有 llm 分支**（走到末尾直接 ``raise ValueError``）；
  * 真实 LLM 通路其实在 ``phase2_llm.py`` 里，是一条**旁路**：它自己拼 prompt、
    自己解析、自己落地边操作，与主引擎 ``CoEvolutionEngine`` 完全没有共享代码；
  * 于是「LLM 参与认知」这件事只在一个 3 种子的小侧实验里成立，旗舰引擎
    （产生 H1/H2/H3/基准结果的��个）用的是**标量规则启发式**。

本模块把 LLM 参与认知与关系决策的实现收敛到**唯一一处**，三层职责清晰分离：

  1. **传输层** ``LLMClient``
     HTTP(/messages) 调用 + **内容寻址归档** + 预算硬上限 + **完整收据**
     （模型标识、temperature、input/output/thinking token 数、response id、stop reason）。
     收据直接补上了此前真实 LLM 臂缺失的那几个字段（审计 F-12）。
     归档同时充当缓存：同一批请求重跑时零 API 调用、零成本、逐位可复现。

  2. **决策层** ``LLMCognitivePolicy``
     构造 prompt（维度自适应的议题图例）、批量请求、严格解析 JSON 决策。
     解析失败按显式规则 fallback（action=keep）并计数，**绝不伪造决策**。

  3. **效应层** ``apply_decision``
     把决策落到信念与边上：keep / mute / unfollow / block → 权重更新或断边。

统一之后：
  * ``VecWorld(backend="llm", llm_policy=...)`` 用本模块；
  * ``CoEvolutionEngine(llm_policy=..., llm_event_budget=...)`` 用本模块（旗舰引擎
    因此可以真的挂上 LLM 认知层）；
  * ``phase2_llm`` 保留原有的 plan/apply/finish 离线归档工作流，但其**决策逻辑改为
    调用本模块**，消除旁路造成的语义分叉。

鉴权
----
默认从环境变量 ``NETWORK_EVOLVE_LLM_BASE_URL`` / ``NETWORK_EVOLVE_LLM_API_KEY`` 读取；
未设置时回退到本机 MCode 的配置（``~/.minimax/config.yaml`` 给 baseURL，
``~/.minimax/auth/prod/cn/mcode-public/auth.json`` 给 OAuth accessToken）。
密钥与令牌**只在本进程内使用，不写盘、不打印**。
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# 默认配置
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "MiniMax-M3.1-Flash"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 4096
DEFAULT_THINKING_BUDGET = 512
DEFAULT_TIMEOUT = 120
ARCHIVE_DIR = "data/llm_archive"

VALID_ACTIONS = ("keep", "mute", "unfollow", "block")

#: 项目默认的三维议题图例（与论文附录的 prompt 契约一致）
DEFAULT_TOPICS = [
    ("binding government regulation of large AI models", "oppose", "support"),
    ("trading user privacy for faster innovation", "support", "oppose"),
    ("open-sourcing frontier model weights", "support", "oppose"),
]

#: 标量引擎（d=1）使用的单维图例 —— 让同一套 prompt 契约服务两种认知几何
SCALAR_TOPICS = [
    ("the contested policy axis", "strongly against the incumbent policy direction",
     "strongly for the incumbent policy direction"),
]

#: 原型 → 发言风格
ARCHETYPE_STYLE = {
    "hub": "Announcing loudly: ",
    "bridger": "Trying to see both sides: ",
    "tail": "Just my two cents: ",
}


def split_topics(triples: Sequence[Any]) -> Tuple[List[str], List[Tuple[str, str]]]:
    """把 (topic, neg, pos) 三元组列表拆成 (topics, poles) —— 策略内部使用的形式。"""
    return [str(t[0]) for t in triples], [(str(t[1]), str(t[2])) for t in triples]


def verbalize(b: Sequence[float], topics: Sequence[str],
              poles: Sequence[Tuple[str, str]], first_person: bool = False) -> str:
    """把信念向量渲染成一句可读的话（供 LLM 阅读的「发言内容」）。"""
    def word(x: float) -> str:
        a = abs(x)
        return "strongly" if a > 0.6 else "moderately" if a > 0.3 else "weakly"

    parts = []
    for k, topic in enumerate(topics):
        neg, pos = poles[k]
        x = float(b[k])
        verb = pos if x > 0 else neg
        if not first_person:
            verb = verb + "s"
        if abs(x) <= 0.1:
            parts.append(f"{'am' if first_person else 'is'} neutral about {topic}")
        else:
            parts.append(f"{word(x)} {verb} {topic}")
    return "; ".join(parts)


def message_text(archetype: str, b: Sequence[float], topics: Sequence[Any],
                 poles: Sequence[Tuple[str, str]]) -> str:
    """生成发言文本（听者侧看到的「对方说了什么」）。"""
    style = ARCHETYPE_STYLE.get(archetype, "")
    return style + "I " + verbalize(b, topics, poles, first_person=True) + "."


def archetype_of(hub: set, bridger: set, i: int) -> str:
    """由原型集合判定智能体原型。"""
    return "hub" if i in hub else "bridger" if i in bridger else "tail"


SYSTEM_CONTRACT = """You are an autonomous agent in a social network simulation.
You hold a personal stance on each issue below, expressed as a number in [-1.0, 1.0]
(-1.0 = most negative pole, 0.0 = neutral, +1.0 = most positive pole).
A peer has just spoken to you. Decide (a) how your own stance shifts and
(b) what you do with the relationship.

Output STRICT JSON only, with no prose and no code fence:
{"decisions": [{"id": "<event id>", "new_belief": [ ...one number per issue... ],
"action": "keep" | "mute" | "unfollow" | "block", "reasoning": "<max 25 words>"}]}

Rules:
- Return exactly one decision per input event, in the same order, reusing each "id".
- "new_belief" must have the same length as the issue list of that event.
- Use "keep" when you accept the statement, "mute" when you want to see less of this
  speaker, "unfollow" when you end the follow, "block" when you also refuse any future
  contact.
- Shifts must be small (at most 0.5 in absolute value) and must be plausible given your
  archetype. A hub is stubborn, a bridger is tolerant, a long-tail responder is volatile.
- If the statement is close to your own stance, keep the tie; if it is far, consider
  muting, unfollowing or blocking.
"""


# ---------------------------------------------------------------------------
# 配置与鉴权
# ---------------------------------------------------------------------------
def _read_json(path: str) -> Optional[Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _mcode_base_url() -> Optional[str]:
    cfg_path = os.path.expanduser("~/.minimax/config.yaml")
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            text = f.read()
    except OSError:
        return None
    for line in text.splitlines():
        if "baseURL:" in line:
            return line.split("baseURL:", 1)[1].strip().strip("\"'")
    return None


def _mcode_token_record() -> Optional[Dict[str, Any]]:
    """读取本机 MCode OAuth 凭据记录（含 accessToken 与过期时间）。

    只在本机读取、只在进程内使用：凭据**绝不写盘、绝不打印**。
    """
    path = os.path.expanduser("~/.minimax/auth/prod/cn/mcode-public/auth.json")
    data = _read_json(path)
    if not isinstance(data, dict):
        return None
    for rec in (data.get("records") or {}).values():
        if isinstance(rec, dict) and rec.get("accessToken"):
            return rec
    return None


def _mcode_access_token() -> Optional[str]:
    rec = _mcode_token_record()
    return str(rec["accessToken"]) if rec else None


def _token_expiry_ms(rec: Optional[Dict[str, Any]]) -> Optional[int]:
    if not rec:
        return None
    exp = rec.get("expiresAtMs")
    try:
        return int(exp) if exp is not None else None
    except (TypeError, ValueError):
        return None


def resolve_endpoint(base_url: Optional[str] = None,
                     api_key: Optional[str] = None) -> Tuple[Optional[str], Optional[str]]:
    """返回 (base_url, api_key)。优先级：显式参数 > 环境变量 > 本机 MCode 配置。"""
    base = (base_url or os.environ.get("NETWORK_EVOLVE_LLM_BASE_URL")
            or _mcode_base_url())
    key = (api_key or os.environ.get("NETWORK_EVOLVE_LLM_API_KEY")
           or _mcode_access_token())
    if base and not base.startswith("http"):
        base = None
    return base, key


# ---------------------------------------------------------------------------
# 收据 / 归档
# ---------------------------------------------------------------------------
@dataclass
class LLMReceipt:
    """一次（或一批）LLM 调用的完整收据。"""
    model: str
    temperature: float
    max_tokens: int
    request_hash: str
    response_id: Optional[str] = None
    stop_reason: Optional[str] = None
    input_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0
    cached: bool = False
    truncated: bool = False
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    n_events: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "request_hash": self.request_hash,
            "response_id": self.response_id,
            "stop_reason": self.stop_reason,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "thinking_tokens": self.thinking_tokens,
            "cached": self.cached,
            "truncated": self.truncated,
            "created_at": self.created_at,
            "n_events": self.n_events,
        }


class LLMArchive:
    """内容寻址的响应归档，同时充当缓存。

    归档项落盘后，**同一批请求再跑一次不会产生任何 API 调用**，
    因此实验可零成本复现，也可以用 ``--check`` 核对论文数字是否来自真实调用。
    """

    def __init__(self, root: str = ARCHIVE_DIR):
        self.root = root

    def _path(self, model: str, request_hash: str) -> str:
        safe_model = "".join(ch for ch in model if ch.isalnum() or ch in "-_.")
        return os.path.join(self.root, safe_model, f"{request_hash}.json")

    def get(self, model: str, request_hash: str) -> Optional[Dict[str, Any]]:
        return _read_json(self._path(model, request_hash))

    def put(self, model: str, request_hash: str, payload: Dict[str, Any]) -> str:
        path = self._path(model, request_hash)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=1)
        return path

    def count(self, model: Optional[str] = None) -> int:
        if not os.path.isdir(self.root):
            return 0
        total = 0
        for dirpath, _dirnames, filenames in os.walk(self.root):
            if model and os.path.basename(dirpath) != model:
                continue
            total += sum(1 for fn in filenames if fn.endswith(".json"))
        return total


# ---------------------------------------------------------------------------
# 传输层
# ---------------------------------------------------------------------------
class LLMClient:
    """HTTP(/messages) 客户端：归档优先、预算硬上限、收据完整。"""

    def __init__(self, model: str = DEFAULT_MODEL,
                 temperature: float = DEFAULT_TEMPERATURE,
                 max_tokens: int = DEFAULT_MAX_TOKENS,
                 base_url: Optional[str] = None,
                 api_key: Optional[str] = None,
                 budget_calls: int = 1000,
                 timeout: int = DEFAULT_TIMEOUT,
                 archive: Optional[LLMArchive] = None,
                 thinking_budget: int = DEFAULT_THINKING_BUDGET,
                 api_style: Optional[str] = None):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.thinking_budget = int(thinking_budget)
        self.base_url, self.api_key = resolve_endpoint(base_url, api_key)
        style = (api_style or os.environ.get("NETWORK_EVOLVE_LLM_API_STYLE") or "").lower()
        if style not in ("anthropic", "openai"):
            style = ("openai" if any(x in (self.base_url or "").lower()
                                     for x in ("deepseek", "openai", "chat/completions"))
                     or "deepseek" in (self.model or "").lower()
                     else "anthropic")
        self.api_style = style
        self.budget_calls = budget_calls
        self.timeout = timeout
        self.archive = archive or LLMArchive()
        self.calls_made = 0
        self.cache_hits = 0
        self.n_credential_refreshes = 0
        self.seen_hashes: set = set()
        self.totals = {"input_tokens": 0, "output_tokens": 0, "thinking_tokens": 0}
        self.receipts: List[LLMReceipt] = []

    # ---- 统计 ----
    @property
    def n_calls(self) -> int:
        return self.calls_made

    def ledger(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "api_style": self.api_style,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "budget_calls": self.budget_calls,
            "api_calls": self.calls_made,
            "cache_hits": self.cache_hits,
            "credential_refreshes": self.n_credential_refreshes,
            "totals": dict(self.totals),
            "receipts": [r.to_dict() for r in self.receipts],
        }

    # ---- 请求哈希 ----
    @staticmethod
    def request_hash(system: str, prompt: str, model: str, temperature: float,
                     max_tokens: int, thinking_budget: int = 0,
                     api_style: str = "anthropic") -> str:
        # Keep the historical hash for the default Anthropic/MiniMax path so
        # that existing content-addressed receipts still replay at zero cost;
        # only the OpenAI-compatible path adds the style marker.
        parts = [model]
        if api_style != "anthropic":
            parts.append(api_style)
        parts += [f"{temperature}", str(max_tokens), str(thinking_budget), system, prompt]
        blob = "\x1f".join(parts)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    # ---- 凭据保鲜 ----
    def _refresh_credentials(self, wait: bool = False,
                             max_wait: float = 300.0,
                             poll: float = 10.0) -> bool:
        """凭据过期时重新从本机 MCode 凭据文件取 access token。

        本机 MCode 运行时会周期性地把刷新的 OAuth access token 写回
        ``~/.minimax/auth/prod/cn/mcode-public/auth.json``（TTL 约 1 小时）。
        长跑实验必然跨越这个边界，因此传输层必须能自愈，否则一个 401
        就会让整轮实验前功尽弃。

        * 凭据只在进程内使用，**绝不写盘、绝不打印**；
        * ``wait=True`` 时会等待磁盘上的 token 变成未过期（最多 ``max_wait`` 秒）；
        * 返回 True 表示拿到一个可用（未过期）的新 token。
        """
        deadline = time.time() + (max_wait if wait else 0.0)
        while True:
            rec = _mcode_token_record()
            token = str(rec["accessToken"]) if rec else None
            exp_ms = _token_expiry_ms(rec)
            fresh = exp_ms is None or exp_ms > time.time() * 1000 + 60_000
            if token and fresh:
                changed = (token != self.api_key)
                self.api_key = token
                if changed:
                    self.n_credential_refreshes += 1
                return True
            if time.time() >= deadline:
                return False
            time.sleep(poll)

    # ---- 主调用 ----
    def complete(self, system: str, prompt: str,
                 n_events: int = 0) -> Tuple[str, LLMReceipt]:
        """返回 (响应文本, 收据)。命中归档时不产生 API 调用。"""
        rh = self.request_hash(system, prompt, self.model, self.temperature,
                                self.max_tokens, self.thinking_budget, self.api_style)
        self.seen_hashes.add(rh)
        cached = self.archive.get(self.model, rh)
        if cached is not None:
            self.cache_hits += 1
            receipt = LLMReceipt(
                model=cached.get("model", self.model),
                temperature=cached.get("temperature", self.temperature),
                max_tokens=cached.get("max_tokens", self.max_tokens),
                request_hash=rh,
                response_id=cached.get("response_id"),
                stop_reason=cached.get("stop_reason"),
                input_tokens=int(cached.get("input_tokens", 0)),
                output_tokens=int(cached.get("output_tokens", 0)),
                thinking_tokens=int(cached.get("thinking_tokens", 0)),
                cached=True,
                truncated=bool(cached.get("truncated", False)),
                created_at=cached.get("created_at", ""),
                n_events=int(cached.get("n_events", n_events)),
            )
            self.totals["input_tokens"] += receipt.input_tokens
            self.totals["output_tokens"] += receipt.output_tokens
            self.totals["thinking_tokens"] += receipt.thinking_tokens
            self.receipts.append(receipt)
            return cached.get("response_text", ""), receipt

        if self.calls_made >= self.budget_calls:
            raise RuntimeError(
                f"LLM 预算耗尽：{self.calls_made}/{self.budget_calls} 次调用。"
                " 请提高 budget_calls 或缩小实验规模。")
        if not self.base_url or not self.api_key:
            raise RuntimeError(
                "未找到 LLM 端点或凭据：请设置 NETWORK_EVOLVE_LLM_BASE_URL / "
                "NETWORK_EVOLVE_LLM_API_KEY 环境变量。")
        # 发送前先做一次凭据保鲜：若磁盘上的 access token 已过期（本机 TTL ~1h），
        # 先等到运行时刷新出新 token，避免白白消耗一次 401。
        if self.api_style != "openai":
            self._refresh_credentials(wait=True, max_wait=120.0)

        if self.api_style == "openai":
            body = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }
            endpoint = self.base_url.rstrip("/") + "/chat/completions"
        else:
            body = {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "system": system,
                "messages": [{"role": "user", "content": prompt}],
            }
            # 该模型强制 adaptive thinking（thinking.type=disabled 会被拒），
            # 但思考预算可封顶，否则会吃满 max_tokens 导致 JSON 响应被截断。
            if self.thinking_budget > 0:
                body["thinking"] = {"type": "enabled", "budget_tokens": self.thinking_budget}
            endpoint = self.base_url.rstrip("/") + "/messages"
        payload_bytes = json.dumps(body).encode("utf-8")

        def _make_req() -> urllib.request.Request:
            return urllib.request.Request(
                endpoint,
                data=payload_bytes,
                headers={"Content-Type": "application/json",
                         "Authorization": f"Bearer {self.api_key}"},
                method="POST",
            )

        last_err: Optional[Exception] = None
        payload: Dict[str, Any] = {}
        # 传输层重试：网络抖动/握手超时是常态，用指数退避重试多次。
        # 每次重试都重新发送同一请求（请求哈希已定，内容寻址归档保证
        # 成功的调用会被缓存，重跑零成本、逐位可复现）。
        attempts = 6
        for attempt in range(attempts):
            try:
                with urllib.request.urlopen(_make_req(), timeout=self.timeout) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
                last_err = None
                break
            except urllib.error.HTTPError as exc:
                last_err = exc
                # 401/403 = 凭据过期。本机 MCode 运行时会在磁盘上刷新
                # ~/.minimax/auth/.../auth.json（OAuth access token TTL ~1h），
                # 因此这里**重新读盘取新 token 并立刻重试**，而不是当成硬失败。
                if exc.code in (401, 403):
                    if self._refresh_credentials(wait=True):
                        continue
                if attempt < attempts - 1:
                    time.sleep(min(30.0, 2.0 * (2 ** attempt)))
            except (urllib.error.URLError, TimeoutError,
                    json.JSONDecodeError, ConnectionError, OSError) as exc:
                last_err = exc
                if attempt < attempts - 1:
                    time.sleep(min(30.0, 2.0 * (2 ** attempt)))
        if last_err is not None:
            raise RuntimeError(f"LLM 调用失败（重试 {attempts} 次后仍失败）："
                               f"{type(last_err).__name__}: {last_err}")

        self.calls_made += 1
        if self.api_style == "openai":
            choice = (payload.get("choices") or [{}])[0]
            response_text = str((choice.get("message") or {}).get("content", "")).strip()
            usage = payload.get("usage", {}) or {}
            details = usage.get("completion_tokens_details", {}) or {}
            receipt = LLMReceipt(
                model=payload.get("model", self.model),
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                request_hash=rh,
                response_id=payload.get("id"),
                stop_reason=choice.get("finish_reason"),
                input_tokens=int(usage.get("prompt_tokens", 0)),
                output_tokens=int(usage.get("completion_tokens", 0)),
                thinking_tokens=int(details.get("reasoning_tokens", 0)),
                cached=False,
                truncated=(choice.get("finish_reason") == "length"),
                n_events=n_events,
            )
        else:
            text_parts = [c.get("text", "") for c in payload.get("content", [])
                          if isinstance(c, dict) and c.get("type") == "text"]
            response_text = "".join(text_parts).strip()
            usage = payload.get("usage", {}) or {}
            details = usage.get("output_tokens_details", {}) or {}
            receipt = LLMReceipt(
                model=payload.get("model", self.model),
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                request_hash=rh,
                response_id=payload.get("id"),
                stop_reason=payload.get("stop_reason"),
                input_tokens=int(usage.get("input_tokens", 0)),
                output_tokens=int(usage.get("output_tokens", 0)),
                thinking_tokens=int(details.get("thinking_tokens", 0)),
                cached=False,
                truncated=(payload.get("stop_reason") == "max_tokens"),
                n_events=n_events,
            )
        self.totals["input_tokens"] += receipt.input_tokens
        self.totals["output_tokens"] += receipt.output_tokens
        self.totals["thinking_tokens"] += receipt.thinking_tokens
        self.receipts.append(receipt)
        self.archive.put(self.model, rh, {
            "model": receipt.model,
            "temperature": receipt.temperature,
            "max_tokens": receipt.max_tokens,
            "request_hash": rh,
            "response_id": receipt.response_id,
            "stop_reason": receipt.stop_reason,
            "input_tokens": receipt.input_tokens,
            "output_tokens": receipt.output_tokens,
            "thinking_tokens": receipt.thinking_tokens,
            "truncated": receipt.truncated,
            "created_at": receipt.created_at,
            "n_events": n_events,
            "system": system,
            "prompt": prompt,
            "response_text": response_text,
        })
        return response_text, receipt


# ---------------------------------------------------------------------------
# 决策层
# ---------------------------------------------------------------------------
@dataclass
class LLMEvent:
    """一次「听者面对发言者」的认知事件。"""
    event_id: str
    listener: int
    speaker: int
    listener_archetype: str
    listener_belief: List[float]
    listener_profile: str
    edge_weight: float
    message: str
    tolerance: float = 0.5


@dataclass
class LLMDecision:
    event_id: str
    action: str
    new_belief: Optional[List[float]] = None
    reasoning: str = ""
    valid: bool = True

    @property
    def severs(self) -> bool:
        return self.action in ("unfollow", "block")


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """从模型输出中提取 JSON 对象（容忍前后多余文字与代码围栏）。"""
    if not text:
        return None
    t = text.strip()
    if t.startswith("```"):
        t = t.split("```", 2)[1] if t.count("```") >= 2 else t.strip("`")
        t = t.strip()
        if t.startswith("json"):
            t = t[4:].strip()
    start = t.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(t)):
        ch = t[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(t[start:i + 1])
                except ValueError:
                    return None
    return None


class LLMCognitivePolicy:
    """把「认知事件」翻译成 LLM 决策的统一策略。

    一个策略实例对应**一个认知几何**（维度数与议题图例），
    因此标量引擎（d=1）与多维 VecWorld（d=3）可以共用同一份实现。
    """

    def __init__(self, client: LLMClient,
                 topics: Sequence[str],
                 poles: Sequence[Tuple[str, str]],
                 system: str = SYSTEM_CONTRACT,
                 arch_tolerance: Optional[Dict[str, float]] = None,
                 round_budget: int = 0):
        if len(topics) != len(poles):
            raise ValueError("topics 与 poles 长度必须一致")
        self.client = client
        self.topics = list(topics)
        self.poles = [tuple(p) for p in poles]
        self.system = system
        self.arch_tolerance = arch_tolerance or {}
        #: 每「轮」允许的 LLM 认知事件总数（0 = 不限，由调用方自行控制）
        self.round_budget = int(round_budget)

    # ---- prompt 构造 ----
    def legend(self) -> List[Dict[str, str]]:
        return [{"issue": t, "negative_pole": p[0], "positive_pole": p[1]}
                for t, p in zip(self.topics, self.poles)]

    def build_prompt(self, events: Sequence[LLMEvent]) -> str:
        payload = {
            "issues": self.legend(),
            "archetypes": {
                "hub": "high-reach agenda setter, stubborn, amplifies its own position",
                "bridger": "spans opposing clusters, tolerant, keeps weak ties alive",
                "tail": "low-reach responder, volatile, easily swayed and easily lost",
            },
            "events": [
                {
                    "id": e.event_id,
                    "your_archetype": e.listener_archetype,
                    "your_stance": [round(float(x), 3) for x in e.listener_belief],
                    "your_profile": e.listener_profile,
                    "your_tolerance": round(float(e.tolerance), 3),
                    "tie_strength": round(float(e.edge_weight), 3),
                    "peer_says": e.message,
                }
                for e in events
            ],
        }
        return ("Decide for every event in the JSON below. "
                "Reply with the strict JSON object described in your instructions.\n\n"
                + json.dumps(payload, ensure_ascii=False))

    # ---- 决策 ----
    def decide(self, events: Sequence[LLMEvent]) -> Tuple[List[LLMDecision], Dict[str, Any]]:
        """返回 (决策列表, 本批统计)。解析失败的条目标记为 invalid 并 fallback 为 keep。"""
        if not events:
            return [], {"n_events": 0, "n_valid": 0, "n_invalid": 0, "receipt": None}
        prompt = self.build_prompt(events)
        text, receipt = self.client.complete(self.system, prompt, n_events=len(events))
        obj = _extract_json(text) or {}
        raw_list = obj.get("decisions") if isinstance(obj, dict) else None
        if not isinstance(raw_list, list):
            raw_list = []
        by_id = {}
        for item in raw_list:
            if isinstance(item, dict) and "id" in item:
                by_id[str(item["id"])] = item

        decisions: List[LLMDecision] = []
        n_valid = 0
        for ev in events:
            item = by_id.get(ev.event_id)
            dec = self._parse_one(ev, item)
            if dec.valid:
                n_valid += 1
            decisions.append(dec)
        stats = {
            "n_events": len(events),
            "n_valid": n_valid,
            "n_invalid": len(events) - n_valid,
            "receipt": receipt.to_dict(),
        }
        return decisions, stats

    def _parse_one(self, ev: LLMEvent, item: Optional[Dict[str, Any]]) -> LLMDecision:
        if not isinstance(item, dict):
            return LLMDecision(ev.event_id, "keep", None, "missing/invalid response", False)
        action = str(item.get("action", "")).strip().lower()
        if action not in VALID_ACTIONS:
            return LLMDecision(ev.event_id, "keep", None,
                               f"invalid action {action!r}", False)
        belief = item.get("new_belief")
        if not isinstance(belief, list) or len(belief) != len(ev.listener_belief):
            return LLMDecision(ev.event_id, action, None,
                               "invalid new_belief", False)
        try:
            vals = [float(x) for x in belief]
        except (TypeError, ValueError):
            return LLMDecision(ev.event_id, action, None, "non-numeric belief", False)
        vals = [max(-1.0, min(1.0, v)) for v in vals]
        return LLMDecision(ev.event_id, action, vals,
                           str(item.get("reasoning", ""))[:200], True)


# ---------------------------------------------------------------------------
# 效应层
# ---------------------------------------------------------------------------
#: LLM 动作 → 边权重增量（与既有 rule 后端的语义保持一致）
ACTION_WEIGHT_DELTA = {"keep": 0.05, "mute": -0.15, "unfollow": None, "block": None}


def apply_decision(world, listener: int, speaker: int,
                   decision: LLMDecision, eta: float = 0.2) -> Dict[str, Any]:
    """把一个 LLM 决策落到世界状态上（信念 + 边）。

    与 ``phase2_llm.apply_round`` 原有语义**逐条一致**：
      · 信念：阻尼更新，位移被裁剪到 ±0.5；
      · keep  → 权重 +0.05；mute → 权重 −0.15（≤0 则断边）；
      · unfollow / block → 立即断边并写入黑名单（禁止被三元闭包重连）。
    """
    res = {"action": decision.action, "dropped": False, "weight_delta": 0.0}
    b_u = world.b[listener]
    if decision.valid and decision.new_belief is not None:
        delta = np_clip([float(x) - float(b_u[k]) for k, x in enumerate(decision.new_belief)],
                        -0.5, 0.5)
        world.b[listener] = np_clip([b_u[k] + eta * delta[k]
                                     for k in range(len(b_u))], -1.0, 1.0)
    if not world.g.has_edge(listener, speaker):
        return res
    if decision.action == "keep":
        dw = 0.05
    elif decision.action == "mute":
        dw = -0.15
    else:  # unfollow / block
        world._drop_edge(listener, speaker)
        res["dropped"] = True
        res["weight_delta"] = None
        return res
    w = world.g[listener][speaker]["weight"] + dw
    if w <= 0:
        world._drop_edge(listener, speaker)
        res["dropped"] = True
    else:
        world.g[listener][speaker]["weight"] = min(1.0, w)
    res["weight_delta"] = dw
    return res


def np_clip(vals: Sequence[float], lo: float, hi: float) -> List[float]:
    return [max(lo, min(hi, float(v))) for v in vals]
