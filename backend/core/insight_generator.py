"""InsightGenerator — LLM-enhanced data insight engine (compact)."""
from __future__ import annotations
import hashlib, json, logging, math, os, re, time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import requests

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
_env = os.getenv
OLLAMA_URL      = _env("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL    = _env("OLLAMA_MODEL", "llama3.2:latest")
LLM_TIMEOUT     = int(_env("LLM_TIMEOUT", "30"))
LLM_BATCH_TO    = int(_env("LLM_BATCH_TIMEOUT", "60"))
# Context inference generates a larger structured JSON response and is on the
# critical path before the parallel batch.  Give it its own, higher ceiling.
LLM_CONTEXT_TIMEOUT = int(_env("LLM_CONTEXT_TIMEOUT", str(max(60, int(_env("LLM_TIMEOUT", "30")) * 2))))
USE_LLM         = _env("USE_LLM", "true").lower() == "true"
LLM_MAX_WORKERS = min(int(_env("LLM_MAX_WORKERS", "2")), 4)
LLM_MAX_RETRIES = int(_env("LLM_MAX_RETRIES", "2"))
LLM_PROVIDER    = _env("LLM_PROVIDER", "ollama")
OPENAI_API_KEY  = _env("OPENAI_API_KEY", "")
OPENAI_MODEL    = _env("OPENAI_MODEL", "gpt-4o-mini")
GROQ_API_KEY    = _env("GROQ_API_KEY", "")
GROQ_MODEL      = _env("GROQ_MODEL", "llama-3.1-8b-instant")
PERSONA         = _env("INSIGHT_PERSONA", "general")
ENABLE_TRACE    = _env("INSIGHT_TRACE", "false").lower() == "true"
CACHE_SIZE      = int(_env("PROMPT_CACHE_SIZE", "128"))
TOKEN_BUDGET    = int(_env("TOKEN_BUDGET", "4096"))

# ── Dataset Context ───────────────────────────────────────────────────────────
@dataclass
class DatasetContext:
    """Holds LLM-inferred domain knowledge about a specific dataset."""
    domain: str = "unknown"
    purpose: str = ""
    column_meanings: Dict[str, str] = field(default_factory=dict)
    target_meaning: str = ""
    key_risks: List[str] = field(default_factory=list)
    leakage_suspects: List[str] = field(default_factory=list)  # post-event / future data
    metadata_cols: List[str] = field(default_factory=list)      # collection-process artefacts
    confidence: float = 0.0

    def is_useful(self) -> bool:
        return self.confidence >= 0.4 and self.domain != "unknown"

    def summary(self) -> str:
        if not self.is_useful():
            return ""
        parts = [f"Domain: {self.domain}"]
        if self.purpose:
            parts.append(f"Purpose: {self.purpose}")
        if self.target_meaning:
            parts.append(f"Target means: {self.target_meaning}")
        if self.key_risks:
            parts.append("Domain risks: " + "; ".join(self.key_risks[:2]))
        if self.metadata_cols:
            parts.append("Likely metadata: " + ", ".join(f'"{c}"' for c in self.metadata_cols[:4]))
        return " | ".join(parts)

    def column_glossary(self, cols: list, max_cols: int = 8) -> str:
        entries = [(c, self.column_meanings[c]) for c in cols[:max_cols] if c in self.column_meanings]
        return "\n".join(f'  "{c}": {m}' for c, m in entries) if entries else ""


# ── Metrics ───────────────────────────────────────────────────────────────────
@dataclass
class InsightMetrics:
    total_llm_calls: int = 0; successful_calls: int = 0; failed_calls: int = 0
    retried_calls: int = 0; cache_hits: int = 0; cache_misses: int = 0
    total_tokens_used: int = 0
    task_latencies: Dict[str, float] = field(default_factory=dict)
    task_failures: Dict[str, str] = field(default_factory=dict)

    def record_call(self, ok: bool, retried=False, tokens=0):
        self.total_llm_calls += 1
        if ok: self.successful_calls += 1
        else: self.failed_calls += 1
        if retried: self.retried_calls += 1
        self.total_tokens_used += tokens

    def record_latency(self, t, s): self.task_latencies[t] = round(s, 3)
    def record_failure(self, t, r): self.task_failures[t] = r

    def to_dict(self) -> dict:
        tot = max(self.cache_hits + self.cache_misses, 1)
        return {
            'total_llm_calls': self.total_llm_calls, 'successful_calls': self.successful_calls,
            'failed_calls': self.failed_calls, 'retried_calls': self.retried_calls,
            'cache_hits': self.cache_hits, 'cache_misses': self.cache_misses,
            'cache_hit_rate': round(self.cache_hits / tot, 3),
            'total_tokens_used': self.total_tokens_used,
            'task_latencies': self.task_latencies, 'task_failures': self.task_failures,
        }

# ── Personas ──────────────────────────────────────────────────────────────────
PERSONAS = {
    'general': ('a friendly, knowledgeable guide explaining data findings to someone curious but not an expert',
        ['Use plain, everyday language — no jargon whatsoever',
         'Explain what the issue actually IS before saying why it matters',
         'Use relatable analogies from everyday life (cooking, sports, directions)',
         'Focus on what this means for the data quality and analysis results',
         'End with one clear, simple action anyone can follow'],
        'ROI, stakeholder, KPI, synergy, leverage, strategic, monetize, revenue impact'),
    'executive': ('a trusted senior advisor speaking to a CEO or VP',
        ['No technical jargon whatsoever', 'Focus on business impact and ROI',
         'Use analogies from business (finance, operations)', 'Be direct and confident',
         'End with a clear next step'],
        'multicollinearity, heteroscedasticity, cardinality, p-value, feature importance, overfitting'),
    'data_scientist': ('a senior ML engineer doing a peer code review',
        ['Be precise and technical — use proper statistical terms',
         'Reference specific sklearn/pandas APIs', 'Mention effect sizes, confidence intervals where relevant',
         'Suggest experiments, not just fixes', 'Include code snippets'], ''),
    'product_manager': ('a data-literate colleague explaining to a PM who is comfortable with data but not ML',
        ['Explain the "so what?" for the product or users',
         'Use light technical terms but define any ML-specific ones',
         'Connect findings to user impact or KPIs', 'Be concise and actionable',
         'Prioritise recommendations by effort vs impact'],
        'heteroscedasticity, eigenvalue, gradient, kernel'),
}

def _persona_block(key: str) -> str:
    tone, rules, avoid = PERSONAS.get(key, PERSONAS['executive'])
    b = f"Tone: {tone}\nRules:\n" + '\n'.join(f'  - {r}' for r in rules)
    return b + f"\nNEVER use these words: {avoid}" if avoid else b

# ── Prompt Cache ──────────────────────────────────────────────────────────────
class PromptCache:
    __slots__ = ('_c', '_o', '_mx')
    def __init__(self, mx=CACHE_SIZE):
        self._c, self._o, self._mx = {}, [], mx

    @staticmethod
    def _key(p, m): return hashlib.sha256(f"{m}:{p}".encode()).hexdigest()[:16]

    def get(self, p, m): return self._c.get(self._key(p, m))

    def put(self, p, m, r):
        k = self._key(p, m)
        if k in self._c: return
        if len(self._c) >= self._mx: self._c.pop(self._o.pop(0), None)
        self._c[k] = r; self._o.append(k)

    def fingerprint(self, p, m): return self._key(p, m)

# ── Output Parser ─────────────────────────────────────────────────────────────
_JSON_RE = re.compile(r'(\[[\s\S]*\]|\{[\s\S]*\})')

class OutputParser:
    @classmethod
    def extract_json(cls, text: str) -> Optional[Any]:
        if not text: return None
        cleaned = re.sub(r'```(?:json)?\s*', '', text).strip()
        try: return json.loads(cleaned)
        except (json.JSONDecodeError, ValueError): pass
        m = _JSON_RE.search(cleaned)
        if m:
            try: return json.loads(m.group(1))
            except (json.JSONDecodeError, ValueError): pass
        return None

    @classmethod
    def extract_json_array(cls, text: str, expected: int) -> Optional[list]:
        p = cls.extract_json(text)
        if not isinstance(p, list) or not p: return None
        if len(p) != expected: logger.warning(f"JSON array length mismatch: got {len(p)}, expected {expected}")
        return p

    @classmethod
    def sanitize_text(cls, t: str) -> str:
        if not t: return ""
        return re.sub(r'\n{3,}', '\n\n', re.sub(r'```(?:json)?\s*', '', t)).strip()

    @classmethod
    def parse_numbered_list(cls, text: str, expected: int) -> Dict[int, str]:
        result: Dict[int, str] = {}; idx = None; buf = []
        for line in (l.strip() for l in text.strip().split('\n') if l.strip()):
            m = re.match(r'^#?\s*(\d+)\s*[.):\-]\s*(.*)', line)
            if m:
                if idx is not None and buf: result[idx] = ' '.join(buf).strip()
                idx = int(m.group(1)); buf = [m.group(2)] if m.group(2) else []
            elif idx is not None: buf.append(line)
        if idx is not None and buf: result[idx] = ' '.join(buf).strip()
        return {k: v for k, v in result.items() if 1 <= k <= expected}

# ── LLM Providers ─────────────────────────────────────────────────────────────
class LLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 250, timeout: int = 30,
                 json_mode: bool = False) -> str: ...
    @abstractmethod
    def is_available(self) -> bool: ...
    @property
    @abstractmethod
    def model_name(self) -> str: ...


class OllamaProvider(LLMProvider):
    __slots__ = ('_url', '_model', '_base')
    def __init__(self, url=OLLAMA_URL, model=OLLAMA_MODEL):
        self._url, self._model = url, model
        self._base = url.replace("/api/generate", "")

    @property
    def model_name(self): return self._model

    def is_available(self) -> bool:
        try:
            r = requests.get(f"{self._base}/api/tags", timeout=3)
            if r.status_code != 200: return False
            names = [m.get('name', '') for m in r.json().get('models', [])]
            bn = self._model.split(':')[0]
            ok = any(self._model in n or bn == n.split(':')[0] for n in names)
            if not ok: logger.warning(f"Ollama running but model '{self._model}' not found. Available: {', '.join(names[:5])}")
            return ok
        except Exception: return False

    def generate(self, prompt, max_tokens=250, timeout=30, json_mode=False) -> str:
        try:
            payload = {
                "model": self._model, "prompt": prompt, "stream": False,
                "options": {"temperature": 0.3, "num_predict": max_tokens},
            }
            if json_mode:
                payload["format"] = "json"  # forces valid JSON at the Ollama API level
            r = requests.post(self._url, json=payload, timeout=(5, timeout))
            if r.status_code == 200: return r.json().get('response', '').strip()
            logger.warning(f"Ollama status {r.status_code}")
        except requests.exceptions.Timeout: logger.warning(f"Ollama timeout after {timeout}s")
        except requests.exceptions.ConnectionError: logger.warning("Ollama connection refused")
        except Exception as e: logger.error(f"Ollama error: {e}")
        return ""


class _ChatProvider(LLMProvider):
    """Base for OpenAI-compatible chat APIs."""
    __slots__ = ('_key', '_model', '_url')
    def __init__(self, key, model, url):
        self._key, self._model, self._url = key, model, url

    @property
    def model_name(self): return self._model
    def is_available(self): return bool(self._key)

    def generate(self, prompt, max_tokens=250, timeout=30, json_mode=False) -> str:
        try:
            body: dict = {
                "model": self._model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens, "temperature": 0.3,
            }
            if json_mode:
                body["response_format"] = {"type": "json_object"}
            r = requests.post(self._url, headers={"Authorization": f"Bearer {self._key}", "Content-Type": "application/json"},
                json=body, timeout=(5, timeout))
            if r.status_code == 200: return r.json()['choices'][0]['message']['content'].strip()
            logger.warning(f"{self.__class__.__name__} status {r.status_code}: {r.text[:200]}")
        except Exception as e: logger.error(f"{self.__class__.__name__} error: {e}")
        return ""


class OpenAIProvider(_ChatProvider):
    def __init__(self, key=OPENAI_API_KEY, model=OPENAI_MODEL):
        super().__init__(key, model, _env("OPENAI_BASE_URL", "https://api.openai.com/v1") + "/chat/completions")


class GroqProvider(_ChatProvider):
    def __init__(self, key=GROQ_API_KEY, model=GROQ_MODEL):
        super().__init__(key, model, "https://api.groq.com/openai/v1/chat/completions")


def _create_provider(name=LLM_PROVIDER) -> LLMProvider:
    m = {'ollama': OllamaProvider, 'openai': OpenAIProvider, 'groq': GroqProvider}
    cls = m.get(name.lower())
    if not cls: logger.warning(f"Unknown LLM provider '{name}', falling back to Ollama"); return OllamaProvider()
    return cls()

# ── LLM Client ────────────────────────────────────────────────────────────────
class LLMClient:
    __slots__ = ('provider', 'cache', 'metrics', 'max_retries', 'token_budget',
                 'enable_trace', '_consecutive_timeouts')

    def __init__(self, provider, cache, metrics, max_retries=LLM_MAX_RETRIES,
                 token_budget=TOKEN_BUDGET, enable_trace=ENABLE_TRACE):
        self.provider, self.cache, self.metrics = provider, cache, metrics
        self.max_retries, self.token_budget, self.enable_trace = max_retries, token_budget, enable_trace
        self._consecutive_timeouts = 0   # tracks how many successive timeout failures have occurred

    @property
    def is_degraded(self) -> bool:
        """True after ≥2 consecutive timeouts — Ollama is clearly loaded.
        In degraded mode we automatically halve token budgets to speed up generation.
        """
        return self._consecutive_timeouts >= 2

    def call(self, prompt, max_tokens=250, timeout=LLM_TIMEOUT, task_name="",
             json_mode=False) -> str:
        max_tokens = min(max_tokens, self.token_budget)
        # Degrade token budget when the LLM is consistently slow
        if self.is_degraded:
            reduced = max(60, int(max_tokens * 0.6))
            if reduced < max_tokens:
                logger.info(f"[degraded] {task_name}: cutting tokens {max_tokens}→{reduced} "
                            f"(consecutive timeouts: {self._consecutive_timeouts})")
                max_tokens = reduced

        cached = self.cache.get(prompt, self.provider.model_name)
        if cached is not None:
            self.metrics.cache_hits += 1
            if self.enable_trace: logger.info(f"[TRACE] Cache HIT for {task_name}")
            return cached
        self.metrics.cache_misses += 1
        fp = self.cache.fingerprint(prompt, self.provider.model_name)
        if self.enable_trace: logger.info(f"[TRACE] {task_name} fp={fp}, max_tokens={max_tokens}, json={json_mode}")

        last_err = ""
        for att in range(1, self.max_retries + 1):
            t0 = time.time()
            res = self.provider.generate(prompt, max_tokens=max_tokens, timeout=timeout,
                                         json_mode=json_mode)
            dt = time.time() - t0

            if res:
                res = OutputParser.sanitize_text(res)
                self.cache.put(prompt, self.provider.model_name, res)
                self.metrics.record_call(True, retried=(att > 1), tokens=max_tokens)
                if task_name: self.metrics.record_latency(task_name, dt)
                self._consecutive_timeouts = 0   # successful response resets the counter
                return res

            # ── Classify failure type ───────────────────────────────────────────
            # If the call consumed ≥90% of the allowed timeout it was a genuine timeout
            # (not just an empty/refused response).  Timeouts and empty responses need
            # different handling:
            #   • Timeout  → LLM is slow/loaded.  Don't sleep (that wastes more time).
            #                Increment counter; if we've seen many, reduce tokens further.
            #   • Empty    → Transient refusal or parse error.  Brief sleep before retry.
            was_timeout = (dt >= timeout * 0.9)
            if was_timeout:
                self._consecutive_timeouts += 1
                last_err = f"Timeout attempt {att} ({dt:.1f}s)"
                if att < self.max_retries:
                    # Retry immediately with a smaller token budget instead of sleeping.
                    # On a loaded Ollama instance, sleeping just means more requests queue up.
                    max_tokens = max(60, int(max_tokens * 0.7))
                    logger.info(f"Timeout retry {att}/{self.max_retries} for {task_name} "
                                f"(reducing tokens→{max_tokens})")
            else:
                self._consecutive_timeouts = max(0, self._consecutive_timeouts - 1)
                last_err = f"Empty response attempt {att}"
                if att < self.max_retries:
                    time.sleep(2 ** (att - 1))
                    logger.info(f"LLM retry {att}/{self.max_retries} for {task_name}")

        self.metrics.record_call(False)
        if task_name: self.metrics.record_failure(task_name, last_err)
        return ""

    def call_json(self, prompt, max_tokens=250, timeout=LLM_TIMEOUT, task_name="",
                  expected_array_len=0) -> Optional[Any]:
        # json_mode=True: Ollama uses "format":"json", OpenAI uses response_format JSON object
        res = self.call(prompt, max_tokens=max_tokens, timeout=timeout, task_name=task_name,
                        json_mode=True)
        if not res: return None
        parsed = (OutputParser.extract_json_array(res, expected_array_len) if expected_array_len
                  else OutputParser.extract_json(res))
        if parsed is not None: return parsed
        logger.info(f"JSON parse failed for {task_name}, retrying strict")
        res2 = self.call(
            f"{prompt}\n\nCRITICAL: Return ONLY valid JSON. No text, no fences.",
            max_tokens=max_tokens, timeout=timeout, task_name=f"{task_name}_retry",
            json_mode=True)
        if not res2: return None
        return (OutputParser.extract_json_array(res2, expected_array_len) if expected_array_len
                else OutputParser.extract_json(res2))

# ── Insight Ranker ────────────────────────────────────────────────────────────
_IW = {'class_imbalance': 10, 'data_leakage_risk': 10, 'simpsons_paradox': 9, 'high_missingness': 8,
       'metadata_leakage': 8,
       'constant_column': 7, 'multicollinearity': 6, 'high_correlation': 5, 'high_outliers': 5,
       'mixed_types': 5, 'near_zero_variance': 4, 'heteroscedasticity': 4, 'moderate_missingness': 3,
       'high_cardinality': 3, 'disguised_missing': 3, 'infinite_values': 3, 'duplicate_rows': 2,
       'whitespace_issues': 1}
_SM = {'critical': 3, 'high': 2, 'medium': 1}

def _rank(issues): return sorted(issues, key=lambda i: _IW.get(i.get('type', ''), 1) * _SM.get(i.get('severity', 'medium'), 1), reverse=True)

# ── Headline Map ──────────────────────────────────────────────────────────────
def _headline(i: dict) -> str:
    t, col = i.get('type', ''), i.get('column', '')
    cols, v1, v2 = i.get('columns', []), i.get('var1', ''), i.get('var2', '')

    # Handle aggregated findings (Fix 2: Intelligent Aggregation)
    if i.get('aggregated'):
        count = i.get('count', 0)
        affected = i.get('columns', [])
        n_cols = len(affected) if isinstance(affected, list) else 0
        _AGG_H = {
            'high_correlation':    f'\U0001f4ca {count} correlated column pairs ({n_cols} variables)',
            'multicollinearity':   f'\U0001f4ca {count} nearly identical column pairs ({n_cols} variables)',
            'high_outliers':       f'\U0001f4c8 {count} columns with significant outliers',
            'near_zero_variance':  f'\U0001f4c9 {count} columns with near-zero variance',
            'whitespace_issues':   f'\U0001f4a1 {count} columns with whitespace issues',
            'disguised_missing':   f'\u26a0 {count} columns with disguised null values',
            'extreme_skewness':    f'\u26a0 {count} columns with extreme skewness (data corruption)',
        }
        return _AGG_H.get(t, f'{count} {t.replace("_", " ")} issues')

    _H = {
        'class_imbalance':    f'\u26a0 Target "{col}" is severely imbalanced',
        'data_leakage_risk':  f'\U0001f6a8 "{col}" may be leaking target information',
        'metadata_leakage':   f'\U0001f9f9 "{col}" is a data-collection artefact, not a feature',
        'multicollinearity':  f'\U0001f4ca "{v1}" and "{v2}" are nearly identical',
        'near_zero_variance': f'\U0001f4c9 "{col}" has near-zero variance',
        'high_cardinality':   f'\U0001f4cb "{col}" has too many unique values',
        'constant_column':    f'\u274c "{col}" has only one value',
        'mixed_types':        f'\u26a0 "{col}" mixes numeric and text values',
        'disguised_missing':  f'\u26a0 "{col}" has disguised null values',
        'infinite_values':    f'\u26a0 "{col}" contains infinite values',
        'whitespace_issues':  f'\U0001f4a1 "{col}" has whitespace issues',
    }
    if t in _H: return _H[t]
    if t == 'simpsons_paradox' and len(cols) >= 3: return f'\u26a0 "{cols[0]}" vs "{cols[1]}" reverses by "{cols[2]}"'
    if t == 'high_correlation': return f'\U0001f4ca "{v1}" and "{v2}" correlated ({i.get("correlation", "")})'
    if t == 'high_outliers': return f'\U0001f4c8 "{col}" has {i.get("percentage", "?")}% outliers'
    if t == 'heteroscedasticity' and len(cols) >= 2: return f'\U0001f4c9 Error spread in "{cols[1]}" varies with "{cols[0]}"'
    if t in ('high_missingness', 'moderate_missingness'):
        return f'{"❌" if t == "high_missingness" else "⚠"} "{col}" is {i.get("percentage", "?")}% missing'
    if t == 'duplicate_rows': return f'\U0001f501 {i.get("count", "")} duplicate rows'
    return f"Issue: {t.replace('_', ' ').title()}"

# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN CLASS
# ═══════════════════════════════════════════════════════════════════════════════
class InsightGenerator:
    __slots__ = ('use_llm', 'persona', '_metrics', '_cache', '_trace', '_traces',
                 '_provider', '_llm', '_llm_available', '_context', '_user_context')

    def __init__(self, use_llm=None, persona=None, provider=None, enable_trace=None):
        self.use_llm = use_llm if use_llm is not None else USE_LLM
        self.persona = persona or PERSONA
        self._metrics, self._cache = InsightMetrics(), PromptCache()
        self._trace = enable_trace if enable_trace is not None else ENABLE_TRACE
        self._traces = []
        self._provider = provider or _create_provider()
        self._llm = LLMClient(self._provider, self._cache, self._metrics, enable_trace=self._trace)
        self._llm_available = None
        self._context: Optional[DatasetContext] = None  # set at the start of generate_insights()
        self._user_context: str = ""  # user-provided data dictionary / domain context

    @property
    def metrics(self): return self._metrics.to_dict()
    @property
    def traces(self): return list(self._traces)

    # ── Public API ────────────────────────────────────────────────────────
    def generate_insights(self, blueprint: dict, stats: dict, recommendations: dict,
                          user_context: str = "",
                          deterministic_summary: dict = None) -> Dict[str, Any]:
        t0 = time.time()
        self._user_context = user_context or ""

        # ── Step 0: infer dataset domain/context before any other LLM work ──
        if self.use_llm and self._check_llm():
            self._context = self._llm_infer_context(blueprint, stats, recommendations)
        else:
            self._context = DatasetContext()

        all_issues = _rank(list(blueprint.get('quality_issues', [])) +
                           list(stats.get('red_flags', [])) + list(stats.get('warnings', [])))

        # ── Fix 5: Fuse LLM-detected leakage suspects with keyword-based flags ──
        # The keyword check (_LEAK_KW) catches names like "result" or "outcome".
        # The LLM catches semantic suspects the keyword list could never know about
        # (e.g. "days_in_hospital" in a readmission dataset, "final_balance" in a churn one).
        # We deduplicate so we never emit two flags for the same column.
        if self._context and self._context.leakage_suspects:
            already_flagged = {i.get('column') for i in all_issues if i.get('type') == 'data_leakage_risk'}
            tgt = (blueprint.get('target_candidates') or [{}])[0].get('column', '')
            for col in self._context.leakage_suspects:
                if col not in already_flagged and col != tgt and col in {
                    p['name'] for p in blueprint.get('column_profiles', [])
                }:
                    meaning = self._context.column_meanings.get(col, col)
                    all_issues.append({
                        'type': 'data_leakage_risk', 'severity': 'critical', 'column': col,
                        'source': 'llm_semantic',
                        'message': (f'Column "{col}" ({meaning}) was flagged by semantic analysis as '
                                    f'likely post-event data in a {self._context.domain} dataset.'),
                        'plain_english': (f'"{col}" ({meaning}) sounds like information you would only '
                                          f'know AFTER the event you\'re trying to predict — meaning the '
                                          f'model would be learning from the future. It\'ll ace your tests '
                                          f'and fail completely in production.'),
                        'recommendation': (f'Verify whether "{col}" was collected before or after '
                                           f'the target event. If after — remove it immediately.'),
                    })
            all_issues = _rank(all_issues)  # re-rank now that we may have added new criticals

        # ── Fuse LLM-detected operational metadata columns ────────────────────────
        # These differ from leakage_suspects (temporal: post-event data) —
        # metadata_cols are collection-process artefacts available at prediction time
        # but that teach the model irrelevant patterns: batch effects, instrument drift,
        # operator habits, plate numbers etc.  A keyword list (_LEAK_KW) can catch
        # obvious names, but only the LLM can recognise domain-specific codes like
        # "RUN_4B", "PLATE_NO", "ASSAY_ID", or "OPERATOR_CODE_7".
        if self._context and self._context.metadata_cols:
            known_cols = {p['name'] for p in blueprint.get('column_profiles', [])}
            tgt = (blueprint.get('target_candidates') or [{}])[0].get('column', '')
            already_flagged = {i.get('column') for i in all_issues
                               if i.get('type') in ('data_leakage_risk', 'metadata_leakage')}
            for col in self._context.metadata_cols:
                if col not in known_cols or col == tgt or col in already_flagged:
                    continue
                meaning = self._context.column_meanings.get(col, col)
                all_issues.append({
                    'type': 'metadata_leakage', 'severity': 'high', 'column': col,
                    'source': 'llm_semantic',
                    'message': (f'Column "{col}" ({meaning}) appears to be a data-collection '
                                f'artefact — a {self._context.domain} operational identifier '
                                f'rather than a real predictive feature.'),
                    'plain_english': (f'"{col}" describes HOW this data was recorded, not WHAT '
                                      f'was measured. If the model learns from it, it will memorise '
                                      f'coincidences tied to specific batches, machines, or operators '
                                      f'rather than learning the real underlying pattern — like '
                                      f'a student who memorises which teacher always uses "C" as the '
                                      f'answer rather than actually understanding the subject.'),
                    'recommendation': (f'Remove "{col}" from your feature set before training. '
                                       f'Keep it only for data auditing purposes. '
                                       f'If you suspect a genuine batch effect exists, investigate '
                                       f'it separately rather than feeding it to the model.'),
                })
            if any(i.get('source') == 'llm_semantic' for i in all_issues):
                all_issues = _rank(all_issues)

        # ── Aggregation: group identical findings to reduce document bloat ────
        from core.aggregation_engine import AggregationEngine
        model_name = recommendations.get('primary_model', '')
        all_issues = AggregationEngine().aggregate(all_issues, model_name=model_name)

        # ── Relevance Filter: contextualize warnings for the recommended model ──
        from core.relevance_filter import RelevanceFilter
        all_issues = RelevanceFilter().filter(
            all_issues, model_name=model_name,
            task_type=recommendations.get('task_type', 'classification')
        )

        es = self._build_executive_summary(blueprint, stats, recommendations, all_issues)
        crit = [i for i in all_issues if i.get('severity') == 'critical']
        high = [i for i in all_issues if i.get('severity') == 'high']
        med  = [i for i in all_issues if i.get('severity') == 'medium']
        cr = self._build_column_relationships(stats)
        mg = self._build_model_guidance(recommendations)
        qw = self._build_quick_wins(blueprint, stats, recommendations)
        ig = self._build_imbalance_guidance(blueprint, stats)
        llm_ok = False
        data_story = ''

        if self.use_llm and self._check_llm():
            llm_ok = True
            try:
                r = self._run_parallel_llm(blueprint, stats, recommendations, es, crit, high, med, cr, mg, qw, ig, all_issues)
                es, crit, high, med = r.get('executive_summary', es), r.get('critical', crit), r.get('high', high), r.get('medium', med)
                cr, mg, qw, ig = r.get('column_relationships', cr), r.get('model_guidance', mg), r.get('quick_wins', qw), r.get('imbalance_guidance', ig)
                data_story = r.get('data_story', '')
            except Exception as e:
                logger.warning(f"Parallel LLM failed, using templates: {e}")
                llm_ok = False; self._metrics.record_failure('parallel_orchestration', str(e))
                data_story = ''

        elapsed = round(time.time() - t0, 2)
        self._metrics.record_latency('total_insight_generation', elapsed)

        def fmt(lst):
            out = []
            for i in lst:
                entry = {
                    'type': i.get('type', 'unknown'),
                    'severity': i.get('severity', 'medium'),
                    'headline': _headline(i),
                    'what_it_means': i.get('plain_english') or i.get('message', ''),
                    'business_impact': i.get('business_impact', ''),
                    'what_to_do': i.get('recommendation', ''),
                    'deep_dive': i.get('deep_dive', ''),
                    'column': i.get('column') or i.get('columns'),
                    # New fields for interactive features
                    'aggregated': i.get('aggregated', False),
                    'count': i.get('count'),
                    'model_context_note': i.get('model_context_note', ''),
                    'action_priority': i.get('action_priority', ''),
                }
                # Include aggregated sub-items for frontend expansion
                if i.get('aggregated') and i.get('pairs'):
                    entry['pairs'] = i['pairs']
                if i.get('aggregated') and i.get('columns'):
                    entry['affected_columns'] = i['columns']
                out.append(entry)
            return out

        result = {
            'executive_summary': es, 'data_story': data_story,
            'domain_context': {
                'domain': self._context.domain,
                'purpose': self._context.purpose,
                'target_meaning': self._context.target_meaning,
                'key_risks': self._context.key_risks,
                'leakage_suspects': self._context.leakage_suspects,
                'metadata_cols': self._context.metadata_cols,
                'column_meanings': self._context.column_meanings,
                'confidence': self._context.confidence,
            } if self._context else None,
            'user_context_provided': bool(self._user_context),
            'critical_insights': fmt(crit),
            'high_priority_insights': fmt(high), 'medium_priority_insights': fmt(med),
            'column_relationships': cr, 'class_imbalance_guidance': ig, 'model_guidance': mg,
            'quick_wins': qw, 'total_insights': len(all_issues),
            'severity_breakdown': {'critical': len(crit), 'high': len(high), 'medium': len(med)},
            'llm_enhanced': llm_ok, 'persona': self.persona,
            'llm_provider': self._provider.model_name if llm_ok else None,
            'generation_time_seconds': elapsed,
            'metrics': self._metrics.to_dict() if self._trace else None,
        }
        if self._trace:
            self._traces.append({'timestamp': time.time(), 'llm_enhanced': llm_ok,
                'persona': self.persona, 'provider': self._provider.model_name,
                'metrics': self._metrics.to_dict()})
        return result

    # kept for backward compat
    def _collect_all_issues(self, bp, stats):
        return list(bp.get('quality_issues', [])) + list(stats.get('red_flags', [])) + list(stats.get('warnings', []))

    def _format_insights(self, issues):
        return [{'type': i.get('type', 'unknown'), 'severity': i.get('severity', 'medium'),
            'headline': _headline(i), 'what_it_means': i.get('plain_english') or i.get('message', ''),
            'business_impact': i.get('business_impact', ''), 'what_to_do': i.get('recommendation', ''),
            'deep_dive': i.get('deep_dive', ''), 'column': i.get('column') or i.get('columns')} for i in issues]

    def _make_headline(self, issue): return _headline(issue)

    # ── LLM check ─────────────────────────────────────────────────────────
    def _check_llm(self):
        if self._llm_available is not None: return self._llm_available
        self._llm_available = self._provider.is_available()
        if not self._llm_available: logger.info(f"LLM '{self._provider.model_name}' not available — templates only")
        return self._llm_available

    # ── Parallel LLM ──────────────────────────────────────────────────────
    def _run_parallel_llm(self, bp, stats, recs, es, crit, high, med, cr, mg, qw, ig, all_issues):
        # ── Cap issue lists before they're sent to LLM ───────────────────────────
        # max_tokens per task scales with len(issues), so unbounded lists directly
        # cause timeouts.  Keep the most important ones (already ranked by _rank).
        crit_llm = crit[:4]
        high_llm = high[:5]
        med_llm  = med[:5]

        res = {'executive_summary': es, 'critical': crit, 'high': high, 'medium': med,
               'column_relationships': cr, 'model_guidance': mg, 'quick_wins': qw,
               'imbalance_guidance': ig, 'data_story': ''}

        with ThreadPoolExecutor(max_workers=LLM_MAX_WORKERS) as pool:
            futs = {}
            futs[pool.submit(self._llm_executive_summary, bp, stats, recs, es)] = 'executive_summary'
            futs[pool.submit(self._llm_data_story, bp, stats, recs, all_issues)] = 'data_story'
            if crit_llm: futs[pool.submit(self._llm_deep_dive_critical, crit_llm, bp, recs)] = 'critical'
            if high_llm: futs[pool.submit(self._llm_enhance_high, high_llm, bp)] = 'high'
            if med_llm:  futs[pool.submit(self._llm_enhance_medium, med_llm, bp)] = 'medium'
            if qw:       futs[pool.submit(self._llm_enhance_quick_wins, qw, bp)] = 'quick_wins'
            futs[pool.submit(self._llm_enhance_model_guidance, mg, bp, recs)] = 'model_guidance'
            if cr:       futs[pool.submit(self._llm_enhance_relationships, cr)] = 'relationships'
            if ig:       futs[pool.submit(self._llm_enhance_imbalance, ig, bp)] = 'imbalance_guidance'

            # ── Dynamic batch timeout ─────────────────────────────────────────────
            # With LLM_MAX_WORKERS=2 and 9 tasks, ceil(9/2)=5 rounds are needed.
            # Using a flat 60s means only 1 round (2 tasks) ever completes.
            # We calculate how long all queued tasks could realistically take, then
            # add a generous buffer so slower models still finish.
            # Formula: ceil(tasks / workers) × (per_task_timeout + 15s overhead) + 30s buffer
            n_tasks  = len(futs)
            per_task = LLM_BATCH_TO          # per-task timeout (user-configured)
            rounds   = math.ceil(n_tasks / max(LLM_MAX_WORKERS, 1))
            dynamic_to = rounds * (per_task + 15) + 30
            effective_to = max(LLM_BATCH_TO, dynamic_to)
            logger.info(f"Batch: {n_tasks} tasks / {LLM_MAX_WORKERS} workers = "
                        f"{rounds} rounds × {per_task+15}s + 30 = {effective_to}s budget")

            try:
                for f in as_completed(futs, timeout=effective_to):
                    k = futs[f]
                    try:
                        v = f.result()
                        if v is not None:
                            if k == 'relationships':
                                em = {(r['col_a'], r['col_b']): r for r in v}
                                res['column_relationships'] = [em.get((r['col_a'], r['col_b']), r) for r in cr]
                            elif k == 'critical':
                                # LLM only enhanced the capped slice; splice back into full crit list
                                res['critical'] = v + crit[len(crit_llm):]
                            elif k == 'high':
                                res['high'] = v + high[len(high_llm):]
                            elif k == 'medium':
                                res['medium'] = v + med[len(med_llm):]
                            else:
                                res[k] = v
                    except Exception as e:
                        logger.warning(f"LLM '{k}' failed: {e}"); self._metrics.record_failure(k, str(e))
            except FuturesTimeoutError:
                done = [futs[f] for f in futs if f.done()]
                pend = [futs[f] for f in futs if not f.done()]
                logger.warning(f"Batch timeout ({effective_to}s). Done: {done}. Pending: {pend}")
                self._metrics.record_failure('batch_timeout', f"pending: {pend}")
                for f in futs:
                    if not f.done(): f.cancel()
        return res

    # ── Template Engine ───────────────────────────────────────────────────
    def _build_executive_summary(self, bp, stats, recs, all_issues) -> str:
        b = bp.get('basic_info', {})
        rows, cols, mpct = b.get('rows', 0), b.get('columns', 0), b.get('missing_percentage', 0)
        dups, pri, task = b.get('duplicate_rows', 0), recs.get('primary_model', 'a machine learning model'), recs.get('task_type', 'classification')
        cn, hn = sum(1 for i in all_issues if i.get('severity') == 'critical'), sum(1 for i in all_issues if i.get('severity') == 'high')
        hp = max(0, 100 - cn*20 - hn*10 - min(mpct*2, 30) - (10 if dups else 0))
        lab = "good shape" if hp >= 80 else "fair condition with some issues" if hp >= 60 else "several problems that need fixing" if hp >= 40 else "significant quality issues that must be resolved"
        parts = [f"Your dataset has {rows:,} rows and {cols} columns and is currently in {lab}."]
        if mpct > 0: parts.append(f"About {mpct:.1f}% of values are missing, which can affect the quality of any analysis.")
        if dups > 0: parts.append(f"There are {dups:,} duplicate rows — removing these will give cleaner, more reliable results.")
        if cn > 0: parts.append(f"There {'is' if cn==1 else 'are'} {cn} critical issue{'s' if cn>1 else ''} that need attention before getting useful results.")
        strong = [p for p in stats.get('correlations', []) if abs(p.get('correlation', 0)) >= 0.7]
        if strong:
            t = strong[0]; parts.append(f'"{t["col_a"]}" and "{t["col_b"]}" move very closely together ({t["correlation"]:+.2f}), which is worth knowing about.')
        for f in all_issues:
            if f.get('type') == 'class_imbalance':
                parts.append(f'The "{f.get("column","")}" column is heavily skewed ({f.get("majority_pct",0):.0f}% of values in one group) — a model trained on this as-is would be unreliable.'); break
        parts.append(f"{pri} looks like a good fit for this {task} task.")
        return " ".join(parts)

    def _build_model_guidance(self, recs):
        c = recs.get('confidence', 0.5)
        return {
            'recommended_model': recs.get('primary_model', 'Unknown'),
            'task_type': recs.get('task_type', 'classification'),
            'confidence_label': "High confidence" if c >= 0.75 else "Moderate confidence" if c >= 0.55 else "Low confidence — explore alternatives",
            'confidence_score': c, 'why_this_model': recs.get('why_this_model', ''),
            'key_reasons': recs.get('reasoning', []), 'alternatives': recs.get('alternatives', []),
            'before_you_train': recs.get('preprocessing_steps', []),
            'how_to_validate': recs.get('cv_strategy', ''),
            'how_to_measure_success': recs.get('recommended_metrics', []),
        }

    def _build_quick_wins(self, bp, stats, recs):
        wins, b, ch = [], bp.get('basic_info', {}), recs.get('characteristics', {})
        if b.get('duplicate_rows', 0) > 0: wins.append(f"Remove {b['duplicate_rows']:,} duplicates — df.drop_duplicates(inplace=True)")
        if ch.get('has_missing'): wins.append("Fix missing values — df.fillna(df.median(numeric_only=True), inplace=True)")
        if any(f['type'] == 'class_imbalance' for f in stats.get('red_flags', [])): wins.append("Address class imbalance — add class_weight='balanced' to your model")
        const = [i['column'] for i in bp.get('quality_issues', []) if i.get('type') == 'constant_column']
        if const: wins.append(f"Drop constant columns {', '.join(repr(c) for c in const[:3])}")
        mc = [(w['var1'], w['var2']) for w in stats.get('warnings', []) if w.get('type') == 'multicollinearity']
        if mc: wins.append(f'Remove one of "{mc[0][0]}" or "{mc[0][1]}" — nearly identical')
        return wins[:5] or ["Data looks clean — proceed to model training"]

    def _build_imbalance_guidance(self, bp, stats):
        flags = [f for f in stats.get('red_flags', []) if f.get('type') == 'class_imbalance']
        if not flags: return None
        f = flags[0]; col, maj = f.get('column', 'unknown'), f.get('majority_pct', 0)
        info = next((t for t in bp.get('target_candidates', []) if t.get('column') == col), {})
        return {
            'target_column': col, 'majority_pct': maj,
            'n_classes': info.get('n_classes', 2), 'imbalance_ratio': info.get('imbalance_ratio'),
            'why_it_matters': f'Target "{col}" has {maj:.0f}% in one class. A model always predicting majority would be {maj:.0f}% "accurate" but useless.',
            'techniques': [
                {'name': 'Class Weights', 'difficulty': 'easy',
                 'description': f'Set class_weight="balanced" to penalise minority misclassification in "{col}".',
                 'code_hint': 'LogisticRegression(class_weight="balanced")'},
                {'name': 'SMOTE', 'difficulty': 'moderate',
                 'description': 'Synthetic minority oversampling. Apply to training data ONLY.',
                 'code_hint': 'from imblearn.over_sampling import SMOTE; SMOTE().fit_resample(X_train, y_train)'},
                {'name': 'Random Undersampling', 'difficulty': 'easy',
                 'description': 'Remove majority rows at random. Simple but discards data.',
                 'code_hint': 'from imblearn.under_sampling import RandomUnderSampler'},
                {'name': 'Threshold Tuning', 'difficulty': 'moderate',
                 'description': f'Lower prediction threshold for "{col}" using precision-recall curve.',
                 'code_hint': 'from sklearn.metrics import precision_recall_curve'},
            ],
            'wrong_metrics': ['accuracy'],
            'right_metrics': ['F1-score', 'AUC-ROC', 'Precision-Recall AUC', "Cohen's Kappa"],
            'metric_explanation': f'Do NOT use accuracy. Use F1-score or AUC-ROC for "{col}".',
        }

    def _build_column_relationships(self, stats):
        rels = []
        for p in stats.get('correlations', []):
            a, b, v = p['col_a'], p['col_b'], p['correlation']
            if abs(v) < 0.3: continue
            if abs(v) >= 0.9:   expl, act, sev = f'"{a}" and "{b}" nearly perfectly correlated ({v:+.2f}).', 'Drop one or combine with PCA.', 'high'
            elif abs(v) >= 0.7: expl, act, sev = f'"{a}" and "{b}" strongly correlated ({v:+.2f}).', 'Fine for tree models. For linear models, consider removing one.', 'medium'
            else:               expl, act, sev = f'"{a}" and "{b}" moderately related ({v:+.2f}).', 'No action needed.', 'info'
            rels.append({'col_a': a, 'col_b': b, 'correlation': v, 'strength': p.get('strength', ''),
                'direction': 'positive' if v > 0 else 'negative', 'explanation': expl, 'action': act, 'severity': sev})
        for f in stats.get('red_flags', []):
            if f.get('type') == 'simpsons_paradox':
                c = f.get('columns', [])
                if len(c) >= 3:
                    rels.append({'col_a': c[0], 'col_b': c[1], 'split_by': c[2], 'correlation': None,
                        'strength': 'reversal', 'direction': 'reverses by group',
                        'explanation': f'"{c[0]}" vs "{c[1]}" reverses by "{c[2]}".', 'action': f'Analyse separately for each group in "{c[2]}".',
                        'severity': 'critical'})
        return sorted(rels, key=lambda r: abs(r['correlation'] or 0), reverse=True)

    # ── Persona + Context instruction ─────────────────────────────────────
    def _persona_instruction(self) -> str:
        base = _persona_block(self.persona)
        parts = [base]

        # Inject user-provided data dictionary / domain context (Fix 3)
        if self._user_context:
            parts.append(
                f"\nUSER-PROVIDED DATA DICTIONARY (treat as ground truth for domain and column meanings):\n"
                f"{self._user_context[:2000]}"  # cap to prevent prompt explosion
            )

        if self._context and self._context.is_useful():
            ctx = self._context.summary()
            parts.append(f"\nDATASET CONTEXT (use this to make insights specific, not generic):\n{ctx}")
        return "\n".join(parts)

    # ── LLM Task 0: Domain Context Inference ─────────────────────────────
    def _llm_infer_context(self, bp: dict, stats: dict, recs: dict) -> DatasetContext:
        """One upfront call to figure out what this dataset is actually about.

        Prompt design choices:
        - NO persona block: this is a classification task, not a user-facing response.
          The persona instructions add ~150 tokens of overhead with zero benefit here.
        - Target detail block first: the actual class labels (e.g. "setosa, versicolor,
          virginica") are extracted from top_values and placed prominently before the
          column snapshot.  Without real class labels the LLM is forced to hallucinate
          the target_meaning (the root cause of bugs like "whether a flower will bloom").
        - Columns capped at 20 × 5 samples: enough for domain inference.  Categorical
          columns use top_values keys (deduplicated category labels) instead of raw
          sample_values, which are both more informative and more compact.
        - max_tokens=550 (raised from 350): the richer target detail + larger snapshot
          legitimately require more output space.  Truncated responses silently force the
          LLM to skip fields and hallucinate fill-ins for anything left unwritten.
        - Uses LLM_CONTEXT_TIMEOUT (≥60s by default) instead of LLM_TIMEOUT (default 30s)
          because this task generates more tokens than a typical single-field call.
        """
        b = bp.get('basic_info', {})
        profiles = bp.get('column_profiles', [])
        target_candidates = bp.get('target_candidates', [])
        target_cols = [t['column'] for t in target_candidates[:2]]

        # Build a map of column name → profile for quick lookup
        profile_map = {p['name']: p for p in profiles}

        # ── Column snapshot: up to 20 cols × 5 sample values ─────────────────
        # More samples give the LLM enough signal to recognise well-known datasets
        # (e.g. seeing "setosa, versicolor, virginica" immediately identifies Iris).
        col_snapshot_lines = []
        for p in profiles[:20]:
            name = p['name']
            dtype = p.get('dtype', '?')
            samples = [str(v) for v in p.get('sample_values', [])[:5]]
            # For categorical columns, prefer top_values keys (the actual category labels)
            # over raw sample_values — they are deduplicated and more informative.
            if p.get('top_values') and p.get('kind') in ('categorical', 'numeric_categorical'):
                samples = [str(k) for k in list(p['top_values'].keys())[:5]]
            col_snapshot_lines.append(
                f'  "{name}" [{dtype}]: {", ".join(samples) if samples else "no samples"}'
            )
        col_snapshot = "\n".join(col_snapshot_lines)

        # ── Target column detail block ────────────────────────────────────────
        # This is the single most important piece of context for accurate target_meaning.
        # Without the actual class labels the LLM is forced to hallucinate.
        target_detail_lines = []
        for t in target_candidates[:2]:
            col = t['column']
            p = profile_map.get(col, {})
            task_type = t.get('task_type', recs.get('task_type', 'unknown'))
            n_classes = t.get('n_classes')
            top_vals = p.get('top_values', {})
            if top_vals:
                class_list = ", ".join(
                    f'"{k}" ({v} rows)' for k, v in list(top_vals.items())[:10]
                )
                target_detail_lines.append(
                    f'  "{col}" [{task_type}] — {n_classes} classes: {class_list}'
                )
            else:
                imbalance = t.get('imbalance_ratio')
                target_detail_lines.append(
                    f'  "{col}" [{task_type}] — {n_classes or "?"} classes'
                    + (f', imbalance ratio {imbalance:.1f}x' if imbalance else '')
                )
        target_detail = "\n".join(target_detail_lines) or "  unknown"

        # Include user-provided data dictionary if available (Fix 3: Dynamic Context Injection)
        user_ctx_block = ""
        if self._user_context:
            user_ctx_block = (
                f"\n\nUSER-PROVIDED DATA DICTIONARY (treat as authoritative ground truth):\n"
                f"{self._user_context[:1500]}\n"
            )

        prompt = f"""Classify this dataset for ML. Respond ONLY with valid JSON.

Shape: {b.get('rows',0):,} rows × {b.get('columns',0)} cols
Task: {recs.get('task_type', 'unknown')}

TARGET COLUMN(S) — use the actual class labels below to write an accurate target_meaning:
{target_detail}

All columns (name [dtype]: sample/category values):
{col_snapshot}{user_ctx_block}

JSON schema (return ONLY this, no other text):
{{
  "domain": "one of: healthcare|finance|e-commerce|telecom|hr|logistics|marketing|real-estate|education|manufacturing|social-media|sports|government|other",
  "purpose": "one sentence: what ML task this dataset is for, referencing the actual target classes",
  "target_meaning": "plain English: what predicting the target means — name the actual classes/values, do NOT invent meanings",
  "column_meanings": {{"col": "brief meaning", ...}},
  "key_risks": ["up to 3 domain-specific data quality risks"],
  "leakage_suspects": ["cols with POST-EVENT data unavailable at prediction time — [] if none"],
  "metadata_cols": ["cols that are collection artefacts: batch IDs, machine codes, run numbers, operator IDs, ETL timestamps — [] if none"],
  "confidence": 0.0
}}"""
        # max_tokens raised from 350 → 550: the richer column snapshot + longer class lists
        # mean the JSON response legitimately needs more space.  Truncated responses were
        # forcing the LLM to hallucinate fill-ins for fields it never got to write.
        raw = self._llm.call_json(prompt, max_tokens=550, timeout=LLM_CONTEXT_TIMEOUT,
                                  task_name='context_inference')
        if not raw or not isinstance(raw, dict):
            logger.info("Context inference returned nothing — proceeding without domain context")
            return DatasetContext()
        ctx = DatasetContext(
            domain=raw.get('domain', 'unknown'),
            purpose=raw.get('purpose', ''),
            column_meanings=raw.get('column_meanings', {}),
            target_meaning=raw.get('target_meaning', ''),
            key_risks=raw.get('key_risks', []),
            leakage_suspects=[c for c in raw.get('leakage_suspects', []) if isinstance(c, str)],
            metadata_cols=[c for c in raw.get('metadata_cols', []) if isinstance(c, str)],
            confidence=float(raw.get('confidence', 0.0)),
        )
        logger.info(f"Domain context inferred: {ctx.domain} (confidence={ctx.confidence:.2f})")
        return ctx

    def _context_glossary(self, cols: list) -> str:
        """Returns a formatted column glossary for use inside prompts."""
        if not self._context or not self._context.is_useful():
            return ""
        glossary = self._context.column_glossary(cols)
        return f"\nColumn meanings in this {self._context.domain} dataset:\n{glossary}" if glossary else ""

    # ── LLM Task 1: Executive Summary ─────────────────────────────────────
    def _llm_executive_summary(self, bp, stats, recs, fallback):
        b, ch = bp.get('basic_info', {}), recs.get('characteristics', {})
        issues = "\n".join(
            [f"- CRITICAL: {f.get('message', f.get('type', ''))}" for f in stats.get('red_flags', [])[:3]] +
            [f"- WARNING: {w.get('message', w.get('type', ''))}" for w in stats.get('warnings', [])[:3]]
        ) or "No major issues."
        strong = [p for p in stats.get('correlations', []) if abs(p.get('correlation', 0)) >= 0.7]
        ct = ("\nCorrelations: " + ", ".join(
            f"{p['col_a']} & {p['col_b']} ({p['correlation']:+.2f})" for p in strong[:3]
        )) if strong else ""
        ctx = [x for cond, x in [(ch.get('has_missing'), f"{ch.get('missing_pct',0):.0f}% missing"),
               (ch.get('has_imbalance'), "imbalanced target"), (ch.get('has_outliers'), "outliers"),
               (ch.get('is_small'), "small dataset"), (ch.get('is_large'), "large dataset")] if cond]
        prompt = f"""{self._persona_instruction()}
DATASET: {b.get('rows',0):,} rows x {b.get('columns',0)} cols
Missing: {b.get('missing_percentage',0):.1f}% | Dupes: {b.get('duplicate_rows',0)}
Model: {recs.get('primary_model','')} ({recs.get('task_type','')})
Context: {', '.join(ctx) or 'clean'}{ct}
FINDINGS:\n{issues}
Write 4-5 flowing sentences: honest assessment of data quality, biggest problem and how it would affect results, recommended model and why it fits, most important first step to take. Ground your language in the actual domain — say what these columns mean in plain life. No bullets."""
        return self._llm.call(prompt, max_tokens=280, task_name='executive_summary') or fallback

    # ── LLM Task 2: Critical Deep Dive ────────────────────────────────────
    def _llm_deep_dive_critical(self, ins, bp, recs):
        if not ins: return ins
        b, model = bp.get('basic_info', {}), recs.get('primary_model', 'your model')
        secs = "\n".join(f"ISSUE {i} [{x.get('type','').replace('_',' ').upper()}]\n  Column: {x.get('column') or x.get('columns','')}\n  Explanation: {x.get('plain_english') or x.get('message','')}\n  Fix: {x.get('recommendation','')}"
            for i, x in enumerate(ins, 1))
        involved_cols = list({x.get('column') or '' for x in ins} | {c for x in ins for c in (x.get('columns') or [])})
        glossary = self._context_glossary(involved_cols)
        prompt = f"""{self._persona_instruction()}
Dataset: {b.get('rows',0):,} rows x {b.get('columns',0)} cols. Model: {model}.{glossary}
{secs}
For EACH issue return JSON: {{"plain_english":"explain the issue in simple terms a non-expert can understand","analysis_impact":"what goes wrong in the analysis or model if this is left unfixed","what_to_do":"specific step-by-step fix in plain language","deep_dive":"a relatable analogy or deeper explanation of why this matters"}}
Return ONLY a JSON array of {len(ins)} objects."""
        p = self._llm.call_json(prompt, max_tokens=min(220*len(ins), 700),
            timeout=LLM_BATCH_TO, task_name='critical_deep_dive', expected_array_len=len(ins))
        if not p: return ins
        return [{**x, 'plain_english': u.get('plain_english', x.get('plain_english', '')),
                 'business_impact': u.get('business_impact', ''), 'recommendation': u.get('what_to_do', x.get('recommendation', '')),
                 'deep_dive': u.get('deep_dive', '')} if i < len(p) and isinstance(p[i], dict) else x
                for i, (x, u) in enumerate((x, p[i] if i < len(p) else {}) for i, x in enumerate(ins))]

    # ── LLM Task 3: High Enhancement ─────────────────────────────────────
    def _llm_enhance_high(self, ins, bp):
        if not ins: return ins
        b = bp.get('basic_info', {})
        secs = "\n".join(f"{i}. [{x.get('type','').replace('_',' ')}] {x.get('column') or x.get('columns','')} — {x.get('plain_english') or x.get('message','')}"
            for i, x in enumerate(ins, 1))
        involved_cols = [x.get('column') or x.get('var1') or '' for x in ins]
        glossary = self._context_glossary(involved_cols)
        prompt = f"""{self._persona_instruction()}
Dataset: {b.get('rows',0):,} rows x {b.get('columns',0)} cols.{glossary}
HIGH issues:\n{secs}
For each: JSON {{"plain_english":"clear explanation anyone can understand","analysis_impact":"what goes wrong if this is ignored","what_to_do":"specific fix in plain steps"}}
Return ONLY a JSON array of {len(ins)} objects."""
        p = self._llm.call_json(prompt, max_tokens=min(130*len(ins), 500),
            timeout=LLM_BATCH_TO, task_name='high_enhance', expected_array_len=len(ins))
        if not p: return ins
        return [{**x, 'plain_english': u.get('plain_english', x.get('plain_english', '')),
                 'business_impact': u.get('analysis_impact', ''), 'recommendation': u.get('what_to_do', x.get('recommendation', ''))}
                if i < len(p) and isinstance(u := p[i], dict) else x for i, x in enumerate(ins)]

    # ── LLM Task 4: Medium Enhancement ───────────────────────────────────
    def _llm_enhance_medium(self, ins, bp):
        if not ins: return ins
        b = bp.get('basic_info', {})
        secs = "\n".join(
            f"{i}. [{x.get('type','').replace('_',' ')}] col={x.get('column') or x.get('columns','')} — {x.get('plain_english') or x.get('message','')}"
            for i, x in enumerate(ins, 1)
        )
        involved_cols = [x.get('column') or x.get('var1') or '' for x in ins]
        glossary = self._context_glossary(involved_cols)
        prompt = f"""{self._persona_instruction()}
Dataset: {b.get('rows',0):,} rows x {b.get('columns',0)} cols.{glossary}
MEDIUM issues:\n{secs}
For each return JSON: {{"plain_english":"1-2 sentence plain rewrite, keep column names quoted","analysis_impact":"what could go wrong if this is ignored","what_to_do":"specific actionable fix in plain language"}}
Return ONLY a JSON array of {len(ins)} objects."""
        p = self._llm.call_json(prompt, max_tokens=min(110*len(ins), 450),
            timeout=LLM_BATCH_TO, task_name='medium_enhance', expected_array_len=len(ins))
        if not p: return ins
        return [{**x,
                 'plain_english': u.get('plain_english', x.get('plain_english', '')),
                 'business_impact': u.get('analysis_impact', x.get('business_impact', '')),
                 'recommendation': u.get('what_to_do', x.get('recommendation', ''))}
                if i < len(p) and isinstance(u := p[i], dict) else x
                for i, x in enumerate(ins)]

    # ── LLM Task 5: Quick Wins ────────────────────────────────────────────
    def _llm_enhance_quick_wins(self, qw, bp):
        if not qw: return qw
        b = bp.get('basic_info', {})
        numbered = "\n".join(f"{i+1}. {w}" for i, w in enumerate(qw))
        prompt = f"""{self._persona_instruction()}
Data cleanup steps for a {b.get('rows',0):,}-row dataset:\n{numbered}
Rewrite each step in plain, friendly language: explain what it does and why it helps. 1-2 sentences + code. ONLY numbered rewrites."""
        r = self._llm.call(prompt, max_tokens=min(80*len(qw), TOKEN_BUDGET), task_name='quick_wins')
        if not r: return qw
        rw = OutputParser.parse_numbered_list(r, len(qw))
        return [rw.get(i+1, w) for i, w in enumerate(qw)]

    # ── LLM Task 6: Model Guidance ────────────────────────────────────────
    def _llm_enhance_model_guidance(self, mg, bp, recs):
        model, task, conf = mg.get('recommended_model', ''), mg.get('task_type', ''), mg.get('confidence_score', 0)
        ch, b = recs.get('characteristics', {}), bp.get('basic_info', {})
        reasons = "\n".join(f"- {r}" for r in mg.get('key_reasons', [])[:5])
        alts = ", ".join(a.get('model', '') for a in mg.get('alternatives', [])[:2]) or "none"
        facts = [x for cond, x in [(ch.get('has_missing'), "missing values"), (ch.get('has_imbalance'), "class imbalance"),
                 (ch.get('has_categoricals'), "categorical data"), (ch.get('is_large'), f"large ({b.get('rows',0):,} rows)"),
                 (ch.get('is_small'), f"small ({b.get('rows',0)} rows)")] if cond]
        existing_steps = mg.get('before_you_train', [])
        cv_hint = mg.get('how_to_validate', '')
        metrics_hint = mg.get('how_to_measure_success', [])
        domain_note = f"\nDomain: {self._context.purpose}" if self._context and self._context.is_useful() and self._context.purpose else ""
        target_note = f"\nTarget means: {self._context.target_meaning}" if self._context and self._context.target_meaning else ""
        prompt = f"""{self._persona_instruction()}
Model: {model} | Task: {task} | Confidence: {conf:.0%} | Alternatives: {alts}
Data characteristics: {', '.join(facts) or 'clean'}{domain_note}{target_note}
Reasons for recommendation:
{reasons}
Preprocessing steps already identified: {existing_steps}
Validation hint: {cv_hint}
Metrics hint: {metrics_hint}

Return a JSON object with:
{{
  "why_this_model": "3-4 sentences in plain language: what {model} actually IS (like a simple analogy), why it suits THIS specific data, and if confidence < 60% mention alternatives worth trying",
  "before_you_train": "2-3 sentences: the most important things to sort out in the data before using {model}, explained simply",
  "how_to_validate": "2 sentences: how to check that the model is actually working well for this task and dataset size",
  "success_metrics": "1-2 sentences: what numbers to look at to know if it worked, and what a good result looks like"
}}
Return ONLY valid JSON."""
        r = self._llm.call_json(prompt, max_tokens=320, task_name='model_guidance')
        if not r or not isinstance(r, dict): return mg
        return {**mg,
                'why_this_model': r.get('why_this_model', mg.get('why_this_model', '')),
                'before_you_train_narrative': r.get('before_you_train', ''),
                'how_to_validate_narrative': r.get('how_to_validate', ''),
                'success_metrics_narrative': r.get('success_metrics', '')}

    # ── LLM Task 7: Column Relationships ──────────────────────────────────
    def _llm_enhance_relationships(self, rels):
        if not rels: return rels
        capped = rels[:8]
        numbered = [
            f'{i}. "{r["col_a"]}" & "{r["col_b"]}" (r={r.get("correlation","N/A")}, severity={r.get("severity","")}, direction={r.get("direction","")}): {r.get("explanation","")}'
            for i, r in enumerate(capped, 1)
        ]
        involved_cols = list({r['col_a'] for r in capped} | {r['col_b'] for r in capped})
        glossary = self._context_glossary(involved_cols)
        prompt = f"""{self._persona_instruction()}{glossary}
Column relationships found in the dataset:
{chr(10).join(numbered)}
For each, return JSON: {{"explanation":"a memorable, everyday analogy in 1-2 sentences that explains why these columns move together, keep column names quoted","action":"what to actually do about this in the analysis"}}
Return ONLY a JSON array of {len(capped)} objects."""
        p = self._llm.call_json(prompt, max_tokens=min(100*len(capped), TOKEN_BUDGET),
            timeout=LLM_BATCH_TO, task_name='relationships', expected_array_len=len(capped))
        if not p: return rels
        enhanced = []
        for i, r in enumerate(capped):
            u = p[i] if i < len(p) and isinstance(p[i], dict) else {}
            enhanced.append({**r,
                'explanation': u.get('explanation', r['explanation']),
                'action': u.get('action', r['action'])})
        return enhanced + rels[8:]

    # ── LLM Task 8: Imbalance Guidance ────────────────────────────────────
    def _llm_enhance_imbalance(self, g, bp):
        if not g: return g
        col, maj, nc = g.get('target_column', ''), g.get('majority_pct', 0), g.get('n_classes', 2)
        b = bp.get('basic_info', {})
        techniques = "\n".join(
            f"- {t['name']} ({t['difficulty']}): {t['description']}" for t in g.get('techniques', [])
        )
        target_note = f"\nIn this dataset: {self._context.target_meaning}" if self._context and self._context.target_meaning else ""
        domain_risks = (f"\nDomain risks: " + "; ".join(self._context.key_risks[:2])) if self._context and self._context.key_risks else ""
        prompt = f"""{self._persona_instruction()}
Dataset: {b.get('rows',0):,} rows. Target "{col}": {maj:.0f}% majority class ({nc} classes).{target_note}{domain_risks}
Techniques available: {techniques or 'class weights, SMOTE, undersampling, threshold tuning'}

Return a JSON object with:
{{
  "why_it_matters": "3-4 sentences: use a vivid everyday analogy (like a coin that lands heads 90% of the time) to explain why {maj:.0f}% imbalance makes the model untrustworthy, in plain language",
  "technique_guidance": "2-3 sentences: given this specific ratio and dataset size, which technique to try first and why, explained simply",
  "metric_reasoning": "2 sentences: explain in everyday terms why the usual accuracy measure fails here and what to use instead",
  "first_step": "one plain, encouraging sentence describing the immediate action the user should take right now"
}}
Return ONLY valid JSON."""
        r = self._llm.call_json(prompt, max_tokens=280, task_name='imbalance_guidance')
        if not r or not isinstance(r, dict): return g
        return {**g,
                'why_it_matters': r.get('why_it_matters', g.get('why_it_matters', '')),
                'technique_guidance': r.get('technique_guidance', ''),
                'metric_reasoning': r.get('metric_reasoning', g.get('metric_explanation', '')),
                'first_step': r.get('first_step', '')}

    # ── LLM Task 9: Data Story ─────────────────────────────────────────────
    def _llm_data_story(self, bp, stats, recs, all_issues):
        """Cohesive narrative connecting all findings into a single story arc."""
        b = bp.get('basic_info', {})
        model = recs.get('primary_model', 'the model')
        task  = recs.get('task_type', 'prediction')
        crit_msgs = [i.get('plain_english') or i.get('message', '') for i in all_issues if i.get('severity') == 'critical'][:3]
        high_msgs = [i.get('plain_english') or i.get('message', '') for i in all_issues if i.get('severity') == 'high'][:3]
        strong = [p for p in stats.get('correlations', []) if abs(p.get('correlation', 0)) >= 0.7]
        cor_note = ", ".join(f'"{p["col_a"]}"↔"{p["col_b"]}" ({p["correlation"]:+.2f})' for p in strong[:2]) or "none notable"

        ctx_summary = self._context.summary() if self._context and self._context.is_useful() else ""
        domain_line = f"\nDomain: {ctx_summary}" if ctx_summary else ""
        prompt = f"""{self._persona_instruction()}
Dataset: {b.get('rows',0):,} rows × {b.get('columns',0)} cols | Task: {task} | Recommended model: {model}{domain_line}
Missing: {b.get('missing_percentage',0):.1f}% | Duplicates: {b.get('duplicate_rows',0)} | Strong correlations: {cor_note}
Critical issues: {'; '.join(crit_msgs) or 'none'}
High issues: {'; '.join(high_msgs) or 'none'}

Write a 5-7 sentence "data story" paragraph that:
1. Opens with a plain description of what kind of data this is and its overall condition — use the domain context to name it specifically
2. Describes the most important issues as a connected narrative (not a list), using language specific to this domain
3. Explains how these issues relate to each other and compound the problem
4. Closes with the single most important thing to fix first, and why
Be vivid, use everyday language, and write as flowing prose — no bullets, no headers. Imagine explaining this to a curious friend who works in {self._context.domain if self._context and self._context.is_useful() else 'this field'}, not a boardroom."""
        return self._llm.call(prompt, max_tokens=320, task_name='data_story') or ""