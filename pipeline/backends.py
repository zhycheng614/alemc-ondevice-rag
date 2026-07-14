"""Generation backend with graceful fallback and fine-grained timing.

Preference order:
  1. llama.cpp CLI binary (env ALEMC_LLAMA_CLI) — preferred on-device because
     `--timings` / stderr expose prefill vs. decode timing and token counts,
     which feed eq:latency (t_prefill, t_decode, n_tokens).
  2. llama-cpp-python — Python bindings; timing derived from the response dict.
  3. deterministic mock — echoes a context-derived answer with a fixed
     synthetic timing model, so the pipeline runs end-to-end without a model.
     Flagged is_mock so it is never reported as real.

Each generate() returns a GenResult with the text and the timing breakdown
needed by the latency and (via the sampler) energy dimensions.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import GEN_MODEL_GGUF, MAX_TOKENS, TEMPERATURE


@dataclass
class GenResult:
    text: str
    ttft_ms: float                 # submit -> first token
    e2e_ms: float                  # submit -> last token
    t_prefill_ms: float            # prompt evaluation time
    t_decode_ms: float             # per-token decode latency (mean)
    n_tokens: int                  # generated tokens
    backend: str = ""
    is_mock: bool = False
    raw_timing: dict = field(default_factory=dict)


class Generator:
    def __init__(self, prefer: str = "auto", model_path: Optional[str] = None,
                 n_threads: Optional[int] = None):
        self.model_path = model_path or GEN_MODEL_GGUF
        self.n_threads = (n_threads or int(os.environ.get("ALEMC_N_THREADS", 0))
                          or (os.cpu_count() or 4))
        self.backend = None
        self._llm = None
        self._cli = os.environ.get("ALEMC_LLAMA_CLI", "")
        # llama-server base URL, e.g. http://127.0.0.1:8080 (preferred: model
        # stays resident, /completion returns exact prompt/decode timings).
        self._server = os.environ.get("ALEMC_LLAMA_SERVER", "")
        # adb on-device CLI: ALEMC_ADB_DIR is the on-phone dir holding llama-cli,
        # lib/, and the GGUF; generation runs on the phone via `adb shell`.
        self._adb_dir = os.environ.get("ALEMC_ADB_DIR", "")
        self._adb_model = os.environ.get("ALEMC_ADB_MODEL",
                                         "gemma-2-2b-it-Q4_K_M.gguf")

        if prefer in ("auto", "adb") and self._adb_dir:
            self.backend = "llama.cpp-adb"
        elif prefer in ("auto", "server") and self._server and self._server_ready():
            self.backend = "llama.cpp-server"
        elif prefer in ("auto", "cli") and self._cli and Path(self._cli).exists():
            self.backend = "llama.cpp-cli"
        elif prefer in ("auto", "python") and self._try_llama_python():
            pass
        else:
            self.backend = "mock"

    def _server_ready(self) -> bool:
        try:
            import urllib.request
            req = urllib.request.Request(self._server.rstrip("/") + "/health")
            with urllib.request.urlopen(req, timeout=3) as resp:
                return resp.status == 200
        except Exception:
            return False

    def _try_llama_python(self) -> bool:
        if not Path(self.model_path).exists():
            return False
        try:
            from llama_cpp import Llama
        except Exception:
            return False
        try:
            self._llm = Llama(
                model_path=self.model_path,
                n_ctx=4096,
                n_threads=self.n_threads,
                logits_all=False,
                verbose=False,
            )
            self.backend = "llama-cpp-python"
            return True
        except Exception:
            return False

    @property
    def is_mock(self) -> bool:
        return self.backend == "mock"

    # -----------------------------------------------------------------------
    def generate(self, prompt: str, max_tokens: int = MAX_TOKENS) -> GenResult:
        if self.backend == "llama.cpp-adb":
            return self._gen_adb(prompt, max_tokens)
        if self.backend == "llama.cpp-server":
            return self._gen_server(prompt, max_tokens)
        if self.backend == "llama.cpp-cli":
            return self._gen_cli(prompt, max_tokens)
        if self.backend == "llama-cpp-python":
            return self._gen_python(prompt, max_tokens)
        return self._gen_mock(prompt, max_tokens)

    # -- llama.cpp on Android via adb ---------------------------------------
    def _gen_adb(self, prompt: str, max_tokens: int) -> GenResult:
        """Run llama-cli on the phone through `adb shell` (single-turn).

        Timing is parsed from the CLI's `[ Prompt: X t/s | Generation: Y t/s ]`
        footer; e2e is measured as host wall-clock (dominated by on-device
        compute, small adb overhead). The prompt is passed base64 to survive
        the shell safely.
        """
        import base64
        import shlex
        b64 = base64.b64encode(prompt.encode()).decode()
        # decode base64 on-device into a var, pass via -p
        remote = (
            f"cd {self._adb_dir} && "
            f"P=$(echo {b64} | base64 -d) && "
            f"LD_LIBRARY_PATH=./lib ./llama-cli -m {self._adb_model} "
            f"-p \"$P\" -n {max_tokens} -t {self.n_threads} --temp {TEMPERATURE} "
            f"-st --no-display-prompt 2>&1"
        )
        t0 = time.perf_counter()
        proc = subprocess.run(["adb", "shell", remote],
                              capture_output=True, text=True, errors="replace")
        e2e_ms = (time.perf_counter() - t0) * 1e3
        out = proc.stdout
        text, timing = self._parse_adb_output(out, prompt)
        prompt_tps = timing.get("prompt_tps", 0.0)
        gen_tps = timing.get("gen_tps", 0.0)
        n_tokens = _count_tokens(text)
        # IMPORTANT: each adb invocation reloads the model (~5-6 s), so
        # wall-clock e2e is NOT representative deployment latency. We instead
        # derive inference latency from llama-cli's own throughput footer
        # (prompt/generation t/s), which excludes model-load time — the honest
        # on-device compute latency. Prompt token count is estimated from the
        # prompt length (~1 token / 4 chars).
        t_decode = (1000.0 / gen_tps) if gen_tps else 0.0
        prompt_tokens = max(1, len(prompt) // 4)
        t_prefill = (prompt_tokens / prompt_tps * 1000.0) if prompt_tps else 0.0
        t_decode_total = n_tokens * t_decode
        infer_e2e_ms = t_prefill + t_decode_total   # excludes model reload
        timing["wallclock_e2e_ms"] = round(e2e_ms, 1)  # keep for audit
        timing["model_reload_overhead_ms"] = round(e2e_ms - infer_e2e_ms, 1)
        return GenResult(
            text=text, ttft_ms=t_prefill, e2e_ms=infer_e2e_ms,
            t_prefill_ms=t_prefill, t_decode_ms=t_decode, n_tokens=n_tokens,
            backend=self.backend, is_mock=False, raw_timing=timing,
        )

    @staticmethod
    def _parse_adb_output(out: str, prompt: str = ""):
        """Extract the answer text and t/s timing from llama-cli REPL output."""
        timing = {}
        m = re.search(r"Prompt:\s*([\d.]+)\s*t/s\s*\|\s*Generation:\s*([\d.]+)\s*t/s", out)
        if m:
            timing["prompt_tps"] = float(m.group(1))
            timing["gen_tps"] = float(m.group(2))
        # Answer: text after the last "> <prompt line>" up to the timing footer.
        # The prompt is echoed after "> "; the generated text follows on the
        # same/next lines until the "[ Prompt:" footer.
        body = out.split("[ Prompt:")[0]
        # The llama-cli spinner animates with backspaces and ends with a
        # "\x08 \x08" (backspace-space-backspace) sequence immediately before
        # the first generated token. That boundary is the most reliable
        # prompt-echo / generation separator, independent of prompt length or
        # terminal line-wrapping. Split on the LAST such spinner sequence.
        spin = re.search(r"[|/\\\-]\x08(?:[|/\\\- ]\x08)*", body)
        if spin:
            # take everything after the final spinner run
            last = None
            for mm in re.finditer(r"(?:[|/\\\- ]\x08)+", body):
                last = mm
            if last:
                body = body[last.end():]
        # Now collapse any residual backspaces and normalize.
        prev = None
        while prev != body:
            prev = body
            body = re.sub(r".\x08", "", body)

        def _norm(s):
            return re.sub(r"\s+", " ", s).strip()
        njoined = _norm(body).split("Exiting...")[0].strip()
        npr = _norm(prompt)
        cut = bool(spin)
        # If the spinner split didn't fire, fall back to prompt-text stripping.
        if not cut and npr and npr in njoined:
            njoined = njoined.split(npr, 1)[1]
            cut = True
        if not cut:
            anchor = _norm(prompt.strip().splitlines()[-1]) if prompt.strip() else ""
            if anchor and anchor in njoined:
                njoined = njoined.rsplit(anchor, 1)[1]
                cut = True
        if not cut:
            tail = npr[-40:]
            if tail and tail in njoined:
                njoined = njoined.split(tail, 1)[1]
        text = re.sub(r"^[|/\\\- ]+", "", njoined).strip()
        return text, timing

    # -- llama.cpp server ---------------------------------------------------
    def _gen_server(self, prompt: str, max_tokens: int) -> GenResult:
        """POST /completion to a resident llama-server.

        The response `timings` block gives exact prefill/decode timing:
          prompt_ms, prompt_n, predicted_ms, predicted_n, predicted_per_token_ms.
        We measure wall-clock e2e ourselves; TTFT is approximated by the
        prompt (prefill) time plus retrieval (added by the pipeline).
        """
        import json as _json
        import urllib.request
        url = self._server.rstrip("/") + "/completion"
        body = _json.dumps(dict(
            prompt=prompt, n_predict=max_tokens, temperature=TEMPERATURE,
            cache_prompt=False, stream=False,
        )).encode()
        req = urllib.request.Request(
            url, data=body, headers={"Content-Type": "application/json"})
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = _json.loads(resp.read())
        e2e_ms = (time.perf_counter() - t0) * 1e3
        tm = data.get("timings", {}) or {}
        t_prefill = float(tm.get("prompt_ms", 0.0))
        n_tokens = int(tm.get("predicted_n", 0)) or _count_tokens(data.get("content", ""))
        t_decode = float(tm.get("predicted_per_token_ms", 0.0)) or (
            float(tm.get("predicted_ms", 0.0)) / n_tokens if n_tokens else 0.0)
        return GenResult(
            text=_clean_answer(data.get("content", ""), prompt),
            ttft_ms=t_prefill, e2e_ms=e2e_ms, t_prefill_ms=t_prefill,
            t_decode_ms=t_decode, n_tokens=n_tokens,
            backend=self.backend, is_mock=False, raw_timing=tm,
        )

    # -- llama.cpp CLI ------------------------------------------------------
    def _gen_cli(self, prompt: str, max_tokens: int) -> GenResult:
        """Invoke the llama.cpp `llama-cli` binary and parse its stderr timings.

        We measure wall-clock ourselves for e2e and rely on llama.cpp's own
        prefill/decode timing lines for the breakdown.
        """
        cmd = [
            self._cli, "-m", self.model_path, "-p", prompt,
            "-n", str(max_tokens), "-t", str(self.n_threads),
            "--temp", str(TEMPERATURE), "-no-cnv",
        ]
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
        e2e_ms = (time.perf_counter() - t0) * 1e3
        text = proc.stdout.strip()
        timing = self._parse_llama_timings(proc.stderr)
        # llama.cpp reports prompt eval (prefill) and eval (decode) blocks.
        t_prefill = timing.get("prompt_eval_ms", 0.0)
        n_tokens = int(timing.get("eval_tokens", 0)) or _count_tokens(text)
        t_decode = (timing.get("eval_ms", 0.0) / n_tokens) if n_tokens else 0.0
        ttft = t_prefill + timing.get("query_embed_ms", 0.0)
        return GenResult(
            text=_clean_answer(text, prompt), ttft_ms=ttft, e2e_ms=e2e_ms,
            t_prefill_ms=t_prefill, t_decode_ms=t_decode, n_tokens=n_tokens,
            backend=self.backend, is_mock=False, raw_timing=timing,
        )

    @staticmethod
    def _parse_llama_timings(stderr: str) -> dict:
        out = {}
        # e.g. "prompt eval time =  123.45 ms /   42 tokens"
        m = re.search(r"prompt eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)", stderr)
        if m:
            out["prompt_eval_ms"] = float(m.group(1))
            out["prompt_tokens"] = int(m.group(2))
        m = re.search(r"[^p]eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)", stderr)
        if m:
            out["eval_ms"] = float(m.group(1))
            out["eval_tokens"] = int(m.group(2))
        return out

    # -- llama-cpp-python ---------------------------------------------------
    def _gen_python(self, prompt: str, max_tokens: int) -> GenResult:
        t0 = time.perf_counter()
        first_token_t = {"t": None}
        # Stream to capture TTFT accurately.
        chunks = []
        for i, out in enumerate(self._llm(
            prompt, max_tokens=max_tokens, temperature=TEMPERATURE, stream=True,
        )):
            if first_token_t["t"] is None:
                first_token_t["t"] = time.perf_counter()
            chunks.append(out["choices"][0]["text"])
        t_end = time.perf_counter()
        text = "".join(chunks)
        ttft_ms = ((first_token_t["t"] or t_end) - t0) * 1e3
        e2e_ms = (t_end - t0) * 1e3
        n_tokens = _count_tokens(text)
        # Approx: prefill ~= TTFT; decode ~= (e2e - ttft) / n_tokens.
        t_decode = (e2e_ms - ttft_ms) / n_tokens if n_tokens else 0.0
        return GenResult(
            text=_clean_answer(text, prompt), ttft_ms=ttft_ms, e2e_ms=e2e_ms,
            t_prefill_ms=ttft_ms, t_decode_ms=t_decode, n_tokens=n_tokens,
            backend=self.backend, is_mock=False,
        )

    # -- mock ---------------------------------------------------------------
    def _gen_mock(self, prompt: str, max_tokens: int) -> GenResult:
        """Deterministic pseudo-answer + synthetic-but-plausible timing.

        The 'answer' is the first sentence of the retrieved context, so EM/F1
        against gold behave sensibly enough to exercise the scoring code.
        """
        ctx = ""
        m = re.search(r"Context:\n(.*?)\n\nQuestion:", prompt, re.S)
        if m:
            ctx = m.group(1)
        answer = re.split(r"(?<=[.!?])\s+", ctx.strip())[0][:200] if ctx else "unknown"
        n_tokens = min(max_tokens, max(8, _count_tokens(answer)))
        # Synthetic timing model (ms): stamped deterministically from prompt len.
        base = 40 + (len(prompt) % 60)
        t_prefill = float(base)
        t_decode = 8.0
        e2e = t_prefill + n_tokens * t_decode
        return GenResult(
            text=answer, ttft_ms=t_prefill, e2e_ms=e2e,
            t_prefill_ms=t_prefill, t_decode_ms=t_decode, n_tokens=n_tokens,
            backend=self.backend, is_mock=True,
        )


def _count_tokens(text: str) -> int:
    return max(1, len(re.findall(r"\S+", text)))


def _clean_answer(text: str, prompt: str) -> str:
    """Strip any echoed prompt and trailing chatter from CLI output."""
    if "Answer:" in text:
        text = text.split("Answer:")[-1]
    # llama-cli sometimes echoes the full prompt; drop it if present.
    if prompt[-60:] in text:
        text = text.split(prompt[-60:])[-1]
    return text.strip()


if __name__ == "__main__":
    g = Generator()
    r = g.generate("Context:\nParis is the capital of France.\n\n"
                   "Question: What is the capital of France?\nAnswer:")
    print(f"backend={g.backend} mock={g.is_mock}")
    print(f"answer={r.text!r}")
    print(f"ttft={r.ttft_ms:.1f}ms e2e={r.e2e_ms:.1f}ms "
          f"prefill={r.t_prefill_ms:.1f} decode/tok={r.t_decode_ms:.2f} "
          f"ntok={r.n_tokens}")
