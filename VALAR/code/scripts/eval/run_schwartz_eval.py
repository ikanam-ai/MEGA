from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import sys
import time
import traceback
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import httpx
except ImportError:
    sys.exit("[ERR] httpx not installed. Run: pip install httpx")

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    _HAS_PLOTLY = True
except ImportError:
    _HAS_PLOTLY = False
    print("[WARN] plotly not installed — windrose charts will be skipped")


SCHWARTZ_10 = [
    "Self-Direction", "Stimulation", "Hedonism", "Achievement", "Power",
    "Security", "Conformity", "Tradition", "Benevolence", "Universalism",
]
LABEL_SET = ", ".join(SCHWARTZ_10)

SHORT = {
    "Self-Direction": "SD", "Stimulation": "ST", "Hedonism": "HE",
    "Achievement": "AC", "Power": "PO", "Security": "SE",
    "Conformity": "CO", "Tradition": "TR", "Benevolence": "BE",
    "Universalism": "UN",
}

_ALIASES: dict[str, str] = {}
for _v in SCHWARTZ_10:
    for _a in (_v, _v.lower(), _v.replace("-", " "), _v.replace("-", ""), _v.replace("-", "_")):
        _ALIASES[re.sub(r"[^a-z0-9]+", "", _a.lower())] = _v
_ALIASES.update({
    "selfdirection": "Self-Direction", "selfdirectionvalue": "Self-Direction",
    "stim": "Stimulation",
    "achievementvalue": "Achievement",
    "powerdominance": "Power",
    "safetysecurity": "Security",
    "obedience": "Conformity",
    "religiontradition": "Tradition",
    "benevolent": "Benevolence",
    "universalist": "Universalism",
})

HERE = Path(__file__).parent
REPO_ROOT = HERE.parent.parent
GOLD_BANK  = REPO_ROOT / "data" / "item_banks" / "gold" / "schwartz10_ru_1000.jsonl"
L0_BANK    = REPO_ROOT / "data" / "item_banks" / "l0_h0" / "h0_all_instances.jsonl"


def canonicalize(raw: Any) -> str | None:
    key = re.sub(r"[^a-z0-9]+", "", str(raw or "").lower())
    return _ALIASES.get(key)


def gold_prompt(item_text: str) -> str:
    return (
        "Определите, какую базовую ценность Шварца наиболее явно выражает следующий текст.\n\n"
        f"Выберите три наиболее подходящие ценности из этого набора:\n{LABEL_SET}\n\n"
        f"Текст:\n{item_text}\n\n"
        "Верните только один корректный JSON-объект.\n"
        "Не используйте markdown. Не заворачивайте JSON в блоки кода.\n"
        "Не добавляйте объяснений вне JSON.\n"
        "Значения top1, top2, top3 должны быть скопированы точно из набора выше.\n"
        "Обоснование — одна строка, не более 25 слов.\n\n"
        "Обязательная схема:\n"
        '{\n  "top1": "точная метка из набора",\n'
        '  "top2": "точная метка из набора",\n'
        '  "top3": "точная метка из набора",\n'
        '  "rationale": "краткое обоснование"\n}'
    )


def _call_api(
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict],
    max_tokens: int,
    temperature: float,
    timeout_sec: int,
    use_thinking_kwargs: bool,
) -> tuple[str, dict]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if use_thinking_kwargs:
        payload["chat_template_kwargs"] = {"enable_thinking": False}

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    with httpx.Client(timeout=timeout_sec) as client:
        resp = client.post(
            f"{base_url.rstrip('/')}/chat/completions",
            json=payload,
            headers=headers,
        )
    resp.raise_for_status()
    data = resp.json()
    content = data["choices"][0]["message"].get("content") or ""
    return content, data.get("usage") or {}


def _call_one_safe(call_fn, *args, **kwargs) -> tuple[str, dict, str | None]:
    try:
        content, usage = call_fn(*args, **kwargs)
        return content, usage, None
    except Exception as exc:
        return "", {}, repr(exc)


def _strip_thinking(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()


def parse_json(text: str) -> dict | None:
    text = _strip_thinking(text)
    m = re.search(r"\{.*\}", text, re.DOTALL)
    candidate = m.group() if m else None
    if candidate is None:
        # Try recovering truncated JSON missing the closing brace
        m2 = re.search(r"\{.*", text, re.DOTALL)
        candidate = (m2.group() + "}") if m2 else None
    if candidate is None:
        return None
    for attempt in (candidate, re.sub(r"[\x00-\x1f\x7f]", " ", candidate)):
        try:
            return json.loads(attempt)
        except json.JSONDecodeError:
            pass
    return None


_DEFAULT_SYSTEM_PROMPT = (
    "Вы — ассистент по классификации ценностей. "
    "Анализируйте тексты строго по схеме Шварца. "
    "Всегда возвращайте только JSON."
)


def _build_messages_for_gold(
    item_text: str,
    system_prompt: str | None = None,
    no_system_role: bool = False,
) -> list[dict]:
    sys_content = system_prompt or _DEFAULT_SYSTEM_PROMPT
    if no_system_role:
        return [{"role": "user", "content": sys_content + "\n\n" + gold_prompt(item_text)}]
    return [
        {"role": "system", "content": sys_content},
        {"role": "user", "content": gold_prompt(item_text)},
    ]


def _run_batch(
    *,
    items: list[dict],
    item_to_messages,
    stable_id_fn,
    args: argparse.Namespace,
    done_ids: set[str],
    bank_name: str,
) -> list[dict]:
    todo = [it for it in items if stable_id_fn(it) not in done_ids]
    print(f"[{bank_name}] planned={len(todo)}  already_done={len(done_ids)}")

    results: list[dict] = []

    def _work(idx: int, item: dict) -> dict:
        t0 = time.monotonic()
        sid = stable_id_fn(item)
        messages = item_to_messages(item)
        content, usage, err = _call_one_safe(
            _call_api,
            base_url=args.api_base,
            api_key=args.api_key,
            model=args.model,
            messages=messages,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            timeout_sec=args.timeout_sec,
            use_thinking_kwargs=args.use_thinking_kwargs,
        )
        latency = round(time.monotonic() - t0, 4)
        row = {
            "item_id":  sid,
            "model":    args.model,
            "content":  content,
            "usage":    usage,
            "latency":  latency,
            "api_ok":   err is None,
        }
        if err:
            row["error"] = err
        for k in ("task_family", "gold_value", "gold_value_a", "gold_value_b",
                  "gold_pair_unordered", "candidate_value", "answer",
                  "item_text", "scenario_text", "source_dataset"):
            if k in item:
                row[k] = item[k]
        return row

    def _run_pass(pass_items: list[dict], pass_label: str) -> list[dict]:
        pass_results: list[dict] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallelism) as pool:
            futures = {pool.submit(_work, i, it): it for i, it in enumerate(pass_items, 1)}
            completed = 0
            for future in concurrent.futures.as_completed(futures):
                completed += 1
                row = future.result()
                pass_results.append(row)
                status = "OK" if row["api_ok"] else "ERR"
                print(
                    f"[{bank_name}]{pass_label}[{status}] {completed}/{len(pass_items)} "
                    f"id={row['item_id']} lat={row['latency']}s "
                    f"preview={row['content'][:80].replace(chr(10), ' ')}"
                )
        return pass_results

    results = _run_pass(todo, "")

    failed_ids = {r["item_id"] for r in results if not r["api_ok"]}
    if failed_ids:
        retry_items = [it for it in todo if stable_id_fn(it) in failed_ids]
        print(f"[{bank_name}] retrying {len(retry_items)} failed item(s) …")
        time.sleep(2)
        retry_results = _run_pass(retry_items, "[retry]")
        retry_map = {r["item_id"]: r for r in retry_results}
        results = [retry_map.get(r["item_id"], r) for r in results]

    return results


def _parse_gold_row(row: dict) -> dict:
    obj = parse_json(row.get("content", ""))
    top1 = top2 = top3 = None
    parse_ok = False
    if obj:
        top1 = canonicalize(obj.get("top1"))
        top2 = canonicalize(obj.get("top2"))
        top3 = canonicalize(obj.get("top3"))
        parse_ok = top1 is not None
    return {**row, "top1": top1, "top2": top2, "top3": top3, "parse_ok": parse_ok}


def score_gold(rows: list[dict]) -> dict:
    scored = [_parse_gold_row(r) for r in rows]
    n = len(scored)
    parse_ok = sum(1 for r in scored if r["parse_ok"])

    per_value: dict[str, dict] = {v: {"tp1": 0, "tp3": 0, "total": 0} for v in SCHWARTZ_10}
    for r in scored:
        gv = r.get("gold_value")
        if gv not in per_value:
            continue
        per_value[gv]["total"] += 1
        if r["top1"] == gv:
            per_value[gv]["tp1"] += 1
        if gv in (r["top1"], r["top2"], r["top3"]):
            per_value[gv]["tp3"] += 1

    recall1 = {v: s["tp1"] / s["total"] if s["total"] else 0.0 for v, s in per_value.items()}
    recall3 = {v: s["tp3"] / s["total"] if s["total"] else 0.0 for v, s in per_value.items()}
    macro_acc1 = sum(recall1.values()) / len(SCHWARTZ_10)
    macro_acc3 = sum(recall3.values()) / len(SCHWARTZ_10)
    micro_acc1 = sum(s["tp1"] for s in per_value.values()) / max(n, 1)

    return {
        "n_items":      n,
        "parse_ok":     parse_ok,
        "parse_rate":   round(parse_ok / max(n, 1), 4),
        "micro_acc1":   round(micro_acc1, 4),
        "macro_acc1":   round(macro_acc1, 4),
        "macro_acc3":   round(macro_acc3, 4),
        "recall1_per_value": {v: round(recall1[v], 4) for v in SCHWARTZ_10},
        "recall3_per_value": {v: round(recall3[v], 4) for v in SCHWARTZ_10},
        "per_value_counts": per_value,
        "scored_rows":  scored,
    }


def _parse_l0_row(row: dict) -> dict:
    tf = row.get("task_family", "")
    obj = parse_json(row.get("content", ""))
    parse_ok = False
    hits1 = hits3 = None
    pair_match = None
    relevance_correct = None

    if obj and tf == "h0_item_to_value":
        top1 = canonicalize(obj.get("top1"))
        top2 = canonicalize(obj.get("top2"))
        top3 = canonicalize(obj.get("top3"))
        gv = row.get("gold_value")
        parse_ok = top1 is not None
        hits1 = int(top1 == gv)
        hits3 = int(gv in (top1, top2, top3))

    elif obj and tf == "h0_conflict_recognition":
        va = canonicalize(obj.get("value_a"))
        vb = canonicalize(obj.get("value_b"))
        gold_pair = row.get("gold_pair_unordered", "")
        parse_ok = va is not None and vb is not None
        predicted_pair = "|".join(sorted([str(va), str(vb)]))
        gold_sorted = "|".join(sorted(gold_pair.split("|")))
        pair_match = int(predicted_pair == gold_sorted)

    elif obj and tf == "h0_contextual_relevance":
        ans = str(obj.get("answer", "")).strip().lower()
        parse_ok = ans in ("yes", "no")
        gold_ans = str(row.get("answer", "")).strip().lower()
        relevance_correct = int(ans == gold_ans)

    return {
        **row,
        "parse_ok": parse_ok,
        "hits1": hits1,
        "hits3": hits3,
        "pair_match": pair_match,
        "relevance_correct": relevance_correct,
    }


def score_l0(rows: list[dict]) -> dict:
    scored = [_parse_l0_row(r) for r in rows]
    n = len(scored)

    itv  = [r for r in scored if r.get("task_family") == "h0_item_to_value"]
    conf = [r for r in scored if r.get("task_family") == "h0_conflict_recognition"]
    relv = [r for r in scored if r.get("task_family") == "h0_contextual_relevance"]

    def _safe(rows_, field):
        vals = [r[field] for r in rows_ if r.get(field) is not None]
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    pv_hits1: dict[str, dict] = {v: {"tp": 0, "total": 0} for v in SCHWARTZ_10}
    for r in itv:
        gv = r.get("gold_value")
        if gv in pv_hits1:
            pv_hits1[gv]["total"] += 1
            if r.get("hits1"):
                pv_hits1[gv]["tp"] += 1
    recall1_pv = {v: s["tp"] / s["total"] if s["total"] else 0.0 for v, s in pv_hits1.items()}

    return {
        "n_items": n,
        "parse_rate_itv":  _safe(itv, "parse_ok"),
        "parse_rate_conf": _safe(conf, "parse_ok"),
        "parse_rate_relv": _safe(relv, "parse_ok"),
        "h0_hits_at_1":   _safe(itv, "hits1"),
        "h0_hits_at_3":   _safe(itv, "hits3"),
        "h0_pair_match":  _safe(conf, "pair_match"),
        "h0_relv_acc":    _safe(relv, "relevance_correct"),
        "recall1_per_value": {v: round(recall1_pv[v], 4) for v in SCHWARTZ_10},
        "scored_rows": scored,
    }


def _windrose(recall_per_value: dict[str, float], title: str):
    if not _HAS_PLOTLY:
        return None
    circle = SCHWARTZ_10 + [SCHWARTZ_10[0]]
    r_vals = [recall_per_value.get(v, 0.0) for v in circle]
    theta  = [SHORT.get(v, v) for v in circle]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=r_vals, theta=theta,
        fill="toself", mode="lines+markers",
        line={"color": "#1f77b4", "width": 2},
        marker={"size": 6},
        name="recall@1",
    ))
    fig.update_layout(
        title=title,
        polar={"radialaxis": {"visible": True, "range": [0, 1]}},
        template="plotly_white",
        height=500, width=500,
    )
    return fig


def _combined_windrose(gold_recall: dict[str, float], l0_recall: dict[str, float], model: str):
    if not _HAS_PLOTLY:
        return None
    circle = SCHWARTZ_10 + [SCHWARTZ_10[0]]
    theta  = [SHORT.get(v, v) for v in circle]
    fig = go.Figure()
    for label, recall, color in [
        ("gold-1000 (RU)", gold_recall, "#1f77b4"),
        ("L0-h0 (EN)",     l0_recall,   "#ff7f0e"),
    ]:
        fig.add_trace(go.Scatterpolar(
            r=[recall.get(v, 0.0) for v in circle],
            theta=theta,
            fill="toself", mode="lines+markers",
            line={"color": color, "width": 2},
            marker={"size": 6},
            name=label,
        ))
    fig.update_layout(
        title=f"Schwartz-10 recognition — {model}",
        polar={"radialaxis": {"visible": True, "range": [0, 1]}},
        template="plotly_white",
        height=560, width=560,
        legend={"orientation": "h", "y": -0.15},
    )
    return fig


def _log_clearml(
    *,
    task,
    model: str,
    gold_scores: dict,
    l0_scores: dict,
    run_dir: Path,
    gold_fig,
    l0_fig,
    combined_fig,
) -> None:
    logger = task.get_logger()

    for k, v in gold_scores.items():
        if isinstance(v, (int, float)) and k != "per_value_counts":
            logger.report_scalar("gold1000", k, v, iteration=0)
    for k, v in l0_scores.items():
        if isinstance(v, (int, float)):
            logger.report_scalar("l0_h0", k, v, iteration=0)

    for val, rec in gold_scores.get("recall1_per_value", {}).items():
        logger.report_scalar("gold1000/recall1", SHORT.get(val, val), rec, iteration=0)
    for val, rec in l0_scores.get("recall1_per_value", {}).items():
        logger.report_scalar("l0_h0/recall1", SHORT.get(val, val), rec, iteration=0)

    for title, fig in [
        ("Windrose — gold-1000 (RU)",  gold_fig),
        ("Windrose — L0-h0 (EN)",      l0_fig),
        ("Windrose — combined",         combined_fig),
    ]:
        if fig is not None:
            logger.report_plotly(title=title, series=model, figure=fig, iteration=0)

    for name, path in [
        ("gold1000/generation_rows",  run_dir / "gold1000" / "generation_rows.jsonl"),
        ("gold1000/scores",           run_dir / "gold1000" / "scores.json"),
        ("l0_h0/generation_rows",     run_dir / "l0_h0"    / "generation_rows.jsonl"),
        ("l0_h0/scores",              run_dir / "l0_h0"    / "scores.json"),
        ("combined_summary",          run_dir / "combined_summary.json"),
    ]:
        if path.exists():
            task.upload_artifact(name, artifact_object=str(path))


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
        encoding="utf-8",
    )


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _safe_model_id(name: str) -> str:
    out = re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_")
    return out or "model"


def _utc_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _load_done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {r["item_id"] for r in _load_jsonl(path) if r.get("api_ok")}


def _ensure_clearml_project(full_name: str) -> None:
    if "/" not in full_name:
        return
    try:
        from clearml import Task
        from clearml.backend_api.services import projects as proj_svc

        session = Task._get_default_session()

        parts = full_name.split("/")
        parent_id: str | None = None

        for depth, part in enumerate(parts):
            path_so_far = "/".join(parts[: depth + 1])
            res = session.send(
                proj_svc.GetAllRequest(
                    name=f"^{re.escape(path_so_far)}$",
                    search_hidden=False,
                    only_fields=["id", "name"],
                )
            )
            found = res.response.projects if res and res.response else []
            match = next((p for p in found if p.name == path_so_far), None)
            if match:
                parent_id = match.id
            else:
                create_kwargs: dict = {"name": path_so_far}
                if parent_id:
                    create_kwargs["parent"] = parent_id
                cr = session.send(proj_svc.CreateRequest(**create_kwargs))
                parent_id = cr.response.id if cr and cr.response else None

        print(f"[INFO] ClearML project hierarchy ensured: {full_name}")
    except Exception as exc:
        print(f"[WARN] _ensure_clearml_project failed ({exc}) — Task.init may still work")


def _smoke(args: argparse.Namespace) -> None:
    print("[SMOKE] calling model with minimal prompt …")
    content, usage = _call_api(
        base_url=args.api_base,
        api_key=args.api_key,
        model=args.model,
        messages=[{"role": "user", "content": 'Return exactly: {"ok": true}'}],
        max_tokens=32,
        temperature=0.0,
        timeout_sec=args.timeout_sec,
        use_thinking_kwargs=args.use_thinking_kwargs,
    )
    print(f"[SMOKE] response: {content[:200].replace(chr(10), ' ')}")
    if "thinking process" in content.lower():
        raise SystemExit("[ERR] smoke response contains thinking preamble — use --thinking never")
    print("[SMOKE] OK")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="VALAR Schwartz-10 eval: gold-1000 (RU) + L0-h0 (EN)"
    )
    p.add_argument("--model",       required=True,  help="Model name served by the endpoint")
    p.add_argument("--api-base",    required=True,  help="Base URL of the OpenAI-compatible endpoint")
    p.add_argument("--api-key",     default=os.environ.get("VALAR_EVAL_API_KEY", "dummy"))
    p.add_argument("--parallelism", type=int, default=int(os.environ.get("VALAR_PARALLELISM", "50")))
    p.add_argument("--max-tokens",  type=int, default=int(os.environ.get("VALAR_MAX_TOKENS",  "256")))
    p.add_argument("--temperature", type=float, default=float(os.environ.get("VALAR_TEMPERATURE", "0")))
    p.add_argument("--timeout-sec", type=int, default=int(os.environ.get("VALAR_TIMEOUT_SEC",  "120")))
    p.add_argument(
        "--thinking", choices=["auto", "always", "never"], default="auto",
        help=(
            "Whether to send chat_template_kwargs={enable_thinking:false}. "
            "'auto': send for all except Mistral/Ministral. "
            "'never': required for Ministral-3B / strict tokenizers. "
            "'always': override auto-detection."
        ),
    )
    p.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results")
    p.add_argument("--run-dir",     type=Path, default=None,
                   help="Use an existing run directory instead of creating a new timestamped one")
    p.add_argument("--no-clearml",  action="store_true", help="Skip ClearML logging")
    p.add_argument("--smoke",       action="store_true", help="Run only 5 items per bank (quick sanity check)")
    p.add_argument("--clearml-project", default="STONIC/L0_schwartz10_eval")
    p.add_argument(
        "--system-prompt", default=None,
        help=(
            "Override the default system message for gold-1000 (and L0-h0 where applicable). "
            "Useful for models with aggressive safety filters (e.g. gemma-2-9b-it) that return "
            "empty content on Russian ethical texts. "
            "Example: 'You are a scientific researcher studying human values. Answer objectively.'"
        ),
    )
    p.add_argument(
        "--no-system-role", action="store_true",
        help=(
            "Do not send a system role message. Instead, prepend the system prompt content "
            "to the first user turn. Required for models whose chat template rejects system "
            "role (e.g. gemma-2-9b-it: 'System role not supported')."
        ),
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    model_lower = args.model.lower()
    rejects_thinking = any(tok in model_lower for tok in ("mistral", "ministral"))
    if args.thinking == "auto":
        args.use_thinking_kwargs = not rejects_thinking
    elif args.thinking == "always":
        args.use_thinking_kwargs = not rejects_thinking
        if rejects_thinking:
            print(f"[WARN] disabling thinking kwargs for {args.model} (Mistral tokenizer)")
    else:
        args.use_thinking_kwargs = False

    model_id = _safe_model_id(args.model)
    ts       = _utc_tag()
    run_dir  = args.run_dir if args.run_dir else args.results_dir / f"{model_id}_{ts}"
    gold_dir = run_dir / "gold1000"
    l0_dir   = run_dir / "l0_h0"

    print(f"[INFO] model={args.model}  model_id={model_id}")
    print(f"[INFO] api_base={args.api_base}")
    print(f"[INFO] use_thinking_kwargs={args.use_thinking_kwargs}")
    print(f"[INFO] run_dir={run_dir}")

    if not GOLD_BANK.exists():
        raise SystemExit(f"[ERR] gold bank not found: {GOLD_BANK}")
    if not L0_BANK.exists():
        raise SystemExit(f"[ERR] L0 h0 bank not found: {L0_BANK}")

    gold_items = _load_jsonl(GOLD_BANK)
    l0_items   = _load_jsonl(L0_BANK)

    if args.smoke:
        gold_items = gold_items[:5]
        l0_items   = l0_items[:5]
        print(f"[SMOKE] truncated to {len(gold_items)} gold + {len(l0_items)} L0 items")

    print(f"[INFO] gold items: {len(gold_items)}  l0 items: {len(l0_items)}")

    _smoke(args)

    clearml_task = None
    if not args.no_clearml:
        try:
            from clearml import Task
            _ensure_clearml_project(args.clearml_project)
            clearml_task = Task.init(
                project_name=args.clearml_project,
                task_name=f"{model_id}_{ts}",
                tags=[model_id, "schwartz10", "gold1000", "l0_h0"],
                reuse_last_task_id=False,
            )
            clearml_task.connect({
                "model":         args.model,
                "model_id":      model_id,
                "api_base":      args.api_base,
                "parallelism":   args.parallelism,
                "temperature":   args.temperature,
                "max_tokens":    args.max_tokens,
                "n_gold":        len(gold_items),
                "n_l0":          len(l0_items),
                "thinking":      args.thinking,
                "run_id":        f"{model_id}_{ts}",
            })
            print("[INFO] ClearML task initialized")
        except Exception as exc:
            print(f"[WARN] ClearML init failed ({exc}) — continuing without logging")
            clearml_task = None

    gold_rows_path = gold_dir / "generation_rows.jsonl"
    done_gold = _load_done_ids(gold_rows_path)

    gold_new = _run_batch(
        items=gold_items,
        item_to_messages=lambda it: _build_messages_for_gold(
            it["item_text"], args.system_prompt, getattr(args, "no_system_role", False)
        ),
        stable_id_fn=lambda it: str(it["item_id"]),
        args=args,
        done_ids=done_gold,
        bank_name="gold1000",
    )

    all_gold = list(_load_jsonl(gold_rows_path)) + gold_new if gold_rows_path.exists() else gold_new
    _write_jsonl(gold_rows_path, all_gold)

    gold_scores = score_gold(all_gold)
    _write_jsonl(gold_dir / "scored_rows.jsonl", gold_scores.pop("scored_rows"))
    _write_json(gold_dir / "scores.json", {k: v for k, v in gold_scores.items() if k != "per_value_counts"})
    print(f"\n[gold1000] micro_acc1={gold_scores['micro_acc1']}  macro_acc1={gold_scores['macro_acc1']}  parse_rate={gold_scores['parse_rate']}")

    l0_rows_path = l0_dir / "generation_rows.jsonl"
    done_l0 = _load_done_ids(l0_rows_path)

    def _l0_messages(item: dict) -> list[dict]:
        return [{"role": "user", "content": item["prompt_text"]}]

    l0_new = _run_batch(
        items=l0_items,
        item_to_messages=_l0_messages,
        stable_id_fn=lambda it: str(it.get("task_id") or it.get("item_id") or it.get("scenario_id")),
        args=args,
        done_ids=done_l0,
        bank_name="l0_h0",
    )

    all_l0 = list(_load_jsonl(l0_rows_path)) + l0_new if l0_rows_path.exists() else l0_new
    _write_jsonl(l0_rows_path, all_l0)

    l0_scores = score_l0(all_l0)
    _write_jsonl(l0_dir / "scored_rows.jsonl", l0_scores.pop("scored_rows"))
    _write_json(l0_dir / "scores.json", {k: v for k, v in l0_scores.items() if k != "per_value_counts"})
    print(f"\n[l0_h0] hits@1={l0_scores['h0_hits_at_1']}  hits@3={l0_scores['h0_hits_at_3']}  pair_match={l0_scores['h0_pair_match']}  relv_acc={l0_scores['h0_relv_acc']}")

    combined = {
        "model": args.model, "model_id": model_id, "run_ts": ts,
        "gold1000": {k: v for k, v in gold_scores.items()},
        "l0_h0":    {k: v for k, v in l0_scores.items()},
    }
    _write_json(run_dir / "combined_summary.json", combined)

    gold_fig     = _windrose(gold_scores["recall1_per_value"], f"Gold-1000 recall@1 — {args.model}")
    l0_fig       = _windrose(l0_scores["recall1_per_value"],   f"L0-h0 recall@1 — {args.model}")
    combined_fig = _combined_windrose(gold_scores["recall1_per_value"], l0_scores["recall1_per_value"], args.model)

    if _HAS_PLOTLY:
        plots_dir = run_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        for name, fig in [("windrose_gold", gold_fig), ("windrose_l0", l0_fig), ("windrose_combined", combined_fig)]:
            if fig is not None:
                fig.write_html(str(plots_dir / f"{name}.html"), include_plotlyjs="cdn")
                (plots_dir / f"{name}.plotly.json").write_text(fig.to_json(), encoding="utf-8")
        print(f"[INFO] plots saved to {plots_dir}")

    if clearml_task is not None:
        try:
            _log_clearml(
                task=clearml_task,
                model=args.model,
                gold_scores=gold_scores,
                l0_scores=l0_scores,
                run_dir=run_dir,
                gold_fig=gold_fig,
                l0_fig=l0_fig,
                combined_fig=combined_fig,
            )
            clearml_task.close()
            print("[INFO] ClearML task closed")
        except Exception as exc:
            print(f"[WARN] ClearML logging failed: {exc}")

    print("\n" + "=" * 60)
    print(f"RESULTS  model={args.model}")
    print(f"  gold-1000  micro_acc1={gold_scores['micro_acc1']:.3f}  macro_acc1={gold_scores['macro_acc1']:.3f}")
    print(f"  L0-h0      hits@1={l0_scores['h0_hits_at_1']:.3f}  pair={l0_scores['h0_pair_match']:.3f}  relv={l0_scores['h0_relv_acc']:.3f}")
    print(f"  run_dir={run_dir}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
