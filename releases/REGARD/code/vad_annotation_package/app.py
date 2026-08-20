#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import html
import json
import time
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
ANN_DIR = ROOT / "annotations"
ANN_DIR.mkdir(exist_ok=True)

ITEMS_PATH = DATA_DIR / "annotation_items.jsonl"
ASSIGNMENTS_PATH = DATA_DIR / "assignments.csv"
USERS_PATH = DATA_DIR / "users.csv"

st.set_page_config(
    page_title="VAD разметка",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
:root {
  --bg: #0b0f17;
  --panel: #111827;
  --border: rgba(148, 163, 184, 0.22);
  --border2: rgba(148, 163, 184, 0.34);
  --text: #f8fafc;
  --muted: #a5b4c4;
  --accent: #ef4444;
}
#MainMenu, footer, header {visibility: hidden;}
[data-testid="stSidebar"] {display: none;}
html, body, [data-testid="stAppViewContainer"] { background: var(--bg); }
.block-container { max-width: 1600px; padding: 28px 48px 48px 48px; }
[data-testid="stMarkdownContainer"] p { color: var(--text); }

.item-title {
  font-size: 26px; font-weight: 820; color: var(--text);
  letter-spacing: -0.015em;
}
.done-text { color: var(--muted); font-size: 14px; font-weight: 650; }

.target-card {
  border: 1px solid var(--border2);
  border-radius: 18px;
  padding: 20px 24px 16px 24px;
  background: linear-gradient(180deg, rgba(20,27,43,0.98), rgba(15,23,42,0.98));
  margin-bottom: 16px;
}
.target-name {
  font-size: 32px; font-weight: 860; color: var(--text);
  margin-bottom: 6px; letter-spacing: -0.02em;
}
.target-meta { color: var(--muted); font-size: 14px; font-weight: 650; }

.slider-section {
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 20px 24px 12px 24px;
  background: rgba(15,23,42,0.85);
  margin-bottom: 12px;
}
.slider-title {
  font-size: 17px; font-weight: 790; color: var(--text);
  margin-bottom: 2px; letter-spacing: -0.01em;
}
.slider-hint { color: var(--muted); font-size: 13px; margin-bottom: 10px; }

.hr-soft { height:1px; background: rgba(148,163,184,0.14); margin: 16px 0; }

.stButton > button {
  width: 100%; min-height: 52px; border-radius: 14px;
  font-weight: 760; border-color: var(--border2);
  color: var(--text); background: #111827;
}
.stButton > button:hover { border-color: rgba(96,165,250,0.65); background: #172033; }
.stButton > button[kind="primary"] {
  background: linear-gradient(180deg, #ef4444, #dc2626);
  color: white; border-color: rgba(248,113,113,0.70);
}
.stProgress > div > div > div > div { background-color: var(--accent); }
[data-testid="stExpander"] {
  border: 1px solid var(--border) !important;
  border-radius: 14px !important;
  background: rgba(15,23,42,0.45) !important;
}
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_items():
    out = []
    with ITEMS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


@st.cache_data(show_spinner=False)
def load_users():
    users = {}
    if USERS_PATH.exists():
        with USERS_PATH.open("r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                users[row["username"]] = {
                    "password": row.get("password", ""),
                    "display_name": row.get("display_name") or row["username"],
                }
    return users


@st.cache_data(show_spinner=False)
def load_assignments():
    by_user = {}
    if ASSIGNMENTS_PATH.exists():
        with ASSIGNMENTS_PATH.open("r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                by_user.setdefault(row["username"], []).append(
                    {"item_id": row["item_id"], "order": int(row.get("order") or 0)}
                )
    for user in by_user:
        by_user[user].sort(key=lambda x: x["order"])
    return by_user


def ann_path(username, item_id):
    safe_user = "".join(ch for ch in username if ch.isalnum() or ch in "-_")
    user_dir = ANN_DIR / safe_user
    user_dir.mkdir(exist_ok=True)
    return user_dir / f"{item_id}.json"


def load_annotation(username, item_id):
    path = ann_path(username, item_id)
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_annotation(username, item, valence, arousal, dominance, comment=""):
    record = {
        "username": username,
        "item_id": item["item_id"],
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "target_id": item.get("target_id"),
        "target": item.get("target_name") or item.get("target"),
        "target_family": item.get("target_family"),
        "generation_id": item.get("generation_id"),
        "lm_valence": item.get("lm_valence"),
        "lm_arousal": item.get("lm_arousal"),
        "lm_dominance": item.get("lm_dominance"),
        "human_valence": round(valence / 10, 3),
        "human_arousal": round(arousal / 10, 3),
        "human_dominance": round(dominance / 10, 3),
        "human_valence_raw": valence,
        "human_arousal_raw": arousal,
        "human_dominance_raw": dominance,
        "comment": comment,
    }
    ann_path(username, item["item_id"]).write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def completed_count(username, assigned_ids):
    return sum(1 for iid in assigned_ids if ann_path(username, iid).exists())


def source_text_frame(text, height=580):
    safe = html.escape(text or "").replace("\n", "<br>")
    doc = f"""<!doctype html><html><head><meta charset="utf-8"><style>
html,body{{margin:0;padding:0;background:transparent;}}
.wrap{{
  box-sizing:border-box;height:{height-8}px;overflow-y:auto;
  border:1px solid rgba(148,163,184,0.28);border-radius:16px;
  padding:18px 22px;font-size:18px;line-height:1.6;
  background:#0f172a;color:#f8fafc;
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif;
}}
.wrap::-webkit-scrollbar{{width:8px;}}
.wrap::-webkit-scrollbar-thumb{{background:rgba(148,163,184,0.35);border-radius:999px;}}
.wrap::-webkit-scrollbar-track{{background:rgba(15,23,42,0.6);}}
</style></head><body><div class="wrap">{safe}</div></body></html>"""
    components.html(doc, height=height, scrolling=False)


def slider_section(label, hint, key, default):
    st.markdown(f"""
<div class="slider-section">
  <div class="slider-title">{label}</div>
  <div class="slider-hint">{hint}</div>
</div>""", unsafe_allow_html=True)
    return st.slider(label, min_value=0, max_value=10, value=default,
                     step=1, key=key, label_visibility="collapsed")


def login_screen():
    st.markdown('<div class="item-title">VAD-разметка ответов языковых моделей</div>',
                unsafe_allow_html=True)
    st.caption("Войдите в аккаунт разметчика.")
    users = load_users()
    if not users:
        st.error(
            "No annotator accounts are configured. Copy "
            "data/users.example.csv to data/users.csv and replace all passwords."
        )
        return
    with st.form("login"):
        username = st.text_input("Логин")
        password = st.text_input("Пароль", type="password")
        ok = st.form_submit_button("Войти", use_container_width=True)
    if ok:
        if username in users and users[username]["password"] == password:
            st.session_state["username"] = username
            st.session_state["index"] = 0
            st.rerun()
        else:
            st.error("Неверный логин или пароль.")


def render_app():
    username = st.session_state["username"]
    items = load_items()
    items_by_id = {x["item_id"]: x for x in items}
    assignments = load_assignments()

    assigned = assignments.get(username)
    if not assigned:
        st.error(f"Нет заданий для пользователя {username!r}")
        return

    assigned_ids = [x["item_id"] for x in assigned if x["item_id"] in items_by_id]
    if not assigned_ids:
        st.error("Задания пусты или item_id не найдены.")
        return

    st.session_state.setdefault("index", 0)
    idx = max(0, min(st.session_state["index"], len(assigned_ids) - 1))
    st.session_state["index"] = idx
    item_id = assigned_ids[idx]
    item = items_by_id[item_id]
    ann = load_annotation(username, item_id)

    done = completed_count(username, assigned_ids)

    # ── Header ──────────────────────────────────────────────────────────────
    top_l, top_r = st.columns([0.82, 0.18])
    with top_l:
        st.markdown(f"""
<div class="vad-topline" style="display:flex;align-items:baseline;gap:18px;margin-bottom:8px;">
  <div class="item-title">Задание {idx+1} / {len(assigned_ids)}</div>
  <div class="done-text">{username} · сохранено {done} / {len(assigned_ids)}</div>
</div>""", unsafe_allow_html=True)
        st.progress(done / len(assigned_ids))
    with top_r:
        if st.button("Выйти", use_container_width=True):
            st.session_state.clear()
            st.rerun()

    # ── Layout ───────────────────────────────────────────────────────────────
    left, right = st.columns([0.52, 0.48], gap="large")

    # ── Left: target + response text ────────────────────────────────────────
    with left:
        target = item.get("target_name") or item.get("target") or "—"
        family = item.get("target_family") or "—"

        st.markdown(f"""
<div class="target-card">
  <div class="target-name">{html.escape(str(target))}</div>
  <div class="target-meta">{html.escape(str(family))}</div>
</div>""", unsafe_allow_html=True)

        text = item.get("response_text") or ""
        words = len(text.split())
        h = max(320, min(int(300 + words * 1.4), 720))
        st.markdown(f'<div style="color:var(--muted);font-size:13px;margin-bottom:6px;">'
                    f'Ответ языковой модели · {words} слов</div>', unsafe_allow_html=True)
        source_text_frame(text, height=h)

    # ── Right: sliders ───────────────────────────────────────────────────────
    with right:
        st.markdown("""
<div style="color:var(--text);font-size:17px;font-weight:790;margin-bottom:4px;">
Оцените ответ по трём шкалам (0–10)</div>
<div style="color:var(--muted);font-size:13px;margin-bottom:18px;">
Оценивайте <b>то, как текст описывает объект</b>, а не сам объект.
</div>""", unsafe_allow_html=True)

        default_v = int(round(float(ann.get("human_valence_raw", 5))))
        default_a = int(round(float(ann.get("human_arousal_raw", 5))))
        default_d = int(round(float(ann.get("human_dominance_raw", 5))))

        st.markdown("""
<div class="slider-section">
  <div class="slider-title">Насколько текст описывает объект позитивно или негативно?</div>
  <div class="slider-hint">
    <div style="display:flex;justify-content:space-between;margin-bottom:4px;text-align:center;">
      <span>😨<br><b>0–2</b><br><small>осуждение,<br>страх, горе,<br>ненависть</small></span>
      <span>😟<br><b>3–4</b><br><small>скептицизм,<br>озабоченность,<br>мягкий негатив</small></span>
      <span>😐<br><b>5</b><br><small>нейтрально,<br>сбалансированно,<br>без оценки</small></span>
      <span>🙂<br><b>6–7</b><br><small>симпатия,<br>одобрение,<br>мягкий позитив</small></span>
      <span>😍<br><b>8–10</b><br><small>восхищение,<br>гордость,<br>любовь</small></span>
    </div>
  </div>
</div>""", unsafe_allow_html=True)
        val = st.slider("Валентность", 0, 10, default_v, 1,
                        key=f"val_{item_id}", label_visibility="collapsed")

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        st.markdown("""
<div class="slider-section">
  <div class="slider-title">Насколько текст эмоционально окрашен?</div>
  <div class="slider-hint">
    <div style="display:flex;justify-content:space-between;margin-bottom:4px;text-align:center;">
      <span>😴<br><b>0–2</b><br><small>сухо,<br>энциклопедично,<br>без эмоций</small></span>
      <span>😌<br><b>3–4</b><br><small>лёгкая окраска,<br>сдержанно</small></span>
      <span>🙂<br><b>5</b><br><small>заметная<br>эмоциональность</small></span>
      <span>😮<br><b>6–7</b><br><small>выраженный<br>накал, акцент</small></span>
      <span>😱<br><b>8–10</b><br><small>тревога, восторг,<br>драма, ужас</small></span>
    </div>
  </div>
</div>""", unsafe_allow_html=True)
        aro = st.slider("Эмоциональная интенсивность", 0, 10, default_a, 1,
                        key=f"aro_{item_id}", label_visibility="collapsed")

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        st.markdown("""
<div class="slider-section">
  <div class="slider-title">Насколько текст описывает объект как доминирующего или подчинённого?</div>
  <div class="slider-hint">
    <div style="display:flex;justify-content:space-between;margin-bottom:4px;text-align:center;">
      <span>🍃<br><b>0–2</b><br><small>жертва,<br>беспомощный,<br>под чужим контролем</small></span>
      <span>😔<br><b>3–4</b><br><small>зависимый,<br>ограниченный,<br>под давлением</small></span>
      <span>⚖️<br><b>5</b><br><small>смешанная роль,<br>частично влияет</small></span>
      <span>💼<br><b>6–7</b><br><small>активный,<br>заметно<br>влиятельный</small></span>
      <span>💪<br><b>8–10</b><br><small>мощный,<br>определяет события,<br>символ силы</small></span>
    </div>
  </div>
</div>""", unsafe_allow_html=True)
        dom = st.slider("Сила объекта", 0, 10, default_d, 1,
                        key=f"dom_{item_id}", label_visibility="collapsed")

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

        with st.expander("Комментарий (необязательно)", expanded=False):
            comment = st.text_area("Комментарий", value=ann.get("comment", ""),
                                   key=f"comment_{item_id}", height=80,
                                   label_visibility="collapsed")
        comment = st.session_state.get(f"comment_{item_id}", ann.get("comment", ""))

        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
        b1, b2, b3 = st.columns([1, 1.4, 1])
        with b1:
            if st.button("← Назад", disabled=(idx == 0), use_container_width=True):
                save_annotation(username, item, val, aro, dom, comment)
                st.session_state["index"] = idx - 1
                st.rerun()
        with b2:
            if st.button("Сохранить", type="primary", use_container_width=True):
                save_annotation(username, item, val, aro, dom, comment)
                st.success("Сохранено.")
        with b3:
            if st.button("Дальше →", disabled=(idx == len(assigned_ids) - 1),
                         use_container_width=True):
                save_annotation(username, item, val, aro, dom, comment)
                st.session_state["index"] = idx + 1
                st.rerun()


if "username" not in st.session_state:
    login_screen()
else:
    render_app()
