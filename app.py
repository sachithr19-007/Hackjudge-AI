import os
from datetime import datetime, date

import streamlit as st
import pandas as pd

from github_fetcher import fetch_github_repo_data
from summarizer import generate_repo_score
from similarity import compute_file_hashes, hash_overlap_pct, get_embedding, cosine_similarity

# ------------------------------------------------------------------
# Page config
# ------------------------------------------------------------------
st.set_page_config(page_title="HackJudge AI", page_icon="⚡", layout="wide")

# ------------------------------------------------------------------
# Styling (same as before)
# ------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
.stApp { background: radial-gradient(circle at 20% 0%, #1a1033 0%, #0b0d17 45%, #0b0d17 100%); color: #e8e6f0; }
#MainMenu, header, footer {visibility: hidden;}
.hero { text-align: center; padding: 1.2rem 0 0.4rem 0; }
.hero h1 { font-size: 2.6rem; font-weight: 700; background: linear-gradient(90deg, #7c3aed, #06b6d4, #a855f7); background-size: 200% auto; -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0; }
.hero p { color: #9691ad; font-size: 0.95rem; margin-top: 0.2rem; }
.card { background: rgba(255,255,255,0.035); border: 1px solid rgba(124,58,237,0.25); border-radius: 16px; padding: 1.4rem 1.6rem; margin-bottom: 1rem; }
.overall-badge { text-align: center; padding: 1.2rem; }
.overall-badge .num { font-family: 'JetBrains Mono', monospace; font-size: 3.6rem; font-weight: 700; line-height: 1; }
.overall-badge .label { color: #9691ad; font-size: 0.85rem; letter-spacing: 0.15em; text-transform: uppercase; margin-top: 0.3rem; }
.score-row { margin-bottom: 0.9rem; }
.score-row .top-line { display: flex; justify-content: space-between; font-size: 0.88rem; margin-bottom: 0.25rem; }
.score-row .cat-name { font-weight: 600; color: #e8e6f0; }
.score-row .cat-score { font-family: 'JetBrains Mono', monospace; color: #06b6d4; }
.bar-bg { background: rgba(255,255,255,0.06); border-radius: 6px; height: 8px; overflow: hidden; }
.bar-fill { height: 100%; border-radius: 6px; background: linear-gradient(90deg, #7c3aed, #06b6d4); }
.cat-reason { color: #9691ad; font-size: 0.82rem; margin-top: 0.25rem; }
.redflag-box { background: rgba(239,68,68,0.08); border: 1px solid rgba(239,68,68,0.3); border-radius: 12px; padding: 0.9rem 1.1rem; font-size: 0.88rem; color: #fca5a5; margin-bottom: 1rem; }
.verdict-box { background: rgba(6,182,212,0.06); border: 1px solid rgba(6,182,212,0.25); border-radius: 12px; padding: 1rem 1.2rem; font-size: 0.92rem; color: #cbd5e1; }
.similarity-box { background: rgba(234,179,8,0.08); border: 1px solid rgba(234,179,8,0.35); border-radius: 12px; padding: 0.9rem 1.1rem; font-size: 0.88rem; color: #fde68a; margin-bottom: 1rem; }
.lb-row { display: flex; align-items: center; justify-content: space-between; background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.06); border-radius: 12px; padding: 0.8rem 1.2rem; margin-bottom: 0.6rem; }
.lb-rank { font-family: 'JetBrains Mono', monospace; font-size: 1.2rem; font-weight: 700; width: 2.2rem; color: #7c3aed; }
.lb-name { font-weight: 600; flex: 1; }
.lb-score { font-family: 'JetBrains Mono', monospace; font-size: 1.3rem; font-weight: 700; color: #06b6d4; }
div[data-testid="stTextInput"] input { background: rgba(255,255,255,0.04); border: 1px solid rgba(124,58,237,0.3); border-radius: 10px; color: #e8e6f0; }
div.stButton > button { background: linear-gradient(90deg, #7c3aed, #06b6d4); color: white; border: none; border-radius: 10px; font-weight: 600; padding: 0.5rem 1.2rem; }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# Session state
# ------------------------------------------------------------------
if "leaderboard" not in st.session_state:
    st.session_state.leaderboard = []       # list of {name, overall, result, commit_stats, similarity}
if "fingerprints" not in st.session_state:
    st.session_state.fingerprints = []       # list of {name, hashes, embedding}

# ------------------------------------------------------------------
# Header
# ------------------------------------------------------------------
st.markdown("""
<div class="hero">
    <h1>⚡ HackJudge AI</h1>
    <p>Paste a repo. Get a verdict. Catches copy-paste and last-minute dumps too.</p>
</div>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Setup")
    api_key = st.text_input("Gemini API Key", type="password", placeholder="Paste key here")
    if api_key:
        os.environ["GEMINI_API_KEY"] = api_key
        st.success("Key set", icon="✅")

    st.divider()
    st.markdown("### 📅 Hackathon Window")
    hackathon_start = st.date_input("Hackathon start date", value=None)

    st.divider()
    st.caption(f"Repos judged this session: **{len(st.session_state.leaderboard)}**")
    if st.session_state.leaderboard and st.button("Clear session", use_container_width=True):
        st.session_state.leaderboard = []
        st.session_state.fingerprints = []
        st.rerun()

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def check_pre_hackathon_commits(first_commit_date_iso: str, start_date: date) -> str | None:
    """Return a warning string if the repo's first commit predates the hackathon."""
    if not first_commit_date_iso or not start_date:
        return None
    first_commit = datetime.fromisoformat(first_commit_date_iso).date()
    if first_commit < start_date:
        days_early = (start_date - first_commit).days
        return (
            f"First commit is {days_early} day(s) *before* the hackathon start date "
            f"({first_commit} vs {start_date}). This repo may pre-date the event."
        )
    return None


def render_commit_chart(daily_counts: dict):
    if not daily_counts:
        st.caption("No commit date data available.")
        return
    df = pd.DataFrame(
        {"Commits": list(daily_counts.values())},
        index=pd.to_datetime(list(daily_counts.keys())),
    )
    st.bar_chart(df, color="#7c3aed")


def render_scorecard(result: dict, commit_stats: dict, similarity_warnings: list, repo_label: str, key_suffix: str):
    overall = result["overall_score"]
    color = "#22c55e" if overall >= 7 else "#eab308" if overall >= 5 else "#ef4444"

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown(f"""
        <div class="card overall-badge">
            <div class="num" style="color:{color}">{overall}</div>
            <div class="label">Overall / 10</div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("⭐ Add to Leaderboard", use_container_width=True, key=f"add_{key_suffix}"):
            st.session_state.leaderboard.append({
                "name": repo_label, "overall": overall, "result": result,
                "commit_stats": commit_stats, "similarity": similarity_warnings,
            })
            st.toast(f"{repo_label} added to leaderboard!", icon="🏆")

    with col2:
        rows_html = ""
        for category, data in result["scores"].items():
            score = data["score"]
            pct = score * 10
            rows_html += f"""
            <div class="score-row">
                <div class="top-line">
                    <span class="cat-name">{category.replace('_', ' ').title()}</span>
                    <span class="cat-score">{score}/10</span>
                </div>
                <div class="bar-bg"><div class="bar-fill" style="width:{pct}%"></div></div>
                <div class="cat-reason">{data['reason']}</div>
            </div>
            """
        st.markdown(f'<div class="card">{rows_html}</div>', unsafe_allow_html=True)

    # Similarity warnings
    if similarity_warnings:
        sim_lines = "".join(
            f"<div>⚠️ {w['name']} — {w['hash_pct']}% identical files, {w['embed_pct']}% content similarity</div>"
            for w in similarity_warnings
        )
        st.markdown(f'<div class="similarity-box">{sim_lines}</div>', unsafe_allow_html=True)

    # Red flags (LLM + pre-hackathon check merged)
    if result.get("red_flags"):
        flags = "".join(f"<div>🚩 {f}</div>" for f in result["red_flags"])
        st.markdown(f'<div class="redflag-box">{flags}</div>', unsafe_allow_html=True)

    st.markdown(f'<div class="verdict-box"><b>Verdict:</b> {result["verdict"]}</div>', unsafe_allow_html=True)

    st.markdown("##### 📈 Commit activity")
    render_commit_chart(commit_stats.get("daily_counts", {}))

# ------------------------------------------------------------------
# Tabs
# ------------------------------------------------------------------
tab_judge, tab_leaderboard = st.tabs(["🔍 Judge a Repo", "🏆 Leaderboard"])

with tab_judge:
    repo_url = st.text_input("GitHub URL", placeholder="https://github.com/owner/repo", label_visibility="collapsed")
    judge_btn = st.button("Judge This Repo", type="primary", use_container_width=True)

    if judge_btn:
        if not os.environ.get("GEMINI_API_KEY"):
            st.error("Add your Gemini API key in the sidebar first.")
        elif not repo_url.strip():
            st.warning("Paste a GitHub repo URL.")
        else:
            with st.spinner("Reading code, checking commits, comparing against other repos..."):
                try:
                    repo_data = fetch_github_repo_data(repo_url)

                    if not repo_data["file_contents"]:
                        st.warning("No recognizable source files found in that repo.")
                    else:
                        repo_text_parts = [
                            f"Repository: {repo_data['owner']}/{repo_data['repo']} "
                            f"(branch: {repo_data['branch']})\n"
                        ]
                        for path, content in repo_data["file_contents"].items():
                            repo_text_parts.append(f"### {path}\n```\n{content}\n```")
                        repo_text = "\n\n".join(repo_text_parts)

                        result = generate_repo_score(repo_text, repo_data["commit_stats"])
                        repo_label = f"{repo_data['owner']}/{repo_data['repo']}"

                        # --- Feature 3: pre-hackathon commit check ---
                        pre_event_warning = check_pre_hackathon_commits(
                            repo_data["commit_stats"].get("first_commit_date"),
                            hackathon_start,
                        )
                        if pre_event_warning:
                            result.setdefault("red_flags", []).append(pre_event_warning)

                        # --- Feature 1: similarity detection ---
                        file_hashes = compute_file_hashes(repo_data["file_contents"])
                        try:
                            embedding = get_embedding(repo_text)
                        except Exception:
                            embedding = None

                        similarity_warnings = []
                        for prev in st.session_state.fingerprints:
                            hash_pct = hash_overlap_pct(file_hashes, prev["hashes"])
                            embed_pct = 0.0
                            if embedding and prev.get("embedding"):
                                embed_pct = cosine_similarity(embedding, prev["embedding"])
                            if hash_pct > 20 or embed_pct > 85:
                                similarity_warnings.append({
                                    "name": prev["name"], "hash_pct": hash_pct, "embed_pct": embed_pct,
                                })

                        st.session_state.fingerprints.append({
                            "name": repo_label, "hashes": file_hashes, "embedding": embedding,
                        })

                        st.success(f"Judged `{repo_label}`", icon="✅")
                        st.divider()
                        render_scorecard(result, repo_data["commit_stats"], similarity_warnings, repo_label, key_suffix=repo_label)

                        with st.expander("📊 Raw commit statistics"):
                            st.json(repo_data["commit_stats"])
                        with st.expander("📁 Files analyzed"):
                            for f in repo_data["files_found"]:
                                st.code(f, language=None)

                except ValueError as exc:
                    st.error(f"Invalid repository URL: {exc}")
                except Exception as exc:
                    err_msg = str(exc)
                    if "rate limit" in err_msg.lower() or "403" in err_msg:
                        st.error("GitHub rate limit hit. Wait a few minutes or add a token.")
                    elif "api_key" in err_msg.lower() or "401" in err_msg:
                        st.error("Gemini API key invalid. Check the sidebar.")
                    else:
                        st.error(f"Unexpected error: {exc}")

with tab_leaderboard:
    if not st.session_state.leaderboard:
        st.info("No repos on the leaderboard yet. Judge a repo and hit **⭐ Add to Leaderboard**.")
    else:
        ranked = sorted(st.session_state.leaderboard, key=lambda x: x["overall"], reverse=True)
        medals = ["🥇", "🥈", "🥉"]

        for i, entry in enumerate(ranked):
            rank_display = medals[i] if i < 3 else f"#{i+1}"
            flag_icon = " ⚠️" if entry["similarity"] else ""
            st.markdown(f"""
            <div class="lb-row">
                <div class="lb-rank">{rank_display}</div>
                <div class="lb-name">{entry['name']}{flag_icon}</div>
                <div class="lb-score">{entry['overall']}/10</div>
            </div>
            """, unsafe_allow_html=True)

        with st.expander("🔍 View full scorecards"):
            for i, entry in enumerate(ranked):
                st.markdown(f"#### {entry['name']}")
                render_scorecard(entry["result"], entry["commit_stats"], entry["similarity"], entry["name"], key_suffix=f"lb_{i}")
                st.divider()