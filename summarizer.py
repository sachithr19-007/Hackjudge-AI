import json
from google import genai
from google.genai import types


_MODEL = "gemini-3.5-flash-lite"

_SYSTEM_INSTRUCTION = """You are a strict, experienced hackathon judge. You will be given
a repository's source files/README and objective commit statistics. Score the project
on a scale of 1-10 for each category below. Be honest and critical — most hackathon
projects score 4-7, reserve 8+ for genuinely polished, complete work.

Return ONLY valid JSON in this exact shape, no markdown fences, no extra text:

{
  "scores": {
    "originality": {"score": <1-10>, "reason": "<one sentence>"},
    "technical_execution": {"score": <1-10>, "reason": "<one sentence>"},
    "completeness": {"score": <1-10>, "reason": "<one sentence>"},
    "commit_quality": {"score": <1-10>, "reason": "<one sentence>"},
    "documentation": {"score": <1-10>, "reason": "<one sentence>"}
  },
  "overall_score": <1-10, weighted average>,
  "red_flags": ["<any concerning signs, e.g. single-day commit dump, no README, etc>"],
  "verdict": "<2-3 sentence summary of whether this is a credible hackathon submission>"
}

For commit_quality specifically, judge based on: message descriptiveness (not just
"fix"/"update"), whether commits are spread across development time vs dumped in
one sitting, and number of contributors."""


def generate_repo_score(repo_text: str, commit_stats: dict) -> dict:
    """Score a repository for hackathon judging. Returns parsed JSON dict."""
    client = genai.Client()

    commit_summary = (
        f"\n\n### Commit Statistics (objective, computed from GitHub API)\n"
        f"Total commits: {commit_stats['total_commits']}\n"
        f"Unique contributors: {commit_stats['unique_authors']}\n"
        f"Average commit message length: {commit_stats['avg_message_length']} chars\n"
        f"Trivial/low-effort messages: {commit_stats['trivial_message_pct']}%\n"
        f"Active days (days with at least 1 commit): {commit_stats['active_days']}\n"
        f"Total span (days): {commit_stats['span_days']}\n"
        f"% of commits made on the single busiest day: {commit_stats['last_day_commit_pct']}%\n"
        f"Sample commit messages: {commit_stats['messages']}\n"
    )

    full_input = repo_text + commit_summary

    response = client.models.generate_content(
        model=_MODEL,
        contents=full_input,
        config=types.GenerateContentConfig(
            system_instruction=_SYSTEM_INSTRUCTION,
            response_mime_type="application/json",
        ),
    )

    return json.loads(response.text)