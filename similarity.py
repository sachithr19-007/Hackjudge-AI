import hashlib
import re
import numpy as np
from google import genai

_EMBED_MODEL = "gemini-embedding-001"

_COMMENT_PATTERNS = [
    re.compile(r"#.*"),                      # Python/shell style
    re.compile(r"//.*"),                     # JS/TS/C style single-line
    re.compile(r"/\*.*?\*/", re.DOTALL),     # C style block
]


def _normalize_code(content: str) -> str:
    """Strip comments, blank lines, and whitespace so trivial formatting
    or renaming differences don't affect the hash."""
    text = content
    for pattern in _COMMENT_PATTERNS:
        text = pattern.sub("", text)

    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines).lower()


def compute_file_hashes(file_contents: dict) -> dict:
    """Return {file_path: sha256_hash} using normalized file content."""
    hashes = {}
    for path, content in file_contents.items():
        normalized = _normalize_code(content)
        if not normalized:
            continue
        hashes[path] = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return hashes


def hash_overlap_pct(hashes_a: dict, hashes_b: dict) -> float:
    """% of file-content hashes shared between two repos, based on the
    smaller repo (so a small repo fully copied into a big one still flags)."""
    set_a, set_b = set(hashes_a.values()), set(hashes_b.values())
    if not set_a or not set_b:
        return 0.0
    shared = set_a & set_b
    smaller = min(len(set_a), len(set_b))
    return round((len(shared) / smaller) * 100, 1)


def get_embedding(text: str) -> list:
    """Get an embedding vector for a repo's README + code sample."""
    client = genai.Client()
    truncated = text[:20000]  # keep requests fast/cheap
    result = client.models.embed_content(
        model=_EMBED_MODEL,
        contents=truncated,
    )
    return result.embeddings[0].values


def cosine_similarity(vec_a: list, vec_b: list) -> float:
    a, b = np.array(vec_a), np.array(vec_b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return round(float(np.dot(a, b) / denom) * 100, 1)