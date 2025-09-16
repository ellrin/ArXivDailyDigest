#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ArXiv Daily Digest

Fetch latest arXiv papers, score them with an LLM, generate a daily
markdown report and a PDF, then email the results.
"""

import os
import re
import json
import time
import smtplib
import sqlite3
import requests
import argparse
import logging
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Any, Literal, Dict, Tuple
from itertools import islice

import arxiv
from tenacity import retry, wait_fixed, stop_after_attempt, after_log
from pydantic import BaseModel, Field, field_validator, ConfigDict
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication

# Set up logging for tenacity
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ========= Load config from config.py =========
try:
    print("📖 Loading config file...")
    from config import * # noqa: F401,F403
    print("✅ Config file loaded successfully.")
except ImportError:
    print("❌ Error: config.py not found. Please ensure it exists in the same directory.")
    raise SystemExit(1)
except Exception as e:
    print(f"❌ Error loading config file: {e}")
    raise SystemExit(1)

# ========= Load .env =========
def load_env() -> None:
    """Loads environment variables from a .env file."""
    env_file = Path(".env")
    if env_file.exists():
        print("🌍 Loading .env environment variables...")
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, v = line.split("=", 1)
                os.environ[k.strip()] = v.strip()
        print("✅ Environment variables loaded successfully.")

load_env()

# ========= Defaults and fallbacks =========
DB_FILE = globals().get("DB_FILE", "arxiv_seen.sqlite3")
OUTPUT_DIR = Path(globals().get("OUTPUT_DIR", "output"))
CACHE_DIR = Path(globals().get("CACHE_DIR", ".cache"))
CACHE_RETENTION_DAYS = int(globals().get("CACHE_RETENTION_DAYS", 14))
OUTPUT_RETENTION_DAYS = int(globals().get("OUTPUT_RETENTION_DAYS", 30))

_email_cfg = globals().get("EMAIL_CONFIG", {})
SMTP_HOST = _email_cfg.get("smtp_server", os.getenv("SMTP_HOST", ""))
SMTP_PORT = int(_email_cfg.get("smtp_port", os.getenv("SMTP_PORT", 587)))
MAIL_FROM = _email_cfg.get("sender_email", os.getenv("MAIL_FROM", ""))
MAIL_TO = _email_cfg.get("recipient_email", os.getenv("MAIL_TO", ""))
MAIL_SUBJECT_PREFIX = _email_cfg.get("subject_prefix", "ArXiv Daily Digest")
SMTP_USER = os.getenv("SMTP_USER", _email_cfg.get("sender_email", ""))
SMTP_PASS = os.getenv("SMTP_PASS", os.getenv("EMAIL_PASSWORD", ""))

# ========= DB Operations =========
def ensure_db() -> None:
    """Ensures the SQLite database and 'seen' table exist."""
    print("📝 Checking database...")
    con = sqlite3.connect(DB_FILE)
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS seen(
            id TEXT PRIMARY KEY,
            ver INT,
            url TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    con.commit()
    con.close()
    print("✅ Database is ready.")

def seen_before(pid: str, ver: int) -> bool:
    """Checks if a paper has been processed before."""
    con = sqlite3.connect(DB_FILE)
    row = con.execute("SELECT ver FROM seen WHERE id=?", (pid,)).fetchone()
    con.close()
    return row is not None and ver <= (row[0] or 0)

def mark_seen(pid: str, ver: int, url: str = "") -> None:
    """Marks a paper as seen in the database."""
    con = sqlite3.connect(DB_FILE)
    con.execute("INSERT OR REPLACE INTO seen(id, ver, url) VALUES(?,?,?)", (pid, ver, url))
    con.commit()
    con.close()

# ========= Fetch arXiv Papers =========
def verify_arxiv_link(arxiv_id: str) -> Dict[str, Any]:
    """Verifies an arXiv paper's existence and returns its details."""
    try:
        client = arxiv.Client()
        search = arxiv.Search(id_list=[arxiv_id])
        results = list(client.results(search))
        if results:
            r = results[0]
            return {"exists": True, "pdf_url": r.pdf_url, "entry_url": r.entry_id, "title": r.title.strip()}
        return {"exists": False}
    except Exception as e:
        print(f"verify_arxiv_link error for {arxiv_id}: {e}")
        return {"exists": False}

def fetch_candidates():
    """Fetches candidate papers from arXiv based on categories and keywords."""
    query = " OR ".join([f"cat:{c}" for c in CATEGORIES])
    print(f"🔎 Fetching papers from arXiv... (Categories: {', '.join(CATEGORIES)})")
    client = arxiv.Client(page_size=min(100, MAX_CANDIDATES), delay_seconds=4, num_retries=4)

    search = arxiv.Search(
        query=query,
        max_results=MAX_CANDIDATES,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )

    since = datetime.now(TZ) - timedelta(days=DAYS_BACK)
    found_count = 0
    keyword_count = 0

    try:
        for r in islice(client.results(search), min(MAX_CANDIDATES, 100)):
            found_count += 1
            updated_utc = r.updated.replace(tzinfo=timezone.utc)
            if updated_utc < since.astimezone(timezone.utc):
                continue

            pid = r.get_short_id()
            ver = 1
            if hasattr(r, "_raw"):
                try:
                    ver = int(str(r._raw.get("version", "1")).lstrip("v"))
                except Exception:
                    ver = 1

            title = r.title.strip()
            abstract = (r.summary or "").strip()
            comments = getattr(r, "comment", "") or ""
            text = f"{title} {abstract}".lower()

            if KEYWORDS and not any(k.lower() in text for k in KEYWORDS):
                continue

            keyword_count += 1
            yield {
                "id": pid,
                "ver": ver,
                "title": title,
                "abstract": abstract,
                "comments": comments,
                "primary": r.primary_category or "",
                "authors": [a.name for a in r.authors],
                "pdf": r.pdf_url,
                "link": r.entry_id,
                "updated": r.updated,
            }
    except arxiv.UnexpectedEmptyPageError:
        print("⚠️ arXiv returned an empty page, terminating fetch early.")
    
    print(f"Found {found_count} candidate papers, with {keyword_count} matching keywords.")

# ========= LLM Review Schema and Calls =========
class LLMReview(BaseModel):
    model_config = ConfigDict(extra="ignore")

    topic: str = ""
    task: str = ""
    method: str = ""
    dataset: str = ""
    key_claim: str = ""
    code_link: Optional[str] = ""
    evidence_level: float = Field(default=0, ge=0, le=5)
    novelty_score: float = Field(default=0, ge=0, le=5)
    red_flags: List[str] = Field(default_factory=list)
    decision: Literal["keep", "reject"] = "reject"
    rationale: str = ""

    @field_validator("code_link", mode="before")
    def coerce_code_link(cls, v: Any) -> str:
        if v is None:
            return ""
        if isinstance(v, (list, dict)):
            return ""
        s = str(v).strip()
        return "" if s.lower() in {"n/a", "na", "none"} else s

    @field_validator("red_flags", mode="before")
    def coerce_red_flags(cls, v: Any) -> List[str]:
        if v is None:
            return []
        if isinstance(v, str):
            return [v]
        if isinstance(v, list):
            return [str(x) for x in v]
        return []

def safe_parse_json(text: str) -> Dict[str, Any]:
    """Safely parses JSON from a string, handling common LLM output formatting issues."""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except (json.JSONDecodeError, TypeError):
        pass

    m = re.search(r"\{.*\}", text, flags=re.S)
    if m:
        try:
            data = json.loads(m.group(0))
            if isinstance(data, dict):
                return data
        except (json.JSONDecodeError, TypeError):
            pass

    print("⚠️ Warning: Failed to parse JSON. Returning default dict.")
    return {
        "topic": "", "task": "", "method": "", "dataset": "", "key_claim": "",
        "code_link": "", "evidence_level": 0, "novelty_score": 0,
        "red_flags": ["JSON parse error"], "decision": "reject", "rationale": "Could not parse LLM output",
    }

def get_gemini_client():
    """Initializes and returns a Gemini API client."""
    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=gemini_key)
            return genai
        except Exception as e:
            print(f"❌ Gemini API client failed to initialize. Error: {e}")
            return None
    return None

# --- 重試策略變更：從 exponential 改為 fixed，重試2次，間隔2秒 ---
@retry(wait=wait_fixed(2), stop=stop_after_attempt(2), after=after_log(logger, logging.WARNING))
def call_gemini(user_prompt: str, model: str) -> str:
    """Calls the Gemini API with a specific model and retry logic."""
    genai = get_gemini_client()
    if not genai:
        raise RuntimeError("No Gemini client available. GEMINI_API_KEY not found or invalid.")
    
    print(f"🤖 Calling Gemini API with model: {model}...")
    
    llm = genai.GenerativeModel(model_name=model)
    resp = llm.generate_content(
        contents=[{"role": "user", "parts": [{"text": SYSTEM_PROMPT + "\n\n" + user_prompt}]}],
        generation_config=genai.GenerationConfig(response_mime_type="application/json"),
    )
    
    # Check if response has text content
    if not resp.candidates:
        raise ValueError("API returned no candidates.")
    return resp.candidates[0].content.parts[0].text

def call_llm(prompt: str) -> dict:
    """Manages the full LLM review process with a fallback strategy."""
    parsed: Dict[str, Any] = {}
    
    # 僅嘗試 Gemini 2.0-flash-exp 模型
    try:
        raw = call_gemini(prompt, model="gemini-2.0-flash-exp")
        parsed_data = safe_parse_json(str(raw))
        
        if not isinstance(parsed_data, dict):
            raise TypeError("Parsed data is not a dictionary.")
            
        parsed = LLMReview(**parsed_data).model_dump()
        print("✅ LLM call succeeded.")
        return parsed
    except Exception as e:
        print(f"❌ LLM call failed. Error: {e}")
    
    # 如果失敗，直接回傳一個預設的 rejected 字典
    parsed = LLMReview().model_dump()
    parsed["decision"] = "reject"
    parsed["rationale"] = f"LLM error: {str(e)[:40]}"
    return parsed

# ========= Scoring Adjustments =========
def adjust_score_by_keywords(text: str, base_score: float, keyword_cfg: dict) -> float:
    """Adjusts scores based on custom keyword matches."""
    text_lower = text.lower()
    total = 0.0
    for cfg in keyword_cfg.values():
        keywords = cfg.get("keywords", [])
        boost = float(cfg.get("score_boost", 0))
        penalty = float(cfg.get("score_penalty", 0))
        matches = sum(1 for k in keywords if k.lower() in text_lower)
        if matches > 0:
            if boost > 0:
                total += min(boost, boost * matches * 0.3)
            if penalty < 0:
                total += max(penalty, penalty * matches * 0.3)
    return max(0.0, min(5.0, float(base_score) + total))

def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip().lower()

def _company_hits(text: str, companies: List[str]) -> List[str]:
    """Finds matching company names in a text."""
    t = _normalize(text)
    hits = []
    for name in companies:
        n = _normalize(name)
        if re.search(rf"\b{re.escape(n)}\b", t):
            hits.append(name)
    return hits

def count_unique_keywords(text: str, keywords: list) -> int:
    """Counts the number of unique keywords found in a text."""
    text_lower = text.lower()
    found_keywords = set()
    for keyword in keywords:
        if keyword.lower() in text_lower:
            found_keywords.add(keyword.lower())
    return len(found_keywords)

def apply_company_evidence_bonus(paper: Dict[str, Any], review: Dict[str, Any]) -> None:
    """Applies an evidence bonus if the paper mentions specified companies."""
    cfg = getattr(ScoringConfig, "COMPANY_EVIDENCE", None)
    if not cfg:
        return

    tiers = cfg.get("tiers", {})
    decay = cfg.get("decay", [1.0, 0.6, 0.4])
    cap = float(cfg.get("cap", 1.5))
    strong_mul = float(cfg.get("strong_multiplier", 1.2))
    weak_mul = float(cfg.get("weak_multiplier", 1.0))
    gate_mode = int(cfg.get("gate_mode", 2))
    gate_terms = [w.lower() for w in cfg.get("gate_terms", [])]
    redflag_penalty = float(cfg.get("redflag_penalty", 0.5))

    title = paper.get("title", "")
    abstract = paper.get("abstract", "")
    comments = paper.get("comments", "")
    authors_joined = " ".join(paper.get("authors", []))

    strong_text = comments
    weak_text = f"{title} {abstract} {authors_joined}"

    hits_with_score = []
    for tier in tiers.values():
        base = float(tier.get("score", 0.0))
        companies = tier.get("companies", [])
        strong_hits = _company_hits(strong_text, companies)
        if strong_hits:
            hits_with_score.extend([(h, base * strong_mul) for h in strong_hits])
        weak_hits = _company_hits(weak_text, companies)
        for h in weak_hits:
            if not any(h == x for x, _ in hits_with_score):
                hits_with_score.append((h, base * weak_mul))

    if not hits_with_score:
        return

    combo_text = _normalize(f"{title} {abstract} {comments}")
    gate_has_terms = any(w in combo_text for w in gate_terms) if gate_terms else False

    if gate_mode == 3 and not gate_has_terms:
        return

    soft_gate_boost = 1.2 if (gate_mode == 2 and gate_has_terms) else 1.0

    hits_with_score.sort(key=lambda x: x[1], reverse=True)
    total_bonus = 0.0
    for i, (_, s) in enumerate(hits_with_score):
        w = decay[i] if i < len(decay) else decay[-1]
        total_bonus += s * w
        if total_bonus >= cap:
            total_bonus = cap
            break

    if review.get("red_flags"):
        total_bonus *= redflag_penalty

    total_bonus *= soft_gate_boost

    review["evidence_level"] = float(max(0.0, min(5.0, review.get("evidence_level", 0.0) + total_bonus)))

# ========= Main LLM Review Pipeline =========
def llm_review(paper: Dict[str, Any]) -> dict:
    """Manages the full LLM review process with a fallback strategy."""
    prompt = build_user_prompt(paper)
    parsed = call_llm(prompt)

    text = paper["title"] + " " + paper["abstract"]
    parsed["novelty_score"] = round(
        adjust_score_by_keywords(text, parsed["novelty_score"], ScoringConfig.NOVELTY_KEYWORDS), 1
    )
    parsed["evidence_level"] = round(
        adjust_score_by_keywords(text, parsed["evidence_level"], ScoringConfig.EVIDENCE_KEYWORDS), 1
    )
    apply_company_evidence_bonus(paper, parsed)

    is_keep = (
        parsed["evidence_level"] >= ScoringConfig.KEEP_THRESHOLD["min_evidence_level"]
        and parsed["novelty_score"] >= ScoringConfig.KEEP_THRESHOLD["min_novelty_score"]
        and (parsed["evidence_level"] + parsed["novelty_score"]) >= ScoringConfig.KEEP_THRESHOLD["min_combined_score"]
    )

    if is_keep:
        parsed["decision"] = "keep"
        print(f"✅ Evaluation complete: {paper['title'][:50]}... -> KEEP (Evidence: {parsed['evidence_level']}, Novelty: {parsed['novelty_score']})")
    else:
        parsed["decision"] = "reject"
        print(f"❌ Evaluation complete: {paper['title'][:50]}... -> REJECT (Evidence: {parsed['evidence_level']}, Novelty: {parsed['novelty_score']})")
        
    return parsed

# ========= Final Score Calculation =========
def calculate_score(paper: dict, review: dict) -> float:
    """Calculates a final score for ranking a paper based on its review."""
    cfg = ScoringConfig
    base = cfg.NOVELTY_WEIGHT * review["novelty_score"] + cfg.EVIDENCE_WEIGHT * review["evidence_level"]
    bonus = 0.0
    if review.get("code_link"):
        bonus += cfg.CODE_BONUS
    if review.get("dataset") and len(review["dataset"]) > 3:
        bonus += cfg.DATASET_BONUS
    penalty = cfg.RED_FLAG_PENALTY * min(cfg.MAX_RED_FLAG_PENALTY, len(review.get("red_flags", [])))

    text = paper["title"] + " " + paper["abstract"]
    unique_keyword_count = count_unique_keywords(text, KEYWORDS)

    keyword_bonus = 0.2 * unique_keyword_count
    bonus += keyword_bonus

    hours_old = (datetime.now(TZ) - paper["updated"].replace(tzinfo=timezone.utc).astimezone(TZ)).total_seconds() / 3600
    recency_bonus = 0.0
    if hours_old < cfg.RECENCY_HOURS_WINDOW:
        recency_bonus = max(0.0, (cfg.RECENCY_HOURS_WINDOW - hours_old) / cfg.RECENCY_HOURS_WINDOW * cfg.RECENCY_BONUS_MAX)

    return max(0.0, base + bonus - penalty + recency_bonus)

# ========= Report Generation =========
def generate_markdown(papers_and_reviews: List[Tuple], date_str: str) -> str:
    """Generates a Markdown report string for the selected papers."""
    lines = [
        f"# ArXiv Daily Digest · {date_str}",
        "",
        f"**Categories**: {', '.join(CATEGORIES)}",
        f"**Keywords**: {', '.join(KEYWORDS)}",
        f"**Total Papers Reviewed**: {len(papers_and_reviews)}",
        f"**Scoring**: Novelty×{ScoringConfig.NOVELTY_WEIGHT} + Evidence×{ScoringConfig.EVIDENCE_WEIGHT} + bonuses − penalties",
        "",
        "---",
        "",
    ]

    for idx, (score, paper, review) in enumerate(papers_and_reviews, 1):
        v = verify_arxiv_link(paper["id"])
        pdf_url = v.get("pdf_url", paper["pdf"]) if v.get("exists") else paper["pdf"]
        entry_url = v.get("entry_url", paper["link"]) if v.get("exists") else paper["link"]
        author_list = ", ".join(paper["authors"][:5])
        more = f" (+{len(paper['authors'])-5} more)" if len(paper["authors"]) > 5 else ""

        lines.extend(
            [
                f"## {idx}. {paper['title']}",
                "",
                f"**ArXiv ID**: [{paper['id']}]({entry_url}) v{paper['ver']} | **Category**: {paper['primary']} | **Score**: {score:.2f}/5.0",
                "",
                f"**Authors**: {author_list}{more}",
                "",
                f"**PDF**: [Download]({pdf_url})",
                "",
                "### Analysis",
                f"- **Topic**: {review['topic']}",
                f"- **Task**: {review['task']}",
                f"- **Method**: {review['method']}",
                f"- **Dataset**: {review['dataset'] or 'Not specified'}",
                f"- **Key Claim**: {review['key_claim']}",
                "",
                "### Scores",
                f"- **Evidence Level**: {review['evidence_level']}/5",
                f"- **Novelty Score**: {review['novelty_score']}/5",
                f"- **Code Available**: {'✅' if review['code_link'] else '❌'}" + (f" ([Link]({review['code_link']}))" if review["code_link"] else ""),
                "",
                "### Assessment",
                f"**Red Flags**: {', '.join(review['red_flags']) if review['red_flags'] else 'None'}",
                "",
                f"**Why Selected**: {review['rationale']}",
                "",
                "### Abstract",
                f"{paper['abstract'][:500]}{'...' if len(paper['abstract']) > 500 else ''}",
                "",
                "---",
                "",
            ]
        )

    return "\n".join(lines)

# ========= Markdown to PDF Conversion =========
def markdown_to_pdf(markdown_file: str, pdf_file: str) -> bool:
    """Converts a Markdown file to a PDF file."""
    print("📄 Generating PDF file...")
    try:
        import markdown
        import weasyprint

        md = Path(markdown_file).read_text(encoding="utf-8")
        html = markdown.markdown(md, extensions=["tables", "fenced_code"])
        styled = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8" />
            <style>
                body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; line-height: 1.6; margin: 40px; color: #333; }}
                h1 {{ color: #2c3e50; border-bottom: 2px solid #eee; padding-bottom: 6px; }}
                h2 {{ color: #2c3e50; margin-top: 28px; }}
                code, pre {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, 'Liberation Mono', monospace; }}
                a {{ color: #0b6; text-decoration: none; }}
                a:hover {{ text-decoration: underline; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 6px 8px; }}
            </style>
        </head>
        <body>{html}</body>
        </html>
        """
        pdf = weasyprint.HTML(string=styled).write_pdf()
        Path(pdf_file).write_bytes(pdf)
        print("✅ PDF file generated successfully.")
        return True
    except ImportError:
        print("❌ Error: PDF generation requires 'markdown' and 'weasyprint'. Please run 'pip install markdown weasyprint'.")
        return False
    except Exception as e:
        print(f"❌ markdown_to_pdf failed: {e}")
        return False

# ========= Email Helpers =========
def send_email(subject: str, html_body: str, attachments: Optional[List[Tuple[str, bytes]]] = None) -> None:
    """Sends an email with an optional attachment."""
    if not SMTP_HOST or not MAIL_FROM or not MAIL_TO:
        print("❌ Email not sent. SMTP or email configuration is missing.")
        return

    msg = MIMEMultipart()
    msg["From"] = MAIL_FROM
    msg["To"] = MAIL_TO
    msg["Subject"] = subject
    msg.attach(MIMEText(html_body, "html", "utf-8"))

    for att_name, att_bytes in attachments or []:
        part = MIMEApplication(att_bytes)
        part.add_header("Content-Disposition", "attachment", filename=att_name)
        msg.attach(part)
    
    print(f"📧 Sending email to {MAIL_TO}...")
    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as s:
            s.starttls()
            if SMTP_USER and SMTP_PASS:
                s.login(SMTP_USER, SMTP_PASS)
            s.send_message(msg)
        print("✅ Email sent successfully.")
    except smtplib.SMTPAuthenticationError:
        print("❌ Email sending failed: SMTP authentication failed. Please check your sender email and app password.")
    except Exception as e:
        print(f"❌ Email sending failed: {e}")

def send_email_no_results(date_str: str) -> None:
    """Sends an email notification when no papers are found or selected."""
    subject = f"{MAIL_SUBJECT_PREFIX} · {date_str} · No results today"
    html = (
        "<p>Hello,</p>"
        "<p>Your arXiv digest ran successfully but found no papers matching the filters today."
        " This can happen when category filters are narrow or keyword filters are strict.</p>"
        "<p>The job will try again in the next cycle.</p>"
        "<p>Regards,<br>ArXiv Daily Digest</p>"
    )
    send_email(subject, html)

# ========= Cleanup Operations =========
def _remove_older_than(path: Path, days: int) -> int:
    """Removes files older than a specified number of days from a directory."""
    if not path.exists():
        return 0
    now = time.time()
    ttl = days * 86400
    removed = 0
    for p in path.rglob("*"):
        if not p.is_file():
            continue
        try:
            age = now - p.stat().st_mtime
            if age > ttl:
                p.unlink(missing_ok=True)
                removed += 1
        except Exception:
            continue
    return removed

def cleanup_environment() -> None:
    """Cleans up old cache and output files."""
    print("🗑️ Cleaning up old files...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    removed_cache = _remove_older_than(CACHE_DIR, CACHE_RETENTION_DAYS)
    removed_out = _remove_older_than(OUTPUT_DIR, OUTPUT_RETENTION_DAYS)
    print(f"✅ Cleanup complete: Removed {removed_cache} cache files and {removed_out} output files.")

    try:
        con = sqlite3.connect(DB_FILE)
        con.execute("VACUUM")
        con.close()
    except Exception as e:
        print(f"⚠️ Database VACUUM failed: {e}")

# ========= Main Job Execution =========
def main() -> None:
    """The main function to run the daily digest job."""
    print("🚀 Starting ArXiv Digest...")
    ensure_db()
    cleanup_environment()

    date_str = datetime.now(TZ).strftime("%Y-%m-%d")

    candidates: List[Dict[str, Any]] = []
    
    all_fetched_papers = list(fetch_candidates())
    print(f"Found {len(all_fetched_papers)} papers. Starting filtering...")
    
    for paper in all_fetched_papers:
        if seen_before(paper["id"], paper["ver"]):
            print(f"➡️ Skipping already seen paper: {paper['id']} v{paper['ver']}")
            continue
        candidates.append(paper)
    
    print(f"After filtering, {len(candidates)} new candidate papers are waiting for evaluation.")

    if len(candidates) == 0:
        print("No new papers to evaluate. Sending no-results email.")
        send_email_no_results(date_str)
        return

    scored: List[Tuple[float, dict, dict]] = []
    for i, p in enumerate(candidates):
        try:
            review = llm_review(p)
            mark_seen(p["id"], p["ver"], p.get("link", ""))
            
            if review.get("decision") != "keep":
                continue
            
            score = calculate_score(p, review)
            scored.append((score, p, review))
        except Exception as e:
            # --- 簡化錯誤處理：遇到任何錯誤都直接跳過這篇論文 ---
            print(f"❌ An unexpected error occurred while processing paper {p['id']}: {e}")
            mark_seen(p["id"], p["ver"], p.get("link", ""))
            # 這裡不需額外處理，因為迴圈結束後就會繼續下一篇
            continue
        finally:
            # Add a deliberate pause after each paper is processed, regardless of outcome
            print("⏳ Pausing for 7 seconds to respect API rate limits...")
            time.sleep(7)

    print(f"Out of {len(candidates)} papers evaluated, {len(scored)} were selected.")

    if len(scored) == 0:
        print("All new papers were rejected. Sending no-results email.")
        send_email_no_results(date_str)
        return

    scored.sort(key=lambda x: x[0], reverse=True)
    if TOP_K and TOP_K > 0:
        scored = scored[:TOP_K]
    
    print(f"Generating report for the top {len(scored)} papers.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    md_name = f"arxiv_digest_{date_str}.md"
    md_path = OUTPUT_DIR / md_name
    pdf_name = f"arxiv_digest_{date_str}.pdf"
    pdf_path = OUTPUT_DIR / pdf_name

    md = generate_markdown(scored, date_str)
    md_path.write_text(md, encoding="utf-8")
    print(f"✅ Markdown report generated: {md_path}")

    made_pdf = markdown_to_pdf(str(md_path), str(pdf_path))

    subject = f"{MAIL_SUBJECT_PREFIX} · {date_str} · {len(scored)} picks"
    html = (
        f"<p>Hello,</p><p>Your digest for {date_str} is ready.</p>"
        f"<p>Total reviewed: {len(scored)}.</p>"
        f"<p>Attachments include markdown and PDF.</p>"
        f"<p>Regards,<br>ArXiv Daily Digest</p>"
    )

    attachments: List[Tuple[str, bytes]] = [(md_name, md.encode("utf-8"))]
    if made_pdf and pdf_path.exists():
        attachments.append((pdf_name, pdf_path.read_bytes()))

    send_email(subject, html, attachments)
    print("✨ Program execution finished.")

# ========= Daemon Loop =========
def _daemon_loop() -> None:
    """Runs the main job once a day at the configured time."""
    cfg = globals().get("DAEMON_CONFIG", {})
    run_hour = cfg.get("run_hour", 4)
    run_minute = cfg.get("run_minute", 0)
    run_second = cfg.get("run_second", 0)

    print(f"⏳ ArXiv Digest is now running in daemon mode, set to run daily at {run_hour:02d}:{run_minute:02d}:{run_second:02d}.")
    print("Press Ctrl+C to exit.")

    while True:
        now = datetime.now(TZ)
        next_run = now.replace(
            hour=run_hour, minute=run_minute, second=run_second, microsecond=0
        )
        if now >= next_run:
            next_run += timedelta(days=1)
        
        sleep_seconds = (next_run - now).total_seconds()
        
        print(f"\n💤 Next run time: {next_run.strftime('%Y-%m-%d %H:%M:%S')}")

        while sleep_seconds > 0:
            hours = int(sleep_seconds // 3600)
            minutes = int((sleep_seconds % 3600) // 60)
            seconds = int(sleep_seconds % 60)
            
            # 格式化並在同一行輸出，使用 \r 實現覆蓋
            countdown_str = f"⏰ Remaining time: {hours:02d}h {minutes:02d}m {seconds:02d}s."
            print(countdown_str.ljust(50), end="\r") # 使用 ljust 確保清除舊的字串

            # 每次休眠一秒
            time.sleep(1)
            sleep_seconds -= 1
        
        print("                                                                ", end="\r")

        print("\n⏰ Time to run the daily job!")
        try:
            main()
        except Exception as e:
            print(f"🚨 Daily job execution failed: {e}")
            time.sleep(300) # Wait 5 minutes before checking again
        
        print("--- Daily job execution finished ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ArXiv Daily Digest CLI tool.")
    parser.add_argument(
        "--daemon", action="store_true", help="Run in daily daemon mode."
    )
    args = parser.parse_args()

    if args.daemon:
        _daemon_loop()
    else:
        main()