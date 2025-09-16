# ArXivDailyDigest

ArXiv Daily Digest by **Vibe Coding**. An automated system for discovering, scoring, and delivering daily research updates with multi-LLM support.

> Fetch the latest arXiv papers, filter by your interests, score with LLMs, and deliver a clean daily digest in Markdown and PDF. If no papers qualify, a short notification email is sent.

---

## 📋 Table of Contents
- 🖼️ Screenshot
- 🎯 Scoring Mechanism
- ⚙️ Installation & Setup
- 🚀 Usage
- 🛡️ Handling API Errors
- 🧹 Cache & Cleanup
- 📡 API Status

---

## 🖼️ Screenshot

[Insert your screenshot here: fig.png]

---

## 🎯 Scoring Mechanism

This system favors verifiable signals over vague prose. Papers must pass minimum thresholds
for evidence, novelty, and overall score to enter the digest.

🏛️ Author Affiliation (Primary Signal)
    Papers from organizations with a history of highly cited work receive a score bonus.
    The bonus decays over time and has a cap to prevent inflation.

🔬 Experimental Keywords (Secondary Signal)
    Terms like dataset, benchmark, ablation, SOTA, cross-validation add a smaller evidence boost.

🌐 Keyword Diversity Bonus
    More diverse matched keywords raise the score and promote interdisciplinary work.

⚠️ Low-Innovation Words (Penalty)
    Words such as survey, review, comparison, analysis, preliminary reduce the novelty score.

🚩 Red Flags
    Missing experiments, no baselines, or vague claims can reduce the evidence bonus.

✅ Decision Rule
    A paper is included in the digest only if it passes evidence and novelty thresholds and reaches a sufficient overall score.

---

## ⚙️ Installation & Setup

### 📦 Dependencies

pip install -r requirements.txt

# Or install manually:
pip install arxiv tenacity pydantic httpx python-dateutil google-genai openai markdown weasyprint

---

## 🔑 Environment Variables

The system prefers Gemini first, then falls back to OpenAI if Gemini is unavailable.

EMAIL_PASSWORD = your-app-password
GEMINI_API_KEY = your-gemini-key
OPENAI_API_KEY = your-openai-key

### Linux/macOS

echo 'export EMAIL_PASSWORD="your-app-password"' >> ~/.bashrc
echo 'export GEMINI_API_KEY="your-gemini-key"' >> ~/.bashrc
echo 'export OPENAI_API_KEY="your-openai-key"' >> ~/.bashrc
source ~/.bashrc

### Windows PowerShell

$Env:EMAIL_PASSWORD = "your-app-password"
$Env:GEMINI_API_KEY = "your-gemini-key"
$Env:OPENAI_API_KEY = "your-openai-key"

### .env File

EMAIL_PASSWORD=your-app-password
GEMINI_API_KEY=your-gemini-key
OPENAI_API_KEY=your-openai-key

---

## 🚀 Usage

### ▶️ Single Run

python arxiv_digest.py

### 🔁 Daemon Mode

python arxiv_digest.py --daemon

### ⏱️ Schedule (config.py)

DAEMON_CONFIG = {
    "run_hour": 4,
    "run_minute": 0,
    "run_second": 0,
}

---

## 🛡️ Handling API Errors

The script uses exponential backoff and model fallback to handle common failures such as **429 Too Many Requests**. When a key or service is unavailable, it automatically pauses, retries, or switches to a working provider.

---

## 🧹 Cache & Cleanup

The system manages caches and outputs to save space.

🗂️ Cache retention: 14 days (configurable in config.py)
📑 Output retention: 30 days (configurable in config.py)
🗃️ Periodic SQLite VACUUM to reclaim storage

---

## 📡 API Status

🔵 OpenAI Status → https://status.openai.com
🟢 Google AI Studio Status → https://aistudio.google.com/status
