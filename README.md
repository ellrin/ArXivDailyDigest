# ArXivDailyDigest
ArXiv Daily Digest by Vibe Coding. An automated system for discovering, scoring, and delivering daily research updates with multi-LLM support.

This project, developed by **Vibe Coding**, automates the process of finding relevant academic papers. It leverages multiple large language models, including **GPT-5**, **Claude.ai Sonnet 4**, and the **Gemini API**. The system fetches the latest papers from arXiv, filters them by user-defined categories and keywords, evaluates their quality using a custom scoring mechanism, and compiles a daily email digest in both Markdown and PDF formats. If no papers match your filters, you'll receive a brief notification email instead.

---

## 🖼️ Screenshot

-----

## 🎯 Scoring Mechanism

Our scoring system prioritizes **verifiable signals** over subjective descriptions.

- 🏛️ **Author Affiliation (Primary Signal):** Papers from organizations with a proven track record of publishing highly-cited and impactful research receive a significant score bonus. The system recognizes different levels of contributions based on the organization's historical output, applying varying bonus scores to reflect their influence in the field. The bonus decays over time and has a cap to prevent score inflation.

- 🔬 **Experimental Keywords (Secondary Signal):** Terms like *dataset*, *benchmark*, *ablation*, *SOTA*, and *cross-validation* provide a smaller boost to the evidence score.

- 🌐 **Keyword Diversity Bonus:** Papers with a greater variety of matching keywords receive a higher score. This encourages the discovery of unique, interdisciplinary research.

- ⚠️ **Low-Innovation Words (Penalty):** Words such as *survey*, *review*, *comparison*, *analysis*, and *preliminary* reduce the novelty score.

- 🚩 **Red Flags:** The absence of experiments, lack of baselines, or vague claims can reduce the evidence bonus, even if the paper is from a major organization.

- ✅ **Decision Rule:** A paper is included in the digest only if it meets the minimum thresholds for evidence, novelty, and a combined overall score.

---

## ⚙️ Installation & Setup

### 📦 Dependencies

You can install all required dependencies using `pip`:

```bash
pip install -r requirements.txt
```

Alternatively, you can install them manually:

```bash
pip install arxiv tenacity pydantic httpx python-dateutil google-genai openai markdown weasyprint
```

🔑 Environment Variables

You must set the following environment variables. The system will use the Gemini API key as the primary choice and fall back to the OpenAI API key if the Gemini key is not available.

🔐 EMAIL_PASSWORD — Your email application password.

☄️ GEMINI_API_KEY — Your Google Gemini API key.

🤖 OPENAI_API_KEY — Your OpenAI API key.

Linux/macOS
echo 'export EMAIL_PASSWORD="your-app-password"' >> ~/.bashrc
echo 'export GEMINI_API_KEY="your-gemini-key"' >> ~/.bashrc
echo 'export OPENAI_API_KEY="your-openai-key"' >> ~/.bashrc
source ~/.bashrc

Windows PowerShell
$Env:EMAIL_PASSWORD = "your-app-password"
$Env:GEMINI_API_KEY = "your-gemini-key"
$Env:OPENAI_API_KEY = "your-openai-key"

.env File
EMAIL_PASSWORD=your-app-password
GEMINI_API_KEY=your-gemini-key
OPENAI_API_KEY=your-openai-key

🚀 Usage
▶️ Single Run
python arxiv_digest.py

🔁 Daemon Mode
python arxiv_digest.py --daemon


You can adjust the schedule in config.py:

DAEMON_CONFIG = {
    "run_hour": 4,
    "run_minute": 0,
    "run_second": 0,
}

🛡️ Handling API Errors

The script is built with robust error handling to deal with common issues like 429 Too Many Requests errors, which can occur when API usage is too high. The built-in exponential backoff and fallback strategy will automatically pause and retry requests, ensuring your daily digest runs successfully. If an API key or service is unavailable, the system will seamlessly switch to a working alternative.

🧹 Cache and Cleanup

The system automatically manages old files and caches to save disk space.

🗂️ Cache Retention: 14 days (configurable in config.py)

📑 Output Retention: 30 days (configurable in config.py)

🗃️ The SQLite database is periodically vacuumed to free up space.

📡 API Status

🔵 OpenAI Status

🟢 Google AI Studio Status



