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



