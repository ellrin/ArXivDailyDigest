"""
ArXiv Daily Digest - Configuration File
User-editable settings are at the top in ALL CAPS.
"""

from dateutil.tz import gettz

# ============= USER SETTINGS (EDIT THESE) =============

# Email addresses
SENDER_EMAIL_ADDRESS = "MAIL_FROM@XXX.com"
RECIPIENT_EMAIL_ADDRESS = "MAIL_TO@XXX.com"

# Optional alias for backward compatibility
SENT_EMAIL_ADDRESS = SENDER_EMAIL_ADDRESS

# Database file
DB_FILE = "seen.db"

# Timezone and recency window
TZ = gettz("Asia/Taipei")
DAYS_BACK = 1  # how many past days to check

# ArXiv search filters
CATEGORIES = [
    "cs.CV",
    "cs.LG",
    "cs.AI",
    "stat.ML",
    "eess.IV",
]
KEYWORDS = [
    "segmentation", "uncertainty", "graph", "gait", "fuzzy",
    "LoRA", "prefix tuning", "SLAM", "pose estimation",
    "medical imaging", "LiDAR", "GCN", "GNN", "robot", "robotics", "transformer",
    "vision", "image", "video", "3D", "multimodal", "self-supervised",
    "few-shot", "zero-shot", "reinforcement learning", "adversarial",
    "NLP", "natural language", "GAN", "diffusion", "neural rendering",
    "point cloud", "attention", "explainable", "XAI",
    "graph neural network", "graph representation", "graph convolution",
    "continual learning", "domain adaptation", "out-of-distribution",
    "anomaly detection", "time series", "medical", "healthcare"
]
MAX_CANDIDATES = 200
TOP_K = 15

# SMTP and subject
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SUBJECT_PREFIX = "ArXiv Daily Digest"

# Daily run schedule
DAEMON_CONFIG = {
    "run_hour": 4,
    "run_minute": 0,
    "run_second": 0,
}

# Required environment variables
REQUIRED_ENV_VARS = {
    "EMAIL_PASSWORD": "App password for email",
    "GEMINI_API_KEY": "Google Gemini API key (preferred)",
}

# ============= DERIVED SETTINGS (DO NOT EDIT) =============

EMAIL_CONFIG = {
    "smtp_server": SMTP_SERVER,
    "smtp_port": SMTP_PORT,
    "sender_email": SENDER_EMAIL_ADDRESS,
    "sender_password": None,  # read from EMAIL_PASSWORD at runtime
    "recipient_email": RECIPIENT_EMAIL_ADDRESS,
    "subject_prefix": SUBJECT_PREFIX,
}

# ============= ADVANCED SETTINGS (EXPERTS ONLY) =============

class ScoringConfig:
    # Base weights
    NOVELTY_WEIGHT = 0.4
    EVIDENCE_WEIGHT = 0.4

    # Small bonuses
    CODE_BONUS = 0.1
    DATASET_BONUS = 0.05

    # Penalties
    RED_FLAG_PENALTY = 0.15
    MAX_RED_FLAG_PENALTY = 3

    # Recency bonus
    RECENCY_BONUS_MAX = 0.1
    RECENCY_HOURS_WINDOW = 24

    # Company evidence is the main evidence signal
    COMPANY_EVIDENCE = {
        # Tiers and base scores per company hit
        "tiers": {
            "A": {
                "score": 1.0,
                "companies": [
                    "google deepmind", "deepmind",
                    "google research",
                    "openai",
                    "meta ai", "fair", "facebook ai research",
                    "microsoft research", "msr",
                    "nvidia", "nvlabs",
                    "anthropic",
                    "apple", "apple machine learning research",
                ],
            },
            "B": {
                "score": 1.0,
                "companies": [
                    "amazon", "aws ai labs", "adobe research",
                    "nvidia research", "meta genai",
                ],
            },
            "C": {
                "score": 1.0,
                "companies": [
                    "alibaba", "alibaba damo", "damo",
                    "huawei", "noah's ark lab",
                    "baidu",
                    "tencent", "tencent ai lab",
                    "bytedance",
                    "sensetime",
                ],
            },
        },

        # Soft gate: company bonus always applies
        # and gets an extra multiplier if gate terms also appear
        "gate_mode": 2,  # 1=no gate, 2=soft gate, 3=hard gate
        "gate_terms": ["dataset", "benchmark", "code", "github", "evaluation", "ablation"],

        # Multipliers and stacking
        "strong_multiplier": 1.2,  # match in authors or comments
        "weak_multiplier": 1.0,    # match only in title or abstract
        "decay": [1.0, 0.6, 0.4],  # first, second, third company hit

        # Cap and red flag attenuation
        "cap": 1.5,         # maximum total company bonus
        "redflag_penalty": 0.5,  # multiply company bonus if red flags are present
    }

    # Novelty keywords: no positive boosts from adjectives
    # Keep small penalty for low-innovation signals
    NOVELTY_KEYWORDS = {
        "low_innovation": {
            "keywords": [
                "survey", "review", "comparison", "analysis", "study of",
                "evaluation of", "investigation", "exploration", "preliminary",
            ],
            "score_penalty": -0.5,
        },
    }

    # Evidence keywords: weaker than company evidence
    EVIDENCE_KEYWORDS = {
        "strong_evidence": {
            "keywords": [
                "experiment", "benchmark", "dataset", "baseline", "quantitative",
                "statistical", "significant", "outperform", "state-of-the-art",
                "sota", "ablation", "cross-validation",
            ],
            "score_boost": 0.3,
        },
        "weak_evidence": {
            "keywords": [
                "propose", "theoretical", "conceptual", "preliminary",
                "qualitative", "case study", "proof-of-concept",
            ],
            "score_penalty": -0.3,
        },
    }

    # Thresholds for keeping a paper
    KEEP_THRESHOLD = {
        "min_evidence_level": 3,
        "min_novelty_score": 2,
        "min_combined_score": 4,
    }

SYSTEM_PROMPT = """You are a strict but rational research assistant.
Read a paper's title and abstract and output an evaluation in strict JSON.

Scoring rules:
- evidence_level (0 to 5)
- novelty_score (0 to 5)

Company affiliation is a strong evidence signal.
Low-innovation adjectives do not give positive boosts.

Common red flags include lack of experiments or baselines, missing datasets,
vague descriptions, and overclaims.

Decision rules:
- evidence_level >= 3 and novelty_score >= 2 and (evidence + novelty) >= 4 → keep
- obvious red flags → reject
- off-topic → reject

Output JSON only.
"""

def build_user_prompt(paper: dict) -> str:
    return f"""Evaluate the following paper:

Title: {paper['title']}

Abstract: {paper['abstract']}

Output JSON with fields:
- topic
- task
- method
- dataset
- key_claim
- code_link
- evidence_level (0-5)
- novelty_score (0-5)
- red_flags
- decision (keep/reject)
- rationale"""
