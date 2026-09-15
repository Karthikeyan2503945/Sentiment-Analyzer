"""
Aspect-Based Sentiment Analysis (ABSA) Extension Module.
Distinct from the primary document-level LSTM classifier.
Extracts specific product/service aspects and resolves sentiment per aspect.
"""

import re
from typing import List, Dict, Any
from src.preprocessing import clean_text

# Common domain aspect patterns
ASPECT_PATTERNS = {
    "Camera / Image Quality": [r"\bcamera\b", r"\blens\b", r"\bphotos?\b", r"\bpicture quality\b", r"\bvideo\b", r"\bzoom\b"],
    "Battery Life & Charging": [r"\bbattery\b", r"\bbattery life\b", r"\bcharging\b", r"\bcharger\b", r"\bpower\b", r"\bbackup\b"],
    "Display & Screen": [r"\bscreen\b", r"\bdisplay\b", r"\bbrightness\b", r"\bresolution\b", r"\bpanel\b", r"\bcolors?\b"],
    "Sound & Audio": [r"\bsound\b", r"\baudio\b", r"\bspeaker\b", r"\bmicrophone\b", r"\bmic\b", r"\bvolume\b", r"\bbass\b", r"\bnoise cancellation\b"],
    "Performance & Speed": [r"\bperformance\b", r"\bspeed\b", r"\bprocessor\b", r"\bram\b", r"\blag\b", r"\bfreeze\b", r"\bfast\b", r"\bslow\b", r"\bsoftware\b", r"\bapp\b"],
    "Build Quality & Design": [r"\bbuild quality\b", r"\bmaterials?\b", r"\bdesign\b", r"\bfinish\b", r"\bergonomics?\b", r"\bdurability\b", r"\bweight\b", r"\bhinge\b"],
    "Customer Service": [r"\bcustomer service\b", r"\bsupport\b", r"\bhelpdesk\b", r"\brepresentative\b", r"\brefund\b", r"\breturn policy\b", r"\bstaff\b"],
    "Shipping & Delivery": [r"\bshipping\b", r"\bdelivery\b", r"\bpackage\b", r"\bpackaging\b", r"\barrival\b", r"\bcourier\b"],
    "Price & Value": [r"\bprice\b", r"\bcost\b", r"\bvalue\b", r"\bexpensive\b", r"\baffordable\b", r"\bworth\b", r"\bdeal\b"],
    "Comfort & Fit": [r"\bcomfort\b", r"\bfit\b", r"\bcomfortable\b", r"\bcushioning\b", r"\bwear\b", r"\bshoes?\b", r"\bfabric\b"]
}

POSITIVE_INDICATORS = {
    "excellent", "great", "amazing", "superb", "fantastic", "good", "love", "loved", 
    "outstanding", "solid", "perfect", "clear", "crisp", "fast", "durable", "responsive",
    "top", "pleased", "satisfied", "impressive", "high quality", "stellar", "phenomenal"
}

NEGATIVE_INDICATORS = {
    "terrible", "awful", "horrible", "bad", "poor", "broken", "worst", "hate", "hated",
    "slow", "sluggish", "flimsy", "cheap", "laggy", "crashes", "overheats", "disappointed",
    "unusable", "defective", "waste", "underwhelming", "substandard", "scam"
}


NEGATION_WORDS = {"not", "never", "no", "neither", "nor", "barely", "hardly", "none", "cannot"}


def split_into_clauses(text: str) -> List[str]:
    """
    Split complex sentences along punctuation and contrastive clause boundaries.
    Preserves compound subjects/predicates joined by 'and'.
    """
    # Split by contrastive conjunctions (but, however, yet, although, while, whereas) and punctuation (,;.)
    clauses = re.split(r"[,;.]|\bbut\b|\bhowever\b|\byet\b|\balthough\b|\bwhile\b|\bwhereas\b", text, flags=re.IGNORECASE)
    return [c.strip() for c in clauses if c.strip()]


def _resolve_aspect_sentiment_in_clause(clause: str, aspect_pattern: str) -> tuple:
    """
    Resolve sentiment and negation status for a specific aspect in a clause.
    Uses token proximity when a clause contains mixed sentiment signals.
    """
    clause_clean = clean_text(clause)
    words = clause_clean.split()
    if not words:
        return "Neutral", False

    # Find position of aspect in token list
    aspect_match = re.search(aspect_pattern, clause, re.IGNORECASE)
    if not aspect_match:
        return "Neutral", False

    # Find approximate word index of aspect match
    aspect_text_match = clean_text(aspect_match.group(0)).split()
    aspect_word_idx = 0
    if aspect_text_match:
        for idx, w in enumerate(words):
            if w == aspect_text_match[0]:
                aspect_word_idx = idx
                break

    # Find all sentiment tokens with distance and negation status
    sentiment_candidates = []
    for idx, w in enumerate(words):
        is_pos = w in POSITIVE_INDICATORS
        is_neg = w in NEGATIVE_INDICATORS
        if is_pos or is_neg:
            # Check negation in preceding 3 words (e.g. "not very good", "not good")
            preceding_window = words[max(0, idx - 3):idx]
            has_neg = any(nw in preceding_window for nw in NEGATION_WORDS)
            
            base_sent = "Positive" if is_pos else "Negative"
            final_sent = ("Negative" if base_sent == "Positive" else "Positive") if has_neg else base_sent
            dist = abs(idx - aspect_word_idx)
            sentiment_candidates.append((dist, final_sent, has_neg))

    if not sentiment_candidates:
        # Check if clause has general negation
        has_neg = any(nw in words for nw in NEGATION_WORDS)
        return "Neutral", has_neg

    # Pick closest sentiment keyword to this aspect
    sentiment_candidates.sort(key=lambda x: x[0])
    _, chosen_sent, chosen_neg = sentiment_candidates[0]
    return chosen_sent, chosen_neg


def extract_aspect_sentiments(text: str) -> List[Dict[str, Any]]:
    """
    Extract entity aspects from text and assign sentiment to each aspect individually.
    
    Example:
    "The camera quality is excellent but the battery life is terrible."
    ->
    [
        {"aspect": "Camera / Image Quality", "sentiment": "Positive", "matched_text": "camera quality is excellent"},
        {"aspect": "Battery Life & Charging", "sentiment": "Negative", "matched_text": "battery life is terrible"}
    ]
    """
    if not text or not isinstance(text, str):
        return []

    clauses = split_into_clauses(text)
    detected_aspects = []
    seen_aspects = set()

    for clause in clauses:
        for aspect_name, patterns in ASPECT_PATTERNS.items():
            for pat in patterns:
                if re.search(pat, clause, re.IGNORECASE):
                    if aspect_name not in seen_aspects:
                        seen_aspects.add(aspect_name)
                        aspect_sent, has_neg = _resolve_aspect_sentiment_in_clause(clause, pat)
                        detected_aspects.append({
                            "aspect": aspect_name,
                            "sentiment": aspect_sent,
                            "clause": clause.strip(),
                            "negation_detected": has_neg
                        })
                    break

    return detected_aspects
