"""
asl_gloss.py
============
Converts plain English sentences into simplified ASL gloss — the word-order
and grammar used in American Sign Language.

Key rules applied:
  1. Lowercase + strip punctuation
  2. Remove: articles (a, an, the)
  3. Remove: be-verb copulas (am, is, are, was, were, be, been, being)
     — unless preceded by "not" (to preserve negation)
  4. Remove: auxiliary/modal verbs (do, does, did, have, has, had, will,
     would, could, should, shall, might, may, must, can)
  5. Remove: infinitive 'to' when followed by a verb
  6. Remove: filler words (just, very, really, quite, rather, so, still)
  7. Expand common contractions (I'm → I, you're → YOU, etc.)
  8. Reorder to Topic-Comment where possible:
     Subject … Object … Verb  (simple SVO→SOV for action sentences)
  9. Map personal pronouns to ASL forms (I→ME, my→MY, he→HE, etc.)
  10. Map common English words that have direct ASL equivalents but differ
      in spelling (want to → WANT, going to → GO, etc.)

Note: This is a rule-based approximation. For full accuracy you would need
an NLP parser (spaCy) and a proper ASL grammar engine. This implementation
covers the most common patterns without external dependencies.
"""

import re

# ── Contraction expansion ──────────────────────────────────────────────────────
CONTRACTIONS = {
    "i'm":     "i",
    "i've":    "i have",
    "i'll":    "i will",
    "i'd":     "i would",
    "you're":  "you",
    "you've":  "you have",
    "you'll":  "you will",
    "you'd":   "you would",
    "he's":    "he",
    "she's":   "she",
    "it's":    "it",
    "we're":   "we",
    "we've":   "we have",
    "we'll":   "we will",
    "we'd":    "we would",
    "they're": "they",
    "they've": "they have",
    "they'll": "they will",
    "they'd":  "they would",
    "that's":  "that",
    "there's": "there",
    "here's":  "here",
    "what's":  "what",
    "isn't":   "not",
    "aren't":  "not",
    "wasn't":  "not",
    "weren't": "not",
    "can't":   "cannot",
    "won't":   "will not",
    "don't":   "not",
    "doesn't": "not",
    "didn't":  "not",
    "haven't": "not have",
    "hasn't":  "not have",
    "hadn't":  "not have",
    "wouldn't":"would not",
    "couldn't":"could not",
    "shouldn't":"should not",
    "let's":   "let us",
    "gonna":   "go",
    "wanna":   "want",
    "gotta":   "must",
    "kinda":   "kind of",
    "sorta":   "sort of",
}

# ── Words to drop entirely ─────────────────────────────────────────────────────
ARTICLES    = {"a", "an", "the"}
BE_VERBS    = {"am", "is", "are", "was", "were", "be", "been", "being"}
AUX_VERBS   = {"do", "does", "did", "have", "has", "had",
               "will", "would", "could", "should", "shall",
               "might", "may", "shall"}
FILLER_WORDS = {"just", "very", "really", "quite", "rather", "so",
                "still", "even", "only", "also", "too", "already",
                "then", "well", "actually", "basically", "literally"}

# Prepositions, conjunctions, relative pronouns — rarely signed individually
PREPOSITIONS = {
    "of", "in", "on", "at", "to", "for", "with", "by", "from",
    "into", "onto", "upon", "within", "without", "through", "during",
    "before", "after", "between", "among", "against", "about",
    "above", "below", "under", "over", "around", "along", "across",
    "behind", "beside", "near", "off", "out", "up", "down", "per",
    "via", "than", "despite", "except", "like",
}
CONJUNCTIONS = {
    "and", "or", "but", "nor", "yet", "so", "although", "because",
    "since", "unless", "while", "whereas", "if", "though", "whether",
    "both", "either", "neither", "as", "when", "where",
}
REL_WORDS = {
    "that", "which", "who", "whom", "whose", "this", "these",
    "those", "such",
}

DROP_WORDS = ARTICLES | BE_VERBS | AUX_VERBS | FILLER_WORDS | PREPOSITIONS | CONJUNCTIONS | REL_WORDS

# ── Pronoun mapping (English → ASL form) ──────────────────────────────────────
PRONOUN_MAP = {
    "i":       "me",
    "my":      "my",
    "mine":    "mine",
    "myself":  "myself",
    "he":      "he",
    "his":     "his",
    "him":     "him",
    "himself": "himself",
    "she":     "she",
    "her":     "her",
    "hers":    "hers",
    "herself": "herself",
    "we":      "we",
    "our":     "our",
    "ours":    "ours",
    "us":      "us",
    "they":    "they",
    "their":   "their",
    "them":    "them",
    "it":      "it",
    "its":     "its",
    "you":     "you",
    "your":    "your",
    "yours":   "yours",
    "yourself":"yourself",
}

# ── Common English phrases mapped to ASL token(s) ─────────────────────────────
# Checked as bigrams/trigrams BEFORE splitting
PHRASE_MAP = [
    # Order matters: longer phrases first
    ("going to",      "go"),
    ("want to",       "want"),
    ("need to",       "need"),
    ("have to",       "must"),
    ("has to",        "must"),
    ("able to",       "can"),
    ("not able to",   "cannot"),
    ("a lot",         "many"),
    ("a lot of",      "many"),
    ("kind of",       "sort"),
    ("sort of",       "sort"),
    ("right now",     "now"),
    ("how are you",   "you fine"),
    ("how do you do", "you fine"),
    ("thank you",     "thank you"),
    ("nice to meet",  "nice meet"),
    ("nice meeting",  "nice meet"),
    ("what is your name", "your name what"),
    ("my name is",    "my name"),
    ("i am",          "me"),
    ("you are",       "you"),
    ("he is",         "he"),
    ("she is",        "she"),
    ("we are",        "we"),
    ("they are",      "they"),
    ("there is",      "have"),
    ("there are",     "have"),
]


def _expand_contractions(text: str) -> str:
    for contraction, expansion in CONTRACTIONS.items():
        text = re.sub(r'\b' + re.escape(contraction) + r'\b', expansion, text, flags=re.IGNORECASE)
    return text


def _apply_phrase_map(text: str) -> str:
    for phrase, replacement in PHRASE_MAP:
        text = re.sub(r'\b' + re.escape(phrase) + r'\b', replacement, text, flags=re.IGNORECASE)
    return text


def english_to_asl(text: str) -> list:
    """
    Convert an English sentence to a list of ASL gloss tokens.

    Returns a list of sign words in ASL word-order.
    Example:
        "I am going to the store" → ["me", "store", "go"]
        "Are you hungry?"         → ["you", "hungry"]
    """
    # Step 1: lowercase + expand contractions
    text = text.lower().strip()
    text = _expand_contractions(text)

    # Step 2: strip punctuation (keep hyphens inside words)
    text = re.sub(r"[^\w\s\-]", " ", text)

    # Step 3: apply phrase-level substitutions
    text = _apply_phrase_map(text)

    # Step 4: tokenise
    tokens = text.split()

    # Step 5: filter drop-words and map pronouns, handle 'to' before verbs
    cleaned = []
    skip_next_to = False
    for i, tok in enumerate(tokens):
        if tok == "to" and i + 1 < len(tokens):
            # Drop infinitive 'to' (before the next word)
            continue
        if tok == "not":
            cleaned.append("not")
            continue
        if tok in DROP_WORDS:
            continue
        # Map pronoun
        tok = PRONOUN_MAP.get(tok, tok)
        if tok:
            cleaned.append(tok)

    # Step 6: simple SOV reorder for short sentences (subject + verb + object)
    # Heuristic: if first token is a pronoun and last token looks like a verb,
    # move the object between subject and verb.
    # This is a very light-touch reorder — full reordering requires an NLP parser.
    cleaned = _light_sov_reorder(cleaned)

    return cleaned


def _light_sov_reorder(tokens: list) -> list:
    """
    Very simple Subject-Object-Verb reorder for common patterns.
    Only fires when: len >= 3, first token is a known pronoun,
    and there are recognisable content words in the middle.
    Falls back to original order if pattern not detected.
    """
    asl_pronouns = {"me", "you", "he", "she", "we", "they", "it"}
    # Common action / stative verbs that typically come at end in ASL
    asl_verbs = {
        "go", "eat", "drink", "buy", "want", "like", "love", "hate",
        "see", "know", "think", "feel", "have", "need", "give", "take",
        "help", "meet", "learn", "study", "finish", "start", "work",
        "play", "run", "walk", "sleep", "wake", "read", "write", "cook",
        "make", "come", "leave", "return", "find", "lose", "tell", "ask",
        "show", "watch", "listen", "talk", "speak", "sign"
    }

    if len(tokens) < 3:
        return tokens

    # If last token is a verb and first is a pronoun → already good for SOV
    if tokens[0] in asl_pronouns and tokens[-1] in asl_verbs:
        return tokens  # already SOV-ish

    # If first token is a pronoun and second or third is a verb → move verb to end
    if tokens[0] in asl_pronouns:
        for idx in range(1, min(3, len(tokens))):
            if tokens[idx] in asl_verbs and idx < len(tokens) - 1:
                verb = tokens.pop(idx)
                tokens.append(verb)
                break

    return tokens


def asl_gloss_string(text: str) -> str:
    """Return the ASL gloss as a readable uppercase string."""
    return "  ".join(t.upper() for t in english_to_asl(text))
