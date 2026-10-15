#!/usr/bin/env python3
"""
Myanmar Tokenizer Playground: Interactive exploration of Unicode normalization, syllable and word segmentation, and subword tokenizers.
"""

import os
import re
import unicodedata
import itertools
import streamlit as st
import pandas as pd
from typing import List, Dict

try:
    import sentencepiece as spm
    from tokenizers import Tokenizer
    from mmdt_tokenizer import MyanmarTokenizer
    from mmdt_tokenizer.rule_segmenter.engine import (
        clean_wordnum_tag, merge_num_classifier, merge_predicate,
        clean_sfp_chunks, clean_cls_tag, clean_postp_tag, clean_chunks, _flatten_if_nested,
        SKIP, _check_pre_defined_tag, scan_longest_at, PIPELINE, Chunk
    )
    from mmdt_tokenizer.rule_segmenter.collapse import collapse_to_phrase_chunks
    from mmdt_tokenizer.tokenizer.syllable_tokenizer import (
        MyanmarSyllableTokenizer as MMDTSyllableTokenizer,
        SYLLABLE_BREAK_PATTERN
    )
    from mmdt_tokenizer.tokenizer.word_tokenizer import get_syllabus_from_tokenizer
    from mmdt_tokenizer.preprocessing import preprocess_burmese_text
except ImportError:
    st.error("Please install dependencies: pip install -r requirements.txt")
    st.stop()

# ---------------------------------------------------------
# Page Configuration (No Emojis)
# ---------------------------------------------------------
st.set_page_config(
    page_title="Myanmar Tokenizer Playground",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Minimalist, Clean Styling (Native Streamlit Theme Integration)
# ---------------------------------------------------------
st.html("""
<style>
    @font-face {
        font-family: 'Padauk';
        src: local('Padauk'), local('Padauk-Bold');
    }

    /* Minimalist Token Pill (Theme Adaptive) */
    .token-pill {
        display: inline-block;
        padding: 4px 10px;
        margin: 3px 2px;
        border-radius: 6px;
        font-family: 'Pyidaungsu', 'Padauk', 'Noto Sans Myanmar', 'Myanmar Text', sans-serif;
        font-size: 14.5px;
        font-weight: 600;
        background: rgba(128, 128, 128, 0.12);
        color: inherit !important;
        border: 1px solid rgba(128, 128, 128, 0.25);
        vertical-align: middle;
    }
    .token-id {
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        font-size: 11.5px;
        opacity: 0.65;
        font-weight: 500;
        margin-left: 4px;
        vertical-align: baseline;
    }
    .token-protected {
        background: rgba(245, 158, 11, 0.15) !important;
        color: inherit !important;
        border: 1px solid #d97706 !important;
        font-weight: 700;
    }
</style>
""")

# ---------------------------------------------------------
# Syllable Breaking & Boundary-Accurate Integrity Engine
# ---------------------------------------------------------
def get_syllables(text: str) -> List[str]:
    """Break Burmese text into syllables using sylbreak++ regex."""
    text_clean = re.sub(r"\s+", " ", text.strip())
    delim = "\u200b"
    segmented = SYLLABLE_BREAK_PATTERN.sub(delim + r"\1", text_clean)
    if segmented.startswith(delim):
        segmented = segmented[len(delim):]
    return [t.strip() for t in segmented.split(delim) if t.strip()]

def check_syllable_integrity(text: str, token_pieces: List[str], prefix_marker: str = "sp") -> Dict:
    """
    Mathematically precise syllable integrity check based on character boundary spans.
    A syllable span (start, end) in text is broken iff a token boundary falls strictly inside it.
    """
    syllables = get_syllables(text)

    # 1. Map character spans of each syllable in text
    syl_spans = []
    curr = 0
    for s in syllables:
        idx = text.find(s, curr)
        if idx != -1:
            syl_spans.append((idx, idx + len(s), s))
            curr = idx + len(s)
        else:
            syl_spans.append((-1, -1, s))

    # 2. Extract token boundaries
    tok_boundaries = set()
    curr_pos = 0
    for t in token_pieces:
        clean_t = t.replace(" ", "").replace("▁", "") if prefix_marker == "sp" else t.replace("##", "")
        if not clean_t:
            continue
        idx = text.find(clean_t, curr_pos)
        if idx != -1:
            tok_boundaries.add(idx)
            tok_boundaries.add(idx + len(clean_t))
            curr_pos = idx + len(clean_t)

    # 3. Check for boundaries strictly inside syllable spans
    broken_syls = []
    matched = 0
    for start, end, s in syl_spans:
        if start == -1:
            broken_syls.append(s)
            continue
        is_broken = any(start < b < end for b in tok_boundaries)
        if is_broken:
            broken_syls.append(s)
        else:
            matched += 1

    total = len(syllables)
    pct = (matched / total * 100.0) if total > 0 else 100.0
    return {
        "integrity_pct": pct,
        "broken": broken_syls,
        "total_syllables": total,
        "intact_count": matched
    }

def show_dataframe(df, height=None, hide_index=True):
    """Safely render dataframes across all Streamlit versions without deprecation warnings."""
    kwargs = {"hide_index": hide_index}
    if height is not None:
        kwargs["height"] = height
    try:
        st.dataframe(df, width="stretch", **kwargs)
    except TypeError:
        st.dataframe(df, use_container_width=True, **kwargs)

def format_token_display(t: str) -> str:
    """Prepend dotted circle (U+25CC) to isolated combining marks so they render without tofu boxes."""
    if t.startswith("##"):
        prefix = "##"
        body = t[2:]
    elif t.startswith("\u2581") or t.startswith(" "):
        prefix = t[0]
        body = t[1:]
    else:
        prefix = ""
        body = t

    if body:
        c0 = body[0]
        cat0 = unicodedata.category(c0)
        if cat0.startswith("M") or ord(c0) in (0x1039, 0x103A, 0x103B, 0x103C, 0x103D, 0x103E):
            return prefix + "\u25CC" + body
    return t

def render_tokens(tokens: List[str], token_type: str = "sp", ids: List[int] = None) -> str:
    html_parts = []
    for i, t in enumerate(tokens):
        disp = format_token_display(t)
        id_badge = f'<span class="token-id">[{ids[i]}]</span>' if (ids and i < len(ids)) else ""
        html_parts.append(f'<span class="token-pill">{disp}{id_badge}</span>')
    return "".join(html_parts)

# ---------------------------------------------------------
# Model Loading (Cached)
# ---------------------------------------------------------
@st.cache_resource
def load_models(syllable_aware: bool = False):
    suffix = "_syl" if syllable_aware else ""
    bpe_path = f"models/sp_bpe{suffix}.model"
    uni_path = f"models/sp_unigram{suffix}.model"
    wp_path = f"models/wordpiece{suffix}.json"

    if not os.path.exists(bpe_path) or not os.path.exists(uni_path) or not os.path.exists(wp_path):
        bpe_path = "models/sp_bpe.model"
        uni_path = "models/sp_unigram.model"
        wp_path = "models/wordpiece.json"

    sp_bpe = spm.SentencePieceProcessor()
    sp_bpe.load(bpe_path)

    sp_uni = spm.SentencePieceProcessor()
    sp_uni.load(uni_path)

    wp = Tokenizer.from_file(wp_path)
    return sp_bpe, sp_uni, wp

TRIE_DESCRIPTIONS = {
    "REGION": "Geographic Regions & Countries",
    "SWORD": "Special Compound Words",
    "REG": "Religious & Heritage Landmarks",
    "CONJ": "Conjunctions",
    "TITLE": "Official Titles & Designations",
    "CL": "Numerical Classifiers",
    "POSTP": "Postpositions (Grammar Particles)",
    "SFP": "Sentence-Final Particles (Politeness)",
    "MCONJ": "Multi-word Conjunctions",
    "DAY": "Day & Time Expressions",
    "VEP": "Verb Enclitics & Modals",
    "PRN": "Personal Pronouns",
    "MONTH": "Burmese Calendar Months",
    "QW": "Question Words",
    "CLEP": "Classifier Enclitics & Quantifiers"
}

@st.cache_data
def load_mmdt_trie_vocabs() -> Dict[str, List[str]]:
    """Extract and sort all unique words stored in MMDT's 15 Trie trees."""
    def _extract_words(trie, prefix=""):
        words = []
        for k, v in trie.items():
            if k == "_END_":
                words.append(prefix)
            elif isinstance(v, dict):
                words.extend(_extract_words(v, prefix + k))
        return sorted(set(words))

    return {tag: _extract_words(trie) for trie, tag in PIPELINE}

# ---------------------------------------------------------
# Sidebar Controls & Presets
# ---------------------------------------------------------
st.sidebar.markdown("### Controls")

presets = {
    "1. Stacked Virama (ပါဌ်ဆင့်)": "တက္ကသိုလ်ကျောင်းသားများ",
    "2. Medial & Killer (ဗျည်းတွဲ အသတ်)": "မြန်မာ့ယဉ်ကျေးမှုသည် ကမ္ဘာတွင် ထင်ရှားသည်။",
    "3. Predicate Merging (အငြင်း ကြိယာတွဲ)": "သူ ထမင်း မစားခဲ့ပါဘူး။",
    "4. Protected Entities (ရက်စွဲ၊ ရာထူး၊ နေရာ)": "သမ္မတဦးဝင်းမြင့်သည် ၂၀၁၈ ခုနှစ် မတ်လ ၃ ရက်နေ့တွင် ရန်ကုန်မြို့သို့ သွားရောက်ခဲ့သည်။",
    "5. Normalization Order (စဉ့် vs စဉ့်)": "\u1005\u1009\u103A\u1037ကူးမင်းလက်ထက် မြန်မာ့ယဉ်ကျေးမှု",
}

selected_preset = st.sidebar.selectbox("Presets", list(presets.keys()), index=3)
default_text = presets[selected_preset]

input_text = st.sidebar.text_area(
    "Input Text:",
    value=default_text,
    height=235,
    help="Type or paste any Myanmar sentence."
)

syl_mode = st.sidebar.checkbox(
    "Syllable-Aware Pre-tokenization",
    value=False,
    help="Pre-segments text with sylbreak++ before passing to BPE / Unigram / WordPiece."
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="font-size: 13px; opacity: 0.85; line-height: 1.5;">
  Developed by <a href="https://tklwin.github.io/" target="_blank" style="text-decoration: underline; font-weight: 600;">Thein Kyaw Lwin</a><br>
  <span style="font-size: 11px; opacity: 0.65;">with the help of Google Antigravity</span>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Prepare Inputs & Tokenization
# ---------------------------------------------------------
text = input_text.strip()
if not text:
    st.warning("Please provide input text in the sidebar.")
    st.stop()

sp_bpe, sp_uni, wp = load_models(syllable_aware=syl_mode)

eval_text = text
if syl_mode:
    eval_text = " ".join(get_syllables(text))

bpe_tokens = sp_bpe.encode(eval_text, out_type=str)
bpe_ids = sp_bpe.encode(eval_text, out_type=int)

uni_tokens = sp_uni.encode(eval_text, out_type=str)
uni_ids = sp_uni.encode(eval_text, out_type=int)

wp_enc = wp.encode(eval_text)
wp_tokens = wp_enc.tokens
wp_ids = wp_enc.ids

syllables = get_syllables(text)

bpe_integ = check_syllable_integrity(text, bpe_tokens, "sp")
uni_integ = check_syllable_integrity(text, uni_tokens, "sp")
wp_integ = check_syllable_integrity(text, wp_tokens, "wp")

char_count = len(text.replace(" ", ""))

# ---------------------------------------------------------
# Header & Metrics
# ---------------------------------------------------------
st.markdown("## Myanmar Tokenizer Playground")
st.caption("Interactive exploration of Unicode normalization, syllable and word segmentation, and subword tokenizers.")

# ---------------------------------------------------------
# 4 Clean Streamlined Tabs
# ---------------------------------------------------------
tab_unicode, tab_seg, tab_mmdt, tab_corpus = st.tabs([
    "[01] UNICODE & NFC INSPECTOR",
    "[02] TOKENIZER PLAYGROUND",
    "[03] mmdt-tokenizer PIPELINE",
    "[04] SPECIFICATIONS"
])

# =========================================================
# TAB 1: Unicode Inspector & NFC
# =========================================================
with tab_unicode:
    st.markdown("#### Unicode Codepoint Table")
    st.caption("Character breakdown with hex codepoints and official Unicode standard names.")

    char_map = {}
    table_rows = []
    char_num = 1
    for i, c in enumerate(text):
        if c == " ":
            continue
        code = ord(c)
        cat = unicodedata.category(c)
        # Prepend dotted circle (U+25CC) for combining marks so they render properly in isolation
        is_combining = cat.startswith("M") or code in (0x1039, 0x103A, 0x103B, 0x103C, 0x103D, 0x103E)
        display_char = ("\u25CC" + c) if is_combining else c
        name = unicodedata.name(c, f"MYANMAR CHARACTER (U+{code:04X})")

        char_map[i] = char_num
        table_rows.append({
            "#": char_num,
            "Codepoint": f"U+{code:04X}",
            "Character": display_char,
            "Name": name
        })
        char_num += 1

    show_dataframe(pd.DataFrame(table_rows), height=320, hide_index=True)

    st.markdown("---")
    st.markdown("#### Unicode Normalization Form C (NFC)")
    nfc_text = unicodedata.normalize("NFC", text)
    if text == nfc_text:
        st.success("Canonical Match: Input is already compliant with Unicode NFC standard.")
    else:
        st.warning("Reordered: Bytes were normalized to canonical order.")
        diffs = []
        for i, (r, n) in enumerate(itertools.zip_longest(text, nfc_text, fillvalue="")):
            if r != n:
                if r:
                    code_r = f"U+{ord(r):04X}"
                    cat_r = unicodedata.category(r)
                    is_comb_r = cat_r.startswith("M") or ord(r) in (0x1039, 0x103A, 0x103B, 0x103C, 0x103D, 0x103E)
                    disp_r = ("\u25CC" + r) if is_comb_r else r
                    name_r = unicodedata.name(r, code_r)
                else:
                    code_r = "-"
                    disp_r = "-"
                    name_r = "[Inserted by NFC]"

                if n:
                    code_n = f"U+{ord(n):04X}"
                    cat_n = unicodedata.category(n)
                    is_comb_n = cat_n.startswith("M") or ord(n) in (0x1039, 0x103A, 0x103B, 0x103C, 0x103D, 0x103E)
                    disp_n = ("\u25CC" + n) if is_comb_n else n
                else:
                    code_n = "-"
                    disp_n = "-"

                diffs.append({
                    "#": char_map.get(i, i + 1),
                    "Raw Codepoint": code_r,
                    "Raw Character": disp_r,
                    "NFC Codepoint": code_n,
                    "NFC Character": disp_n,
                    "Character Name": name_r
                })
        show_dataframe(pd.DataFrame(diffs), hide_index=True)

# =========================================================
# TAB 2: Tokenizer Playground (4-Tier Granularity Hierarchy)
# =========================================================
with tab_seg:
    # 1. Character-Level
    chars = [c for c in text if c != " "]
    st.markdown("**1. Character-Level (Unicode Codepoints)**")
    st.caption(f"{len(chars)} characters (excluding whitespace)")
    st.markdown(render_tokens(chars, "char"), unsafe_allow_html=True)

    # 2. Syllable-Level
    st.markdown("**2. Syllable-Level (sylbreak++)**")
    st.caption(f"{len(syllables)} syllables")
    st.markdown(render_tokens(syllables, "syl"), unsafe_allow_html=True)

    # 3. Word-Level
    mmdt_inst = MyanmarTokenizer()
    mmdt_words_list = mmdt_inst.word_tokenize(text)
    mmdt_words = mmdt_words_list[0] if mmdt_words_list else []
    st.markdown("**3. Word-Level (mmdt-tokenizer)**")
    st.caption(f"{len(mmdt_words)} linguistic words")
    st.markdown(render_tokens(mmdt_words, "syl"), unsafe_allow_html=True)

    # 4. Subword-Level
    st.markdown("**4. Subword-Level (Vocab: 4,000)**")
    c_bpe, c_uni, c_wp = st.columns(3)
    with c_bpe:
        st.caption(f"SentencePiece BPE ({len(bpe_tokens)} tokens)")
        st.markdown(render_tokens(bpe_tokens, "sp", ids=bpe_ids), unsafe_allow_html=True)
    with c_uni:
        st.caption(f"SentencePiece Unigram ({len(uni_tokens)} tokens)")
        st.markdown(render_tokens(uni_tokens, "sp", ids=uni_ids), unsafe_allow_html=True)
    with c_wp:
        st.caption(f"WordPiece / BERT ({len(wp_tokens)} tokens)")
        st.markdown(render_tokens(wp_tokens, "wp", ids=wp_ids), unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### Summary Benchmark Table")
    n_words = max(1, len(mmdt_words))
    summary_data = [
        {
            "Granularity Tier": "Subword",
            "Model / Tool": "SentencePiece BPE",
            "Tokens": len(bpe_tokens),
            "Fertility (Tok/Word)": f"{len(bpe_tokens)/n_words:.2f}",
            "Chars/Token": f"{char_count/max(1, len(bpe_tokens)):.2f}",
            "Syllable Intact %": f"{bpe_integ['integrity_pct']:.1f}%",
            "Broken Syllables": ", ".join(sorted(set(bpe_integ["broken"]))) if bpe_integ["broken"] else "None (100% Intact)"
        },
        {
            "Granularity Tier": "Subword",
            "Model / Tool": "SentencePiece Unigram",
            "Tokens": len(uni_tokens),
            "Fertility (Tok/Word)": f"{len(uni_tokens)/n_words:.2f}",
            "Chars/Token": f"{char_count/max(1, len(uni_tokens)):.2f}",
            "Syllable Intact %": f"{uni_integ['integrity_pct']:.1f}%",
            "Broken Syllables": ", ".join(sorted(set(uni_integ["broken"]))) if uni_integ["broken"] else "None (100% Intact)"
        },
        {
            "Granularity Tier": "Subword",
            "Model / Tool": "WordPiece (BERT)",
            "Tokens": len(wp_tokens),
            "Fertility (Tok/Word)": f"{len(wp_tokens)/n_words:.2f}",
            "Chars/Token": f"{char_count/max(1, len(wp_tokens)):.2f}",
            "Syllable Intact %": f"{wp_integ['integrity_pct']:.1f}%",
            "Broken Syllables": ", ".join(sorted(set(wp_integ["broken"]))) if wp_integ["broken"] else "None (100% Intact)"
        },
        {
            "Granularity Tier": "Word",
            "Model / Tool": "mmdt-tokenizer",
            "Tokens": len(mmdt_words),
            "Fertility (Tok/Word)": "1.00",
            "Chars/Token": f"{char_count/max(1, len(mmdt_words)):.2f}",
            "Syllable Intact %": "100.0%",
            "Broken Syllables": "None (100% Intact)"
        },
        {
            "Granularity Tier": "Syllable",
            "Model / Tool": "sylbreak++",
            "Tokens": len(syllables),
            "Fertility (Tok/Word)": f"{len(syllables)/n_words:.2f}",
            "Chars/Token": f"{char_count/max(1, len(syllables)):.2f}",
            "Syllable Intact %": "100.0%",
            "Broken Syllables": "None (100% Intact)"
        },
        {
            "Granularity Tier": "Character",
            "Model / Tool": "Unicode Codepoints",
            "Tokens": len(chars),
            "Fertility (Tok/Word)": f"{len(chars)/n_words:.2f}",
            "Chars/Token": "1.00",
            "Syllable Intact %": "—",
            "Broken Syllables": "—"
        },
    ]
    show_dataframe(pd.DataFrame(summary_data), hide_index=True)

# =========================================================
# TAB 3: mmdt-tokenizer Pipeline (All 5 Complete Steps)
# =========================================================
with tab_mmdt:
    st.markdown("#### mmdt-tokenizer: 5-Stage Linguistic Pipeline")
    st.caption("Complete step-by-step breakdown across all 5 architectural transformation stages.")

    # Stage 1: Preprocessing & Protected Patterns
    phrase_tokens, protected_dict = preprocess_burmese_text(text)
    with st.expander("Stage 1: Pattern Protection & Entity Shielding", expanded=True):
        if protected_dict:
            masked_stream = " ".join([p if p not in protected_dict else f"<{repr(p)[1:-1]}>" for p in phrase_tokens])
            st.markdown(f"**Masked Stream:** `{masked_stream}`")
            prot_df = pd.DataFrame([{"Placeholder Byte": repr(k), "Protected Entity": v} for k, v in protected_dict.items()])
            show_dataframe(prot_df, hide_index=True)
        else:
            st.caption("No special entity (date, phone, URL) detected. Text proceeds directly as plain text.")

    # Stage 2: Syllable Breaking with Shielding
    syl_tokenizer = MMDTSyllableTokenizer()
    get_syl = get_syllabus_from_tokenizer(syl_tokenizer)
    with st.expander("Stage 2: Syllable Segmentation (sylbreak++) with Entity Shielding", expanded=True):
        stage2_html = []
        for p in phrase_tokens:
            if p in protected_dict:
                stage2_html.append(f'<span class="token-pill token-protected">[PROTECTED: {protected_dict[p]}]</span>')
            else:
                syls = _flatten_if_nested(get_syl(p))
                for s in syls:
                    stage2_html.append(f'<span class="token-pill">{s}</span>')
        st.markdown("".join(stage2_html), unsafe_allow_html=True)

    # Stage 3: Multi-Trie Lexicon Scanning (15 Categories with Longest Match)
    tokens = []
    for phrase in phrase_tokens:
        if phrase in protected_dict:
            tokens.append(protected_dict[phrase])
        else:
            syllable_tokens = _flatten_if_nested(get_syl(phrase))
            syllable_tokens.append(' ')
            tokens.extend(syllable_tokens)

    pure_stage3 = []
    i = 0
    n = len(tokens)
    while i < n:
        t = tokens[i]
        if t in SKIP:
            pure_stage3.append(Chunk((i, i), t, "PUNCT"))
            i += 1
            continue
        tag = _check_pre_defined_tag(t)
        if tag:
            pure_stage3.append(Chunk((i, i), t, tag))
            i += 1
            continue
        m = scan_longest_at(tokens, i, PIPELINE)
        if m:
            pure_stage3.append(m)
            i = m.span[1] + 1
        else:
            pure_stage3.append(Chunk((i, i), t, "RAW"))
            i += 1

    with st.expander("Stage 3: Multi-Trie Lexicon Scanning (15 Categories with Longest Match)", expanded=True):
        st.caption("Scans 15 Trie dictionaries with longest-match-first priority. Notice individual syllables before syntactic fusing.")
        chunk_df = pd.DataFrame([{"Piece": c.text, "Grammar Tag": c.tag} for c in pure_stage3 if c.text.strip()])
        if not chunk_df.empty:
            show_dataframe(chunk_df.T, hide_index=False)
        else:
            st.caption("No tokens scanned.")

    # Stage 4: Structural Merges (Predicate & Classifier Merging)
    c1 = clean_wordnum_tag(pure_stage3)
    c2 = merge_num_classifier(c1)
    c3 = merge_predicate(c2)
    c4 = clean_sfp_chunks(c3)
    c5 = clean_cls_tag(c4)
    c6 = clean_postp_tag(c5)
    stage4_chunks = clean_chunks(c6)

    with st.expander("Stage 4: Structural Merging (Predicates & Classifiers)", expanded=True):
        st.caption("Fuses grammatical structures: negative predicates (e.g. မ + စား + ခဲ့ + ပါ + ဘူး -> မစားခဲ့ပါဘူး : PRED) and classifiers.")
        stage4_df = pd.DataFrame([{"Unit": c.text, "Syntactic Tag": c.tag} for c in stage4_chunks if c.text.strip()])
        if not stage4_df.empty:
            show_dataframe(stage4_df.T, hide_index=False)
        else:
            st.caption("No merged structures.")

    # Stage 5: Final Buffer & Word Collapsing
    _, all_tokens = collapse_to_phrase_chunks([stage4_chunks])
    stage5_words = all_tokens[0] if all_tokens else mmdt_words

    with st.expander("Stage 5: Final Buffer & Word Collapsing", expanded=True):
        st.caption("Flushes buffered RAW chunks into complete linguistic words upon encountering functional tags.")
        st.markdown(render_tokens(stage5_words, "syl"), unsafe_allow_html=True)

    st.markdown("---")
    with st.expander("Explore MMDT 15 Trie Dictionaries (1,533 Entries)", expanded=False):
        st.caption("Inspect the rule-based lexical trees that drive Stage 3 longest-match token scanning.")
        trie_vocabs = load_mmdt_trie_vocabs()
        col_cat, col_filter = st.columns([1, 1])
        with col_cat:
            selected_cat = st.selectbox(
                "Select Lexicon Category:",
                options=list(trie_vocabs.keys()),
                index=4,
                format_func=lambda c: f"{c} ({len(trie_vocabs[c])} entries) — {TRIE_DESCRIPTIONS.get(c, '')}"
            )
        with col_filter:
            filter_query = st.text_input(
                "Filter entries in category:",
                placeholder="e.g. သမ္မတ or ညွှန်ကြား",
                key="trie_filter_input"
            )

        cat_words = trie_vocabs[selected_cat]
        if filter_query.strip():
            display_words = [w for w in cat_words if filter_query.strip() in w]
            st.caption(f"Showing **{len(display_words)}** of {len(cat_words)} entries in `{selected_cat}` matching '{filter_query.strip()}':")
        else:
            display_words = cat_words
            st.caption(f"All **{len(cat_words)}** entries in `{selected_cat}` ({TRIE_DESCRIPTIONS.get(selected_cat, '')}):")

        if display_words:
            st.markdown(
                f'<div style="max-height: 260px; overflow-y: auto; padding: 10px; border: 1px solid rgba(128,128,128,0.2); border-radius: 6px; background: rgba(128,128,128,0.04);">'
                f'{render_tokens(display_words, "syl")}'
                f'</div>',
                unsafe_allow_html=True
            )
        else:
            st.info(f"No entries in `{selected_cat}` matched '{filter_query.strip()}'.")

# =========================================================
# TAB 4: Specifications
# =========================================================
with tab_corpus:
    st.markdown("#### Dataset & Subword Specifications")
    st.markdown("""
    - **Training Dataset:** [myNews](https://huggingface.co/datasets/LULab/mynews) (5,844 news paragraphs, ~316K words, Aung et al., 2025)
    - **Subword Budget:** 4,000 tokens | 99.95% Character Coverage
    - **Tokenizers & Architectures:**
      - **SentencePiece BPE** *(GPT-4, LLaMA 3, Mistral)*
      - **SentencePiece Unigram** *(Gemma, T5, ALBERT)*
      - **Hugging Face WordPiece** *(BERT, mBERT)*
    """)
