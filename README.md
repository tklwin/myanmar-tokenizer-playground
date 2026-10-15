# Myanmar Tokenizer Playground

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://myanmar-tokenizer-playground.streamlit.app/)

Interactive exploration of Unicode normalization, syllable and word segmentation, and subword tokenizers for the Myanmar (Burmese) language.

**Live Application:** [https://myanmar-tokenizer-playground.streamlit.app/](https://myanmar-tokenizer-playground.streamlit.app/)

Developed by [Thein Kyaw Lwin](https://tklwin.github.io/) with the help of Google Antigravity.

---

## Overview

Natural Language Processing for the Myanmar language presents distinct structural challenges compared to whitespace-delimited languages:
- **Absence of Word Delimiters**: Burmese text is written continuously without standard whitespace between words; spaces are used primarily for phrasal pauses.
- **Complex Unicode Grapheme Clusters**: Burmese syllables combine consonants, medials, vowels, anaphora, asat (killers), and virama-based stacked consonants (Pali consonant stacking / ပါဌ်ဆင့်).
- **Subword Syllable Fragmentation**: Standard subword tokenizers (BPE, Unigram, WordPiece) trained directly on raw character streams often break syllables across illegal boundaries, detaching dependent combining marks and producing invalid linguistic units.

The **Myanmar Tokenizer Playground** is a unified interactive laboratory built with Streamlit to visualize, analyze, and benchmark Myanmar text across four hierarchical tiers of granularity:
1. **Character Tier**: Unicode codepoints, categories, and Unicode Normalization Form C (NFC).
2. **Syllable Tier**: Rule-based syllable boundary segmentation using `sylbreak++`.
3. **Word Tier**: Rule-based linguistic word segmentation via `mmdt-tokenizer`'s 5-stage transformation pipeline.
4. **Subword Tier**: Production-grade subword models (SentencePiece BPE, SentencePiece Unigram, Hugging Face WordPiece) with character-span syllable integrity benchmarking.

---

## Key Features

- **Unicode Codepoint & NFC Inspector**:
  - Detailed breakdown of each character with hex codepoints and official Unicode standard names.
  - Safe combining mark visualization using dotted circle (`U+25CC`) to avoid rendering artifacts.
  - Live comparison between raw input and Unicode Normalization Form C (NFC), highlighting reordered combining marks.

- **4-Tier Granularity Hierarchy**:
  - Visual token pills for character, syllable, word, and subword levels.
  - Real-time token ID visualization (`[ID]`) for all subword representations.
  - Summary benchmark table calculating token counts, fertility ratios (tokens per word), characters per token, and boundary-accurate syllable integrity percentages.

- **mmdt-tokenizer 5-Stage Transformation Pipeline**:
  - Step-by-step interactive inspection of the entire linguistic pipeline:
    - Stage 1: Preprocessing and entity shielding (dates, numerals, URLs).
    - Stage 2: Syllable breaking with entity preservation.
    - Stage 3: Multi-trie lexicon scanning across 15 grammatical categories using longest-match-first priority.
    - Stage 4: Structural merges for negative predicates (e.g., `မ + စား + ခဲ့ + ပါ + ဘူး` -> `မစားခဲ့ပါဘူး`) and numerical classifiers.
    - Stage 5: Final buffer word collapsing into complete linguistic tokens.
  - **15 Trie Dictionaries Explorer**: Searchable and browsable inspection of 1,533 unique lexical entries across 15 categories (geographic regions, titles, postpositions, classifiers, conjunctions, pronouns, etc.).

- **Dual-Track Subword Models**:
  - Standard subword models trained on continuous raw text.
  - Syllable-aware subword models trained with syllable boundary preservation.
  - Precise mathematical syllable integrity evaluation based on character span boundaries.

---

## Project Structure

```text
myanmar-tokenizer-playground/
|-- app.py                     # Streamlit interactive web application
|-- train_tokenizers.py        # Reproducible subword training pipeline
|-- requirements.txt           # Python package dependencies
|-- LICENSE                    # MIT License
|-- README.md                  # Documentation
|-- data/
|   `-- myNews_train.raw.txt   # myNews training corpus (5,844 news paragraphs, ~316K words)
`-- models/
    |-- sp_bpe.model           # SentencePiece BPE model (4,000 vocab)
    |-- sp_bpe.vocab           # SentencePiece BPE vocabulary
    |-- sp_bpe_syl.model       # Syllable-aware SentencePiece BPE model
    |-- sp_bpe_syl.vocab       # Syllable-aware SentencePiece BPE vocabulary
    |-- sp_unigram.model       # SentencePiece Unigram model (4,000 vocab)
    |-- sp_unigram.vocab       # SentencePiece Unigram vocabulary
    |-- sp_unigram_syl.model   # Syllable-aware SentencePiece Unigram model
    |-- sp_unigram_syl.vocab   # Syllable-aware SentencePiece Unigram vocabulary
    |-- wordpiece.json         # Hugging Face WordPiece model (4,000 vocab)
    `-- wordpiece_syl.json     # Syllable-aware Hugging Face WordPiece model
```

---

## Installation & Setup

### Prerequisites
- Python 3.9, 3.10, 3.11, or 3.12
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/tklwin/myanmar-tokenizer-playground.git
cd myanmar-tokenizer-playground
```

### 2. Create and Activate a Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```
*(On Windows: `.venv\Scripts\activate`)*

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the Streamlit Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## Retraining Subword Tokenizers

To retrain or benchmark the subword models from the provided `myNews` corpus:

```bash
python train_tokenizers.py
```

This pipeline automatically:
1. Normalizes the dataset to Unicode Normalization Form C (NFC).
2. Generates both raw and syllable-segmented training corpora.
3. Trains SentencePiece BPE, SentencePiece Unigram, and Hugging Face WordPiece models with a 4,000-token budget and 99.95% character coverage.
4. Outputs models into the `models/` directory and evaluates syllable integrity across test samples.

---

## Specifications & Dataset

- **Corpus**: [myNews](https://huggingface.co/datasets/LULab/mynews) (Aung et al., 2025)
  - Paragraphs: 5,844
  - Words: ~316,000
  - Raw size: 4.4 MB
- **Subword Budget**: 4,000 tokens
- **Character Coverage**: 99.95%
- **MMDT Lexicon Tries**: 15 categories, 1,533 unique entries
  - REGION (291), SWORD (316), REG (273), CONJ (134), TITLE (111), CL (89), POSTP (60), SFP (51), MCONJ (50), DAY (33), VEP (31), PRN (28), MONTH (27), QW (23), CLEP (16).

---

## References

- Aung, T., Kyaw, E. K. K., Thu, Y. K., Myint Oo, T., & Supnithi, T. (2025). *Enhancing Burmese News Classification with Kolmogorov-Arnold Network Head Fine-tuning*. 20th International Joint Symposium on Artificial Intelligence and Natural Language Processing (iSAI-NLP).
- Kudo, T. (2018). *Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates*. ACL 2018.
- Kudo, T., & Richardson, J. (2018). *SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing*. EMNLP 2018.
- `mmdt-tokenizer`: Myanmar Multi-Dictionary and Rule-Based Tokenizer.

---

## Author & Attribution

- **Developer**: [Thein Kyaw Lwin](https://tklwin.github.io/)
- **Assisted by**: Google Antigravity
- **License**: [MIT License](LICENSE)

