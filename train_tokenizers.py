#!/usr/bin/env python3
"""
Myanmar Subword Tokenizers: Production-Grade Training Pipeline
=============================================================
A self-contained, reproducible training suite for Myanmar NLP subword tokenizers:
1. Unicode NFC Normalization (Canonical character ordering)
2. Enhanced Syllable Segmentation (sylbreak++)
3. Dual-Track Subword Training:
   - Standard Subword Models: BPE, Unigram, WordPiece (4,000 vocab)
   - Syllable-Aware Models:   BPE, Unigram, WordPiece (4,000 vocab, bounded by sylbreak++)
4. Automated Syllable Integrity & Coverage Benchmarking

Dataset: myNews (Aung et al., 2025)
License: MIT
"""

import os
import re
import time
import argparse
import unicodedata
from typing import List, Dict
import sentencepiece as spm
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from mmdt_tokenizer.tokenizer.syllable_tokenizer import SYLLABLE_BREAK_PATTERN

DELIM = "\u200b"

def normalize_burmese_text(text: str) -> str:
    """Canonicalize text to Unicode Normalization Form C (NFC) and normalize spaces."""
    # 1. Unicode NFC canonical decomposition followed by canonical composition
    text = unicodedata.normalize("NFC", text.strip())
    # 2. Collapse redundant whitespace but preserve newlines
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()

def segment_syllables(text: str) -> List[str]:
    """Break Burmese text into syllables using sylbreak++ (handles Aukmyit-Asat clusters)."""
    text_clean = normalize_burmese_text(text)
    segmented = SYLLABLE_BREAK_PATTERN.sub(DELIM + r"\1", text_clean)
    if segmented.startswith(DELIM):
        segmented = segmented[len(DELIM):]
    return [t.strip() for t in segmented.split(DELIM) if t.strip()]

def prepare_corpora(raw_file: str, clean_file: str, syl_file: str):
    """
    Process raw myNews dataset:
    - Clean TSV labels and extract raw text sentences.
    - Normalize all sentences to Unicode NFC.
    - Generate two training corpora:
      1. clean_file: Standard continuous text (NFC normalized).
      2. syl_file: Syllable-segmented text (sylbreak++ with space boundaries).
    """
    print(f"\n[1/3] Preparing Corpora from: {raw_file}")
    if not os.path.exists(raw_file):
        raise FileNotFoundError(f"Input file not found: {raw_file}")

    with open(raw_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    clean_sentences = []
    syl_sentences = []

    for line in lines:
        parts = line.strip().split("\t")
        # Extract text field (myNews raw format: __label__category \t text)
        text = parts[1].strip() if len(parts) >= 2 else line.strip()
        # Remove any residual fasttext label tags
        text = re.sub(r"^__label__\S+\s*", "", text)
        norm_text = normalize_burmese_text(text)
        if not norm_text:
            continue

        clean_sentences.append(norm_text)
        # Syllable pre-tokenized line (space between each syllable)
        syls = segment_syllables(norm_text)
        syl_sentences.append(" ".join(syls))

    print(f"  Extracted & normalized {len(clean_sentences):,} sentences.")

    # Save NFC cleaned corpus
    os.makedirs(os.path.dirname(clean_file), exist_ok=True)
    with open(clean_file, "w", encoding="utf-8") as f:
        f.write("\n".join(clean_sentences) + "\n")
    print(f"  Saved clean NFC corpus: {clean_file}")

    # Save Syllable-segmented NFC corpus
    with open(syl_file, "w", encoding="utf-8") as f:
        f.write("\n".join(syl_sentences) + "\n")
    print(f"  Saved syllable-pretokenized corpus: {syl_file}")

def train_sentencepiece(input_file: str, model_prefix: str, model_type: str = "bpe", vocab_size: int = 4000):
    """Train SentencePiece model (BPE or Unigram)."""
    print(f"  Training SentencePiece {model_type.upper()} ({model_prefix})...")
    t0 = time.time()
    spm.SentencePieceTrainer.train(
        input=input_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=0.9995,
        pad_id=0, unk_id=1, bos_id=2, eos_id=3,
        hard_vocab_limit=False,
        pad_piece="<pad>", unk_piece="<unk>", bos_piece="<s>", eos_piece="</s>"
    )
    dur = time.time() - t0
    print(f"    Completed in {dur:.2f}s -> {model_prefix}.model ({vocab_size} vocab)")

def train_wordpiece(input_file: str, output_file: str, vocab_size: int = 4000):
    """Train Hugging Face WordPiece model (BERT-style)."""
    print(f"  Training Hugging Face WordPiece ({output_file})...")
    t0 = time.time()
    tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    trainer = trainers.WordPieceTrainer(
        vocab_size=vocab_size,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
        min_frequency=2,
        continuing_subword_prefix="##"
    )
    tokenizer.train([input_file], trainer)
    tokenizer.save(output_file)
    dur = time.time() - t0
    print(f"    Completed in {dur:.2f}s -> {output_file} ({vocab_size} vocab)")

def evaluate_models(models_dir: str):
    """Run verification benchmark across all 6 models on canonical Burmese test sentences."""
    print(f"\n[3/3] Evaluating Models in: {models_dir}")
    test_sentences = [
        ("Entity & Aukmyit", "သမ္မတဦးဝင်းမြင့်သည် ၂၀၁၈ ခုနှစ် မတ်လ ၃ ရက်နေ့တွင် ရန်ကုန်မြို့သို့ သွားရောက်ခဲ့သည်။"),
        ("Stacked Virama", "တက္ကသိုလ်ကျောင်းသားများသည် စာမေးပွဲ ဖြေဆိုကြသည်။"),
        ("Negative Predicate", "သူ ထမင်း မစားခဲ့ပါဘူး။"),
    ]

    sp_bpe = spm.SentencePieceProcessor(model_file=os.path.join(models_dir, "sp_bpe.model"))
    sp_uni = spm.SentencePieceProcessor(model_file=os.path.join(models_dir, "sp_unigram.model"))
    wp = Tokenizer.from_file(os.path.join(models_dir, "wordpiece.json"))

    sp_bpe_syl = spm.SentencePieceProcessor(model_file=os.path.join(models_dir, "sp_bpe_syl.model"))
    sp_uni_syl = spm.SentencePieceProcessor(model_file=os.path.join(models_dir, "sp_unigram_syl.model"))
    wp_syl = Tokenizer.from_file(os.path.join(models_dir, "wordpiece_syl.json"))

    def check_integrity(text: str, tokens: List[str], marker: str = "sp") -> float:
        syls = segment_syllables(text)
        syl_spans = []
        curr = 0
        for s in syls:
            idx = text.find(s, curr)
            if idx != -1:
                syl_spans.append((idx, idx + len(s)))
                curr = idx + len(s)
            else:
                syl_spans.append((-1, -1))

        tok_bounds = set()
        c_pos = 0
        for t in tokens:
            clean = t.replace(" ", "").replace("\u2581", "") if marker == "sp" else t.replace("##", "")
            if not clean:
                continue
            idx = text.find(clean, c_pos)
            if idx != -1:
                tok_bounds.add(idx)
                tok_bounds.add(idx + len(clean))
                c_pos = idx + len(clean)

        matched = sum(1 for st, en in syl_spans if st != -1 and not any(st < b < en for b in tok_bounds))
        return (matched / len(syls) * 100.0) if syls else 100.0

    for name, s in test_sentences:
        s_norm = normalize_burmese_text(s)
        s_syl_eval = " ".join(segment_syllables(s_norm))

        # Standard models
        bpe_std = sp_bpe.encode(s_norm, out_type=str)
        uni_std = sp_uni.encode(s_norm, out_type=str)
        wp_std = wp.encode(s_norm).tokens

        # Syllable-aware models
        bpe_syl = sp_bpe_syl.encode(s_syl_eval, out_type=str)
        uni_syl = sp_uni_syl.encode(s_syl_eval, out_type=str)
        wp_syl_toks = wp_syl.encode(s_syl_eval).tokens

        print(f"\n--- Benchmark: {name} ---")
        print(f"Text: {s_norm}")
        print(f"  Standard SP BPE:      {check_integrity(s_norm, bpe_std, 'sp'):5.1f}% intact ({len(bpe_std)} tokens)")
        print(f"  Standard SP Unigram:  {check_integrity(s_norm, uni_std, 'sp'):5.1f}% intact ({len(uni_std)} tokens)")
        print(f"  Standard WordPiece:   {check_integrity(s_norm, wp_std, 'wp'):5.1f}% intact ({len(wp_std)} tokens)")
        print(f"  [+] Syllable SP BPE:      {check_integrity(s_norm, bpe_syl, 'sp'):5.1f}% intact ({len(bpe_syl)} tokens)")
        print(f"  [+] Syllable SP Unigram:  {check_integrity(s_norm, uni_syl, 'sp'):5.1f}% intact ({len(uni_syl)} tokens)")
        print(f"  [+] Syllable WordPiece:   {check_integrity(s_norm, wp_syl_toks, 'wp'):5.1f}% intact ({len(wp_syl_toks)} tokens)")

def main():
    parser = argparse.ArgumentParser(description="Train Production-Grade Myanmar Subword Tokenizers")
    parser.add_argument("--raw_data", type=str, default="data/myNews_train.raw.txt", help="Path to raw myNews dataset")
    parser.add_argument("--vocab_size", type=int, default=4000, help="Vocabulary size (default: 4000)")
    parser.add_argument("--models_dir", type=str, default="models", help="Output directory for trained models")
    args = parser.parse_args()

    clean_file = "data/corpus_cleaned.txt"
    syl_file = "data/corpus_syllables.txt"

    # Step 1: Preprocessing & Normalization
    prepare_corpora(args.raw_data, clean_file, syl_file)

    # Step 2: Training
    print(f"\n[2/3] Training 6 Subword Models (Vocab: {args.vocab_size:,})...")
    os.makedirs(args.models_dir, exist_ok=True)

    # Track A: Standard Models (trained on continuous clean text)
    print("\n--- Track A: Standard Subword Models ---")
    train_sentencepiece(clean_file, os.path.join(args.models_dir, "sp_bpe"), "bpe", args.vocab_size)
    train_sentencepiece(clean_file, os.path.join(args.models_dir, "sp_unigram"), "unigram", args.vocab_size)
    train_wordpiece(clean_file, os.path.join(args.models_dir, "wordpiece.json"), args.vocab_size)

    # Track B: Syllable-Aware Models (trained on sylbreak++ pre-tokenized text)
    print("\n--- Track B: Syllable-Aware Subword Models (sylbreak++) ---")
    train_sentencepiece(syl_file, os.path.join(args.models_dir, "sp_bpe_syl"), "bpe", args.vocab_size)
    train_sentencepiece(syl_file, os.path.join(args.models_dir, "sp_unigram_syl"), "unigram", args.vocab_size)
    train_wordpiece(syl_file, os.path.join(args.models_dir, "wordpiece_syl.json"), args.vocab_size)

    # Step 3: Evaluation
    evaluate_models(args.models_dir)
    print("\nTraining and evaluation successfully completed!")

if __name__ == "__main__":
    main()
