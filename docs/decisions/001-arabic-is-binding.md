# 001: The Arabic text is authoritative

**Status:** accepted (2026-11-19)

## Context

The corpus has every document in Arabic and English. What the source files say about which version prevails:

| Document | Statement in the PDFs |
|---|---|
| NCA ECC-2:2024, CCC-2:2024, DCC-1:2022 | English PDFs: *"the Arabic version will be the binding language for all matters relating to the meaning or interpretation of this document."* |
| PDPL, its Implementing Regulation, the Transfer Regulation (SDAIA) | No statement in either language. Saudi laws are issued and gazetted in Arabic, and the English texts are translations, but these files do not say so. Verify against SDAIA's site before stating it as fact. |

A check of all 335 English/Arabic chunk pairs (structure and length screening, then reading every flagged pair) found one real content difference. NCA's Arabic and English ECC-2:2024 files number subdomains 1-7 and 2-7 differently:

| Content | English | Arabic |
|---|---|---|
| Comply with national cybersecurity legislation | (listed as deleted in the appendix) | 1-7-1 |
| Comply with international agreements | 1-7-1 | 1-7-2 |
| Cover at least the DCC requirements | (absent) | 2-7-3 |
| Periodically review data-protection requirements | 2-7-3 | 2-7-4 |

The screening catches missing or extra controls and large content gaps. It does not catch subtle wording differences ("shall" vs "may"). A model-based pairwise meaning check is planned as part of the evaluation.

## Decision

1. **Arabic numbering is canonical.** Every chunk has `canonical_key`, which is the Arabic key. English keys that differ are mapped through `aliases` in `data/sources.yaml`. Pairing, citation scoring and deduplication use `canonical_key`, never the raw `ref_key`.
2. **Each chunk records its standing** in `authority`:
   - `binding`: NCA Arabic.
   - `official`: SDAIA Arabic. The source files don't state which version prevails.
   - `translation`: all English texts.
3. **Citations always show the Arabic text.** Retrieval searches both languages, whatever the question's language. Each cited article or control shows the Arabic text, plus the English translation when the question is in English. The answer is written in the question's language.
4. **If the versions differ, the Arabic wins, and the answer says so.** For example: "the English translation numbers this 1-7-1; the binding Arabic text numbers it 1-7-2". The text comes from each chunk's `note`.
5. **OCR'd Arabic chunks** (`source: ocr`) can contain misread digits. For exact numbers and cross-references inside them, rely on the paired English text, and quote the Arabic with that caveat.
6. **The UI and API show a disclaimer:** English texts are translations, and the Arabic version prevails (stated by NCA for its controls).
7. **The evaluation includes questions on ECC 1-7 and 2-7**, so citation accuracy is tested where the versions disagree (`eval/questions.jsonl`, tag `version-difference`).

## Consequences

- An English-only answer can't silently cite a control number that means something else in the binding text.
- Citations are longer, because they include both languages.
- New documents need `authority` (and `aliases` if their numbering differs) in `data/sources.yaml`. `tests/test_processed_chunks.py` fails if the canonical keys stop lining up.
