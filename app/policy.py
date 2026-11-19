"""Answering policy shared by the API and the evaluation (see docs/decisions/001-arabic-is-binding.md)."""

DISCLAIMER = {
    "en": (
        "English texts are translations. The Arabic text prevails; NCA states this for its "
        "controls (ECC, CCC, DCC). This assistant is not legal advice."
    ),
    "ar": (
        "النصوص الإنجليزية ترجمة، والنص العربي هو المعتمد (وتنص الهيئة الوطنية للأمن السيبراني على ذلك "
        "لضوابطها). هذه الإجابات ليست استشارة قانونية."
    ),
}

# Authority ranking used when the two language versions of a citation disagree.
AUTHORITY_RANK = {"binding": 0, "official": 1, "translation": 2}
