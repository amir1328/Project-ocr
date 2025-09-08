from typing import List
import re
from symspellpy import SymSpell, Verbosity

try:
    import arabic_reshaper
    from bidi.algorithm import get_display
except Exception:  # optional
    arabic_reshaper = None
    get_display = None


class Postprocessor:
    def __init__(
        self,
        lexicon_paths: List[str],
        max_edit_distance_dictionary: int = 2,
        normalize_digits: bool = True,
        normalize_diacritics: bool = True,
    ) -> None:
        self.normalize_digits = normalize_digits
        self.normalize_diacritics = normalize_diacritics
        self.sym = SymSpell(max_dictionary_edit_distance=max_edit_distance_dictionary)
        for path in lexicon_paths:
            try:
                self.sym.create_dictionary(path)
            except Exception:
                pass

    def _normalize_digits(self, text: str) -> str:
        if not self.normalize_digits:
            return text
        digits_map = str.maketrans({
            "٠": "0", "١": "1", "٢": "2", "٣": "3", "٤": "4",
            "٥": "5", "٦": "6", "٧": "7", "٨": "8", "٩": "9",
        })
        return text.translate(digits_map)

    def _normalize_rtl(self, text: str) -> str:
        if arabic_reshaper and get_display:
            try:
                return get_display(arabic_reshaper.reshape(text))
            except Exception:
                return text
        return text

    def _spell_correct(self, text: str) -> str:
        tokens = re.findall(r"\w+|\W+", text, flags=re.UNICODE)
        corrected = []
        for tok in tokens:
            if tok.isalpha() and len(tok) > 2 and self.sym.words:
                suggestions = self.sym.lookup(tok, Verbosity.TOP, max_edit_distance=2)
                corrected.append(suggestions[0].term if suggestions else tok)
            else:
                corrected.append(tok)
        return "".join(corrected)

    def clean_text(self, text: str) -> str:
        text = self._normalize_digits(text)
        text = self._spell_correct(text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = text.strip()
        text = self._normalize_rtl(text)
        return text
