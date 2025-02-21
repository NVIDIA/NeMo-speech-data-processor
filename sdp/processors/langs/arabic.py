import re
import unicodedata

from sdp.processors.base_processor import BaseParallelProcessor, DataEntry

# Arabic letters
HAMZA = "\u0621"
ALEF_MADDA = "\u0622"
ALEF_HAMZA_ABOVE = "\u0623"
WAW_HAMZA = "\u0624"
ALEF_HAMZA_BELOW = "\u0625"
YEH_HAMZA_ABOVE = "\u0626"
ALEF = "\u0627"
BEH = "\u0628"
TEH_MARBUTA = "\u0629"
TEH = "\u062A"
THEH = "\u062B"
JEEM = "\u062C"
HAH = "\u062D"
KHAH = "\u062E"
DAL = "\u062F"
THAL = "\u0630"
REH = "\u0631"
ZAIN = "\u0632"
SEEN = "\u0633"
SHEEN = "\u0634"
SAD = "\u0635"
DAD = "\u0636"
TAH = "\u0637"
ZAH = "\u0638"
AIN = "\u0639"
GHAIN = "\u063A"
FEH = "\u0641"
QAF = "\u0642"
KAF = "\u0643"
LAM = "\u0644"
MEEM = "\u0645"
NOON = "\u0646"
HEH = "\u0647"
WAW = "\u0648"
ALEF_MAKSURA = "\u0649"
YEH = "\u064A"

# Harakats (diacritics)
FATHAT = "\u064E"
KASRAH = "\u0650"
DAMMAH = "\u064F"
SUKUN = "\u0652"
SHADDAH = "\u0651"
KASRATAN = "\u064D"
DAMMATAN = "\u064C"
FATHATAN = "\u064B"

# Ligatures
LAM_ALEF = u'\uFEFB'
LAM_ALEF_HAMZA_ABOVE = u'\uFEF7'
LAM_ALEF_HAMZA_BELOW = u'\uFEF9'
LAM_ALEF_MADDA_ABOVE = u'\uFEF5'
LIGATURES=(LAM_ALEF, LAM_ALEF_HAMZA_ABOVE, LAM_ALEF_HAMZA_BELOW, LAM_ALEF_MADDA_ABOVE)

# Punctuation marks
QUESTION_MARK = "\u061F"
SAMICOLON = "\u061B"
COMMA = "\u060C"

DIACRITICS = [chr(x) for x in range(0x0600, 0x06ff) if unicodedata.category(chr(x)) == "Mn"]
PUNCTUATION_MARKS = ["?", "!", ":", ";", "-", ".", ",", "؟","،", "؛"]
ALEFS = (ALEF, ALEF_MADDA, ALEF_HAMZA_ABOVE, ALEF_HAMZA_BELOW)

class ArabicTextPreprocessor(BaseParallelProcessor):
    """Class for Arabic text preprocessing.

    Operates on the text in the ``input_text_key``, and saves output text in
    the ``output_text_key``.
    
    Args:
        input_text_key (str):       the text field that will be the input to the processor.
        output_text_key (str):      the text field that will contain processed text.
        remove_extra_spaces (bool): replaces consequent spaces by one. Defaults to True.
        remove_empty_lines (bool):  joins multiline input into single-line text. Defaults to True.
        remove_diacritics (bool):   removes Arabic diacritical marks from the input text. Defaults to False.
        remove_punctuation (bool):  removes punctuation marks from the input text. Defaults to False.
            Processed punctuation marks are: Question mark, Exclamation mark, Colon,Semicolon,
            Hypen-Minus, Full stop, Comma, Arabic Question Mark, Arabic Comma, Arabic Semicolon.
        remove_tatweel (bool):      removes tatweel justification sign from the text. Defaults to False.
        apply_nfkc (bool):   applies compatability decomposition followed by canonical composition.
            Useful for replacing Arabic letters positional forms with general unicode and ensuring consistent diacritical marks ordering.
            Find more here https://docs.python.org/3/library/unicodedata.html#unicodedata.normalize.
            Defaults to False.
        normalize (bool): normalizes the input text. Normalization includes:    removing diacritical marks,
            normalization of letter `ALEF`-- `ALEF_HAMZA_BELOW`, `ALEF_HAMZA_ABOVE`, `ALEF_MADDA_ABOVE` will be replaced by `ALEF`,
            normalization of ligatures: `LAM_ALEF`, `LAM_ALEF_HAMZA_ABOVE`, `LAM_ALEF_HAMZA_BELOW`, `LAM_ALEF_MADDA_ABOVE` ligatures will be replaces by two letters `LAM` and `ALEF`.
            letter `TEH_MARBUTA` will be replaced by `HEH`. Defaults to False.
    """
    def __init__(
        self,
        input_text_key: str = "text",
        output_text_key: str = "text",
        remove_extra_spaces: bool = True,
        remove_empty_lines: bool = True,
        remove_diacritics: bool = False,
        remove_punctuation: bool = False,
        remove_tatweel: bool = False,
        normalize_ligature: bool = False,
        apply_nfkc: bool = False,
        normalize: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_text_key = input_text_key
        self.output_text_key = output_text_key
        self.output_text_key = output_text_key
        self.remove_extra_spaces = remove_extra_spaces
        self.remove_empty_lines = remove_empty_lines
        self.remove_diacritics = remove_diacritics
        self.remove_punctuation = remove_punctuation
        self.remove_tatweel = remove_tatweel
        self.normalize_ligature = normalize_ligature
        self.normalize = normalize
        self.apply_nfkc = apply_nfkc

    def process_dataset_entry(self, data_entry):
        data_entry[self.output_text_key] = self.clean_data(
            data_entry[self.input_text_key]
        )
        return [DataEntry(data=data_entry)]

    def _remove_diacritics(self, text):
        for char in DIACRITICS:
            text = text.replace(char, '')
        return text

    def _remove_punctuation(self, text):
        for char in PUNCTUATION_MARKS:
            text = text.replace(char, '')
        return text

    def _normalize_teh(self, text):
        text = text.replace(TEH_MARBUTA, HEH)
        return text
    
    def _normalize_ligature(self, text):
        LIGUATURES_PATTERN = re.compile(u"[" + u"".join(LIGATURES) + u"]", re.UNICODE)
        return LIGUATURES_PATTERN.sub(u'%s%s' % (LAM, ALEF), text)
    
    def _normalize_alef(self, text):
        ALEFS_PATTERN = re.compile(u"[" + u"".join(ALEFS) + u"]", re.UNICODE)
        return re.sub(ALEFS_PATTERN, ALEF, text)

    def _remove_extra_spaces(self, text):
        text = re.sub(" +", " ", text)
        return text

    def _remove_empty_lines(self, text):
        lines = text.split("\n")
        return ("\n").join([line for line in lines if len(line) >= 1])

    def _normalize(self, text):
        text = self._remove_diacritics(text)
        text = self._normalize_alef(text)
        text = self._normalize_ligature(text)
        text = self._normalize_teh(text)

        return text

    def clean_data(self, text):
        if self.remove_extra_spaces:
            text = self._remove_extra_spaces(text)
        if self.remove_empty_lines:
            text = self._remove_empty_lines(text)
        if self.remove_diacritics:
            text = self._remove_diacritics(text)
        if self.remove_tatweel:
            text = text.replace("ـ", "")
        if self.remove_punctuation:
            text = self._remove_punctuation(text)
        if self.normalize_ligature:
            text = self._normalize_ligature(text)
        if self.normalize:
            text = self._normalize(text)
        if self.apply_nfkc:
            text = unicodedata.normalize("NFKC", text)
        return text