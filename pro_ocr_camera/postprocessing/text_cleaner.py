import re

class TextCleaner:
    def __init__(self, regex_pattern=None, corrections=None):
        self.regex_pattern = regex_pattern
        self.corrections = corrections or {}

    def clean(self, text, mode="LPR"):
        """
        Cleans OCR output.
        """
        if not text:
            return ""
            
        # Basic cleaning: remove special chars and strip
        text = text.upper().strip()
        text = re.sub(r'[^A-Z0-9-]', '', text)
        
        # Apply character corrections (common misreads)
        if mode == "LPR":
            # For plates, we can be more aggressive
            corrected = ""
            for char in text:
                corrected += self.corrections.get(char, char)
            text = corrected

        # Regex validation
        if self.regex_pattern:
            if not re.match(self.regex_pattern, text):
                # Optionally return empty or handle invalid format
                pass
                
        return text

    @staticmethod
    def is_valid_plate(text, pattern):
        if not text:
            return False
        return bool(re.match(pattern, text))
