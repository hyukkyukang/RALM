import re

STOPWORDS_FROM_GPT2 = set(
    {
        "ourselves",
        "hers",
        "between",
        "yourself",
        "but",
        "again",
        "there",
        "about",
        "once",
        "during",
        "out",
        "very",
        "having",
        "with",
        "they",
        "own",
        "an",
        "be",
        "some",
        "for",
        "do",
        "its",
        "yours",
        "such",
        "into",
        "of",
        "most",
        "itself",
        "other",
        "off",
        "is",
        "s",
        "am",
        "or",
        "who",
        "as",
        "from",
        "him",
        "each",
        "the",
        "themselves",
        "until",
        "below",
        "are",
        "we",
        "these",
        "your",
        "his",
        "through",
        "don",
        "nor",
        "me",
        "were",
        "her",
        "more",
        "himself",
        "this",
        "down",
        "should",
        "our",
        "their",
        "while",
        "above",
        "both",
        "up",
        "to",
        "ours",
        "had",
        "she",
        "all",
        "no",
        "when",
        "at",
        "any",
        "before",
        "them",
        "same",
        "and",
        "been",
        "have",
        "in",
        "will",
        "on",
        "does",
        "yourselves",
        "then",
        "that",
        "because",
        "what",
        "over",
        "why",
        "so",
        "can",
        "did",
        "not",
        "now",
        "under",
        "he",
        "you",
        "herself",
        "has",
        "just",
        "where",
        "too",
        "only",
        "myself",
        "which",
        "those",
        "i",
        "after",
        "few",
        "whom",
        "t",
        "being",
        "if",
        "theirs",
        "my",
        "against",
        "a",
        "by",
        "doing",
        "it",
        "how",
        "further",
        "was",
        "here",
        "than",
    }
)


def normalize_quotes(text: str) -> str:
    """
    Normalize various types of single and double quotes to standard quotes (' and ").
    Handles curly quotes, prime marks, and their combinations.
    Uses regex for efficient pattern matching.
    """
    # Map of quote patterns to be replaced with standard quotes
    quote_patterns = [
        (r'[""‟„″]', '"'),  # various double quotes to standard double quote
        (r"``", '"'),  # double backticks to double quote
        (r"[" "‛′]", "'"),  # various single quotes to standard single quote
        (r"`", "'"),  # single backtick to single quote
    ]

    result = text
    for pattern, replacement in quote_patterns:
        result = re.sub(pattern, replacement, result)

    return result


if __name__ == "__main__":
    # Test cases with different types of quotes and combinations
    test_strings = [
        # Basic cases
        "``Hello, world!``",  # Double backticks
        '"Hello, world!"',  # Standard double quotes
        '"Hello, world!"',  # Curly double quotes (left and right)
        '‟Hello, world!"',  # Bottom double quote with standard
        '„Hello, world!"',  # Double low-9 quote with standard
        "″Hello, world!″",  # Double prime quotes
        # Single quote cases
        "'Hello, world!'",  # Standard single quotes
        "'Hello, world!'",  # Curly single quotes
        "‛Hello, world!'",  # Single bottom quote with standard
        "′Hello, world!′",  # Prime marks
        "`Hello, world!`",  # Backticks
        # Mixed single and double quote cases
        "\"Hello, 'world' indeed!\"",  # Nested single in double
        "'Hello, \"world\" indeed!'",  # Nested double in single
        '′Quote′ with "mixed" styles',  # Mixed prime and curly quotes
        "`Single` with „double‟ mix",  # Mixed backtick and double quotes
        # Mixed and complex cases
        '``Hello`` "world" "testing" ‟mixed″',  # Mixed quote types
        "No quotes here",  # No quotes to test
        '""',  # Empty quotes
        '"Hello" in "the" "middle"',  # Multiple quote pairs
        # Edge cases
        "`` ``",  # Multiple backticks with spaces
        '""""',  # Multiple consecutive quotes
        '"Hello"world"',  # No space between quoted sections
        '``"Hello"``',  # Nested quotes
        # Real-world examples
        'He said ``I don"t know`` and left',  # Contractions with quotes
        '"It"s a "beautiful" day"',  # Multiple quotes with apostrophes
        '„Quote" in "different" ″styles″',  # Mixed international quotes
        # Long text examples
        '``This is a longer piece of text that "contains" multiple types of "quotes" within it``',
        '"First quote" then „second quote" and finally ″third quote″',
        "'Single quote' then \"double quote\" and `backtick` mix",  # Mixed quote styles
        # Special cases
        ' "Space before"',  # Leading space
        '"Space after" ',  # Trailing space
        '\n"New line"\n',  # With newlines
    ]

    # Test each string and print results
    for test_str in test_strings:
        result = normalize_quotes(test_str)
        print(f"Original: {test_str}")
        print(f"Normalized: {result}")
        print("-" * 50)
