"""
Define the things we'll search for when tokenizing
"""

normal_words = [
    'a', 'akesi', 'ala', 'alasa', 'ale', 'anpa',
    'ante', 'anu', 'awen',
    'e', 'en', 'esun',
    'ijo', 'ike', 'ilo', 'insa',
    'jaki', 'jan', 'jelo', 'jo',
    'kala', 'kalama', 'kama', 'kasi', 'ken', 'kepeken',
    'kili', 'kiwen', 'ko', 'kon', 'kule', 'kulupu', 'kute',
    'la', 'lape', 'laso', 'lawa', 'len', 'lete', 'li', 'lili', 'linja',
    'lipu', 'loje', 'lon', 'luka', 'lukin', 'lupa',
    'ma', 'mama', 'mani', 'meli', 'mi', 'mije', 'moku', 'moli',
    'monsi', 'mu', 'mun', 'musi', 'mute',
    'nanpa', 'nasa', 'nasin', 'nena', 'ni', 'nimi', 'noka',
    'o', 'olin', 'ona', 'open',
    'pakala', 'pali', 'palisa', 'pan', 'pana', 'pi', 'pilin', 'pimeja', 'pini',
    'pipi', 'poka', 'poki', 'pona', 'pu',
    'sama', 'seli', 'selo', 'seme', 'sewi', 'sijelo', 'sike', 'sin', 'sina',
    'sinpin', 'sitelen', 'sona', 'soweli', 'suli', 'suno', 'supa', 'suwi',
    'tan', 'taso', 'tawa', 'telo', 'tenpo', 'toki', 'tomo', 'tu',
    'unpa', 'uta', 'utala',
    'walo', 'wan', 'waso', 'wawa', 'weka', 'wile',
    # less normal. Remove?
    'apeja', 'kijetesantakalu', 'kipisi', 'leko',
    'majuna', 'monsuta', 'po', 'powe', 'tuli'
]

punctuation = [
    '.', '?', '!', ',',  # basic needs
    '"', ':', ';', '*', '#', '\n', '(', ')',  # markdown and more complex
]


# special rules for tokens -> sentences
no_space_before = [
    '.', '?', '!', ',',
    ':', ';', '\n', ')'
]

no_space_after = [
    '(', '\n'
]


# special rules for sentences -> tokens
# remove left, use right instead
equivalence = {
    'ali': 'ale',
    'kin': 'a',
    'oko': 'lukin',
    'namako': 'sin'
}
_punct_equiv = {
    ('\'', '`', '“', '”'): '"',
    ('-', '~', '=', '_', '\\', '|', '+'): '*',
    ('[', '{', '<'): '(',
    (']', '}', '>'): ')',
}

# turn the above into a 1-1 lookup
for k, v in _punct_equiv.items():
    for key in k:
        equivalence[key] = v
