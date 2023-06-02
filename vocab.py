"""
Define the things we'll search for when tokenizing
"""

normal_words = [
    'a', 'akesi', 'ala', 'alasa', 'ale', 'ali', 'anpa',
    'ante', 'anu', 'apeja', 'awen',
    'e', 'en', 'esun',
    'ijo', 'ike', 'ilo', 'insa',
    'jaki', 'jan', 'jelo', 'jo',
    'kala', 'kalama', 'kama', 'kasi', 'ken', 'kepeken', 'kijetesantakalu',
    'kili', 'kin', 'kipisi', 'kiwen', 'ko', 'kon', 'kule', 'kulupu', 'kute',
    'la', 'lape', 'laso', 'lawa', 'leko', 'len', 'lete', 'li', 'lili', 'linja',
    'lipu', 'loje', 'lon', 'luka', 'lukin', 'lupa',
    'ma', 'majuna', 'mama', 'mani', 'meli', 'mi', 'mije', 'moku', 'moli',
    'monsi', 'monsuta', 'mu', 'mun', 'musi', 'mute',
    'namako', 'nanpa', 'nasa', 'nasin', 'nena', 'ni', 'nimi', 'noka',
    'o', 'oko', 'olin', 'ona', 'open',
    'pakala', 'pali', 'palisa', 'pan', 'pana', 'pi', 'pilin', 'pimeja', 'pini',
    'pipi', 'po', 'poka', 'poki', 'pona', 'powe', 'pu',
    'sama', 'seli', 'selo', 'seme', 'sewi', 'sijelo', 'sike', 'sin', 'sina',
    'sinpin', 'sitelen', 'sona', 'soweli', 'suli', 'suno', 'supa', 'suwi',
    'tan', 'taso', 'tawa', 'telo', 'tenpo', 'toki', 'tomo', 'tu', 'tuli',
    'unpa', 'uta', 'utala',
    'walo', 'wan', 'waso', 'wawa', 'weka', 'wile'
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
_equiv = {
    ('\'', '`', '“', '”'): '"',
    ('-', '~', '=', '_', '\\', '|', '+'): '*',
    ('[', '{', '<'): '(',
    (']', '}', '>'): ')',
}

# turn the above into a 1-1 lookup
equivalence = {}
for k, v in _equiv.items():
    for key in k:
        equivalence[key] = v
