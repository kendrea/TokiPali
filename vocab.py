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
    'kipisi',
    'leko',
    'monsuta',
    # 'kijetesantakalu',
    # # uncommon
    # 'majuna', 'powe', 'apeja', 'po', 'tuli'
]

punctuation = [
    '.', '?', '!', ','
]

# special rules for sentences -> tokens
# remove left, use right instead
equivalence = {
    'ali': 'ale',
    'kin': 'a',
    'oko': 'lukin',
    'namako': 'sin',
    # debatable
    'kijetesantakalu': 'soweli',  # raccoon->mammal
    # uncommon
    'majuna': 'suli',  # old->adult
    'powe': 'ike',  # false->bad
    'apeja': 'alasa',  # embarrass->hunt
    'po': 'luka',  # 4->5
    'tuli': 'luka',  # 3->5

    # punctuation
    ':': '.',
    ';': '.',
    '-': '.',
    '~': '.',
}
