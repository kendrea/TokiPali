"""
Define the things we'll search for when tokenizing
"""
from enum import Enum
from dataclasses import dataclass
from vocab import normal_words

# # turn the above into a 1-1 lookup
# equivalence = {}
# for k, v in _equiv.items():
#  for key in k:
#      equivalence[key] = v


class Part(Enum):
    """
    Parts of speech
    """
    NOUN = 0
    VERB = 1
    NUMBER = 2
    PRE_VERB = 3
    PARTICLE = 4
    ADJECTIVE = 5
    PREPOSITION = 6


@dataclass
class TranslationEntry:
    """
    One "line" of a dictionary lookup
    """
    part: Part
    definition: str

    def __str__(self):
        names = ["n", "v", "num", "pre-verb", "particle", "adj", "prep"]
        part_s = names[self.part.value]
        return f"{part_s}. {self.definition}"


tokipona_to_en = {}  # filled below


# https://jan-ne.github.io/tp/dictionary
# Changes after paste: prefer ale, a, lukin, sin. Remove line wrapping
_DICT_STR = """
a
     PARTICLE: (emphasis, emotion or confirmation)
akesi
     NOUN: non-cute animal; reptile, amphibian
ala
     ADJECTIVE: no, not, zero
alasa
     VERB: to hunt, forage
ale
     ADJECTIVE: all; abundant, countless, bountiful, every, plentiful
     NOUN: abundance, everything, life, universe
     NUMBER: 100
anpa
     ADJECTIVE: bowing down, downward, humble, lowly, dependent
ante
     ADJECTIVE: different, altered, changed, other
anu
     PARTICLE: or
awen
     ADJECTIVE: enduring, kept, protected, safe, waiting, staying
     PRE-VERB: continuer à
e
     PARTICLE: (before the direct object)
en
     PARTICLE: (between multiple subjects)
esun
     NOUN: market, shop, fair, bazaar, business transaction
ijo
     NOUN: thing, phenomenon, object, matter
ike
     ADJECTIVE: bad, negative; non-essential, irrelevant
ilo
     NOUN: tool, implement, machine, device
insa
     NOUN: centre, content, inside, between; internal organ, stomach
jaki
     ADJECTIVE: disgusting, obscene, sickly, toxic, unclean, unsanitary
jan
     NOUN: human being, person, somebody
jelo
     ADJECTIVE: yellow, yellowish
jo
     VERB: to have, carry, contain, hold
kala
     NOUN: fish, marine animal, sea creature
kalama
     VERB: to produce a sound; recite, utter aloud
kama
     ADJECTIVE: arriving, coming, future, summoned
     PRE-VERB: to become, manage to, succeed in
kasi
     NOUN: plant, vegetation; herb, leaf
ken
     PRE-VERB: to be able to, be allowed to, can, may
     ADJECTIVE: possible
kepeken
     PREPOSITION: to use, with, by means of
kili
     NOUN: fruit, vegetable, mushroom
kiwen
     NOUN: hard object, metal, rock, stone
ko
     NOUN: clay, clinging form, dough, semi-solid, paste, powder
kon
     NOUN: air, breath; essence, spirit; hidden reality, unseen agent
kule
     ADJECTIVE: colourful, pigmented, painted
kulupu
     NOUN: community, company, group, nation, society, tribe
kute
     NOUN: ear
     VERB: to hear, listen; pay attention to, obey
la
     PARTICLE: (between the context phrase and the main sentence)
lape
     ADJECTIVE: sleeping, resting
laso
     ADJECTIVE: blue, green
lawa
     NOUN: head, mind
     VERB: to control, direct, guide, lead, own, plan, regulate, rule
len
     NOUN: cloth, clothing, fabric, textile; cover, layer of privacy
lete
     ADJECTIVE: cold, cool; uncooked, raw
li
     PARTICLE: (between any subject except mi alone or sina alone and its verb; also to introduce a new verb for the same subject)
lili
     ADJECTIVE: little, small, short; few; a bit; young
linja
     NOUN: long and flexible thing; cord, hair, rope, thread, yarn
lipu
     NOUN: flat object; book, document, card, paper, record, website
loje
     ADJECTIVE: red, reddish
lon
     PREPOSITION: located at, present at, real, true, existing
luka
     NOUN: arm, hand, tactile organ
     NUMBER: five
lukin
     NOUN: eye
     VERB: to look at, see, examine, observe, read, watch
     PRE-VERB: to seek, look for, try to
lupa
     NOUN: door, hole, orifice, window
ma
     NOUN: earth, land; outdoors, world; country, territory; soil
mama
     NOUN: parent, ancestor; creator, originator; caretaker, sustainer
mani
     NOUN: money, cash, savings, wealth; large domesticated animal
meli
     NOUN: woman, female, feminine person; wife
mi
     NOUN: I, me, we, us
mije
     NOUN: man, male, masculine person; husband
moku
     VERB: to eat, drink, consume, swallow, ingest
moli
     ADJECTIVE: dead, dying
monsi
     NOUN: back, behind, rear
mu
     PARTICLE: (animal noise or communication)
mun
     NOUN: moon, night sky object, star
musi
     ADJECTIVE: artistic, entertaining, frivolous, playful, recreational
mute
     ADJECTIVE: many, a lot, more, much, several, very
     NOUN: quantity
nanpa
     PARTICLE: -th (ordinal number)
     NOUN: numbers
nasa
     ADJECTIVE: unusual, strange; foolish, crazy; drunk, intoxicated
nasin
     NOUN: way, custom, doctrine, method, path, road
nena
     NOUN: bump, button, hill, mountain, nose, protuberance
ni
     ADJECTIVE: that, this
nimi
     NOUN: name, word
noka
     NOUN: foot, leg, organ of locomotion; bottom, lower part
o
     PARTICLE: hey! O! (vocative or imperative)
olin
     VERB: to love, have compassion for, respect, show affection to
ona
     NOUN: he, she, it, they
open
     VERB: to begin, start; open; turn on
pakala
     ADJECTIVE: botched, broken, damaged, harmed, messed up
pali
     VERB: to do, take action on, work on; build, make, prepare
palisa
     NOUN: long hard thing; branch, rod, stick
pan
     NOUN: cereal, grain; barley, corn, oat, rice, wheat; bread, pasta
pana
     VERB: to give, send, emit, provide, put, release
pi
     PARTICLE: of
pilin
     NOUN: heart (physical or emotional)
     ADJECTIVE: feeling (an emotion, a direct experience)
pimeja
     ADJECTIVE: black, dark, unlit
pini
     ADJECTIVE: ago, completed, ended, finished, past
pipi
     NOUN: bug, insect, ant, spider
poka
     NOUN: hip, side; next to, nearby, vicinity
poki
     NOUN: container, bag, bowl, box, cup, cupboard, drawer, vessel
pona
     ADJECTIVE: good, positive, useful; friendly, peaceful; simple
pu
     ADJECTIVE: interacting with the official Toki Pona book
sama
     ADJECTIVE: same, similar; each other; sibling, peer, fellow
     PREPOSITION: as, like
seli
     ADJECTIVE: fire; cooking element, chemical reaction, heat source
selo
     NOUN: outer form, outer layer; bark, peel, shell, skin; boundary
seme
     PARTICLE: what? which?
sewi
     NOUN: area above, highest part, something elevated
     ADJECTIVE: awe-inspiring, divine, sacred, supernatural
sijelo
     NOUN: body (of person or animal), physical state, torso
sike
     NOUN: round or circular thing; ball, circle, cycle, sphere, wheel
     ADJECTIVE: of one year
sin
     ADJECTIVE: new, fresh; additional, another, extra
sina
     NOUN: you
sinpin
     NOUN: face, foremost, front, wall
sitelen
     NOUN: image, picture, representation, symbol, mark, writing
sona
     VERB: to know, be skilled in, be wise about, have information on
     PRE-VERB: to know how to
soweli
     NOUN: animal, beast, land mammal
suli
     ADJECTIVE: big, heavy, large, long, tall; important; adult
suno
     NOUN: sun; light, brightness, glow, radiance, shine; light source
supa
     NOUN: horizontal surface, thing to put or rest something on
suwi
     ADJECTIVE: sweet, fragrant; cute, innocent, adorable
tan
     PREPOSITION: by, from, because of
taso
     PARTICLE: but, however
     ADJECTIVE: only
tawa
     PREPOSITION: going to, toward; for; from the perspective of
     ADJECTIVE: moving
telo
     NOUN: water, liquid, fluid, wet substance; beverage
tenpo
     NOUN: time, duration, moment, occasion, period, situation
toki
     VERB: to communicate, say, speak, say, talk, use language, think
tomo
     NOUN: indoor space; building, home, house, room
tu
     NUMBER: two
unpa
     VERB: to have sexual or marital relations with
uta
     NOUN: mouth, lips, oral cavity, jaw
utala
     VERB: to battle, challenge, compete against, struggle against
walo
     ADJECTIVE: white, whitish; light-coloured, pale
wan
     ADJECTIVE: unique, united
     NUMBER: one
waso
     NOUN: bird, flying creature, winged animal
wawa
     ADJECTIVE: strong, powerful; confident, sure; energetic, intense
weka
     ADJECTIVE: absent, away, ignored
wile
     PRE-VERB: must, need, require, should, want, wish
"""

WORD = ""
for line in _DICT_STR.split('\n'):
    if not line:
        continue
    if line[0] != ' ':
        WORD = line.strip()
        continue
    split = line.strip().split(':')
    if not tokipona_to_en.get(WORD):
        tokipona_to_en[WORD] = []
    part = Part[split[0].replace('-', '_')]
    newdef = TranslationEntry(part, split[1].strip())
    tokipona_to_en[WORD].append(newdef)

# Mostly-duplicate words. There are slight differences in connotation
# tokipona_to_en['ali'] = tokipona_to_en['ale']
# tokipona_to_en['kin'] = tokipona_to_en['a']
# tokipona_to_en['oko'] = tokipona_to_en['lukin']
# tokipona_to_en['namako'] = tokipona_to_en['sin']

# uncommon words, not in minimal dictionary set.
tokipona_to_en['monsuta'] = [  # 85% usage
    TranslationEntry(Part.NOUN, "fear, dread, monster, predator, threat")]
tokipona_to_en['leko'] = [  # 72% usage
    TranslationEntry(Part.NOUN, "stairs, square, block, corner, cube"),
    TranslationEntry(Part.ADJECTIVE, "square, blocky")]
# tokipona_to_en['kipisi'] = [  # 73% usage
#     TranslationEntry(Part.VERB, "split, cut, slice, sever"),
#     TranslationEntry(Part.ADJECTIVE, "sharp")]
# tokipona_to_en['kijetesantakalu'] = [  # 73% usage
#     TranslationEntry(Part.NOUN, "racoon, ringtail")]
# tokipona_to_en['majuna'] = [  # 32% usage
#     TranslationEntry(Part.ADJECTIVE, "old, aged, ancient")]
# tokipona_to_en['powe'] = [  # 21% usage
#     TranslationEntry(Part.ADJECTIVE, "unreal, false, untrue"),
#     TranslationEntry(Part.VERB, "pretend, deceive, trick")]
# tokipona_to_en['apeja'] = [  # 20% usage
#     TranslationEntry(Part.NOUN, "guilt, shame, stigma"),
#     TranslationEntry(Part.VERB, "shun, accuse, expose, dishonor, embarrass")]
# tokipona_to_en['po'] = [  # 5% usage
#     TranslationEntry(Part.NUMBER, "four")]
# tokipona_to_en['tuli'] = [  # 4% usage
#     TranslationEntry(Part.NUMBER, "three")]

# ensure token list and definition list match
if __name__ == "__main__":
    if set(tokipona_to_en.keys()) != set(normal_words):
        print("Definition list and token list don't match...")
        # check to see what's missing
        for word in normal_words:
            if not tokipona_to_en.get(word):
                print("missing definition for", word)

        # check to see what's extra
        for key in tokipona_to_en:
            if key not in normal_words:
                print("extra definition for", key)
    else:
        print("Consistency check passed!")
