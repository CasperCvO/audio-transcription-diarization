"""Versioned prompt templates for summarization (plan B6).

The `PROMPT_VERSION` constant is recorded in every run's `meta.json` so
outputs are reproducible and comparable across runs. Bump the version
whenever any of the strings below changes.
"""

from __future__ import annotations

PROMPT_VERSION = "v0.1.0"

SYSTEM_PROMPT_NL = """\
Je bent een nauwkeurige vergaderassistent. Je krijgt het gediariseerde \
transcript van een Nederlandstalige vergadering met sprekers en \
tijdstempels. Werk altijd in het Nederlands. Verzin nooit informatie; \
baseer elke uitspraak direct op het transcript. Citeer waar mogelijk de \
sprekers. Geef de uitvoer terug als strikt geldige JSON volgens het \
gevraagde schema."""

MAP_PROMPT_NL = """\
Hieronder volgt een deel van een gediariseerd vergadertranscript. Dit \
deel staat niet op zichzelf: het is een venster in een langere vergadering. \
Lees het zorgvuldig en vat per deel samen wat er gebeurde.

Geef **uitsluitend** een JSON-object terug met exact deze velden:
- `local_topics`: lijst van {"title": str, "bullets": [str, ...]}
- `local_decisions`: lijst van {"text": str, "segment_idx": int | null}
- `local_actions`: lijst van {"task": str, "owner": str | null, "due": str | null, \
"segment_idx": int | null}
- `local_questions`: lijst van strings (open vragen uit dit venster)
- `quotes`: lijst van {"speaker": str, "text": str, "segment_idx": int | null}

`segment_idx` verwijst naar het indexnummer van een segment in dit venster \
(begint bij 0 in het gegeven venster).

Richtlijnen:
- Gebruik dezelfde naam/label voor een spreker als in het transcript.
- Laat lijsten leeg als er niets relevants in dit venster staat; verzin niets.
- Houd bullets kort en feitelijk.
"""

REDUCE_PROMPT_NL = """\
Je krijgt een lijst met JSON-samenvattingen, elk over een opeenvolgend deel \
van dezelfde vergadering. Consolideer ze tot één globale samenvatting. \
Ontdubbel vergelijkbare beslissingen en voeg actiepunten samen wanneer \
`owner` en de kern van de taak overeenkomen. Bewaar de oorspronkelijke \
volgorde van gebeurtenissen.

Geef **uitsluitend** een JSON-object terug met dit schema:
{
  "title": str,               # korte titel in het Nederlands, max ~8 woorden
  "tldr": [str, ...],         # 3-5 bullets, elk 1 zin
  "topics": [                 # chronologisch, niet alfabetisch
    {"title": str, "bullets": [str, ...]}
  ],
  "decisions": [{"text": str}],
  "action_items": [
    {"task": str, "owner": str | null, "due": str | null}
  ],
  "open_questions": [str, ...],
  "next_steps": [str, ...]
}

Regels:
- Alles in het Nederlands.
- Verzin niets; werk alleen met de gegeven lokale samenvattingen.
- Wees specifiek over wie welke actie neemt; laat `owner` op null als dit \
  niet duidelijk uit het transcript blijkt.
"""

CRITIQUE_PROMPT_NL = """\
Je bent een kritische reviewer. Ik geef je (1) het volledige gediariseerde \
transcript en (2) een conceptsamenvatting in JSON.

Jouw taak:
- Controleer of elke beslissing en elk actiepunt echt uit het transcript komt.
- Markeer ontbrekende beslissingen of actiepunten die in het transcript \
  voorkomen maar in de samenvatting ontbreken.
- Markeer hallucinaties (uitspraken die niet in het transcript staan).
- Stel correcties voor.

Geef **uitsluitend** een JSON-object terug met dit schema:
{
  "missing_decisions": [{"text": str, "segment_idx": int | null}],
  "missing_actions": [{"task": str, "owner": str | null, "due": str | null, \
"segment_idx": int | null}],
  "hallucinations": [{"field": str, "index": int, "reason": str}],
  "corrections": [{"path": str, "new_value": <any>, "reason": str}]
}

`path` gebruikt dezelfde sleutels als de samenvatting (bijv. \
"action_items[2].owner"). Laat lijsten leeg als er niets aan te merken is.
"""

NAME_RESOLUTION_PROMPT_NL = """\
Hieronder volgt het begin van een vergadertranscript (eerste paar minuten). \
Probeer voor elke speaker-label (bijv. SPEAKER_00) de echte naam te \
achterhalen uit introducties ("Ik ben ...", "Mijn naam is ...").

Geef **uitsluitend** een JSON-object terug met het patroon:
{"SPEAKER_00": "Naam" of null, "SPEAKER_01": "Naam" of null, ...}

Als een naam niet duidelijk uit het transcript volgt, geef null terug.
Verzin geen namen.
"""
