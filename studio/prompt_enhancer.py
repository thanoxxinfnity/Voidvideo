"""
Prompt enhancer: a short or broken prompt (English or Hinglish) -> the caption-style prompt our LoRA was trained on.

Rule-based (instant, no GPU): Hinglish word map, @character / #effect cards (scripts/expand_prompt.py), then fills what is
missing - shot size, camera move, setting/light, effects, style tail - from keywords in the prompt. `enhance_llm()` is an
optional second pass with a small local instruct model for free-form rewriting (needs transformers; not used by default).
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import expand_prompt as ep  # noqa: E402

HINGLISH = {
    "ladai": "fight", "lad": "fight", "ladta": "fights", "karta": "", "karti": "", "maar": "hits", "marta": "hits", "mukka": "punch",
    "lat": "kick", "talwar": "sword", "bhaag": "run", "bhagta": "runs", "daudta": "runs", "daud": "run", "kood": "jump",
    "udta": "flies", "rota": "cries", "roti": "cries", "hansta": "laughs", "hansti": "laughs", "gussa": "angry", "gusse": "angry",
    "dar": "fear", "raat": "night", "din": "day", "baarish": "rain", "barish": "rain", "aag": "fire", "bijli": "lightning",
    "dhuan": "smoke", "pani": "water", "barf": "snow", "sheher": "city", "gali": "alley", "chhat": "rooftop", "jungle": "forest",
    "ladka": "boy", "ladki": "girl", "aadmi": "man", "aurat": "woman", "raakshas": "monster", "rakshas": "monster",
    "shaitan": "demon", "bheed": "crowd", "dhimi": "slow", "tez": "fast", "dhire": "slowly", "khoon": "blood",
    "ka": "of", "ki": "of", "ke": "of", "mein": "in", "me": "in", "par": "on", "pe": "on", "aur": "and", "se": "against",
    "ko": "to", "hai": "", "hain": "",
}
SHOTS = [(r"close[- ]?up|face|expression|eyes|cries|laughs|angry", "Close-up shot"),
         (r"crowd|army|city|street|battlefield|landscape|wide|rooftop", "Wide shot"),
         (r".", "Medium shot")]
MOVES = [(r"chase|run|runs|race|fly|flies|dash", "camera tracks sideways with the subject"),
         (r"fight|punch|kick|slash|clash|battle|explod", "handheld camera with slight shake on impacts"),
         (r"sad|cries|tears|alone|slow|remember", "slow push-in"),
         (r"reveal|monster|giant|tower|castle", "camera tilts up slowly"),
         (r".", "static camera")]
LIGHT = [(r"night|neon|city lights", "night scene, neon and moonlight, deep blue shadows"),
         (r"sunset|dusk|evening", "golden-hour light, long orange shadows"),
         (r"rain|storm|fog", "overcast moody light, wet reflections"),
         (r"fire|explod|lava", "strong orange firelight on the faces"),
         (r".", "clear daylight, soft shadows")]
EFFECTS = [(r"punch|kick|slash|hit|clash", "impact flash, speed lines, debris"),
           (r"fire|burn|explod", "hand-drawn flames and smoke"),
           (r"lightning|electric|thunder", "branching lightning sparks"),
           (r"water|rain|wave|splash", "splashing water droplets"),
           (r"run|dash|chase", "speed lines, dust trail")]
STYLE = "Traditional 2D hand-drawn anime"


def _first(rules, text):
    for pat, out in rules:
        if re.search(pat, text, re.I):
            return out
    return ""


def translate(prompt):
    toks = re.findall(r"[#@\w']+|[.,;]", prompt)
    return " ".join(w for w in (HINGLISH.get(t.lower(), t) for t in toks) if w).replace(" ,", ",").replace(" .", ".")


def enhance(prompt, cards=None, vfx=None, camera=None):
    """-> (enhanced prompt, report dict of what was added)."""
    cards = cards if cards is not None else {k.lower(): v for k, v in json.load(open(ep.DEFAULT_CARDS)).items()}
    vfx = vfx if vfx is not None else {k.lower(): v for k, v in json.load(open(ep.VFX_CARDS)).items()}
    text = translate(prompt)
    added = {}
    has_cam = re.search(r"shot|close-up|camera|pan|zoom", text, re.I)
    shot = camera or ("" if has_cam else f"{_first(SHOTS, text)}, {_first(MOVES, text)}.")
    if shot:
        added["camera"] = shot
    base = ep.expand(text, cards, camera=shot or "", vfx=vfx).strip()
    out = base.replace(f" {ep.STYLE}", "").replace(ep.STYLE, "").rstrip()
    extra = []
    if not re.search(r"night|day|sunset|dusk|light|sun|rain", text, re.I):
        light = _first(LIGHT, text)
        added["lighting"] = light
        extra.append(light.capitalize() + ".")
    if "Effects:" not in out:
        eff = _first(EFFECTS, text)
        if eff:
            added["effects"] = eff
            extra.append(f"Effects: {eff}.")
    out = (out if out.endswith(".") else out + ".") + (" " + " ".join(extra) if extra else "") + f" {STYLE}"
    if re.search(r"fast|fight|punch|slash|run", text, re.I):
        out += ", animated on ones"
        added["timing"] = "animated on ones"
    return re.sub(r"\s+", " ", out + ".").strip(), added


def enhance_llm(prompt, model_id="Qwen/Qwen2.5-1.5B-Instruct"):
    """Optional free-form rewrite with a small local model (downloads it). Returns the rule-based text on any failure."""
    base, _ = enhance(prompt)
    try:
        from transformers import pipeline
        gen = pipeline("text-generation", model=model_id, device_map="auto")
        msg = [{"role": "system", "content": "Rewrite as one dense English video prompt for a 2D anime generator: shot size, camera move, "
                                             "subjects with looks and left/right positions, explicit body motion, setting, light, effects. "
                                             "No names, no commentary."},
               {"role": "user", "content": base}]
        return gen(msg, max_new_tokens=220, do_sample=False)[0]["generated_text"][-1]["content"].strip()
    except Exception:
        return base
