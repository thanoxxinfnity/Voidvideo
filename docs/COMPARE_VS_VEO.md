# Our model vs Google Veo (same prompts)

Idea: give the **same text** to both, then compare **blind** (you do not know which clip is which) and score. Prompts 1-6 are the exact ones our model has already generated (clips are in the chat); 7-10 are new, harder ones.

## Fair rules
1. Paste the prompt **exactly as written** into Veo. Do not add words. 16:9, no audio, 720p is enough.
2. Veo makes longer clips: compare only the **first 2 seconds** (the script cuts both to 2.04 s).
3. Make **3 takes** on each side and keep the best take of each (video AI is a lottery; one take proves nothing).
4. Name the files with the same name on both sides (`monster.mp4` in `ours/` and `monster.mp4` in `veo/`).
5. Run `python scripts/compare_blind.py ours veo blind` (or send me both folders), watch `blind/*.mp4` (left = A, right = B), score, **then** `--reveal`.

## Prompts
**1. `monster`**

> Close-up shot, slight zoom-in as a monster lunges forward. One monster with red and black fur, yellow eyes and sharp teeth snarls with its mouth open, its body tensed for attack, outdoors in daylight. Traditional 2D hand-drawn anime, smear frames.

**2. `cheetah`**

> Wide shot, camera tracks sideways. A cheetah sprints across a golden savanna, its legs stretching and folding in a fast gallop, its spotted body flexing, dust rising behind it. Traditional 2D hand-drawn anime.

**3. `crowd`**

> Wide high-angle shot, static camera. A big crowd of about twenty people in different clothes runs down a wide city street towards the camera, arms swinging, some looking back in panic. Traditional 2D hand-drawn anime.

**4. `lightning`**

> Medium shot, static camera. A young man with spiky black hair raises his fist and bright blue lightning crackles around his arm, branching sparks flashing and lighting up his face and the dark walls. Anime special effects, hand-drawn.

**5. `ice`**

> Wide shot, static camera. A girl swings her hand and a wave of ice spreads across the ground, sharp ice crystals shooting up and freezing a wall, white frost mist drifting. Anime special effects, hand-drawn.

**6. `fire`**

> Medium shot, slow push-in. A swordsman swings a sword wrapped in orange flames, the fire trailing behind the blade in a bright arc, sparks and embers flying, smoke curling up. Anime special effects, hand-drawn.

**7. `expression`**

> Close-up shot, static camera. A teenage girl with long black hair faces the camera. Her expression shifts from calm to shock: her eyes widen, her pupils shrink, her mouth drops slightly open, a single tear forms. Traditional 2D hand-drawn anime.

**8. `punch`**

> Medium wide shot, handheld camera with slight shake on impact. On the left a man with spiky pink hair punches, on the right a tall man with white hair blocks with his forearm, dust and debris flying, impact flash and speed lines. Traditional 2D hand-drawn anime.

**9. `talk`**

> Medium shot, static camera. Two students stand facing each other on a school rooftop at sunset: on the left a tall boy with short white hair, on the right a shorter boy with spiky black hair in a dark uniform. The black-haired boy talks and gestures with his hand while the other listens with his arms crossed. Traditional 2D hand-drawn anime.

**10. `battle`**

> Wide high-angle shot, static camera. A huge battle in a ruined city: dozens of warriors clash with a giant monster, swords flashing, dust and smoke rising, blood splatter on the impact frames. Traditional 2D hand-drawn anime.

## Score sheet (1 = bad ... 5 = excellent, per clip A and B)

| Prompt | Motion / physics | Faces + hands | Anime look (hand-drawn, not 3D/real) | Effects (fire, lightning, impact) | Followed the prompt | Total A | Total B |
|---|---|---|---|---|---|---|---|
| monster |  |  |  |  |  |  |  |
| cheetah |  |  |  |  |  |  |  |
| crowd |  |  |  |  |  |  |  |
| lightning |  |  |  |  |  |  |  |
| ice |  |  |  |  |  |  |  |
| fire |  |  |  |  |  |  |  |
| expression |  |  |  |  |  |  |  |
| punch |  |  |  |  |  |  |  |
| talk |  |  |  |  |  |  |  |
| battle |  |  |  |  |  |  |  |

## What to expect (honest)
- Veo will almost surely win on raw sharpness, faces, physics and length. That is a model 100x bigger than ours.
- Where we can still win or tie: **hand-drawn anime look + timing** (ones/twos/threes, smear frames, impact frames), because we train on real anime clips and Veo tends to drift to a smooth 3D-ish look.
- Judge **anime look** and **effects** separately from **faces/hands**, so you see where the gap is and what to train next.
- Our clips are 2 s, 720x480, from a 1.3B model with ~200 LoRA steps, with no layering or retakes yet.
