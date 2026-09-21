#!/usr/bin/env python3
"""
Runs INSIDE a Kaggle GPU kernel. Generates short anime-style clips and
uploads each one straight to the VoidVideo Vault site
(voidvideo-vault.vercel.app) as it finishes, tagged with the category the
prompt was written for and captioned from that same prompt.

Uses CogVideoX-2b -- the model this project actually trains a LoRA on,
already verified working end-to-end on Kaggle's free GPU. Wan2.2-TI2V-5B
was tried and dropped: it survived a cheap smoke test but got OOM-killed
by the OS during real generation (VAE decode of a full 49-frame clip),
wasting a full model-load-plus-40-minute cycle for zero output.

Uploads happen one clip at a time as soon as each is generated, so a kernel
that gets cut off by Kaggle's session time limit still keeps everything
generated up to that point -- nothing is lost by stopping early.

Two quality gates run before anything is allowed to upload, and a clip that
fails either one is re-rolled with a new seed (a few times) and then dropped
entirely rather than ever reaching the site:
  1. frame_looks_blank() -- catches solid white/gray garbage frames.
  2. frame_looks_uncolored() -- catches real-but-uncolored sketch/lineart
     output. This one is a hard, non-negotiable requirement: an uncolored
     clip must never be uploaded, even if everything else about it is fine.
Quality now takes priority over wall-clock time (MAX_SECONDS is a generous
backstop against a stuck session, not a race to finish fast).

This is a stopgap for padding out under-filled categories when real footage
is running low on time, not a replacement for real hand-drawn reference
clips: whichever base model ends up running here is un-tuned on this
project's art style, so output quality and anime-style adherence will be
inconsistent. Treat generated clips as filler, review them if you can.

Pushed to Kaggle by scripts/push_clip_generator_to_kaggle.py.
"""
import random
import re
import signal
import subprocess
import sys
import time
from pathlib import Path


def pip_install():
    subprocess.run(
        [
            sys.executable, "-m", "pip", "install", "-q", "-U",
            "diffusers>=0.35.0", "transformers>=4.44", "accelerate>=0.33",
            "imageio[ffmpeg]", "sentencepiece", "requests",
        ],
        check=True,
    )


pip_install()

import numpy as np  # noqa: E402
import requests  # noqa: E402
import torch  # noqa: E402
from diffusers import CogVideoXPipeline  # noqa: E402
from diffusers.utils import export_to_video  # noqa: E402

SITE = "https://voidvideo-vault.vercel.app"
COGVIDEOX_MODEL_ID = "THUDM/CogVideoX-2b"
NUM_FRAMES = 49  # matches configs/training_config.yaml (t2v) exactly
FPS = 8
# Reduced from 480x720: the real OOM wasn't about model weight dtype at all
# (bf16 correctly halved weight memory to ~6GB) -- it was CogVideoX's
# spatio-temporal attention trying to allocate a single 70GB tensor, which
# scales with total tokens (frames x spatial patches) and has nothing to do
# with dtype. Smaller resolution directly shrinks that token count.
HEIGHT = 320
WIDTH = 480
NUM_INFERENCE_STEPS = 30  # reverted from 40: at 4 attempts/prompt worst case, 40 steps meant
# up to ~60 real minutes stuck on one hard prompt with zero visible progress or way to check
# in (Kaggle gives no live logs for a running kernel). 30 is the config already verified
# working end-to-end earlier in this project -- faster feedback loop matters more here.
GUIDANCE_SCALE = 6.0
MAX_RETRIES_PER_PROMPT = 2  # reverted from 3, same reasoning -- fail faster, move on sooner
# User's stance shifted from "hard 5-6h cap no matter what" to "take the time you
# need for real quality, just don't burn the whole day for nothing" -- so this is
# now a backstop against an actually-stuck/runaway session, not the primary
# quality control (the blank/color gates below are). Kaggle free GPU sessions cap
# out around 9h on their own anyway, so 8h leaves a safety margin under that.
MAX_SECONDS = 8 * 3600

OUT_DIR = Path("/kaggle/working/generated")
OUT_DIR.mkdir(parents=True, exist_ok=True)

STYLE_SUFFIX = (
    ", modern 2D anime film style, fully colored with natural cel shading and "
    "soft gradient shadows, detailed painted background art, clean inked "
    "outlines, natural realistic anime color palette, consistent character "
    "design, smooth cinematic timing"
)
NEGATIVE_PROMPT = (
    "black and white, grayscale, monochrome, sketch, pencil sketch, line art "
    "only, uncolored, unfinished, storyboard, rough draft, flat cartoon "
    "colors, oversaturated, plain empty background"
)

# Base prompts per category, matching the motion sub-types in data/README.md.
# User-authored, professional-grade cinematic prompts (each already bakes in
# its own "strictly no 3D/CGI/photorealism" art-direction). "Special power"
# VFX prompts (energy blasts, auras, teleports, domain expansions) the user
# also sent are deliberately NOT included here: this LoRA's stated scope is
# human motion primitives (locomotion/gestures/expressions/secondary-motion,
# see data/README.md), not supernatural VFX authoring -- flagged to the user,
# easy to add back as a 5th category if they actually want it trained too.
# The combat prompts they sent were split by which primitive they actually
# demonstrate: kicks/throws/footwork -> locomotion, punches/blocks/strikes
# with the feet planted -> gestures.
PROMPTS = {
    "locomotion": [
        "A lean teenage sorcerer in a tattered navy-black high-collar uniform sprinting frantically down a narrow, rain-slicked Tokyo back-alley. Low-angle tracking shot focusing on boots splashing through shallow puddles. 2D traditional cel animation, sharp ink lines, hand-drawn splash in-betweens, flat shadows. Strictly no 3D CGI, no motion blur, no glowing trails, no photorealism.",
        "Side-view silhouette of a tall figure with spiky white hair and an oversized black trench coat striding deliberately along a concrete rooftop ledge at dusk. Horizon lit by deep crimson sunset. Snappy 24fps walk cycle, crisp anime keyframing, stylized high-contrast cel shading. Strictly no 3D mesh, no plastic textures, no digital warp.",
        "An athletic martial artist in grey linen forearm wraps jumping backwards off one foot to create distance, landing low on both boots and skidding across a wooden dojo floor with white dust kicking up. Dynamic keyframe physics, fluid hand-drawn smear frames, locked static camera. Strictly no 3D ragdoll, no CGI particles.",
        "Full-body front shot of a battle-worn fighter dragging their boots forward on broken asphalt, shoulders slumped, knees trembling and buckling slightly before recovering balance. Authentic 2D anatomical struggle, crisp linework, muted cel-shaded colors, locked camera. Strictly no rubbery limbs, no CGI.",
        "Medium shot from behind a warrior with a low braided ponytail walking out from deep subway tunnel shadows into glaring platform daylight, dark coat draping naturally, smooth weight transitions per step. Classic MAPPA pacing, pure hand-drawn 2D. Strictly no 3D bloom, no lens flare.",
        "A young operative in a beige utility vest sprinting along a corridor wall, planting their right heel hard, and pivoting sharply around a 90-degree corner with heavy inertia control. Crisp hand-drawn in-betweens, flat lineart. Strictly no 3D rendering, no digital warp.",
        "Medium side-angle tracking shot of a monk in white and charcoal robes stepping up mossy stone shrine stairs two at a time, athletic knee lifts, balanced cloth drape. Traditional Japanese cel animation, clean ink strokes. Strictly no photorealism.",
        "Front 3/4 view of a tall woman in a slate-grey business suit walking forward calmly across a floor littered with shattered glass, glass shards remaining flat 2D drawings, steady rhythmic heels. Sharp line weight, locked frame. Strictly no 3D reflections, no CGI.",
        "Side view of an unarmed fighter in black hakama pants sliding forward in a low combative kendo shuffle across tatami mats, heels staying grounded, upper body motionless. Clean anatomical posture, pure 2D. Strictly no 3D models.",
        "Full shot of a parkour courier in cargo joggers running at a galvanized rooftop air-vent, planting one palm on the metal, and vaulting cleanly over to land in a running stride. Dynamic keyframe arcs, crisp flat shading. Strictly no CGI blur.",
        "A schoolboy holding a yellow umbrella walking slowly across a misty pedestrian bridge, stopping mid-step, hesitating as boots hover, then planting down. Moody atmospheric 2D watercolor background with bold ink character lines. Strictly no 3D mesh.",
        "Wide side tracking shot of a rebel warrior sprinting through a dense bamboo grove, boots sinking slightly into dark mud with small hand-drawn mud clods flying backward. Authentic 2D animation timing. Strictly no 3D particle systems.",
        "A boy with messy blonde undercut walking forward with both hands stuffed deep into leather bomber jacket pockets, shoulders rolling casually with each step, calm gaze straight ahead. Classic cel shading, locked shot. Strictly no plastic textures.",
        "Medium shot of an injured youth pressing a bloodied palm against a tiled wall, limping forward one dragging step at a time, fingers sliding leaving a faint red hand-drawn smear. Intense dramatic 2D linework. Strictly no CGI.",
        "A girl with twin buns walking down a spiraling metal overpass staircase, hand gliding along the green railing, skirts swaying in sync with footsteps, high-angle framing. Traditional 24fps anime pacing. Strictly no 3D clipping.",
        "Side profile of a scout crawling forward on hands and knees through a cramped concrete utility shaft, knees shifting forward deliberately, head tilted to avoid ceiling pipes. Accurate 2D foreshortening. Strictly no digital distortion.",
        "Frontal camera shot of a fighter bursting forward in a short explosive two-meter sprint, braking suddenly with right foot forward and fists raised into a high guard. Sharp 2D impact timing, crisp outlines. Strictly no 3D models.",
        "A student in an unbuttoned school blazer striding confidently through a vacant Shibuya crosswalk at dusk, long purposeful strides, static camera positioned low to the ground. Pure 2D hand-drawn aesthetic. Strictly no CGI glow.",
        "Medium full shot of a martial artist stepping rapidly to the left to dodge an incoming line of fire, weight shifting smoothly onto the ball of the left foot, torso leaning with discipline. Crisp 2D linework. Strictly no motion blur.",
        "Wide shot of a figure running along an elevated rusted catwalk inside an abandoned warehouse, silhouette backlit by cold moonlight, steady side-panning motion. Traditional cel animation. Strictly no 3D particle lighting.",
        "Low-angle macro shot of black military combat boots stepping down into a shallow puddle, water fanning out in stylized flat 2D droplet arcs, sharp rim light. Classic sakuga anime splash frames. Strictly no 3D fluid sim.",
        "A young detective stepping backward three consecutive paces while keeping eyes locked straight ahead, heels lifting smoothly, trench coat trailing slightly behind. Steady locked-off camera, clean lines. Strictly no CGI.",
        "Side view of a street rogue jumping down from a rusted metal dumpster, bending knees upon landing on concrete to absorb shock, then standing tall. Natural squash-and-stretch anime physics. Strictly no 3D ragdoll.",
        "A doctor in a flying white lab coat sprinting urgently down a linoleum corridor, perspective lines converging sharply toward center, frantic hand-drawn foot cycles. Flat cel shading, locked angle. Strictly no digital blur.",
        "Front view of an adventurer in heavy winter coat leaning their upper torso forward into a harsh blizzard, boots planting heavily to resist the wind, face shielded by a forearm. Pure 2D hand-drawn resistance. Strictly no 3D snow particles.",
        "A cheerful high-school girl taking two playful skips along the curb of a suburban residential street, swinging her canvas bag in rhythm, bright daylight, clean crisp outlines. Traditional anime timing. Strictly no rubber limbs.",
        "Low-angle side shot of an assassin moving forward in a deep crouch along the shadow of a wooden fence, feet rolling quietly from heel to toe, eyes fixed ahead. High-contrast 2D shadows. Strictly no 3D models.",
        "A warrior trudging forward through ankle-deep swamp water, lifting each boot with visible effort, water dripping off the leather in clean 2D line droplets. Heavy kinetic weight, static camera. Strictly no CGI.",
        "Medium wide shot of a traveler walking away from camera into the pitch-black entryway of an ancient wooden temple, silhouette gradually vanishing into deep black cel shadows. Classic hand-drawn lighting. Strictly no 3D bloom.",
        "Side profile of a stoic security officer in a grey peaked cap marching with rigid, precise military strides along a chain-link perimeter fence. Crisp geometric ink lines, locked framing. Strictly no photorealism.",
        "Front medium shot of a passenger stepping off a moving rubber escalator threshold onto a shiny tile floor, fluid transition into a steady forward stride. Clean flat colors, 2D art. Strictly no 3D reflections.",
        "A distance runner with tied-back dark hair jogging at a relaxed warm-up pace down a riverside embankment, arms bent at ninety degrees pumping smoothly, side tracking shot. Authentic anime line art. Strictly no CGI.",
        "Wide profile of an explorer running toward a jagged rocky cliff edge, planting boots hard and skidding to a dead stop two inches from the drop, loose pebbles rolling off. Dynamic 2D anime framing. Strictly no 3D physics engine.",
        "A tall youth walking forward casually with both hands laced behind the back of his head, elbows flared out, whistling silently with relaxed shoulders. Clean cel shading, static camera. Strictly no CGI warp.",
        "Side view of a hunter stepping high over a mossy fallen log in an ancient forest, placing the leading boot down firmly on wet leaves, natural balance control. Hand-drawn 2D anatomy. Strictly no 3D clipping.",
        "A rogue sprinting at top speed, dropping down onto their side into a baseball slide to glide under a lowered metal shutter, popping up on the other side. Dynamic keyframed action, crisp lines. Strictly no 3D mesh.",
        "Back view of an underworld boss walking down a dimly lit concrete ramp, the heavy weighted hem of a fur-collared longcoat swaying rhythmically with each step. Moody 2D palette, locked shot. Strictly no plastic textures.",
        "A scout creeping along a concrete barrier, pausing at the edge, leaning head and shoulders forward to peek around the corner, then pulling back smoothly. Classic anime suspense pacing. Strictly no CGI.",
        "Side view tracking shot of bare feet walking along wet shoreline sand, gentle foam lapping over the toes, footprints drawn as simple flat 2D outlines. Muted colors, traditional anime art. Strictly no 3D water shaders.",
        "Medium shot from below of a technician in blue coveralls climbing an industrial iron wall ladder, alternating hands and boots with clean mechanical discipline. Flat cel shading, static camera. Strictly no 3D models.",
        "A battered martial artist walking forward, losing balance, and dropping straight down onto both knees on the dirt ground, palms slamming down to hold up the torso. Heavy 2D impact, settling dust lines. Strictly no 3D ragdoll.",
        "Low-angle dynamic shot of a fighter spinning low to the floor on one palm, extending their right leg in a wide circular sweep kick along the concrete, hand-drawn dust arcs kicking off the soles. Crisp 2D linework, heavy kinetic momentum. Strictly no 3D ragdoll, no digital warp.",
        "Full-body shot of a fighter leaping high into the air, driving a vertical heel drop axe kick straight down, leg extended in a clean foreshortened perspective arc, striking the ground with cracked stone lines. 2D cel animation. Strictly no 3D rendering.",
        "Side profile wide shot of a fighter pivoting 360 degrees on the ball of their front foot, driving a back-kick straight into the opponent's chest with explosive follow-through, stylized 2D impact rings. Snappy in-betweens, flat shading. Strictly no 3D models.",
        "Low-angle dynamic side shot of a fighter launching forward off one foot, tucking arms in and driving a devastating lead knee strike forward into the frame, hair whipping back with momentum. 2D cel-shaded anatomy, crisp lines. Strictly no 3D ragdoll.",
        "Full shot of a fighter feinting a low front kick, twisting the hip mid-air, and whipping the shin up high into a snap head kick, perfect anatomical rotation drawn in traditional anime frames. Flat cel shadows, static shot. Strictly no 3D mesh.",
        "Dynamic medium shot of a fighter grabbing the collar and sleeve of an attacker, turning their back, and cleanly flipping the opponent over their hip, slamming them flat onto tatami mats with bouncing dust lines. Pure 2D hand-drawn physics. Strictly no 3D ragdoll.",
        "Ground-level shot of a fighter jumping and wrapping both legs around the standing opponent's waist, twisting their core to drag them crashing down to the pavement, fluid 2D anatomical grappling. Crisp outlines, locked camera. Strictly no 3D clipping.",
        "A fighter pushing both palms hard into the opponent's torso to push off, backflipping through the air, and landing lightly in a low crouch three meters back. Snappy anime bounce physics, clean flat cel art. Strictly no CGI warp.",
        "Side profile of a fighter executing a reverse hook kick, heel snapping across the upper frame, clothes rippling with sudden inertia, settling back into an athletic southpaw stance. Crisp ink outlines. Strictly no 3D models.",
        "Full wide shot of an anime fighter taking two rapid strides up a vertical brick wall, leaping backward off the bricks, and delivering a dual-foot drop kick into the frame with heavy 2D kinetic force. Sakuga style timing. Strictly no CGI.",
        "Medium shot of a martial artist firing three consecutive, snapping high kicks off the same lead leg without touching the ground, balance pivoting on the rear foot, sharp 2D line weight. Flat cel colors. Strictly no rubbery limbs.",
    ],
    "gestures": [
        "A hand-drawn anime girl waving her hand cheerfully at the camera.",
        "A hand-drawn anime boy pointing forward with his finger.",
        "Close-up of scarred hands winding white cotton wraps around knuckles, holding the loose cloth end between clenched teeth to pull it tight before tucking it in. Detailed anatomical line art, flat tone shading. Strictly no 3D modeling, no rubber fingers.",
        "A fighter lunging forward and slamming their flat palm against the cracked pavement, fingers splaying outward as white stylized hand-drawn shockwave rings burst across the concrete. Classic impact frames, crisp cel shading. Strictly no 3D smoke, no digital particles.",
        "Close-up side profile of a hand gripping the braided hilt of a sheathed katana at the hip, thumb firmly clicking the brass tsuba guard forward by half an inch, exposing an inch of steel line art. Traditional 2D keyframing. Strictly no 3D CGI render.",
        "Medium close-up of a scarred antagonist using the back of a black-gloved thumb to wipe a streak of dark crimson blood from their chin, eyes locked forward with cold malice. Bold linework, dramatic chiaroscuro cel shadows. Strictly no photorealistic skin.",
        "Macro close-up of fingers striking a red sulfur match head against a matchbox strip, a small stylized 2D flame flaring to life instantly and illuminating the fingers with flat orange highlights. Traditional anime fire art. Strictly no 3D particle sim.",
        "Medium close-up of slender fingers tapping the bottom of a crumpled cardboard cigarette pack, catching the single exposed filter between index and middle finger, pulling it out cleanly. Crisp lineart. Strictly no CGI.",
        "Close-up of hands swinging open the fluted cylinder of a steel revolver, dropping three brass cartridge shells into the chambers with metallic precision, then snapping the cylinder shut. Masterclass mechanical 2D line work. Strictly no 3D assets.",
        "Close-up of a fighter using their left thumb to press against the knuckles of the right hand one by one, visible tendon flex and subtle finger pops, locked shot. Detailed hand anatomy, clean cel lines. Strictly no rubber limbs.",
        "Medium close-up of an assassin reaching across their chest to pull a curved combat knife from a shoulder sheath in a smooth reverse grip, spinning the blade once between fingers to lock it in place. Sharp 2D keyframes. Strictly no 3D warp.",
        "Medium shot of a street gambler flicking an old silver coin into the air with their thumb, tracking it with their eyes, and snatching it out of mid-air with a sharp downward fist grab. Crisp 2D arcs. Strictly no motion blur.",
        "Close-up of a cold intellectual character using the tip of their middle finger to push up the bridge of wireframe glasses, light glancing across the lenses as flat white diagonal opaque streaks. Pure anime visual language. Strictly no 3D glare.",
        "Medium close-up of a mentor tapping two fingers against a student's forehead with mild exasperation, finger recoil settling naturally, calm locked-off camera. Traditional Kyoto Animation style lineart. Strictly no CGI.",
        "Close-up of trembling fingers tightening around a folded parchment paper, crushing it into a tight, crumpled paper ball in a clenched fist, sharp creases drawn in bold ink lines. Hand-drawn 2D physics. Strictly no 3D mesh deformation.",
        "Medium shot of a street vigilante reaching behind with both hands to yank a dark oversized hood up and over their cap, casting the upper half of their face in heavy pitch-black shadow. High-contrast cel shading. Strictly no CGI bloom.",
        "Close-up of a thumb flicking open the hinged lid of a silver Zippo lighter with a crisp mechanical snap, thumb rolling the flint wheel to create a clean stylized hand-drawn spark. Crisp lineart. Strictly no 3D smoke.",
        "Low-angle shot of a sorcerer crouching, dragging a thick stick of white chalk across rough slate stone, forming the arc of a ritual seal with powdery white line fragments. Traditional 2D effects. Strictly no digital glow.",
        "Close-up of two hands holding wooden disposable chopsticks horizontally, breaking them apart cleanly down the middle seam with a sharp snap, fingers steady. Everyday natural anime observation, clean outlines. Strictly no 3D.",
        "Medium shot of a weary soldier extending their right arm forward, fingers relaxed, palm open flat toward the camera in an earnest gesture of ceasefire, subtle hand tremor. Delicate line weight, static shot. Strictly no CGI.",
        "Extreme close-up of a hand clenching down around a shattered jade pendant, sharp edges pressing into the palm, a single drop of crimson blood welling between fingers. Dramatic Shonen ink lines. Strictly no 3D assets.",
        "Medium shot of a wanderer gathering the loose edges of their travel cloak and tucking both forearms inside the fabric across their chest, smooth draped cloth folds settling. Clean flat colors. Strictly no 3D cloth physics.",
        "Frontal medium close-up of a detective extending an arm straight toward the camera lens, holding a black automatic pistol with two-handed grip, intense steady eyes behind iron sights. Bold foreshortened lineart. Strictly no 3D CGI models.",
        "Close medium tracking shot of an athletic sorcerer throwing a lightning-fast left jab followed by a heavy right straight punch, crisp hand-drawn motion blur smears, snapping back into guard. Authentic 2D sakuga animation, flat cel shading, locked-in framing. Strictly no 3D CGI, no plastic models, no digital bloom.",
        "Medium close-up of a martial artist parrying an incoming fist with their forearm, stepping in close and driving an upward elbow strike toward the opponent's chin, bold impact line burst. Traditional 2D keyframing, sharp black outlines. Strictly no CGI particles.",
        "Medium shot of a fighter raising both forearms to absorb a frontal hit, absorbing the kinetic shock, then ducking low and firing a vicious left hook to the body, muscles flexing with hand-drawn tension. Classic anime fighting weight. Strictly no CGI.",
        "Frontal medium view of two fighters locked in close combat, both exchanging a rapid rhythm of fists with alternating hand-drawn white smear frames, focused aggressive eyes, locked camera. Sakuga 2D line art, raw energy. Strictly no digital blur, no CGI mesh.",
        "Medium close-up of a fighter wrapping both hands behind the opponent's neck in a Muay Thai clinch, pulling them down while driving two rapid, rhythmic knee strikes into the midsection. Masterclass 2D kinetic weight, sharp ink lines. Strictly no CGI.",
        "A fighter slapping an incoming punch away with their open left hand, twisting hips smoothly, and slamming their open right palm into the opponent's chest with a visible white hand-drawn shockwave ring. Crisp 2D impact timing. Strictly no 3D particle simulation.",
        "Close medium shot of an anime boxer slipping beneath a wide hook, bending at the waist, and launching an explosive upward uppercut that shears through the air with clean white speedlines. Traditional 24fps keyframes. Strictly no CGI.",
        "Extreme close-up on two fighters trading simultaneous punches across each other's shoulders, fists grazing jaws, white hand-drawn shockwave smears radiating between their snarling faces. Intense Shonen 2D linework. Strictly no 3D models.",
        "Low-angle shot looking up from the ground at a dominant fighter mounted on top, raining down rhythmic alternating hammerfists with intense facial fury, heavy down-strokes drawn in thick brush ink. Pure traditional cel animation. Strictly no CGI.",
    ],
    "expressions": [
        "Extreme close-up on a pale antagonist's face, pupils shrinking to needlepoints, lower eyelids twitching upward, mouth tearing open into a wide, jagged, maniacal smile displaying pointed teeth. Heavy Shonen ink hatching lines, intense cel shadows. Strictly no 3D models, no plastic skin.",
        "Close-up of a girl's face drenched in rain, eyebrows pinching upward at the center in sheer heartbreak, lower lip trembling silently, stylized hand-drawn water streaks running down cheeks without digital blur. Pure traditional 2D animation drama. Strictly no photorealism.",
        "Close-up side profile of a scarred veteran, eye completely deadpan and unblinking, slowly rotating his gaze to stare directly at the viewer with an ominous downward brow furrow. Heavy ink line weight, pitch-black cel shadows. Strictly no CGI glow, no digital warping.",
        "Extreme close-up on wide hazel eyes, the irises vibrating with a subtle two-frame hand-drawn tremor, pupils instantly contracting in raw horror, mouth dropping slightly open. Crisp psychological anime framing. Strictly no 3D eyes.",
        "Close-up of a mischievous rogue winking their left eye shut with an arched brow, while the right eye sparkles with a confident hand-drawn glint, corner of mouth curved in a sharp smirk. Clean cel shading. Strictly no 3D face mesh.",
        "Extreme close-up on a fighter's jaw and mouth, chest rising and falling violently, parted lips releasing rapid silent gasps, sweat droplets drawn as sharp geometric cel-shaded highlights. High-tension anime drawing. Strictly no CGI.",
        "Close-up of a student forcing a wide, trembling smile while tears well up at the corners of shut eyes, eyebrows tilted in painful gratitude, soft morning light cel shading. Emotional Kyoto Animation aesthetic. Strictly no 3D skin.",
        "Close-up of completely empty, glazed anime eyes with no highlight reflections, blank expressionless mouth, utterly motionless head holding a disturbing deadpan gaze. Crisp 2D line work, locked frame. Strictly no digital blur.",
        "Close-up of an anime antagonist curling their upper lip back in disgust, nose wrinkling with sharp hand-drawn crease lines, eyes looking down with cold supremacy. Dramatic high-contrast shadows. Strictly no 3D textures.",
        "Close-up of an investigator, eyes widening as puzzle pieces click, blinking twice in rapid succession, mouth parting to form an unspoken word, head lifting an inch. Fluid anime acting keyframes. Strictly no CGI.",
        "Close-up of a tired programmer or researcher, heavy dark bags under eyes, eyelids drifting halfway down over tired irises, head giving a subtle downward nod before jerking awake. Realistic anime timing. Strictly no 3D.",
        "Extreme close-up on mouth, small white teeth sinking into the soft fleshy lower lip, leaving an indented white mark as tension builds, subtle chin tremor. Delicate hand-drawn lines. Strictly no photorealism.",
        "Close-up front view of a dark-haired assassin, face tilted downward so eyes peer out from underneath heavy shadowed bangs, pupils blazing with cold hatred. Flat high-contrast anime tones. Strictly no glowing eye effects, no 3D.",
        "Close-up of a character closing their eyes gently, furrowed brows softening, lips parting as a long silent breath of tension leaves the body, shoulders visibly dropping out of frame. Delicate 2D keyframing. Strictly no CGI.",
        "Medium close-up of a sharp-tongued girl rolling her eyes upward in an exaggerated circular arc of pure annoyance, folding her arms and turning her chin away with a haughty 'hmph' posture. Crisp cel art. Strictly no rubber limbs.",
        "Extreme macro close-up on a man's sharp jawline, the masseter muscle bulging and clenching tight under the skin in suppressed anger, teeth grinding invisibly behind closed lips. Anatomical 2D precision. Strictly no 3D mesh.",
        "Extreme close-up on enormous anime eyes reflecting a hand-drawn starry night sky, pupils wide, delicate white specular reflection dots painted on the cornea, lips parted in wonder. Masterclass hand-drawn beauty. Strictly no CGI lens flare.",
        "Side profile close-up of a youth looking down at the ground, long eyelashes casting delicate ink shadows over the cheek, a soft lonely breeze fluttering their bangs. Muted watercolor background, crisp 2D line art. Strictly no 3D.",
        "Close-up on a guilty student's eyes darting rapidly from left to right and back to center without turning the head, beads of nervous sweat forming along the hairline. Comedic-dramatic anime pacing. Strictly no digital warp.",
        "Close-up of a commander barking orders in silence, jaw dropping wide and snapping shut rhythmically with powerful stylized mouth movements, eyes fiery and commanding. Bold 2D ink lines. Strictly no 3D human lip sync.",
        "Close-up of an anime heroine puffing both cheeks out like a chipmunk, lips jutting forward in an exaggerated cute pout of frustration, eyebrows furrowed in a flat line. Traditional 2D comedic tropes. Strictly no 3D rendering.",
        "Medium close-up of a fallen hero tilting their head back slightly, eyes closed in self-mockery, shoulders shaking with a silent, bitter laugh before their face hardens into stone. Expressive 2D acting. Strictly no CGI.",
        "Extreme close-up on cloudy, unfocused anime eyes suddenly snapping into sharp clarity, pupils dilating, specular highlights reappearing in an instant as consciousness returns. Crisp 2D transition. Strictly no 3D bloom.",
        "Close-up on a battered fighter wiping blood from their eye, eye locking back onto the opponent with unbreakable iron will, teeth bared in a defiant battle snarl. High-octane MAPPA-style Shonen line weight. Strictly no CGI.",
    ],
    # A character-less ambient scene is fine here as long as the anime style/
    # coloring is good (user explicitly walked back the earlier "character in
    # every clip" rule) -- mix of character-present and character-less shots.
    "secondary-motion": [
        "Full shot of an occult detective standing completely still on a cathedral roof during a storm, the long tails of their heavy black wool trench coat whipping violently in complex hand-drawn cloth waves. Masterclass 2D cloth animation. Strictly no 3D cloth physics, no CGI mesh.",
        "Low-angle static shot of an alleyway brick wall following an explosion, jagged chunks of 2D hand-drawn concrete tumbling down, bouncing off asphalt with white stylized impact dust rings. Traditional cel animation. Strictly no 3D particle systems.",
        "Extreme macro close-up of a polished black steel blade held horizontally, heavy raindrops falling into the frame, striking the cold metal edge, and exploding into hand-drawn flat splash droplets. Pure 2D sakuga liquid art. Strictly no 3D fluid sim.",
        "Static shot of an ancient paper charm burning from bottom to top, parchment turning to black curling ash, small hand-drawn red ember specks floating lazily upward into darkness. Classic anime hand-drawn FX. Strictly no 3D fire sim.",
        "Low-angle exterior shot looking up at wooden telephone poles, three thick black power cables hanging in parabolic arcs, undulating slowly up and down with heavy harmonic inertia after an earthquake. Clean ink outlines. Strictly no 3D rendering.",
        "Static interior shot of an open classroom window, translucent white sheer curtains billowing deep into the room in an afternoon breeze, sunlight casting moving hand-drawn folds onto wooden floorboards. Kyoto Animation style serenity. Strictly no CGI.",
        "Medium tracking shot of an anime girl standing on an overpass, her long knitted crimson scarf caught in a high wind, streaming out behind her in fluid, serpentine cloth in-betweens. Sharp cel outlines. Strictly no 3D simulation.",
        "Close-up of a dark ceramic matcha cup sitting on a tatami mat, delicate tendrils of stylized 2D white steam rising in looping serpentine shapes, dissolving smoothly into the air. Traditional painted background, crisp 2D art. Strictly no CGI smoke.",
        "Close-up of a fragile glass wind chime hanging from a rustic wooden porch, its central paper strip fluttering wildly in an incoming storm, causing the glass rod to clatter and swing in an arc. Hand-drawn clean physics. Strictly no 3D models.",
        "Top-down view of a clear rainwater puddle on asphalt, a single circular raindrop hitting the center, sending out three perfect concentric hand-drawn line ripples that reflect dark clouds. Clean flat cel graphics. Strictly no 3D liquid engines.",
        "Medium shot from behind an anime swordswoman with hip-length black hair who has just dashed to a halt, the mass of hair overshooting forward before whipping back and settling naturally down her back. Snappy secondary follow-through. Strictly no 3D hair cards.",
        "Static shot of a heavy iron chain hanging from a factory crane hook, swinging in a wide, slow pendulum arc across dusty light shafts, individual chain links drawn with sharp ink outlines. Authentic 2D mechanical weight. Strictly no CGI.",
        "Locked camera shot of a suburban residential corner, dozens of pink hand-drawn cherry blossom petals sweeping across the frame in sudden turbulent wind gusts, tumbling realistically in 2D space. Classic anime spring aesthetic. Strictly no 3D particles.",
        "Medium shot of a vintage green iron streetlamp glowing in thick fog, the bulb filament buzzing with subtle hand-drawn light vibration, a small paper flyer taped to the pole fluttering vigorously in the breeze. Gritty 2D cel art. Strictly no 3D bloom.",
        "Close-up of an anime student's canvas backpack resting on a desk, two cute acrylic character keychains swinging back and forth with decaying kinetic momentum after being set down. Sharp line work. Strictly no 3D physics.",
        "Top-down macro shot of water draining out of a clean porcelain sink, forming a smooth rotating 2D whirlpool vortex with stylized white foam line accents swirling inward. Clean graphic anime water animation. Strictly no CGI fluid.",
        "Interior shot of a commuter train window, an off-white vinyl pull-down blind rattling and vibrating rapidly against the metal frame due to the high-speed track motion. Flat shading, authentic transit details. Strictly no 3D warp.",
        "Close-up shot of a dark garden pond surface, a single crimson maple leaf drifting down from top of frame, gently kissing the water surface, and creating a single expanding circular 2D wave ring. Masterful 2D nature aesthetic. Strictly no 3D simulation.",
    ],
}


def build_queue():
    # Prompt pools are now large and specific enough (52/32/24/18 unique
    # prompts) to roughly track the site's real targets (locomotion 50,
    # gestures 30, expressions 40, secondary-motion 20) on their own --
    # expressions is the one category whose pool is still proportionally
    # behind its target, so it gets an extra pass.
    weights = {"expressions": 2, "secondary-motion": 1, "gestures": 1, "locomotion": 1}
    queue = []
    for cat, w in weights.items():
        for _ in range(w):
            queue.extend((cat, p) for p in PROMPTS[cat])
    random.shuffle(queue)
    return queue


def caption_from_prompt(base_prompt: str) -> str:
    """Turn a generation prompt into a clean training caption. The newer
    prompts end with "Strictly no 3D CGI / no motion blur / ..." sentences --
    those are negative instructions for the generator, not a description of
    what's actually in the clip, so they're stripped rather than uploaded as
    caption metadata."""
    sentences = re.split(r"(?<=[.!?])\s+", base_prompt.strip())
    kept = [s for s in sentences if s and not s.lower().startswith("strictly")]
    caption = " ".join(kept).strip() or base_prompt.strip()
    if not caption.endswith("."):
        caption += "."
    return caption[0].upper() + caption[1:]


def upload_clip(path: Path, category: str, caption: str) -> dict:
    sign_res = requests.post(f"{SITE}/api/sign", json={"tag": category, "caption": caption}, timeout=30)
    sign_res.raise_for_status()
    sd = sign_res.json()

    with open(path, "rb") as f:
        files = {"file": f}
        data = {
            "api_key": sd["apiKey"],
            "timestamp": sd["timestamp"],
            "signature": sd["signature"],
            "folder": sd["folder"],
            "tags": sd["tags"],
        }
        if sd.get("context"):
            data["context"] = sd["context"]
        upload_res = requests.post(
            f"https://api.cloudinary.com/v1_1/{sd['cloudName']}/video/upload",
            data=data, files=files, timeout=180,
        )
    upload_res.raise_for_status()
    return upload_res.json()


def frame_looks_blank(frames) -> bool:
    """CogVideoX-2b in fp16 has a known failure mode on older GPUs (e.g.
    Kaggle's T4) where the VAE decode produces near-uniform white/gray
    frames instead of an actual image -- silent garbage, not an exception.
    Catch it by checking pixel variance on a couple of sample frames."""
    for idx in (len(frames) // 2, len(frames) - 1):
        arr = np.asarray(frames[idx], dtype=np.float32)
        if arr.std() < 8.0:  # a real rendered frame has far more spread than this
            return True
    return False


# User's hard requirement: "AI colors hona chahiye hi chahiye, nahi hoga toh
# video upload nahi hoga" -- an uncolored/sketch-only clip must NEVER reach the
# site, full stop. frame_looks_blank() alone does not catch this: flat gray
# lineart on white paper has plenty of pixel variance (so it passes the blank
# check) while still having almost no actual color in it.
UNCOLORED_SATURATION_THRESHOLD = 18.0  # 0-255 scale avg; real colored anime frames sit far above this
COLORED_PIXEL_SATURATION = 40.0  # a pixel above this is "clearly colored", not just antialiasing noise
MIN_COLORED_PIXEL_FRACTION = 0.02  # at least 2% of the frame must actually be colored, not just noise


def frame_looks_uncolored(frames) -> bool:
    """Reject grayscale/sketch-only output. Checks HSV-style saturation
    (max(R,G,B) - min(R,G,B)) both as a frame average AND as a coverage
    fraction (percent of pixels that are clearly colored), over several
    sample frames. Two checks instead of one average alone: a pure
    black-and-white/pencil-sketch frame has ~0 on both, but relying on the
    average by itself could be fooled by widespread faint compression/model
    noise that nudges the mean up without any real colored area existing."""
    sample_idxs = sorted({0, len(frames) // 4, len(frames) // 2, (3 * len(frames)) // 4, len(frames) - 1})
    frame_sats = []
    frame_colored_fractions = []
    for idx in sample_idxs:
        arr = np.asarray(frames[idx], dtype=np.float32)
        cmax = arr.max(axis=-1)
        cmin = arr.min(axis=-1)
        saturation = np.where(cmax > 1.0, (cmax - cmin) / cmax * 255.0, 0.0)
        frame_sats.append(float(saturation.mean()))
        frame_colored_fractions.append(float((saturation > COLORED_PIXEL_SATURATION).mean()))
    avg_saturation = sum(frame_sats) / len(frame_sats)
    avg_colored_fraction = sum(frame_colored_fractions) / len(frame_colored_fractions)
    # AND, not OR: a colored character shot on a plain/white background has a
    # LOW average (background pixels drag it down) but a healthy colored-pixel
    # fraction -- that must pass. Only reject when neither signal shows color.
    return avg_saturation < UNCOLORED_SATURATION_THRESHOLD and avg_colored_fraction < MIN_COLORED_PIXEL_FRACTION


class StepTimeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise StepTimeout()


def with_timeout(seconds, fn, *args, **kwargs):
    """Runs fn with a hard wall-clock deadline. Kaggle gives no way to peek
    at a running kernel's logs, so a hang here (a slow/throttled model
    download, a stuck CUDA call) previously looked identical to normal
    progress from the outside -- multiple runs sat RUNNING for over an hour
    with zero output and no crash before anyone could tell something was
    actually stuck. This turns a silent hang into a clear, loud failure that
    shows up in the kernel's logs once it exits."""
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(seconds)
    try:
        return fn(*args, **kwargs)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def load_cogvideox():
    print(f"Loading {COGVIDEOX_MODEL_ID}...", flush=True)
    # 10-minute hard deadline: a slow/throttled HuggingFace download (this
    # script sends unauthenticated requests -- no HF_TOKEN set) has caused
    # silent multi-hour hangs here before. A real load finishes in a couple
    # of minutes, so 600s is already a generous margin, not a tight cutoff.
    # Whole pipeline in bfloat16 -- no mixed dtypes, same memory footprint as
    # fp16 (float32 OOM'd on a 14.5GB T4: "Tried to allocate 20.00 MiB ...
    # 14.54 GiB memory in use", confirmed via the manual Gradio tool).
    # bf16 has fp32's wide exponent range (unlike fp16, which is what
    # actually overflows to NaN/white during VAE decode on T4-class GPUs)
    # while fitting in the same memory fp16 does, and it's a single dtype
    # throughout so it doesn't hit the fp16/fp32-VAE mismatch that crashed
    # under enable_model_cpu_offload() either. This was very likely the
    # real cause behind a chunk of this script's own silent "failed" count
    # all along, since its per-clip except-block only ever logged to output
    # nobody could see live.
    pipe = with_timeout(
        600, CogVideoXPipeline.from_pretrained, COGVIDEOX_MODEL_ID, torch_dtype=torch.bfloat16
    )
    print("Model weights loaded.", flush=True)
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    try:
        # Directly targets the attention memory spike (a single 70GB alloc
        # attempt was observed) by processing attention in chunks instead of
        # materializing the full attention matrix at once.
        pipe.enable_attention_slicing()
    except AttributeError:
        pass
    return pipe


def main():
    # Wan2.2-TI2V-5B was tried here earlier (user-requested, better quality
    # than CogVideoX-2b if it fits). It's dropped now: on a real run it got
    # OOM-killed by the OS ("Killed", not a catchable CUDA exception) right
    # after the diffusion loop finished, during VAE decode of the full
    # 49-frame clip -- the cheap 9-frame smoke test didn't stress the decode
    # step enough to catch this, so it passed while the real generation
    # silently ate ~40 minutes of quota for zero output. CogVideoX-2b is
    # also literally this project's actual target base model (see
    # CLAUDE.md), so there's no quality tradeoff being made by dropping Wan.
    pipe = load_cogvideox()
    model_name = "cogvideox-2b"
    print(f"Generating with: {model_name}")

    queue = build_queue()
    print(f"Queue built: {len(queue)} clips planned across {len(PROMPTS)} categories")

    done = 0
    failed = 0
    blanked = 0
    uncolored = 0
    timed_out = 0
    start = time.time()

    for i, (category, base_prompt) in enumerate(queue):
        elapsed = time.time() - start
        if elapsed > MAX_SECONDS:
            print(f"\nApproaching the session time limit ({elapsed/3600:.1f}h elapsed), stopping here.")
            break

        prompt = base_prompt + STYLE_SUFFIX
        print(f"\n[{i + 1}/{len(queue)}] category={category} elapsed={elapsed/60:.1f}min")
        print(f"  prompt: {prompt}")

        out_path = OUT_DIR / f"clip_{i:04d}.mp4"
        try:
            frames = None
            reject_reason = None
            for attempt in range(1, MAX_RETRIES_PER_PROMPT + 2):
                seed = random.randint(0, 2**31 - 1)
                generator = torch.Generator(device="cuda").manual_seed(seed)

                def _generate():
                    return pipe(
                        prompt=prompt,
                        negative_prompt=NEGATIVE_PROMPT,
                        num_frames=NUM_FRAMES,
                        height=HEIGHT,
                        width=WIDTH,
                        num_inference_steps=NUM_INFERENCE_STEPS,
                        guidance_scale=GUIDANCE_SCALE,
                        output_type="pil",
                        generator=generator,
                    ).frames[0]

                try:
                    # 15-minute hard deadline per attempt: a normal generation
                    # takes ~10 min at 30 steps, so this only fires on a real
                    # hang (the failure mode that previously caused multi-hour
                    # silent stalls with no crash and no visible progress).
                    candidate = with_timeout(900, _generate)
                except StepTimeout:
                    reject_reason = "timed out (>15min, likely a stuck/hung call)"
                    timed_out += 1
                    more_left = attempt <= MAX_RETRIES_PER_PROMPT
                    print(f"  attempt {attempt} {reject_reason}, "
                          f"{'retrying...' if more_left else 'giving up on this prompt.'}", flush=True)
                    continue

                if frame_looks_blank(candidate):
                    reject_reason = "blank"
                elif frame_looks_uncolored(candidate):
                    # Hard mandatory gate -- never upload uncolored/sketch output.
                    reject_reason = "uncolored/sketch (no real color)"
                else:
                    frames = candidate
                    break

                more_left = attempt <= MAX_RETRIES_PER_PROMPT
                print(f"  attempt {attempt} rejected ({reject_reason}, seed={seed}), "
                      f"{'retrying...' if more_left else 'giving up on this prompt.'}", flush=True)

            if frames is None:
                if reject_reason == "blank":
                    blanked += 1
                elif reject_reason and reject_reason.startswith("timed out"):
                    pass  # already counted in the except block above
                else:
                    uncolored += 1
                continue

            export_to_video(frames, str(out_path), fps=FPS)

            caption = caption_from_prompt(base_prompt)
            result = upload_clip(out_path, category, caption)
            print(f"  uploaded: {result.get('public_id')}")
            done += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            failed += 1
        finally:
            if out_path.exists():
                out_path.unlink()
            torch.cuda.empty_cache()

    total_min = (time.time() - start) / 60
    print(
        f"\n=== DONE: {done} uploaded, {blanked} blanked (skipped), "
        f"{uncolored} uncolored/sketch (skipped, never uploaded), "
        f"{timed_out} timed out (>15min hangs), {failed} failed, "
        f"{total_min:.1f} min elapsed ==="
    )


if __name__ == "__main__":
    main()
