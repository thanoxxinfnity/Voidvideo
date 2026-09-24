#!/usr/bin/env python3
"""
Runs INSIDE a Kaggle GPU kernel. Launches a Gradio app (public share link)
backed by CogVideoX-2b so a human can generate one anime clip at a time,
watch it immediately, and decide whether it's good enough -- no automated
blank/color gate, no retry loop, no auto-upload.

Why this exists: the fully-automated version (kaggle_clip_generator_entrypoint.py)
kept silently hanging for 1-2+ hours with zero visible progress and no crash,
because Kaggle's API gives no way to see a running kernel's logs, and even a
signal.alarm()-based timeout watchdog didn't reliably interrupt whatever was
actually stuck. A human watching a live UI solves the visibility problem
directly instead of trying to engineer around it blind.

Usage: open the kernel's page on kaggle.com after it starts running -- Kaggle
streams stdout live in its own web UI (unlike its API). The Gradio public
URL (https://xxxxx.gradio.live) is printed there within a minute or so of
the model finishing loading. Open that URL, pick or type a prompt, hit
Generate, wait ~8-10 minutes, watch the result, download it if it's good,
and upload it to the site yourself via the existing Smart Upload page.

Pushed to Kaggle by scripts/push_gradio_generator_to_kaggle.py.
"""
import subprocess
import sys


def pip_install():
    subprocess.run(
        [
            sys.executable, "-m", "pip", "install", "-q", "-U",
            "diffusers>=0.35.0", "transformers>=4.44", "accelerate>=0.33",
            "imageio[ffmpeg]", "sentencepiece", "gradio",
        ],
        check=True,
    )


pip_install()

import torch  # noqa: E402
from diffusers import CogVideoXPipeline, CogVideoXImageToVideoPipeline  # noqa: E402
from diffusers.utils import export_to_video  # noqa: E402
import gradio as gr  # noqa: E402

# Native generation stays at the crash-safe 8fps below -- CogVideoX itself
# was never re-tuned to output more frames per second (that's what triggered
# the original OOMs). Instead the exported clip is upsampled to this fps
# afterwards with ffmpeg motion interpolation, a pure CPU/ffmpeg post-process
# that doesn't touch GPU memory at all.
OUTPUT_FPS = 24


def interpolate_to_output_fps(src_path: str) -> str:
    """Upsample an exported clip from its native FPS to OUTPUT_FPS with
    ffmpeg's motion-compensated frame interpolation. Runs after the GPU has
    already been released for this generation, so it can't contribute to an
    OOM. Falls back to the original (native-fps) file if ffmpeg is missing
    or the filter fails, rather than losing the result."""
    dst_path = src_path.replace(".mp4", f"_{OUTPUT_FPS}fps.mp4")
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", src_path,
                "-filter:v", f"minterpolate=fps={OUTPUT_FPS}:mi_mode=mci:mc_mode=aobmc:vsbmc=1",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
                dst_path,
            ],
            check=True, capture_output=True, timeout=180,
        )
        return dst_path
    except Exception as e:
        print(f"24fps interpolation failed, returning native-fps clip instead: {e}", flush=True)
        return src_path

COGVIDEOX_MODEL_ID = "THUDM/CogVideoX-2b"
# Cut from 49 (the training config's frame count) after three straight OOM
# crashes on this 14.5GB T4 -- fewer frames means less resident at once
# throughout the whole pipeline, not just in the attention op. ~3.1s clips
# at this fps instead of ~6.1s; shorter but actually generates.
NUM_FRAMES = 25
FPS = 8
# Reduced from 480x720: the real OOM wasn't about model weight dtype at all
# (bf16 correctly halved weight memory to ~6GB) -- it was CogVideoX's
# spatio-temporal attention trying to allocate a single 70GB tensor, which
# scales with total tokens (frames x spatial patches) and has nothing to do
# with dtype. Smaller resolution directly shrinks that token count.
HEIGHT = 256
WIDTH = 384

# I2V uses the 5B model (2.5x the 2B T2V model's params) -- loaded lazily
# (only on first Image-to-Video generate call) so a T2V-only session never
# pays its ~20GB download/load cost. Kept even smaller than the T2V settings
# above since the bigger transformer leaves less VRAM headroom on the same
# 14.5GB T4, even with identical sequential-offload/slicing/tiling applied.
COGVIDEOX_I2V_MODEL_ID = "THUDM/CogVideoX-5b-I2V"
I2V_NUM_FRAMES = 17
I2V_FPS = 8
I2V_HEIGHT = 208
I2V_WIDTH = 304

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

# Same prompt library as kaggle_clip_generator_entrypoint.py -- kept here too
# since a Kaggle script kernel is a single self-contained file. Used to
# populate the dropdown of ready-made prompts; free typing also works.
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

CHOICES = []
for _cat, _prompts in PROMPTS.items():
    for _p in _prompts:
        label = f"[{_cat}] {_p[:80]}..."
        CHOICES.append((label, _p))

print(f"Loading {COGVIDEOX_MODEL_ID}...", flush=True)
# Whole pipeline in bfloat16 -- no mixed dtypes, same memory footprint as
# fp16 (float32 just OOM'd on this 14.5GB T4: "Tried to allocate 20.00 MiB
# ... 14.54 GiB memory in use"). bf16 has the same wide exponent range as
# fp32 (unlike fp16), which is what actually avoids the VAE decode
# overflowing to NaN/white on T4-class GPUs -- fp16 has a narrow exponent
# range and overflows there, float32 fixed that but doesn't fit, and mixing
# fp16-transformer with fp32-VAE crashed under enable_model_cpu_offload().
# bf16 throughout sidesteps all three failure modes at once. T4 lacks
# native bf16 tensor cores so this may compute a bit slower than fp16, but
# that's a fine tradeoff for actually fitting and not crashing.
def _harden_pipe(p):
    """Apply the same low-VRAM tricks (bf16 already set at from_pretrained)
    to any CogVideoX pipeline instance -- shared between T2V and I2V so the
    I2V path gets the identical crash-safety as the battle-tested T2V one."""
    p.enable_sequential_cpu_offload()
    p.vae.enable_slicing()
    p.vae.enable_tiling()
    try:
        p.enable_attention_slicing()
    except AttributeError:
        pass
    return p


print(f"Loading {COGVIDEOX_MODEL_ID}...", flush=True)
# Whole pipeline in bfloat16 -- no mixed dtypes, same memory footprint as
# fp16 (float32 just OOM'd on this 14.5GB T4: "Tried to allocate 20.00 MiB
# ... 14.54 GiB memory in use"). bf16 has the same wide exponent range as
# fp32 (unlike fp16), which is what actually avoids the VAE decode
# overflowing to NaN/white on T4-class GPUs -- fp16 has a narrow exponent
# range and overflows there, float32 fixed that but doesn't fit, and mixing
# fp16-transformer with fp32-VAE crashed under enable_model_cpu_offload().
# bf16 throughout sidesteps all three failure modes at once. T4 lacks
# native bf16 tensor cores so this may compute a bit slower than fp16, but
# that's a fine tradeoff for actually fitting and not crashing.
t2v_pipe = _harden_pipe(CogVideoXPipeline.from_pretrained(COGVIDEOX_MODEL_ID, torch_dtype=torch.bfloat16))
print("T2V model loaded. Launching Gradio...", flush=True)

# I2V (5B, ~2.5x the T2V model) is NOT loaded here -- only on first actual
# use, from inside generate() below. Eagerly loading both up front would
# double the startup download/VRAM-mapping time for sessions that only ever
# use one mode.
_i2v_pipe = None


def get_i2v_pipe():
    global _i2v_pipe
    if _i2v_pipe is None:
        print(f"First Image-to-Video request -- loading {COGVIDEOX_I2V_MODEL_ID} (larger model, may take a few minutes)...", flush=True)
        _i2v_pipe = _harden_pipe(
            CogVideoXImageToVideoPipeline.from_pretrained(COGVIDEOX_I2V_MODEL_ID, torch_dtype=torch.bfloat16)
        )
        print("I2V model loaded.", flush=True)
    return _i2v_pipe


import math  # noqa: E402
import time  # noqa: E402

# CogVideoX has a fixed trained frame window -- there's no VRAM/time setting
# that makes one generation call produce more than a few seconds. A longer
# requested duration is instead built by chaining multiple short segments:
# generate one, feed its LAST FRAME into the I2V pipeline as the next
# segment's starting image (so motion continues instead of hard-cutting to
# an unrelated scene), repeat until the target duration is covered, then
# concatenate. This is slow (each segment is its own full generation) but
# that slowness is exactly the "low VRAM, take longer, don't crash"
# trade-off asked for -- no segment is any bigger than the already-hardened
# single-clip settings above.
T2V_SEGMENT_SECONDS = NUM_FRAMES / FPS
I2V_SEGMENT_SECONDS = I2V_NUM_FRAMES / I2V_FPS
# Rough wall-clock estimates from this repo's own prior runs on a Kaggle T4,
# shown to set expectations before a long chained run starts -- not a
# guarantee, actual time varies with steps/queue load.
EST_MINUTES_PER_T2V_SEGMENT = 9
EST_MINUTES_PER_I2V_SEGMENT = 14


def plan_segments(mode: str, duration_seconds: float):
    """Returns (total_segments, estimated_minutes) for a target duration.
    Segment 0 uses `mode`; every segment after that is always I2V
    (continuing from the previous segment's last frame), regardless of
    starting mode."""
    first_seg_seconds = T2V_SEGMENT_SECONDS if mode == "Text-to-Video" else I2V_SEGMENT_SECONDS
    remaining = max(0.0, duration_seconds - first_seg_seconds)
    extra_segments = math.ceil(remaining / I2V_SEGMENT_SECONDS) if remaining > 0 else 0
    total_segments = 1 + extra_segments
    est_minutes = (
        (EST_MINUTES_PER_T2V_SEGMENT if mode == "Text-to-Video" else EST_MINUTES_PER_I2V_SEGMENT)
        + extra_segments * EST_MINUTES_PER_I2V_SEGMENT
    )
    return total_segments, est_minutes


def concat_segments(paths: list, seed: int) -> str:
    if len(paths) == 1:
        return paths[0]
    list_path = f"/kaggle/working/concat_list_{seed}.txt"
    with open(list_path, "w") as f:
        for p in paths:
            f.write(f"file '{p}'\n")
    out_path = f"/kaggle/working/final_seed{seed}_{OUTPUT_FPS}fps.mp4"
    # All segments already share the same codec/pix_fmt/fps (both came out of
    # interpolate_to_output_fps), so a stream-copy concat is safe and instant
    # -- no re-encode needed.
    subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_path, "-c", "copy", out_path],
        check=True, capture_output=True, timeout=120,
    )
    return out_path


def generate(mode, prompt_text, image, steps, seed, duration_seconds):
    if mode == "Image-to-Video" and image is None:
        yield None, "ERROR: Image-to-Video ke liye ek image upload karo pehle."
        return
    if not prompt_text or not prompt_text.strip():
        yield None, "ERROR: Prompt khali hai -- kuch likho ya dropdown se pick karo."
        return

    full_prompt = prompt_text.strip() + STYLE_SUFFIX
    seed = int(seed) if seed is not None and seed >= 0 else torch.randint(0, 2**31 - 1, (1,)).item()
    duration_seconds = float(duration_seconds or T2V_SEGMENT_SECONDS)

    total_segments, est_minutes = plan_segments(mode, duration_seconds)
    yield None, (
        f"Planning {total_segments} chained segment(s) to cover ~{duration_seconds:.0f}s "
        f"of video. Estimated time: ~{est_minutes} min (rough estimate, varies). Starting..."
    )

    segment_paths = []
    current_image = image
    t_start = time.time()
    try:
        for i in range(total_segments):
            seg_generator = torch.Generator(device="cuda").manual_seed(seed + i)
            if i == 0 and mode == "Text-to-Video":
                frames = t2v_pipe(
                    prompt=full_prompt,
                    negative_prompt=NEGATIVE_PROMPT,
                    num_frames=NUM_FRAMES,
                    height=HEIGHT,
                    width=WIDTH,
                    num_inference_steps=int(steps),
                    guidance_scale=6.0,
                    output_type="pil",
                    generator=seg_generator,
                ).frames[0]
                native_fps = FPS
            else:
                pipe = get_i2v_pipe()
                frames = pipe(
                    prompt=full_prompt,
                    image=current_image,
                    negative_prompt=NEGATIVE_PROMPT,
                    num_frames=I2V_NUM_FRAMES,
                    height=I2V_HEIGHT,
                    width=I2V_WIDTH,
                    num_inference_steps=int(steps),
                    guidance_scale=6.0,
                    output_type="pil",
                    generator=seg_generator,
                ).frames[0]
                native_fps = I2V_FPS

            raw_seg_path = f"/kaggle/working/seg{i}_seed{seed}_native.mp4"
            export_to_video(frames, raw_seg_path, fps=native_fps)
            segment_paths.append(interpolate_to_output_fps(raw_seg_path))
            current_image = frames[-1]  # feeds the next segment's continuation
            torch.cuda.empty_cache()

            elapsed_min = (time.time() - t_start) / 60
            more = "Concatenating..." if i == total_segments - 1 else "Generating next segment..."
            yield None, f"Segment {i + 1}/{total_segments} done ({elapsed_min:.1f} min elapsed). {more}"

        final_path = concat_segments(segment_paths, seed)
        total_min = (time.time() - t_start) / 60
        yield final_path, (
            f"Done: {total_segments} segment(s), ~{duration_seconds:.0f}s target, "
            f"{total_min:.1f} min actual, {OUTPUT_FPS}fps. Seed base: {seed} "
            "(same seed+prompt/image = same result, change seed for variety)."
        )
    except Exception as e:
        # Yielded directly (not raised) so both the UI and any API caller
        # see the real reason instead of Gradio's generic hidden AppError.
        import traceback
        tb = traceback.format_exc()
        print(f"GENERATE FAILED: {tb}", flush=True)
        yield None, f"ERROR: {type(e).__name__}: {e}"
    finally:
        torch.cuda.empty_cache()


with gr.Blocks(title="VoidVideo Clip Studio") as demo:
    gr.Markdown(
        "# VoidVideo Clip Studio (manual)\n"
        "Pick Text-to-Video or Image-to-Video, pick a ready-made prompt or "
        "type your own, hit **Generate**, wait (T2V ~8-10 min; I2V is on a "
        "bigger 5B model and slower, especially the first run while it "
        "loads). Output is generated on a safe low-VRAM native fps then "
        "upsampled to 24fps automatically. If it's good, download it and "
        "upload it yourself via the site's Smart Upload page, tagged with "
        "the right category. Nothing here auto-uploads -- you decide what's "
        "good enough."
    )
    with gr.Row():
        with gr.Column():
            mode = gr.Radio(
                ["Text-to-Video", "Image-to-Video"], value="Text-to-Video", label="Mode"
            )
            image_in = gr.Image(label="Source image (Image-to-Video only)", type="pil", visible=False)
            mode.change(
                fn=lambda m: gr.update(visible=(m == "Image-to-Video")),
                inputs=mode, outputs=image_in,
            )
            dropdown = gr.Dropdown(
                choices=CHOICES, label="Ready-made prompt (pick one, or clear and type your own below)"
            )
            prompt_box = gr.Textbox(
                label="Prompt", lines=4, placeholder="A hand-drawn anime boy walking down a rainy street..."
            )
            dropdown.change(fn=lambda x: x, inputs=dropdown, outputs=prompt_box)
            steps_slider = gr.Slider(10, 50, value=30, step=1, label="Inference steps (30 = good default, higher = slower but sometimes cleaner)")
            duration_slider = gr.Slider(
                3, 60, value=20, step=1,
                label="Target duration (seconds) -- longer than ~3-6s chains multiple "
                      "segments together (each continuing from the last one's final frame), "
                      "which takes proportionally longer but stays in the same low-VRAM settings",
            )
            seed_box = gr.Number(value=-1, label="Seed (-1 = random each time)")
            btn = gr.Button("Generate", variant="primary")
        with gr.Column():
            video_out = gr.Video(label="Result (right-click / use the download icon to save)")
            status = gr.Textbox(label="Status (updates per segment on longer runs)", interactive=False)

    btn.click(
        fn=generate,
        inputs=[mode, prompt_box, image_in, steps_slider, seed_box, duration_slider],
        outputs=[video_out, status],
    )

demo.queue().launch(share=True, show_error=True)
