"""Per-frame Kontext pose prompts for animation presets.

Each preset is a list of Kontext edit instructions, one per frame. Kontext
prompts follow the "Change X. Keep Y exactly the same." format so character
identity is preserved while only the pose varies.

The "Keep..." clause is appended to every frame automatically by
`tools.animate.animate_sprite` to avoid repeating boilerplate. These prompts
define only the pose delta.

Frame counts per preset:
    idle:   4 frames (subtle breathing)
    walk:   8 frames (standard 8-frame walk cycle)
    run:    8 frames (faster, more extreme poses)
    attack: 6 frames (windup → swing → recoil)
    jump:   6 frames (crouch → launch → airborne → peak → fall → land)
    cast:   6 frames (prep → raise → release → return)
    death:  6 frames (hit → stagger → falling → down)
"""

from __future__ import annotations

# Appended to every preset frame so callers don't have to repeat it
_PRESERVATION_SUFFIX = (
    "Keep the exact same character, clothing, armor, weapon, "
    "color palette, and art style unchanged. White background."
)


def _with_suffix(prompts: list[str]) -> list[str]:
    return [f"{p} {_PRESERVATION_SUFFIX}" for p in prompts]


ANIMATION_PRESETS: dict[str, list[str]] = {
    "idle": _with_suffix([
        "Change the pose to a neutral standing idle, facing forward, weapon relaxed at the side.",
        "Change the pose to a subtle breath intake, chest slightly raised, weight shifted to the right foot.",
        "Change the pose to a subtle breath exhale, chest lowered, weight shifted to the left foot.",
        "Change the pose to a neutral standing idle, facing forward, head tilted slightly to the side.",
    ]),
    "walk": _with_suffix([
        "Change the pose to the left leg forward planted flat on the ground, right leg behind, mid-stride walk, side view.",
        "Change the pose to the left leg fully forward, right leg lifted off the ground behind, arms swinging naturally, side view.",
        "Change the pose to both legs near together passing mid-step, weight centered, side view.",
        "Change the pose to the right leg forward planted flat on the ground, left leg behind, mid-stride walk, side view.",
        "Change the pose to the right leg fully forward, left leg lifted off the ground behind, arms swinging naturally, side view.",
        "Change the pose to both legs near together passing mid-step, weight centered, side view.",
        "Change the pose to the left leg forward planted flat on the ground, right leg behind, mid-stride walk, side view.",
        "Change the pose to the left leg fully forward, right leg lifted off the ground behind, arms swinging naturally, side view.",
    ]),
    "run": _with_suffix([
        "Change the pose to a full sprint, left leg extended forward, right leg back in full extension, leaning forward, arms pumping, side view.",
        "Change the pose to a full sprint, knees high, both feet off the ground in mid-bound, side view.",
        "Change the pose to a full sprint, right leg extended forward, left leg back in full extension, leaning forward, side view.",
        "Change the pose to a full sprint, left leg planting on the ground, right leg driving forward, side view.",
        "Change the pose to a full sprint, right leg extended forward, left leg back in full extension, leaning forward, side view.",
        "Change the pose to a full sprint, knees high, both feet off the ground in mid-bound, side view.",
        "Change the pose to a full sprint, left leg extended forward, right leg back in full extension, leaning forward, side view.",
        "Change the pose to a full sprint, right leg planting on the ground, left leg driving forward, side view.",
    ]),
    "attack": _with_suffix([
        "Change the pose to the character winding up a weapon swing, weapon pulled back behind the body, shoulders rotated.",
        "Change the pose to the character beginning the weapon swing, weapon raised overhead, arms extending forward.",
        "Change the pose to the character mid-swing, weapon horizontal at shoulder height, body leaning into the swing.",
        "Change the pose to the character completing the swing, weapon fully extended forward at hip height, body weight on the front foot.",
        "Change the pose to the character recovering from the swing, weapon lowered across the body, stepping back.",
        "Change the pose to the character returning to a ready stance, weapon held in front, knees slightly bent.",
    ]),
    "jump": _with_suffix([
        "Change the pose to the character crouched, knees deeply bent, preparing to jump upward.",
        "Change the pose to the character launching upward, legs beginning to extend, arms swinging up.",
        "Change the pose to the character in the air, legs fully extended downward, arms raised overhead.",
        "Change the pose to the character at peak height in the air, body slightly curled, knees tucked.",
        "Change the pose to the character falling, legs beginning to extend for landing, arms out for balance.",
        "Change the pose to the character landing, knees bent absorbing impact, weight low to the ground.",
    ]),
    "cast": _with_suffix([
        "Change the pose to the character standing ready, arms lowered, preparing to cast a spell.",
        "Change the pose to the character raising one hand forward, fingers spread, channeling energy.",
        "Change the pose to the character with both hands raised forward, glowing energy forming in front, focused expression.",
        "Change the pose to the character releasing the spell, arms thrust forward, energy burst leaving the hands.",
        "Change the pose to the character with arms lowered after the cast, slight recoil stance.",
        "Change the pose to the character returning to a neutral standing stance, arms at sides.",
    ]),
    "death": _with_suffix([
        "Change the pose to the character taking a hit, body jerking backward, weapon dropping.",
        "Change the pose to the character staggering, knees beginning to buckle, one arm reaching out for balance.",
        "Change the pose to the character falling to one knee, body hunched forward, head down.",
        "Change the pose to the character falling backward, arms flung out, off balance.",
        "Change the pose to the character lying on the ground, fallen, body sprawled.",
        "Change the pose to the character lying still on the ground, defeated, weapon on the ground beside them.",
    ]),
}


def preset_prompts(animation: str) -> list[str] | None:
    """Return the list of per-frame pose prompts for a preset, or None."""
    return ANIMATION_PRESETS.get(animation)


def available_presets() -> list[str]:
    return sorted(ANIMATION_PRESETS.keys())
