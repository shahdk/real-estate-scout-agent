"""Image analysis using Gemma 4 (via Ollama) for property condition inspection.

This is the most expensive step in the pipeline — it runs LAST, only after
all preference-based filtering has narrowed the listing set.
"""

import base64
import json
import os
import re
import threading
from pathlib import Path

import httpx

# ---------------------------------------------------------------------------
# Ollama configuration
# ---------------------------------------------------------------------------
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma4:31b")

# ---------------------------------------------------------------------------
# Analysis cache — persisted to disk so re-runs skip already-analyzed listings
# ---------------------------------------------------------------------------
CACHE_DIR = Path(__file__).parent / "cache"
CACHE_FILE = CACHE_DIR / "analysis_cache.json"
_cache_lock = threading.Lock()
_analysis_cache: dict[str, dict] | None = None  # lazy-loaded


def _load_cache() -> dict[str, dict]:
    global _analysis_cache
    if _analysis_cache is not None:
        return _analysis_cache
    with _cache_lock:
        if _analysis_cache is not None:
            return _analysis_cache
        if CACHE_FILE.exists():
            try:
                _analysis_cache = json.loads(CACHE_FILE.read_text())
                print(f"  [cache] Loaded {len(_analysis_cache)} cached analyses")
            except (json.JSONDecodeError, OSError):
                _analysis_cache = {}
        else:
            _analysis_cache = {}
    return _analysis_cache


def _save_cache():
    with _cache_lock:
        if _analysis_cache is None:
            return
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        CACHE_FILE.write_text(json.dumps(_analysis_cache, indent=2))


def get_cached_analysis(listing_id: str) -> dict | None:
    """Return cached analysis for a listing, or None."""
    cache = _load_cache()
    return cache.get(listing_id)


CONDITION_PROMPT = """You are a professional home inspector, Vastu Shastra consultant, and Feng Shui master analyzing real estate listing photos.

## CRITICAL — Determining Property Facing Direction:
The LAST image provided is a Google Maps satellite view with a RED MARKER on the property. NORTH IS AT THE TOP of this image.
Use this satellite image to determine the front entrance direction:
1. Find the RED MARKER — this marks the exact property.
2. Identify the STREET that runs along the front of the house (where the driveway connects to).
3. The house "faces" the direction from the house TOWARD the street. For example:
   - If the street is BELOW (south of) the house → the house faces SOUTH
   - If the street is ABOVE (north of) the house → the house faces NORTH
   - If the street is to the LEFT (west of) the house → the house faces WEST
   - If the street is to the RIGHT (east of) the house → the house faces EAST
4. Remember: TOP of satellite image = NORTH, RIGHT = EAST, BOTTOM = SOUTH, LEFT = WEST.
5. This is the ONLY method you should use — it is the primary source of truth.
- Do NOT use shadow analysis — it is unreliable and WRONG.
- Do NOT guess or assume any default direction.
- DOUBLE CHECK your answer by verifying the street position relative to the house.
If no satellite image is available, state that direction could not be determined.

## Part 1: Property Condition Inspection
Examine these property images and identify:

1. STRUCTURAL ISSUES: foundation cracks, roof damage, sagging, water stains on ceilings/walls
2. WATER DAMAGE: stains, mold, warped flooring, discoloration
3. ELECTRICAL/PLUMBING: visible issues, outdated panels, exposed wiring
4. COSMETIC CONDITION: paint condition, flooring quality, fixture age
5. RENOVATION QUALITY: signs of DIY vs professional work, recent updates
6. RED FLAGS: staged to hide problems, missing photos of key areas, unusual angles
7. POSITIVES: recent renovations, high-end finishes, well-maintained

## Part 2: Vastu Shastra Analysis
Assess the following Vastu Shastra principles based ONLY on what you can observe in the images:

1. ENTRANCE & FACING: Main entrance direction and placement (ideal: N, E, or NE)
2. KITCHEN PLACEMENT: Position relative to the house (ideal: SE corner, cooking facing E)
3. MASTER BEDROOM: Location in the house (ideal: SW quadrant)
4. LIVING AREAS: Open space, natural light, central area (Brahmasthan)
5. BATHROOM/TOILET: Placement (ideal: NW or S, never NE)
6. STAIRCASE: Location and direction of ascent (ideal: S, SW, or W, clockwise)
7. WATER ELEMENTS: Swimming pool, water features (ideal: NE)
8. PLOT & SHAPE: Lot shape regularity, slopes, surroundings
9. COLORS & MATERIALS: Wall colors, flooring materials, overall energy
10. NATURAL LIGHT & VENTILATION: Window placement, cross-ventilation

## Part 3: Feng Shui Analysis
Assess the following Feng Shui principles based ONLY on what you can observe in the images:

1. FRONT ENTRANCE (Ming Tang): Is the approach open and welcoming? Clear path to the door? Bright hall effect? Any 'poison arrows' (sharp corners, T-intersections) aimed at entrance?
2. COMMAND POSITION: Can key rooms (bedroom, office, kitchen) be arranged so occupants see the door without being directly in line with it?
3. FIVE ELEMENTS BALANCE: Presence of Wood (plants, columns), Fire (lighting, triangles), Earth (stone, ceramics), Metal (fixtures, round shapes), Water (fountains, mirrors, curves)
4. CHI FLOW: Does the layout allow smooth energy flow? Long corridors? Staircase facing front door? Doors aligned creating 'rushing chi'?
5. YIN-YANG BALANCE: Balance of light/dark, soft/hard, quiet/active areas
6. NATURAL LIGHT: Window placement, quality of natural lighting
7. CLUTTER & SPACE: Open vs crowded spaces, storage visibility
8. KITCHEN (Health center): Stove placement, relationship to water elements, triangle between stove/sink/fridge
9. BEDROOM (Rest): Bed wall placement, mirror positions, electronics visibility
10. EXTERIOR & LANDSCAPING: Lot shape, tree placement, neighboring structures, water features, garden balance

## Part 4: South-Facing Vastu Pada Analysis (CONDITIONAL)
If the house appears to be SOUTH-FACING based on the entrance photos:
1. Identify the image with the CLEAREST front-on DAYTIME view of the main entrance/front facade. STRONGLY prefer image index 0 — on Redfin, the first photo is almost always the primary exterior shot and the best front-on view. Only pick a different index if image 0 is clearly NOT an exterior front view. Report the 0-based index.
2. Mentally divide the south wall of the house into 9 equal segments (padas), numbered 1–9 from the SOUTHEAST corner to the SOUTHWEST corner (Pada 1 = east/SE end, Pada 9 = west/SW end).
3. The 9 padas and their Vastu qualities for a south-facing house are:
   - Pada 1 (SE corner) — "Anil": INAUSPICIOUS — brings instability and health issues
   - Pada 2 — "Pusha": INAUSPICIOUS — causes financial losses
   - Pada 3 — "Vitatha": AUSPICIOUS — brings wealth and prosperity
   - Pada 4 — "Gruhakshata": AUSPICIOUS — best pada for south entrance, brings success and growth
   - Pada 5 (center) — "Yama": INAUSPICIOUS — ruled by Lord of Death, brings negativity
   - Pada 6 — "Gandharva": INAUSPICIOUS — causes instability
   - Pada 7 — "Bhringraj": INAUSPICIOUS — causes obstacles
   - Pada 8 — "Mriga": INAUSPICIOUS — causes fear and anxiety
   - Pada 9 (SW corner) — "Pitra": VERY INAUSPICIOUS — brings severe misfortune
4. Estimate which pada the main door falls into based on its horizontal position along the south wall in the entrance image. Also estimate the door's approximate horizontal position as a percentage from left edge (0%) to right edge (100%) of the facade visible in the image.
5. Report the pada number, name, whether it is auspicious, and the specific effect.

If the house is NOT south-facing or you cannot determine the facing direction, set "pada_analysis" to null.

For each finding across all four parts, note what you CAN observe and what you CANNOT determine from photos alone.
Rate severity: INFO / MINOR / MODERATE / MAJOR / CRITICAL

Also provide:
- Overall condition score (1-10)
- Vastu compliance score (1-10, 10 = excellent compliance)
- Feng Shui harmony score (1-10, 10 = excellent)
- Estimated deferred maintenance cost range
- Estimated total buyer cost: the listing price PLUS estimated costs for deferred maintenance, necessary repairs, closing costs (~3% of price), and any immediate renovations needed. Provide a low and high range.
- Recommended negotiation price: based on the condition issues found, suggest what price the buyer should offer/negotiate. Factor in deferred maintenance, cosmetic issues, and any red flags that justify a lower offer. Provide a low and high range with reasoning.
- Areas that need further in-person inspection

Return ONLY valid JSON (no markdown fences):
{
  "condition_score": <number 1-10>,
  "vastu_score": <number 1-10>,
  "vastu_assessment": "<EXCELLENT|GOOD|MODERATE|NEEDS_REMEDIES|POOR>",
  "feng_shui_score": <number 1-10>,
  "feng_shui_assessment": "<EXCELLENT|GOOD|MODERATE|NEEDS_CURES|POOR>",
  "summary": "<2-3 sentence overall assessment including Vastu/Feng Shui highlights>",
  "findings": [
    {
      "category": "<STRUCTURAL|WATER_DAMAGE|ELECTRICAL_PLUMBING|COSMETIC|RENOVATION|RED_FLAG|POSITIVE|VASTU|FENG_SHUI>",
      "severity": "<INFO|MINOR|MODERATE|MAJOR|CRITICAL>",
      "description": "<specific finding>",
      "action": "<recommended action, remedy, cure, or 'None - positive feature'>"
    }
  ],
  "estimated_maintenance": {"low": <number>, "high": <number>},
  "estimated_buyer_cost": {"low": <number>, "high": <number>, "notes": "<breakdown of what's included>"},
  "negotiation_price": {"low": <number>, "high": <number>, "reasoning": "<why this price range is justified based on findings>"},
  "vastu_remedies": ["<recommended Vastu remedies>"],
  "feng_shui_cures": ["<recommended Feng Shui cures>"],
  "pada_analysis": {
    "is_south_facing": <true|false>,
    "entrance_image_index": <number — which image (0-based) shows the entrance most clearly>,
    "door_position_pct": <number 0-100 — approximate horizontal position of the door in the entrance image, 0=left edge, 100=right edge>,
    "estimated_pada": <number 1-9>,
    "pada_name": "<Anil|Pusha|Vitatha|Gruhakshata|Yama|Gandharva|Bhringraj|Mriga|Pitra>",
    "is_auspicious": <true|false>,
    "effect": "<specific effect of this pada placement>",
    "all_padas": [
      {"number": 1, "name": "Anil", "auspicious": false, "has_door": false},
      {"number": 2, "name": "Pusha", "auspicious": false, "has_door": false},
      {"number": 3, "name": "Vitatha", "auspicious": true, "has_door": false},
      {"number": 4, "name": "Gruhakshata", "auspicious": true, "has_door": false},
      {"number": 5, "name": "Yama", "auspicious": false, "has_door": false},
      {"number": 6, "name": "Gandharva", "auspicious": false, "has_door": false},
      {"number": 7, "name": "Bhringraj", "auspicious": false, "has_door": false},
      {"number": 8, "name": "Mriga", "auspicious": false, "has_door": false},
      {"number": 9, "name": "Pitra", "auspicious": false, "has_door": false}
    ],
    "confidence": "<HIGH|MEDIUM|LOW — how confident you are in the facing direction and pada estimate>",
    "notes": "<any caveats about the analysis>"
  },
  "inspection_priorities": ["<area to inspect in person>"]
}"""


def fetch_image_as_base64(url: str) -> tuple[str, str] | None:
    """Download an image and return (base64_data, media_type). Synchronous."""
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/131.0.0.0 Safari/537.36"
            ),
            "Referer": "https://www.redfin.com/",
        }
        with httpx.Client(timeout=15.0, follow_redirects=True) as http:
            resp = http.get(url, headers=headers)
            resp.raise_for_status()

        content_type = resp.headers.get("content-type", "image/jpeg")
        if "png" in content_type:
            media_type = "image/png"
        elif "webp" in content_type:
            media_type = "image/webp"
        else:
            media_type = "image/jpeg"

        # Skip tiny images (icons, spacers, placeholders)
        if len(resp.content) < 5000:
            return None

        # Skip SVGs and GIFs
        if "svg" in content_type or "gif" in content_type:
            return None

        b64 = base64.standard_b64encode(resp.content).decode("utf-8")
        return b64, media_type
    except Exception as e:
        print(f"  [image_analyzer] Failed to fetch {url[:80]}...: {e}")
        return None


def fetch_satellite_image(address: str) -> dict | None:
    """Fetch a Google Maps satellite image for direction determination.

    Returns an image block dict or None. North is at the top of the image.
    """
    api_key = os.getenv("GOOGLE_MAPS_API_KEY")
    if not api_key:
        print("  [image_analyzer] No GOOGLE_MAPS_API_KEY set, skipping satellite image")
        return None

    url = "https://maps.googleapis.com/maps/api/staticmap"
    params = {
        "center": address,
        "zoom": "20",
        "size": "640x640",
        "maptype": "satellite",
        "markers": f"color:red|{address}",
        "key": api_key,
    }

    try:
        with httpx.Client(timeout=15.0) as http:
            resp = http.get(url, params=params)
            resp.raise_for_status()

        ct = resp.headers.get("Content-Type", "image/png")
        media_type = "image/png" if "png" in ct else "image/jpeg"
        b64 = base64.standard_b64encode(resp.content).decode("utf-8")
        print(f"  [image_analyzer] Fetched satellite image for {address}")
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": b64,
            },
        }
    except Exception as e:
        print(f"  [image_analyzer] Failed to fetch satellite image: {e}")
        return None


def analyze_listing_images(
    listing_id: str,
    image_urls: list[str],
    address: str = "",
) -> dict:
    """Run Gemma 4 (via Ollama) analysis on listing photos. Synchronous.

    Returns a structured dict with condition_score, findings, etc.
    Uses cache to skip already-analyzed listings.
    """
    # Check cache first
    cached = get_cached_analysis(listing_id)
    if cached:
        print(f"  [image_analyzer] Using cached analysis for {listing_id}")
        return cached

    urls_to_analyze = image_urls

    image_blocks = []
    for url in urls_to_analyze:
        result = fetch_image_as_base64(url)
        if result is None:
            continue
        b64_data, media_type = result
        image_blocks.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": b64_data,
            },
        })

    if not image_blocks:
        return {
            "condition_score": None,
            "summary": "No images could be loaded for analysis.",
            "findings": [],
            "estimated_maintenance": {"low": 0, "high": 0},
            "inspection_priorities": [],
            "error": "no_images",
        }

    # Fetch Google Maps satellite image for direction determination
    satellite_block = None
    if address:
        satellite_block = fetch_satellite_image(address)

    # Split into batches; smaller size for local Ollama inference
    BATCH_SIZE = 5
    batches = [
        image_blocks[i : i + BATCH_SIZE]
        for i in range(0, len(image_blocks), BATCH_SIZE)
    ]
    # Append satellite image to each batch for direction determination
    if satellite_block:
        for batch in batches:
            batch.append(satellite_block)

    all_findings = []
    all_scores = []
    all_vastu_scores = []
    all_feng_shui_scores = []
    all_summaries = []
    all_maintenance_low = []
    all_maintenance_high = []
    all_priorities = []
    all_vastu_remedies = []
    all_feng_shui_cures = []
    pada_analysis = None
    buyer_cost = None
    negotiation_price = None
    total_analyzed = 0

    for batch_idx, batch in enumerate(batches):
        print(
            f"  [image_analyzer] Batch {batch_idx + 1}/{len(batches)} "
            f"({len(batch)} images)"
        )
        result = _analyze_batch(batch)
        if result is None:
            continue

        total_analyzed += len(batch)
        if result.get("condition_score") is not None:
            all_scores.append(result["condition_score"])
        if result.get("vastu_score") is not None:
            all_vastu_scores.append(result["vastu_score"])
        if result.get("feng_shui_score") is not None:
            all_feng_shui_scores.append(result["feng_shui_score"])
        if result.get("summary"):
            all_summaries.append(result["summary"])
        all_findings.extend(result.get("findings", []))
        maint = result.get("estimated_maintenance", {})
        if maint.get("low") is not None:
            all_maintenance_low.append(maint["low"])
            all_maintenance_high.append(maint.get("high", 0))
        all_priorities.extend(result.get("inspection_priorities", []))
        all_vastu_remedies.extend(result.get("vastu_remedies", []))
        all_feng_shui_cures.extend(result.get("feng_shui_cures", []))
        if pada_analysis is None and result.get("pada_analysis"):
            pada_analysis = result["pada_analysis"]
        if buyer_cost is None and result.get("estimated_buyer_cost"):
            buyer_cost = result["estimated_buyer_cost"]
        if negotiation_price is None and result.get("negotiation_price"):
            negotiation_price = result["negotiation_price"]

    if not all_scores:
        return {
            "listing_id": listing_id,
            "condition_score": None,
            "summary": "All image batches failed analysis.",
            "findings": [],
            "estimated_maintenance": {"low": 0, "high": 0},
            "inspection_priorities": [],
            "error": "all_batches_failed",
        }

    # Merge results across batches
    avg_score = round(sum(all_scores) / len(all_scores), 1)
    avg_vastu = round(sum(all_vastu_scores) / len(all_vastu_scores), 1) if all_vastu_scores else None
    avg_feng_shui = round(sum(all_feng_shui_scores) / len(all_feng_shui_scores), 1) if all_feng_shui_scores else None
    # Deduplicate priorities, remedies, cures
    unique_priorities = list(dict.fromkeys(all_priorities))
    unique_vastu_remedies = list(dict.fromkeys(all_vastu_remedies))
    unique_feng_shui_cures = list(dict.fromkeys(all_feng_shui_cures))

    result = {
        "listing_id": listing_id,
        "condition_score": avg_score,
        "vastu_score": avg_vastu,
        "feng_shui_score": avg_feng_shui,
        "summary": " ".join(all_summaries) if all_summaries else "",
        "findings": all_findings,
        "estimated_maintenance": {
            "low": max(all_maintenance_low) if all_maintenance_low else 0,
            "high": max(all_maintenance_high) if all_maintenance_high else 0,
        },
        "vastu_remedies": unique_vastu_remedies,
        "feng_shui_cures": unique_feng_shui_cures,
        "estimated_buyer_cost": buyer_cost,
        "negotiation_price": negotiation_price,
        "pada_analysis": pada_analysis,
        "inspection_priorities": unique_priorities,
        "images_analyzed": total_analyzed,
    }

    # Save to cache
    cache = _load_cache()
    cache[listing_id] = result
    _save_cache()
    print(f"  [cache] Saved analysis for {listing_id}")

    return result


def _analyze_batch(image_blocks: list[dict]) -> dict | None:
    """Send a single batch of images to Gemma 4 via Ollama. Returns parsed JSON or None."""
    # Extract base64 image data for Ollama's format
    images = [block["source"]["data"] for block in image_blocks]

    payload = {
        "model": OLLAMA_MODEL,
        "messages": [{
            "role": "user",
            "content": CONDITION_PROMPT,
            "images": images,
        }],
        "stream": False,
        "options": {
            "num_predict": 10000,
        },
    }

    for attempt in range(2):
        try:
            with httpx.Client(timeout=600.0) as http:
                resp = http.post(
                    f"{OLLAMA_BASE_URL}/api/chat",
                    json=payload,
                )
                resp.raise_for_status()

            result = resp.json()
            raw = result["message"]["content"]

            # Try to extract JSON from markdown fences
            json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
            if json_match:
                raw = json_match.group(1)
            else:
                # Try to find raw JSON object
                obj_match = re.search(r"\{[\s\S]*\}", raw)
                if obj_match:
                    raw = obj_match.group(0)

            return json.loads(raw)

        except httpx.HTTPStatusError as e:
            if attempt == 0 and len(images) > 1:
                # Retry with fewer images
                print("  [image_analyzer] Ollama error, retrying with fewer images...")
                images = images[: len(images) // 2]
                payload["messages"][0]["images"] = images
                continue
            print(f"  [image_analyzer] Ollama HTTP error: {e}")
            return None

        except json.JSONDecodeError as e:
            if attempt == 0:
                print(f"  [image_analyzer] JSON parse failed, retrying... ({e})")
                continue
            print(f"  [image_analyzer] Could not parse JSON from Ollama response")
            return None

        except Exception as e:
            print(f"  [image_analyzer] Batch error: {e}")
            return None

    return None
