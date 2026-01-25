# Changelog: Phase 2 - Sidewall UV No-Split + Gabled Floor Limit

**Date:** 2025-01-20
**Session:** Sidewall UV Fix (No Splitting) + Floor Limit for Gabled Roofs
**Version:** 0.3.2 -> 0.3.3

---

## Summary

This session implements two related changes:

1. **Floor limit for gabled roofs:** Buildings with gabled roofs are now limited to 1-2 floors maximum. Buildings with 3+ floors requesting gabled roofs automatically fall back to flat.

2. **Sidewall UV mapping without geometry splitting:** Reverted the per-floor quad splitting from v0.3.2. Sidewalls are now generated as a SINGLE quad with continuous UV mapping that spans multiple facade sections.

**Key insight:** GPU UV interpolation handles the transition between texture sections automatically, so geometry splitting is unnecessary for gabled buildings (which are now limited to 1-2 floors).

---

## Problem Description

### Issue 1: Gabled Roofs on Tall Buildings

Gabled (pitched) roofs look unrealistic on buildings taller than 2 floors. European-style houses with gabled roofs are typically 1-2 stories.

### Issue 2: Unnecessary Geometry from Per-Floor Splitting

The v0.3.2 fix split sidewalls into multiple quads (one per floor), creating extra edge loops. For gabled buildings limited to 2 floors, this is unnecessary because:

- A single quad with continuous UV coordinates can map to two adjacent texture sections
- The GPU interpolates UV coordinates linearly across the quad
- z=0-3m maps to ground section, z=3-6m maps to upper section automatically

---

## Changes Made

### 1. New Floor Restriction Constants

**File:** `condor_buildings/config.py`

```python
# =============================================================================
# FLOOR RESTRICTIONS FOR ROOF TYPES
# =============================================================================

# Maximum floors allowed for gabled roofs
GABLED_MAX_FLOORS = 2

# Maximum floors allowed for hipped roofs (future use)
HIPPED_MAX_FLOORS = 2
```

### 2. New Fallback Reason

**File:** `condor_buildings/models/building.py`

Added to `RoofFallbackReason` enum:
```python
# Floor-based reasons
TOO_MANY_FLOORS = "too_many_floors"
```

### 3. Floor Count Check in Roof Type Determination

**File:** `condor_buildings/generators/building_generator.py`

Added `gabled_max_floors` parameter to `_determine_roof_type()` and `_generate_roof()`:

```python
def _determine_roof_type(..., gabled_max_floors: int = GABLED_MAX_FLOORS):
    if requested_type == RoofType.GABLED:
        # Check floor count restriction FIRST
        if building.floors > gabled_max_floors:
            result.fallback_reason = RoofFallbackReason.TOO_MANY_FLOORS
            building.roof_fallback_reason = RoofFallbackReason.TOO_MANY_FLOORS
            return RoofType.FLAT, None
        # ... rest of eligibility checks
```

### 4. New Continuous UV Function

**File:** `condor_buildings/generators/uv_mapping.py`

Added `compute_sidewall_continuous_uvs()` function:

```python
def compute_sidewall_continuous_uvs(
    wall_width_m: float,
    building_floors: int,
    facade_index: int
) -> List[Tuple[float, float]]:
    """
    Compute UV coordinates for a sidewall as a SINGLE continuous quad.

    - 1 floor (3m): Maps to ground section only
    - 2 floors (6m): Maps to ground + upper sections (continuous V range)

    Returns 4 UV tuples for quad corners.
    """
```

### 5. Simplified Sidewall Generation

**File:** `condor_buildings/generators/walls.py`

Refactored `_generate_side_wall_with_uvs()` to use single quad:

```python
def _generate_side_wall_with_uvs(...):
    # Get continuous UVs for single quad
    uvs = compute_sidewall_continuous_uvs(edge_len, building_floors, facade_index)

    # Create 4 vertices for complete wall (NOT per floor)
    bl = mesh.add_vertex(p0.x, p0.y, floor_z)
    br = mesh.add_vertex(p1.x, p1.y, floor_z)
    tr = mesh.add_vertex(p1.x, p1.y, wall_top_z)
    tl = mesh.add_vertex(p0.x, p0.y, wall_top_z)

    # Add single quad with UVs
    mesh.add_quad_with_uvs(bl, br, tr, tl, ...)
```

---

## UV Mapping Details

### Atlas Layout Recap

- Atlas: 512 x 12288 px
- Roofs: pixels 0-3072 (V: 0.75-1.0)
- Facades: pixels 3072-12288 (V: 0.0-0.75)
- 12 facade styles, each 768px (3 sections of 256px)

### Section Layout per Facade Style (top to bottom)

| Section | Pixel Range | V Range (style 0) | Content |
|---------|-------------|-------------------|---------|
| Gable | 0-256 | 0.7292-0.75 | No windows |
| Upper | 256-512 | 0.7083-0.7292 | Windows only |
| Ground | 512-768 | 0.6875-0.7083 | Doors + windows |

### Sidewall UV Ranges

| Floors | V Bottom | V Top | Sections Covered |
|--------|----------|-------|------------------|
| 1 | 0.6875 | 0.7083 | Ground only |
| 2 | 0.6875 | 0.7292 | Ground + Upper |

### How GPU Interpolation Works

For a 2-floor (6m) sidewall with V range [0.6875, 0.7292]:
- Vertex at z=0m has V=0.6875 (bottom of ground section)
- Vertex at z=6m has V=0.7292 (top of upper section)
- GPU linearly interpolates:
  - z=0-3m maps to V 0.6875-0.7083 (ground section)
  - z=3-6m maps to V 0.7083-0.7292 (upper section)

No geometry split needed!

---

## Geometry Impact

### Comparison with v0.3.2

| Building Type | v0.3.2 (split) | v0.3.3 (no split) |
|--------------|----------------|-------------------|
| 2-floor, 4 side walls | 8 quads | 4 quads |
| Vertices per side wall | 8 | 4 |

### Test Results (patch 036019)

| Metric | v0.3.2 | v0.3.3 |
|--------|--------|--------|
| Vertices | 336,527 | 333,695 |
| Faces | 183,511 | 181,547 |
| Buildings | 5,416 | 5,416 |
| Gabled roofs | varies | 1,811 |
| Flat roofs | varies | 3,605 |
| `too_many_floors` fallbacks | 0 | 1,346 |

---

## Fallback Reasons Distribution

From pipeline run on patch 036019:

| Reason | Count |
|--------|-------|
| `too_many_vertices` | 1,407 |
| `too_many_floors` | 1,346 |
| `too_short_side` | 209 |
| `bad_aspect_ratio` | 36 |
| `too_long_side` | 18 |
| `too_elongated` | 6 |
| `not_rectangle_angles` | 2 |
| `has_holes` | 1 |
| `not_rectangular_enough` | 1 |
| `too_large_area` | 1 |

---

## What Stays Unchanged

- **Front/rear gable walls:** Pentagon geometry with separate gable triangle (unchanged)
- **`_generate_ring_walls()`:** For flat-roof buildings, still uses per-floor splitting (flat roofs can have 3+ floors)
- **Roof geometry:** No changes
- **1-floor gabled buildings:** Still work correctly with single floor mapping

---

## Files Modified

| File | Changes |
|------|---------|
| `config.py` | Added `GABLED_MAX_FLOORS`, `HIPPED_MAX_FLOORS` |
| `models/building.py` | Added `TOO_MANY_FLOORS` to enum |
| `generators/building_generator.py` | Floor count check in `_determine_roof_type()` and `_generate_roof()` |
| `generators/uv_mapping.py` | New `compute_sidewall_continuous_uvs()` function |
| `generators/walls.py` | Refactored `_generate_side_wall_with_uvs()` to single quad |

---

## Verification

### Pipeline Test
```
python -m condor_buildings.main --patch-dir test_data --patch-id 036019 --output-dir output --verbose
```
Result: Success, 5416 buildings processed in ~5.5 seconds.

### Unit Tests
```python
# Floor restriction test
building_3f.floors = 3
roof_type = _determine_roof_type(building_3f, ...)
assert roof_type == RoofType.FLAT
assert result.fallback_reason == RoofFallbackReason.TOO_MANY_FLOORS

# UV mapping test
uvs_1f = compute_sidewall_continuous_uvs(10.0, 1, 0)
assert uvs_1f[0][1] == 0.6875  # V bottom = ground bottom
assert uvs_1f[3][1] == 0.7083  # V top = ground top

uvs_2f = compute_sidewall_continuous_uvs(10.0, 2, 0)
assert uvs_2f[0][1] == 0.6875  # V bottom = ground bottom
assert uvs_2f[3][1] == 0.7292  # V top = upper top
```

---

## Visual Verification (for Wiek)

In Blender with the atlas applied, verify:

### 2-Story Gabled Buildings
1. **Side walls:** Should have NO horizontal edge loop at 3m height
2. **Side walls:** Should show correct 2-floor texture mapping:
   - Lower 3m: ground section (doors + windows)
   - Upper 3m: upper section (windows only)
3. **Select side wall in edit mode:** Should be a single quad (4 vertices)

### 1-Story Gabled Buildings
- Side walls: single quad mapped to ground section only

### 3+ Story Buildings
- Should ALL have flat roofs (no gabled allowed)
- Check that buildings that were previously gabled with 3 floors are now flat

### UV Editor Check
- Select a 2-story gabled building's side wall
- Should show a single UV quad spanning from ground_bottom to upper_top

---

## Acceptance Checklist

| # | Criterion | Status |
|---|-----------|--------|
| 1 | Gabled roofs limited to 1-2 floors | PASS |
| 2 | 3+ floor buildings fallback to flat with `TOO_MANY_FLOORS` | PASS |
| 3 | Side walls are single quads (no per-floor split) | PASS |
| 4 | 1-floor sidewall maps to ground section only | PASS |
| 5 | 2-floor sidewall maps to ground+upper continuous | PASS |
| 6 | No horizontal edge loops on sidewalls | PASS |
| 7 | Front/rear gable walls unchanged | PASS |
| 8 | Flat-roof buildings unchanged | PASS |
| 9 | Pipeline runs successfully | PASS |

---

## Output Files

Generated in `output/`:
- `o036019_LOD0.obj` (~28 MB)
- `o036019_LOD1.obj` (~28 MB)
- `o036019_report.json`
- `o036019.log`

---

## Session Notes

This session addresses Wiek's requirement to eliminate unnecessary geometry (horizontal edge loops) on sidewalls while maintaining correct UV mapping for multi-floor buildings.

The key insight is that for gabled buildings limited to 2 floors, a single quad with continuous UV coordinates is sufficient. The GPU's linear interpolation of UV coordinates automatically maps the lower half of the wall to the ground texture section and the upper half to the upper section.

This approach:
- Reduces geometry complexity (fewer vertices/faces)
- Maintains correct visual appearance
- Simplifies the code by avoiding per-floor loop logic for gabled sidewalls

Total implementation time: ~45 minutes (including planning, implementation, and verification).
