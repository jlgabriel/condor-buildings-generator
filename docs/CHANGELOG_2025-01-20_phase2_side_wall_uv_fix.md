# Changelog: Phase 2 - Side Wall UV Multi-Floor Fix

**Date:** 2025-01-20
**Session:** Fix Side Wall UV Mapping for Multi-Floor Buildings
**Version:** 0.3.1 -> 0.3.2

---

## Summary

Fixed a bug where side walls (walls parallel to the ridge) in multi-floor gabled buildings were only mapped to the 'ground' texture section, ignoring upper floors. The fix also applies to all walls in flat-roof buildings.

**Before:** A 2-story building's side walls showed only ground floor texture (doors+windows) stretched across both floors.

**After:** A 2-story building's side walls correctly show:
- Ground floor (0-3m): 'ground' section (doors + windows)
- Upper floor (3-6m): 'upper' section (windows only)

---

## Problem Description

### Root Cause

In `walls.py`, the function `_generate_side_wall_with_uvs()` created a single quad for the entire wall height and always used `section='ground'` for UV mapping, ignoring the `building_floors` parameter.

```python
# BEFORE (incorrect)
section = 'ground'  # Always ground, ignoring building_floors
uvs = compute_wall_quad_uvs(edge_len, facade_index, section)
# Single quad from floor_z to wall_top_z
```

### Technical Constraint

Facade textures do NOT wrap vertically (V). Each section ('ground', 'upper', 'gable') occupies a distinct V range in the atlas. A single quad cannot map to multiple V sections.

**Solution:** Split wall geometry into multiple quads (one per floor, 3m each), with each quad mapped to the appropriate texture section.

---

## Files Modified

### `condor_buildings/generators/walls.py`

**1. Added imports (lines 22-24):**
```python
from .uv_mapping import (
    ...
    compute_multi_floor_wall_uvs,  # NEW
)
from ..config import DEFAULT_FLOOR_HEIGHT  # NEW
```

**2. Rewrote `_generate_side_wall_with_uvs()` (lines 196-245):**

Now generates one quad per floor instead of a single quad:

```python
def _generate_side_wall_with_uvs(..., building_floors):
    # Get UVs for each floor segment
    floor_uvs = compute_multi_floor_wall_uvs(edge_len, building_floors, facade_index)

    # Generate one quad per floor
    for floor_idx in range(building_floors):
        z_bottom = floor_z + (floor_idx * DEFAULT_FLOOR_HEIGHT)
        z_top = z_bottom + DEFAULT_FLOOR_HEIGHT

        # Create vertices and UVs for this floor segment
        uvs = floor_uvs[floor_idx]
        # ... add quad with UVs
```

**3. Rewrote `_generate_ring_walls()` (lines 413-495):**

Same fix applied for flat-roof buildings and hole walls:

```python
def _generate_ring_walls(..., building_floors):
    for each edge:
        floor_uvs = compute_multi_floor_wall_uvs(edge_len, building_floors, facade_index)

        for floor_idx in range(building_floors):
            # Generate quad for this floor with correct UV section
```

---

## UV Mapping Details

### Existing Function Reused

`compute_multi_floor_wall_uvs()` in `uv_mapping.py` (lines 370-421) already existed but was not being used. It returns:
- `[0]` = ground floor UVs ('ground' section: doors + windows)
- `[1..n]` = upper floor UVs ('upper' section: windows only)

### UV Ranges (facade_index=0 example)

| Floor | Section | V Min | V Max |
|-------|---------|-------|-------|
| 0 (ground) | ground | 0.0000 | 0.0208 |
| 1 (upper) | upper | 0.0208 | 0.0417 |
| 2+ (upper) | upper | 0.0208 | 0.0417 |

All floors use the same U span for horizontal alignment (bricks line up vertically).

---

## Geometry Impact

The fix increases vertex/face count proportionally to the number of floors:

| Building Type | Before | After |
|--------------|--------|-------|
| 1-floor, 4 walls | 4 quads | 4 quads (unchanged) |
| 2-floor, 4 walls | 4 quads | 8 quads |
| 3-floor, 4 walls | 4 quads | 12 quads |

### Test Patch 036019 Results

| Metric | Before (v0.3.1) | After (v0.3.2) |
|--------|-----------------|----------------|
| Vertices | ~185,639 | 336,527 |
| Faces | ~108,067 | 183,511 |
| UVs | ~193,987 | 344,875 |
| Buildings | 5,416 | 5,416 |
| Processing time | ~4.5s | ~10s |

The increase is expected and necessary for correct UV mapping.

---

## Verification

### Pipeline Test
```
python -m condor_buildings.main --patch-dir test_data --patch-id 036019 --output-dir output
```
Result: Success, 5416 buildings processed.

### UV Verification in OBJ

Sample UV coordinates from output file:
```
vt 0.000000 0.000000      # Ground floor bottom
vt 11.000000 0.000000
vt 11.000000 0.020833     # Ground floor top
vt 0.000000 0.020833
vt 0.000000 0.020833      # Upper floor bottom (same as ground top)
vt 11.000000 0.020833
vt 11.000000 0.041667     # Upper floor top
vt 0.000000 0.041667
```

Confirms: Two separate quads with distinct V ranges for ground and upper floors.

---

## Visual Verification (for Wiek)

In Blender with the atlas applied, verify:

### 2-Story Gabled Buildings
1. **Front/rear gable walls:** Should show 2-story mapping (already worked before)
2. **Side walls (NEW FIX):** Should now ALSO show 2-story mapping:
   - Lower half: ground floor texture (doors + windows)
   - Upper half: upper floor texture (windows only)

### 1-Story Buildings
- All walls should show single ground floor section (unchanged behavior)

### Flat-Roof Buildings
- Multi-story flat buildings should now show correct per-floor textures on all walls

### Check in UV Editor
- Select a 2-story building
- Side walls should show 2 stacked UV regions in the editor
- Both regions should have the same horizontal U extent (brick alignment)

---

## Acceptance Checklist

| # | Criterion | Status |
|---|-----------|--------|
| 1 | Side walls split into per-floor quads | PASS |
| 2 | Ground floor uses 'ground' section | PASS |
| 3 | Upper floors use 'upper' section | PASS |
| 4 | Same U span across all floor segments | PASS |
| 5 | 1-floor buildings unchanged | PASS |
| 6 | Flat-roof buildings also fixed | PASS |
| 7 | Hole walls (courtyards) also fixed | PASS |
| 8 | Pipeline runs successfully | PASS |
| 9 | No changes to roof geometry | PASS |
| 10 | Gable walls unchanged | PASS |

---

## Output Files

Generated in `output/`:
- `o036019_LOD0.obj` (~27 MB)
- `o036019_LOD1.obj` (~27 MB)
- `o036019_report.json`
- `o036019.log`

---

## Session Notes

This fix addresses the side wall UV mapping bug identified by Wiek during visual testing in Blender. The root cause was that `_generate_side_wall_with_uvs()` was designed as a provisional implementation with a TODO comment indicating that multi-floor support would require geometry splitting.

The fix leverages the existing `compute_multi_floor_wall_uvs()` helper function that was already implemented in `uv_mapping.py` but never called from the wall generation code.

Total implementation time: ~30 minutes (including planning and verification).
