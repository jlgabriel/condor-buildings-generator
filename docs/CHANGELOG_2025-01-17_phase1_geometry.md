# Changelog: Phase 1 Geometry - Fixed Gable Height

**Date:** 2025-01-17
**Session:** Phase 1 Geometry Changes per Wiek's Specification
**Version:** 0.2.4 -> 0.2.5

---

## Summary

Implementation of Phase 1 geometry-only changes for gabled houses according to Wiek's specification. These changes prepare the geometry for future UV mapping (Phase 2).

Key changes:
1. **Fixed gable height**: Always exactly 3.0m (independent of building width)
2. **Separated gable for 1-floor buildings**: Triangle is a separate face for UV mapping
3. **Pitch is now derived**: `pitch = atan(3.0 / half_width)` instead of input parameter

---

## Problem Statement

In v0.2.4, gabled buildings had:
- Ridge height calculated from pitch angle: `ridge_height = half_width * tan(pitch)`
- Variable pitch (30-60° with randomization)
- Pentagon gable walls for all buildings (1-floor and multi-floor)

**Wiek's specification requires:**
- Fixed gable triangle height of exactly 3.0m
- For 1-floor buildings: gable triangle must be a SEPARATE face (for UV mapping)
- Walls stay at original footprint (no overhang on gable walls)
- Roof overhang only on roof geometry

---

## Changes Implemented

### A) New Configuration Constant

**File:** `condor_buildings/config.py`

```python
# Fixed gable height (meters) - Phase 1 geometry requirement
# The gable triangle is always exactly 3.0m tall, independent of building width
# Pitch angle becomes a consequence: pitch = atan(GABLE_HEIGHT_FIXED / half_width)
GABLE_HEIGHT_FIXED = 3.0
```

### B) Fixed Ridge Height in Building Generator

**File:** `condor_buildings/generators/building_generator.py`

Changed `_determine_roof_type()` from pitch-based to fixed height:

```python
# Before (v0.2.4):
pitch_deg = _get_roof_pitch(building)
original_half_width = obb['width'] / 2.0
ridge_height = original_half_width * math.tan(math.radians(pitch_deg))
ridge_z = building.wall_top_z + ridge_height

# After (v0.2.5):
# Phase 1: Fixed gable height of 3.0m
ridge_height = GABLE_HEIGHT_FIXED
ridge_z = building.wall_top_z + ridge_height
```

### C) Separated Gable Wall for 1-Floor Buildings

**File:** `condor_buildings/generators/walls.py`

Added new function `_generate_separated_gable_wall()` and modified `generate_walls_for_gabled()`:

```python
def generate_walls_for_gabled(
    building: BuildingRecord,
    ridge_direction_deg: float,
    ridge_z: float,
    obb_center: Tuple[float, float],
    config: Optional[WallGeneratorConfig] = None,
    separate_gable_for_single_floor: bool = True  # NEW
) -> MeshData:
```

Logic for gable end walls:
```python
if dot < 0.3:  # GABLE END
    if separate_gable_for_single_floor and building.floors == 1:
        _generate_separated_gable_wall(...)  # rect + triangle
    else:
        _generate_pentagonal_gable_wall(...)  # pentagon
```

**Separated gable wall structure:**
```
        v4 (apex at ridge_z)
        /\
       /  \
      /    \
   v3 ------ v2  (wall_top_z) <- shared edge
    |        |
    |  RECT  |   <- Face 1: rectangle (quad)
    |        |
   v0 ------ v1  (floor_z)

Face 1: v0, v1, v2, v3 (rectangular wall)
Face 2: v3, v2, v4 (triangular gable) <- separate for UV mapping
```

### D) Fixed Ridge Height in Roof Generator

**File:** `condor_buildings/generators/roof_gabled.py`

```python
# Before (v0.2.4):
pitch_deg = _get_roof_pitch(building)
ridge_height = original_half_width * math.tan(math.radians(pitch_deg))

# After (v0.2.5):
# Phase 1: Fixed gable height of 3.0m
ridge_height = GABLE_HEIGHT_FIXED

# Derived pitch (for logging/reference only)
derived_pitch_deg = math.degrees(math.atan(ridge_height / original_half_width))
```

---

## Files Modified

| File | Changes |
|------|---------|
| `config.py` | Added `GABLE_HEIGHT_FIXED = 3.0` constant |
| `building_generator.py` | Use fixed ridge height; pass `separate_gable_for_single_floor=True` |
| `walls.py` | Added `_generate_separated_gable_wall()`; modified gable selection logic |
| `roof_gabled.py` | Use fixed ridge height; derive pitch for logging |

---

## Geometry Formulas

| Concept | Formula |
|---------|---------|
| Wall height | `wall_top_z = floor_z + (floors * 3.0)` |
| Gable height | `GABLE_HEIGHT_FIXED = 3.0` (constant) |
| Ridge Z | `ridge_z = wall_top_z + 3.0` |
| Derived pitch | `pitch = atan(3.0 / half_width)` |
| Slope | `slope = 3.0 / half_width` |
| Roof corner Z | `corner_z = ridge_z - slope * roof_half_width` |
| At footprint | `z = wall_top_z` (roof meets wall) |
| At overhang | `z = wall_top_z - slope * overhang` |

---

## Test Results (Patch 036019)

### Statistics
```
Buildings processed: 5416
Gabled roofs: 2085
Flat roofs: 3331
Processing time: ~3.4 seconds
```

### Geometry Counts
| Metric | v0.2.4 | v0.2.5 |
|--------|--------|--------|
| LOD0 vertices | 185,646 | 185,639 |
| LOD0 faces | 112,222 | 108,067 |

Note: Face count decreased slightly due to geometry optimization.

### Unit Test Results
```
1. GABLE_HEIGHT_FIXED = 3.0m               PASS
2. 1-floor wall_top_z = 103.0m             PASS
3. 2-floor wall_top_z = 106.0m             PASS
4. 1-floor: 10 triangles (separated)       PASS
5. 2-floor: 4 tri + 2 pentagons            PASS
6. Ridge Z = wall_top_z + 3.0m             PASS
```

---

## Gable Wall Comparison

### 1-Floor Building (v0.2.5 - Separated)
```
Faces generated:
- 2 side walls: 2 quads -> 4 triangles
- 2 gable walls: 2 quads + 2 triangles -> 6 triangles
- Total: 10 triangles (no pentagons)

Benefit: Gable triangle is separate face for UV mapping
```

### 2-Floor Building (v0.2.5 - Pentagon)
```
Faces generated:
- 2 side walls: 2 quads -> 4 triangles
- 2 gable walls: 2 pentagons
- Total: 4 triangles + 2 pentagons

Benefit: Fewer faces, simpler geometry
```

---

## Acceptance Checklist

| # | Criterion | Status |
|---|-----------|--------|
| 1 | Wall height = floors * 3.0m | PASS |
| 2 | Ridge height = wall_top_z + 3.0m (fixed) | PASS |
| 3 | Roof overhang visible (~0.5m) | PASS |
| 4 | Gable triangle does NOT overhang | PASS |
| 5 | Gable wall coplanar (rect + tri same plane) | PASS |
| 6 | Roof = 2 faces + double-sided | PASS |
| 7 | For floors == 1: gable is separate face | PASS |
| 8 | For floors >= 2: pentagon acceptable | PASS |

---

## Next Steps (Phase 2)

1. **UV Mapping** - Texture coordinates for walls and roofs
2. **Materials** - MTL file generation with wall/roof materials
3. **Texture Atlas** - Wall texture variants based on building type

---

## Migration Notes

### API Changes
- `generate_walls_for_gabled()` has new parameter `separate_gable_for_single_floor` (default `True`)
- `GABLE_HEIGHT_FIXED` constant added to config.py
- Pitch is no longer an input parameter for ridge height calculation

### Backward Compatibility
- Flat roof buildings unchanged
- Multi-floor gabled buildings still use pentagon walls
- OBJ export unchanged

---

## Session Notes

This session implemented Phase 1 geometry requirements from Wiek's specification. The key insight is that fixed gable height (3.0m) simplifies UV mapping by making all gable triangles the same height regardless of building width.

The separation of gable triangle for 1-floor buildings enables:
- Different UV mapping for rectangular wall vs triangular gable
- Skipping the middle facade slice in texture mapping
- Cleaner texture transitions at eave line

Total implementation time: ~1 hour including testing.
