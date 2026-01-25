# Changelog: Phase 2 - UV Mapping + Texture Atlas

**Date:** 2025-01-18
**Session:** Phase 2 UV Mapping Implementation per Wiek's Specification
**Version:** 0.2.5 -> 0.3.0

---

## Summary

Implementation of Phase 2: UV mapping and texture atlas system for building walls and roofs. This enables textured rendering of buildings using a single combined atlas with roof patterns and facade styles.

Key features:
1. **6 roof texture patterns** - Deterministically selected per building
2. **12 facade styles** - Each with 3 sections (gable, upper floors, ground floor)
3. **Proper UV wrapping** - Horizontal wrap allowed, vertical wrap prevented
4. **Meter-based scaling** - 3m = 1.0 UV unit for aligned textures
5. **1-floor special case** - Skips middle section (gable + ground only)

---

## Texture Atlas Layout

```
Atlas: 512 x 12288 pixels
U: [0..1] horizontal (can exceed 1.0 for wrapping)
V: [0..1] vertical (NO wrapping - stays within slice)

ROOF REGION (V in [0, 0.25]):
├── Pattern 0: V [0.0000, 0.0417]
├── Pattern 1: V [0.0417, 0.0833]
├── Pattern 2: V [0.0833, 0.1250]
├── Pattern 3: V [0.1250, 0.1667]
├── Pattern 4: V [0.1667, 0.2083]
└── Pattern 5: V [0.2083, 0.2500]

FACADE REGION (V in [0.25, 1.0]):
12 styles, each with 3 sections:
├── GABLE (no windows):     top of block
├── UPPER (windows only):   middle of block
└── GROUND (doors+windows): bottom of block

Each style occupies V range of 0.0625 (0.75 / 12)
Each section occupies V range of 0.0208 (0.0625 / 3)
```

---

## Files Created

### `condor_buildings/generators/uv_mapping.py` (NEW)

Core UV mapping module with:

| Function | Purpose |
|----------|---------|
| `select_building_variations(seed)` | Deterministic roof/facade selection |
| `get_roof_v_range(roof_index)` | V range for roof pattern |
| `get_facade_section_v_range(facade_index, section)` | V range for facade section |
| `compute_wall_u_span(wall_width_m)` | U span with meter rounding |
| `compute_roof_slope_uvs(length, width, index)` | 4 UVs for roof quad |
| `compute_wall_quad_uvs(width, index, section)` | 4 UVs for wall quad |
| `compute_gable_triangle_uvs(width, index)` | 3 UVs for gable triangle |
| `compute_pentagon_wall_uvs(width, index, floors)` | 5 UVs for pentagon |

---

## Files Modified

### `condor_buildings/config.py`

Added texture atlas constants:

```python
# Atlas dimensions
ATLAS_WIDTH_PX = 512
ATLAS_HEIGHT_PX = 12288

# Roof patterns
ROOF_PATTERN_COUNT = 6
ROOF_SLICE_V = 512 / 12288  # 0.0417

# Facade styles
FACADE_STYLE_COUNT = 12
FACADE_REGION_V_MIN = 0.25

# Wall module
WALL_MODULE_M = 3.0  # 3m = 1.0 UV unit
```

### `condor_buildings/io/obj_exporter.py`

Updated to export UV coordinates:

- Added `total_uvs` to `ExportStats`
- Writes `vt u v` lines after vertices
- Changed face format from `f v1 v2 v3` to `f v1/vt1 v2/vt2 v3/vt3`
- Handles UV index offsets during mesh merge

### `condor_buildings/generators/walls.py`

Added UV coordinates to all wall types:

- `generate_walls()` - Calls `select_building_variations()` for facade index
- `generate_walls_for_gabled()` - UV-aware wall generation
- `_generate_side_wall_with_uvs()` - New function for side walls
- `_generate_pentagonal_gable_wall()` - Now with UV coordinates
- `_generate_separated_gable_wall()` - Ground + gable section UVs
- `_generate_ring_walls()` - Added facade_index parameter

### `condor_buildings/generators/roof_gabled.py`

Added UV coordinates to roof slopes:

- `generate_gabled_roof()` - Gets roof_index from building seed
- `_generate_obb_roof_v3()` - Added roof_index parameter, computes and applies UVs
- `_duplicate_faces_reversed()` - Now also duplicates face_uvs

### `condor_buildings/generators/roof_flat.py`

Added UV coordinates to flat roofs (bugfix during session):

- `generate_flat_roof()` - Gets roof_index from building seed
- `_generate_simple_roof()` - Added roof_index parameter, computes UVs based on world XY
- `_generate_roof_with_holes()` - UV mapping for polygons with holes
- `_generate_fan_roof()` - UV mapping for fallback fan triangulation
- UVs use world-space XY coordinates scaled to 3m = 1.0 UV unit for seamless tiling

**Note:** This was a critical fix discovered during testing. Without UVs on flat roofs,
the OBJ exporter would skip all UV data because `len(face_uvs) != len(faces)`.

---

## UV Mapping Rules

### Roof Slopes
- V: ridge (v_max) → eave (v_min), full slice height
- U: `u_span = roof_length / roof_width` (preserves aspect ratio)
- Can wrap horizontally (U > 1.0)

### Wall Quads
- U: `u_span = round(wall_width_m) / 3.0`
- V: Maps to single facade section (no vertical wrap)
- Same u_span for all segments ensures brick alignment

### 1-Floor Gable Buildings
- Rectangle → 'ground' section (doors + windows)
- Triangle → 'gable' section (no windows)
- Skips 'upper' section entirely

### Multi-Floor Buildings
- Pentagon face spans ground + upper + gable sections
- UV mapping distributes across sections proportionally

---

## Variation Selection

Deterministic per-building using seed:

```python
rng = random.Random(building.seed)
roof_index = rng.randint(0, 5)    # 6 patterns
facade_index = rng.randint(0, 11) # 12 styles
```

Same seed always produces same variations across runs.

---

## Test Results

### Unit Tests
```
Test 1 PASS: Variations deterministic
Test 2 PASS: Roof V ranges in [0, 0.25]
Test 3 PASS: Facade V ranges in [0.25, 1.0]
Test 4 PASS: Wall U span calculation correct
Test 5 PASS: Roof UVs preserve aspect ratio
Test 6 PASS: Wall quad UVs have 4 points
Test 7 PASS: Gable triangle UVs have 3 points
Test 8 PASS: Last facade ends at V=1.0
```

### Integration Test (10x6m 1-floor building)
```
Walls: 18 vertices, 10 faces, 22 UVs
Roof: 6 vertices, 8 faces, 8 UVs
Combined: 24 vertices, 18 faces, 30 UVs
OBJ export: vt lines present, f v/vt format confirmed
```

### Full Pipeline Test (patch 036019)
```
Buildings: 5,416 (2,085 gabled + 3,331 flat)
Vertices: 185,639
UVs: 193,987
Faces: 108,067
Output files: o036019_LOD0.obj, o036019_LOD1.obj
```

---

## OBJ Output Format

Before (v0.2.5):
```obj
v 0.000000 0.000000 100.000000
v 10.000000 0.000000 100.000000
f 1 2 3
```

After (v0.3.0):
```obj
v 0.000000 0.000000 100.000000
v 10.000000 0.000000 100.000000
vt 0.000000 0.979167
vt 3.333333 0.979167
f 1/1 2/2 3/3
```

---

## Acceptance Checklist

| # | Criterion | Status |
|---|-----------|--------|
| 1 | Roof textures in V [0, 0.25] | PASS |
| 2 | Roof horizontal wrap (U > 1) | PASS |
| 3 | Roof preserves aspect ratio | PASS |
| 4 | Facade textures in V [0.25, 1.0] | PASS |
| 5 | Ground floor uses doors+windows section | PASS |
| 6 | Upper floors use windows-only section | PASS |
| 7 | Gable triangle uses no-windows section | PASS |
| 8 | 1-floor buildings skip middle section | PASS |
| 9 | Wall U uses meter rounding | PASS |
| 10 | No geometry changes from Phase 1 | PASS |
| 11 | Deterministic variation per building | PASS |
| 12 | OBJ exports vt and f v/vt format | PASS |

---

## Next Steps (Phase 3)

1. **MTL file generation** - Material definitions for Condor
2. **Texture atlas creation** - Actual texture images
3. **Visual verification** - Test in Blender with atlas texture
4. **Multi-floor UV refinement** - Split walls per-floor for proper section mapping

---

## Migration Notes

### API Changes
- `_generate_obb_roof_v3()` now requires `roof_index` parameter
- `_duplicate_faces_reversed()` now accepts `uv_start_idx` parameter
- `_generate_ring_walls()` now accepts `facade_index` and `building_floors`
- Wall generation functions now require `edge_len` and `facade_index`

### Backward Compatibility
- Geometry unchanged from Phase 1
- OBJ files without UVs still supported (legacy export)
- Building seed mechanism unchanged

---

## Session Notes

Phase 2 implements the complete UV mapping system per Wiek's specification. The atlas layout was adjusted to ensure all 12 facade styles fit within the available V range [0.25, 1.0].

Key insight: The original specification had facade blocks of 1536px each (12 * 1536 = 18432px), which exceeded the available space (9216px). The implementation scales the 12 styles proportionally to fit.

### Bugfix: Flat Roof UV Mapping

During testing, we discovered that the OBJ exporter was not writing any UV coordinates. The cause was that flat roofs (3,331 buildings in the test patch) had no UV coordinates, causing `len(face_uvs) != len(faces)`. The exporter's safety check:

```python
has_uvs = len(merged.uvs) > 0 and len(merged.face_uvs) == len(merged.faces)
```

...would evaluate to `False`, skipping all UV output. The fix was to add UV mapping to `roof_flat.py` using world-space XY coordinates scaled to match wall textures (3m = 1.0 UV unit).

Total implementation time: ~3 hours (including bugfix).
