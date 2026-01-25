# Changelog: Phase 2 - UV V Coordinate Inversion Fix

**Date:** 2025-01-19
**Session:** Fix UV V Coordinate Inversion per Wiek's Specification
**Version:** 0.3.0 -> 0.3.1

---

## Summary

Fixed the UV V coordinate inversion issue reported by Wiek. The texture atlas has:
- **Roofs** at the TOP of the atlas (pixels 0-3071)
- **Facades** at the BOTTOM of the atlas (pixels 3072-12287)

The correct UV convention is:
- **V = 1.0** at atlas TOP (pixel y = 0)
- **V = 0.0** at atlas BOTTOM (pixel y = 12288)

The previous implementation had this inverted, causing roofs to show facade textures and vice versa.

---

## Problem Description

### Before (Incorrect)
```
Roofs mapped to:    V in [0.0, 0.25]   (BOTTOM of atlas - WRONG)
Facades mapped to:  V in [0.25, 1.0]  (TOP of atlas - WRONG)
```

### After (Correct)
```
Roofs mapped to:    V in [0.75, 1.0]  (TOP of atlas - CORRECT)
Facades mapped to:  V in [0.0, 0.75]  (BOTTOM of atlas - CORRECT)
```

---

## Files Modified

### `condor_buildings/config.py`

**Changed constants (lines 145-168):**

Before:
```python
ROOF_REGION_V_MAX = ROOF_PATTERN_COUNT * ROOF_SLICE_V    # 0.25
FACADE_REGION_V_MIN = ROOF_REGION_V_MAX  # 0.25
```

After:
```python
# CORRECTED: Roof region is at TOP of atlas (high V values)
ROOF_REGION_V_MIN = 1.0 - (ROOF_PATTERN_COUNT * ROOF_SLICE_V)  # 0.75
ROOF_REGION_V_MAX = 1.0  # Top of atlas

# Facade styles (BELOW roofs - low V values)
FACADE_STYLE_HEIGHT_PX = 768
FACADE_SECTION_HEIGHT_PX = 256

# CORRECTED: Facade region is BELOW roof region
FACADE_REGION_V_MAX = ROOF_REGION_V_MIN  # 0.75 (top of facade region)
FACADE_REGION_V_MIN = 0.0  # Bottom of atlas
```

### `condor_buildings/generators/uv_mapping.py`

**Updated imports (lines 19-27):**
```python
from ..config import (
    ROOF_PATTERN_COUNT,
    ROOF_SLICE_V,
    ROOF_REGION_V_MAX,  # NEW
    FACADE_STYLE_COUNT,
    FACADE_REGION_V_MAX,  # CHANGED (was FACADE_REGION_V_MIN)
    FACADE_REGION_V_MIN,
    WALL_MODULE_M,
)
```

**Updated module constants (lines 29-35):**
```python
FACADE_REGION_V_SIZE = FACADE_REGION_V_MAX - FACADE_REGION_V_MIN  # 0.75
FACADE_BLOCK_V = FACADE_REGION_V_SIZE / FACADE_STYLE_COUNT  # 0.0625
FACADE_SECTION_V = FACADE_BLOCK_V / 3.0  # 0.020833...
```

**Rewritten `get_roof_v_range()` (lines 62-87):**
```python
def get_roof_v_range(roof_index: int) -> Tuple[float, float]:
    # V decreases as roof_index increases (going DOWN the atlas)
    v_max = ROOF_REGION_V_MAX - roof_index * ROOF_SLICE_V       # = 1.0 - i * 0.0417
    v_min = ROOF_REGION_V_MAX - (roof_index + 1) * ROOF_SLICE_V # = 1.0 - (i+1) * 0.0417
    return v_min, v_max
```

**Rewritten `get_facade_section_v_range()` (lines 90-135):**
```python
def get_facade_section_v_range(facade_index: int, section: str) -> Tuple[float, float]:
    section_offsets = {
        'gable': 0,   # Top of block (highest V)
        'upper': 1,   # Middle of block
        'ground': 2,  # Bottom of block (lowest V)
    }

    # Block V top - V decreases as facade_index increases
    block_v_top = FACADE_REGION_V_MAX - facade_index * FACADE_BLOCK_V

    # Section within block (offset from top)
    section_offset = section_offsets[section]
    v_max = block_v_top - section_offset * FACADE_SECTION_V
    v_min = v_max - FACADE_SECTION_V

    return v_min, v_max
```

**Updated docstrings in `compute_roof_slope_uvs()` (lines 160-201):**
- Clarified that eave maps to v_min (lower V) and ridge maps to v_max (higher V)

---

## Corrected UV Ranges

### Roofs (V in [0.75, 1.0])

| Roof Index | V Min | V Max |
|------------|-------|-------|
| 0 | 0.9583 | 1.0000 |
| 1 | 0.9167 | 0.9583 |
| 2 | 0.8750 | 0.9167 |
| 3 | 0.8333 | 0.8750 |
| 4 | 0.7917 | 0.8333 |
| 5 | 0.7500 | 0.7917 |

### Facades (V in [0.0, 0.75])

| Facade Index | Section | V Min | V Max |
|--------------|---------|-------|-------|
| 0 | gable | 0.7292 | 0.7500 |
| 0 | upper | 0.7083 | 0.7292 |
| 0 | ground | 0.6875 | 0.7083 |
| 5 | gable | 0.4167 | 0.4375 |
| 5 | upper | 0.3958 | 0.4167 |
| 5 | ground | 0.3750 | 0.3958 |
| 11 | gable | 0.0417 | 0.0625 |
| 11 | upper | 0.0208 | 0.0417 |
| 11 | ground | 0.0000 | 0.0208 |

---

## Verification

### Unit Tests
```
Roofs (V in [0.75, 1.0]):     OK
Facades (V in [0.0, 0.75]):   OK
Section order (gable > upper > ground): OK
All verifications passed!
```

### Pipeline Test (patch 036019)
```
Buildings: 5,416 (2,085 gabled + 3,331 flat)
Vertices: 185,639
UVs: 193,987
Faces: 108,067
Execution time: ~4.5 seconds
```

### UV Value Distribution in Output OBJ
```
Unique V values range from 0.0 to 1.0:
- Facades: 0.0, 0.0208, 0.0417, ..., 0.75
- Roofs: 0.75, 0.7708, 0.7917, ..., 1.0
```

---

## Output Files

Generated in `output/`:
- `o036019_LOD0.obj` (15.9 MB)
- `o036019_LOD1.obj` (15.9 MB)
- `o036019_report.json` (185 KB)
- `o036019.log` (1.1 MB)

---

## Visual Verification (for Wiek)

After applying the texture atlas in Blender:
1. **Roofs** should show tile/shingle patterns (not brick textures)
2. **Ground floor walls** should show doors + windows
3. **Upper floor walls** should show windows only
4. **Gable triangles** should show plain texture (no windows)

---

## Acceptance Checklist

| # | Criterion | Status |
|---|-----------|--------|
| 1 | Roof textures in V [0.75, 1.0] | PASS |
| 2 | Facade textures in V [0.0, 0.75] | PASS |
| 3 | V=1.0 at atlas TOP | PASS |
| 4 | V=0.0 at atlas BOTTOM | PASS |
| 5 | Gable section has highest V within facade style | PASS |
| 6 | Ground section has lowest V within facade style | PASS |
| 7 | Pipeline generates correct output | PASS |
| 8 | No geometry changes | PASS |

---

## Session Notes

This fix addresses the UV V coordinate inversion issue identified by Wiek after testing the Phase 2 output in Blender. The root cause was that the original implementation assumed V=0 at atlas top, but the correct convention (per Wiek's specification) is V=1.0 at atlas top.

The fix only required changes to two files (`config.py` and `uv_mapping.py`) and did not affect the geometry generation or the structure of the UV mapping functions - only the V range calculations were corrected.

Total implementation time: ~30 minutes (including planning and verification).
