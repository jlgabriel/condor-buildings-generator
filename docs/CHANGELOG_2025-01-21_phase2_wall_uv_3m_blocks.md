# Changelog: Phase 2 - Wall UV Mapping Update (3m U-Blocks + Door Offset)

**Date:** 2025-01-21
**Session:** Update Wall UV Mapping per Wiek's New Specification
**Version:** 0.3.3 -> 0.3.4

---

## Summary

Updated wall UV mapping in the U direction to use 3-meter blocks instead of 1-meter blocks, and added a U offset of 0.33 to reduce door frequency on facades.

**Before:**
- Scale: 1 meter = 0.33 U (details too dense)
- U start: 0.0 (doors appear frequently)
- Rounding: round() to nearest 1m

**After:**
- Scale: 3 meters = 0.33 U (features larger/more realistic)
- U start: 0.33 (skip door section, doors less frequent)
- Rounding: ceil() to nearest 3m multiple

---

## Problem Description

### Wiek's Feedback

The previous UV mapping scale (1m = 0.33U) produced facade textures with details that were too small and dense. Additionally, doors appeared too frequently across walls because the UV mapping started at U=0.0 (the door section).

### New Texture Layout in U

The facade texture is divided horizontally into three equal bands:
```
U 0.00 → 0.33 = Door section
U 0.33 → 0.66 = Window section
U 0.66 → 1.00 = Window section
```

By starting wall UV mapping at U=0.33, we skip the door section by default, making doors appear only when tiling wraps around (less frequently).

---

## Changes Made

### 1. New Constants in `config.py`

Added at lines 181-186:
```python
# Wall UV mapping constants (Phase 2 update - 2025-01-21)
# New Wiek rule: 3m blocks with U offset to reduce door frequency
WALL_BLOCK_METERS = 3.0       # Each 3m block maps to WALL_BLOCK_U
WALL_BLOCK_U = 1.0 / 3.0      # U width per 3m block (0.3333...)
WALL_U_OFFSET = 1.0 / 3.0     # Start U offset (skip door section at U 0.0-0.33)
WALL_MIN_METERS = 3.0         # Minimum wall length for UV mapping
```

### 2. Updated `uv_mapping.py`

#### 2.1 Added Imports (lines 27-30)
```python
from ..config import (
    ...
    WALL_BLOCK_METERS,
    WALL_BLOCK_U,
    WALL_U_OFFSET,
    WALL_MIN_METERS,
)
```

#### 2.2 Rewrote `compute_wall_u_span()` (lines 145-170)

**Before:**
```python
def compute_wall_u_span(wall_width_m: float) -> float:
    rounded_width = round(wall_width_m)  # Round to nearest 1m
    if rounded_width < 1:
        rounded_width = 1
    return rounded_width / WALL_MODULE_M  # WALL_MODULE_M = 3.0
```

**After:**
```python
def compute_wall_u_span(wall_width_m: float) -> float:
    import math
    # Round UP to nearest 3m multiple
    rounded_width = math.ceil(wall_width_m / WALL_BLOCK_METERS) * WALL_BLOCK_METERS
    if rounded_width < WALL_MIN_METERS:
        rounded_width = WALL_MIN_METERS
    return (rounded_width / WALL_BLOCK_METERS) * WALL_BLOCK_U
```

#### 2.3 Added `compute_wall_u_range()` (lines 173-190)

New helper function that returns (u_start, u_end) with the offset applied:
```python
def compute_wall_u_range(wall_width_m: float) -> Tuple[float, float]:
    u_span = compute_wall_u_span(wall_width_m)
    u_start = WALL_U_OFFSET  # 0.3333...
    u_end = u_start + u_span
    return u_start, u_end
```

#### 2.4 Updated All Wall UV Functions

The following functions were updated to use `compute_wall_u_range()` instead of starting at U=0.0:

| Function | Lines | Change |
|----------|-------|--------|
| `compute_wall_quad_uvs()` | 221-254 | Uses `u_start`, `u_end` from `compute_wall_u_range()` |
| `compute_gable_triangle_uvs()` | 257-288 | Uses `u_start`, `u_end`, `u_center` |
| `compute_pentagon_wall_uvs()` | 291-363 | Uses `u_start`, `u_end`, `u_center` |
| `compute_multi_floor_wall_uvs()` | 370-423 | Uses `u_start`, `u_end` |
| `compute_sidewall_continuous_uvs()` | 426-506 | Uses `u_start`, `u_end` |

---

## Rounding Rule Details

### Before (round to 1m)
```
5.2m → round(5.2) = 5m → 5/3 = 1.667 U span
6.6m → round(6.6) = 7m → 7/3 = 2.333 U span
```

### After (ceil to 3m multiple)
```
2.0m  → ceil(2.0/3)*3 = 3.0m  → (3/3) * 0.33 = 0.33 U span
10.6m → ceil(10.6/3)*3 = 12.0m → (12/3) * 0.33 = 1.33 U span
1.0m  → ceil(1.0/3)*3 = 3.0m  → (3/3) * 0.33 = 0.33 U span
```

The minimum clamp ensures walls < 3m still get a valid UV mapping.

---

## Test Cases (All Passed)

| Wall Length | Rounded To | U Start | U End | Expected U End |
|-------------|-----------|---------|-------|----------------|
| 2.0m | 3.0m | 0.3333 | 0.6667 | 0.6667 |
| 10.6m | 12.0m | 0.3333 | 1.6667 | 1.6667 |
| 1.0m | 3.0m | 0.3333 | 0.6667 | 0.6667 |
| 3.0m | 3.0m | 0.3333 | 0.6667 | 0.6667 |

---

## Verification

### Pipeline Test
```
python -m condor_buildings.main --patch-dir test_data --patch-id 036019 --output-dir output
```
Result: Success, 5416 buildings processed.

### UV Verification in OBJ

Sample UV coordinates from generated file:
```
vt 0.333333 0.125000    # Wall starts at U=0.33, not U=0.0
vt 4.000000 0.125000    # Wall ends at U=4.0 (large wall)
vt 4.000000 0.145833
vt 0.333333 0.145833
```

Confirms: All walls now start at U ≈ 0.333333 instead of U = 0.0.

### UV Analysis

Example from output: `u_end = 4.0, u_start = 0.333333`
- `u_span = 4.0 - 0.333333 = 3.6667`
- `blocks = 3.6667 / 0.3333 = 11 blocks`
- `wall_length_rounded = 11 * 3m = 33m`

---

## Visual Impact

### Doors
- **Before:** Doors appeared at U=0.0-0.33 on every wall tile
- **After:** Doors only appear when U wraps past 1.0 and re-enters the 0.0-0.33 range

### Windows
- **Before:** Window sections at U=0.33-0.66 and U=0.66-1.0
- **After:** Walls start in window section (U=0.33), doors appear less frequently

### Texture Scale
- **Before:** 1m of wall = 0.33 U (dense details)
- **After:** 3m of wall = 0.33 U (larger, more realistic features)

---

## Files Modified

| File | Changes |
|------|---------|
| `condor_buildings/config.py` | Added 4 new constants |
| `condor_buildings/generators/uv_mapping.py` | Updated imports, rewrote `compute_wall_u_span()`, added `compute_wall_u_range()`, updated 5 wall UV functions |

---

## Files NOT Modified

- `walls.py` - No changes needed (uses uv_mapping.py functions)
- `roof_gabled.py` - Roof UV mapping unchanged
- `roof_flat.py` - Roof UV mapping unchanged
- V direction mapping - Unchanged

---

## Acceptance Checklist

| # | Criterion | Status |
|---|-----------|--------|
| 1 | New scale: 3m = 0.33 U | PASS |
| 2 | U offset = 0.33 (skip door section) | PASS |
| 3 | Rounding: ceil() to 3m multiples | PASS |
| 4 | Minimum wall length clamp: 3.0m | PASS |
| 5 | All wall UV functions updated | PASS |
| 6 | V mapping unchanged | PASS |
| 7 | Roof UV mapping unchanged | PASS |
| 8 | Test cases pass | PASS |
| 9 | Pipeline runs successfully | PASS |
| 10 | UV values verified in OBJ | PASS |

---

## Output Files

Generated in `output/`:
- `o036019_LOD0.obj`
- `o036019_LOD1.obj`
- `o036019_report.json`
- `o036019.log`

---

## Session Notes

This update implements Wiek's new UV mapping specification to make facade features (doors/windows) visually larger and more realistic, while reducing the frequency of doors appearing on walls.

The key insight is that starting UV mapping at U=0.33 (instead of U=0.0) skips the door section of the texture by default. Doors will only appear when the U coordinate wraps around past 1.0, which happens less frequently with the new 3m block scale.

Total implementation time: ~45 minutes (including planning, implementation, and verification).
