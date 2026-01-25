# Changelog: Gabled Roof Fixes

**Date:** 2025-01-11
**Session:** Gabled roof geometry and size-gate improvements
**Version:** 0.2.0 -> 0.2.1

---

## Summary

This session addressed two issues with gabled roof generation:
1. **Ridge direction bug** - Roofs appeared too tall because the ridge was aligned with the wrong axis
2. **Large buildings getting gabled roofs** - Rectangular apartments/industrial buildings were incorrectly receiving house-style gabled roofs

---

## Fix 1: Ridge Direction Correction

### Problem

In Blender, gabled roofs appeared disproportionately tall. Investigation revealed that the ridge was being aligned **perpendicular** to the longest edge instead of **parallel** to it.

For a rectangular building:
- **Before (wrong):** Ridge perpendicular to long edge → span = long side → tall roof
- **After (correct):** Ridge parallel to long edge → span = short side → natural height

### Root Cause

In `_get_ridge_direction()`, the code was adding 90 degrees to the longest edge direction:

```python
# WRONG - was perpendicular to longest edge
ridge_direction = (eave_direction + 90.0) % 180.0
```

### Solution

Changed to use the longest edge direction directly:

```python
# CORRECT - parallel to longest edge
ridge_direction = compute_longest_edge_axis(ring)
```

### Files Modified

| File | Change |
|------|--------|
| `generators/roof_gabled.py` | Fixed `_get_ridge_direction()` to use `compute_longest_edge_axis()` directly |
| `config.py` | Added `DEBUG_GABLED_ROOFS` flag for per-building debug logging |

### Debug Logging

When `DEBUG_GABLED_ROOFS = True` in config, each gabled building logs:
```
GABLED DEBUG [osm_id]: vertices=4, OBB_length=15.8m (along ridge),
OBB_width=10.2m (span), ridge_dir=45.0° (longest_axis_heuristic),
pitch=42.3°, ridge_height=4.63m, wall_height=6.0m, total_height=10.6m
```

---

## Fix 2: House-Scale Size Gate for Gabled Roofs

### Problem

Large rectangular buildings (apartments, industrial, office) were receiving gabled roofs because they passed the geometry check (4 vertices, convex, rectangular). Visually, these buildings should have flat roofs.

### Solution

Added a **house-scale** size validation that runs after the geometry check. Buildings must pass BOTH:
1. Gabled eligibility (geometry): 4 vertices, convex, rectangular, no holes
2. House-scale (size): area, side lengths, and aspect ratio within house limits

### New Configuration Parameters

Added to `config.py`:

```python
# Maximum footprint area for house classification (m²)
HOUSE_MAX_FOOTPRINT_AREA = 300.0

# Maximum side length for house classification (m)
HOUSE_MAX_SIDE_LENGTH = 25.0

# Minimum side length for house classification (m)
# Below this = shed/garage -> flat roof
HOUSE_MIN_SIDE_LENGTH = 4.0

# Maximum aspect ratio for house classification
HOUSE_MAX_ASPECT_RATIO = 4.0
```

### New CLI Arguments

```bash
python -m condor_buildings.main --patch-dir . --patch-id 036019 \
    --house-max-area 300 \
    --house-max-side 25 \
    --house-min-side 4 \
    --house-max-aspect 4
```

### New Fallback Reasons

| Reason | Description |
|--------|-------------|
| `too_large_area` | Footprint area > max (apartments/industrial) |
| `too_long_side` | Side length > max (long buildings) |
| `too_short_side` | Side length < min (sheds/garages) |
| `too_elongated` | Aspect ratio > max (row houses/industrial) |

### Files Modified

| File | Change |
|------|--------|
| `config.py` | Added `HOUSE_*` constants and `PipelineConfig` fields |
| `models/building.py` | Added size-based `RoofFallbackReason` values |
| `processing/footprint.py` | Added `check_house_scale()` function, new `GabledEligibility` values, `FootprintAnalysis.is_house_scale` field |
| `generators/building_generator.py` | Updated `_generate_roof()` to check both geometry AND size |
| `main.py` | Added CLI arguments, stats tracking, enhanced report output |

### Report Output

The `report.json` now includes:
```json
{
  "stats": {
    "gabled_eligible": 2486,
    "house_scale_pass": 4582,
    "house_scale_fail": 834,
    "gabled_roofs": 2085,
    "flat_roofs": 3331
  },
  "fallback_reasons": {
    "too_many_vertices": 2462,
    "too_short_side": 209,
    "too_long_side": 32,
    "too_elongated": 6,
    "too_large_area": 4
  },
  "config_used": {
    "house_max_footprint_area": 300.0,
    "house_max_side_length": 25.0,
    "house_min_side_length": 4.0,
    "house_max_aspect_ratio": 4.0
  }
}
```

---

## Test Results (Patch 036019)

### Before This Session
- 2336 gabled roofs
- 3080 flat roofs

### After Ridge Fix Only
- 2336 gabled roofs (same count, but correct proportions)

### After House-Scale Gate
- **2085 gabled roofs** (-251 from ridge fix)
- **3331 flat roofs** (+251)
- 251 rectangular buildings now correctly get flat roofs

### Fallback Breakdown
| Reason | Count |
|--------|-------|
| `too_many_vertices` | 2462 |
| `too_short_side` | 209 |
| `bad_aspect_ratio` | 36 |
| `too_long_side` | 32 |
| `too_elongated` | 6 |
| `too_large_area` | 4 |
| `not_rectangle_angles` | 2 |
| `has_holes` | 1 |
| `not_rectangular_enough` | 1 |

---

## Usage Examples

### Standard run (default house-scale thresholds)
```bash
python -m condor_buildings.main --patch-dir . --patch-id 036019
```

### More permissive (larger houses allowed)
```bash
python -m condor_buildings.main --patch-dir . --patch-id 036019 \
    --house-max-area 400 \
    --house-max-side 30
```

### More restrictive (smaller houses only)
```bash
python -m condor_buildings.main --patch-dir . --patch-id 036019 \
    --house-max-area 200 \
    --house-max-side 20
```

### Debug single building
```bash
python -m condor_buildings.main --patch-dir . --patch-id 036019 \
    --debug-osm-id 306462456
```

---

## Document Naming Convention

For changelog documents created after successful sessions:

```
CHANGELOG_YYYY-MM-DD_<brief_description>.md
```

Examples:
- `CHANGELOG_2025-01-11_gabled_fixes.md`
- `CHANGELOG_2025-01-15_uv_mapping.md`
- `CHANGELOG_2025-01-20_texture_atlas.md`

This allows:
1. Chronological sorting by date prefix
2. Quick identification of changes by description
3. Clear distinction from design docs (like `ITERATION_A_GABLED_ROOFS.md`)
