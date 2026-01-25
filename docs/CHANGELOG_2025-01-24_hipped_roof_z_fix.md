# Changelog: Hipped Roof Z Positioning Fix

**Date:** 2025-01-24
**Session:** Fix hipped roof floating above walls + parameter tuning
**Version:** 0.3.5 -> 0.3.6

---

## Summary

Fixed a geometry issue where hipped roofs were "floating" above the walls (gap between wall top and roof underside). Also adjusted house-scale parameters to allow more buildings to receive pitched roofs.

---

## Problem Statement

### Issue Reported by Wiek

The hipped roof was hovering above the walls:
- The bottom of the roof overhang was at the same Z position as the top of the wall
- This created a visible gap between wall top and roof underside
- Same issue that was previously fixed for gabled roofs

### Expected Behavior

- The roof slope plane should pass through `wall_top_z` at the original footprint boundary
- When there is overhang, the eave corners should be BELOW `wall_top_z` due to the slope
- Wall height must remain unchanged (3m per floor)

---

## Root Cause Analysis

In the original `roof_hipped.py`, the eave vertices were placed at `eave_z = wall_top_z`:

```python
# OLD CODE (incorrect)
eave_z = building.wall_top_z
# ... expand footprint for overhang ...
for vx, vy in verts:
    v_indices.append(mesh.add_vertex(vx, vy, eave_z))  # All at wall_top_z!
```

This meant that even when the footprint was expanded for overhang, the eave vertices were still at wall height. The roof appeared to "hover" because the overhang edge should have been lower following the roof slope.

### Comparison with Gabled Roofs

Gabled roofs correctly calculate corner Z using the slope:

```python
# GABLED (correct)
corner_z = ridge_z - slope * roof_half_width
```

This makes the overhang corners naturally lower than `wall_top_z`.

---

## Solution Implemented

### 1. Calculate `tan_pitch` from Original Footprint

```python
# Compute slope using ORIGINAL footprint geometry
original_distances = original_edge_geom.distances
max_dist = max(original_distances)
tan_pitch = roof_height / max_dist if max_dist > 0.01 else 1.0
```

### 2. Calculate `eave_z` Based on Overhang

```python
if config.overhang > 0:
    # Eave corners are BELOW wall_top_z due to slope
    eave_z = wall_top_z - tan_pitch * config.overhang
    verts_2d = _expand_footprint(original_verts, config.overhang)
else:
    eave_z = wall_top_z
    verts_2d = original_verts
```

### 3. Pass `tan_pitch` to Roof Generation

```python
_generate_hipped_roof_quadrangle(
    mesh, verts_2d, edge_geom, eave_z, roof_height, roof_index, tan_pitch
)
```

This ensures consistent slope calculation regardless of overhang.

---

## Verification Results

Test building: 10m x 8m footprint, 6m wall height (2 floors), floor_z = 100m

| Metric | LOD0 (0.5m overhang) | LOD1 (no overhang) |
|--------|---------------------|-------------------|
| Wall top Z | 106.000m | 106.000m |
| Eave Z | 105.700m | 106.000m |
| Drop from wall | 0.300m | 0.000m |
| Ridge Z | 108.400m | 108.400m |

**Results confirm:**
- LOD0: Eave is 0.3m BELOW wall top (correct overhang drop)
- LOD1: Eave is exactly AT wall top (roof touches wall perfectly)
- Ridge height consistent in both LODs

---

## Parameter Tuning

Increased house-scale thresholds by 20% to allow more buildings to receive pitched roofs:

| Parameter | Old Value | New Value | Change |
|-----------|-----------|-----------|--------|
| `HOUSE_MAX_FOOTPRINT_AREA` | 300.0 m² | 360.0 m² | +20% |
| `HOUSE_MAX_SIDE_LENGTH` | 25.0 m | 30.0 m | +20% |
| `HOUSE_MIN_SIDE_LENGTH` | 4.0 m | 3.2 m | -20% |
| `HOUSE_MAX_ASPECT_RATIO` | 4.0 | 4.8 | +20% |

---

## Test Results (Patch 036019)

### With `--random-hipped` (geometry mode)

| Metric | Value |
|--------|-------|
| Buildings processed | 5,416 |
| **Gabled roofs** | **1,065** |
| **Hipped roofs** | **900** |
| **Flat roofs** | **3,451** |
| Total pitched roofs | 1,965 (36.3%) |
| Gabled fallbacks | 2,754 |
| Hipped fallbacks | 154 |

### Fallback Distribution

| Reason | Count |
|--------|-------|
| `too_many_vertices` | 1,434 |
| `too_many_floors` | 1,346 |
| `too_short_side` | 77 |
| `bad_aspect_ratio` | 36 |
| `too_long_side` | 6 |
| `too_elongated` | 3 |
| `not_rectangle_angles` | 2 |
| `too_large_area` | 2 |
| `has_holes` | 1 |
| `not_rectangular_enough` | 1 |

---

## Files Modified

| File | Changes |
|------|---------|
| `condor_buildings/generators/roof_hipped.py` | Fixed eave_z calculation, added tan_pitch parameter |
| `condor_buildings/config.py` | Increased house-scale thresholds by 20% |

---

## Technical Details

### Updated Function Signature

```python
def _generate_hipped_roof_quadrangle(
    mesh: MeshData,
    verts: List[Tuple[float, float]],
    edge_geom: EdgeGeometry,
    eave_z: float,
    roof_height: float,
    roof_index: int,
    tan_pitch: float  # NEW: passed from caller
) -> None:
```

### Updated Debug Logging

```python
logger.info(
    f"HIPPED DEBUG [{building.osm_id}]: "
    f"wall_top_z={wall_top_z:.2f}m, "
    f"eave_z={eave_z:.2f}m (drop={eave_drop:.2f}m), "
    f"ridge_z={ridge_z:.2f}m, "
    f"tan_pitch={tan_pitch:.3f}, "
    ...
)
```

---

## Geometry Explanation

```
SIDE VIEW (with overhang):

        ridge_z ─────────────────── ridge
                    /\
                   /  \
                  /    \
    wall_top_z ──┼──────┼── wall top (roof plane passes through here)
                /        \
     eave_z ───/──────────\─── eave (lower due to slope)
              │            │
              │   WALL     │
              │            │
    floor_z ──┴────────────┴── ground

The overhang extends beyond the wall, and the eave_z is calculated as:
    eave_z = wall_top_z - tan_pitch * overhang
```

---

## Future Considerations

### Straight Skeleton for Complex Footprints

The current hipped roof algorithm only works for 4-vertex quadrilaterals. Supporting 5+ vertex polygons would require implementing a straight skeleton algorithm. This is feasible but complex, especially for non-convex polygons (L-shaped, T-shaped buildings).

Current fallback statistics show 1,434 buildings rejected due to `too_many_vertices`, which could potentially receive proper hipped/gabled roofs with straight skeleton support.

---

## Session Notes

- Fix was similar to the gabled roof floating fix implemented earlier
- The key insight is that `eave_z` must account for the slope when overhang is present
- Parameter tuning was requested to increase the number of pitched roofs for visual variety
