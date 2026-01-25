# Changelog 2025-01-23: Hipped Roofs Implementation

## Summary

Implemented hipped (four-slope) roofs for quadrilateral buildings based on BLOSM's analytical solution. Also fixed a floating roof bug affecting buildings with non-standard floor heights.

## New Features

### Hipped Roof Generation (`roof_hipped.py`)

New module implementing hipped roofs for simple 4-vertex quadrilateral buildings:

- **Algorithm**: Adapted from BLOSM's analytical solution for quadrilaterals
  - Computes edge geometry (vectors, lengths, angles)
  - Calculates "edge event" distances where bisectors meet
  - Finds ridge endpoints from minimum distance edges
  - Creates 4 roof faces: 2 triangular hips + 2 trapezoidal sides

- **Special case**: Square footprints generate pyramidal roofs (single apex point)

- **Configuration**:
  - `HIPPED_HEIGHT_FIXED = 3.0m` (same as gabled for visual consistency)
  - `HIPPED_MAX_FLOORS = 2` (same restriction as gabled)
  - `DEBUG_HIPPED_ROOFS = False` (debug logging flag)

- **Eligibility**: Same constraints as gabled roofs
  - 4 vertices only
  - Strictly convex
  - No holes
  - House-scale (area < 300m², sides 4-25m, aspect ratio < 4:1)

### Roof Type Selection

- Hipped roofs are assigned **only by OSM tag**: `roof:shape=hipped`
- Gabled remains the default for eligible houses
- No automatic selection between gabled/hipped (predictable, data-driven)

### Statistics and Testing

- Added `hipped_roofs` and `hipped_fallbacks` counters to pipeline stats
- Added `--random-hipped` CLI flag for visual testing
  - Randomly assigns hipped to 50% of eligible buildings
  - Uses deterministic seed per building for reproducibility

## Bug Fixes

### Floating Roofs Fix (`walls.py`)

**Problem**: Some buildings had roofs floating above the walls.

**Root Cause**: `_generate_ring_walls()` generated walls floor-by-floor using fixed `DEFAULT_FLOOR_HEIGHT = 3.0m`, but roofs were placed at `wall_top_z = floor_z + height_m`. If `height_m` was not exactly `floors * 3.0m`, walls didn't reach the roof.

**Solution**: Calculate actual floor height dynamically:
```python
total_wall_height = top_z - floor_z
actual_floor_height = total_wall_height / building_floors
```

This ensures walls always reach exactly `top_z`, regardless of whether `height_m` is a multiple of 3.0m.

## Files Changed

| File | Changes |
|------|---------|
| `condor_buildings/generators/roof_hipped.py` | **NEW** - Hipped roof generation |
| `condor_buildings/generators/building_generator.py` | Added HIPPED case in roof type determination and generation |
| `condor_buildings/generators/walls.py` | Added `generate_walls_for_hipped()`, fixed floating roof bug |
| `condor_buildings/config.py` | Added `HIPPED_HEIGHT_FIXED`, `DEBUG_HIPPED_ROOFS`, `random_hipped` |
| `condor_buildings/main.py` | Added hipped stats, `--random-hipped` CLI argument |

## Algorithm Details (from BLOSM)

### Edge Geometry Computation
```
For each edge i:
  vector[i] = vertex[i+1] - vertex[i]
  length[i] = |vector[i]|
  cos[i] = -dot(vector[i], vector[i-1]) / (length[i] * length[i-1])
  sin[i] = -cross(vector[i], vector[i-1]).z / (length[i] * length[i-1])
  distance[i] = length[i] / ((1+cos[i])/sin[i] + (1+cos[i+1])/sin[i+1])
```

### Ridge Calculation
- Find two edges with minimum distance (opposite edges)
- Ridge endpoints are where bisectors from these edges meet
- Ridge vertex position computed by moving inward from edge by `distance[i]`

### Roof Faces
- 2 triangular "hip" faces at the short ends
- 2 trapezoidal faces along the long sides
- All faces duplicated with reversed winding for double-sided rendering

## Test Results

With `--random-hipped` on test patch 036019:
- Total buildings: 5416
- Gabled roofs: 910
- Hipped roofs: 909
- Flat roofs: 3597

Visual verification confirmed:
- Hipped roofs render correctly with proper 4-slope geometry
- Pyramidal roofs work for square footprints
- Overhang works correctly (LOD0 has 0.5m overhang)
- No more floating roofs on complex structures
