# Iteration A: Robust Gabled Roofs

**Date:** January 2025
**Version:** 0.1.0 -> 0.2.0

## Objective

Make gabled roofs robust by restricting them to "safe" footprints (rectangles), add clear fallback reasons in report.json, and prepare for Phase B (UV mapping + texture atlases).

## Key Design Choice

For Milestone A, gabled roofs are generated **only for footprints with exactly 4 vertices** (rectangles) by default. This is configurable via `GABLED_MAX_VERTICES` for future experimentation with 6/8 vertex shapes.

### Why This Is Acceptable

1. **Eliminates almost all roof-mesh failures** - No more self-intersecting roofs
2. **Matches Condor house style** - Simple rectangular footprints are the norm
3. **Sets up for UV/atlas improvements** - Which will deliver the biggest visual gain

## Changes Implemented

### A) Tightened Gabled Eligibility Gate

**Files:** `config.py`, `processing/footprint.py`

New configuration constants:
```python
GABLED_MAX_VERTICES = 4          # Only rectangles (was 20)
GABLED_REQUIRE_CONVEX = True     # Must be strictly convex
GABLED_REQUIRE_NO_HOLES = True   # No inner rings allowed
GABLED_MIN_RECTANGULARITY = 0.70 # Area/OBB ratio threshold
GABLED_ANGLE_TOLERANCE_DEG = 25.0 # Tolerance from 90 degrees
```

New helper functions:
- `get_unique_vertex_count(ring)` - Counts vertices excluding closing point
- `is_strictly_convex(ring)` - Checks all cross products have same sign
- `is_rectangle_like(ring)` - Verifies angles are close to 90 degrees
- `check_gabled_eligibility_strict()` - Comprehensive eligibility check

**Critical Fix:** Changed OBB calculation to use `compute_longest_edge_axis()` instead of `compute_longest_axis()`. The old method used the diagonal (farthest points), causing rectangles to have only 50% rectangularity. The new method aligns the OBB with the actual edges.

### B) Explicit Fallback Reasons

**Files:** `models/building.py`, `generators/building_generator.py`

New enum `RoofFallbackReason`:
```python
class RoofFallbackReason(Enum):
    NONE = "none"
    HAS_HOLES = "has_holes"
    TOO_MANY_VERTICES = "too_many_vertices"
    NOT_CONVEX = "not_convex"
    NOT_CONVEX_ENOUGH = "not_convex_enough"
    NOT_RECTANGULAR_ENOUGH = "not_rectangular_enough"
    BAD_ASPECT_RATIO = "bad_aspect_ratio"
    NOT_RECTANGLE_ANGLES = "not_rectangle_angles"
    HIPPED_NOT_SUPPORTED = "hipped_not_supported"
    DEGENERATE = "degenerate"
```

New fields in `BuildingRecord`:
- `roof_fallback_reason: RoofFallbackReason`
- `footprint_vertex_count: int`

New fields in `BuildingGeneratorResult`:
- `fallback_reason: RoofFallbackReason`
- `footprint_analysis: FootprintAnalysis`

### C) Rectangle-Focused Gabled Generation

**File:** `generators/roof_gabled.py`

The existing OBB-based generator already works optimally for rectangles. Updated documentation to clarify the design assumptions:
- Footprint has exactly 4 vertices (rectangle)
- Footprint is convex and has no holes
- This eliminates all roof-mesh self-intersection issues

### D) UV/Texture Groundwork (Phase B Preparation)

**File:** `models/mesh.py`

Added UV coordinate support:
```python
@dataclass
class MeshData:
    vertices: List[Tuple[float, float, float]]
    uvs: List[Tuple[float, float]]           # NEW
    faces: List[List[int]]
    face_uvs: List[List[int]]                # NEW
```

New methods:
- `add_uv(u, v)` - Add UV coordinate
- `add_vertex_with_uv(x, y, z, u, v)` - Add vertex and UV together
- `add_face_with_uvs(vertex_indices, uv_indices)`
- `add_triangle_with_uvs(v1, v2, v3, uv1, uv2, uv3)`
- `add_quad_with_uvs(v1, v2, v3, v4, uv1, uv2, uv3, uv4)`

UV Strategy documented:
- Walls use a "strip atlas": tileable in U, variants stacked in V
- U coordinate can exceed [0,1] for wrapping without atlas jumps
- V coordinate selects texture variant in the atlas

### E) Config Changes

**File:** `config.py`

`PipelineConfig` now includes:
```python
gabled_max_vertices: int = GABLED_MAX_VERTICES
gabled_require_convex: bool = GABLED_REQUIRE_CONVEX
gabled_require_no_holes: bool = GABLED_REQUIRE_NO_HOLES
gabled_min_rectangularity: float = GABLED_MIN_RECTANGULARITY
debug_osm_id: Optional[str] = None  # For single-building debugging
```

### F) Testing/Validation

**File:** `main.py`

New CLI arguments:
- `--gabled-max-vertices <n>` - Override max vertices (default 4)
- `--gabled-allow-non-convex` - Allow non-convex footprints
- `--debug-osm-id <id>` - Process only a single building by OSM ID

Enhanced `report.json`:
```json
{
  "vertex_count_stats": {
    "4_vertices": 2529,
    "5_to_6_vertices": 1181,
    "7_to_8_vertices": 841,
    "9_plus_vertices": 865
  },
  "fallback_reasons": {
    "too_many_vertices": 2462,
    "bad_aspect_ratio": 36,
    "not_rectangle_angles": 2,
    "has_holes": 1,
    "not_rectangular_enough": 1
  },
  "config_used": {
    "gabled_max_vertices": 4,
    "gabled_require_convex": true,
    "gabled_require_no_holes": true,
    "gabled_min_rectangularity": 0.7,
    "global_seed": 42,
    "roof_overhang_lod0": 0.5
  }
}
```

## Test Results (Patch 036019)

```
Buildings processed: 5416
LOD0: 181,978 vertices, 104,886 faces
LOD1: 181,978 vertices, 104,886 faces

Footprint vertex distribution:
  4 vertices (rectangles): 2529 (46.7%)
  5-6 vertices: 1181 (21.8%)
  7-8 vertices: 841 (15.5%)
  9+ vertices: 865 (16.0%)

Roof types:
  Gabled eligible: 2486 (45.9%)
  Actual gabled: 2336 (43.1%)
  Flat roofs: 3080 (56.9%)
  Gabled->Flat fallbacks: 2502

Fallback reasons:
  too_many_vertices: 2462
  bad_aspect_ratio: 36
  not_rectangle_angles: 2
  has_holes: 1
  not_rectangular_enough: 1

Processing time: ~3.2 seconds
```

## Files Modified

| File | Changes |
|------|---------|
| `config.py` | New gabled eligibility constants, PipelineConfig fields |
| `processing/footprint.py` | New helpers, strict eligibility check, OBB fix |
| `models/building.py` | RoofFallbackReason enum, new BuildingRecord fields |
| `models/mesh.py` | UV support (uvs, face_uvs, new methods) |
| `generators/building_generator.py` | Fallback reason tracking, updated imports |
| `generators/roof_gabled.py` | Updated documentation |
| `main.py` | New CLI args, enhanced report, debug mode |

## Usage Examples

### Standard run (rectangles only)
```bash
python -m condor_buildings.main --patch-dir ./CLT3 --patch-id 036019
```

### Allow more complex shapes
```bash
python -m condor_buildings.main --patch-dir ./CLT3 --patch-id 036019 --gabled-max-vertices 6
```

### Debug single building
```bash
python -m condor_buildings.main --patch-dir ./CLT3 --patch-id 036019 --debug-osm-id 192558470
```

## Next Steps (Phase B)

1. **UV Mapping Implementation**
   - Generate UVs for walls based on wall dimensions
   - Generate UVs for roof faces
   - Support strip atlas texture format

2. **Texture Atlas System**
   - Create facade atlas with tiled variants
   - Create roof texture atlas
   - Material assignment based on building category

3. **OBJ Export Enhancement**
   - Export `vt` (texture coordinates) lines
   - Export `f v/vt` face format
   - MTL file generation with texture references
