# Changelog: Geometry v4 - Pentagonal Gable Walls

**Date:** 2025-01-16
**Session:** Gabled Roof Architecture Redesign per Wiek's Specification
**Version:** 0.2.3 -> 0.2.4

---

## Summary

Major architectural redesign of gabled building geometry to match Wiek's reference model exactly:

1. **Walls include gable** - Pentagonal walls extend from floor to ridge (not separate pieces)
2. **Roof is just slopes** - Only 2 slope planes, open at gable ends
3. **Single pentagon face** - Gable walls as one n-gon (better for UV mapping)
4. **Cleaner geometry** - Walls and roof are independent bodies

---

## Problem Statement

In v0.2.3, gabled buildings had:
- Rectangular walls (floor_z to wall_top_z)
- Triangular gable walls on top of rectangular walls (wall_top_z to ridge_z)
- Roof with gable end caps closing the triangles

This created a "fused" appearance where the roof seemed merged with the walls.

**Wiek's specification:**
> "La base, la parte rectangular es un cuerpo, y el techo es otro cuerpo independiente"
> (The base rectangular part is one body, and the roof is another independent body)

The walls should be pentagonal (single solid from floor to ridge), and the roof should be just 2 floating slope planes.

---

## Changes Implemented

### A) New Pentagonal Wall Generation

**File:** `condor_buildings/generators/walls.py`

Added new function `generate_walls_for_gabled()` that creates:
- **Gable end walls** (perpendicular to ridge): Pentagonal faces from floor_z to ridge_z
- **Side walls** (parallel to ridge): Rectangular faces from floor_z to wall_top_z

```python
def generate_walls_for_gabled(
    building: BuildingRecord,
    ridge_direction_deg: float,
    ridge_z: float,
    obb_center: Tuple[float, float],
    config: Optional[WallGeneratorConfig] = None
) -> MeshData:
```

**Pentagon vertices (CCW order):**
```
        v3 (ridge apex)
        /\
       /  \
      /    \
   v4--------v2  (wall_top_z / eave)
    |        |
    |        |
   v0--------v1  (floor_z)
```

### B) Pentagon as Single N-gon Face

Instead of triangulating the pentagon into 3 triangles, we now generate it as a single 5-vertex polygon face.

**Benefits:**
- Better for UV mapping (single continuous face)
- Cleaner geometry (no internal edges)
- Fewer faces in the mesh

**New method in `MeshData`:**
```python
def add_polygon(self, *vertices: int) -> None:
    """Add an n-gon face (polygon with any number of vertices)."""
    self.faces.append(list(vertices))
```

### C) Simplified Roof Generation

**File:** `condor_buildings/generators/roof_gabled.py`

Removed gable end caps from roof - the roof is now just 2 slope quads:
- Right slope: `c1, c0, r0, r1`
- Left slope: `c3, r0, r1, c2`

The roof is "open" at the gable ends because the pentagonal walls now close the building visually.

```python
# NOTE: Gable end triangles removed in v0.2.4
# The roof is now "open" at the gable ends - the pentagonal walls
# in walls.py close the building visually
```

### D) Building Generator Orchestration

**File:** `condor_buildings/generators/building_generator.py`

New function `_determine_roof_type()` determines the actual roof type BEFORE generating walls:

```python
def _determine_roof_type(building, overhang, result, ...)
    -> Tuple[RoofType, Optional[Tuple[ridge_dir, ridge_z, obb_center]]]:
```

This allows:
1. Check if building will have gabled roof
2. Calculate ridge parameters (direction, height, center)
3. Pass parameters to `generate_walls_for_gabled()`
4. Generate appropriate wall geometry before roof

---

## Files Modified

| File | Changes |
|------|---------|
| `models/mesh.py` | Added `add_polygon()` and `add_polygon_with_uvs()` methods for n-gon faces |
| `generators/walls.py` | Added `generate_walls_for_gabled()` and `_generate_pentagonal_gable_wall()` |
| `generators/roof_gabled.py` | Removed gable end caps, updated to Geometry v4, deprecated `include_gable_walls` |
| `generators/building_generator.py` | Added `_determine_roof_type()`, modified LOD0/LOD1 to use pentagonal walls for gabled buildings |

---

## Test Results (Patch 036019)

### Statistics
```
Buildings processed: 5416
Gabled roofs: 2085
Flat roofs: 3331
Processing time: ~3.6 seconds
```

### Face Distribution
```
Triangles (3 vertices): 99,726
Pentagons (5 vertices): 4,163
```

The ~4,163 pentagons correspond to gable walls (2 per gabled building × 2,085 buildings).

### Vertex/Face Count
| Metric | v0.2.3 | v0.2.4 | Change |
|--------|--------|--------|--------|
| LOD0 vertices | 193,986 | 185,646 | -8,340 |
| LOD0 faces | 120,562 | 112,222 | -8,340 |

Reduction due to:
- Pentagon as 1 face instead of 3 triangles (-2 faces per gable wall)
- Removed gable end caps from roof (-2 faces per building)

---

## Geometry Comparison

### v0.2.3 (Before)
```
Building structure:
- Rectangular walls (floor → wall_top)
- Triangular gable walls (wall_top → ridge)
- Roof with gable end caps (closed triangles)

Problem: Walls and roof "fused" visually
```

### v0.2.4 (After)
```
Building structure:
- Pentagonal walls at gable ends (floor → ridge)
- Rectangular walls at sides (floor → wall_top)
- Roof = 2 slope planes only (open at ends)

Result: Clean separation of wall body and roof body
```

---

## Visual Verification

### Reference Model
`reference-house/Correct house.obj` (Wiek's specification):
- 18 faces total
- 10 wall triangles (including pentagonal gable walls triangulated)
- 8 roof triangles (4 top + 4 bottom for double-sided)
- Gable walls extend from floor to ridge as single body
- Roof is independent floating planes

### Generated Output
`output/o036019_LOD0.obj`:
- Pentagonal gable walls visible in Blender
- Roof slopes floating independently
- Overhang clearly visible
- Matches reference geometry

---

## Edge Detection Logic

To determine which walls are gable ends vs side walls:

```python
# dot product of edge direction and ridge direction
dot = abs(edge_dx_n * ridge_dx + edge_dy_n * ridge_dy)

if dot < 0.3:
    # GABLE END: Edge perpendicular to ridge
    # Generate pentagonal wall (floor → ridge)
else:
    # SIDE WALL: Edge parallel to ridge
    # Generate rectangular wall (floor → wall_top)
```

- `dot ≈ 0`: Edge perpendicular to ridge (gable end, short wall)
- `dot ≈ 1`: Edge parallel to ridge (side wall, long wall under slope)

---

## Migration Notes

### API Changes
- `GabledRoofConfig.include_gable_walls` is now deprecated (default `False`)
- New function `generate_walls_for_gabled()` in `walls.py`
- New function `_determine_roof_type()` in `building_generator.py`
- New method `add_polygon()` in `MeshData`

### Backward Compatibility
- Flat roof buildings unchanged
- LOD1 uses same pentagonal wall logic
- OBJ export supports n-gon faces natively

---

## Next Steps

1. **UV Mapping** - Texture coordinates for pentagonal walls
2. **Materials** - MTL file with wall/roof materials
3. **Texture Atlas** - Wall texture variants

---

## Session Notes

This session resolved the final geometry issue from Wiek's review. The buildings now match the reference model with:
- Clean pentagonal gable walls as single faces
- Independent roof slope planes
- Proper architectural separation of wall and roof bodies

Total implementation time: ~2 hours including debugging and iteration.
