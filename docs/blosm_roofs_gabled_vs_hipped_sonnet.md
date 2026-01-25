# BLOSM Roof Construction Analysis: Gabled vs Hipped Roofs
## Investigation Report for Condor Buildings Pipeline Integration

**Date:** 2025-01-14  
**Purpose:** Research-only investigation of BLOSM roof generation (gabled/hipped) for planning Condor 3 pipeline integration  
**Scope:** No code modifications; analysis and recommendations only

---

## 1. Executive Summary

### Key Findings

1. **Two Implementation Paths**: BLOSM has two building systems:
   - `building2/` (newer, action/volume-based) - primary implementation analyzed
   - `building/` (older, legacy) - mostly superseded

2. **Gabled Roofs**: Implemented via **RoofProfile** class using a profile-based approach
   - Profile defined as 2D shape: `[(0., 0.), (0.5, 1.), (1., 0.)]` (triangle)
   - Swept along building orientation (perpendicular to longest edge by default)
   - Complex slot-based algorithm for arbitrary footprints
   - Handles non-rectangular, concave, and multi-polygon cases

3. **Hipped Roofs**: Two distinct implementations:
   - **Simple quadrangles**: Analytical solution for 4-sided convex polygons
   - **Complex polygons**: Uses **straight skeleton** algorithm (bpypolyskel library)

4. **Straight Skeleton**: Core algorithm for hipped roofs (non-quadrangle cases)
   - Computes medial axis (skeleton) of polygon by propagating edges inward
   - Creates roof faces from skeleton edges
   - Supports holes, concave polygons, multipolygons
   - Computationally expensive (iterative event-driven algorithm)

5. **Decision Logic**: Roof shape determined by:
   - OSM tag `roof:shape` (if present)
   - Style system fallback (default: flat for most categories)
   - No automatic heuristic to choose gabled vs hipped based on footprint geometry

6. **Integration Challenges**:
   - Profile-based gabled approach is over-engineered for simple rectangles
   - Straight skeleton is heavy for real-time/batch generation
   - Our pipeline focuses on house-scale rectangles; can use simpler geometry
   - No direct OBB or PCA usage in BLOSM (uses edge iteration)

---

## 2. Code Map

### Key Files and Entry Points

```
blosm-release/
├── action/volume/
│   ├── __init__.py                  # Volume action entry point, roof type dispatcher
│   ├── roof.py                       # Base Roof class (direction calculation, height)
│   ├── roof_gabled.py                # Gabled roof (thin wrapper around RoofProfile)
│   ├── roof_profile.py               # Profile-based roof implementation (1077 lines)
│   ├── roof_hipped.py                # Hipped roof (quadrangle + general polygon)
│   ├── roof_hipped_multi.py          # Hipped roof for multipolygons with holes
│   ├── roof_flat.py                  # Flat roof implementation
│   └── geometry/
│       ├── rectangle.py              # Rectangle geometry helpers
│       └── trapezoid.py              # Trapezoid geometry helpers
├── lib/bpypolyskel/
│   ├── bpypolyskel.py                # Straight skeleton algorithm
│   ├── bpyeuclid.py                  # 2D geometry primitives
│   └── poly2FacesGraph.py            # Polygon to face graph conversion
├── item/
│   ├── roof_profile.py               # Data structure for profile roofs
│   ├── roof_hipped.py                # Data structure for hipped roofs
│   └── roof_hipped_multi.py          # Data structure for hipped multipolygons
└── util/
    └── polygon.py                    # Polygon utilities
```

### Call Chain Diagram

```
Volume.do(building, ...) [action/volume/__init__.py:54]
  │
  ├──> Volume.generateVolume(footprint, coords) [:95]
  │      │
  │      └──> volumeGenerators[roofShape].do(footprint, coords) [:100]
  │            │
  │            ├─ roofShape="gabled"  ──> RoofProfile.do(footprint, coords)
  │            │                             │
  │            │                             └──> RoofProfile.render(footprint, roofItem)
  │            │                                    [roof_profile.py:620]
  │            │                                    • processDirection() - compute ridge direction
  │            │                                    • createProfileVertices() - sweep profile
  │            │                                    • trackUp/trackDown() - form roof faces
  │            │
  │            └─ roofShape="hipped"  ──> RoofHipped.do(footprint, coords)
  │                                          │
  │                                          └──> RoofHipped.render(footprint, roofItem)
  │                                                 [roof_hipped.py:60]
  │                                                 • if n==4: generateRoofQuadrangle()
  │                                                 • else:    generateRoof() -> polygonize()
  │                                                            [lib/bpypolyskel/bpypolyskel.py:882]
  │
  └──> if multipolygon with holes:
         └──> volumeGeneratorMultiHipped.do(footprint) [:65]
                └──> RoofHippedMulti.generateRoof()
                       └──> polygonize(verts, ..., holesInfo, ...)
```

---

## 3. Gabled Roof Implementation

### Decision Logic

**File:** `action/volume/__init__.py:64-105`

```python
# Roof type selection
volumeGenerators = {
    'flat': RoofFlat(...),
    'gabled': RoofProfile(roofDataGabled, ...),  # <- Profile with triangular cross-section
    'hipped': RoofHipped(...),
    'pyramidal': RoofGeneratrix(...),
    # ... other shapes
}

volumeGenerator = self.volumeGenerators.get(
    footprint.getStyleBlockAttr("roofShape"),
    self.volumeGenerators[Volume.defaultRoofShape]  # default: "flat"
)
```

- **Input:** `roof:shape` OSM tag or style system default
- **No heuristic:** BLOSM does NOT decide gabled vs hipped based on footprint geometry
- **Default:** Most categories default to "flat" unless explicitly tagged

### Geometry Construction Steps

**File:** `action/volume/roof_profile.py`

**Profile Definition** (lines 13-23):
```python
roofDataGabled = (
    (
        (0., 0.),   # Left eave  (x, z)
        (0.5, 1.),  # Ridge apex
        (1., 0.)    # Right eave
    ),
    {
        "numSamples": 10,       # Quantization for fast slot lookup
        "angleToHeight": 0.5    # Factor for angle-to-height conversion
    }
)
```

**Algorithm Overview** (RoofProfile class):

1. **Direction Computation** (`processDirection`, line 144):
   ```python
   # Get roof direction (ridge direction)
   if roofDirection is None:
       if roofOrientation == "across":
           # Ridge ACROSS longest edge (profile ALONG longest)
           d = max(polygon.edges).normalized()
       else:
           # Ridge PERPENDICULAR to longest edge (default, more natural)
           d = getDefaultDirection(polygon)  # perpendicular to longest edge
   ```

2. **Vertex Projection** (lines 168-177):
   - Project each footprint vertex onto direction vector `d`
   - Find min/max projections to get polygon width
   - Each vertex gets X-coordinate in profile space: `x = (proj[i] - proj[minIndex]) / polygonWidth`
   - Range: 0.0 (left end) to 1.0 (right end)

3. **Profile Slots** (line 506):
   - Create "slots" at each profile point (3 slots for gabled: left eave, ridge, right eave)
   - Slots collect vertex "parts" as algorithm traverses footprint

4. **Profiled Vertices** (`ProfiledVert` class, line 79):
   - For each footprint vertex at basement level:
     - Compute X in profile space (along direction)
     - Compute Y in profile space (perpendicular to direction)
     - Find which slot(s) vertex projects to
     - Create new vertex at `z = roofVerticalPosition + roofHeight * h` where `h` is profile height

5. **Slot Tracking** (`createProfileVertices`, line 733):
   - Walk around footprint polygon
   - For each edge between vertices `pv1` and `pv2`:
     - Create intermediate vertices at slot intersections
     - Build wall faces below roof edge
     - Track "parts" in each slot (sequences of vertices)

6. **Face Formation** (`trackUp`/`trackDown`, lines 380/265):
   - After all vertices placed, walk slots upward/downward
   - Connect parts into roof faces (triangles/quads)
   - Complex logic handles reflections, islands, nested structures

### Inputs/Parameters

**From Footprint:**
- `footprint.polygon.verts` - footprint vertices (2D or 3D)
- `footprint.polygon.n` - vertex count
- `footprint.roofHeight` - height of roof (from pitch, angle, or default)
- `footprint.roofVerticalPosition` - Z where roof starts (top of walls)
- `footprint.direction` - ridge direction vector (computed or from tag)
- `footprint.polygonWidth` - width along profile direction

**From Roof Class:**
- `self.profile` - profile shape (list of (x, z) points)
- `self.height` - default roof height (4.0m)
- `self.angleToHeight` - conversion factor if using roof angle

**From OSM Tags (optional):**
- `roof:direction` - cardinal direction or degrees
- `roof:orientation` - "along" or "across"
- `roof:height` - explicit height
- `roof:angle` - pitch angle (converted to height)

### Output Mesh Structure

**Vertices:**
- Original footprint vertices at `roofVerticalPosition` (roof base)
- Profiled vertices at elevated Z (along profile)
- Slot intersection vertices (intermediate points)

**Faces:**
- **Roof faces:** Triangles/quads formed by slot tracking
  - CCW winding for correct normals
  - UV coordinates computed for texturing
- **Wall faces:** Trapezoids/rectangles below roof edges
  - Connect basement to profiled vertices
  - Handled by facade renderer

**Normals:**
- Automatically derived from CCW winding
- Profile slopes determine roof face normals

### Edge Cases and Limitations

1. **Non-rectangular footprints:**
   - Algorithm handles arbitrary polygons (even concave)
   - Creates complex roof with multiple faces
   - May produce unexpected results for highly irregular shapes

2. **Concave polygons:**
   - Supported but may create self-intersecting roof geometry
   - No validation for concavity

3. **Multipolygons:**
   - Not handled by standard RoofProfile
   - Treated as multiple separate buildings

4. **Holes/inner rings:**
   - Not supported in RoofProfile
   - Would need RoofProfileMulti (not implemented for gabled)

5. **Near-zero polygon width:**
   - Validation check prevents division by zero
   - Falls back to invalid if width too small

6. **Degenerate profiles:**
   - Profile with repeated X values may cause issues
   - No explicit validation for profile shape

### Wall-Roof Connection

**File:** `action/volume/roof_profile.py:733-966`

**Gable Walls:**
- Triangular walls at ends perpendicular to ridge
- Created as trapezoid faces during `createProfileVertices`
- Base at `roofVerticalPosition`, apex at ridge height
- Coplanar with rectangular walls (shares footprint edge)

**Eave Connection:**
- Roof edges meet walls at `roofVerticalPosition`
- No explicit overhang in base implementation (can be added via offset action)
- Wall face extends from ground to `roofVerticalPosition`
- Roof starts at `roofVerticalPosition`

**Overhang:**
- Not natively supported in core algorithm
- Can be added via separate "offset" action before roof generation
- Offset expands footprint polygon outward by overhang distance

---

## 4. Hipped Roof Implementation

### Decision Logic

**File:** `action/volume/roof_hipped.py:60-82`

```python
def render(self, footprint, roofItem):
    firstVertIndex = self.getRoofFirstVertIndex(footprint)
    super().extrude(footprint, roofItem)  # Build walls
    
    # Route based on footprint vertex count
    if footprint.polygon.n == 4:
        ok = self.generateRoofQuadrangle(footprint, roofItem, firstVertIndex)
    else:
        ok = self.generateRoof(footprint, roofItem, firstVertIndex)
    
    if ok:
        self.facadeRenderer.render(footprint, self.data)
        self.roofRenderer.render(roofItem)
    else:
        # Fallback to flat roof if hipped generation fails
        self.volumeAction.volumeGenerators["flat"].do(footprint, ...)
```

- **Quadrangle path:** Analytical solution for 4-sided convex polygons
- **General path:** Straight skeleton for all other cases
- **Fallback:** Flat roof if straight skeleton fails (degenerate geometry)

### Geometry Construction: Quadrangle Case

**File:** `action/volume/roof_hipped.py:83-196`

**Algorithm:**

1. **Edge Analysis** (lines 94-115):
   ```python
   # For each edge i, compute:
   vector[i] = verts[i+1] - verts[i]           # Edge vector
   length[i] = vector[i].length                 # Edge length
   cos[i] = -vector[i].dot(vector[i-1]) / (length[i] * length[i-1])  # Interior angle cosine
   sin[i] = -(vector[i].cross(vector[i-1])[2]) / (length[i] * length[i-1])  # Sine
   
   # Distance to edge event (where bisectors meet):
   distance[i] = length[i] / ((1+cos[i])/sin[i] + (1+cos[i+1])/sin[i+1])
   ```

2. **Find Ridge Endpoints** (lines 117-130):
   - Find edge with minimum `distance[i]` → first event vertex
   - Opposite edge → second event vertex
   - Special case: square (all distances equal) → skip (pyramidal)

3. **Compute Ridge Vertices** (lines 132-141):
   ```python
   # Tangent of roof pitch
   tan = roofHeight / max(distance[0], distance[2])
   
   # Ridge vertices:
   vert1 = getRoofVert(baseVert[i], i, tan)
   vert2 = getRoofVert(baseVert[j], j, tan)
   
   # getRoofVert():
   #   Move inward from edge by distance[i]
   #   Raise by distance[i] * tan
   ```

4. **Create Roof Faces** (lines 143-195):
   - 2 triangular hips (at first/second event edges)
   - 2 trapezoidal sides (connecting ridges)
   - CCW winding for upward normals

**Inputs:**
- Quadrangle footprint vertices (must be convex)
- `roofHeight` - maximum height at ridge
- `firstVertIndex` - offset in verts array

**Output:**
- 4 roof faces (2 triangles, 2 trapezoids)
- 2 new ridge vertices

**Limitations:**
- Only works for convex quadrangles
- Non-convex or non-planar quads may produce incorrect geometry
- No validation for convexity

### Geometry Construction: General Polygon Case

**File:** `action/volume/roof_hipped.py:198-287`  
**Library:** `lib/bpypolyskel/bpypolyskel.py:882-956`

**Algorithm: Straight Skeleton**

1. **Unit Vector Computation** (lines 209-219):
   ```python
   unitVector = [(verts[i+1] - verts[i]).normalized() for i in range(n)]
   length = [edge.length for edge in unitVector]
   ```

2. **Call Straight Skeleton** (lines 224-234):
   ```python
   polygonize(
       verts,              # In/out: vertex list (skeleton nodes appended)
       firstVertIndex,     # Start of footprint vertices
       numPolygonVerts,    # Footprint vertex count
       None,               # holesInfo (None for simple polygon)
       footprint.roofHeight,  # Maximum height
       0,                  # tan (unused if height provided)
       roofSideIndices,    # Out: list of face vertex indices
       unitVector          # Precomputed edge directions
   )
   ```

3. **Straight Skeleton Details** (`lib/bpypolyskel/bpypolyskel.py`):

   **Core Concept:**
   - Propagate polygon edges inward at constant speed
   - Edges shrink as they move
   - When edges meet, create "events" (vertices of skeleton)
   - Edge events → skeleton edges → roof faces

   **Event Types:**
   - **Edge event:** Two adjacent edges collapse to a point
   - **Split event:** Edge hits opposite edge, splits skeleton

   **Output:**
   - New vertices appended to `verts` (skeleton nodes)
   - `roofSideIndices` filled with face definitions (lists of vertex indices)
   - Each face has first edge on original footprint

4. **Face Construction** (lines 239-286):
   ```python
   for indices in roofSideIndices:
       edgeIndex = indices[0] - firstVertIndex
       if edgeIndex < numPolygonVerts:
           # Normal case: face has edge on footprint
           roofItem.addRoofSide(indices, uvs, edgeIndex, ...)
       else:
           # Exotic case: face has no footprint edge (internal face)
           # Compute face normal and UV basis vectors
           roofItem.addRoofSide(indices, uvs, -1, ...)
   ```

   **UV Mapping:**
   - For normal faces: U along footprint edge, V up slope
   - For internal faces: U/V in arbitrary face-aligned basis
   - Slope factor: `factor = sqrt(1 + tan²)` for texture scaling

**Inputs:**
- Arbitrary polygon (can be concave, any vertex count)
- `roofHeight` - maximum skeleton height
- Optional `tan` (roof pitch tangent) - alternative to height

**Output:**
- N roof faces (triangles/quads/polygons)
- M new skeleton vertices
- Faces connect footprint edges to skeleton edges

**Limitations:**
- Computationally expensive (iterative event queue)
- May fail for degenerate polygons (very narrow, self-intersecting)
- Numerical precision issues with near-parallel edges
- No explicit handling of concave inputs (may produce overlapping faces)

### Defaults and Fallbacks

**File:** `action/volume/roof_hipped.py:72-81`

```python
if ok:
    self.facadeRenderer.render(footprint, self.data)
    self.roofRenderer.render(roofItem)
else:
    # Unable to generate the hipped roof.
    # Generate a flat roof as a fallback solution
    self.volumeAction.volumeGenerators["flat"].do(footprint, ...)
```

**Fallback Conditions:**
- Straight skeleton fails validation (`validatePolygonizeOutput`, line 54)
  - Faces with < 3 vertices
  - Faces with duplicate vertex indices
- Quadrangle case: square footprint (distance[0] == distance[1])
  - Returns early, no roof generated (should use pyramidal instead)

**Validation:**
```python
def validatePolygonizeOutput(self, roofSideIndices):
    for indices in roofSideIndices:
        if not (len(indices) >= 3 and len(indices) == len(set(indices))):
            return False
    return True
```

### Wall-Roof Connection

**Hipped roofs have NO gable walls** (triangular end walls)
- All sides slope toward center
- Roof faces meet walls at all footprint edges
- Edge connection at `roofVerticalPosition`

**Eaves:**
- Every footprint edge is an eave (bottom of roof slope)
- Roof faces start at footprint vertices
- No explicit overhang (same as gabled)

**Hip Lines:**
- Intersection of adjacent roof faces
- Rise from footprint corners to ridge/apex
- Slope determined by edge event geometry

---

## 5. Defaults and Fallbacks

### Roof Type Selection

**File:** `action/volume/__init__.py:95-100`

```python
volumeGenerator = self.volumeGenerators.get(
    footprint.getStyleBlockAttr("roofShape"),
    self.volumeGenerators[Volume.defaultRoofShape]  # "flat"
)
```

**Default Roof Shape:** `"flat"` (per-category styling may override)

**OSM Tag Priority:**
1. `roof:shape` tag (if present)
2. Style system rules (based on `building` tag)
3. Global default: `"flat"`

### Style System Defaults

**File:** `style/default.py`

**Examples:**
```python
# Residential (line 41):
roofShape = Value(Alternatives(
    #RandomWeighted(( ("gabled", 10), ("flat", 40) ))
))

# Commercial (line 272):
roofShape = Value(Alternatives(...))

# Industrial (line 329):
roofShape = Value(Alternatives(...))
```

- Most categories DO NOT specify default roof shape
- Falls back to global default (`"flat"`)
- Gabled/hipped only used when explicitly tagged

### Missing Tag Behavior

**Height Calculation** (`roof.py:179-213`):
```python
# If no roof:height tag:
if h is None:
    if not self.angleToHeight is None and "roofAngle" in footprint.styleBlock:
        # Use roof:angle if present
        angle = footprint.getStyleBlockAttr("roofAngle")
        h = self.angleToHeight * footprint.polygonWidth * math.tan(radians(angle))
    else:
        # Use default height
        h = self.height  # 4.0m for gabled, 4.0m for hipped
```

**Direction Calculation** (`roof.py:144-166`):
```python
# If no roof:direction tag:
if d is None:
    if self.hasRidge and roofOrientation == "across":
        # Ridge across longest edge
        d = max(polygon.edges).normalized()
    else:
        # Ridge perpendicular to longest edge (default)
        d = getDefaultDirection(polygon)  # perpendicular to max edge
```

### Multipolygon Behavior

**File:** `action/volume/__init__.py:61-93`

```python
if element.t is parse.multipolygon:
    if element.hasInner():
        # Multipolygon with holes
        if footprint.getStyleBlockAttr("roofShape") in ("hipped", "gabled"):
            self.volumeGeneratorMultiHipped.do(footprint)
        else:
            self.volumeGeneratorMultiFlat.do(footprint)
    else:
        # Multipolygon without holes (multiple separate polygons)
        # Treat each polygon as separate building
        for polygon in element.ls:
            footprint = Footprint.getItem(...)
            self.generateVolume(footprint, coords)
```

**Notes:**
- Gabled multipolygon NOT SUPPORTED (uses `RoofHippedMulti` fallback)
- Hipped multipolygon uses straight skeleton with hole support
- Multipolygon without holes → split into separate buildings

---

## 6. Algorithm Notes

### Polygon Processing

**No OBB/PCA in BLOSM:**
- Uses iterative edge analysis instead
- Finds longest edge directly: `max(polygon.edges, key=lambda e: e.length)`
- Direction perpendicular to longest edge: `longestEdge.cross(normal).normalized()`

**Edge Iteration:**
```python
# Typical pattern:
for i in range(n):
    j = (i + 1) % n
    edge = verts[j] - verts[i]
    # ... process edge
```

**Convexity:**
- Not explicitly checked
- Algorithms assume counterclockwise winding
- Concave polygons may produce unexpected results

### Straight Skeleton

**File:** `lib/bpypolyskel/bpypolyskel.py`

**Algorithm Summary:**
1. Initialize event queue with edge events
2. Pop earliest event
3. Process event:
   - Create skeleton vertex
   - Update adjacent edges
   - Add new events to queue
4. Repeat until all edges collapsed
5. Convert skeleton to faces via graph traversal

**Complexity:**
- Time: O(n² log n) worst case (n = vertex count)
- Space: O(n) for skeleton storage

**Limitations:**
- Precision-sensitive (uses EPSILON = 0.00001)
- May fail for near-degenerate cases
- No explicit error handling (validation post-facto)

**Libraries:**
- `bpypolyskel.py` - straight skeleton implementation
- `bpyeuclid.py` - 2D geometry primitives (Vector2, Line2, Ray2, Edge2)
- `poly2FacesGraph.py` - skeleton to face conversion

### Direction Computation

**File:** `action/volume/roof.py:144-177`

**Method:** `processDirection(footprint)`

1. **Check OSM tag `roof:direction`:**
   - Cardinal direction (e.g., "N", "SE") → predefined vector
   - Numeric degrees → convert to vector

2. **Check `roof:orientation`:**
   - "across" → ridge ALONG longest edge (profile across)
   - Default → ridge PERPENDICULAR to longest edge (profile along)

3. **Compute default:**
   ```python
   def getDefaultDirection(polygon):
       # Perpendicular to longest edge
       return max(polygon.edges).cross(polygon.normal).normalized()
   ```

4. **Project vertices:**
   ```python
   projections = [d[0]*v[0] + d[1]*v[1] for v in polygon.verts]
   minProjIndex = argmin(projections)
   maxProjIndex = argmax(projections)
   polygonWidth = projections[maxProjIndex] - projections[minProjIndex]
   ```

**Result:**
- `footprint.direction` - unit vector (ridge direction)
- `footprint.projections` - list of vertex projections
- `footprint.polygonWidth` - span along direction
- `footprint.minProjIndex`, `footprint.maxProjIndex` - boundary vertices

### Height Calculation

**File:** `action/volume/roof.py:179-266`

**Priority:**
1. `roof:height` tag (explicit)
2. `roof:angle` tag (convert to height via `polygonWidth * tan(angle)`)
3. Roof levels (`numRoofLevels`, `roofLevelHeight`)
4. Default height (`self.height`, typically 4.0m)

**Angle-to-Height:**
```python
if "roofAngle" in footprint.styleBlock:
    angle = footprint.getStyleBlockAttr("roofAngle")
    h = self.angleToHeight * footprint.polygonWidth * math.tan(radians(angle))
```
- `angleToHeight = 0.5` for gabled (half-width to ridge)
- Adjusts for profile shape

**Roof Levels:**
- Multi-story roofs (e.g., mansard)
- Sum level heights to get total roof height
- Not commonly used for simple gabled/hipped

---

## 7. Integration Ideas for `condor_buildings`

### Current Condor Pipeline vs BLOSM

| **Aspect** | **Condor Buildings** | **BLOSM** |
|------------|---------------------|-----------|
| **Target** | House-scale rectangles (4-vertex convex) | Arbitrary building footprints |
| **Gabled** | OBB-based, ridge along long axis | Profile-swept, direction-based |
| **Hipped** | Not yet implemented | Quadrangle (analytical) + general (straight skeleton) |
| **Complexity** | Optimized for simple case | Generic, handles all cases |
| **Algorithm** | Direct geometry from OBB | Profile slots / straight skeleton |
| **Overhang** | Built into geometry (expanded footprint) | Separate offset action |
| **Performance** | Fast (< 1ms per building) | Slower (profile tracking, skeleton) |

### Recommended Approach: Hipped Roofs for Condor

**For House-Scale Rectangles (4 vertices, convex):**

1. **Adopt BLOSM's Quadrangle Algorithm** (`roof_hipped.py:83-196`):
   - Reusable for our rectangle case
   - Analytical solution (no iterative algorithms)
   - Computes ridge endpoints directly from edge geometry

2. **Simplify for Rectangles:**
   - Our OBB already gives us length, width, center
   - Can skip BLOSM's edge iteration (we know it's a rectangle)
   - Use OBB "width" (short dimension) as half-span
   - Ridge endpoints at `±length/2` along OBB "along" axis

3. **Overhang Handling:**
   - Apply overhang BEFORE roof geometry (like BLOSM's offset action)
   - Expand footprint by overhang distance
   - Compute roof from expanded footprint
   - Ensures eaves drop below wall top (natural overhang)

**Algorithm Sketch:**
```python
def generate_hipped_roof_quadrangle(building, overhang=0.5):
    # 1. Get OBB
    obb = compute_obb(building.footprint.outer_ring, ridge_direction)
    
    # 2. Expand footprint by overhang
    expanded_footprint = expand_polygon(footprint, overhang)
    
    # 3. For each edge, compute distance to bisector intersection
    #    (port BLOSM's edge event calculation)
    vectors, lengths, cos, sin, distances = compute_edge_geometry(expanded_footprint)
    
    # 4. Find minimum distance edges (ridge endpoints)
    min_idx1 = argmin(distances)
    min_idx2 = (min_idx1 + 2) % 4  # opposite edge
    
    # 5. Compute ridge vertices
    tan = building.roof_height / max(distances[min_idx1], distances[min_idx2])
    ridge_v1 = get_ridge_vertex(expanded_footprint[min_idx1], distances[min_idx1], tan)
    ridge_v2 = get_ridge_vertex(expanded_footprint[min_idx2], distances[min_idx2], tan)
    
    # 6. Create 4 roof faces:
    #    - 2 triangular hips (at min_idx1, min_idx2)
    #    - 2 trapezoidal sides (connecting ridges)
    mesh.add_triangle(footprint[min_idx1], footprint[min_idx1+1], ridge_v1)
    mesh.add_quad(footprint[min_idx1+1], footprint[min_idx2], ridge_v2, ridge_v1)
    mesh.add_triangle(footprint[min_idx2], footprint[min_idx2+1], ridge_v2)
    mesh.add_quad(footprint[min_idx2+1], footprint[min_idx1], ridge_v1, ridge_v2)
```

**Key Functions to Port:**
- Edge geometry calculation (vectors, angles, distances)
- Ridge vertex calculation (`getRoofVert`)
- UV mapping (for texturing)

### Gabled Roofs: Comparison

**BLOSM Gabled:**
- Over-engineered for rectangles
- Profile slots are overkill for simple case
- Complex tracking algorithm unnecessary

**Condor Gabled (current):**
- Simple, direct: ridge from OBB, 2 slope quads, 2 gable triangles
- Already optimal for our use case
- No need to adopt BLOSM's approach

**Recommendation:**
- **Keep current gabled implementation** (already superior for rectangles)
- **Port BLOSM's hipped quadrangle logic** (fills capability gap)

### Data Needs for Integration

**To Port BLOSM Hipped Quadrangle:**

| **Input** | **Source in Condor** | **Notes** |
|-----------|---------------------|-----------|
| Footprint vertices | `building.footprint.outer_ring` | Already available |
| Roof height | `building.ridge_height_m` | Already computed for gabled |
| Overhang | `config.overhang` | Already used for gabled |
| Wall top Z | `building.wall_top_z` | Already computed |
| Ridge direction | `building.roof_direction_deg` | Already computed (OBB long axis) |

**New Calculations Needed:**
- Edge vectors (trivial: `v[i+1] - v[i]`)
- Edge lengths (trivial: `edge.length()`)
- Interior angles (BLOSM: `cos[i]`, `sin[i]`)
- Bisector distances (BLOSM: `distance[i]`)

**New Geometry Functions:**
- `compute_edge_geometry(vertices)` → vectors, lengths, angles, distances
- `get_ridge_vertex(base_vertex, distance, tan)` → 3D vertex
- `create_hip_face(...)` → triangle/quad face

### Validation and Testing

**Test Footprints:**
1. **Perfect rectangle:**
   - 10m × 6m
   - Expect symmetric hip roof, ridge length ~8m

2. **Near-square:**
   - 8m × 7m
   - Expect short ridge (~6m), nearly pyramidal

3. **Elongated rectangle:**
   - 15m × 5m
   - Expect long ridge (~13m), steep hips

4. **Slightly irregular quadrangle:**
   - Vertices slightly off rectangle
   - Test robustness of algorithm

5. **Non-convex quadrangle (if allowed):**
   - Concave corner
   - Should fallback to flat or error gracefully

**Edge Cases to Handle:**
- Square footprint (BLOSM special case: `distance[0] == distance[1]`)
  - Should produce pyramidal (no ridge, single apex)
- Very elongated (aspect ratio > 10)
  - Ensure ridge vertices don't go out of bounds
- Near-degenerate (width < 0.1m)
  - Fallback to flat

### Coordinate System Alignment

**BLOSM:**
- Right-handed: X (east), Y (north), Z (up)
- CCW winding for upward normals
- Blender coordinate system

**Condor:**
- Same: X (east), Y (north), Z (up)
- Same winding convention
- No conversion needed

**Units:**
- BLOSM: meters (OpenStreetMap standard)
- Condor: meters (Condor flight sim standard)
- No conversion needed

### Potential Pitfalls

1. **Non-convex footprints:**
   - BLOSM's quadrangle algorithm assumes convexity
   - Our pipeline already filters for convex (via rectangularity check)
   - Add explicit convexity validation before hipped generation

2. **Inverted normals:**
   - Ensure CCW winding for all faces
   - BLOSM uses careful vertex ordering (can reuse)

3. **Wall-roof gap:**
   - BLOSM: roof starts at `roofVerticalPosition`
   - Condor: walls go to `wall_top_z`
   - **Must ensure:** `roofVerticalPosition == wall_top_z`

4. **Overhang edge cases:**
   - Overhang may cause self-intersection for small footprints
   - BLOSM doesn't validate overhang size
   - **Add check:** `overhang < min(width, length) / 2`

5. **Square fallback:**
   - BLOSM returns early for squares (no ridge)
   - Our implementation should generate pyramidal roof instead
   - **Add logic:** If square, create apex vertex + 4 triangular faces

6. **Straight skeleton not needed:**
   - BLOSM's general polygon path is overkill for our rectangles
   - **Do NOT port straight skeleton** (too complex, unnecessary)

---

## 8. Appendix

### Code Excerpts

**A. BLOSM Quadrangle Hipped Roof (Simplified)**

```python
# action/volume/roof_hipped.py:83-196

def generateRoofQuadrangle(self, footprint, roofItem, firstVertIndex):
    verts = footprint.building.verts
    vector, length, cos, sin, distance = [], [], [], [], []
    
    # Compute edge geometry
    vector = [verts[firstVertIndex + (i+1)%4] - verts[firstVertIndex + i] for i in range(4)]
    length = [v.length for v in vector]
    cos = [-(vector[i].dot(vector[i-1])) / (length[i] * length[i-1]) for i in range(4)]
    sin = [-(vector[i].cross(vector[i-1])[2]) / (length[i] * length[i-1]) for i in range(4)]
    distance = [length[i] / ((1+cos[i])/sin[i] + (1+cos[(i+1)%4])/sin[(i+1)%4]) for i in range(4)]
    
    # Find ridge endpoints
    minDistanceIndex1 = min(range(4), key=lambda i: distance[i])
    minDistanceIndex2 = (minDistanceIndex1 + 2) % 4  # opposite edge
    
    # Compute ridge vertices
    tan = footprint.roofHeight / max(distance[minDistanceIndex1], distance[minDistanceIndex2])
    factor = sqrt(1 + tan*tan)
    
    vert1 = getRoofVert(verts[firstVertIndex + minDistanceIndex1], minDistanceIndex1, tan)
    vert2 = getRoofVert(verts[firstVertIndex + minDistanceIndex2], minDistanceIndex2, tan)
    
    # Create roof faces (2 triangles + 2 quads)
    roofItem.addRoofSide([firstVertIndex + minDistanceIndex1, firstVertIndex + (minDistanceIndex1+1)%4, vert1], ...)
    roofItem.addRoofSide([firstVertIndex + (minDistanceIndex1+1)%4, firstVertIndex + minDistanceIndex2, vert2, vert1], ...)
    # ... (2 more faces)

def getRoofVert(self, vert, i, tan):
    """Compute ridge vertex by moving inward and up from edge."""
    return vert + \
        self.distance[i] / self.length[i] * (zAxis.cross(self.vector[i]) + (1+self.cos[i])/self.sin[i] * self.vector[i]) + \
        self.distance[i] * tan * zAxis
```

**B. BLOSM Direction Calculation**

```python
# action/volume/roof.py:144-177

def processDirection(self, footprint):
    polygon = footprint.polygon
    d = footprint.getStyleBlockAttr("roofDirection")
    
    if d is None:
        if self.hasRidge and footprint.getStyleBlockAttr("roofOrientation") == "across":
            d = max(polygon.edges).normalized()  # Ridge ALONG longest edge
        else:
            d = getDefaultDirection(polygon)  # Ridge PERPENDICULAR to longest edge
    elif d in Roof.directions:
        d = Roof.directions[d]  # Cardinal direction lookup
    else:
        d = math.radians(d)
        d = Vector((math.sin(d), math.cos(d), 0.))  # Convert degrees to vector
    
    footprint.direction = d
    
    # Project vertices onto direction
    projections = [d[0]*v[0] + d[1]*v[1] for v in polygon.verts]
    footprint.minProjIndex = min(range(polygon.n), key=lambda i: projections[i])
    footprint.maxProjIndex = max(range(polygon.n), key=lambda i: projections[i])
    footprint.polygonWidth = projections[footprint.maxProjIndex] - projections[footprint.minProjIndex]

def getDefaultDirection(polygon):
    """Perpendicular to longest edge."""
    return max(polygon.edges).cross(polygon.normal).normalized()
```

**C. BLOSM Gabled Profile Sweeping (Excerpt)**

```python
# action/volume/roof_profile.py:733-966

def createProfileVertices(self, pv1, pv2, _pv, footprint):
    """Create vertices at slot intersections between profiled vertices pv1 and pv2."""
    verts = footprint.building.verts
    slots = self.slots
    slot = self.slot  # current slot
    
    # Determine if pv1 is on a slot
    if pv1.onSlot:
        # Check for reflection, change slot, create new part
        if should_change_slot(pv1, pv2, _pv):
            self.originSlot = slot
            slot = slots[pv1.index]
            slot.append(pv1.vertIndex, pv1.y, self.originSlot, reflection)
    
    # Create in-between vertices at slot intersections
    if pv1.index != pv2.index:
        for slotIndex in range(pv2.index, pv1.index, direction):
            # Compute vertex at slot intersection
            x = slots[slotIndex].x
            vert = interpolate(pv1, pv2, x, footprint.roofHeight)
            verts.append(vert)
            
            # Append to slot and create new part
            slot.append(vertIndex)
            self.originSlot = slot
            slot = slots[slotIndex]
            slot.append(vertIndex, y, self.originSlot)
    
    # Append pv2 to current slot
    slot.append(pv2.vertIndex)
    self.slot = slot
```

### Test Footprints for Validation

**1. Perfect Rectangle (10m × 6m):**
```
Vertices (CCW):
  (0, 0), (10, 0), (10, 6), (0, 6)

Expected Hipped Roof:
  - Ridge: (2, 3) to (8, 3), length = 6m, height = 3m (pitch ~45°)
  - 4 faces: 2 triangular hips (at short ends), 2 trapezoidal sides
  - Ridge endpoints 2m from ends (distance = width/2 = 3m, offset = 2m)
```

**2. Near-Square (8m × 7m):**
```
Vertices (CCW):
  (0, 0), (8, 0), (8, 7), (0, 7)

Expected Hipped Roof:
  - Ridge: (1.5, 3.5) to (6.5, 3.5), length = 5m, height = 3.5m
  - Nearly pyramidal (short ridge)
  - Steeper hip angles
```

**3. Elongated Rectangle (15m × 5m):**
```
Vertices (CCW):
  (0, 0), (15, 0), (15, 5), (0, 5)

Expected Hipped Roof:
  - Ridge: (2.5, 2.5) to (12.5, 2.5), length = 10m, height = 2.5m
  - Long ridge, steep hips at ends
  - Shallower roof pitch overall
```

**4. Slightly Irregular Quadrangle:**
```
Vertices (CCW):
  (0, 0), (10.1, 0.2), (10, 6), (0.1, 5.9)

Expected Hipped Roof:
  - Ridge slightly off-center
  - Asymmetric hip faces
  - Algorithm should handle gracefully
```

**5. L-Shape (Non-convex, should REJECT):**
```
Vertices (CCW):
  (0, 0), (10, 0), (10, 5), (5, 5), (5, 10), (0, 10)

Expected Behavior:
  - Fallback to flat roof
  - Log warning: "Hipped roof requires convex footprint"
```

**6. Square (8m × 8m):**
```
Vertices (CCW):
  (0, 0), (8, 0), (8, 8), (0, 8)

Expected Behavior (BLOSM):
  - Returns early (no ridge, distance[0] == distance[1])
  - Should generate pyramidal instead

Expected Behavior (Condor):
  - Detect square (aspect ratio ~1.0)
  - Create pyramidal: apex at (4, 4, wall_top_z + 4), 4 triangular faces
```

### File Reference Table

| **File** | **Lines** | **Purpose** |
|----------|-----------|-------------|
| `action/volume/__init__.py` | 119 | Main entry point, roof type dispatcher |
| `action/volume/roof.py` | 266 | Base Roof class, direction/height calculation |
| `action/volume/roof_gabled.py` | 25 | Gabled roof wrapper (uses RoofProfile) |
| `action/volume/roof_profile.py` | 1077 | Profile-based roof implementation |
| `action/volume/roof_hipped.py` | 304 | Hipped roof (quadrangle + general) |
| `action/volume/roof_hipped_multi.py` | 171 | Hipped multipolygon with holes |
| `lib/bpypolyskel/bpypolyskel.py` | 1600+ | Straight skeleton algorithm |
| `item/roof_profile.py` | - | Data structure for profile roofs |
| `item/roof_hipped.py` | - | Data structure for hipped roofs |

### Key Observations Summary

1. **BLOSM is a generic solution** for arbitrary building footprints
   - Handles concave, holes, multipolygons, any vertex count
   - Our pipeline targets house-scale rectangles (much simpler)

2. **Gabled implementation is complex** due to generality
   - Profile slots handle arbitrary footprints
   - Tracking algorithm handles reflections, islands, nested structures
   - **Not needed for our rectangles** (current OBB approach is better)

3. **Hipped quadrangle algorithm is portable** and useful
   - Analytical solution for 4-sided convex polygons
   - Direct computation (no iterative algorithms)
   - **Good fit for Condor pipeline** (same constraints)

4. **Straight skeleton is overkill** for rectangles
   - O(n² log n) complexity
   - Precision-sensitive, may fail
   - Only needed for complex polygons (not our target)

5. **Direction calculation differs** from our approach
   - BLOSM: iterates edges, finds longest, computes perpendicular
   - Condor: computes OBB, uses long axis directly
   - **OBB is superior** (avoids edge iteration, already computed)

6. **Overhang handled differently**
   - BLOSM: separate offset action expands footprint
   - Condor: built into roof geometry (expanded footprint in roof gen)
   - **Both approaches work** (conceptually equivalent)

7. **Fallback strategy is simple**
   - Hipped generation fails → use flat roof
   - No multi-level fallback chain
   - **Adopt same strategy** for Condor

---

## End of Report

**Next Steps:**
1. Prototype hipped quadrangle algorithm in `condor_buildings/generators/roof_hipped.py`
2. Port BLOSM's edge geometry calculations
3. Test with rectangular footprints (4-vertex, convex)
4. Validate overhang behavior
5. Add square → pyramidal special case
6. Integrate with building generator (roof type selection)
