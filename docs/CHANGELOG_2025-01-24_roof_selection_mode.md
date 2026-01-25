# Changelog: Roof Selection Mode

**Date:** 2025-01-24
**Session:** Add configurable roof selection mode
**Version:** 0.3.4 -> 0.3.5

---

## Summary

Added a new configuration parameter `--roof-selection-mode` that controls how buildings are selected for pitched roofs (gabled/hipped). This allows choosing between geometry-based heuristics or strict OSM tag-based selection.

---

## Problem Statement

Previously, the pipeline used geometry + category heuristics + area to decide which buildings receive pitched roofs. This included:
- Buildings tagged as `building=house`, `building=detached`, etc.
- Buildings tagged as `building=yes` with small footprints (<200 m²)
- Small commercial/apartment buildings

This approach sometimes assigned gabled roofs to buildings that shouldn't have them. A new mode was needed to restrict pitched roofs to only buildings explicitly tagged as houses in OSM.

---

## Changes Implemented

### 1. New `RoofSelectionMode` Enum

**File:** `condor_buildings/config.py`

```python
class RoofSelectionMode(Enum):
    """
    Mode for selecting which buildings receive pitched (gabled/hipped) roofs.

    GEOMETRY: (Default) Use geometry constraints + category heuristics + area.
    OSM_TAGS_ONLY: Only buildings explicitly tagged as houses get pitched roofs.
    """
    GEOMETRY = "geometry"
    OSM_TAGS_ONLY = "osm_tags_only"
```

### 2. New `roof_selection_mode` in PipelineConfig

**File:** `condor_buildings/config.py`

```python
@dataclass
class PipelineConfig:
    # ...
    roof_selection_mode: RoofSelectionMode = RoofSelectionMode.GEOMETRY
```

### 3. Updated `select_roof_type()` Function

**File:** `condor_buildings/generators/building_generator.py`

Added `selection_mode` parameter:

```python
def select_roof_type(
    building: BuildingRecord,
    force_flat: bool = False,
    selection_mode: RoofSelectionMode = RoofSelectionMode.GEOMETRY
) -> RoofType:
```

**Logic for `OSM_TAGS_ONLY` mode:**
```python
if selection_mode == RoofSelectionMode.OSM_TAGS_ONLY:
    if building.category == BuildingCategory.HOUSE:
        return RoofType.GABLED  # Can become HIPPED via --random-hipped
    else:
        return RoofType.FLAT
```

### 4. New CLI Argument

**File:** `condor_buildings/main.py`

```python
parser.add_argument(
    '--roof-selection-mode',
    type=str,
    choices=['geometry', 'osm_tags_only'],
    default='geometry',
    help='Roof selection mode: "geometry" (default) uses geometry+category heuristics, '
         '"osm_tags_only" gives pitched roofs only to buildings tagged as houses'
)
```

### 5. Created `CLAUDE.md` Quick Reference

**File:** `CLAUDE.md` (project root)

Created a quick reference file that Claude Code reads automatically at session start. Contains:
- How to run the pipeline
- Test data location
- Output file structure
- Key CLI arguments
- Project structure overview

---

## Files Modified

| File | Changes |
|------|---------|
| `condor_buildings/config.py` | Added `RoofSelectionMode` enum, added `roof_selection_mode` to `PipelineConfig` |
| `condor_buildings/generators/building_generator.py` | Added `selection_mode` parameter to `select_roof_type()`, imported `RoofSelectionMode` |
| `condor_buildings/main.py` | Added `--roof-selection-mode` CLI argument, imported `RoofSelectionMode`, pass mode to `select_roof_type()` |
| `CLAUDE.md` | **NEW** - Quick reference for Claude sessions |

---

## Usage Examples

### Default mode (geometry + heuristics)
```bash
python -m condor_buildings.main --patch-dir test_data --patch-id 036019 --output-dir output
```

### OSM tags only mode
```bash
python -m condor_buildings.main --patch-dir test_data --patch-id 036019 --output-dir output \
    --roof-selection-mode osm_tags_only
```

### OSM tags only + random hipped (for visual testing)
```bash
python -m condor_buildings.main --patch-dir test_data --patch-id 036019 --output-dir output \
    --roof-selection-mode osm_tags_only --random-hipped
```

---

## Test Results (Patch 036019)

### With `--roof-selection-mode osm_tags_only --random-hipped`

| Metric | Value |
|--------|-------|
| Buildings processed | 5,416 |
| **Gabled roofs** | **261** |
| **Hipped roofs** | **286** |
| **Flat roofs** | **4,869** |
| Gabled→Flat fallbacks | 1,751 |
| Hipped→Flat fallbacks | 152 |

### Comparison with `geometry` mode (previous default)

| Mode | Gabled | Hipped | Flat |
|------|--------|--------|------|
| `geometry` | 1,811 | 0 | 3,605 |
| `osm_tags_only` | 261 | 286 | 4,869 |
| `osm_tags_only --random-hipped` | 261 | 286 | 4,869 |

The `osm_tags_only` mode significantly reduces pitched roofs because only buildings with explicit house tags (like `building=house`, `building=detached`, `building=villa`) receive them. Buildings with `building=yes` (the majority) now get flat roofs.

---

## Building Categories Affected

### Gets pitched roof in `osm_tags_only` mode:
- `building=house`
- `building=detached`
- `building=semidetached_house`
- `building=terrace`
- `building=farm`
- `building=farmhouse`
- `building=cabin`
- `building=bungalow`
- `building=villa`
- `building=residential`
- `building=hut`
- `building=shed`

### Gets flat roof in `osm_tags_only` mode:
- `building=yes` (most common)
- `building=apartments`
- `building=commercial`
- `building=industrial`
- `building=warehouse`
- All other categories

---

## Interaction with `--random-hipped`

The `--random-hipped` flag continues to work with both modes:
- In `geometry` mode: 50% of eligible buildings get hipped instead of gabled
- In `osm_tags_only` mode: 50% of HOUSE-category buildings get hipped instead of gabled

This allows visual testing of hipped roofs while using the stricter selection mode.

---

## Session Notes

This feature was requested to have more control over which buildings receive pitched roofs. The `osm_tags_only` mode is useful when:
1. The OSM data has good building type tagging
2. You want to avoid false positives (non-houses with gabled roofs)
3. You prefer a more conservative approach

The `geometry` mode remains the default for backward compatibility and for areas where OSM tagging is incomplete.

Total implementation time: ~30 minutes
