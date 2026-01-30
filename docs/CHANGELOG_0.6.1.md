# Changelog v0.6.1 - Configurable Parameters in Blender UI

**Release Date:** January 30, 2025

## Overview

This release exposes key pipeline configuration parameters in the Blender addon UI, allowing users to customize building generation without modifying code. Previously hardcoded values can now be adjusted directly from the Blender interface.

## New Features

### Configurable Roof Geometry Parameters

New controls in the "Roof Options" panel:

| Parameter | Default | Description |
|-----------|---------|-------------|
| **Gable Height** | 3.0m | Height of gable/hipped roof peak above walls |
| **Roof Overhang** | 0.5m | Overhang distance beyond walls (LOD0 only) |
| **Max Floors (Gabled)** | 2 | Maximum floors for gabled/hipped roof eligibility |

### Advanced Geometry Constraints

New controls in the "Advanced" panel:

| Parameter | Default | Description |
|-----------|---------|-------------|
| **Min Rectangularity** | 0.70 | Minimum area/OBB ratio for gabled eligibility |
| **Max Vertices (Polyskel)** | 12 | Maximum footprint vertices for polyskel hipped roofs |

### Terrain Integration

| Parameter | Default | Description |
|-----------|---------|-------------|
| **Floor Z Offset** | 0.3m | Distance to sink buildings below terrain (prevents gaps on slopes) |

### Reproducibility

| Parameter | Default | Description |
|-----------|---------|-------------|
| **Random Seed** | 42 | Seed for deterministic texture/style variation |

## Technical Changes

### Runtime Configuration System

Added a new runtime configuration system in `building_generator.py`:

```python
from condor_buildings.generators import configure_generator

# Configure before processing a batch
configure_generator(
    gable_height=4.0,
    roof_overhang_lod0=0.3,
    gabled_max_floors=3,
    # ... other parameters
)

# Process buildings with new configuration
result = generate_building_lod0(building)
```

**New functions:**
- `configure_generator(**kwargs)` - Set runtime parameters
- `reset_generator_config()` - Reset to defaults
- `get_runtime_config()` - Get current configuration

### Files Changed

| File | Changes |
|------|---------|
| `blender/properties.py` | Added 7 new UI properties |
| `blender/panels.py` | Added UI sections for new parameters |
| `blender/operators.py` | Calls `configure_generator()` before processing |
| `generators/building_generator.py` | Added `GeneratorRuntimeConfig` class and configuration functions |
| `generators/__init__.py` | Exported new configuration functions |
| `config.py` | Added new fields to `PipelineConfig` |

## UI Layout

### Roof Options Panel (expanded)
```
Roof Options
├── Roof Selection Mode
├── Random Hipped Roofs
└── Roof Geometry
    ├── Gable Height: [3.0] m
    ├── Roof Overhang: [0.5] m
    └── Max Floors (Gabled): [2]
```

### Advanced Panel (expanded)
```
Advanced
├── House-Scale Constraints
│   ├── Max House Area
│   ├── Max Side Length
│   ├── Min Side Length
│   └── Max Aspect Ratio
├── Geometry Constraints
│   ├── Min Rectangularity: [0.70]
│   └── Max Vertices (Polyskel): [12]
├── Terrain Integration
│   └── Floor Z Offset: [0.3] m
└── Reproducibility
    └── Random Seed: [42]
```

## Backward Compatibility

- All default values match previous hardcoded constants
- Existing workflows will produce identical results
- CLI tool continues to work as before (uses defaults)

## Migration Notes

No migration required. Simply update the addon ZIP in Blender.

---

**Full Changelog:** [v0.6.0...v0.6.1](https://github.com/condor-buildings/condor-buildings-generator/compare/v0.6.0...v0.6.1)
