# Condor Buildings Generator - Quick Reference for Claude

## Project Overview

This is a Python pipeline that generates 3D building meshes from OpenStreetMap (OSM) data for Condor 3 flight simulator. It produces OBJ files with UV coordinates for texture mapping.

## How to Run the Pipeline

```bash
# Basic run
python -m condor_buildings.main --patch-dir test_data --patch-id 036019 --output-dir output

# With verbose logging
python -m condor_buildings.main --patch-dir test_data --patch-id 036019 --output-dir output --verbose

# OSM tags only mode (only tagged houses get pitched roofs)
python -m condor_buildings.main --patch-dir test_data --patch-id 036019 --output-dir output --roof-selection-mode osm_tags_only

# With random hipped roofs (50% gabled, 50% hipped for testing)
python -m condor_buildings.main --patch-dir test_data --patch-id 036019 --output-dir output --random-hipped

# Combined: OSM tags only + random hipped
python -m condor_buildings.main --patch-dir test_data --patch-id 036019 --output-dir output --roof-selection-mode osm_tags_only --random-hipped
```

## Test Data Location

```
test_data/
├── h036019.txt      # Patch metadata (UTM zone, translation offsets)
├── h036019.obj      # Terrain mesh
└── map_21.osm       # OSM building data
```

## Output Files

```
output/
├── o036019_LOD0.obj     # Detailed mesh (with 0.5m roof overhang)
├── o036019_LOD1.obj     # Simplified mesh (no overhang)
├── o036019_report.json  # Statistics and configuration used
└── o036019.log          # Detailed processing log
```

## Key CLI Arguments

| Argument | Description |
|----------|-------------|
| `--patch-dir` | Directory with input files (required) |
| `--patch-id` | Patch ID like 036019 (required) |
| `--output-dir` | Output directory (default: ./output) |
| `--verbose` | Enable debug logging |
| `--roof-selection-mode` | `geometry` (default) or `osm_tags_only` |
| `--random-hipped` | Mix 50% gabled / 50% hipped for testing |
| `--debug-osm-id <id>` | Process single building by OSM ID |

## Project Structure

```
condor_buildings/
├── main.py              # CLI entry point - START HERE
├── config.py            # All configuration constants
├── models/              # Data models (BuildingRecord, MeshData, etc.)
├── io/                  # File I/O (OSM parser, OBJ exporter)
├── processing/          # Footprint analysis, floor Z solver
└── generators/          # Mesh generation (walls, roofs, UV mapping)
    ├── walls.py
    ├── roof_gabled.py
    ├── roof_hipped.py
    ├── roof_flat.py
    └── uv_mapping.py
```

## Documentation

- `docs/TECHNICAL_DOCUMENTATION.md` - Complete technical reference
- `docs/CHANGELOG_*.md` - Detailed changelogs for each version

## Current Version

v0.3.6 - Phase 2 complete (UV mapping + texture atlas + hipped roofs with Z fix)
