# Condor Buildings Generator - Quick Reference for Claude

## Project Overview

This is a Python pipeline that generates 3D building meshes from OpenStreetMap (OSM) data for Condor 3 flight simulator. It produces OBJ files with UV coordinates for texture mapping.

**Available as:** CLI tool + Blender addon (v0.6.1+)

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

## Fake Condor Structure (for testing Blender addon without Condor3)

```
fake_condor/
└── Landscapes/
    └── TestLandscape/
        └── Working/
            ├── Heightmaps/
            │   ├── h036019.txt
            │   └── h036019.obj
            └── Autogen/  (output folder)
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
├── __init__.py          # Package version + Blender addon registration
├── main.py              # CLI entry point - START HERE
├── config.py            # All configuration constants + PipelineConfig
├── models/              # Data models (BuildingRecord, MeshData, etc.)
├── io/                  # File I/O (OSM parser, OBJ exporter)
├── processing/          # Footprint analysis, floor Z solver
├── generators/          # Mesh generation (walls, roofs, UV mapping)
│   ├── walls.py
│   ├── roof_gabled.py
│   ├── roof_hipped.py
│   ├── roof_flat.py
│   └── uv_mapping.py
└── blender/             # Blender addon (v0.5.0+)
    ├── __init__.py      # Addon initialization
    ├── properties.py    # UI properties (Condor path, landscape, patch range)
    ├── operators.py     # Import/clear operators with Condor workflow
    ├── panels.py        # UI panels (sidebar)
    ├── mesh_converter.py # MeshData → Blender mesh conversion
    └── osm_downloader.py # Download OSM data from Overpass API
```

## Documentation

- `docs/TECHNICAL_DOCUMENTATION.md` - Complete technical reference
- `docs/CHANGELOG_*.md` - Detailed changelogs for each version

## Blender Addon Usage (v0.6.1+)

The addon supports the real Condor folder structure:

1. **Condor Directory**: Path to Condor3 installation (e.g., `C:\Condor3`)
2. **Landscape**: Auto-detected from `Landscapes/` folder
3. **Patch Range**: X/Y min/max for batch processing, or single patch mode
4. **OSM Source**: Download from Overpass API or use local file
5. **Output**: Save to `Working/Autogen/` and/or import to Blender

### Programmatic Usage

```python
from condor_buildings.main import run_pipeline
from condor_buildings.config import PipelineConfig

config = PipelineConfig(
    patch_id="036019",
    patch_dir="/path/to/heightmaps",
    zone_number=0,  # Auto-loaded from h*.txt
    translate_x=0.0,
    translate_y=0.0,
    osm_path="/path/to/downloaded.osm",  # Optional: explicit OSM path
)

# Memory mode returns meshes directly (for Blender)
result = run_pipeline(config, output_mode="memory")
# result.lod0_meshes, result.lod1_meshes contain MeshData objects
```

### OSM Downloader

```python
from condor_buildings.blender.osm_downloader import download_osm_for_patch
from condor_buildings.io.patch_metadata import load_patch_metadata

metadata = load_patch_metadata("h036019.txt")
result = download_osm_for_patch(metadata, output_dir="./", filename_prefix="map")
# result.filepath contains path to downloaded .osm file
```

## Creating ZIP for Blender Installation

```bash
# From project root
powershell -Command "Compress-Archive -Path 'condor_buildings' -DestinationPath 'condor_buildings_v0.6.1.zip' -Force"
```

## Runtime Configuration (v0.6.1+)

Configure generator parameters before processing:

```python
from condor_buildings.generators import configure_generator

# Override defaults before processing
configure_generator(
    gable_height=4.0,           # Roof peak height (default 3.0m)
    roof_overhang_lod0=0.3,     # Overhang distance (default 0.5m)
    floor_z_epsilon=0.5,        # Sink below terrain (default 0.3m)
    gabled_max_floors=3,        # Max floors for gabled (default 2)
    polyskel_max_vertices=15,   # Max verts for polyskel (default 12)
)
```

## Current Version

v0.6.1 - Configurable parameters in Blender UI:
- Gable height, roof overhang, floor Z offset adjustable in UI
- Max floors for gabled/hipped roofs configurable
- Min rectangularity and polyskel max vertices in Advanced panel
- Random seed for reproducible results
- Runtime configuration system (`configure_generator()`)

Previous versions:
- v0.6.0: Polyskel hipped roofs for 5-12 vertex buildings
- v0.5.0: Condor workflow support (auto-detect landscapes, download OSM, batch processing)
