"""
Mesh generators for Condor Buildings Generator.

Contains wall generator, flat roof generator, gabled roof generator,
and the main building generator that orchestrates them all.
"""

from .walls import generate_walls, WallGeneratorConfig
from .roof_flat import generate_flat_roof, FlatRoofConfig
from .roof_gabled import generate_gabled_roof, GabledRoofConfig
from .building_generator import (
    generate_building,
    generate_building_lod0,
    generate_building_lod1,
    BuildingGeneratorResult,
)

__all__ = [
    'generate_walls',
    'WallGeneratorConfig',
    'generate_flat_roof',
    'FlatRoofConfig',
    'generate_gabled_roof',
    'GabledRoofConfig',
    'generate_building',
    'generate_building_lod0',
    'generate_building_lod1',
    'BuildingGeneratorResult',
]
