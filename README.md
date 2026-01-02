# PULCHRA - Protein Chain Restoration Algorithm

A modern Python implementation of [PULCHRA](https://github.com/euplotes/pulchra) for reconstructing full-atom protein models from reduced C-alpha representations.

## Installation

```bash
pip install pulchra
```

### From Source

```bash
# Clone and enter the directory
cd pulchra-py

# Install with uv
uv sync
```

## Data Setup

Before using PULCHRA, you need to convert the C header data files to NumPy format:

```bash
uv run python scripts/convert_data.py
```

This reads the `nco_data.h`, `rot_data_coords.h`, and `rot_data_idx.h` files from the project root directory and creates the corresponding `.npy` files in `src/pulchra/data/`.

## Usage

### Command Line

```bash
# Basic reconstruction
uv run pulchra input_ca.pdb

# With verbose output
uv run pulchra -v input.pdb

# Specify output file
uv run pulchra input.pdb -O output.pdb

# Skip certain steps
uv run pulchra --no-xvol --no-chirality input.pdb

# Use PDB-SG format (CA + sidechain centers)
uv run pulchra --pdbsg input_sg.pdb

# Optimize H-bonds (optional refinement)
uv run pulchra -q input.pdb
```

### Python API

```python
from pulchra import Pulchra

# Basic usage
p = Pulchra()
molecule = p.reconstruct("input_ca.pdb", "output.pdb")

# With options
p = Pulchra(
    verbose=True,
    ca_optimize=True,
    optimize_hbonds=True
)
molecule = p.reconstruct("input.pdb")

# From raw coordinates
import numpy as np
ca_coords = np.array([[x1, y1, z1], [x2, y2, z2], ...])
sequence = "MKWVTFISLLLF..."
molecule = p.reconstruct_coords(ca_coords, sequence)
```

## Options

| Option | Description |
|--------|-------------|
| `-v, --verbose` | Verbose output |
| `-c, --no-ca-optimize` | Skip C-alpha optimization |
| `-b, --no-backbone` | Skip backbone reconstruction |
| `-s, --no-sidechains` | Skip sidechain reconstruction |
| `-o, --no-xvol` | Skip excluded volume optimization |
| `-z, --no-chirality` | Skip chirality check |
| `-q, --optimize-hbonds` | Optimize backbone H-bonds |
| `-n, --center` | Center chain at origin |
| `-g, --pdbsg` | Use PDB-SG input format |
| `-p, --detect-cispro` | Auto-detect cis-prolines |
| `-r, --random-start` | Start from random CA chain |
| `-e, --amber-order` | AMBER backbone atom ordering |
| `-u, --max-shift` | Max CA shift in Angstroms |
| `-O, --output` | Output file path |

## Algorithm Overview

PULCHRA reconstructs protein structures through the following steps:

1. **C-alpha Optimization**: Adjusts CA positions to satisfy ideal geometry using steepest descent
2. **Backbone Reconstruction**: Places N, C, O atoms using statistical templates
3. **Sidechain Reconstruction**: Places sidechain atoms using rotamer library
4. **Excluded Volume Optimization**: Fixes steric clashes by selecting alternative rotamers
5. **Chirality Check**: Detects and fixes D-amino acids
6. **H-bond Optimization** (optional): Refines backbone to improve hydrogen bonding

## Development

```bash
# Install dev dependencies
uv sync --dev

# Run tests
uv run pytest

# Type checking
uv run mypy src/pulchra

# Linting
uv run ruff check src/
```

## License

MIT License - see LICENSE file.

## Credits

Original PULCHRA algorithm by Piotr Rotkiewicz.
Python implementation based on PULCHRA v3.06.
