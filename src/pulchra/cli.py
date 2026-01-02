"""
Command-line interface for PULCHRA.

Provides the `pulchra` command for reconstructing protein structures.
"""

import sys
from pathlib import Path

import click

from pulchra import __version__


@click.command()
@click.argument("pdb_file", type=click.Path(exists=True))
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
@click.option("-c", "--no-ca-optimize", is_flag=True, help="Skip C-alpha optimization")
@click.option("-b", "--no-backbone", is_flag=True, help="Skip backbone reconstruction")
@click.option(
    "-s", "--no-sidechains", is_flag=True, help="Skip sidechain reconstruction"
)
@click.option(
    "-o", "--no-xvol", is_flag=True, help="Skip excluded volume optimization"
)
@click.option("-z", "--no-chirality", is_flag=True, help="Skip chirality check")
@click.option("-q", "--optimize-hbonds", is_flag=True, help="Optimize backbone H-bonds")
@click.option("-n", "--center", is_flag=True, help="Center chain at origin")
@click.option("-g", "--pdbsg", is_flag=True, help="Use PDB-SG input format (CA + SC)")
@click.option("-p", "--detect-cispro", is_flag=True, help="Auto-detect cis-prolines")
@click.option("-r", "--random-start", is_flag=True, help="Start from random CA chain")
@click.option(
    "-e", "--amber-order", is_flag=True, help="Use AMBER backbone atom ordering"
)
@click.option("-u", "--max-shift", type=float, default=0.5, help="Max CA shift (A)")
@click.option(
    "-i",
    "--initial",
    type=click.Path(exists=True),
    help="Initial CA coordinates file",
)
@click.option("-t", "--trajectory", is_flag=True, help="Save optimization trajectory")
@click.option("-O", "--output", type=click.Path(), help="Output file path")
@click.option("--version", is_flag=True, help="Show version and exit")
def main(
    pdb_file,
    verbose,
    no_ca_optimize,
    no_backbone,
    no_sidechains,
    no_xvol,
    no_chirality,
    optimize_hbonds,
    center,
    pdbsg,
    detect_cispro,
    random_start,
    amber_order,
    max_shift,
    initial,
    trajectory,
    output,
    version,
):
    """
    PULCHRA: Protein Chain Restoration Algorithm

    Reconstructs full-atom protein models from C-alpha traces or
    reduced representations.

    Example usage:

        pulchra input_ca.pdb

        pulchra -v input.pdb -O output.pdb

        pulchra --pdbsg --no-xvol input_sg.pdb
    """
    if version:
        click.echo(f"PULCHRA version {__version__}")
        return

    # Determine output filename
    if output is None:
        input_path = Path(pdb_file)
        output = str(input_path.stem) + ".rebuilt.pdb"

    if verbose:
        click.echo(f"PULCHRA v{__version__}")
        click.echo(f"Input: {pdb_file}")
        click.echo(f"Output: {output}")
        click.echo()

    try:
        from pulchra import Pulchra

        # Create PULCHRA instance with options
        p = Pulchra(
            verbose=verbose,
            ca_optimize=not no_ca_optimize,
            rebuild_backbone=not no_backbone,
            rebuild_sidechains=not no_sidechains,
            optimize_xvol=not no_xvol,
            check_chirality=not no_chirality,
            optimize_hbonds=optimize_hbonds,
            use_pdbsg=pdbsg,
            detect_cispro=detect_cispro,
            random_start=random_start,
            center_chain=center,
            ca_max_shift=max_shift,
            rearrange_backbone=amber_order,
        )

        # Run reconstruction
        molecule = p.reconstruct(pdb_file, output)

        if verbose:
            click.echo()
            click.echo(f"Reconstruction complete!")
            click.echo(f"  Residues: {molecule.nres}")
            click.echo(f"  Atoms: {molecule.natoms}")
            click.echo(f"  Output: {output}")

    except FileNotFoundError as e:
        click.echo(f"Error: {e}", err=True)
        click.echo(
            "\nData files not found. Please run the data conversion script:",
            err=True,
        )
        click.echo("  uv run python scripts/convert_data.py", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        if verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
