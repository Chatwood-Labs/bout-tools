# bout-tools

**bout-tools** is a collection of lightweight, focused utilities designed to support simulations and data workflows built around the **BOUT++ 5.x** framework.

Many of the tools in this repository were originally developed for internal research and simulation workflows and have been refined and generalised for broader public use.

The emphasis is on:

- practical utility
- scientific correctness
- reproducibility
- and clean integration into real MHD and plasma simulation workflows

Tools are added gradually as they reach a stable, well-documented state suitable for public release.

## Included Tools

Each tool is fully documented in its own directory.

### 1. Axisymmetric Tokamak Grid Generator - v1.1-public

Analytic geometry generator supporting:

- circular and shaped tokamak equilibria
- elongation (κ) and triangularity (δ)
- analytic magnetic field construction
- metric tensor, curvature, and Jacobian calculations
- BOUT++ compatible netCDF output

Located in:

```bash
bout_tokamak_grid/
```

This tool is intended for generating clean, well-conditioned tokamak-style grids suitable for testing, prototyping, and controlled simulation studies.

### 2. BOUT++ Tokamak Grid Diagnostics Utility - v1.0-public

Validation and visualisation tool for BOUT++ grid files, with a focus on tokamak-style axisymmetric geometry.

Provides:

- automated sanity checks on metric tensors, Jacobians, and geometry
- detection of common failure modes (sign flips, singularities, non-finite metrics, etc.)
- consistency checks between det(g) and J conventions
- optional magnetic field and shift-angle diagnostics
- interactive HTML reports with embedded plots
- machine-readable JSON output for CI and batch workflows

Located in:

```bash
bout_tokamak_grid_diagnostics/
```

This tool is designed to catch the class of errors where a grid “looks fine” but causes solvers to fail or behave pathologically.

It is intended as a **diagnostic aid**, not a replacement for full equilibrium validation.

## Philosophy

This repository aims to provide tools that are:

- **practical** - built to solve real workflow problems
- **scientifically meaningful** - not just technically correct
- **well documented** - usable without reverse engineering
- **reproducible** - deterministic behaviour and clear provenance
- **unopinionated** - easy to integrate into existing pipelines

The tools here are intentionally small and focused. This is not a framework. It is a toolbox.

## Roadmap (General)

Future additions may include:

- additional diagnostics and visualisation helpers
- extended analytic geometry tools
- workflow automation utilities
- small support scripts for common BOUT++ simulation tasks

The roadmap is intentionally flexible. Tools appear here as they mature and stabilise.

## Contributing

Issues and pull requests are welcome.

For substantial feature proposals or structural changes, please open an issue first so the design can be discussed.

This repository favours clarity and stability over rapid feature growth.

## License

Released under the MIT License, unless otherwise noted in individual tool directories.
