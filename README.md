# bout-tools

**bout-tools** is a collection of lightweight utilities designed to support simulations and data workflows built around the **BOUT++ 5.x** framework. Many of the utilities in this repository were originally developed for internal simulation workflows and have been refined and generalised for broader public use.

The repository focuses on providing clean, practical tools that simplify geometry generation, pre-processing, post-processing, and general MHD modeling workflows.

Tools are added gradually as they reach a stable, polished state suitable for public release.

## Currently Included

Full documentation for each tool is provided in its own directory.

**Axisymmetric Tokamak Grid Generator - v1.1-public**

A fully analytic geometry generator supporting:

- circular and shaped tokamak equilibria
- elongation (κ) and triangularity (δ)
- analytic magnetic field construction
- metric tensor, curvature, and Jacobian calculations
- BOUT++ compatible netCDF output

Documentation and examples can be found in:

```bash
bout_tokamak_grid/
```

## Philosophy

This repository aims to provide tools that are:

- practical and easy to use
- scientifically meaningful
- well documented and reproducible
- focused on solving real workflow needs

Development is incremental by design; tools appear here once they are tidy, generalised, and broadly useful.

## Roadmap (General)

Future additions may include:

- diagnostic/visualisation helpers
- extended analytic geometry tools
- workflow automation scripts
- various small utilities supporting BOUT++ simulations

This roadmap is intentionally flexible; features are added as they mature.

## Contributing

Issues and pull requests are welcome.

For substantial feature proposals or structural changes, please open an issue so they can be discussed beforehand.

## License

Released under the MIT License, unless otherwise noted in individual tool directories.