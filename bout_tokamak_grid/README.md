# Axisymmetric Tokamak Grid Generator for BOUT++ 5.x (**Version: 1.1-Public**)

This script generates an axisymmetric tokamak geometry grid compatible with **BOUT++ 5.x**, which is a framework for simulating edge-plasma interactions in tokamaks. It supports both **circular geometry** and **shaped equilibria** via **elongation (κ)** and **triangularity (δ)**, making it versatile for different tokamak configurations. The grid generation includes the calculation of the **coordinate mapping**, **metric tensor**, **Jacobian**, **magnetic geometry**, and **field-aligned shift angle**.

The grid generation follows the large-aspect-ratio tokamak assumption (i.e. the major radius **R₀** is treated as constant). This approximation is widely used in theoretical tokamak models and is fully compatible with BOUT++'s grid loading mechanism.

## Installation

Before running the script, ensure that the following dependencies are installed:

### Dependencies:
- **NumPy**: For numerical computations and array handling.
- **SciPy**: For integration functions (cumulative trapezoid integration).
- **NetCDF4**: For reading and writing netCDF files.

To install the dependencies, run:

```bash
pip install numpy scipy netCDF4
```

## Usage
### Command Line Arguments:

To run the script, use the following command structure:

```bash
python bout_tokamak_grid_generator.py \
    --R0 6.2 \
    --a 2.0 \
    --B0 5.3 \
    --kappa 1.7 \
    --delta 0.33 \
    --q0 1.05 \
    --qa 3.5 \
    --qform cubic \
    --nx 64 \
    --ny 64 \
    --nz 128 \
    --outfile grid.nc
```

### Description of Arguments:

- **&nbsp;--R0 (default: 6.2):** Major radius of the tokamak in meters.
- **&nbsp;--a (default: 2.0):** Minor radius of the tokamak in meters.
- **&nbsp;--B0 (default: 5.3):** Toroidal magnetic field at R0 in Tesla.
- **&nbsp;--xmin_frac (default: 0.1):** Fraction of minor radius to avoid r = 0 singularity.
- **&nbsp;--kappa (default: 1.7):** Elongation factor (1.7 corresponds to ITER-ish baseline). Larger values elongate the plasma.
- **&nbsp;--delta (default: 0.33):** Triangularity factor (0.33 corresponds to ITER-ish baseline).
- **&nbsp;--q0 (default: 1.05):** Central safety factor.
- **&nbsp;--qa (default: 3.5):** Edge safety factor at r = a.
- **&nbsp;--qform (default: "quadratic"):** Functional form of the safety factor profile. Options: "quadratic", "linear", or "cubic".
- **&nbsp;--nx (default: 64):** Number of grid points in the radial direction.
- **&nbsp;--ny (default: 64):** Number of grid points in the poloidal direction.
- **&nbsp;--nz (default: 128):** Number of grid points in the toroidal direction.
- **&nbsp;--precision (default: "f8"):** Floating-point precision. Options: "f4" or "f8".
- **&nbsp;--outfile (default: "grid.nc"):** Output netCDF filename where the grid data will be saved.
- **&nbsp;--curvature (default: "exact"):** Curvature model for the tokamak grid. Options: "exact", "simple", or "none".

### Example Usage:

To generate a grid for a tokamak with a major radius of 6.2 m, a minor radius of 2.0 m, and a toroidal field of 5.3 T, with an elongation (kappa) of 1.7, and triangularity (delta) of 0.33:

```bash
python bout_tokamak_grid_generator.py \
    --R0 6.2 \
    --a 2.0 \
    --B0 5.3 \
    --kappa 1.7 \
    --delta 0.33 \
    --q0 1.05 \
    --qa 3.5 \
    --qform cubic \
    --nx 128 \
    --ny 128 \
    --nz 128 \
    --outfile grid.nc
```

This command will create a netCDF file **grid.nc** containing the generated grid.

## Output

The script generates a netCDF file with the following datasets:

- x, y, z: The grid points in the radial, poloidal, and toroidal directions.
- xcoord, ycoord, zcoord: The 3D coordinate arrays.
- dx, dy, dz: The 3D spacing arrays in each direction.
- R, Z: The radial and poloidal coordinates of the plasma surface.
- g11, g22, g33, g12, g13, g23: Covariant components of the metric tensor.
- g_11, g_22, g_33, g_12, g_13, g_23: Contravariant components of the metric tensor.
- J: The Jacobian determinant.
- surfvol: The differential volume element.
- shiftAngle: The field-aligned shift angle.
- Bxy, Bpxy: The total magnetic field magnitude and the physical poloidal field.
- G1, G2: The curvature components along eₜₕᵉₜₐ and eᵣ.

This output can be directly used with BOUT++ 5.x for simulations.

## Explanation of Key Code Choices

### 1. Shaping Parameters (κ and δ)

- **κ (Elongation):** This parameter controls how elongated the plasma is. Values greater than 1.0 stretch the plasma vertically. While this can lead to numerical instability in extreme cases (e.g., κ > 3), it's generally safe in the ranges you're using.
- **δ (Triangularity):** This controls the "D-shape" of the plasma cross-section. A value of 0.0 gives a circular cross-section, while nonzero values create triangular shaping. The script uses a simple model, where θ -> θ + δ * sin(θ), which is numerically stable and commonly used in tokamak modeling.

### 2. Curvature Calculation

The curvature is calculated in three modes:

- **exact:** Uses full numerical differentiation of the magnetic field components and Christoffel symbols . This is computationally expensive but accurate.
- **simple:** A simple analytic approximation is used for tokamaks with large aspect ratios, yielding a much faster but less precise result.
- **none:** disables curvature calculations entirely. This is useful for specific cases where curvature isn't needed.

### 3. Safety Factor Profile

The **q-profile** is calculated using different forms: quadratic, linear, or cubic. This defines the safety factor (**q**) as a function of radial position. The "quadratic" form is the default and is often used in standard tokamak simulations. If you need smoother or more flexible profiles, "linear" and "cubic" options are available.

### 4. Grid Generation Method

The grid is generated in 3D, with the radial direction using a non-uniform distribution to avoid singularities at the plasma center. This is done by avoiding **r = 0** by a fraction of the minor radius (**xmin_frac**). The radial coordinate starts from a small value, ensuring the grid remains numerically stable.

### 5. Jacobian and Metrics

The Jacobian is calculated using the coordinate basis vectors. This is crucial for understanding the geometry and volume elements of the plasma grid. We ensure that the Jacobian is always positive, preventing any singularities in the grid.

## Known Issues / Limitations

- **Numerical Stability with Extreme Shaping:** If κ or δ are too large, the geometry may become numerically unstable. In these cases, the script will issue warnings or raise errors.
- **Performance with Large Grids:** The "exact" curvature mode can be computationally expensive for large grids, and performance may degrade. Consider using the "simple" mode for larger grid sizes.

## License

This software is released under the MIT License. Feel free to modify and redistribute it, but please provide attribution to Chatwood Labs Ltd.

## Contributing

Issues and pull requests are welcome.  
For large feature additions or major changes, please open an issue first to discuss the proposed implementation.