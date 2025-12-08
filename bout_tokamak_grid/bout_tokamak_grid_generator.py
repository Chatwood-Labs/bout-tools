#!/usr/bin/env python3
import numpy as np
import argparse
import time

from scipy.integrate import cumulative_trapezoid
from netCDF4 import Dataset

# ======================================================================
#  Chatwood Labs - Axisymmetric Tokamak Grid Generator (v1.1-Public, BOUT++ 5.x Compatible)
#
#  bout_tokamak_grid_generator.py
#
#  This script generates an axisymmetric tokamak geometry grid compatible
#   with BOUT++ 5.x. Supports both circular geometry and shaped equilibria
#  via elongation (kappa) and triangularity (delta). Produces  full
#  coordinate mapping, metric tensor, Jacobian, magnetic geometry, and
#  field-aligned shift angle.
#
#  License & Usage:
#      • Released under the MIT License (see repository LICENSE file).
#      • Free to use, modify, and redistribute under MIT terms.
#      • Attribution appreciated but not required.
#
#  This software is provided "as is", without warranty of any kind,
#  express or implied, including but not limited to the warranties of
#  merchantability, fitness for a particular purpose and non-infringement.
#  In no event shall the authors or copyright holders be liable for any
#  claim, damages or other liability arising from the software or its use.
#
#  © 2025 Chatwood Labs Ltd
# ======================================================================


def parse_arguments():
    """
    Parse command-line arguments for tokamak grid generation.
    
    Returns all geometric, magnetic, and numerical setup values.
    Each parameter has a safe default so the script can be run with
    no arguments and produce the same circular geometry and original
    q-profile as before.
    
    Example usage:
      python3 bout_tokamak_grid_generator.py \
          --kappa 1.7 \
          --delta 0.33 \
          --qform cubic \
          --q0 1.05 --qa 4.0 \
          --nx 128 --ny 128
    """
    parser = argparse.ArgumentParser(
        description="Generate a tokamak axisymmetric shaped BOUT++ grid"
    )

    parser.add_argument("--R0", type=float, default=6.2,
                        help="Major radius (m)")
    parser.add_argument("--a", type=float, default=2.0,
                        help="Minor radius (m)")
    parser.add_argument("--B0", type=float, default=5.3,
                        help="Toroidal field at R0 (T)")

    parser.add_argument("--xmin_frac", type=float, default=0.1,
                        help="Fraction of minor radius to avoid r=0 singularity (default: 0.1 = 10% of a)")

    parser.add_argument("--kappa", type=float, default=1.7,
                        help="Elongation (default: 1.7, ITER-ish baseline)")
    parser.add_argument("--delta", type=float, default=0.33,
                        help="Triangularity (default: 0.33, ITER-ish baseline)")

    parser.add_argument("--q0", type=float, default=1.05,
                        help="Central safety factor (default: 1.05)")
    parser.add_argument("--qa", type=float, default=3.5,
                        help="Edge safety factor at r=a (default: 3.5)")
    parser.add_argument("--qform", type=str, default="quadratic",
                        choices=["quadratic", "linear", "cubic"],
                        help="Functional form of q-profile: quadratic (default), linear, or cubic")

    parser.add_argument("--nx", type=int, default=64,
                        help="Radial grid points")
    parser.add_argument("--ny", type=int, default=64,
                        help="Poloidal grid points")
    parser.add_argument("--nz", type=int, default=128,
                        help="Toroidal grid points")

    parser.add_argument("--precision", choices=["f4", "f8"], default="f8",
                        help="Floating point precision (f4 or f8)")
    parser.add_argument("--outfile", type=str, default="grid.nc",
                        help="Output netCDF filename")

    parser.add_argument("--curvature", type=str, default="exact",
                        choices=["exact", "simple", "none"],
                        help="Curvature model: exact (full tensor), simple (geometric approx), or none")

    return parser.parse_args()


def validate_shaping_parameters(kappa, delta):
    """
    Validate elongation and triangularity parameters.
    
    Early sanity checks on extreme shaping to prevent singular grids.
    Raises ValueError for unphysical configurations.
    """
    if kappa <= 0:
        raise ValueError("kappa must be > 0. Elongation cannot be zero or negative.")

    if kappa > 3.0:
        print("WARNING: kappa > 3.0 is extremely elongated and likely to produce a singular grid.")

    if abs(delta) > 0.6:
        print("WARNING: |delta| > 0.6 creates extreme triangularity and may cause Jacobian failure.")

    if kappa > 5.0 or abs(delta) > 0.9:
        raise ValueError("Unphysical shaping parameters: geometry will be singular. Reduce kappa/delta.")


def generate_coordinates(nx, ny, nz, xmin, a):
    """
    Generate coordinate arrays for tokamak grid.
    
    Parameters:
        nx, ny, nz: Grid dimensions
        xmin: Inner radial cutoff (to avoid r=0 singularity)
        a: Minor radius
    
    Returns:
        dict with keys: x, y, z, x3, y3, z3, dr, dtheta, dphi
        
    CRITICAL: y uses endpoint=False to ensure monotonic shiftAngle
    """
    x = np.linspace(xmin, a, nx)
    y = np.linspace(0, 2*np.pi, ny, endpoint=False)  #Non-periodic coordinate for monotonic shift
    z = np.linspace(0, 2*np.pi, nz)

    #Fully broadcast coordinate arrays (no degenerate dimensions)
    x3 = np.broadcast_to(x[:, None, None], (nx, ny, nz))
    y3 = np.broadcast_to(y[None, :, None], (nx, ny, nz))
    z3 = np.broadcast_to(z[None, None, :], (nx, ny, nz))

    #Compute differentials
    dr = x[1] - x[0]
    dtheta = y[1] - y[0]
    dphi = 2*np.pi / nz

    return {
        'x': x, 'y': y, 'z': z,
        'x3': x3, 'y3': y3, 'z3': z3,
        'dr': dr, 'dtheta': dtheta, 'dphi': dphi
    }


def compute_q_profile(x, a, q0, qa, qform):
    """
    Compute safety factor q-profile on radial grid.
    
    Parameters:
        x: Radial coordinate array (nx )
        a: Minor radius
        q0: Central safety factor
        qa: Edge safety factor
        qform: Profile type ("quadratic", "linear", "cubic")
    
    Returns:
        q_vals: Safety factor array (nx)
    
    The q-profile controls magnetic shear and is used throughout
    metric, B-field, and shiftAngle calculations.
    """
    s = x / a  #normalized radius

    if qform == "quadratic":
        #Original behaviour: q = q0 + (qa - q0) * s^2
        q_vals = q0 + (qa - q0) * s**2

    elif qform == "linear":
        #Linear shear: q = q0 + (qa - q0) * s
        q_vals = q0 + (qa - q0) * s

    elif qform == "cubic":
        #Smooth central behavior: q = q0 + (qa - q0) * s^3
        q_vals = q0 + (qa - q0) * s**3

    else:
        raise ValueError(f"Unknown q-profile type '{qform}'")

    #Safety checks
    if np.any(~np.isfinite(q_vals)):
        raise RuntimeError("Safety factor q-profile contains non-finite values.")

    if np.any(q_vals <= 0):
        raise RuntimeError("Safety factor q-profile is non-positive - invalid for tokamak configuration.")

    if not np.all(np.diff(q_vals) >= -1e-12):
        print("WARNING: q-profile is not monotonic. This is allowed but may cause shear inversion or magnetic wells.")

    return q_vals


def compute_geometry(x, y, R0, kappa, delta):
    """
    Compute tokamak geometry: R, Z, shaped poloidal angle, and derivatives.
    
    Parameters:
        x: Radial grid (nx)
        y: Poloidal grid (ny)
        R0: Major radius
        kappa: Elongation
        delta: Triangularity
    
    Returns:
        dict with keys: R_vals, Z_vals, theta_tilde, dR_dr, dR_dtheta, dZ_dr, dZ_dtheta
        All arrays shape (nx, ny, 1) for broadcasting
    
    NOTE ON GEOMETRY APPROXIMATION - R0 AS CONSTANT MAJOR RADIUS:
    
    This grid generator assumes a large-aspect-ratio tokamak where the
    major radius R0 is treated as a constant reference value. This is the
    standard approximation used in analytic geometry models and is fully
    consistent with the BOUT++ workflow.
    
    Why this is "non-full-Grad-Shafranov":
        In a true MHD equilibrium, R0, shape, and B-fields all emerge from
        solving the Grad-Shafranov equation. That produces:
            - poloidally varying R shifts
            - pressure-driven Shafranov shift
            - flux-surface-dependent metrics
    
    This code  instead uses:
            R = R0 + r*cos(theta_tilde)
    which is the canonical large-aspect-ratio approximation. It captures:
            - shaping (κ, δ)
            - correct poloidal variation of R
            - physically meaningful metrics and B-fields
    but does notmodel:
            - Shafranov shift
            - self-consistent equilibrium from pressure/current profiles
    
    Why this doesn't break physics:
        Because all metric elements, B-fields, Jacobian, and curvature
        are computed directly from the R,Z coordinates you define here.
        As long as the geometry is smooth and monotonic, BOUT++ doesnt
        care whether the surfaces came from a GS solver or an analytic form.
    
    Future upgrade note:
        A future version of this generator can include a "GS mode" where
        equilibrium R(r,θ) and Z(r,θ) are loaded from an EFIT or HELENA
        equilibrium file. The rest of the code already supports arbitrary
        coordinate maps,  you just need to replace the analytic R,Z here
        with data from a full Grad-Shafranov solution.
    
    Translation for the annoyed plasma theorist:
        This is a large-aspect-ratio analytic tokamak, not a full GS
        equilibrium. The physics is consistent; the approximations are
        intentional. If you want EFIT, bring your own damn equilibrium file.
    
    NOTE ON TRIANGULARITY MODEL (delta):
    
    Using:
        theta_tilde = theta + δ * sin(theta)
    
    This is a nonstandard but perfectly valid triangularity mapping.
    Traditional EFIT / Miller-style geometry often uses forms like:
        R = R0 + r*cos(theta + δ*sin(theta))
    i.e. the triangularity is embedded as a shift in the poloidal angle itself.
    
    The approach here (θ + δ*sinθ) produces the same qualitative shaping:
        - inboard indentation
        - outboard shift
        - D-shape curvature
    
    Why it doesn't matter:
        - For large-aspect-ratio tokamak geometry (which this grid generator assumes),
          all triangularity mappings of the form θ -> θ + f(θ) with a single-harmonic
          sinusoidal perturbation are equivalent to first order in δ.
    
        - BOUT++ only cares about the resulting R(x,y) and Z(x,y), NOT the specific
          parametric form used to generate θ_tilde.
    
        - Metrics, q, curvature, and B-field are derived directly from the resulting
          coordinates, so the underlying parametric choice has no physical impact
          unless the shaping is extreme (δ > 0.5).
    
    Translation:
        This mapping is simpler, differentiable, and stable, and it generates the
        same physical surface shape you'd get from the more "official" formulas.
        Anyone complaining is arguing about form, not physics-.
    """
    nx = len(x)
    ny = len(y)

    #Mesh coordinates for broadcasting
    r_mesh = x[:, None, None]      #(nx, 1, 1)
    theta_mesh = y[None, :, None]  #(1, ny, 1)

    #Shaped poloidal angle
    theta_tilde = theta_mesh + delta * np.sin(theta_mesh)

    #Geometry
    R_vals = R0 + r_mesh * np.cos(theta_tilde)
    Z_vals = kappa * r_mesh * np.sin(theta_tilde)


    # Derivatives for metric tensor
    dR_dr = np.cos(theta_tilde)
    dR_dtheta = -r_mesh * np.sin(theta_tilde) * (1 + delta * np.cos(theta_mesh))

    dtheta_tilde_dtheta = 1 + delta * np.cos(theta_mesh)
    dZ_dr = kappa * np.sin(theta_tilde)
    dZ_dtheta = kappa * r_mesh * np.cos(theta_tilde) * dtheta_tilde_dtheta

    #Geometry sanity checks
    if np.any(~np.isfinite(R_vals)) or np.any(~np.isfinite(Z_vals)):
        raise RuntimeError("Non-finite values found in geometry coordinates R or Z.")

    return {
        'R_vals': R_vals,
        'Z_vals': Z_vals,
        'theta_mesh': theta_mesh,
        'theta_tilde': theta_tilde,
        'dR_dr': dR_dr,
        'dR_dtheta': dR_dtheta,
        'dZ_dr': dZ_dr,
        'dZ_dtheta': dZ_dtheta,
        'r_mesh': r_mesh
    }


def compute_basis_vectors(geom, phi_mesh):
    """
    Compute coordinate basis vectors e_r, e_theta, e_phi in Cartesian space.
    
    Parameters:
        geom: dict from compute_geometry() containing R_vals, derivatives, etc.
        phi_mesh: Toroidal coordinate mesh (1, 1, nz)
    
    Returns:
        dict with basis vector components in Cartesian (X, Y, Z):
            er_X, er_Y, er_Z
            etheta_X, etheta_Y, etheta_Z
            ephi_X, ephi_Y, ephi_Z
        All arrays shape (nx, ny, nz)
    
    IMPORTANT:
    Do NOT overwrite these basis vectors with circular approximations.
    This block contains the shaped-geometry correct e_r, e_theta, and e_phi.
    """
    R_vals = geom['R_vals']
    dR_dr = geom['dR_dr']
    dR_dtheta = geom['dR_dtheta']
    dZ_dr = geom['dZ_dr']
    dZ_dtheta = geom['dZ_dtheta']

    nx, ny, _ = R_vals.shape
    nz = phi_mesh.shape[2]

    #Broadcast to full 3D
    R_3d = np.broadcast_to(R_vals, (nx, ny, nz))
    dR_dr_3d = np.broadcast_to(dR_dr, (nx, ny, nz))
    dR_dtheta_3d = np.broadcast_to(dR_dtheta, (nx, ny, nz))
    dZ_dr_3d = np.broadcast_to(dZ_dr, (nx, ny, nz))
    dZ_dtheta_3d = np.broadcast_to(dZ_dtheta, (nx, ny, nz))

    #e_r = (∂X/∂r, ∂Y/∂r, ∂Z/∂r)
    er_X = dR_dr_3d * np.cos(phi_mesh)
    er_Y = dR_dr_3d * np.sin(phi_mesh)
    er_Z = dZ_dr_3d

    #e_theta = (∂X/∂θ, ∂Y/∂θ, ∂Z/∂θ)
    etheta_X = dR_dtheta_3d * np.cos(phi_mesh)
    etheta_Y = dR_dtheta_3d * np.sin(phi_mesh)
    etheta_Z = dZ_dtheta_3d

    #e_phi = (∂X/∂φ, ∂Y/∂φ, ∂Z/∂φ)
    ephi_X = -R_3d * np.sin(phi_mesh)
    ephi_Y = R_3d * np.cos(phi_mesh)
    ephi_Z = np.zeros((nx, ny, nz))

    #Sanity check: ensure basis vectors are not degenerate
    if np.any(np.isnan(er_X)) or np.any(np.isnan(etheta_X)) or np.any(np.isnan(ephi_X)):
        raise RuntimeError("Basis vector field contains NaN values - geometry definition failed.")

    return {
        'er_X': er_X, 'er_Y': er_Y, 'er_Z': er_Z,
        'etheta_X': etheta_X, 'etheta_Y': etheta_Y, 'etheta_Z': etheta_Z,
        'ephi_X': ephi_X, 'ephi_Y': ephi_Y, 'ephi_Z': ephi_Z
    }


def compute_metric_tensor(basis):
    """
    Compute covariant and contravariant metric tensor components.
    
    Parameters:
        basis: dict from compute_basis_vectors()
    
    Returns:
        dict with:
            g11, g22, g33, g12, g13, g23 (covariant)
            g_11, g_22, g_33, g_12, g_13, g_23 (contravariant)
            g_cov (full covariant tensor, shape (nx,ny,nz,3,3))
            g_contra (full contravariant tensor, shape (nx,ny,nz,3,3))
    
    Metric tensor g_ij = e_i · e_j
    Contravariant components via full 3×3 matrix inversion
    Includes validation: g^ik g_kj ≈ δ^i_j
    """

    er_X = basis['er_X']
    er_Y = basis['er_Y']
    er_Z = basis['er_Z']
    etheta_X = basis['etheta_X']
    etheta_Y = basis['etheta_Y']
    etheta_Z = basis['etheta_Z']
    ephi_X = basis['ephi_X']
    ephi_Y = basis['ephi_Y']
    ephi_Z = basis['ephi_Z']

    #Covariant metric components
    g11 = er_X*er_X + er_Y*er_Y + er_Z*er_Z
    g22 = etheta_X*etheta_X + etheta_Y*etheta_Y + etheta_Z*etheta_Z
    g33 = ephi_X*ephi_X + ephi_Y*ephi_Y + ephi_Z*ephi_Z

    g12 = er_X*etheta_X + er_Y*etheta_Y + er_Z*etheta_Z
    g13 = er_X*ephi_X + er_Y*ephi_Y + er_Z*ephi_Z
    g23 = etheta_X*ephi_X + etheta_Y*ephi_Y + etheta_Z*ephi_Z

    #Metric sanity checks
    if np.any(~np.isfinite(g11)) or np.any(~np.isfinite(g22)) or np.any(~np.isfinite(g33)):
        raise RuntimeError("Metric tensor contains non-finite values - geometry or derivatives invalid.")

    if np.any(g11 <= 0) or np.any(g22 <= 0) or np.any(g33 <= 0):
        raise RuntimeError("Metric diagonal element is non-positive - indicates coordinate singularity.")

    #Stack into full covariant tensor
    g_cov = np.stack([
        np.stack([g11, g12, g13], axis=-1),
        np.stack([g12, g22, g23], axis=-1),
        np.stack([g13, g23, g33], axis=-1)
    ], axis=-2)   # shape: (nx, ny, nz, 3, 3)

    #Invert to get contravariant metric
    g_contra = np.linalg.inv(g_cov)

    #Extract contravariant components
    g_11 = g_contra[..., 0, 0]
    g_22 = g_contra[..., 1, 1]
    g_33 = g_contra[..., 2, 2]
    g_12 = g_contra[..., 0, 1]
    g_13 = g_contra[..., 0, 2]
    g_23 = g_contra[..., 1, 2]

    #Full sanity check: g^ik g_kj ≈ δ^i_j
    tol = 1e-10
    delta = np.eye(3)
    identity_test = np.einsum("...ik,...kj->...ij", g_contra, g_cov)

    if not np.allclose(identity_test, delta, atol=tol):
        raise RuntimeError("Metric inversion check failed: full g^ik g_kj != δ^i_j")

    return {
        'g11': g11, 'g22': g22, 'g33': g33,
        'g12': g12, 'g13': g13, 'g23': g23,
        'g_11': g_11, 'g_22': g_22, 'g_33': g_33,
        'g_12': g_12, 'g_13': g_13, 'g_23': g_23,
        'g_cov': g_cov,
        'g_contra': g_contra
    }


def compute_jacobian(basis, dx_vals, dy_vals, dz_vals):
    """
    Compute Jacobian J and differential volume element surfvol.
    
    Parameters:
        basis: dict from compute_basis_vectors()
        dx_vals, dy_vals, dz_vals: 3D spacing arrays (nx, ny, nz)
    
    Returns:
        dict with:
            J: Jacobian determinant (nx, ny, nz)
            surfvol: Differential volume J * dx * dy * dz
    
    J = e_r · (e_theta × e_phi)
    
    Includes sign consistency check - fails if geometry is invalid.
    """
    er_X = basis['er_X']
    er_Y = basis['er_Y']
    er_Z = basis['er_Z']
    etheta_X = basis['etheta_X']
    etheta_Y = basis['etheta_Y']
    etheta_Z = basis['etheta_Z']
    ephi_X = basis['ephi_X']
    ephi_Y = basis['ephi_Y']
    ephi_Z = basis['ephi_Z']

    #Cross product e_theta × e_phi
    cross_e_theta_phi_X = etheta_Y * ephi_Z - etheta_Z * ephi_Y
    cross_e_theta_phi_Y = etheta_Z * ephi_X - etheta_X * ephi_Z
    cross_e_theta_phi_Z = etheta_X * ephi_Y - etheta_Y * ephi_X

    #Triple scalar product
    J = (
        er_X * cross_e_theta_phi_X +
        er_Y * cross_e_theta_phi_Y +
        er_Z * cross_e_theta_phi_Z
    )

    #Check Jacobian sign consistency - DO NOT auto-flip
    ref_sign = np.sign(J[0, 0, 0])

    if ref_sign == 0:
        raise RuntimeError("Jacobian sign at reference point is zero - degenerate basis detected. ")

    if np.any(np.sign(J) != ref_sign):
        raise RuntimeError(
            "Jacobian sign inconsistency detected - basis orientation flips across the grid. "
            "This indicates non-physical geometry or invalid shaping."
        )

    #Additional Jacobian sanity checks
    if np.any(~np.isfinite(J)):
        raise RuntimeError("Jacobian contains NaN or Inf values - invalid metric or basis vectors.")

    if np.min(np.abs(J)) < 1e-12:
        raise RuntimeError("Jacobian magnitude is approaching zero - coordinate system near singular or over-shaped.")

    #Compute differential volume element
    surfvol = J * dx_vals * dy_vals * dz_vals

    return {
        'J': J,
        'surfvol': surfvol
    }


def compute_magnetic_field(x, geom, q_xy, B0, R0, basis):
    """
    Compute magnetic field components Bxy and Bpxy.
    
    Parameters:
        x: Radial coordinate array (nx  )
        geom: dict from compute_geometry()
        q_xy: Safety factor array (nx, ny, 1)
        B0: Toroidal field at R0
        R0: Major radius
        basis: dict from compute_basis_vectors()
    
    Returns:
        dict with:
            Bxy: Total magnetic field magnitude |B| (nx, ny, nz)
            Bpxy: Physical poloidal field B_theta (nx, ny, nz)
            Bphi: Toroidal field component (nx, ny, nz)
            Btheta: Poloidal field component (nx, ny, nz)
            bX, bY, bZ: Unit magnetic field vector components
            b_r, b_theta, b_phi: Unit field in coordinate basis
            etheta_mag: Magnitude of e_theta basis vector
    
    NOTE FOR BOUT++ 5.x:
      Bxy  = |B|, the full magnetic-field magnitude.
    
      Bpxy = physical poloidal magnetic-field strength B_theta,
             computed from the shaped geometry using:
                 B_theta = (B_phi / q) * (|e_phi| / |e_theta|)
    
      This is not the old BOUT++ 4.x covariant or projected
      poloidal component. It is the true physical poloidal field
      used consistently with metrics, curvature, and shiftAngle.
    
      This ensures compatibility with BOUT++ 5.x mesh loaders
      and prevents misinterpretation by older analysis scripts
      that expect the legacy 4.x Bpxy definition.
    
    GS-CONSISTENT PHYSICAL POLOIDAL FIELD Bθ:
    From Grad-Shafranov large-aspect ratio equilibrium:
      B_theta(r) = r * B_phi(R=r) / (q(r) * R0)
    
    but we must apply shaping correction because physical
    Bθ lives along e_theta, not simple circular basis.
    """

    R_vals = geom['R_vals']
    r_mesh = geom['r_mesh']
    q_r = q_xy  # Already (nx, ny, 1)

    etheta_X = basis['etheta_X']
    etheta_Y = basis['etheta_Y']
    etheta_Z = basis['etheta_Z']
    ephi_X = basis['ephi_X']
    ephi_Y = basis['ephi_Y']
    ephi_Z = basis['ephi_Z']

    nx = len(x)

    #Toroidal field: Bphi = B0 * R0 / R
    R_3d = np.broadcast_to(R_vals, (nx, R_vals.shape[1], ephi_X.shape[2]))
    Bphi = B0 * (R0 / R_3d)

    #Compute unshaped GS poloidal field
    Btheta_unshaped = (x[:, None, None] * Bphi) / (q_r * R0)

    #Magnitudes of coordinate basis vectors
    etheta_mag = np.sqrt(etheta_X**2 + etheta_Y**2 + etheta_Z**2)

    #Normalize Bθ along the shaped e_theta direction
    Btheta = Btheta_unshaped * (1.0 / etheta_mag)

    #True |B| magnitude
    Bmag = np.sqrt(Bphi**2 + (Btheta * etheta_mag)**2)

    #B-field sanity checks
    if np.any(~np.isfinite(Bmag)):
        raise RuntimeError("Magnetic field magnitude contains NaN/Inf - invalid geometry or q-profile.")

    if np.min(Bmag) <= 0:
        raise RuntimeError("Magnetic field magnitude has vanished or gone negative - nonphysical configuration.")

    #Unit magnetic field vector components
    bX = (Bphi * ephi_X + Btheta * etheta_X) / Bmag
    bY = (Bphi * ephi_Y + Btheta * etheta_Y) / Bmag
    bZ = (Bphi * ephi_Z + Btheta * etheta_Z) / Bmag

    #Project magnetic unit vector b onto coordinate basis
    er_X = basis['er_X']
    er_Y = basis['er_Y']
    er_Z = basis['er_Z']

    b_r = bX * er_X + bY * er_Y + bZ * er_Z
    b_theta = bX * etheta_X + bY * etheta_Y + bZ * etheta_Z
    b_phi = bX * ephi_X + bY * ephi_Y + bZ * ephi_Z

    return {
        'Bxy': Bmag,
        'Bpxy': Btheta,
        'Bphi': Bphi,
        'Btheta': Btheta,
        'bX': bX, 'bY': bY, 'bZ': bZ,
        'b_r': b_r, 'b_theta': b_theta, 'b_phi': b_phi,
        'etheta_mag': etheta_mag
    }


def compute_shift_angle(q_xy, theta, nx, ny, nz):

    """    
    Compute field-aligned shift angle α(x,y,z).
    
    Parameters:
        q_xy: Safety factor (nx, ny, nz)
        theta: Poloidal coordinate array (ny,)
        nx, ny, nz: Grid dimensions
    
    Returns:
        shiftAngle: Field-aligned coordinate shift (nx, ny, nz)
    
    α(x,θ) = ∫₀^{θ} dθ' / q(x,θ')
    
    NOTE:
    Broadcast of shiftAngle must keep the full (nx, ny, nz) structure.
    Avoid squeezing dimensions; it silently destroys the z-axis and relies
    on implicit NumPy broadcasting, which is fragile and confusing.
    """
    #1/q(x,θ) - already (nx,ny,nz) but symmetric in z
    inv_q = 1.0 / q_xy

    #Integrate along poloidal angle dimension (axis=1)
    alpha = cumulative_trapezoid(inv_q[:, :, 0], x=theta, axis=1, initial=0.0)

    #Force alpha into strict (nx, ny) before any reshaping
    alpha = np.asarray(alpha)
    alpha = alpha.reshape(nx, ny)

    #Broadcast alpha into full (nx, ny, nz)
    alpha_3d = alpha[:, :, None]  #(nx, ny, 1)
    alpha_3d = np.broadcast_to(alpha_3d, (nx, ny, nz))

    return alpha_3d


def compute_curvature(mode, bfield, basis, metric, geom, coords):
    """
    Compute magnetic curvature components G1 and G2.
    
    Parameters:
        mode: "exact", "simple", or "none"
        bfield: dict from compute_magnetic_field()
        basis: dict from compute_basis_vectors()
        metric: dict from compute_metric_tensor()
        geom: dict from compute_geometry()
        coords: dict from generate_coordinates()
    
    Returns:
        dict with:
            G1: Curvature component along e_theta (nx, ny, nz)
            G2: Curvature component along e_r (nx, ny, nz)
    
    Curvature calculation methods:
        "exact"  -> full metric, Christoffel symbols, covariant derivatives
        "simple" -> analytic tokamak curvature approximation
        "none"   -> disable curvature calculation entirely
    """
    nx = basis['er_X'].shape[0]
    ny = basis['er_X'].shape[1]
    nz = basis['er_X'].shape[2]

    if mode == "none":
        #Zero curvature
        return {
            'G1': np.zeros((nx, ny, nz)),
            'G2': np.zeros((nx, ny, nz))
        }

    elif mode == "simple":
        #Analytic circular tokamak approximation:
        #G1 ≈ -cos(theta)/R , G2 ≈ -sin(theta)/R
        theta_mesh = geom['theta_mesh']
        R_vals = geom['R_vals']
        R_3d = np.broadcast_to(R_vals, (nx, ny, nz))

        theta_3d = np.broadcast_to(theta_mesh, (nx, ny, nz))
        G1 = -(np.cos(theta_3d) / R_3d)
        G2 = -(np.sin(theta_3d) / R_3d)

        return {'G1': G1, 'G2': G2}

    elif mode == "exact":
        #Full tensor curvature calculation

        bX = bfield['bX']
        bY = bfield['bY']
        bZ = bfield['bZ']
        b_r = bfield['b_r']
        b_theta = bfield['b_theta']
        b_phi = bfield['b_phi']

        er_X = basis['er_X']
        er_Y = basis['er_Y']
        er_Z = basis['er_Z']
        etheta_X = basis['etheta_X']
        etheta_Y = basis['etheta_Y']
        etheta_Z = basis['etheta_Z']
        ephi_X = basis['ephi_X']
        ephi_Y = basis['ephi_Y']
        ephi_Z = basis['ephi_Z']

        g_cov = metric['g_cov']
        g_contra = metric['g_contra']

        dr = coords['dr']
        dtheta = coords['dtheta']
        dphi = coords['dphi']

        #Compute ∂b/∂x^i (numerical partial derivatives)
        db_dr_X = np.gradient(bX, dr, axis=0)
        db_dr_Y = np.gradient(bY, dr, axis=0)
        db_dr_Z = np.gradient(bZ, dr, axis=0)

        db_dtheta_X = np.gradient(bX, dtheta, axis=1)
        db_dtheta_Y = np.gradient(bY, dtheta, axis=1)
        db_dtheta_Z = np.gradient(bZ, dtheta, axis=1)

        db_dphi_X = np.gradient(bX, dphi, axis=2)
        db_dphi_Y = np.gradient(bY, dphi, axis=2)
        db_dphi_Z = np.gradient(bZ, dphi, axis=2)

        #Metric derivatives
        dg_dr = np.gradient(g_cov, dr, axis=0)
        dg_dtheta = np.gradient(g_cov, dtheta, axis=1)
        dg_dphi = np.gradient(g_cov, dphi, axis=2)

        dg = np.stack([dg_dr, dg_dtheta, dg_dphi], axis=3)

        #Christoffel symbols
        Gamma = np.zeros((nx, ny, nz, 3, 3, 3))
        for i in range(3):
            for j in range(3):
                term1 = dg[..., i, :, j]
                term2 = dg[..., j, :, i]
                term3 = dg[..., :, i, j]
                Gamma[..., :, i, j] = 0.5 * np.einsum(
                    "...km,...m->...k",
                    g_contra,
                    term1 + term2 - term3
                )

        #b in coordinate basis
        b_vec = np.stack([b_r, b_theta, b_phi], axis=-1)

        partial_b = np.stack([
            np.stack([db_dr_X, db_dr_Y, db_dr_Z], axis=-1),
            np.stack([db_dtheta_X, db_dtheta_Y, db_dtheta_Z], axis=-1),
            np.stack([db_dphi_X, db_dphi_Y, db_dphi_Z], axis=-1)
        ], axis=-2)

        basis_tensor = np.stack([
            np.stack([er_X, etheta_X, ephi_X], axis=-1),
            np.stack([er_Y, etheta_Y, ephi_Y], axis=-1),
            np.stack([er_Z, etheta_Z, ephi_Z], axis=-1)
        ], axis=-2)

        partial_b_coord = np.einsum("...ik,...kj->...ij", partial_b, basis_tensor)
        covDb = partial_b_coord + np.einsum("...kij,...k->...ij", Gamma, b_vec)

        kappa_coord = np.einsum("...i,...ij->...j", b_vec, covDb)

        kappa_X = kappa_coord[..., 0] * er_X + kappa_coord[..., 1] * etheta_X + kappa_coord[..., 2] * ephi_X
        kappa_Y = kappa_coord[..., 0] * er_Y + kappa_coord[..., 1] * etheta_Y + kappa_coord[..., 2] * ephi_Y
        kappa_Z = kappa_coord[..., 0] * er_Z + kappa_coord[..., 1] * etheta_Z + kappa_coord[..., 2] * ephi_Z

        G1 = kappa_X * etheta_X + kappa_Y * etheta_Y + kappa_Z * etheta_Z
        G2 = kappa_X * er_X + kappa_Y * er_Y + kappa_Z * er_Z

        return {'G1': G1, 'G2': G2}

    else:
        raise ValueError(f"Unknown curvature mode '{mode}'")


def compute_q_poloidal(x, q_vals, geom, B0, R0):
    """
    Compute TRUE GS-CONSISTENT q(x,y) from Bθ definition.
    
    Uses the large-aspect-ratio Grad-Shafranov relation:
      q = r * B_phi(R) / (R * B_theta(r))
    where B_theta is derived from the target q(r) profile.
    
    Parameters:
        x: Radial coordinate array (nx)
        q_vals: Radial q-profile (nx)
        geom: dict from compute_geometry()
        B0: Toroidal field at R0
        R0: Major radius
    
    Returns:
        q_xy: Poloidally varying safety factor (nx, ny, 1)
    """
    R_vals = geom['R_vals']
    r_mesh = geom['r_mesh']
    q_r = q_vals[:, None, None]  #(nx,1,1)

    #Toroidal field
    Bphi_q = B0 * (R0 / R_vals)

    #GS-consistent poloidal field:
    #   B_theta(r) = r * B_phi(R=r) / (q_r * R0)
    #NOTE: use R0 not R_vals; q-profile is radial, not poloidally varying.
    Btheta_q = (x[:, None, None] * B0) / (q_r * R0)

    #Final physical q(x,y):
    #q_xy = r Bφ / (R Bθ(r))
    q_xy = (x[:, None, None] * Bphi_q) / (R_vals * Btheta_q)

    return q_xy


def write_netcdf_grid(outfile, coords, geom, metric, jacobian, bfield, shift, curv, args):
    """
    Write all computed data to BOUT++ 5.x compatible netCDF grid file.
    
    Parameters:
        outfile: Output filename
        coords: dict from generate_coordinates()
        geom: dict from compute_geometry()
        metric: dict from compute_metric_tensor()
        jacobian: dict from compute_jacobian()
        bfield: dict from compute_magnetic_field()
        shift: shiftAngle array
        curv: dict from compute_curvature()
        args: Command line arguments namespace
    
    Writes complete BOUT++ 5.x grid including:
        - Dimensions and coordinate arrays
        - Metric tensor (covariant and contravariant )
        - Jacobian and differential volume
        - Magnetic field components
        - Shift angle
        - Curvature
        - All required metadata
    """
    nx = args.nx
    ny = args.ny
    nz = args.nz
    precision = args.precision

    x = coords['x']
    y = coords['y']
    z = coords['z']
    x3 = coords['x3']
    y3 = coords['y3']
    z3 = coords['z3']

    R_vals = geom['R_vals']
    Z_vals = geom['Z_vals']

    #Compute 3D spacing arrays
    dr = coords['dr']
    dtheta = coords['dtheta']
    dphi = coords['dphi']

    r_mesh = geom['r_mesh']
    theta_mesh = geom['theta_mesh']

    DX = np.full((nx, ny, nz), dr)
    DY = r_mesh * dtheta  #broadcasts to (nx,ny,nz)

    R_mesh = np.broadcast_to(R_vals, (nx, ny, nz))
    DZ = R_mesh * dphi

    with Dataset(outfile, "w", format="NETCDF4") as nc:
        # -----------------------------
        # Dimensions + required metadata
        # -----------------------------
        nc.createDimension("x", nx)
        nc.createDimension("y", ny)
        nc.createDimension("z", nz)

        nc.setncattr("nx", nx)
        nc.setncattr("ny", ny)
        nc.setncattr("nz", nz)

        nx_var = nc.createVariable("nx", "i4")
        nx_var[:] = nx
        ny_var = nc.createVariable("ny", "i4")
        ny_var[:] = ny
        nz_var = nc.createVariable("nz", "i4")
        nz_var[:] = nz

        #Guard cells
        MXG = 2
        MYG = 2
        MZG = 0

        nc.setncattr("MXG", MXG)
        nc.setncattr("MYG", MYG)
        nc.setncattr("MZG", MZG)

        MXG_var = nc.createVariable("MXG", "i4")
        MXG_var[:] = MXG
        MYG_var = nc.createVariable("MYG", "i4")
        MYG_var[:] = MYG
        MZG_var = nc.createVariable("MZG", "i4")
        MZG_var[:] = MZG

        # -----------------------------
        # Coordinate arrays
        # -----------------------------
        nc.createVariable("x", precision, ("x",))[:] = x
        nc.createVariable("y", precision, ("y",))[:] = y
        nc.createVariable("z", precision, ("z",))[:] = z

        x_var = nc.variables["x"]
        y_var = nc.variables["y"]
        z_var = nc.variables["z"]

        x_var.units = "arb"
        y_var.units = "arb"
        z_var.units = "arb"

        # Required coordinate map variables
        xcoord = nc.createVariable("xcoord", precision, ("x", "y", "z"))
        ycoord = nc.createVariable("ycoord", precision, ("x", "y", "z"))
        zcoord = nc.createVariable("zcoord", precision, ("x", "y", "z"))

        xcoord.coordinates = "x y z"
        ycoord.coordinates = "x y z"
        zcoord.coordinates = "x y z"

        xcoord[:] = x3
        ycoord[:] = y3
        zcoord[:] = z3

        # -----------------------------
        # 3D spacing arrays
        # -----------------------------
        dx = nc.createVariable("dx", precision, ("x", "y", "z"))
        dy = nc.createVariable("dy", precision, ("x", "y", "z"))
        dz = nc.createVariable("dz", precision, ("x", "y", "z"))

        dx.coordinates = "xcoord ycoord zcoord"
        dy.coordinates = "xcoord ycoord zcoord"
        dz.coordinates = "xcoord ycoord zcoord"

        dx[:] = DX
        dy[:] = DY
        dz[:] = DZ

        # -----------------------------
        # Geometry R(x,y,z) and Z(x,y,z)
        # -----------------------------
        R = nc.createVariable("R", precision, ("x", "y", "z"))
        Z = nc.createVariable("Z", precision, ("x", "y", "z"))
        R.coordinates = "xcoord ycoord zcoord"
        Z.coordinates = "xcoord ycoord zcoord"
        R.units = "m"
        Z.units = "m"

        R[:] = np.broadcast_to(R_vals, (nx, ny, nz))
        Z[:] = np.broadcast_to(Z_vals, (nx, ny, nz))

        # -----------------------------
        # Covariant metric tensor
        # -----------------------------
        g11 = nc.createVariable("g11", precision, ("x", "y", "z"))
        g22 = nc.createVariable("g22", precision, ("x", "y", "z"))
        g33 = nc.createVariable("g33", precision, ("x", "y", "z"))
        g12 = nc.createVariable("g12", precision, ("x", "y", "z"))
        g13 = nc.createVariable("g13", precision, ("x", "y", "z"))
        g23 = nc.createVariable("g23", precision, ("x", "y", "z"))

        g11.units = "1"
        g22.units = "1"
        g33.units = "1"
        g12.units = "1"
        g13.units = "1"
        g23.units = "1"

        for var in (g11, g22, g33, g12, g13, g23):
            var.coordinates = "xcoord ycoord zcoord"

        g11[:] = metric['g11']
        g22[:] = metric['g22']
        g33[:] = metric['g33']
        g12[:] = metric['g12']
        g13[:] = metric['g13']
        g23[:] = metric['g23']

        # -----------------------------
        # Contravariant metric tensor
        # -----------------------------
        g_11 = nc.createVariable("g_11", precision, ("x", "y", "z"))
        g_22 = nc.createVariable("g_22", precision, ("x", "y", "z"))
        g_33 = nc.createVariable("g_33", precision, ("x", "y", "z"))
        g_12 = nc.createVariable("g_12", precision, ("x", "y", "z"))
        g_13 = nc.createVariable("g_13", precision, ("x", "y", "z"))
        g_23 = nc.createVariable("g_23", precision, ("x", "y", "z"))

        g_11.units = "1"
        g_22.units = "1"
        g_33.units = "1"
        g_12.units = "1"
        g_13.units = "1"
        g_23.units = "1"

        for var in (g_11, g_22, g_33, g_12, g_13, g_23):
            var.coordinates = "xcoord ycoord zcoord"

        g_11[:] = metric['g_11']
        g_22[:] = metric['g_22']
        g_33[:] = metric['g_33']
        g_12[:] = metric['g_12']
        g_13[:] = metric['g_13']
        g_23[:] = metric['g_23']

        # -----------------------------
        # Jacobian and surfvol
        # -----------------------------
        J = nc.createVariable("J", precision, ("x", "y", "z"))
        J.coordinates = "xcoord ycoord zcoord"
        J.units = "m2"

        surfvol = nc.createVariable("surfvol", precision, ("x", "y", "z"))
        surfvol.coordinates = "xcoord ycoord zcoord"
        surfvol.units = "m3"

        J[:] = jacobian['J']
        surfvol[:] = jacobian['surfvol']

        # -----------------------------
        # Shift angle
        # -----------------------------
        shift_var = nc.createVariable("shiftAngle", precision, ("x", "y", "z"))
        shift_var.coordinates = "xcoord ycoord zcoord"
        shift_var[:] = shift

        # -----------------------------
        # Magnetic field
        # -----------------------------
        Bxy = nc.createVariable("Bxy", precision, ("x", "y", "z"))
        Bxy.coordinates = "xcoord ycoord zcoord"
        Bxy.units = "T"

        Bpxy = nc.createVariable("Bpxy", precision, ("x", "y", "z"))
        Bpxy.coordinates = "xcoord ycoord zcoord"
        Bpxy.units = "T"

        Bxy[:] = bfield['Bxy']
        Bpxy[:] = bfield['Bpxy']

        # -----------------------------
        # Curvature
        # -----------------------------
        G1 = nc.createVariable("G1", precision, ("x", "y", "z"))
        G2 = nc.createVariable("G2", precision, ("x", "y", "z"))
        G1.coordinates = "xcoord ycoord zcoord"
        G2.coordinates = "xcoord ycoord zcoord"
        G1.units = "1/m"
        G2.units = "1/m"

        G1[:] = curv['G1']
        G2[:] = curv['G2']

        # -----------------------------
        # BOUT++ metadata
        # -----------------------------
        nc.mesh_type = "tokamak_axisymmetric_shaped"
        nc.description = "Chatwood Labs - Tokamak axisymmetric shaped grid by bout_tokamak_grid_generator.py"
        nc.R0 = args.R0
        nc.a = args.a

        nc.data_format = "BOUT++"
        nc.coord_system = "tridim"
        nc.zperiod = 2 * np.pi
        nc.precision = precision

        # Boundary regions
        nc.setncattr("ixseps1", -1)
        nc.setncattr("ixseps2", -1)
        nc.setncattr("jyseps1_1", -1)
        nc.setncattr("jyseps1_2", -1)
        nc.setncattr("jyseps2_1", -1)
        nc.setncattr("jyseps2_2", -1)
        nc.setncattr("ny_inner", MYG)


def main():
    t0 = time.time()

    #Parse command-line arguments
    args = parse_arguments()

    #Validate shaping parameters
    validate_shaping_parameters(args.kappa, args.delta)

    print(f"Writing grid to {args.outfile}")

    #Generate coordinate arrays
    xmin = args.xmin_frac * args.a
    coords = generate_coordinates(args.nx, args.ny, args.nz, xmin, args.a)

    #Compute safety factor profile
    q_vals = compute_q_profile(coords['x'], args.a, args.q0, args.qa, args.qform)

    #Compute tokamak geometry
    geom = compute_geometry(coords['x'], coords['y'], args.R0, args.kappa, args.delta)

    #Compute GS-consistent poloidal q(x,y)
    q_xy = compute_q_poloidal(coords['x'], q_vals, geom, args.B0, args.R0)

    #Compute basis vectors
    phi_mesh = coords['z3'][:1, :1, :]  # (1,1,nz)
    basis = compute_basis_vectors(geom, phi_mesh)

    #Compute metric tensor
    metric = compute_metric_tensor(basis)

    #Compute spacing arrays for Jacobian
    dr = coords['dr']
    dtheta = coords['dtheta']
    dphi = coords['dphi']

    r_mesh = geom['r_mesh']
    R_vals = geom['R_vals']

    DX = np.full((args.nx, args.ny, args.nz), dr)
    DY = r_mesh * dtheta
    R_mesh = np.broadcast_to(R_vals, (args.nx, args.ny, args.nz))
    DZ = R_mesh * dphi

    #Compute Jacobian
    jacobian = compute_jacobian(basis, DX, DY, DZ)

    #Compute magnetic field
    bfield = compute_magnetic_field(coords['x'], geom, q_xy, args.B0, args.R0, basis)

    #Compute shift angle
    shift = compute_shift_angle(q_xy, coords['y'], args.nx, args.ny, args.nz)

    #Compute curvature
    curv = compute_curvature(args.curvature, bfield, basis, metric, geom, coords)

    ##Write netCDF grid file
    write_netcdf_grid(args.outfile, coords, geom, metric, jacobian, bfield, shift, curv, args)

    print("Grid generation time: %.2f seconds" % (time.time() - t0))
    print("Done.")


if __name__ == "__main__":
    main()


