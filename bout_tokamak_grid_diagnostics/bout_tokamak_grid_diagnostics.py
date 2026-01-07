#!/usr/bin/env python3
__version__ = "1.0.0"

import os
import io
import json
import argparse
import ctypes
import sys
import logging
import html
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server use
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

from importlib.metadata import version, PackageNotFoundError

# ==============================================================
# Chatwood Labs - BOUT++ 5.x Grid Diagnostics Utility (v1.0.0-Public, BOUT++ 4.x / 5.x Compatible)
#
#  bout_tokamak_grid_diagnostics.py
#
# A standalone diagnostic tool for validating BOUT++ 5.x grid files.
# (with robust support for mainstream BOUT++ 4.x grids).
#
# This script performs a suite of consistency checks on grid geometry,
# metric tensor, Jacobian, and related optional fields (e.g. surfvol,
# magnetic field, shiftAngle) commonly used in BOUT++ simulations.
#
# ------------------------------------------------------------------------------
# Design note:
# This utility is intentionally distributed as a single, self-contained Python
# script to maximise portability in typical HPC and laboratory environments,
# where users often execute tools directly from a shared directory without
# installing full Python packages. No project layout, build system, or repository
# structure is required: if Python and the listed dependencies are available,
# the script should run as-is. This is a deliberate design choice, not an
# oversight, and future modularisation will preserve this single-file entry
# point for compatibility.
# ------------------------------------------------------------------------------
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

class PlotUtils:
    @staticmethod
    def robust_norm(data2d, *, diverging=False, center=0.0, log_if_possible=True, clip=(2.0, 98.0)):
        """
        Minimal, consistent normalization for diagnostic / publication-quality plots.

        Features:
          - Percentile clipping to prevent single-point outliers from dominating
          - Optional diverging normalization centered at `center` (for signed fields)
          - Optional log scaling for strictly-positive fields with large dynamic range

        Notes:
          - Returns (norm, vmin, vmax)
          - norm may be None if data is empty/non-finite (caller can fall back)
        """
        a = np.asarray(data2d, dtype=float)
        a = a[np.isfinite(a)]

        if a.size == 0:
            return None, None, None

        lo, hi = np.nanpercentile(a, clip)
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            lo = float(np.nanmin(a))
            hi = float(np.nanmax(a))
            if lo == hi:
                lo -= 1.0
                hi += 1.0

        if diverging:
            span = float(max(abs(lo - center), abs(hi - center)))
            vmin = center - span
            vmax = center + span
            return mcolors.TwoSlopeNorm(vmin=vmin, vcenter=center, vmax=vmax), vmin, vmax

        vmin, vmax = float(lo), float(hi)

        if log_if_possible and vmin > 0 and vmax / vmin >= 1e3:
            return mcolors.LogNorm(vmin=vmin, vmax=vmax), vmin, vmax

        return mcolors.Normalize(vmin=vmin, vmax=vmax), vmin, vmax

# Back-compat wrapper (keeps all existing call-sites unchanged)
def robust_norm(data2d, *, diverging=False, center=0.0, log_if_possible=True, clip=(2.0, 98.0)):
    return PlotUtils.robust_norm(
        data2d,
        diverging=diverging,
        center=center,
        log_if_possible=log_if_possible,
        clip=clip,
    )

# --------------------
# Logging
# --------------------
# Goals:
#   - DO NOT shadow/replace built-in print()
#   - Keep CLI behavior consistent: script prints -> captured into logging after setup
#   - Provide back-compat module-level wrappers: setup_logging(...), log_exception(...)
#   - Avoid duplicate handlers if re-invoked
#
# Notes:
#   - We redirect sys.stdout/sys.stderr to logger-backed writers *after* handlers exist.
#   - We still allow print(..., file=somefile) to bypass this (Python behavior).
# --------------------

LOGGER_NAME = "chatwoodlabs.bout.grid_diagnostics"
logger = logging.getLogger(LOGGER_NAME)

class Logging:
    """
    Central logging configuration for this script.

    This class exists so we can keep all logging policy in one place without
    replacing built-ins. The CLI entrypoint uses module-level wrappers
    (setup_logging/log_exception) to avoid touching the rest of the code.
    """

    class _LoggerWriter:
        """
        File-like object that buffers writes and emits complete lines into the logger.

        Upgrade vs previous version:
          - Supports prefix-based routing: lines beginning with e.g. "[WARN]" / "[ERROR]"
            get routed to the matching logger method.
          - Keeps buffering behavior (log only complete lines unless flushed).
          - Provides pragmatic stderr heuristics so common "noise" on stderr doesn't get
            mislabeled as ERROR while still catching obvious traceback/error markers.

        Prefix format supported (case-insensitive):
          [DEBUG] msg
          [INFO] msg
          [WARN] msg   (or [WARNING])
          [ERROR] msg
          [CRITICAL] msg  (or [FATAL])

        Notes:
          - Prefixes are stripped from the emitted message.
          - Unprefixed lines go to the configured default level.
        """

        _PREFIX_LEVEL = {
            "DEBUG": "debug",
            "INFO": "info",
            "WARN": "warning",
            "WARNING": "warning",
            "ERROR": "error",
            "CRITICAL": "critical",
            "FATAL": "critical",
        }

        def __init__(self, logger_obj, default_level="info", stream_name="stdout"):
            self._logger = logger_obj
            self._default_level = str(default_level).lower().strip()
            self._stream_name = str(stream_name)
            self._buf = ""

            # Guardrail: if someone passes a bogus default level, fall back to info.
            if not hasattr(self._logger, self._default_level):
                self._default_level = "info"

        def _emit_line(self, line):
            """
            Route a single line to the most appropriate logger level.

            Routing priority:
              1) Explicit prefix like "[WARN]" -> mapped level
              2) stderr heuristics (Traceback/Error markers) -> error
              3) default_level configured for this writer
            """
            if line is None:
                return

            # Normalize newlines/carriage returns but preserve the core message.
            line = line.rstrip("\r\n")
            if not line:
                return

            # ---- 1) Prefix-based routing: [LEVEL] message ----
            stripped = line.lstrip()
            if stripped.startswith("["):
                close = stripped.find("]")
                if close > 1:
                    tag = stripped[1:close].strip().upper()
                    if tag in self._PREFIX_LEVEL:
                        msg = stripped[close + 1 :].lstrip(" \t:-")
                        level_name = self._PREFIX_LEVEL[tag]
                        getattr(self._logger, level_name)(msg if msg else "")
                        return

            # ---- 2) stderr heuristics: catch obvious real errors without over-labeling ----
            if self._stream_name == "stderr":
                # Common Python traceback header
                if line.startswith("Traceback (most recent call last):"):
                    self._logger.error(line)
                    return

                # Generic markers that usually mean an actual failure is being printed
                low = line.lower()
                if "exception" in low or low.startswith("error:") or "fatal" in low:
                    self._logger.error(line)
                    return

            # ---- 3) Default routing ----
            getattr(self._logger, self._default_level)(line)

        def write(self, s):
            if not s:
                return 0

            self._buf += s

            # Emit complete lines; keep trailing partial line buffered.
            while "\n" in self._buf:
                line, self._buf = self._buf.split("\n", 1)
                self._emit_line(line)

            return len(s)

        def flush(self):
            # Emit any remaining buffered content as one line.
            if self._buf:
                # Don't silently drop whitespace-only buffers; flush should be faithful.
                self._emit_line(self._buf)
            self._buf = ""

        def isatty(self):
            return False

        def fileno(self):
            """
            Provide a safe fileno() for libraries that expect a real file descriptor.

            We do NOT guarantee that our logger-writer is backed by a real OS-level
            stream. However, in this script we attempt to delegate to the original
            stdout stream if available.
            """
            underlying = getattr(Logging, "_orig_stdout", None)
            if underlying is None:
                underlying = getattr(sys, "__stdout__", None)

            if underlying is None:
                raise io.UnsupportedOperation("fileno")

            fileno = getattr(underlying, "fileno", None)
            if callable(fileno):
                return fileno()

            raise io.UnsupportedOperation("fileno")

    @staticmethod
    def setup_logging(*, verbose=False, quiet=False, log_file=None):
        """
        Configure logging for CLI usage.

        Parameters
        ----------
        verbose : bool
            Enable DEBUG output.
        quiet : bool
            Suppress INFO output; show only WARNING/ERROR.
        log_file : str or None
            If provided, also write logs to this file.

        Side effects
        ------------
        - Installs handlers on the module logger.
        - Redirects sys.stdout -> logger.info by default (unless a [LEVEL] prefix overrides).
        - Redirects sys.stderr -> logger.warning by default (unless a [LEVEL] prefix overrides).
          Additionally, stderr uses lightweight heuristics to route obvious tracebacks/errors
          to logger.error without turning all stderr noise into "ERROR".
        """
        if verbose and quiet:
            verbose = False  # quiet wins

        level = logging.INFO
        if quiet:
            level = logging.WARNING
        elif verbose:
            level = logging.DEBUG

        logger.setLevel(level)

        # Capture original stdio ONCE, and if we're reconfiguring, restore it first.
        if not hasattr(Logging, "_orig_stdout"):
            Logging._orig_stdout = sys.stdout
            Logging._orig_stderr = sys.stderr
        else:
            # If setup_logging() is called again, don't build handlers on top of LoggerWriter
            sys.stdout = Logging._orig_stdout
            sys.stderr = Logging._orig_stderr

        # Prevent handler duplication if invoked multiple times.
        logger.handlers.clear()
        logger.propagate = False

        formatter = logging.Formatter("%(message)s")

        # Always bind console handlers to the ORIGINAL stdio streams, not whatever sys.stdout/sys.stderr
        # currently are (we redirect those later to LoggerWriter).
        #
        # HPC-friendly behavior:
        # - stdout is intended for lower-severity output (DEBUG/INFO), when enabled
        #   by the active logging level and handler configuration.
        # - stderr is intended for higher-severity output (WARNING+), when enabled
        #   by the active logging level and handler configuration.
        #
        # In --quiet mode the handler level is WARNING, so INFO/DEBUG may be suppressed
        # regardless of which stream they originate from.
        #
        # This preserves the common expectation that stderr contains problems, which matters for
        # job log parsing and tooling (and for sane humans reading logs).

        class _MaxLevelFilter(logging.Filter):
            """
            Allow log records up to a maximum severity.

            We use this to keep WARNING+ out of stdout, so those records only appear on stderr.
            """
            def __init__(self, max_level):
                super().__init__()
                self.max_level = int(max_level)

            def filter(self, record):
                return int(record.levelno) <= self.max_level

        # Handler for "normal" output (DEBUG/INFO) -> stdout
        sh_out = logging.StreamHandler(Logging._orig_stdout)
        sh_out.setLevel(level)  # respects --verbose / --quiet via logger.setLevel(level)
        sh_out.addFilter(_MaxLevelFilter(logging.INFO))
        sh_out.setFormatter(formatter)
        logger.addHandler(sh_out)

        # Handler for warnings/errors -> stderr
        sh_err = logging.StreamHandler(Logging._orig_stderr)
        sh_err.setLevel(logging.WARNING)  # always route WARNING+ to stderr when enabled by logger level
        sh_err.setFormatter(formatter)
        logger.addHandler(sh_err)

        if log_file:
            fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
            fh.setLevel(level)
            fh.setFormatter(formatter)
            logger.addHandler(fh)

        # Redirect stdout/stderr (idempotent, and avoids stacking wrappers).
        #
        # Important behavior:
        #   - stdout defaults to INFO unless the line explicitly says otherwise via prefix.
        #   - stderr defaults to WARNING (NOT ERROR) because many libraries write non-fatal
        #     progress/status to stderr. Explicit prefixes like [ERROR] still win.
        #
        # This preserves useful severity signal in logs without turning stderr into a
        # permanent "everything is on fire" channel.
        if not isinstance(sys.stdout, Logging._LoggerWriter):
            sys.stdout = Logging._LoggerWriter(logger, default_level="info", stream_name="stdout")
        if not isinstance(sys.stderr, Logging._LoggerWriter):
            sys.stderr = Logging._LoggerWriter(logger, default_level="warning", stream_name="stderr")

    @staticmethod
    def log_exception(msg):
        """
        Log an exception message with traceback.

        Call inside an except: block.
        """
        logger.exception(msg)

    @staticmethod
    def restore_stdio():
        """
        Restore original sys.stdout/sys.stderr if they were redirected.
        """
        if hasattr(Logging, "_orig_stdout"):
            sys.stdout = Logging._orig_stdout
        if hasattr(Logging, "_orig_stderr"):
            sys.stderr = Logging._orig_stderr

# --------------------------------------------------------------
# Back-compat wrappers (keep existing call-sites unchanged)
#
# The rest of the script uses:
#   - setup_logging(...)
#   - log_exception(...)
#
# These wrappers keep that stable while the implementation lives in Logging.
# --------------------------------------------------------------
def setup_logging(*, verbose=False, quiet=False, log_file=None):
    return Logging.setup_logging(verbose=verbose, quiet=quiet, log_file=log_file)

def log_exception(msg):
    return Logging.log_exception(msg)

class Validation:
    @staticmethod

    def check(cond, msg):
        """
        Helper function to print standardized check results.
        
        Parameters
        ----------
        cond : bool
            Condition to test.
        msg : str
            Description of the check.
        
        Returns
        -------
        None
        """
        if cond:
            print(f"[ OK ] {msg}")
        else:
            print(f"[FAIL] {msg}")

    @staticmethod
    def record_check(checks, name, passed, severity="INFO", details=None):
        """
        Append a structured check record to the checks list.

        Parameters
        ----------
        checks : list
            The list to append the check result to.
        name : str
            The name of the check.
        passed : bool or None
            Boolean indicating whether the check passed. None means skipped.
        severity : str, optional
            Severity of the check result: "CRITICAL", "WARN", or "INFO" (default is "INFO").
        details : dict or str, optional
            Additional details about the check (default is None).
        
        Returns
        -------
        bool
            The value of `passed`.
        """
        if passed is True:
            status = "PASS"
        elif passed is False:
            status = "FAIL"
        else:
            status = "SKIP"

        entry = {
            "name": str(name),
            "status": status,
            "severity": str(severity),
        }
        if details is not None:
            entry["details"] = details
        checks.append(entry)
        return passed     

# Back-compat wrappers
def check(cond, msg):
    return Validation.check(cond, msg)

def record_check(checks, name, passed, severity="INFO", details=None):
    return Validation.record_check(checks, name, passed, severity=severity, details=details)          

class GridIO:
    @staticmethod    
    def detect_and_normalize_grid(ds):
        """
        Detect common BOUT++ grid conventions and normalize to BOUT++ 5-style names.
        
        Parameters
        ----------
        ds : xarray.Dataset
            The dataset containing grid data to be processed.
        
        Returns
        -------
        ds_normalized : xarray.Dataset
            The normalized dataset with updated dimension and variable names.
        grid_format : str
            The format of the grid, either "BOUT5_STYLE" or "BOUT4_STYLE".
        """
        vars_ = set(ds.variables)
        dims_ = set(ds.sizes)

        has_RZ_5 = ("R" in vars_) and ("Z" in vars_)
        has_RZ_4 = ("Rxy" in vars_) and ("Zxy" in vars_)

        #Default label
        grid_format = "UNKNOWN"

        #Decide format label (simple + robust)
        if has_RZ_5:
            grid_format = "BOUT5_STYLE"
        elif has_RZ_4:
            grid_format = "BOUT4_STYLE"

        #Normalize dims first (common BOUT4 naming)
        dim_rename = {}
        if ("x" not in dims_) and ("nx" in dims_):
            dim_rename["nx"] = "x"
        if ("y" not in dims_) and ("ny" in dims_):
            dim_rename["ny"] = "y"
        if ("z" not in dims_) and ("nz" in dims_):
            dim_rename["nz"] = "z"

        if dim_rename:
            ds = ds.rename_dims(dim_rename)
            #xarray may also have coordinates named like the dims
            coord_rename = {k: v for k, v in dim_rename.items() if k in ds.coords}
            if coord_rename:
                ds = ds.rename(coord_rename)

        # ----------------------------------------------------------
        # Normalize variable names (common BOUT4 + toolchain variants)
        #
        # Rule (important):
        #   Only rename -> canonical BOUT++ 5-style names IF the canonical name
        #   does not already exist in the dataset. This prevents clobbering when
        #   a file contains both canonical + legacy variables.
        #
        # Scope:
        #   We normalize only the variables that this script uses later for
        #   geometry, metric/J validation, and optional B/shift/surfvol diagnostics.
        # ----------------------------------------------------------
        var_rename = {}

        def _maybe_alias(canon, *aliases):
            """
            If `canon` is missing but any alias exists, rename the first alias found to `canon`.

            Why this helper exists:
              - Keeps the rename logic consistent and non-destructive
              - Avoids double-renaming or stomping files that already have canon names
              - Makes it easy to extend without changing downstream code
            """
            if canon in ds.variables:
                return
            for a in aliases:
                if a in ds.variables:
                    var_rename[a] = canon
                    return

        # --- Core geometry (BOUT4 commonly uses *xy / lowercase) ---
        _maybe_alias("R", "Rxy", "rxy")
        _maybe_alias("Z", "Zxy", "zxy")

        # --- Metric tensor components (underscore variants occur in some generators) ---
        _maybe_alias("g11", "g_11")
        _maybe_alias("g12", "g_12")
        _maybe_alias("g13", "g_13")
        _maybe_alias("g22", "g_22")
        _maybe_alias("g23", "g_23")
        _maybe_alias("g33", "g_33")

        # --- Jacobian (some older grids use lowercase) ---
        _maybe_alias("J", "j")

        # --- Magnetic fields (common canonical + occasional lowercase variants) ---
        _maybe_alias("Bxy", "bxy")
        _maybe_alias("Bpxy", "bpxy")

        # Optional toroidal/third component naming variants (harmless if unused later)
        _maybe_alias("Bz", "Bzxy", "bz", "bzxy")

        # --- Curvature / geometry operators (optional but used by plots when present) ---
        _maybe_alias("G1", "g1")
        _maybe_alias("G2", "g2")

        # --- Shift fields (optional diagnostics + plots) ---
        _maybe_alias("shiftAngle", "ShiftAngle", "shiftangle")
        _maybe_alias("zShift", "ZShift", "zshift")

        # --- Volume / surface element (optional) ---
        _maybe_alias("surfvol", "SurfVol", "SURFVOL")

        if var_rename:
            ds = ds.rename(var_rename)

        return ds, grid_format

    @staticmethod 
    def maybe_open_dataset(path, byte_threshold=12 * 1024**3, ram_fraction_threshold=0.60, force_chunk=False, chunks=None):
        """
        Open a dataset using an eager "fast path" when it is likely safe, and fall back
        to a dask-chunked open when memory pressure is likely.

        Chunking policy
        ---------------
        Chunking is enabled if any of the following applies:

        1) Baseline heuristic (estimate + RAM fraction):
           - Estimated working-set >= byte_threshold (default 12 GiB), AND
           - Estimated working-set / available RAM >= ram_fraction_threshold (default 0.60)

        2) "Huge file" safety bias:
           - If the on-disk file is very large (currently ~3 GiB+), we may enable chunking
             even if the estimator undercounts heavy variables (with an extra sanity check
             against available RAM when it can be determined).

        3) Explicit override:
           - force_chunk=True always enables chunking (useful when heuristics undercount).

        Chunk sizes
        -----------
        - If `chunks` is provided (dict), it is filtered to dimensions that actually exist.
        - Otherwise, internal defaults are used (moderate x/y, conservative z).
        - If chunking is forced but no dims match, we fall back to xarray's "auto" chunking.

        Notes
        -----
        - The function avoids returning the initial metadata-only open (decode_cf=False,
          mask_and_scale=False). Instead, it reopens with decode_cf=True and mask_and_scale=True
          so the unchunked and chunked paths produce consistent physical values.

        Parameters
        ----------
        path : str
            Path to the dataset file.
        byte_threshold : int, optional
            Threshold in bytes for when to enable chunking (default is 12 GiB).
        ram_fraction_threshold : float, optional
            Threshold for RAM usage fraction when enabling chunking (default is 0.60).
        force_chunk : bool, optional
            Force chunking regardless of heuristics.
        chunks : dict or None, optional
            Chunk sizes to pass to xarray.open_dataset(chunks=...). Keys not present as dataset
            dimensions are ignored.

        Returns
        -------
        ds : xarray.Dataset
            The opened dataset, either eager or dask-chunked depending on policy.
        """    
        def _available_mem_bytes():
            """
            Cross-platform available RAM estimate (bytes).

            Priority:
              1) Linux: /proc/meminfo MemAvailable (best signal for "can I allocate?")
              2) Windows: GlobalMemoryStatusEx (ullAvailPhys)
              3) POSIX: sysconf (AVPHYS_PAGES * PAGE_SIZE) as a fallback
            Returns 0 if not determinable.
            """

            # -----------------------
            # Linux: /proc/meminfo
            # -----------------------
            try:
                if os.name == "posix" and os.path.exists("/proc/meminfo"):
                    with open("/proc/meminfo", "r") as f:
                        meminfo = f.read().splitlines()
                    kv = {}
                    for line in meminfo:
                        parts = line.split(":")
                        if len(parts) != 2:
                            continue
                        key = parts[0].strip()
                        val = parts[1].strip().split()
                        if len(val) >= 2 and val[1].lower() == "kb":
                            kv[key] = int(val[0]) * 1024
                    # Prefer MemAvailable; fall back to MemFree if needed
                    return int(kv.get("MemAvailable", kv.get("MemFree", 0)))
            except Exception:
                pass

            # -----------------------
            # Windows: GlobalMemoryStatusEx
            # -----------------------
            try:
                if os.name == "nt":
                    class MEMORYSTATUSEX(ctypes.Structure):
                        _fields_ = [
                            ("dwLength", ctypes.c_ulong),
                            ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                        ]

                    stat = MEMORYSTATUSEX()
                    stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
                    if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                        return int(stat.ullAvailPhys)
            except Exception:
                pass

            # -----------------------
            # POSIX fallback: sysconf
            # -----------------------
            try:
                if hasattr(os, "sysconf"):
                    page_size = os.sysconf("SC_PAGE_SIZE")
                    av_pages = os.sysconf("SC_AVPHYS_PAGES")
                    if isinstance(page_size, int) and isinstance(av_pages, int) and page_size > 0 and av_pages > 0:
                        return int(page_size * av_pages)
            except Exception:
                pass

            return 0

        # ----------------------------------------------------------
        # Fast-path guardrails: fail cleanly on missing / invalid path
        #
        # Why this exists:
        # - If the file doesn't exist, xarray can throw a confusing ValueError
        #   about "no IO backends found", because it never gets to an OS-level
        #   open. That error is technically true for "unknown extension", but
        #   it's garbage UX for "typo in filename".
        #
        # Contract:
        # - Missing file: raise FileNotFoundError with a clear message.
        # - Directory passed as a "file": raise IsADirectoryError.
        # ----------------------------------------------------------
        if path is None or str(path).strip() == "":
            raise FileNotFoundError("No grid file path provided.")

        path = str(path)

        if not os.path.exists(path):
            raise FileNotFoundError(f"Grid file not found: {path}")

        if os.path.isdir(path):
            raise IsADirectoryError(f"Grid path is a directory, not a file: {path}")

        # Open once (no chunks) to read metadata cheaply.
        #
        # IMPORTANT (safety / "don’t accidentally OOM on open"):
        # Some netCDF backends can do non-trivial work at open time (CF decoding,
        # mask/scale application, datetime decoding, etc.). We *only* need:
        #  - variable names
        #  - shapes/sizes
        #  - dtypes
        # to estimate a working set and decide if chunking is needed.
        #
        # So we explicitly disable CF decoding and scaling here to minimize the chance
        # that "metadata open" does any heavy lifting or causes eager reads.
        #
        # We will reopen the dataset normally (and/or chunked) after we decide the path.
        ds0 = xr.open_dataset(
            path,
            decode_cf=False,       # avoid CF decoding work (e.g., times, bounds, etc.)
            mask_and_scale=False,  # avoid applying scale_factor/add_offset and masks
        )

        # NOTE:
        # If we end up reopening chunked, we will close ds0 before doing so.
        # If we keep the unchunked ds0, it stays open intentionally (xarray lazy IO).
        #
        # Estimate a "working set" size for large, frequently-touched arrays.
        #
        # CRITICAL NOTE (BOUT4 vs BOUT5):
        # We estimate *before* we normalize/rename variables later in the pipeline.
        # So for BOUT4-style grids (e.g. rxy/zxy), we must include common aliases here
        # or the estimator undercounts and we may skip chunking -> OOM on big grids.
        #
        # Implementation detail:
        # We count each *logical* field once by grouping aliases and taking the first
        # name that exists in the file. This avoids double-counting if a file happens
        # to contain both canonical and alias names.
        heavy_var_groups = [
            # ------------------------------------------------------------------
            # "Heavy" variables used to estimate the working-set (RAM pressure).
            #
            # Goal:
            #   - Count large, frequently-touched arrays that typical diagnostics
            #     will access soon after open.
            #
            # Key constraint:
            #   - This estimator runs BEFORE we rename/normalize variable names
            #     later (BOUT4 vs BOUT5 conventions). So we include common aliases
            #     *here* to avoid undercounting and accidentally skipping chunking.
            #
            # Rule:
            #   - Each tuple is a group of aliases for ONE logical field.
            #   - We count the first name that exists to avoid double-counting.
            # ------------------------------------------------------------------

            # Geometry (BOUT5: R/Z, common BOUT4: rxy/zxy, sometimes Rxy/Zxy)
            ("R", "Rxy", "rxy"),
            ("Z", "Zxy", "zxy"),

            # Sometimes grids carry additional geometry coordinate helpers
            # (not always present, but can be big and often used).
            ("psi", "Psi", "psixy", "Psixy"),
            ("theta", "Theta"),
            ("phi", "Phi"),

            # Metric tensor components (canonical names + underscore variants).
            # Underscore variants appear in some custom pipelines.
            ("g11", "g_11"),
            ("g12", "g_12"),
            ("g13", "g_13"),
            ("g22", "g_22"),
            ("g23", "g_23"),
            ("g33", "g_33"),

            # Jacobian (some older grids use lowercase)
            ("J", "j"),

            # Magnetic fields (common canonical + occasional lowercase variants)
            ("Bxy", "bxy"),
            ("Bpxy", "bpxy"),
            ("Bz", "Bzxy", "bz", "bzxy"),  # optional in some grids

            # Curvature / geometry operators (optional but often used in MHD setups)
            ("G1", "g1"),
            ("G2", "g2"),

            # Optional geometry helpers / shift fields
            ("shiftAngle", "ShiftAngle", "shiftangle"),
            ("zShift", "ZShift", "zshift"),

            # Volume / surface elements
            ("surfvol", "SurfVol", "SURFVOL"),

            # Grid spacing / cell-size helpers (varies by generator; optional)
            ("dx", "dX", "DX"),
            ("dy", "dY", "DY"),
            ("dz", "dZ", "DZ"),
        ]

        est_bytes = 0
        for name_group in heavy_var_groups:
            #Pick the first alias that exists in this dataset
            v = next((n for n in name_group if n in ds0.variables), None)
            if v is None:
                continue

            try:
                est_bytes += int(ds0[v].size) * int(ds0[v].dtype.itemsize)
            except Exception:
                #If anything about dtype/size is weird, don't fail the open path
                pass

        try:
            file_bytes = os.path.getsize(path)
        except Exception:
            file_bytes = None

        avail_bytes = _available_mem_bytes()
        frac = (est_bytes / avail_bytes) if avail_bytes > 0 else 0.0

        # Decide whether to chunk.
        #
        # Baseline policy (existing behavior):
        #   - Be conservative: only chunk when estimated RAM pressure is real.
        #
        # Additional safety policy (new behavior):
        #   - If the file on disk is "huge", bias toward chunking even if our
        #     "heavy variables" estimator undercounts.
        #
        # Why we need this:
        #   - Some grids carry big auxiliary fields not listed in heavy_var_groups.
        #   - Some backends can end up touching more than expected during downstream ops.
        #   - Chunking is a cheap insurance policy for very large inputs.
        #
        # Notes:
        #   - This does NOT force eager loading; later logic explicitly avoids ds.load()
        #     when chunked unless it's clearly safe.
        #   - The user can still override everything with --force-chunk.
        #
        use_chunks = (est_bytes >= byte_threshold) and (avail_bytes > 0 and frac >= ram_fraction_threshold)

        # "Huge file" bias:
        #
        # If file size alone suggests this is a monster dataset, prefer chunking.
        # This is intentionally simpler than the RAM heuristic (no guesswork about
        # which variables get touched later).
        #
        # Default threshold: 3 GiB (tunable via this constant).
        huge_file_threshold_bytes = 3 * 1024**3

        if (not use_chunks) and (file_bytes is not None) and (file_bytes >= huge_file_threshold_bytes):
            # If we can estimate available RAM, use it as an extra sanity check:
            # - If the file is a large fraction of available RAM, chunk.
            # If RAM is unknown, still chunk because "huge file" is the stronger signal.
            if avail_bytes > 0:
                file_frac = file_bytes / float(avail_bytes)
                # 0.35 is intentionally lower than ram_fraction_threshold:
                # file size is a coarse proxy and we want earlier chunking for huge files.
                if file_frac >= 0.35:
                    print(
                        "[WARN] File size is huge relative to available RAM "
                        f"(file ~{file_bytes/1024**3:.2f} GiB, avail ~{avail_bytes/1024**3:.2f} GiB, frac ~{file_frac:.2f}) "
                        "-> enabling chunking as a safety bias."
                    )
                    use_chunks = True
            else:
                print(
                    "[WARN] File size is huge "
                    f"(file ~{file_bytes/1024**3:.2f} GiB; available RAM unknown) "
                    "-> enabling chunking as a safety bias."
                )
                use_chunks = True

        # Override: force chunking if explicitly requested.
        # This is the "I don't trust the heuristic; don't risk eager reads" mode.
        if force_chunk:
            use_chunks = True

        if not use_chunks:
            if file_bytes is not None and avail_bytes > 0:
                print(f"[INFO] Estimated working-set ~{est_bytes/1024**2:.1f} MiB (file {file_bytes/1024**2:.1f} MiB, avail {avail_bytes/1024**2:.1f} MiB, frac {frac:.2f}) -> no chunking")
            elif file_bytes is not None:
                print(f"[INFO] Estimated working-set ~{est_bytes/1024**2:.1f} MiB (file {file_bytes/1024**2:.1f} MiB) -> no chunking")
            else:
                print(f"[INFO] Estimated working-set ~{est_bytes/1024**2:.1f} MiB -> no chunking")
            # NOTE:
            # ds0 was intentionally opened with decode_cf=False and mask_and_scale=False
            # for a *metadata-only* sizing estimate. Returning ds0 would make the unchunked
            # path produce different physical values than the chunked path (which reopens
            # the dataset with normal decoding/scaling).
            #
            # Fix: close ds0 and reopen consistently with the "real" settings.
            try:
                ds0.close()
            except Exception:
                pass

            ds = xr.open_dataset(
                path,
                decode_cf=True,       # enable CF decoding consistently
                mask_and_scale=True,  # apply scale_factor/add_offset + masks consistently
            )
            ds.attrs["_chunked_open"] = False

            # Keep the memory sizing signals around for downstream decisions (HTML + runtime).
            # This is presentation/behavioral metadata only; it does not affect the dataset contents.
            ds.attrs["_est_working_set_bytes"] = int(est_bytes)
            ds.attrs["_avail_mem_bytes"] = int(avail_bytes) if avail_bytes is not None else 0
            ds.attrs["_est_working_set_frac"] = float(frac)

            return ds

        #Chunked path: capture dims *before* closing ds0.
        #Some backends/lazy stores can make ds0 metadata unsafe after close().
        dims0 = set(ds0.dims)

        try:
            ds0.close()
        except Exception:
            pass

        # Choose chunk sizes:
        #   1) If user provided --chunks, use those (filtered to existing dims).
        #   2) Otherwise fall back to internal defaults.
        #
        # Note: We filter to dims0 so passing "z=1" on a 2D grid doesn't break anything.
        if isinstance(chunks, dict) and len(chunks) > 0:
            user_chunks = {k: int(v) for k, v in chunks.items() if k in dims0}
            chunks = user_chunks
        else:
            chunks = {}
            # Only define chunk sizes for dims that exist
            # (and keep them moderate; z is often the killer)
            if "x" in dims0:
                chunks["x"] = 64
            if "y" in dims0:
                chunks["y"] = 128
            if "z" in dims0:
                chunks["z"] = 8

        # If chunking was requested but no dims matched (e.g. bizarre dim names),
        # fall back to xarray's automatic chunking behavior.
        if force_chunk and not chunks:
            chunks = "auto"

        if file_bytes is not None and avail_bytes > 0:
            print(f"[WARN] Estimated working-set ~{est_bytes/1024**2:.1f} MiB (file {file_bytes/1024**2:.1f} MiB, avail {avail_bytes/1024**2:.1f} MiB, frac {frac:.2f}) -> enabling chunking {chunks}")
        else:
            print(f"[WARN] Estimated working-set ~{est_bytes/1024**2:.1f} MiB -> enabling chunking {chunks}")

        # Open chunked, but explicitly keep decode/scaling aligned with the unchunked "real open".
        ds = xr.open_dataset(
            path,
            chunks=chunks,
            decode_cf=True,
            mask_and_scale=True,
        )

        ds.attrs["_chunked_open"] = True

        # Preserve the sizing estimate so the main pipeline can decide whether eager-loading
        # is safe. Chunking only helps if we keep the dataset lazy when it is truly large.
        ds.attrs["_est_working_set_bytes"] = int(est_bytes)
        ds.attrs["_avail_mem_bytes"] = int(avail_bytes) if avail_bytes is not None else 0
        ds.attrs["_est_working_set_frac"] = float(frac)

        return ds

# Back-compat wrappers
def detect_and_normalize_grid(ds):
    return GridIO.detect_and_normalize_grid(ds)

def maybe_open_dataset(
    path,
    byte_threshold=12 * 1024**3,
    ram_fraction_threshold=0.60,
    force_chunk=False,
    chunks=None,
):
    return GridIO.maybe_open_dataset(
        path,
        byte_threshold=byte_threshold,
        ram_fraction_threshold=ram_fraction_threshold,
        force_chunk=force_chunk,
        chunks=chunks,
    )


def parse_chunks_spec(spec):
    """
    Parse --chunks argument into a dict suitable for xarray.open_dataset(chunks=...).

    Accepted format:
      "x=256,y=256,z=1"

    Rules:
      - Returns None if spec is None/empty (meaning: use internal defaults).
      - Ignores empty items (extra commas).
      - Requires positive integers.
      - Leaves dimension names as-provided (we later filter to dims that exist).
    """
    if spec is None:
        return None

    spec = str(spec).strip()
    if spec == "":
        return None

    chunks = {}
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue

        if "=" not in item:
            raise ValueError(
                f"Invalid --chunks item '{item}'. Expected 'dim=int' like 'x=256'."
            )

        k, v = item.split("=", 1)
        k = k.strip()
        v = v.strip()

        if not k:
            raise ValueError(f"Invalid --chunks item '{item}': empty dimension name.")
        try:
            n = int(v)
        except Exception:
            raise ValueError(f"Invalid --chunks item '{item}': chunk size must be an integer.")

        if n <= 0:
            raise ValueError(f"Invalid --chunks item '{item}': chunk size must be > 0.")

        chunks[k] = n

    return chunks if chunks else None

def to_scalar(x):
    """
    Convert "scalar-like" values into a plain Python scalar (float/bool/int).

    Why this exists:
      - xarray / dask reductions frequently return numpy scalars, 0-d arrays,
        or 1-element arrays.
      - Some code paths may accidentally hand us a non-scalar ndarray.
      - The old version attempted `float(x)` twice, which is useless and can
        explode on non-scalar arrays.

    Rules:
      - If it's a dask object, compute it once.
      - If it's an xarray object, unwrap `.values`.
      - If it's a numpy scalar / 0-d array / 1-element array, return the item.
      - If it's a larger array, fail loudly with a clear error (caller bug).
      - Prefer to preserve bool/int where possible; otherwise fall back to float.
    """
    # 1) Materialise dask laziness (but do it only once).
    if hasattr(x, "compute"):
        x = x.compute()

    # 2) Unwrap xarray objects to raw numpy-ish values.
    if hasattr(x, "values"):
        x = x.values

    # 3) Normalise numpy-ish scalars / arrays into a Python scalar.
    try:
        import numpy as np
    except Exception:
        np = None

    if np is not None:
        # numpy scalar: np.float64(1.2), np.bool_(True), etc.
        if isinstance(x, np.generic):
            return x.item()

        # numpy array: handle 0-d and 1-element safely.
        if isinstance(x, np.ndarray):
            if x.ndim == 0:
                return x.item()
            if x.size == 1:
                return x.reshape(()).item()

            # Anything bigger is not scalar-like. That's a caller mistake.
            raise TypeError(
                f"to_scalar expected a scalar-like value, got ndarray shape={x.shape}, dtype={x.dtype}"
            )

    # 4) Plain Python scalar types already: return as-is.
    if isinstance(x, (bool, int, float)):
        return x

    # 5) Last resort: attempt numeric conversion.
    # This covers things like strings ("1.23") or objects that implement __float__.
    try:
        return float(x)
    except Exception as e:
        raise TypeError(f"to_scalar could not convert value of type {type(x).__name__}: {x!r}") from e


def da_all(da: xr.DataArray) -> bool:
    """
    Dask-safe boolean reduction for xarray objects.
    Equivalent to `bool(da.all())` but guaranteed to return a Python bool
    without triggering full-array conversion accidentally.
    """
    return bool(to_scalar(da.all()))

def da_any(da: xr.DataArray) -> bool:
    """
    Dask-safe boolean reduction for xarray objects.
    Equivalent to `bool(da.any())` but guaranteed to return a Python bool.
    """
    return bool(to_scalar(da.any()))

def _is_chunked(ds):
    """
    Return True if this dataset is dask-chunked (lazy).
    We also respect the explicit marker set in maybe_open_dataset().
    """
    try:
        if bool(ds.attrs.get("_chunked_open", False)):
            return True
    except Exception:
        pass

    # Fallback: if any variable is chunked, treat dataset as chunked.
    try:
        return any(getattr(ds[v].data, "chunks", None) is not None for v in ds.data_vars)
    except Exception:
        return False


def _as_numpy(da):
    """
    Convert an xarray DataArray to a NumPy array in a way that is safe for chunked data.

    - If da is dask-backed, .compute() materializes only what we've already sliced/reduced.
    - If da is eager, this is basically just .values.
    """
    try:
        if hasattr(da, "compute"):
            da = da.compute()
        return np.asarray(da.values)
    except Exception:
        # Last resort: best-effort conversion
        return np.asarray(da)

# -----------------------------------------------------------------------------
# Theta validation helpers (visualization-only diagnostics)
#
# Why this exists:
# - "theta" is *often* a geometric poloidal angle in tokamak grids (radians).
# - But some grids use "theta" for a remapped/index coordinate, degrees, or
#   something that varies with x/z in a way that breaks tokamak assumptions.
#
# We keep these checks lightweight:
# - Only validate a 1D theta(y) line (sliced first -> Dask-safe).
# - Optionally compare theta(y) to a geometric angle computed from R/Z on that
#   same line (stronger check).
#
# These checks MUST NOT change core logic; they only emit warnings/info.
# -----------------------------------------------------------------------------

TWOPI = 2.0 * np.pi

def _wrap_to_pi(a):
    """Wrap angle to (-pi, pi]."""
    return (a + np.pi) % (2.0 * np.pi) - np.pi

def _theta_sanity_check_1d(theta_vals, *, context=""):
    """
    Basic sanity checks on a 1D theta(y) line.

    Parameters
    ----------
    theta_vals : np.ndarray
        1D array of theta values along poloidal index y (already sliced).
    context : str
        Short label to clarify where warnings are coming from.

    Returns
    -------
    dict
        Info diagnostics (span, endpoint diff, etc.) for optional logging.
    """
    info = {}
    prefix = f"[WARN]{'[' + context + ']' if context else ''} "

    th = np.asarray(theta_vals, dtype=float)
    th = th[np.isfinite(th)]
    if th.size < 4:
        print(prefix + "theta sanity check skipped (too few finite samples).")
        return info

    span = float(np.nanmax(th) - np.nanmin(th))
    info["span_raw"] = span

    # Units heuristic: radians should be ~2pi span (or close-ish).
    # Degrees frequently show ~360 span.
    if 50.0 < span < 400.0:
        print(prefix + f"'theta' span looks like degrees (span≈{span:.1f}). Did you forget to convert to radians?")
    elif span < 1.0:
        print(prefix + f"'theta' span is tiny (span≈{span:.3f}). Suspicious for a poloidal angle.")

    # Unwrap for continuity / monotonicity checks.
    th_u = np.unwrap(th)
    d = np.diff(th_u)

    # Big adjacent jumps (> ~pi) usually means wrapping/discontinuity or bad ordering.
    max_jump = float(np.nanmax(np.abs(d)))
    info["max_adjacent_jump"] = max_jump
    if max_jump > (np.pi + 0.2):
        print(prefix + "'theta' has large adjacent jumps (>π). Possible wrap/discontinuity or y-order issue.")

    # Endpoint periodicity-ish check (mod 2pi).
    end_diff = float(_wrap_to_pi(th_u[-1] - th_u[0]))
    info["endpoint_wrap_diff"] = end_diff
    if abs(end_diff) > 0.2 and abs(abs(end_diff) - TWOPI) > 0.2:
        print(prefix + f"'theta' endpoints don't match cleanly modulo 2π (wrap diff≈{end_diff:.3f} rad).")

    # Old min/max "coverage" heuristic, but with honest messaging.
    # This is not definitive; some valid conventions use 0..2pi or partial segments.
    if (np.nanmin(th) > -np.pi) or (np.nanmax(th) < np.pi):
        print(prefix + "'theta' does not span roughly [-π, π] on this cut. Could be a different convention "
                      "(e.g. 0..2π), a restricted segment, or a non-geometric theta.")

    return info

def _theta_vs_geometry_check(theta_vals, R_line, Z_line, *, context=""):
    """
    Optional stronger check:
    Compare theta(y) to a geometric poloidal angle atan2(Z-Z0, R-R0) on the same line.

    Notes:
    - We only check *shape similarity* (correlation) after removing constant offsets.
    - This does not prove correctness, but it catches the common "theta is something else" case.
    """
    prefix = f"[WARN]{'[' + context + ']' if context else ''} "

    th = np.asarray(theta_vals, dtype=float)
    R = np.asarray(R_line, dtype=float)
    Z = np.asarray(Z_line, dtype=float)

    m = np.isfinite(th) & np.isfinite(R) & np.isfinite(Z)
    if np.count_nonzero(m) < 4:
        print(prefix + "theta vs geometry check skipped (too few finite samples).")
        return

    th = np.unwrap(th[m])
    R = R[m]
    Z = Z[m]

    # Crude "center" estimate (good enough for sanity checking).
    R0 = float(np.nanmedian(R))
    Z0 = float(np.nanmedian(Z))

    th_geom = np.unwrap(np.arctan2(Z - Z0, R - R0))

    # Compare shapes after removing constant offsets.
    th0 = th - float(np.nanmean(th))
    tg0 = th_geom - float(np.nanmean(th_geom))

    # Correlation: 1.0 means "tracks perfectly up to offset/scale".
    # A bad value means theta probably isn’t a geometric poloidal angle.
    try:
        corr = float(np.corrcoef(th0, tg0)[0, 1])
    except Exception:
        corr = np.nan

    if not np.isfinite(corr) or corr < 0.9:
        print(prefix + f"'theta' does not track geometric atan2(Z-Z0, R-R0) well (corr={corr:.3f}). "
                      "It may not be a geometric poloidal angle (still could be a valid coordinate).")


def infer_outboard_midplane_y_index(theta_da, R, Z=None, *, context=""):
    """
    Infer the y-index corresponding to the *outboard midplane*.

    Primary heuristic (tokamak-centric):
      - If `theta_da` is available and includes a 'y' dimension, assume theta≈0
        corresponds to outboard midplane.
      - IMPORTANT: theta is often multi-dimensional; we MUST slice to a 1D theta(y)
        line before converting to numpy (Dask-safe, prevents flat-index bugs).

    Optional validation (visualization-only):
      - Run sanity checks on the sliced theta(y) line (range, unwrap continuity, etc.).
      - Optionally compare theta(y) to a geometric angle from (R,Z) on the same line.

    Fallback heuristic:
      - Use the y-index where R is maximal at mid-radius (x=nx//2).
        Works well for typical tokamak cross-sections, still a heuristic.

    Parameters
    ----------
    theta_da : xarray.DataArray | None
        Grid 'theta' variable (optional). May be multi-dimensional.
    R : xarray.DataArray | np.ndarray
        Major radius array. Expected to be (x,y) after any z-slicing.
    Z : xarray.DataArray | np.ndarray | None
        Vertical coordinate array (optional). If provided, enables theta-vs-geometry check.
    context : str
        Short label for warning messages (e.g. "grid geometry", "mag field").

    Returns
    -------
    int
        y index for outboard midplane.
    """
    # --- Try theta-based inference first (preferred when usable) ---
    try:
        if theta_da is None:
            raise ValueError("theta_da is None")

        if not hasattr(theta_da, "dims") or "y" not in theta_da.dims:
            raise ValueError("'theta' has no 'y' dimension")

        # Build indexers to reduce theta down to a 1D line over y.
        # Pick mid-radius in x if present; pick index 0 for any other dims (z/t/etc).
        indexers = {}
        if "x" in theta_da.dims:
            indexers["x"] = int(theta_da.sizes["x"] // 2)

        for d in theta_da.dims:
            if d in ("y", "x"):
                continue
            indexers[d] = 0

        theta_line = theta_da.isel(indexers)

        # Convert ONLY AFTER slicing down (Dask-safe).
        theta_vals = np.asarray(theta_line.values, dtype=float)

        # Sanity: ensure we truly have a 1D theta(y) line. If not, bail to fallback.
        if theta_vals.ndim != 1:
            raise ValueError(f"theta_line is not 1D after slicing (ndim={theta_vals.ndim})")

        # Visualization-only validation (does not affect returned index)
        _theta_sanity_check_1d(theta_vals, context=context)

        # Optional stronger validation vs geometry (needs R/Z along same x_mid line).
        if Z is not None:
            try:
                # Extract R_line/Z_line in a dimension-safe way.
                if hasattr(R, "dims") and "x" in R.dims and "y" in R.dims:
                    x_mid = int(R.sizes["x"] // 2)
                    R_line = np.asarray(R.isel(x=x_mid).values, dtype=float)
                else:
                    R_np = np.asarray(R, dtype=float)
                    x_mid = int(R_np.shape[0] // 2)
                    R_line = R_np[x_mid, :]

                if hasattr(Z, "dims") and "x" in Z.dims and "y" in Z.dims:
                    x_mid = int(Z.sizes["x"] // 2)
                    Z_line = np.asarray(Z.isel(x=x_mid).values, dtype=float)
                else:
                    Z_np = np.asarray(Z, dtype=float)
                    x_mid = int(Z_np.shape[0] // 2)
                    Z_line = Z_np[x_mid, :]

                if R_line.ndim == 1 and Z_line.ndim == 1 and R_line.size == theta_vals.size:
                    _theta_vs_geometry_check(theta_vals, R_line, Z_line, context=context)
            except Exception:
                # Keep this strictly non-fatal: diagnostics only.
                pass

        # Outboard midplane ≈ theta closest to 0
        #
        # IMPORTANT DEFENSIVE FIX:
        # np.nanargmin() throws if the entire input is NaN (or becomes all-NaN after abs()).
        # That can happen on broken grids, partially-filled grids, or if the slice picks a
        # region where theta is missing. In that case, we *must* fall back to the R-based
        # heuristic below rather than crashing.
        abs_th = np.abs(theta_vals)
        finite = np.isfinite(abs_th)

        if not np.any(finite):
            # No usable theta samples on this cut -> trigger fallback path.
            raise ValueError("theta_line contains no finite values; cannot infer outboard midplane from theta")

        # Map the argmin within the filtered array back to the original y-index.
        return int(np.flatnonzero(finite)[np.argmin(abs_th[finite])])

    except Exception:
        # --- Fallback: R-max at mid-radius ---
        try:
            # xarray path
            if hasattr(R, "dims") and "x" in R.dims and "y" in R.dims:
                R_line = R.isel(x=int(R.sizes["x"] // 2))
                return int(R_line.argmax(dim="y").values)

            # numpy path
            R_np = np.asarray(R, dtype=float)
            x_mid = int(R_np.shape[0] // 2)
            return int(np.nanargmax(R_np[x_mid, :]))

        except Exception:
            # Absolute last-resort fallback: 0
            return 0

def render_grid_visualization(ds, output_path):
    """
    Generate 3D visualization of tokamak grid geometry.
    
    Creates a PNG showing:
    - 3D toroidal surface rendering
    - Poloidal cross-section
    - Flux surface structure
    - Curvature components (if available)
    
    Parameters
    ----------
    ds : xarray.Dataset
        The grid dataset containing geometry and possibly curvature data.
    output_path : str
        The path to save the generated PNG file.
    
    Returns
    -------
    bool
        True if the visualization is successfully saved, False if an error occurred.
    """
    try:
        print("\n=== Generating Grid Visualization ===")
        
        #Extract geometry (dimension-order safe).
        #IMPORTANT:
        # - If ds is chunked, calling .values on full 3D fields can blow RAM.
        # - So we determine dimensionality lazily, then only compute the slices needed.
        R_da = ds["R"].transpose("x", "y", "z", missing_dims="ignore")
        Z_da = ds["Z"].transpose("x", "y", "z", missing_dims="ignore")

        #Support both 2D (x,y) and 3D (x,y,z) grids
        nx = int(R_da.sizes.get("x", R_da.shape[0]))

        is_3d = ("z" in R_da.dims) and (R_da.sizes.get("z", 0) > 1)

        #For all plots below, we primarily use a single toroidal slice z=0 when available.
        #Compute only that slice for chunked datasets; keep fast path for eager datasets.
        if is_3d:
            R = _as_numpy(R_da.isel(z=0))
            Z = _as_numpy(Z_da.isel(z=0))
        else:
            R = _as_numpy(R_da)
            Z = _as_numpy(Z_da)
        
        #Check if curvature data exists
        has_curvature = "G1" in ds and "G2" in ds
        
        # Create figure with subplots (4 if curvature, 3 if not)
        if has_curvature:
            fig = plt.figure(figsize=(20, 10))
        else:
            fig = plt.figure(figsize=(16, 6))
        
        #Plotting sections (Poloidal Cross-Section, 3D Toroidal Surface, Flux Surface Spacing, etc.)
        #Each plot is described clearly in the earlier part of the function
        
        # ------------------------------------------------
        # Plot 1: Poloidal Cross-Section (2D)
        # ------------------------------------------------
        if has_curvature:
            ax1 = fig.add_subplot(2, 3, 1)
        else:
            ax1 = fig.add_subplot(1, 3, 1)
        
        # Plot flux surfaces at multiple radial positions
        for i in [0, nx//4, nx//2, 3*nx//4, nx-1]:
            #R/Z are already 2D poloidal-plane arrays (z=0 slice taken earlier if needed).
            R_slice = R[i, :]
            Z_slice = Z[i, :]

            ax1.plot(R_slice, Z_slice, linewidth=1.5, alpha=0.7)
        
        ax1.set_xlabel('R (m)', fontsize=12)
        ax1.set_ylabel('Z (m)', fontsize=12)
        ax1.set_title('Poloidal Cross-Section', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')
        
        # ------------------------------------------------
        # Plot 2: 3D Toroidal Surface (outer flux surface)
        # ------------------------------------------------
        if has_curvature:
            ax2 = fig.add_subplot(2, 3, 2, projection='3d')
        else:
            ax2 = fig.add_subplot(1, 3, 2, projection='3d')
        
        # Take outer flux surface
        #
        # IMPORTANT:
        # - At the top of this function we intentionally convert R/Z into a 2D poloidal plane (z=0)
        #   to avoid loading full 3D fields for chunked datasets.
        # - For the 3D toroidal surface plot, we DO need the full outer surface (y,z) at x = -1.
        #   So we compute that surface directly from the lazy DataArrays (R_da/Z_da), not from R/Z.
        if is_3d:
            # Outer surface at x = -1: shape ~ (y, z)
            R_outer = _as_numpy(R_da.isel(x=-1))
            Z_outer = _as_numpy(Z_da.isel(x=-1))

            # Use the existing z index as the toroidal angle parameter (0..2π).
            # This is a visualization convention; it does not assume z is a true geometric angle.
            nphi = int(R_outer.shape[1])
            phi = np.linspace(0, 2*np.pi, nphi, endpoint=False)

            X_outer = R_outer * np.cos(phi[np.newaxis, :])
            Y_outer = R_outer * np.sin(phi[np.newaxis, :])
            Z_outer_plot = Z_outer
        else:
            #Fake a toroidal surface by revolving the (R,Z) curve around axis
            R_curve = R[-1, :]   # (y,)
            Z_curve = Z[-1, :]   # (y,)
            nphi = 64
            phi = np.linspace(0, 2*np.pi, nphi, endpoint=False)

            X_outer = R_curve[:, np.newaxis] * np.cos(phi[np.newaxis, :])
            Y_outer = R_curve[:, np.newaxis] * np.sin(phi[np.newaxis, :])
            Z_outer_plot = Z_curve[:, np.newaxis] * np.ones_like(phi[np.newaxis, :])
        
        #Plot surface
        ax2.plot_surface(X_outer, Y_outer, Z_outer_plot, 
                        cmap='plasma', alpha=0.8, 
                        linewidth=0, antialiased=True)
        
        #Add some flux surface contours
        for i in [nx//4, nx//2, 3*nx//4]:
            # R/Z are already 2D poloidal-plane arrays (z=0 slice taken earlier if needed).
            R_surf = R[i, :]
            Z_surf = Z[i, :]
            
            #Close the contour
            R_surf_closed = np.append(R_surf, R_surf[0])
            Z_surf_closed = np.append(Z_surf, Z_surf[0])
            
            #Plot at phi=0
            X_surf = R_surf_closed
            Y_surf = np.zeros_like(R_surf_closed)
            
            ax2.plot(X_surf, Y_surf, Z_surf_closed, 
                    'k-', linewidth=2, alpha=0.6)
        
        ax2.set_xlabel('X (m)', fontsize=10)
        ax2.set_ylabel('Y (m)', fontsize=10)
        ax2.set_zlabel('Z (m)', fontsize=10)

        if is_3d:
            ax2.set_title('3D Toroidal Geometry', fontsize=14, fontweight='bold')
        else:
            ax2.set_title(
                'Axisymmetric Surface\n(2D Grid Revolved - Visualization Only)',
                fontsize=13,
                fontweight='bold'
            )

        if not is_3d:
            ax2.text2D(
                0.5, -0.07,
                'Generated by toroidal revolution of 2D poloidal cross-section',
                transform=ax2.transAxes,
                ha='center',
                va='bottom',
                fontsize=9,
                alpha=0.6
            )
        
        #Set equal aspect ratio for 3D plot
        max_range = np.array([X_outer.max()-X_outer.min(), 
                             Y_outer.max()-Y_outer.min(), 
                             Z_outer_plot.max()-Z_outer_plot.min()]).max() / 2.0
        mid_x = (X_outer.max()+X_outer.min()) * 0.5
        mid_y = (Y_outer.max()+Y_outer.min()) * 0.5
        mid_z = (Z_outer_plot.max()+Z_outer_plot.min()) * 0.5
        ax2.set_xlim(mid_x - max_range, mid_x + max_range)
        ax2.set_ylim(mid_y - max_range, mid_y + max_range)
        ax2.set_zlim(mid_z - max_range, mid_z + max_range)
        
        # ------------------------------------------------
        # Plot 3: Flux Surface Spacing (Radial Structure)
        # ------------------------------------------------
        if has_curvature:
            ax3 = fig.add_subplot(2, 3, 3)
        else:
            ax3 = fig.add_subplot(1, 3, 3)

        # ------------------------------------------------------------------
        # IMPORTANT VISUALIZATION ASSUMPTION (Outboard Midplane Heuristic)
        #
        # Several plots below (e.g. flux surface spacing, radial profiles)
        # require selecting a representative "outboard midplane" index in the
        # poloidal direction.
        #
        # The logic here assumes an AXISYMMETRIC TOKAMAK GEOMETRY, where:
        #   - The outboard midplane corresponds approximately to theta ≈ 0
        #     (if a theta coordinate exists), OR
        #   - The poloidal location of maximum major radius R at mid-radius.
        #
        # This heuristic is reasonable and standard for tokamak grids, but it
        # is NOT invariant across magnetic configurations.
        #
        # In particular:
        #   - Stellarators, helical devices, or strongly 3D equilibria do not
        #     possess a unique or physically privileged "outboard midplane".
        #   - For such grids, these plots should be interpreted as qualitative
        #     diagnostics only, not strict physical profiles.
        #
        # This tool intentionally does NOT attempt to generalize this concept
        # beyond tokamak-like geometry.
        # ------------------------------------------------------------------

        #Infer outboard midplane index for visualization
        if "theta" in ds:
            theta_da = ds["theta"]

            # NOTE / DEFENSIVE WARNING:
            # Some grids provide a variable named "theta" that is not a true
            # geometric poloidal angle (e.g. index-based, remapped, or otherwise
            # non-monotonic). We assume here that theta≈0 corresponds to the
            # outboard midplane, which is standard for tokamak grids but not
            # guaranteed.
            #
            # This affects visualization heuristics only, not validation logic.
            #
            # IMPORTANT (Dask-safe):
            # `theta` is often multi-dimensional and can be lazily loaded. Calling
            # `.values` on the full array can eagerly materialize a *huge* dataset.
            # So we reduce `theta` to a small 1D theta(y) line first, then check
            # its span. This keeps the warning without blowing up memory.
            try:
                # Build indexers to reduce theta down to a 1D line over y.
                # Pick mid-radius in x if present; pick index 0 for any other dims (z/t/etc).
                indexers = {}
                if hasattr(theta_da, "dims") and "x" in theta_da.dims:
                    indexers["x"] = int(theta_da.sizes["x"] // 2)

                for d in getattr(theta_da, "dims", ()):
                    if d in ("y", "x"):
                        continue
                    indexers[d] = 0

                theta_line = theta_da.isel(indexers)

                # Convert ONLY AFTER slicing down (prevents eager-load of full theta).
                theta_vals_check = np.asarray(theta_line.values, dtype=float)

                # Only run the span check if we truly have a 1D theta(y) line.
                # If not, do nothing (we don't want false alarms or eager-load hacks).
                if theta_vals_check.ndim == 1 and theta_vals_check.size > 0:
                    if (np.nanmin(theta_vals_check) > -np.pi) or (np.nanmax(theta_vals_check) < np.pi):
                        print(
                            "[WARN] 'theta' variable detected but may not represent a full "
                            "geometric poloidal angle; outboard midplane inference is heuristic."
                        )
            except Exception:
                # This is visualization-only validation; never fail the workflow due to a warning check.
                pass

            # Use shared inference logic + visualization-only theta validation.
            # Passing Z enables an optional (stronger) check against geometric atan2(Z,Z0,R,R0).
            y_mid_idx = infer_outboard_midplane_y_index(theta_da, R, Z=Z, context="grid geometry")

        else:
            # Fallback (NumPy): choose poloidal index where R is maximal at mid-radius.
            # R is always 2D here (x, y) because we took z=0 earlier if needed.
            x_mid = R.shape[0] // 2
            R_line = R[x_mid, :]
            y_mid_idx = int(np.nanargmax(R_line))
        
        # Plot R vs x at outboard midplane
        # R is 2D (x, y) here
        R_midplane = R[:, y_mid_idx]

        x_coord = np.arange(nx)
        
        ax3.plot(x_coord, R_midplane, 'b-', linewidth=2, label='R (outboard)')
        ax3.set_xlabel('Radial Index (x)', fontsize=12)
        ax3.set_ylabel('R (m)', fontsize=12)
        ax3.set_title('Flux Surface Spacing', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        #Add spacing gradient as secondary axis
        ax3_twin = ax3.twinx()
        dR_dx = np.gradient(R_midplane)
        ax3_twin.plot(x_coord, dR_dx, 'r--', linewidth=1.5, alpha=0.7, label='dR/dx')
        ax3_twin.set_ylabel('dR/dx (m/point)', fontsize=12, color='r')
        ax3_twin.tick_params(axis='y', labelcolor='r')
        ax3_twin.legend(loc='upper right')
        
        # ------------------------------------------------
        # Curvature Plots (if available)
        # ------------------------------------------------
        if has_curvature:
            # ------------------------------------------------
            # Curvature arrays (dimension-order safe, Dask-safe)
            #
            # IMPORTANT:
            # - Do NOT call .values on the full 3D curvature fields. On chunked datasets
            #   that triggers a full compute/load and can blow RAM.
            # - We only need a single toroidal slice (z=0) for these poloidal-plane plots,
            #   so slice first, then convert to numpy.
            # ------------------------------------------------
            G1_da = ds["G1"].transpose("x", "y", "z", missing_dims="ignore")
            G2_da = ds["G2"].transpose("x", "y", "z", missing_dims="ignore")

            # Keep existing behavior: use z=0 when available/meaningful (3D grids),
            # otherwise treat curvature as already 2D.
            is_curv_3d = ("z" in G1_da.dims) and (G1_da.sizes.get("z", 0) > 1)

            if is_curv_3d:
                # Compute ONLY the z=0 slice (shape: x,y)
                G1 = _as_numpy(G1_da.isel(z=0))
                G2 = _as_numpy(G2_da.isel(z=0))
            else:
                # Already 2D (or degenerate z) - compute as-is
                G1 = _as_numpy(G1_da)
                G2 = _as_numpy(G2_da)
            
            #Plot 4: G1 (geodesic curvature) on poloidal plane
            ax4 = fig.add_subplot(2, 3, 4)
            
            # We already sliced z=0 above when needed, so G1/G2 should now be 2D (x,y).
            # Keep the code explicit to prevent accidental regressions.
            G1_slice = G1[:, :]
            G2_slice = G2[:, :]

            # R and Z are already 2D poloidal-plane arrays (x,y).
            # For 3D grids we sliced z=0 earlier, so R/Z are still (x,y).
            R_grid = R
            Z_grid = Z
            
            # Create contour plot
            levels = 20
            norm1, vmin1, vmax1 = robust_norm(G1_slice, diverging=True, center=0.0, log_if_possible=False)

            # Guard against empty/all-NaN curvature fields:
            # - prevents f-string formatting of None
            # - avoids contourf failing when no finite values exist
            if norm1 is None:
                ax4.text(
                    0.5, 0.5,
                    "No finite G1 data to plot",
                    transform=ax4.transAxes,
                    fontsize=11,
                    ha="center", va="center",
                    alpha=0.8
                )
            else:
                contour1 = ax4.contourf(R_grid, Z_grid, G1_slice, levels=levels, cmap='RdBu_r', norm=norm1)
                cbar1 = plt.colorbar(contour1, ax=ax4)
                cbar1.set_label('G1 (1/m)', fontsize=10)

                ax4.text(
                    0.02, 0.02,
                    f"clipped [{vmin1:.3g}, {vmax1:.3g}]",
                    transform=ax4.transAxes,
                    fontsize=9,
                    alpha=0.75,
                    ha="left", va="bottom"
                )

            #Overlay flux surfaces.
            #IMPORTANT: R/Z are 2D (x,y), so the correct indexing is R[i, :] / Z[i, :].
            for i in [0, nx//4, nx//2, 3*nx//4, nx-1]:
                ax4.plot(R[i, :], Z[i, :], 'k-', linewidth=0.5, alpha=0.5)
            
            ax4.set_xlabel('R (m)', fontsize=12)
            ax4.set_ylabel('Z (m)', fontsize=12)
            ax4.set_title('Geodesic Curvature (G1)', fontsize=14, fontweight='bold')
            ax4.set_aspect('equal')
            ax4.grid(True, alpha=0.3)

            #Plot 5: G2 (normal curvature) on poloidal plane
            ax5 = fig.add_subplot(2, 3, 5)

            # G2 is already a 2D (x,y) poloidal-plane field because we sliced z=0 above if needed.
            # Keep this explicit so we never accidentally reintroduce full 3D loads.
            G2_slice = G2[:, :]

            # IMPORTANT:
            # Use the full 2D poloidal-plane mesh (R_grid, Z_grid) for contour plots.
            # Do NOT use R_slice/Z_slice here - those are 1D arrays from earlier flux-surface
            # line plots and may be stale from the last loop iteration.
            norm2, vmin2, vmax2 = robust_norm(G2_slice, diverging=True, center=0.0, log_if_possible=False)

            # Guard against empty/all-NaN curvature fields:
            if norm2 is None:
                ax5.text(
                    0.5, 0.5,
                    "No finite G2 data to plot",
                    transform=ax5.transAxes,
                    fontsize=11,
                    ha="center", va="center",
                    alpha=0.8
                )
            else:
                contour2 = ax5.contourf(R_grid, Z_grid, G2_slice, levels=levels, cmap='RdBu_r', norm=norm2)
                cbar2 = plt.colorbar(contour2, ax=ax5)
                cbar2.set_label('G2 (1/m)', fontsize=10)

                ax5.text(
                    0.02, 0.02,
                    f"clipped [{vmin2:.3g}, {vmax2:.3g}]",
                    transform=ax5.transAxes,
                    fontsize=9,
                    alpha=0.75,
                    ha="left", va="bottom"
                )

            #Overlay flux surfaces
            #R/Z are 2D (x,y): use R[i, :] not R[i, :, 0]
            for i in [0, nx//4, nx//2, 3*nx//4, nx-1]:
                ax5.plot(R[i, :], Z[i, :], 'k-', linewidth=0.5, alpha=0.5)

            ax5.set_xlabel('R (m)', fontsize=12)
            ax5.set_ylabel('Z (m)', fontsize=12)
            ax5.set_title('Normal Curvature (G2)', fontsize=14, fontweight='bold')
            ax5.set_aspect('equal')
            ax5.grid(True, alpha=0.3)

            #Plot 6: Total curvature magnitude
            ax6 = fig.add_subplot(2, 3, 6)
            G_mag = np.sqrt(G1_slice**2 + G2_slice**2)

            # IMPORTANT:
            # Same deal as Plot 5 - contour plots require the 2D mesh.
            # Reusing R_slice/Z_slice here is a bug because those are 1D line arrays.
            norm3, vmin3, vmax3 = robust_norm(G_mag, diverging=False, log_if_possible=True)
            contour3 = ax6.contourf(R_grid, Z_grid, G_mag, levels=levels, cmap='viridis', norm=norm3)
            cbar3 = plt.colorbar(contour3, ax=ax6)
            cbar3.set_label('|G| (1/m)', fontsize=10)

            ax6.text(
                0.02, 0.02,
                f"clipped [{vmin3:.3g}, {vmax3:.3g}]",
                transform=ax6.transAxes,
                fontsize=9,
                alpha=0.75,
                ha="left", va="bottom"
            )

            #Overlay flux surfaces
            #R/Z are 2D (x,y): use R[i, :] not R[i, :, 0]
            for i in [0, nx//4, nx//2, 3*nx//4, nx-1]:
                ax6.plot(R[i, :], Z[i, :], 'k-', linewidth=0.5, alpha=0.5)

            ax6.set_xlabel('R (m)', fontsize=12)
            ax6.set_ylabel('Z (m)', fontsize=12)
            ax6.set_title('Total Curvature |G|', fontsize=14, fontweight='bold')
            ax6.set_aspect('equal')
            ax6.grid(True, alpha=0.3)
            
            print("  ✓ Curvature data included in visualization")
        
        # ------------------------------------------------
        # Save figure
        # ------------------------------------------------
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Visualization saved: {output_path}")
        return True
        
    except Exception as e:
        log_exception(f"  ✗ Visualization failed: {e}")        
        return False

def render_magnetic_field_visualization(ds, output_path):
    """
    Generate magnetic field visualization.
    
    Creates a PNG showing:
    - B-field magnitude distribution
    - Poloidal field Bp distribution
    - Field line traces on poloidal plane
    - Safety factor q profile (if available)
    
    Parameters
    ----------
    ds : xarray.Dataset
        The grid dataset containing magnetic field data.
    output_path : str
        The path to save the generated PNG file.
    
    Returns
    -------
    bool
        True if the visualization is successfully saved, False if an error occurred.
    """
    try:
        print("\n=== Generating Magnetic Field Visualization ===")
        
        #Check if B-field data exists
        if "Bxy" not in ds or "Bpxy" not in ds:
            print("  ✗ No magnetic field data (Bxy, Bpxy) found - skipping B-field visualization")
            return False
        
        # Extract geometry and fields (dimension-order safe)
        # NOTE:
        #   Do NOT call `.values` on these 3D fields up-front.
        #   `.values` forces eager loading of the full arrays (defeats dask chunking),
        #   which is how you get node OOMs on real grids.
        #
        #   Instead:
        #     1) keep as xarray DataArrays
        #     2) slice the plane we actually plot (z=0 if present)
        #     3) convert ONLY that slice to numpy via _as_numpy()

        R_da = ds["R"].transpose("x", "y", "z", missing_dims="ignore")
        Z_da = ds["Z"].transpose("x", "y", "z", missing_dims="ignore")
        Bxy_da = ds["Bxy"].transpose("x", "y", "z", missing_dims="ignore")
        Bpxy_da = ds["Bpxy"].transpose("x", "y", "z", missing_dims="ignore")

        # Take poloidal slice (z=0) if z exists; otherwise use the 2D field directly
        if "z" in R_da.dims:
            R_da = R_da.isel(z=0)
        if "z" in Z_da.dims:
            Z_da = Z_da.isel(z=0)
        if "z" in Bxy_da.dims:
            Bxy_da = Bxy_da.isel(z=0)
        if "z" in Bpxy_da.dims:
            Bpxy_da = Bpxy_da.isel(z=0)

        # Convert ONLY the sliced plane to numpy (small + predictable memory)
        R_slice = _as_numpy(R_da)
        Z_slice = _as_numpy(Z_da)
        Bxy_slice = _as_numpy(Bxy_da)
        Bpxy_slice = _as_numpy(Bpxy_da)

        # Dimensions are now definitely 2D for plotting
        nx = R_slice.shape[0]
        ny = R_slice.shape[1]

        # Use the same outboard-midplane inference as render_grid_visualization(),
        # but also run the improved theta sanity checks without ever eager-loading
        # the full theta array.
        if "theta" in ds:
            theta_da = ds["theta"]

            # Pass R_slice/Z_slice (already 2D numpy planes) so:
            # - fallback works if theta is unusable
            # - optional theta-vs-geometry check can run
            y_mid_idx = infer_outboard_midplane_y_index(
                theta_da,
                R_slice,
                Z=Z_slice,
                context="mag field"
            )
        else:
            # Fallback: choose poloidal index where R is maximal at mid-radius
            x_mid = R_slice.shape[0] // 2
            R_line = R_slice[x_mid, :]
            y_mid_idx = int(np.nanargmax(R_line))

        # ------------------------------------------------------------------
        # Flux-surface overlay selection (visual rings)
        #
        # IMPORTANT:
        #   These are the "rings around the circle" in your plots.
        #   If each subplot uses a different set of indices, the rings will not match.
        #
        # Policy:
        #   Use ONE shared index set across Plot 1/2/3 for consistency.
        # ------------------------------------------------------------------
        flux_surface_idx = [0, nx//4, nx//2, 3*nx//4, nx-1]

        #Create figure with 4 subplots
        fig = plt.figure(figsize=(18, 10))

        # ------------------------------------------------
        # Plot 1: Total B-field magnitude
        # ------------------------------------------------
        ax1 = fig.add_subplot(2, 2, 1)
        
        levels = 20
        norm1, vmin1, vmax1 = robust_norm(Bxy_slice, diverging=False, log_if_possible=True)

        # robust_norm() can return (None, None, None) when the data are empty / all non-finite.
        # Two reasons to guard:
        #   1) f-strings like {vmin:.3g} will crash on None
        #   2) matplotlib contourf frequently errors when z is all-NaN (no valid levels)
        if norm1 is None:
            ax1.text(
                0.5, 0.5,
                "No finite |B| data to plot",
                transform=ax1.transAxes,
                fontsize=11,
                ha="center", va="center",
                alpha=0.8
            )
        else:
            contour1 = ax1.contourf(R_slice, Z_slice, Bxy_slice, levels=levels, cmap='plasma', norm=norm1)
            cbar1 = plt.colorbar(contour1, ax=ax1)
            cbar1.set_label('|B| (T)', fontsize=11)

            # Minimal scale annotation (publication-friendly without being noisy)
            ax1.text(
                0.02, 0.02,
                f"clipped [{vmin1:.3g}, {vmax1:.3g}]",
                transform=ax1.transAxes,
                fontsize=9,
                alpha=0.75,
                ha="left", va="bottom"
            )

        
        # Plot selected flux surfaces (poloidal lines)
        # NOTE:
        #   Plot 2 is treated as the reference for how many rings to show.
        #   Using too many rings in Plot 1 makes it look "different" even when the geometry is identical.
        for i in flux_surface_idx:
            ax1.plot(R_slice[i, :], Z_slice[i, :], 'k-', linewidth=0.8, alpha=0.4)
        
        ax1.set_xlabel('R (m)', fontsize=12)
        ax1.set_ylabel('Z (m)', fontsize=12)
        ax1.set_title('Total Magnetic Field |B|', fontsize=14, fontweight='bold')
        ax1.set_aspect('equal')
        ax1.grid(True, alpha=0.3)
        
        # ------------------------------------------------
        # Plot 2: Poloidal B-field magnitude
        # ------------------------------------------------
        ax2 = fig.add_subplot(2, 2, 2)
        
        norm2, vmin2, vmax2 = robust_norm(Bpxy_slice, diverging=False, log_if_possible=True)

        # Guard against robust_norm() returning None-triplet on empty/all-NaN inputs.
        if norm2 is None:
            ax2.text(
                0.5, 0.5,
                "No finite Bp data to plot",
                transform=ax2.transAxes,
                fontsize=11,
                ha="center", va="center",
                alpha=0.8
            )
        else:
            contour2 = ax2.contourf(R_slice, Z_slice, Bpxy_slice, levels=levels, cmap='viridis', norm=norm2)
            cbar2 = plt.colorbar(contour2, ax=ax2)
            cbar2.set_label('Bp (T)', fontsize=11)

            ax2.text(
                0.02, 0.02,
                f"clipped [{vmin2:.3g}, {vmax2:.3g}]",
                transform=ax2.transAxes,
                fontsize=9,
                alpha=0.75,
                ha="left", va="bottom"
            )
        
        # Overlay flux surfaces (visual rings)
        # NOTE:
        #   Use the SAME flux_surface_idx as Plot 1 so the rings match across subplots.
        for i in flux_surface_idx:
            ax2.plot(R_slice[i, :], Z_slice[i, :], 'k-', linewidth=0.8, alpha=0.4)
        
        ax2.set_xlabel('R (m)', fontsize=12)
        ax2.set_ylabel('Z (m)', fontsize=12)
        ax2.set_title('Poloidal Magnetic Field Bp', fontsize=14, fontweight='bold')
        ax2.set_aspect('equal')
        ax2.grid(True, alpha=0.3)
        
        # ------------------------------------------------
        # Plot 3: Field line structure (quiver plot)
        # ------------------------------------------------
        ax3 = fig.add_subplot(2, 2, 3)
        
        #Compute Bp components for quiver plot
        #For field-aligned coordinates, Bp is along e_theta direction
        #Approximate: Bp_R ≈ -Bp * sin(theta), Bp_Z ≈ Bp * cos(theta)
        
        #Subsample for cleaner quiver plot
        skip = max(1, nx // 20)
        skip_y = max(1, ny // 20)
        
        R_sub = R_slice[::skip, ::skip_y]
        Z_sub = Z_slice[::skip, ::skip_y]
        Bp_sub = Bpxy_slice[::skip, ::skip_y]
        
        #Estimate poloidal angle for direction
        R_center = R_slice.mean()
        Z_center = Z_slice.mean()
        
        theta_sub = np.arctan2(Z_sub - Z_center, R_sub - R_center)
        
        #Bp components (pointing along poloidal direction)
        Bp_R = -Bp_sub * np.sin(theta_sub)
        Bp_Z = Bp_sub * np.cos(theta_sub)
        
        #Normalize for visualization
        Bp_mag = np.sqrt(Bp_R**2 + Bp_Z**2)
        Bp_R_norm = Bp_R / (Bp_mag + 1e-10)
        Bp_Z_norm = Bp_Z / (Bp_mag + 1e-10)
        
        #Plot as quiver
        ax3.quiver(R_sub, Z_sub, Bp_R_norm, Bp_Z_norm, 
                  Bp_mag, cmap='cool', alpha=0.8, scale=30)
        
        # Overlay flux surfaces (visual rings)
        # NOTE:
        #   Use the SAME flux_surface_idx as Plot 1/2 so the rings match across subplots.
        for i in flux_surface_idx:
            ax3.plot(R_slice[i, :], Z_slice[i, :], 'k-', linewidth=1.2, alpha=0.6)
        
        ax3.set_xlabel('R (m)', fontsize=12)
        ax3.set_ylabel('Z (m)', fontsize=12)
        ax3.set_title('Poloidal Field Direction', fontsize=14, fontweight='bold')
        ax3.set_aspect('equal')
        ax3.grid(True, alpha=0.3)
        
        # ------------------------------------------------
        # Plot 4: Radial profiles
        # ------------------------------------------------
        ax4 = fig.add_subplot(2, 2, 4)

        # NOTE:
        # Use the SAME outboard-midplane heuristic as the geometry visualization:
        #   - theta≈0 if available/usable
        #   - otherwise R-max at mid-radius.
        #
        # This keeps the "radial profiles" consistent with the other plots and avoids
        # silently assuming y=0 means outboard midplane.
        #
        # We use the already z-sliced plane arrays (Bxy_slice / Bpxy_slice),
        # which are 2D numpy arrays produced above. This avoids ever loading full 3D fields.
        B_profile = Bxy_slice[:, y_mid_idx]
        Bp_profile = Bpxy_slice[:, y_mid_idx]
        
        x_coord = np.arange(nx)
        
        ax4.plot(x_coord, B_profile, 'r-', linewidth=2, label='|B| (total)')
        ax4.plot(x_coord, Bp_profile, 'b-', linewidth=2, label='Bp (poloidal)')
        
        #Compute toroidal field (approximately)
        #Handle case where Bp > B due to numerical precision
        Bt_squared = B_profile**2 - Bp_profile**2
        Bt_squared = np.maximum(Bt_squared, 0)  #Clamp to non-negative
        Bt_profile = np.sqrt(Bt_squared)
        ax4.plot(x_coord, Bt_profile, 'g--', linewidth=2, label='Bt (toroidal, est.)')
        
        ax4.set_xlabel('Radial Index (x)', fontsize=12)
        ax4.set_ylabel('B-field (T)', fontsize=12)
        ax4.set_title('Radial B-field Profiles', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.legend(fontsize=10)
        
        #Add secondary axis for Bp/Bt ratio (pitch)
        ax4_twin = ax4.twinx()
        pitch_ratio = Bp_profile / (Bt_profile + 1e-10)
        ax4_twin.plot(x_coord, pitch_ratio, 'm:', linewidth=2, alpha=0.7, label='Bp/Bt')
        ax4_twin.set_ylabel('Bp/Bt (pitch)', fontsize=12, color='m')
        ax4_twin.tick_params(axis='y', labelcolor='m')
        ax4_twin.legend(loc='upper right', fontsize=10)
        
        # ------------------------------------------------
        # Save figure
        # ------------------------------------------------
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Magnetic field visualization saved: {output_path}")
        return True
        
    except Exception as e:
        log_exception(f"  ✗ Magnetic field visualization failed: {e}")
        return False

class Reports:
    @staticmethod
    def generate_html_report(path, ds, validation_results, output_html):
        """
        Generate comprehensive HTML report with all diagnostics and visualizations.
        
        Parameters
        ----------
        path : str
            Path to grid file.
        ds : xarray.Dataset
            Grid dataset.
        validation_results : dict
            Dictionary containing all validation results.
        output_html : str
            Path to output HTML file.
        
        Returns
        -------
        bool
            True if the report is successfully generated, False if an error occurred.
        """

        def _h(x):
            return html.escape(str(x))    

        #Local helper: HTML-escape any value before injecting into the report.
        #Prevents accidental HTML injection via metadata/strings from the grid file.

        try:
            print("\n=== Generating HTML Report ===")
            
            # Extract results
            nx = validation_results['nx']
            ny = validation_results['ny']
            nz = validation_results['nz']
            grid_type = validation_results['grid_type']
            grid_format = validation_results.get('grid_format', 'UNKNOWN')
            dimensionality = validation_results.get('dimensionality', 'UNKNOWN')
            config_type = validation_results['config_type']
            kappa = validation_results['kappa']
            aspect_ratio = validation_results['aspect_ratio']
            critical_failures = validation_results['critical_failures']

            metric_interpretation = validation_results.get('metric_interpretation', None)
            metric_det_relation = validation_results.get('metric_det_relation', None)
            metric_det_mean_rel_err = validation_results.get('metric_det_mean_rel_err', None)
            metric_det_max_rel_err = validation_results.get('metric_det_max_rel_err', None)

            
            #Get image paths (prefer actual output paths passed via validation_results;
            #fallback to legacy "same-as-input" naming)
            render_img = validation_results.get('render_img') or path.replace('.nc', '_render.png')
            bfield_img = validation_results.get('bfield_img') or path.replace('.nc', '_bfield.png')
            
            # Check which images exist
            has_render = bool(render_img) and os.path.exists(render_img)
            has_bfield = bool(bfield_img) and os.path.exists(bfield_img)
            
            #Generate HTML
            html_content = f"""<!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <!--
          SECURITY / ROBUSTNESS NOTE:
          Even though `path` normally comes from the local filesystem, it is still untrusted input
          (e.g. weird filenames or upstream wrappers). Always HTML-escape values before injecting
          into markup to prevent accidental HTML/script injection.
        -->
        <title>BOUT++ Grid Diagnostics Report - {_h(os.path.basename(path))}</title>

        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}
            
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 20px;
                line-height: 1.6;
            }}
            
            .container {{
                max-width: 1400px;
                margin: 0 auto;
                background: white;
                border-radius: 12px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                overflow: hidden;
            }}
            
            .header {{
                background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
                color: white;
                padding: 40px;
                text-align: center;
            }}
            
            .header h1 {{
                font-size: 2.5em;
                margin-bottom: 10px;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            }}
            
            .header .filename {{
                font-size: 1.2em;
                opacity: 0.9;
                font-family: monospace;
                background: rgba(255,255,255,0.1);
                padding: 10px 20px;
                border-radius: 5px;
                display: inline-block;
                margin-top: 10px;
            }}
            
            .verdict {{
                padding: 30px 40px;
                text-align: center;
                font-size: 1.5em;
                font-weight: bold;
            }}
            
            .verdict.valid {{
                background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
                color: white;
            }}
            
            .verdict.invalid {{
                background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
                color: white;
            }}
            
            .verdict .icon {{
                font-size: 3em;
                margin-bottom: 10px;
            }}
            
            .content {{
                padding: 40px;
            }}
            
            .section {{
                margin-bottom: 40px;
            }}
            
            .section h2 {{
                color: #1e3c72;
                border-bottom: 3px solid #667eea;
                padding-bottom: 10px;
                margin-bottom: 20px;
                font-size: 1.8em;
            }}
            
            .grid-info {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                margin-bottom: 30px;
            }}
            
            .info-card {{
                background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                padding: 20px;
                border-radius: 8px;
                border-left: 4px solid #667eea;
            }}
            
            .info-card h3 {{
                color: #1e3c72;
                font-size: 0.9em;
                text-transform: uppercase;
                margin-bottom: 5px;
                opacity: 0.8;
            }}
            
            .info-card .value {{
                font-size: 1.8em;
                font-weight: bold;
                color: #2a5298;
            }}
            
            .check-list {{
                background: #f8f9fa;
                padding: 20px;
                border-radius: 8px;
                margin-bottom: 20px;
            }}
            
            .check-item {{
                padding: 10px;
                margin: 5px 0;
                border-radius: 5px;
                display: flex;
                align-items: center;
            }}
            
            .check-item.pass {{
                background: #d4edda;
                color: #155724;
            }}

            .check-item.skip {{
                background: #e2e3e5;
                color: #383d41;
            }}

            
            .check-item.fail {{
                background: #f8d7da;
                color: #721c24;
            }}
            
            .check-item .icon {{
                margin-right: 10px;
                font-weight: bold;
                font-size: 1.2em;
            }}
            
            .warnings {{
                background: #fff3cd;
                border-left: 4px solid #ffc107;
                padding: 20px;
                border-radius: 5px;
                margin-top: 20px;
            }}
            
            .warnings h3 {{
                color: #856404;
                margin-bottom: 10px;
            }}
            
            .warnings ul {{
                list-style: none;
                padding-left: 0;
            }}
            
            .warnings li {{
                padding: 8px 0;
                border-bottom: 1px solid rgba(133,100,4,0.1);
            }}
            
            .warnings li:before {{
                content: "⚠ ";
                margin-right: 8px;
            }}
            
            .failures {{
                background: #f8d7da;
                border-left: 4px solid #dc3545;
                padding: 20px;
                border-radius: 5px;
                margin-top: 20px;
            }}
            
            .failures h3 {{
                color: #721c24;
                margin-bottom: 10px;
            }}
            
            .failures ul {{
                list-style: none;
                padding-left: 0;
            }}
            
            .failures li {{
                padding: 8px 0;
                border-bottom: 1px solid rgba(114,28,36,0.1);
                font-weight: bold;
            }}
            
            .failures li:before {{
                content: "✗ ";
                margin-right: 8px;
            }}
            
            .visualization {{
                margin: 30px 0;
                text-align: center;
            }}
            
            .visualization img {{
                max-width: 100%;
                height: auto;
                border-radius: 8px;
                box-shadow: 0 4px 20px rgba(0,0,0,0.1);
                margin: 10px 0;
            }}
            
            .visualization h3 {{
                color: #2a5298;
                margin-bottom: 15px;
                font-size: 1.4em;
            }}
            
            .metrics-table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
                background: white;
                border-radius: 8px;
                overflow: hidden;
                box-shadow: 0 2px 10px rgba(0,0,0,0.05);
            }}
            
            .metrics-table th {{
                background: #1e3c72;
                color: white;
                padding: 15px;
                text-align: left;
                font-weight: 600;
            }}
            
            .metrics-table td {{
                padding: 12px 15px;
                border-bottom: 1px solid #e9ecef;
            }}
            
            .metrics-table tr:hover {{
                background: #f8f9fa;
            }}
            
            .footer {{
                background: #2a5298;
                color: white;
                padding: 20px;
                text-align: center;
                font-size: 0.9em;
            }}
            
            .footer a {{
                color: #38ef7d;
                text-decoration: none;
            }}

            /* Collapsible grouped checklist (accordion) */
            details.check-group {{
                background: #ffffff;
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.05);
                margin: 12px 0;
                overflow: hidden;
                border: 1px solid #e9ecef;
            }}

            summary.check-group-summary {{
                cursor: pointer;
                list-style: none;
                padding: 14px 16px;
                font-weight: 700;
                display: flex;
                align-items: center;
                justify-content: space-between;
                user-select: none;
            }}

            summary.check-group-summary::-webkit-details-marker {{
                display: none;
            }}

            .group-title {{
                display: flex;
                align-items: center;
                gap: 10px;
                font-size: 1.05em;
            }}

            .group-pill {{
                font-weight: 800;
                padding: 3px 10px;
                border-radius: 999px;
                font-size: 0.85em;
                color: white;
            }}

            .pill-critical {{ background: #dc3545; }}
            .pill-warn     {{ background: #ffc107; color: #212529; }}
            .pill-info     {{ background: #6c757d; }}

            .group-counts {{
                font-family: monospace;
                font-size: 0.9em;
                opacity: 0.85;
            }}

            .group-body {{
                padding: 10px 14px 14px 14px;
                background: #f8f9fa;
            }}

            
            @media print {{
                body {{
                    background: white;
                    padding: 0;
                }}
                .container {{
                    box-shadow: none;
                }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>BOUT++ Grid Diagnostics Report</h1>

                <!--
                  SECURITY / ROBUSTNESS NOTE:
                  Filenames can contain characters that have meaning in HTML.
                  Escape before embedding to avoid broken markup or injection.
                -->
                <div class="filename">{_h(os.path.basename(path))}</div>
            </div>
            
            <div class="verdict {'valid' if len(critical_failures) == 0 else 'invalid'}">
                <div class="icon">{'✓' if len(critical_failures) == 0 else '✗'}</div>
                <div>Grid File {'VALID' if len(critical_failures) == 0 else 'INVALID'}</div>
            </div>
            
            <div class="content">
                <div class="section">
                    <h2>Grid Summary</h2>
                    <div class="grid-info">
                        <div class="info-card">
                            <h3>Grid Type</h3>
                            <div class="value">{grid_type}</div>
                        </div>
                        <div class="info-card">
                            <h3>Grid Format</h3>
                            <div class="value">{grid_format}</div>
                        </div>
                        <div class="info-card">
                            <h3>Configuration</h3>
                            <div class="value">{config_type}</div>
                        </div>
                        <div class="info-card">
                            <h3>Dimensions (x×y×z)</h3>
                            <div class="value">{nx}×{ny}×{nz}</div>
                        </div>
                        <div class="info-card">
                            <h3>Dimensionality</h3>
                            <div class="value">{dimensionality}</div>
                        </div>
                        <div class="info-card">
                            <h3>Elongation κ</h3>
                            <div class="value">{kappa:.2f}</div>
                        </div>
                        <div class="info-card">
                            <h3>Aspect Ratio</h3>
                            <div class="value">{aspect_ratio:.2f}</div>
                        </div>
                        <div class="info-card">
                            <h3>Metric Interpretation</h3>
                            <div class="value">{metric_interpretation or 'N/A'}</div>
                        </div>
                        <div class="info-card">
                            <h3>det(g) vs J Consistency</h3>
                            <div class="value" style="font-size: 1.1em;">
                                {metric_det_relation or 'N/A'}<br>
                                <span style="font-size: 0.9em; font-family: monospace;">
                                    mean={metric_det_mean_rel_err if metric_det_mean_rel_err is not None else 'N/A'}
                                    max={metric_det_max_rel_err if metric_det_max_rel_err is not None else 'N/A'}
                                </span>
                            </div>
                        </div>

                    </div>
            """

            checks_list = validation_results.get("checks", [])

            # ------------------------------------------------------------
            # "Detailed Metrics" status mapping (presentation-only)
            #
            # The metrics table currently uses a naive heuristic (min_val > 0
            # or name contains 'J'/'surf'), which can contradict the real
            # validation checks already recorded in checks_list.
            #
            # Here we define a minimal mapping from each metric row to the
            # relevant check names in `checks_list`. Status in the metrics table
            # will reflect FAIL/PASS/SKIP of these checks.
            #
            # NOTE: This does NOT change any validation logic. It only changes
            # how the HTML status column is derived.
            # ------------------------------------------------------------

            metric_check_map = {
                # Metric tensor positivity/sanity checks
                "g11": ["metric:g11:no_nan", "metric:g11:no_inf"],
                "g22": ["metric:g22:no_nan", "metric:g22:no_inf"],
                "g33": ["metric:g33:no_nan", "metric:g33:no_inf"],

                # Jacobian: BOUT++ v5 allows negative J; key requirement is sign consistency
                "Jacobian J": ["jacobian:sign_consistent"],

                # surfvol: optional; when present you check sign consistency, match to J, ratio finite
                "surfvol": ["surfvol:present", "surfvol:sign_consistent", "surfvol:sign_matches_J", "surfvol:ratio_finite"],

                # Optional B-field (only add this row if Bxy exists)
                "B-field |B|": ["Bxy positive"],
            }

            def _check_index(checks):
                """
                Build a dict: check_name -> check_record.
                If a check name appears more than once, the last one wins (fine for our usage).
                """
                idx = {}
                for c in checks:
                    idx[str(c.get("name", ""))] = c
                return idx

            checks_by_name = _check_index(checks_list)

            def _metric_status_from_checks(metric_label):
                """
                Determine a compact status string for a metric table row based on recorded checks.

                Priority:
                  - Any FAIL in mapped checks => ✗ Issue
                  - Else any SKIP in mapped checks => ⏭ N/A (optional / skipped)
                  - Else (all PASS or no mapped checks found) => ✓ OK

                This keeps the metrics table consistent with the real validation results.
                """
                names = metric_check_map.get(metric_label, [])
                if not names:
                    return "✓ OK"  # fallback (should be rare)

                statuses = []
                for n in names:
                    rec = checks_by_name.get(n)
                    if rec is None:
                        continue
                    statuses.append((rec.get("status") or "SKIP").upper())

                if any(s == "FAIL" for s in statuses):
                    return "✗ Issue"
                if any(s == "SKIP" for s in statuses):
                    return "⏭ N/A"
                return "✓ OK"

            #Local UI helpers (HTML report):
            #Convert check status/severity into CSS classes, icons, and grouped accordion sections.
            #These do not affect validation logic - only presentation in the generated HTML.

            def _status_class(s):
                """Map a check status string to (css_class, icon). Unknown -> SKIP."""
                s = (s or "").upper()
                if s == "PASS":
                    return "pass", "✓"
                if s == "FAIL":
                    return "fail", "✗"
                return "skip", "⏭"

            def _sev_badge(sev):
                sev = (sev or "INFO").upper()
                if sev == "CRITICAL":
                    return '<span style="font-weight:700; padding:2px 8px; border-radius:999px; background:#dc3545; color:white; margin-left:10px;">CRITICAL</span>'
                if sev == "WARN":
                    return '<span style="font-weight:700; padding:2px 8px; border-radius:999px; background:#ffc107; color:#212529; margin-left:10px;">WARN</span>'
                return '<span style="font-weight:700; padding:2px 8px; border-radius:999px; background:#6c757d; color:white; margin-left:10px;">INFO</span>'

            # Summary counts
            pass_n = sum(1 for c in checks_list if c.get("status") == "PASS")
            fail_n = sum(1 for c in checks_list if c.get("status") == "FAIL")
            skip_n = sum(1 for c in checks_list if c.get("status") == "SKIP")

            html_content += f"""
                        <div class="check-list">
                            <div class="check-item {'fail' if critical_failures else 'pass'}">
                                <span class="icon">{'✗' if critical_failures else '✓'}</span>
                                <span>{f"{len(critical_failures)} critical failure(s) detected" if critical_failures else "No critical failures reported"}</span>
                            </div>

                            <div style="margin-top: 12px; font-size: 0.95em;">
                                <strong>Checks:</strong>
                                <span style="font-family: monospace;">PASS={pass_n} FAIL={fail_n} SKIP={skip_n}</span>
                            </div>

                            <div style="margin-top: 15px;">
            """

            # Group checks by severity for collapsible UI
            groups = {
                "CRITICAL": [],
                "WARN": [],
                "INFO": [],
            }

            for c in checks_list:
                sev = (c.get("severity") or "INFO").upper()
                if sev not in groups:
                    sev = "INFO"
                groups[sev].append(c)

            def _group_counts(items):
                p = sum(1 for c in items if (c.get("status") or "SKIP").upper() == "PASS")
                f = sum(1 for c in items if (c.get("status") or "SKIP").upper() == "FAIL")
                s = sum(1 for c in items if (c.get("status") or "SKIP").upper() == "SKIP")
                return p, f, s

            def _render_group(sev, items, pill_class, open_by_default=False):
                # Render one collapsible <details> group for checks of a given severity.
                # Keeps the report readable when there are many INFO/WARN items.

                p, f, s = _group_counts(items)

                #Group header
                html = f"""
                                    <details class="check-group" {'open' if open_by_default else ''}>
                                        <summary class="check-group-summary">
                                            <div class="group-title">
                                                <span class="group-pill {pill_class}">{sev}</span>
                                                <span>{sev} checks</span>
                                            </div>
                                            <div class="group-counts">PASS={p} FAIL={f} SKIP={s}</div>
                                        </summary>
                                        <div class="group-body">
                """

                # Group items
                for c in items:
                    status = (c.get("status") or "SKIP").upper()
                    cls, icon = _status_class(status)
                    name = _h(c.get("name", "unnamed-check"))
                    details = c.get("details", None)

                    detail_html = ""
                    if details is not None:
                        if isinstance(details, dict):
                            kv = ", ".join([f"{_h(k)}={_h(v)}" for k, v in details.items()])
                            detail_html = f'<div style="margin-left: 28px; font-family: monospace; font-size: 0.9em; opacity: 0.85;">{kv}</div>'
                        else:
                            detail_html = f'<div style="margin-left: 28px; font-family: monospace; font-size: 0.9em; opacity: 0.85;">{_h(details)}</div>'

                    #Inside a severity group, the severity badge is redundant (and noisy), so omit it.
                    html += f"""
                                            <div class="check-item {cls}">
                                                <span class="icon">{icon}</span>
                                                <span>{name}</span>
                                            </div>
                                            {detail_html}
                    """

                html += """
                                        </div>
                                    </details>
                """
                return html

            # Render groups (keep them collapsed by default to avoid eye-burn)
            html_content += _render_group("CRITICAL", groups["CRITICAL"], "pill-critical", open_by_default=False)
            html_content += _render_group("WARN",     groups["WARN"],     "pill-warn",     open_by_default=False)
            html_content += _render_group("INFO",     groups["INFO"],     "pill-info",     open_by_default=False)

            # Close the check-list wrapper
            html_content += """
                                </div>
                            </div>
            """

            html_content += """
                    <div style="margin-top: 12px; font-size: 0.95em; opacity: 0.9;">
                        <em>
                            Detailed results are shown in the grouped checklist above
                            (CRITICAL / WARN / INFO).
                        </em>
                    </div>
            """

            html_content += """
                </div>
                
                <div class="section">
                    <h2>Detailed Metrics</h2>
                    <table class="metrics-table">
                        <thead>
                            <tr>
                                <th>Component</th>
                                <th>Minimum</th>
                                <th>Maximum</th>
                                <th>Status</th>
                            </tr>
                        </thead>
                        <tbody>
            """

            # Add metric table rows
            metrics = [
                ('g11', validation_results.get('g11_min'), validation_results.get('g11_max')),
                ('g22', validation_results.get('g22_min'), validation_results.get('g22_max')),
                ('g33', validation_results.get('g33_min'), validation_results.get('g33_max')),
                ('Jacobian J', validation_results.get('J_min'), validation_results.get('J_max')),
                ('surfvol', validation_results.get('surf_min'), validation_results.get('surf_max')),
            ]
            
            if 'Bxy' in ds:
                metrics.append(('B-field |B|', validation_results.get('B_min'), validation_results.get('B_max')))
            
            for name, min_val, max_val in metrics:
                if min_val is not None and max_val is not None:
                    # Use recorded validation checks (real results) rather than a min_val/name heuristic.
                    status = _metric_status_from_checks(name)

                    html_content += f"""
                            <tr>
                                <td><strong>{name}</strong></td>
                                <td>{min_val:.4e}</td>
                                <td>{max_val:.4e}</td>
                                <td>{status}</td>
                            </tr>
                    """

                else:
                    #Explicitly show missing metrics (e.g. surfvol)
                    html_content += f"""
                            <tr>
                                <td><strong>{name}</strong></td>
                                <td colspan="2"><em>missing</em></td>
                                <td>⚠</td>
                            </tr>
                    """

            html_content += """
                        </tbody>
                    </table>
                </div>
            """

            #Add visualizations
            if has_render:
                html_content += f"""
                <div class="section">
                    <h2>Grid Geometry & Curvature</h2>
                    <div class="visualization">
                        <img src="./{os.path.basename(render_img)}" alt="Grid Geometry Visualization">
                    </div>
                </div>
                """

            if has_bfield:
                html_content += f"""
                <div class="section">
                    <h2>Magnetic Field Structure</h2>
                    <div class="visualization">
                        <img src="./{os.path.basename(bfield_img)}" alt="Magnetic Field Visualization">
                    </div>
                </div>
                """

            #Footer
            html_content += """
            </div>
            
            <div class="footer">
                Generated by <strong>Chatwood Labs BOUT++ Grid Diagnostics Utility</strong><br>
                <small>BOUT++ 5.x Compatible | Python Grid Validation Tool</small>
            </div>
        </div>
    </body>
    </html>
    """

            # Write HTML file
            with open(output_html, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            print(f"  ✓ HTML report saved: {output_html}")
            return True
            
        except Exception as e:
            log_exception(f"  ✗ HTML report generation failed: {e}")
            return False
            
    @staticmethod
    def write_json_report(output_html, validation_results):
        """
        Write a machine-readable JSON report next to the HTML report.
        Uses the same base filename as the HTML (e.g., *_report.html -> *_report.json).

        Notes:
          - Ensures report artifact paths in JSON are RELATIVE to the report directory
            (portable on GitHub / between machines).
        """
        try:

            def _jsonable(obj):
                """
                Convert a value into something that json.dump() can serialize.

                Why this exists (and why it's careful):
                  - xarray reductions often return 0-D DataArray objects.
                    If we return those directly, json.dump() will fail.
                  - Calling `.values` on a dask-backed DataArray can *materialize*
                    large arrays. The JSON report must never accidentally load
                    an entire 3D field into RAM.

                Policy:
                  - Prefer scalarization via `to_scalar()` for xarray / dask objects.
                  - If a *non-scalar* array sneaks in, only allow it if it's small;
                    otherwise fail loudly (caller bug / report design issue).
                """
                # numpy scalars/arrays
                if isinstance(obj, np.generic):
                    return obj.item()
                if isinstance(obj, np.ndarray):
                    # Guard against gigantic arrays accidentally being serialized.
                    # JSON reports should contain summaries, not full fields.
                    if obj.size > 1_000_000:
                        raise TypeError(
                            f"Refusing to JSON-serialize large ndarray (shape={obj.shape}, dtype={obj.dtype}). "
                            "Store reductions (min/max/mean) instead."
                        )
                    return obj.tolist()

                # xarray objects (if any sneak in)
                #
                # IMPORTANT:
                #   Do NOT blindly do `obj.values.tolist()`:
                #     - `.values` may compute dask-backed arrays (huge)
                #     - 0-D DataArray `.values` may be a numpy scalar, not ndarray
                if hasattr(obj, "values"):
                    # First attempt: treat it as scalar-like (safe for reductions).
                    try:
                        s = to_scalar(obj)
                        # If scalarization succeeded, we're done.
                        if isinstance(s, (bool, int, float, str)) or s is None:
                            return s
                        # If to_scalar returns something odd but JSONable, still accept.
                        return s
                    except Exception:
                        # Fall back to extracting values, but with size guards.
                        try:
                            v = obj.values
                            if isinstance(v, np.generic):
                                return v.item()
                            if isinstance(v, np.ndarray):
                                if v.size > 1_000_000:
                                    raise TypeError(
                                        f"Refusing to JSON-serialize large xarray values array (shape={v.shape}, dtype={v.dtype}). "
                                        "Store reductions (min/max/mean) instead."
                                    )
                                return v.tolist()
                        except Exception:
                            # Give up; let containers below or final return handle it.
                            pass

                # containers
                if isinstance(obj, dict):
                    return {str(k): _jsonable(v) for k, v in obj.items()}
                if isinstance(obj, (list, tuple)):
                    return [_jsonable(v) for v in obj]
                if isinstance(obj, set):
                    return [_jsonable(v) for v in sorted(obj)]

                return obj


            def _relativize_report_paths(obj, base_dir):
                """
                Convert known artifact path fields to paths relative to the report directory.
                Keeps the JSON portable and avoids leaking local filesystem paths.
                """
                if isinstance(obj, dict):
                    out = {}
                    for k, v in obj.items():
                        kk = str(k)
                        if kk in ("render_img", "bfield_img") and isinstance(v, str) and v:
                            try:
                                v_abs = os.path.abspath(v)
                                out[kk] = os.path.relpath(v_abs, start=base_dir)
                            except Exception:
                                out[kk] = v
                        else:
                            out[kk] = _relativize_report_paths(v, base_dir)
                    return out

                if isinstance(obj, list):
                    return [_relativize_report_paths(v, base_dir) for v in obj]

                return obj

            output_json = os.path.splitext(output_html)[0] + ".json"

            # Base directory = directory containing the report files
            base_dir = os.path.dirname(os.path.abspath(output_html)) or "."

            payload = _jsonable(validation_results)
            payload = _relativize_report_paths(payload, base_dir)

            with open(output_json, "w") as f:
                json.dump(payload, f, indent=2, sort_keys=True)

            print(f"JSON report written: {output_json}")
            return output_json

        except Exception as e:
            print(f"Failed to write JSON report: {e}")
            return None

# Back-compat wrappers
def generate_html_report(path, ds, validation_results, output_html):
    return Reports.generate_html_report(path, ds, validation_results, output_html)

def write_json_report(output_html, validation_results):
    return Reports.write_json_report(output_html, validation_results)

class DiagnosticsRunner:
    @staticmethod
    def main(
        path,
        outdir=None,
        make_plots=True,
        make_html=True,
        json_only=False,
        strict=False,
        det_relerr_mean_max=1e-3,
        det_relerr_max_max=1e-2,
        metric_cond_warn=1e6,
        jacobian_singularity_min=1e-10,
        aspect_ratio_warn_min=2.0,
        ny_warn_min=32,

        # Chunking controls (threaded from CLI -> main -> here)
        force_chunk=False,
        chunks=None,
    ):

        """
        Run diagnostics on a BOUT++ grid file.
        
        Parameters
        ----------
        path : str
            The path to the grid file to be analyzed.
        outdir : str, optional
            The directory where output files (reports, plots) will be saved (default is None, which saves in the same directory as the grid file).
        make_plots : bool, optional
            Whether to generate visual plots (default is True).
        make_html : bool, optional
            Whether to generate an HTML report (default is True).
        json_only : bool, optional
            If True, only generates the JSON report, skipping other outputs (default is False).
        strict : bool, optional
            Whether to apply strict validation rules (default is False).
        det_relerr_mean_max : float, optional
            Maximum allowed relative error in the determinant of the metric (default is 1e-3).
        det_relerr_max_max : float, optional
            Maximum allowed maximum relative error in the determinant of the metric (default is 1e-2).
        metric_cond_warn : float, optional
            The condition number of the metric above which a warning is issued (default is 1e6).
        jacobian_singularity_min : float, optional
            The minimum Jacobian value considered singular (default is 1e-10).
        aspect_ratio_warn_min : float, optional
            The minimum aspect ratio considered valid (default is 2.0).
        ny_warn_min : int, optional
            The minimum number of grid points in the y-direction considered valid (default is 32).
        
        Returns
        -------
        None
        """
        print("=== BOUT++ Grid Diagnostics (5.x) ===")
        print(f"Loading: {path}")
        checks = []

        # ----------------------------------------------------------
        # Output routing:
        # Default is "next to the input grid", unless --outdir is given.
        # ----------------------------------------------------------
        if outdir is None:
            outdir = os.path.dirname(os.path.abspath(path)) or "."

        #Make sure outdir exists (non-fatal if it already exists)
        os.makedirs(outdir, exist_ok=True)

        #Base name: <gridfilename> without extension, plus suffixes later.
        _base = os.path.splitext(os.path.basename(path))[0]

        def _outpath(suffix):
            #suffix examples: "_report.html", "_report.json", "_render.png"
            return os.path.join(outdir, _base + suffix)

        # Load dataset using xarray.
        #
        # Default behavior: use internal heuristics to decide whether chunking is needed.
        # Advanced behavior:
        #   --force-chunk forces chunking on every run (useful when heuristics undercount).
        #   --chunks lets the user control chunk sizes (otherwise internal defaults are used).
        ds = maybe_open_dataset(path, force_chunk=force_chunk, chunks=chunks)

        #Detect/normalize BOUT4 vs BOUT5 naming conventions (in-memory only)
        ds, grid_format = detect_and_normalize_grid(ds)
        print(f"Grid format: {grid_format}")

        # If we opened chunked (dask), DO NOT automatically eager-load the entire dataset.
        #
        # Why:
        # - Chunking is used specifically when the estimated working-set is large vs available RAM.
        # - Calling ds.load() forces *all* variables into memory and defeats the purpose of chunking.
        #
        # What we do instead:
        # - Eager-load only when the estimated working-set is comfortably below available RAM.
        # - Otherwise, keep things lazy; downstream computations can .compute() selectively.
        if ds.attrs.get("_chunked_open", False):
            est_bytes = int(ds.attrs.get("_est_working_set_bytes", 0) or 0)
            avail_bytes = int(ds.attrs.get("_avail_mem_bytes", 0) or 0)

            # Conservative guardrails:
            # - Only eager-load if estimate is <= 25% of available RAM
            # - AND also capped to a hard upper bound (8 GiB) to avoid surprising workstation OOMs.
            eager_load_frac_threshold = 0.25
            eager_load_cap_bytes = 8 * 1024**3

            # If we can't estimate available RAM, default to NOT eager-loading.
            if avail_bytes > 0:
                safe_by_frac = (est_bytes > 0) and (est_bytes <= eager_load_frac_threshold * avail_bytes)
            else:
                safe_by_frac = False

            safe_by_cap = (est_bytes > 0) and (est_bytes <= eager_load_cap_bytes)

            if safe_by_frac and safe_by_cap:
                print("[INFO] Chunked dataset detected -> eager-loading (estimate fits safely in RAM)")
                ds = ds.load()
            else:
                # Keep it lazy: this preserves the value of chunking for genuinely large grids.
                if avail_bytes > 0:
                    print(
                        "[WARN] Chunked dataset detected -> keeping lazy (no ds.load). "
                        f"Estimated working-set ~{est_bytes/1024**3:.2f} GiB, "
                        f"avail ~{avail_bytes/1024**3:.2f} GiB."
                    )
                else:
                    print(
                        "[WARN] Chunked dataset detected -> keeping lazy (no ds.load). "
                        f"Estimated working-set ~{est_bytes/1024**3:.2f} GiB, "
                        "available RAM unknown."
                    )

        # ----------------------------------------------------------
        # Dimension information (sizes, not deprecated .dims)
        # ----------------------------------------------------------
        nx = ds.sizes.get("x", None)
        ny = ds.sizes.get("y", None)
        nz = ds.sizes.get("z", None)

        # Dimensionality label (2D if no z dim or z size <= 1)
        dimensionality = "3D" if (nz is not None and int(nz) > 1) else "2D"

        print(f"Grid size: nx={nx} ny={ny} nz={nz}")

        # Required variables for core geometry/metric validation
        required = ["R", "Z",
                    "g11", "g22", "g33",
                    "g12", "g13", "g23",
                    "J"]

        # Optional variables (nice-to-have)
        optional = ["surfvol"]

        print("\nChecking required variables:")
        for v in required:
            ok = (v in ds.variables)
            msg = f"Variable '{v}' exists"
            check(ok, msg)
            record_check(checks, f"required_var:{v}", bool(ok), severity="CRITICAL", details={"message": msg})

        print("\nChecking optional variables:")
        for v in optional:
            ok = (v in ds.variables)
            msg = f"Variable '{v}' exists (optional)"
            check(ok, msg)
            record_check(checks, f"optional_var:{v}", bool(ok), severity="INFO", details={"message": msg})

        # If the grid is missing core metric/J fields, treat as "partial geometry grid"
        missing_required = [v for v in required if v not in ds.variables]
        partial_grid = len(missing_required) > 0

        if partial_grid:
            print("\n  [WARN] Partial grid detected - missing required fields:")
            for v in missing_required:
                print(f"    - {v}")
            print("  [WARN] Will run geometry + optional B-field plots only; skipping metric/J/surfvol validation.")

        # ----------------------------------------------------------
        # Geometry Checks
        # ----------------------------------------------------------
        print("\n=== Geometry ===")

        # Defaults for geometry-derived scalars so later sections don't explode
        a_R = 0.0
        a_Z = 0.0
        kappa = 1.0
        aspect_ratio = 0.0
        R_at_midplane = None

        #Examine single toroidal slice (z=0) if present
        R = None
        Z = None
        R_avg = None
        R_mean = 0.0

        if ("R" in ds.variables) and ("Z" in ds.variables):
            R_da = ds["R"]
            Z_da = ds["Z"]

            #Only slice z if it exists
            if "z" in R_da.dims:
                R = R_da.isel(z=0)
            else:
                R = R_da

            if "z" in Z_da.dims:
                Z = Z_da.isel(z=0)
            else:
                Z = Z_da

            # Check geometric shape characteristics
            R_avg = R.mean(dim="y")
            R0_expected = to_scalar(ds.R0) if hasattr(ds, 'R0') else to_scalar(R_avg.mean())
            
            # Detect radial coordinate direction
            R_avg_diff = np.diff(R_avg)
            is_increasing = np.all(R_avg_diff > -1e-8)
            is_decreasing = np.all(R_avg_diff < 1e-8)
            
            if is_increasing:
                print("  [INFO] Radial coordinate: OUTWARD (x increases away from axis)")
                check(True, "<R>_theta increases with x (standard ordering)")
                record_check(checks, "geometry:radial_ordering", True, severity="INFO",
                             details={"ordering": "OUTWARD"})
            elif is_decreasing:
                print("  [INFO] Radial coordinate: INWARD (x increases toward axis)")
                check(True, "<R>_theta decreases with x (reversed ordering)")
                record_check(checks, "geometry:radial_ordering", True, severity="INFO",
                             details={"ordering": "INWARD"})
            else:
                print("  [WARN] Radial coordinate: NON-MONOTONIC")
                check(False, "<R>_theta monotonic ordering")
                record_check(checks, "geometry:radial_ordering", False, severity="WARN",
                             details={"ordering": "NON_MONOTONIC"})

                    
            # Check geometry shaping: circular vs elongated/triangular
            R_variation = to_scalar(R_avg.max() - R_avg.min())
            R_mean = to_scalar(R_avg.mean())
            
            if R_mean > 0:
                R_relative_variation = R_variation / R_mean
                
                # Calculate elongation properly: minor radius ratio, not extent ratio
                # For a tokamak: κ = (half-height in Z) / (half-width in R at midplane)
                Z_extent = to_scalar(Z.max() - Z.min())
                
                # Get R at outboard and inboard midplane (infer y index).
                #
                # IMPORTANT: theta can be multi-dimensional (theta(x,y) or with extra dims),
                # and taking theta_da.values directly can:
                #   - eagerly load a large array (memory hit),
                #   - produce a flat index from nanargmin() that is not a y-index.
                #
                # Use the shared helper so the logic matches visualization.
                theta_da = ds["theta"] if "theta" in ds else None
                y_mid_idx = infer_outboard_midplane_y_index(theta_da, R)

                # NOTE (heuristic):
                # We treat y as a poloidal index and infer the "outboard midplane" as the y location
                # where R is maximal on a representative surface (here x=nx//2). This works for typical
                # tokamak-like cross-sections but is not a formal definition for highly distorted grids,
                # rotated poloidal angle conventions, or non-tokamak geometries.            

                # Dask-safe: keep as DataArray slice; compute only scalar reductions.
                R_at_midplane_da = R.isel(y=y_mid_idx)
                R_outboard = float(to_scalar(R_at_midplane_da.max(skipna=True)))
                R_inboard = float(to_scalar(R_at_midplane_da.min(skipna=True)))

                # Keep a numpy-like alias name for downstream code that expects it to exist.
                # (Downstream uses `.mean()` later; DataArray supports that too.)
                R_at_midplane = R_at_midplane_da

                # Minor radius in R direction
                a_R = (R_outboard - R_inboard) / 2.0
                # Minor radius in Z direction (half the vertical extent)
                a_Z = Z_extent / 2.0
                
                if a_R > 1e-6:
                    kappa = a_Z / a_R
                    print(f"  [INFO] Elongation: κ = {kappa:.2f} (a_Z={a_Z:.3f}m, a_R={a_R:.3f}m)")
                else:
                    #Cylindrical limit: no radial extent
                    kappa = 1.0
                    print("  [INFO] Cylindrical geometry: a_R ≈ 0, assuming κ = 1.0")
                
                if R_relative_variation < 0.01:  #Less than 1% variation
                    print(f"  [INFO] Circular/cylindrical cross-section: <R>_theta variation < 1% (ΔR/R = {R_relative_variation:.4f})")
                else:
                    print(f"  [INFO] Shaped geometry: <R>_theta varies by {R_relative_variation:.1%} (shaping present)")
                    
                    #Additional shaping info
                    R_range = R_outboard - R_inboard
                    print(f"  [INFO] Major radius R0 ≈ {(R_outboard + R_inboard)/2:.3f}m, minor radius a ≈ {R_range/2:.3f}m")

            # R at outboard midplane should be monotonic with x, consistent with detected radial ordering
            R_outboard = R.isel(y=y_mid_idx)

            # NOTE:
            # This is a *local* sanity check at the inferred outboard midplane (single y index),
            # not a proof that R is monotonic everywhere on the poloidal plane.
            # It’s intended to catch swapped/garbled radial indexing and obvious coordinate issues.

            if is_increasing:
                ok = np.all(np.diff(R_outboard) > 0)
                msg = "R increases with x at outboard midplane (standard ordering)"
            elif is_decreasing:
                ok = np.all(np.diff(R_outboard) < 0)
                msg = "R decreases with x at outboard midplane (reversed ordering)"
            else:
                #If the global ordering wasn't monotonic, don't pretend this check is meaningful
                ok = False
                msg = "R monotonic with x at outboard midplane (global ordering non-monotonic)"

            check(ok, msg)

            record_check(checks, "geometry:R_outboard_monotonic_x", bool(ok), severity="WARN",
                 details={"message": msg})

            # NOTE (heuristic):
            # We check whether Z is roughly centered about 0 over the domain (mean(Z) ≈ 0),
            # which is typical for symmetric geometries referenced to the midplane (Z=0).
            # This can legitimately fail for vertically shifted plasmas or grids where Z=0
            # is not the geometric midplane.

            # Z mean should be near 0 for geometries symmetric about the midplane (Z=0).
            # (Not guaranteed for shifted/biased geometries; treat as a soft sanity check.)
            Z_scale = max(np.abs(to_scalar(Z.max())), np.abs(to_scalar(Z.min())))
            tol_Z = max(1e-6, 1e-4 * Z_scale) if Z_scale > 0 else 1e-6
            _z_ok = abs(to_scalar(Z.mean())) < tol_Z
            _z_msg = f"Z mean ≈ 0 (tol={tol_Z:.2e}, balanced poloidal extent)"
            check(_z_ok, _z_msg)
            record_check(checks, "geometry:Z_mean_near_zero", bool(_z_ok), severity="WARN",
                         details={"tol": float(tol_Z), "Z_mean": float(to_scalar(Z.mean())), "message": _z_msg})

            # g33 ≈ R^2 is only expected when the toroidal coordinate is the *geometric* toroidal angle (radians).
            # If z is scaled (e.g. length-like) or otherwise non-angle, this check is not valid and will false-fail.
            z_units = ""
            try:
                if "z" in ds.coords:
                    z_units = str(ds.coords["z"].attrs.get("units", "")).lower()
                elif "z" in ds.variables and getattr(ds["z"], "ndim", 0) == 1:
                    z_units = str(ds["z"].attrs.get("units", "")).lower()
            except Exception:
                z_units = ""

            do_g33_R2_check = any(u in z_units for u in ["rad", "radian", "radians"])

            if "g33" in ds.variables:
                g33_da = ds["g33"]
                if "z" in g33_da.dims:
                    g33 = g33_da.isel(z=0)
                else:
                    g33 = g33_da

                if do_g33_R2_check:
                    R_scale = to_scalar(R.mean())
                    tol_g33 = max(1e-6, 1e-6 * R_scale**2)

                    # Dask-safe: do NOT call np.allclose on DataArrays (forces full materialization).
                    # Use xarray elementwise comparison + reduction to a scalar bool.
                    _g33_close = xr.apply_ufunc(
                        np.isclose,
                        g33,
                        R**2,
                        kwargs={"rtol": 0.0, "atol": tol_g33},
                        dask="allowed",
                    )
                    _g33_ok = bool(to_scalar(_g33_close.all()))

                    _g33_msg = f"g33 ≈ R² (tol={tol_g33:.2e}, z units='{z_units or 'unknown'}')"
                    check(_g33_ok, _g33_msg)
                    record_check(checks, "geometry:g33_matches_R2", _g33_ok, severity="WARN",
                                 details={"tol": float(tol_g33), "z_units": (z_units or "unknown"), "message": _g33_msg})

                else:
                    # Dask-safe heuristic stats for g33/R²:
                    # - Do NOT use `.values` (forces eager load).
                    # - Use xarray quantile reductions -> scalar floats.
                    ratio_da = g33 / (R**2)

                    # quantile() returns a DataArray; reduce to Python scalars.
                    ratio_med = float(to_scalar(ratio_da.quantile(0.50, skipna=True)))
                    ratio_p05 = float(to_scalar(ratio_da.quantile(0.05, skipna=True)))
                    ratio_p95 = float(to_scalar(ratio_da.quantile(0.95, skipna=True)))

                    print(
                        f"  [WARN] g33 present - z units='{z_units or 'unknown'}' not radians; "
                        f"weak heuristic g33/R²: median={ratio_med:.3g}, p05={ratio_p05:.3g}, p95={ratio_p95:.3g}"
                    )
                    record_check(
                        checks,
                        "geometry:g33_matches_R2",
                        None,
                        severity="INFO",
                        details={
                            "reason": "z not radians",
                            "z_units": (z_units or "unknown"),
                            "ratio_median": ratio_med,
                            "ratio_p05": ratio_p05,
                            "ratio_p95": ratio_p95,
                        },
                    )
            else:
                print("  [INFO] g33 not present - skipping g33 ≈ R² geometry check")

        else:
            check(False, "Geometry checks skipped: missing 'R' and/or 'Z'")

        # Bail out early for partial grids (no metric/J/surfvol). Still do visualizations + HTML.
        if partial_grid:
            # ----------------------------------------------------------
            # Generate Visualizations (geometry + B-field if present)
            # ----------------------------------------------------------
            if make_plots:
                viz_path = _outpath("_render.png")
                if ("R" in ds.variables) and ("Z" in ds.variables):
                    render_grid_visualization(ds, viz_path)

                bfield_path = _outpath("_bfield.png")
                if ("R" in ds.variables) and ("Z" in ds.variables):
                    render_magnetic_field_visualization(ds, bfield_path)
            else:
                viz_path = None
                bfield_path = None

            # Minimal "verdict" data for report
            critical_failures = [f"Missing required field: {v}" for v in missing_required]
            warnings = []

            validation_results = {
                'nx': nx,
                'ny': ny,
                'nz': nz,
                'grid_type': "UNKNOWN",
                'grid_format': grid_format,
                'dimensionality': dimensionality,
                'config_type': "UNKNOWN",
                'kappa': kappa,
                'aspect_ratio': aspect_ratio,
                'critical_failures': critical_failures,
                'warnings': warnings,
                'checks': checks,

                # Record actual output image paths (may be None if --no-plots)
                'render_img': viz_path,
                'bfield_img': bfield_path,

                'metric_interpretation': None,
                'metric_det_relation': None,
                'metric_det_mean_rel_err': None,
                'metric_det_alt_mean_rel_err': None,
                'metric_det_max_rel_err': None,

                'g11_min': None, 'g11_max': None,
                'g22_min': None, 'g22_max': None,
                'g33_min': None, 'g33_max': None,
                'J_min': None, 'J_max': None,
                'surf_min': None, 'surf_max': None,
            }

            if "Bxy" in ds:
                # Dask-/xarray-safe: avoid `.values` which would eagerly load the whole 3D field.
                # We only need scalar reductions for validation, so compute min/max lazily.
                B_da = ds["Bxy"]

                # xarray reductions return DataArrays; `to_scalar()` converts to Python scalars
                # and computes if dask-backed.
                validation_results["B_min"] = float(to_scalar(B_da.min(skipna=True)))
                validation_results["B_max"] = float(to_scalar(B_da.max(skipna=True)))

            html_path = _outpath("_report.html")

            if make_html and (not json_only):
                generate_html_report(path, ds, validation_results, html_path)
            else:
                print("\n=== HTML Report ===")
                print("  (skipped)")

            write_json_report(html_path, validation_results)
            return 2

        # Metric components (dimension-order safe):
        # Keep these as xarray.DataArray so we preserve lazy/dask behavior.
        # Do NOT call `.values` here; validation only needs reductions (min/max/any/mean),
        # which xarray/dask can compute without loading the full 3D arrays into RAM.
        g11 = ds["g11"].transpose("x", "y", "z", missing_dims="ignore")
        g22 = ds["g22"].transpose("x", "y", "z", missing_dims="ignore")
        g33_full = ds["g33"].transpose("x", "y", "z", missing_dims="ignore")
        g12 = ds["g12"].transpose("x", "y", "z", missing_dims="ignore")
        g13 = ds["g13"].transpose("x", "y", "z", missing_dims="ignore")
        g23 = ds["g23"].transpose("x", "y", "z", missing_dims="ignore")

        # Default grid classification if topology checks are skipped
        grid_type = "UNKNOWN"

        # ----------------------------------------------------------
        # Grid Topology Classification
        # ----------------------------------------------------------
        print("\n=== Grid Topology ===")
        
        if R_avg is None:
            grid_type = "UNKNOWN"
            check(False, "Grid topology checks skipped: missing 'R'/'Z' geometry inputs")
        else:

            #Analyze radial structure
            dR_avg_dx = np.gradient(R_avg)
            dR_range = np.abs(dR_avg_dx.max() - dR_avg_dx.min())
            dR_std = np.std(dR_avg_dx)
            
            #Classify grid type based on radial structure
            uniform_threshold = 1e-4 * R_mean if R_mean > 0 else 1e-6
            
            if dR_range < uniform_threshold:
                grid_type = "UNIFORM_FLUX"
                print("  [INFO] Grid Type: UNIFORM FLUX-COORDINATE")
                print(f"  [INFO] Flux surfaces uniformly spaced (dR/dx variation = {dR_range:.2e})")
                print("  [INFO] Typical for: analytic equilibria, testing, idealized geometries")
            else:
                grid_type = "EQUILIBRIUM"
                print("  [INFO] Grid Type: EQUILIBRIUM WITH RADIAL STRUCTURE")
                print(f"  [INFO] Flux surfaces non-uniformly spaced (dR/dx variation = {dR_range:.2e})")
                print(f"  [INFO] Radial structure std: {dR_std:.2e}")
                print("  [INFO] Typical for: EFIT reconstructions, GS solutions, realistic equilibria")
                
                # Additional equilibrium characterization
                inner_compression = np.abs(dR_avg_dx[:nx//4].mean())
                outer_compression = np.abs(dR_avg_dx[-nx//4:].mean())
                
                if outer_compression > 2.0 * inner_compression:
                    print(f"  [INFO] Edge compression detected: outer spacing {outer_compression/inner_compression:.2f}x tighter")
                    print("  [INFO] Consistent with separatrix approach or steep pressure gradient")
            
        # Topology: Limiter vs Diverted
        ixseps1 = ds.attrs.get('ixseps1', -1)
        jyseps1_1 = ds.attrs.get('jyseps1_1', -1)
        jyseps2_1 = ds.attrs.get('jyseps2_1', -1)
        
        if ixseps1 > 0 and (jyseps1_1 > 0 or jyseps2_1 > 0):
            config_type = "DIVERTED"
            print("  [INFO] Configuration: DIVERTED (X-point present)")
            print(f"  [INFO] Separatrix at x-index: {ixseps1}")
            
            if jyseps1_1 > 0 and jyseps2_1 > 0 and jyseps2_1 != jyseps1_1:
                print("  [INFO] Double-null configuration detected")
            else:
                print("  [INFO] Single-null configuration")
        else:
            config_type = "LIMITER"
            print("  [INFO] Configuration: LIMITER (no X-point)")
            print("  [INFO] Plasma terminated by material surface")
        
        # Grid quality assessment
        print("  [INFO] Grid Quality Assessment:")
        
        # Check aspect ratio
        if R_mean > 0 and a_R > 0:
            aspect_ratio = R_mean / a_R
            print(f"    Aspect ratio A = R0/a = {aspect_ratio:.2f}")
            
            if aspect_ratio > 10:
                print("    [INFO] High aspect ratio - large-aspect-ratio approximations valid")
            elif aspect_ratio < 2:
                print("    [WARN] Low aspect ratio - may need full toroidal effects")
        
        # Check resolution quality
        if nx > 0 and a_R > 0:
            radial_resolution = a_R / nx
            print(f"    Radial resolution: Δr ≈ {radial_resolution*1000:.2f} mm/point")
            
            if radial_resolution > 0.1:
                print("    [WARN] Coarse radial resolution - consider increasing nx")
        
        if ny > 0:
            poloidal_resolution = 2*np.pi / ny
            print(f"    Poloidal resolution: Δθ ≈ {np.degrees(poloidal_resolution):.2f} deg/point")
            
            if ny < 32:
                print("    [WARN] Low poloidal resolution - aliasing possible")
        
        # Shaping parameter extraction (if shaped and geometry present)
        if (R is not None) and (Z is not None) and (R_at_midplane is not None) and (kappa > 1.1):
            print(f"    Elongation κ = {kappa:.2f} {'(moderately elongated)' if kappa < 1.5 else '(highly elongated)'}")
            
            # Estimate triangularity from R asymmetry at top vs bottom
            #
            # Dask-safe:
            # - Avoid Z.values / R.values (full materialization)
            # - Stack to 1D so we can argmax/argmin on a single dimension and only pull scalars.
            Z_stack = Z.stack(_all=Z.dims)
            R_stack = R.stack(_all=R.dims)

            Z_top_idx = int(to_scalar(Z_stack.argmax(dim="_all")))
            Z_bot_idx = int(to_scalar(Z_stack.argmin(dim="_all")))

            R_top = float(to_scalar(R_stack.isel(_all=Z_top_idx)))
            R_bot = float(to_scalar(R_stack.isel(_all=Z_bot_idx)))

            # R_at_midplane may now be a DataArray (from the midplane fix); mean() stays valid.
            R_mid_avg = float(to_scalar(R_at_midplane.mean(skipna=True)))

            
            # Simple triangularity estimate: shift of top/bottom relative to midplane
            delta_approx = (R_mid_avg - (R_top + R_bot)/2) / a_R if a_R > 1e-6 else 0.0
            
            if abs(delta_approx) > 0.05 and abs(delta_approx) < 2.0:
                print(f"    Triangularity δ ≈ {delta_approx:.2f} (crude geometric estimate)")
                print("    [INFO] Triangularity estimate is approximate - use equilibrium solver for precise value")

        #Coordinate system validation
        print("  [INFO] Coordinate System:")
        
        #Check if coordinates are field-aligned
        shift_name = None
        if "shiftAngle" in ds:
            shift_name = "shiftAngle"
        elif "zShift" in ds:
            shift_name = "zShift"

        if shift_name is not None:
            sa_da = ds[shift_name]

            try:
                # Dask-safe: slice first, then do xarray std reduction.
                sa2d = sa_da.isel(z=0) if "z" in sa_da.dims else sa_da

                # Original intent:
                #   np.std(sa_test, axis=0)  -> std over x for each y
                # Dask-safe equivalent:
                sa_variation_da = sa2d.std(dim="x", skipna=True)

                varies = bool(to_scalar((sa_variation_da > 0.1).any()))
                if varies:
                    print(f"    Field-aligned coordinates: YES ({shift_name} varies poloidally)")
                else:
                    print(f"    Field-aligned coordinates: NO ({shift_name} uniform)")

            except Exception as e:
                # Non-fatal: this is a qualitative "is it field-aligned-ish?" heuristic.
                # Never crash the diagnostics or block report generation because of this.
                print(f"    [WARN] Field-aligned coordinate heuristic failed ({shift_name}): {e}")

        # Check metric orthogonality
        #
        # NOTE:
        # xarray reductions (like .max(), .mean()) return xarray objects (often 0-D DataArray),
        # not plain Python floats. If you feed those into Python's max() or an `if` condition,
        # you can trigger:
        #   - "truth value of an array is ambiguous" errors, and/or
        #   - unexpected behaviour with dask-backed arrays.
        #
        # We already have `to_scalar()` in this script. Use it to force true Python floats
        # before doing Python control flow or max().
        g12_max = float(to_scalar(np.abs(g12).max(skipna=True)))
        g13_max = float(to_scalar(np.abs(g13).max(skipna=True)))
        g23_max = float(to_scalar(np.abs(g23).max(skipna=True)))
        g11_typical = float(to_scalar(np.abs(g11).mean(skipna=True)))

        worst_offdiag = max(g12_max, g13_max, g23_max)

        # Dimensionless orthogonality measure: worst off-diagonal relative to typical g11 magnitude
        orthogonality = (worst_offdiag / g11_typical) if (g11_typical > 0.0) else 0.0

        if orthogonality < 0.01:
            print("    Metric orthogonality: ORTHOGONAL (cross terms < 1%)")
        elif orthogonality < 0.1:
            print(f"    Metric orthogonality: NEARLY ORTHOGONAL (cross terms ≈ {orthogonality*100:.1f}%)")
        else:
            print(f"    Metric orthogonality: NON-ORTHOGONAL (cross terms ≈ {orthogonality*100:.1f}%)")
            print("    [INFO] Non-orthogonal coordinates typical for shaped flux-aligned grids")

        # ----------------------------------------------------------
        # Metric Tensor Validation
        # ----------------------------------------------------------
        print("\n=== Metric tensor ===")

        # Metric components already loaded earlier for topology check
        # Dask-/xarray-safe: perform reductions lazily and convert to scalars with `to_scalar()`.
        for g_name, g_arr in [("g11", g11), ("g22", g22), ("g33", g33_full),
                              ("g12", g12), ("g13", g13), ("g23", g23)]:

            # NaN check: `.isnull()` is xarray-native and dask-safe
            nan_any = bool(to_scalar(g_arr.isnull().any()))
            _nan_ok = not nan_any

            # Inf check: use apply_ufunc so it works for both numpy-backed and dask-backed arrays
            inf_any = bool(to_scalar(xr.apply_ufunc(np.isinf, g_arr, dask="allowed").any()))
            _inf_ok = not inf_any

            # Scalar range (skip NaNs)
            g_min = float(to_scalar(g_arr.min(skipna=True)))
            g_max = float(to_scalar(g_arr.max(skipna=True)))

            check(_nan_ok, f"{g_name} has no NaNs")
            record_check(
                checks,
                f"metric:{g_name}:no_nan",
                _nan_ok,
                severity="CRITICAL",
                details={"min": g_min, "max": g_max},
            )

            check(_inf_ok, f"{g_name} has no Infs")
            record_check(
                checks,
                f"metric:{g_name}:no_inf",
                _inf_ok,
                severity="CRITICAL",
                details={"min": g_min, "max": g_max},
            )

            print(f"  {g_name}: min={g_min:.4e} max={g_max:.4e}")

        #BOUT++ only stores one half of the symmetric metric tensor
        check(True, "Metric symmetry assumed OK (BOUT stores only one triangle)")

        # ----------------------------------------------------------
        # Jacobian
        # ----------------------------------------------------------
        print("\n=== Jacobian ===")

        # IMPORTANT (Dask-safe):
        #   Do NOT do: J = ds["J"].values
        #   That forces eager loading of the *entire* J array (often 3D) into RAM.
        #
        # Policy:
        #   - Keep J as an xarray DataArray (lazy if chunked)
        #   - Only compute cheap reductions (min/max, min/max sign, etc.) via to_scalar()
        J_da = ds["J"]

        # Backwards-compatibility alias:
        # A lot of the existing code references `J` later (e.g. metric determinant checks).
        # We keep `J` as a DataArray (NOT .values) so operations stay lazy/Dask-safe.
        J = J_da

        # Basic range reporting (lazy-safe reductions)
        J_min = float(to_scalar(J_da.min()))
        J_max = float(to_scalar(J_da.max()))

        # BOUT++ 5.x does NOT require J > 0 - only that the sign is consistent.
        #
        # NOTE ABOUT ZEROS:
        #   np.sign(0) == 0, and some grids can contain zeros (guard cells, degenerate points,
        #   partially-populated files). We ignore zeros when checking sign consistency.
        #
        # Dask-safe sign consistency check:
        #   - Mask out zeros => NaN
        #   - Take sign() of remaining values
        #   - If both -1 and +1 exist anywhere => sign flip => FAIL
        #
        # We avoid pulling the whole array into memory by reducing the sign field
        # down to scalars: min_sign and max_sign.
        nz_da = J_da.where(J_da != 0)

        # xr.apply_ufunc keeps the operation lazy on chunked arrays
        J_sign_da = xr.apply_ufunc(np.sign, nz_da, dask="allowed")

        # If everything was zero, nz_da becomes all-NaN. min/max then become NaN.
        min_sign = to_scalar(J_sign_da.min(skipna=True))
        max_sign = to_scalar(J_sign_da.max(skipna=True))

        if not np.isfinite(min_sign) or not np.isfinite(max_sign):
            # No non-zero entries => can't define a sign. Suspicious.
            ref_sign = 0.0
            consistent = False

            check(False, "Jacobian sign consistent across domain (no non-zero J to define a sign)")
            record_check(
                checks,
                "jacobian:sign_consistent",
                False,
                severity="CRITICAL",
                details={
                    "ref_sign": float(ref_sign),
                    "J_min": float(J_min),
                    "J_max": float(J_max),
                    "reason": "all J entries have sign==0 (J==0 everywhere)",
                },
            )
        else:
            min_sign_f = float(min_sign)
            max_sign_f = float(max_sign)

            # If min==max, the sign is consistent (either all +1 or all -1).
            consistent = bool(min_sign_f == max_sign_f)
            ref_sign = max_sign_f  # equals min_sign_f when consistent

            # Provide unique signs without materializing the full array:
            # - if consistent => [ref_sign]
            # - else => [-1, +1] (the only possible non-zero signs from np.sign)
            unique_nonzero_signs = [ref_sign] if consistent else [-1.0, 1.0]

            check(consistent, "Jacobian sign consistent across non-zero domain")
            record_check(
                checks,
                "jacobian:sign_consistent",
                consistent,
                severity="CRITICAL",
                details={
                    "ref_sign": float(ref_sign),
                    "J_min": float(J_min),
                    "J_max": float(J_max),
                    "unique_nonzero_signs": [float(s) for s in unique_nonzero_signs],
                },
            )

        print(f"  J: min={J_min:.4e} max={J_max:.4e}")
        print(f"  Jacobian sign = {ref_sign:+.0f}")

        # ----------------------------------------------------------
        # surfvol
        # ----------------------------------------------------------
        print("\n=== surfvol ===")
        surf = None

        # Robust variable discovery:
        # Some producers write SurfVol / SURFVOL / surfVol etc.
        # This block finds a case-insensitive match so we don't falsely report "missing".
        surfvol_name = None
        for v in ds.variables:
            if str(v).lower() == "surfvol":
                surfvol_name = v
                break

        # You can add aliases here if your ecosystem uses other names.
        # Keep it explicit and boring.
        if surfvol_name is None:
            aliases = ["surfacevol", "surf_vol", "surfVol"]  # optional: extend if you encounter real variants
            for a in aliases:
                for v in ds.variables:
                    if str(v).lower() == str(a).lower():
                        surfvol_name = v
                        break
                if surfvol_name is not None:
                    break

        if surfvol_name is None:
            print("  [INFO] surfvol missing (optional) - skipping surfvol checks")

            # Debug breadcrumb so you can prove to yourself what's in the file
            # without dumping the whole dataset.
            try:
                candidates = [v for v in ds.variables if "surf" in str(v).lower()]
                if candidates:
                    print(f"  [DEBUG] Variables containing 'surf': {candidates}")
            except Exception:
                pass

            # Optional-variable semantics:
            # - Missing optional data is not a failure of the grid.
            # - Use passed=None so record_check() emits status="SKIP"
            record_check(
                checks,
                "surfvol:present",
                None,  # None => SKIP (by design in record_check)
                severity="INFO",
                details={"message": "surfvol missing (optional)"},
            )

            # Dependent checks are also skipped
            record_check(checks, "surfvol:sign_consistent", None, severity="INFO",
                         details={"reason": "surfvol missing"})
            record_check(checks, "surfvol:ratio_finite", None, severity="INFO",
                         details={"reason": "surfvol missing"})

        else:
            print(f"  [INFO] Using surfvol variable: '{surfvol_name}'")

            # IMPORTANT (Dask-safe):
            # Keep as DataArray; do not .values the full thing.
            surf_da = ds[surfvol_name]

            # Range (lazy-safe reductions)
            surf_min = float(to_scalar(surf_da.min()))
            surf_max = float(to_scalar(surf_da.max()))

            # Sign consistency for surfvol:
            # - Ignore zeros (mask => NaN), then reduce min/max sign to scalars.
            surf_nz_da = surf_da.where(surf_da != 0)
            surf_sign_da = xr.apply_ufunc(np.sign, surf_nz_da, dask="allowed")

            surf_min_sign = to_scalar(surf_sign_da.min(skipna=True))
            surf_max_sign = to_scalar(surf_sign_da.max(skipna=True))

            if not np.isfinite(surf_min_sign) or not np.isfinite(surf_max_sign):
                _surf_ref_sign = 0.0
                _sign_ok = False
            else:
                _surf_ref_sign = float(surf_max_sign)  # equals min when consistent
                _sign_ok = bool(float(surf_min_sign) == float(surf_max_sign))

            check(_sign_ok, "surfvol sign consistent across domain")
            record_check(
                checks,
                "surfvol:sign_consistent",
                _sign_ok,
                severity="WARN",
                details={
                    "ref_sign": float(_surf_ref_sign),
                    "surf_min": float(surf_min),
                    "surf_max": float(surf_max),
                },
            )

            # If Jacobian sign is known, also check surfvol sign matches it
            _matches_J = (float(_surf_ref_sign) == float(ref_sign))
            check(_matches_J, "surfvol sign matches Jacobian sign")
            record_check(
                checks,
                "surfvol:sign_matches_J",
                _matches_J,
                severity="WARN",
                details={"surf_ref_sign": float(_surf_ref_sign), "J_ref_sign": float(ref_sign)},
            )

            # surfvol / J ratio (lazy-safe)
            # Use J as a DataArray alias (from your Jacobian section: J = J_da)
            ratio_da = surf_da / J

            _finite_ok = bool(to_scalar(np.isfinite(ratio_da).all()))
            check(_finite_ok, "surfvol/J finite")

            ratio_min = float(to_scalar(ratio_da.min(skipna=True)))
            ratio_max = float(to_scalar(ratio_da.max(skipna=True)))
            record_check(
                checks,
                "surfvol:ratio_finite",
                _finite_ok,
                severity="WARN",
                details={"ratio_min": float(ratio_min), "ratio_max": float(ratio_max)},
            )

            # Keep existing behavior: set surf so later summary/warnings logic works
            surf = surf_da

        # ----------------------------------------------------------
        # Magnetic field diagnostics (optional)
        # ----------------------------------------------------------
        B = None
        print("\n=== Magnetic field ===")
        if "Bxy" in ds:
            # IMPORTANT (Dask-safe):
            #   Do NOT do: ds["Bxy"].values
            #   That can materialize the entire 3D B-field in RAM and OOM big grids.
            #
            # Policy:
            #   - Keep Bxy as a DataArray
            #   - Only compute reductions (min/max) via to_scalar()
            B_da = ds["Bxy"]
            B = B_da  # Backwards-compatible alias: keep `B` but as DataArray (not numpy)

            B_min = float(to_scalar(B_da.min()))
            B_max = float(to_scalar(B_da.max()))

            _B_ok = bool(B_min > 0)
            check(_B_ok, "Bxy positive")
            record_check(
                checks,
                "Bxy positive",
                _B_ok,
                severity="CRITICAL",
                details={"B_min": float(B_min), "B_max": float(B_max)},
            )

            print(f"  Bxy: min={B_min:.4e} max={B_max:.4e}")
        else:
            print("No Bxy field found")

        # ----------------------------------------------------------
        # shiftAngle diagnostics (optional)
        # ----------------------------------------------------------
        print("\n=== Shift field (shiftAngle / zShift) ===")

        shift_name = None
        if "shiftAngle" in ds:
            shift_name = "shiftAngle"
        elif "zShift" in ds:
            shift_name = "zShift"

        if shift_name is not None:
            sa_da = ds[shift_name]

            try:
                # Dask-safe: keep as DataArray; slice z first (if present) BEFORE any conversion.
                sa2d = sa_da.isel(z=0) if "z" in sa_da.dims else sa_da

                print(f"  [INFO] Using shift field: {shift_name}")

                # NOTE:
                # We interpret "monotonic in y" as "monotonic along the poloidal index direction".
                #
                # Older / non-normalized grids sometimes do NOT use a literal dimension named "y"
                # (e.g. they may use "ny" or "theta"). If we call diff(dim="y") on such data,
                # xarray will throw: tuple.index(x): x not in tuple.
                #
                # Policy (still Dask-safe):
                # - Detect the poloidal dimension name from sa2d.dims
                # - If we can't identify it, skip this check cleanly (WARN) instead of crashing
                # - Keep the phase-unwrapping behavior for shiftAngle (same intent as before)

                # --- detect the poloidal dimension name for this shift field ---
                poloidal_dim = None
                for cand in ("y", "ny", "theta", "pol", "poloidal", "Y"):
                    if cand in sa2d.dims:
                        poloidal_dim = cand
                        break

                # Debug breadcrumb (small + useful): proves what dims we actually got
                print(f"  [DEBUG] {shift_name}.dims = {sa2d.dims}")

                if poloidal_dim is None:
                    # Can't safely assess monotonicity if we don't know which axis is poloidal.
                    # Treat as a non-fatal warning and keep going so report generation still happens.
                    print(f"  [WARN] Cannot find poloidal dim for {shift_name}; skipping monotonicity check.")
                    record_check(
                        checks,
                        f"shift:{shift_name}:monotonic_y",
                        None,  # None => SKIP per record_check() convention
                        severity="WARN",
                        details={"reason": "no recognizable poloidal dim", "dims": list(sa2d.dims)},
                    )
                else:
                    # Raw monotonicity along the poloidal dimension: d/d(poloidal_dim) >= -1e-8 everywhere
                    d_sa = sa2d.diff(dim=poloidal_dim)
                    shift_mono_raw = bool(to_scalar((d_sa > -1e-8).all()))

                    shift_mono_unwrapped = None
                    shift_wrap_artifact = False
                    shift_mono_used = shift_mono_raw

                    if shift_name == "shiftAngle":
                        # Phase wrapping at ±π can produce artificial sign flips in finite differences.
                        # Unwrap along the detected poloidal dimension without materializing the full array.
                        sa_unwrapped = xr.apply_ufunc(
                            np.unwrap,
                            sa2d,
                            input_core_dims=[[poloidal_dim]],
                            output_core_dims=[[poloidal_dim]],
                            kwargs={"axis": -1},
                            dask="allowed",
                        )

                        d_unwrapped = sa_unwrapped.diff(dim=poloidal_dim)
                        shift_mono_unwrapped = bool(to_scalar((d_unwrapped > -1e-8).all()))

                        shift_wrap_artifact = (not shift_mono_raw) and bool(shift_mono_unwrapped)
                        shift_mono_used = bool(shift_mono_unwrapped)

                    label = f"{shift_name} monotonic in {poloidal_dim}"
                    if shift_wrap_artifact:
                        label += " (after phase unwrap)"
                    check(shift_mono_used, label)
                    record_check(
                        checks,
                        f"shift:{shift_name}:monotonic_{poloidal_dim}",
                        bool(shift_mono_used),
                        severity="WARN",
                        details={
                            "wrap_artifact": bool(shift_wrap_artifact),
                            "label": label,
                            "dims": list(sa2d.dims),
                        },
                    )

            except Exception as e:
                # Non-fatal: shift checks are advisory; do not block report/plots generation.
                print(f"  [WARN] Shift field diagnostics failed for {shift_name}: {e}")
                record_check(
                    checks,
                    f"shift:{shift_name}:diagnostics",
                    None,
                    severity="WARN",
                    details={"reason": "exception", "error": str(e), "dims": list(getattr(sa_da, "dims", []))},
                )

        else:
            print("  [INFO] No shift field found (shiftAngle / zShift)")

            record_check(checks, "shift:present", None, severity="INFO",
                         details={"reason": "no shiftAngle/zShift"})

        # ----------------------------------------------------------
        # Metric tensor positivity (diagonal components)
        # ----------------------------------------------------------
        print("\n=== Metric positivity ===")
        for g in ["g11", "g22", "g33"]:
            da = ds[g]

            #Dask-safe reductions:
            # - min() and sum() compute without materializing the whole field in RAM at once.
            g_min = to_scalar(da.min())
            nonpos_count = int(to_scalar((da <= 0).sum()))

            _ok = bool(g_min > 0)
            _msg = f"{g} > 0 (positive definite)"

            check(_ok, _msg)
            record_check(
                checks,
                f"metric:{g}:positive",
                _ok,
                severity="CRITICAL",
                details={"count_nonpos": nonpos_count, "min": float(g_min), "message": _msg},
            )

            if not _ok:
                print(f"  [WARN] {g} has non-positive values at {nonpos_count} points")

        #Sylvester criterion (partial): check leading 2x2 principal minor > 0
        #This catches cases where diagonals are positive but the metric is still indefinite due to cross-terms.
        try:
            #Dask-safe: stay in xarray for the algebra and reduce at the end.
            g11_da = ds["g11"]
            g22_da = ds["g22"]
            g12_da = ds["g12"]

            minor2_da = g11_da * g22_da - g12_da**2

            _minor_ok = bool(to_scalar((minor2_da > 0).all()))
            _minor_bad = int(to_scalar((minor2_da <= 0).sum()))
            _minor_msg = "Leading 2x2 principal minor > 0 (g11*g22 - g12^2)"

            check(_minor_ok, _minor_msg)
            record_check(
                checks,
                "metric:leading_minor_2x2_positive",
                _minor_ok,
                severity="CRITICAL",
                details={"bad_points": _minor_bad, "message": _minor_msg},
            )

            if not _minor_ok:
                print(f"  [WARN] 2x2 principal minor non-positive at {_minor_bad} points (metric may be indefinite)")

        except Exception as e:
            print(f"  [INFO] Skipping 2x2 principal minor check (missing fields or shape issue): {e}")
            record_check(checks, "metric:leading_minor_2x2_positive", None, severity="INFO",
                 details={"reason": "exception", "error": str(e)})

        # ----------------------------------------------------------
        # Metric determinant consistency
        # ----------------------------------------------------------
        print("\n=== Metric determinant ===")
        # Metric components already loaded earlier
        #
        # Reality check:
        # Some grids provide g11..g23 as covariant (g_ij), others as contravariant (g^{ij}).
        # Different grid generators / toolchains vary, and the same variable names can be used
        # for either convention in the wild.
        #
        # We autodetect by comparing which relation matches the provided Jacobian J better:
        #   covariant:     det(g_ij)   ≈ J^2
        #   contravariant: det(g^{ij}) ≈ 1/J^2
        # NOTE ON "MUTANT" GRIDS (non-canonical conventions):
        #
        # This tool assumes BOUT++-style semantics:
        #   - The metric components (g11..g23) are intended to represent a coherent metric tensor
        #     in the grid's coordinate system (either g_ij or g^ij).
        #   - The Jacobian J is intended to be consistent with that choice via:
        #         det(g_ij)   ≈ J^2        (covariant)
        #         det(g^{ij}) ≈ 1 / J^2    (contravariant)
        #
        # If a grid mixes conventions (e.g. covariant metric with a Jacobian defined for the
        # contravariant convention, or a Jacobian with extra normalization factors), then BOTH
        # comparisons can look "wrong". In that case, failures here may reflect a convention mismatch
        # rather than numerical corruption.
        #
        # We *intentionally* do not try to auto-fix or reinterpret beyond the two standard relations:
        # silently guessing would make it easy to accept a self-inconsistent grid and produce bad physics.

        # Determinant of the 3x3 symmetric metric matrix (same algebra either way)
        # Keep as DataArray math (dask-safe); only reduce to scalars at the end.
        det_g_raw = (
            g11 * (g22 * g33_full - g23**2)
            - g12 * (g12 * g33_full - g13 * g23)
            + g13 * (g12 * g23 - g13 * g22)
        )

        # xarray supports numpy ufuncs; keep this lazy as well
        det_g_abs = xr.apply_ufunc(np.abs, det_g_raw, dask="allowed")

        # Use DataArray Jacobian (lazy-safe). `J` should already be a DataArray alias,
        # but fall back to dataset lookup if someone rearranges code order later.
        J_local = J if "J" in locals() else ds["J"]
        J2 = J_local**2

        J2_safe = J2 + 1e-30
        inv_J2 = 1.0 / J2_safe

        # Compute relative errors for both interpretations (still DataArrays)
        rel_err_cov = xr.apply_ufunc(np.abs, det_g_abs - J2, dask="allowed") / (J2 + 1e-10)
        rel_err_con = xr.apply_ufunc(np.abs, det_g_abs - inv_J2, dask="allowed") / (inv_J2 + 1e-10)

        # Reduce to scalars for comparison + printing
        mean_cov = float(to_scalar(rel_err_cov.mean()))
        mean_con = float(to_scalar(rel_err_con.mean()))

        # Pick the interpretation that matches better
        if mean_cov <= mean_con:
            det_g = det_g_raw
            det_target = J2
            det_label = "det(g_ij) ≈ J²"
            rel_error = rel_err_cov

            metric_interpretation = "COVARIANT"
            metric_det_relation = det_label
            metric_det_mean_rel_err = mean_cov
            metric_det_alt_mean_rel_err = mean_con

            print(
                f"  [INFO] Metric interpretation: COVARIANT (g_ij) selected "
                f"(mean rel err {mean_cov:.3e} vs {mean_con:.3e})"
            )
        else:
            det_g = det_g_raw
            det_target = inv_J2
            det_label = "det(g^{ij}) ≈ 1/J²"
            rel_error = rel_err_con

            metric_interpretation = "CONTRAVARIANT"
            metric_det_relation = det_label
            metric_det_mean_rel_err = float(mean_con)
            metric_det_alt_mean_rel_err = float(mean_cov)

            print(f"  [INFO] Metric interpretation: CONTRAVARIANT (g^{{ij}}) selected (mean rel err {mean_con:.3e} vs {mean_cov:.3e})")

        #If neither interpretation matches well, scream about it
        if min(mean_cov, mean_con) > 1e-2:
            print(f"  [WARN] det(g) does not match J² or 1/J² well (mean rel err cov={mean_cov:.3e}, con={mean_con:.3e})")
            print("  [WARN] Grid may be inconsistent/corrupt or using a nonstandard metric/J convention")

        # Dask-safe checks:
        # - Avoid np.allclose / np.all on DataArrays (forces full compute + materialization).
        # - Use xarray elementwise ops and reduce to scalar with to_scalar().
        _det_close = xr.apply_ufunc(
            np.isclose,
            xr.apply_ufunc(np.abs, det_g, dask="allowed"),
            det_target,
            kwargs={"rtol": 1e-3, "atol": 0.0},
            dask="allowed",
        )
        check(bool(to_scalar(_det_close.all())),
              f"{det_label} (metric consistency)")

        check(bool(to_scalar((det_g > 0).all())),
              "det(g) > 0 (metric positive definite)")

        # Dask-safe formatted stats (avoid formatting DataArray directly).
        _rel_mean = float(to_scalar(rel_error.mean(skipna=True)))
        _rel_max = float(to_scalar(rel_error.max(skipna=True)))
        print(f"  Relative error: mean={_rel_mean:.4e} max={_rel_max:.4e}")

        # ----------------------------------------------------------
        # Spatial localization: where are the worst det(g) vs J mismatches?
        # Keep it small + JSON-friendly.
        #
        # IMPORTANT (Dask-safe):
        # - Do NOT do `np.asarray(rel_error)` (forces full 2D/3D materialization).
        # - Instead compute small reductions + a few candidate slices.
        # ----------------------------------------------------------
        finite_mask_da = xr.apply_ufunc(np.isfinite, rel_error, dask="allowed")

        # "Bad" threshold uses the same policy knob as the final verdict max threshold
        bad_mask_da = finite_mask_da & (rel_error > det_relerr_max_max)

        # Dask-safe counts (reductions only)
        bad_count = int(to_scalar(bad_mask_da.sum()))
        total_count = int(to_scalar(finite_mask_da.sum()))
        bad_frac = (bad_count / total_count) if total_count > 0 else 0.0

        worst_points = []
        try:
            # Report a handful of worst offenders (by relative error)
            #
            # Dask-safe approach:
            # - If we have z, find candidate z-slices by max error per z (1D compute),
            #   then only load a few 2D slices to pick top offenders.
            # - If no z, we must load the 2D field (same as before, but unavoidable).
            N = 10

            # Prepare R/Z DataArrays for context decoration (same as your existing logic)
            R_da = None
            Z_da = None
            if "R" in ds and "Z" in ds:
                R_da = ds["R"].transpose("x", "y", "z", missing_dims="ignore")
                Z_da = ds["Z"].transpose("x", "y", "z", missing_dims="ignore")

            if "z" in rel_error.dims:
                # 1) Find candidate z slices via max over (x,y) -> 1D over z
                rel_xy = rel_error.stack(_xy=("x", "y"))
                max_per_z = rel_xy.max("_xy", skipna=True)

                # Small compute: size ~ nz
                max_per_z_np = np.asarray(max_per_z.compute())
                if max_per_z_np.size > 0 and np.isfinite(max_per_z_np).any():
                    n_candidates = int(min(max_per_z_np.size, max(N, 4)))
                    cand_z = np.argpartition(max_per_z_np, -n_candidates)[-n_candidates:]
                    cand_z = cand_z[np.argsort(max_per_z_np[cand_z])[::-1]]

                    candidates = []
                    for kz in cand_z:
                        # 2) Load only one 2D slice into memory
                        slice_da = rel_error.isel(z=int(kz))
                        slice_np = np.asarray(slice_da.compute())

                        if slice_np.size == 0 or not np.isfinite(slice_np).any():
                            continue

                        flat = slice_np.ravel()
                        k = min(N, flat.size)
                        idx_part = np.argpartition(flat, -k)[-k:]
                        idx_sorted = idx_part[np.argsort(flat[idx_part])[::-1]]

                        for idx in idx_sorted:
                            val = float(flat[int(idx)])
                            if not np.isfinite(val):
                                continue
                            i, j = np.unravel_index(int(idx), slice_np.shape)
                            candidates.append((val, int(i), int(j), int(kz)))

                    # 3) Global top-N across candidates
                    candidates.sort(key=lambda t: t[0], reverse=True)
                    candidates = candidates[:N]

                    for val, i, j, k0 in candidates:
                        entry = {
                            "index": (i, j, k0),
                            "rel_error": float(val),
                        }

                        # Add geometry context if R/Z exist (same behavior as before)
                        try:
                            if R_da is not None and Z_da is not None:
                                entry["R"] = float(to_scalar(R_da.isel(x=i, y=j, z=k0)))
                                entry["Z"] = float(to_scalar(Z_da.isel(x=i, y=j, z=k0)))
                        except Exception:
                            pass

                        worst_points.append(entry)
            else:
                # No z dimension: fallback to old behavior (must materialize the 2D field)
                rel_err_np = np.asarray(rel_error.compute())
                finite_mask = np.isfinite(rel_err_np)

                flat = np.where(finite_mask, rel_err_np, -np.inf).ravel()
                if flat.size > 0:
                    k = min(N, flat.size)
                    idx_part = np.argpartition(flat, -k)[-k:]
                    idx_sorted = idx_part[np.argsort(flat[idx_part])[::-1]]

                    for idx in idx_sorted:
                        if not np.isfinite(flat[idx]):
                            continue
                        ijk = np.unravel_index(int(idx), rel_err_np.shape)
                        entry = {
                            "index": tuple(int(v) for v in ijk),
                            "rel_error": float(flat[idx]),
                        }

                        try:
                            if R_da is not None and Z_da is not None:
                                i, j = ijk
                                entry["R"] = float(to_scalar(R_da.isel(x=i, y=j)))
                                entry["Z"] = float(to_scalar(Z_da.isel(x=i, y=j)))
                        except Exception:
                            pass

                        worst_points.append(entry)

        except Exception as e:
            worst_points = [{"error": str(e)}]

        if bad_count > 0:
            print(f"  [WARN] det(g) vs J mismatch above threshold at {bad_count}/{total_count} points ({bad_frac:.3%})")
            if worst_points:
                print("  [WARN] Worst points (index, rel_error):")
                for wp in worst_points[:5]:
                    if "index" in wp and "rel_error" in wp:
                        print(f"    {wp['index']}  {wp['rel_error']:.3e}")

        # Dask-safe: scalarize everything stored in report details.
        _det_close = xr.apply_ufunc(
            np.isclose,
            xr.apply_ufunc(np.abs, det_g, dask="allowed"),
            det_target,
            kwargs={"rtol": 1e-3, "atol": 0.0},
            dask="allowed",
        )
        _det_ok = bool(to_scalar(_det_close.all()))
        _mean_rel = float(to_scalar(rel_error.mean(skipna=True)))
        _max_rel = float(to_scalar(rel_error.max(skipna=True)))

        record_check(
            checks,
            f"{det_label} (metric consistency)",
            _det_ok,
            severity="CRITICAL",
            details={
                "mean_rel_error": _mean_rel,
                "max_rel_error": _max_rel,
                "bad_threshold": float(det_relerr_max_max),
                "bad_count": int(bad_count),
                "bad_fraction": float(bad_frac),
                "worst_points": worst_points,
            },
        )

        metric_det_max_rel_err = float(_max_rel)

        # Dask-safe positivity count
        if not bool(to_scalar((det_g > 0).all())):
            neg_count = int(to_scalar((det_g <= 0).sum()))
            print(f"  [WARN] det(g) has {neg_count} non-positive values - metric not positive definite!")

        # ----------------------------------------------------------
        # Boundary guard cell check
        # ----------------------------------------------------------
        print("\n=== Boundary regions ===")
        
        # Try to get actual boundary indices from grid metadata
        ixseps1 = ds.attrs.get('ixseps1', None)
        ixseps2 = ds.attrs.get('ixseps2', None)

        if R_avg is None:
            check(False, "Boundary region checks skipped: missing R_avg geometry input")
        elif ixseps1 is not None:
            print(f"  ixseps1 = {ixseps1}")
            if ixseps2 is not None:
                print(f"  ixseps2 = {ixseps2}")
            
            # Check for discontinuities in guard cells (MXG typically = 2)
            MXG = ds.attrs.get('MXG', 2)
            
            if MXG < nx // 4:  # Sanity check
                # Inner guard cells
                if ixseps1 >= MXG:
                    R_guard_inner = R_avg[:MXG]
                    R_core = R_avg[MXG:MXG+5]
                    grad_guard = np.abs(np.diff(R_guard_inner))
                    grad_core = np.abs(np.diff(R_core))
                    
                    if grad_core.mean() > 1e-10:
                        ratio_inner = grad_guard.max() / grad_core.mean()
                        check(ratio_inner < 3.0, f"Inner guard cells consistent (gradient ratio {ratio_inner:.2f})")
                        if ratio_inner >= 3.0:
                            print(f"  [WARN] Inner guard cells show {ratio_inner:.2f}x gradient spike")
                
                # Outer guard cells
                if nx - ixseps1 > MXG:
                    R_guard_outer = R_avg[-MXG:]
                    R_edge = R_avg[-MXG-5:-MXG]
                    grad_guard = np.abs(np.diff(R_guard_outer))
                    grad_edge = np.abs(np.diff(R_edge))
                    
                    if grad_edge.mean() > 1e-10:
                        ratio_outer = grad_guard.max() / grad_edge.mean()
                        check(ratio_outer < 3.0, f"Outer guard cells consistent (gradient ratio {ratio_outer:.2f})")
                        if ratio_outer >= 3.0:
                            print(f"  [WARN] Outer guard cells show {ratio_outer:.2f}x gradient spike")
        else:
            print("  [INFO] No ixseps1 metadata - skipping boundary topology check")
            print("  [INFO] Performing simple gradient consistency check...")
            
            if nx > 10:
                # Fallback: check for gradient anomalies at edges
                grad_R = np.abs(np.diff(R_avg))
                grad_median = np.median(grad_R[2:-2])
                
                if grad_median > 1e-10:
                    grad_inner = grad_R[:2].max()
                    grad_outer = grad_R[-2:].max()
                    
                    check(grad_inner < 5.0 * grad_median, 
                          f"Inner edge gradient reasonable ({grad_inner/grad_median:.2f}x median)")
                    check(grad_outer < 5.0 * grad_median,
                          f"Outer edge gradient reasonable ({grad_outer/grad_median:.2f}x median)")

        # ----------------------------------------------------------
        # B-field and geometry alignment
        # ----------------------------------------------------------
        print("\n=== B-field alignment ===")
        if R_avg is None:
            check(False, "B-field alignment checks skipped: missing R_avg geometry input")
        elif "Bxy" in ds and "Bpxy" in ds:

            # IMPORTANT (Dask-safe):
            #   Avoid .values on full 3D arrays. We only need positivity + a radial average.
            Bxy_da = ds["Bxy"]
            Bpxy_da = ds["Bpxy"]

            Bxy_min = float(to_scalar(Bxy_da.min()))
            Bpxy_min = float(to_scalar(Bpxy_da.min()))

            check(Bxy_min > 0 and Bpxy_min > 0,
                  "B-field components positive")
            
            # Analyze correlation between flux surface spacing and poloidal field
            dR_dx = np.gradient(R_avg)
            
            # Use absolute variation, not just std, for threshold
            dR_dx_range = np.abs(dR_dx.max() - dR_dx.min())
            
            # Threshold: gradient variation > 0.1% of mean R over the domain
            # This catches shaped plasmas even with modest shaping
            threshold = 1e-4 * R_mean if R_mean > 0 else 1e-6
            
            if dR_dx_range > threshold:  # Significant radial structure
                # Compute a radial profile of poloidal field magnitude.
                # Dask-safe approach:
                #   - reduce over y and z dims if present (xarray handles dims cleanly)
                #   - then convert the resulting 1D profile to numpy for corrcoef
                if "z" in Bpxy_da.dims:
                    Bp_profile_da = np.abs(Bpxy_da).mean(dim=("y", "z"))
                else:
                    Bp_profile_da = np.abs(Bpxy_da).mean(dim=("y",))

                Bp_avg = _as_numpy(Bp_profile_da)

                Bp_std = np.std(Bp_avg[5:-5]) if nx > 10 else np.std(Bp_avg)
                
                if Bp_std > 1e-10 * Bp_avg.mean():
                    # Use absolute value of gradient for correlation
                    dR_abs = np.abs(dR_dx[5:-5]) if nx > 10 else np.abs(dR_dx)
                    Bp_corr = Bp_avg[5:-5] if nx > 10 else Bp_avg
                    
                    if len(dR_abs) == len(Bp_corr) and len(dR_abs) > 3:
                        correlation = np.corrcoef(dR_abs, Bp_corr)[0, 1]
                        print(f"  |dR/dx| vs Bp correlation: {correlation:.3f}")
                        
                        # Shaped plasmas: Bp typically increases with flux surface spacing
                        if correlation > 0.3:
                            print("  [INFO] Poloidal field increases with flux surface spacing (typical for shaped plasma)")
                        elif correlation < -0.3:
                            print("  [WARN] Negative correlation - unusual field configuration")
                        else:
                            print("  [INFO] Weak correlation - Bp relatively uniform")
                else:
                    print("  [INFO] Bp uniform across domain (cylindrical field configuration)")
            else:
                print(f"  [INFO] No radial flux surface variation (dR/dx range={dR_dx_range:.2e} < {threshold:.2e})")
                print("  [INFO] Skipping Bp correlation analysis")

        # ----------------------------------------------------------
        # ShiftAngle range validation
        # ----------------------------------------------------------
        print("\n=== ShiftAngle range ===")
        if "shiftAngle" in ds:
            # Dask-safe: compute scalar reductions directly from DataArray
            sa_da = ds["shiftAngle"]
            sa_min = float(to_scalar(sa_da.min(skipna=True)))
            sa_max = float(to_scalar(sa_da.max(skipna=True)))
            sa_range = sa_max - sa_min

            print(f"  shiftAngle: min={sa_min:.4f} max={sa_max:.4f} range={sa_range:.4f}")

            # Typical range should be ~2π for one poloidal transit
            check(sa_range < 10.0, "shiftAngle range reasonable (< 10 rad)")

            if sa_range > 2*np.pi + 0.5:
                print(f"  [WARN] shiftAngle range ({sa_range:.2f}) exceeds 2π - check for wrapping issues")

        # ----------------------------------------------------------
        # Coordinate singularity check
        # ----------------------------------------------------------
        print("\n=== Singularity check ===")
        
        #Check Jacobian near boundaries
        #
        # IMPORTANT (Dask-safe):
        #   Avoid NumPy slicing on a materialized J array.
        #   Work directly with J_da and use reductions (min) that are safe on chunked data.
        #
        # Policy (same as original intent):
        # - "Boundary" = first/last 2 radial indices (x dimension)
        # - "Bulk" = exclude 5 cells on each side if nx > 10, else use full domain

        # Define boundary slice in x: x in [0,1] U [nx-2,nx-1]
        # This keeps semantics similar to your original "J[:2] and J[-2:]" intent.
        x_size = int(J_da.sizes.get("x", J_da.shape[0]))

        # Guard: if x dimension is smaller than 2 (pathological), clamp safely
        n_edge = 2 if x_size >= 2 else x_size

        J_boundary_da = xr.concat(
            [J_da.isel(x=slice(0, n_edge)), J_da.isel(x=slice(x_size - n_edge, x_size))],
            dim="x",
        )

        # Backwards-compatibility alias:
        # Existing code below this section expects `J_boundary` to exist.
        # Keep it as a DataArray (lazy / Dask-safe) - do NOT convert to .values.
        J_boundary = J_boundary_da

        _jb_min = float(to_scalar(np.abs(J_boundary_da).min()))
        _jb_ok = bool(_jb_min > jacobian_singularity_min)
        _jb_msg = "No Jacobian singularities at boundaries"
        check(_jb_ok, _jb_msg)
        record_check(
            checks,
            "jacobian:boundary_min_abs",
            _jb_ok,
            severity="CRITICAL",
            details={
                "min_abs": float(_jb_min),
                "threshold": float(jacobian_singularity_min),
                "message": _jb_msg,
            },
        )

        # Bulk check (if you have one later / want parity with old behavior)
        if x_size > 10:
            J_bulk_da = J_da.isel(x=slice(5, x_size - 5))
        else:
            J_bulk_da = J_da

        # Keep the same "bulk exists" safety, but now it’s robust for xarray shapes
        try:
            _bulk_min = float(to_scalar(np.abs(J_bulk_da).min()))
        except Exception:
            _bulk_min = float(to_scalar(np.abs(J_da).min()))

        #Check for metric blowup
        metric_components = [g11, g22, g33_full]

        # ---------------------------------------------------------------------
        # Metric conditioning ("blowup") check
        #
        # Goal (same intent as original):
        #   condition_ratio = max(metric) / min(metric)    where min(metric) must be > 0
        #
        # Fixes:
        #   - Define and populate `_pos_mins` (list of strictly-positive minima).
        #   - Define `_g` by iterating through `metric_components`.
        #   - Dask-/xarray-safe: convert reductions to Python floats with `to_scalar()`
        #     BEFORE using Python's max()/min() or control-flow comparisons.
        # ---------------------------------------------------------------------

        # Dask-safe maximum over all metric components:
        # Convert each component's scalar max to float, then take Python max.
        max_metric = max(float(to_scalar(g.max(skipna=True))) for g in metric_components)

        # Collect strictly-positive minima only (conditioning requires positive denominator)
        _pos_mins = []
        for _g in metric_components:
            # NOTE (xarray/dask safety):
            # `_g.min()` returns a 0-D DataArray (possibly dask-backed), not a Python scalar.
            # `to_scalar()` forces computation of that reduction (cheap) and returns a real scalar.
            _gmin = float(to_scalar(_g.min(skipna=True)))
            if _gmin > 0.0:
                _pos_mins.append(_gmin)

        if len(_pos_mins) == 0:
            # No strictly-positive minima means we can't define a meaningful conditioning ratio.
            # This usually coincides with earlier metric-positivity checks failing; here we
            # avoid crashing and record what happened.
            min_metric = 0.0
            print("  [WARN] Metric conditioning skipped: no strictly-positive metric minima (min <= 0 present)")

            record_check(
                checks,
                "metric:conditioning_ok",
                None,  # None => SKIP semantics in your reporting pattern
                severity="INFO",
                details={
                    "reason": "non-positive metric component(s) prevent conditioning ratio",
                    "max_metric": float(max_metric),
                },
            )
        else:
            min_metric = min(_pos_mins)

            # Now it's safe and meaningful to compute the conditioning ratio.
            metric_ratio = max_metric / min_metric
            print(f"  Metric condition number: {metric_ratio:.2e}")
            check(metric_ratio < metric_cond_warn, "Metric components well-conditioned")
            record_check(
                checks,
                "metric:conditioning_ok",
                bool(metric_ratio < metric_cond_warn),
                severity="WARN",
                details={"ratio": float(metric_ratio), "threshold": float(metric_cond_warn)},
            )

        # ----------------------------------------------------------
        # Generate Visualizations
        # ----------------------------------------------------------
        if make_plots:
            viz_path = _outpath("_render.png")
            if ("R" in ds.variables) and ("Z" in ds.variables):
                render_grid_visualization(ds, viz_path)
            else:
                print("\n=== Generating Grid Visualization ===")
                print("  ✗ Missing R/Z - skipping geometry visualization")

            # Generate B-field visualization if data exists
            bfield_path = _outpath("_bfield.png")
            if ("R" in ds.variables) and ("Z" in ds.variables):
                render_magnetic_field_visualization(ds, bfield_path)
            else:
                print("\n=== Generating Magnetic Field Visualization ===")
                print("  ✗ Missing R/Z - skipping B-field visualization")
        else:
            viz_path = None
            bfield_path = None

        # ----------------------------------------------------------
        # Final Summary
        # ----------------------------------------------------------
        print("\n" + "="*60)
        print("=== FINAL VERDICT ===")
        print("="*60)
        
        #Collect all critical failures
        critical_failures = []
        warnings = []
        
        # Check for critical issues
        # Reuse the earlier Jacobian sign-consistency result (which ignores zeros safely).
        if not consistent:
            critical_failures.append("Jacobian sign inconsistent")

        # Metric determinant consistency already computed earlier (det_g, rel_error)
        # Treat a bad consistency match as critical (grid is internally inconsistent)
        if 'rel_error' in locals():
            # NOTE (xarray/dask safety):
            # rel_error is an xarray.DataArray (possibly dask-backed). Reductions like `.mean()` / `.max()`
            # return 0-D DataArrays, not Python floats. Comparing those directly in an `if` triggers:
            #   ValueError: The truth value of an array is ambiguous
            #
            # Consistent with the rest of this script: reduce lazily with xarray, then convert to
            # a true Python scalar via `to_scalar()` before doing Python control flow.
            _rel_mean_verdict = float(to_scalar(rel_error.mean(skipna=True)))
            _rel_max_verdict  = float(to_scalar(rel_error.max(skipna=True)))

            if (_rel_mean_verdict > det_relerr_mean_max) and (_rel_max_verdict > det_relerr_max_max):
                critical_failures.append("Metric determinant inconsistent with Jacobian")
            
        # Dask-safe final verdict checks:
        # - Avoid np.any/np.all/np.isnan/np.isinf on xarray objects (can trigger full loads).
        # - Use xarray-native boolean reductions and to_scalar() to get Python bools.

        if not bool(to_scalar((det_g > 0).all())):
            critical_failures.append("Metric determinant non-positive")

        diag_nonpos = (
            bool(to_scalar((g11 <= 0).any()))
            or bool(to_scalar((g22 <= 0).any()))
            or bool(to_scalar((g33_full <= 0).any()))
        )
        if diag_nonpos:
            critical_failures.append("Metric diagonal non-positive")

        J_bad = bool(to_scalar(J.isnull().any())) or bool(to_scalar(xr.apply_ufunc(np.isinf, J, dask="allowed").any()))
        if J_bad:
            critical_failures.append("Jacobian contains NaN/Inf")

        metric_bad = False
        for a in (g11, g22, g33_full, g12, g13, g23):
            # NaN
            if bool(to_scalar(a.isnull().any())):
                metric_bad = True
                break
            # Inf
            if bool(to_scalar(xr.apply_ufunc(np.isinf, a, dask="allowed").any())):
                metric_bad = True
                break

        if metric_bad:
            critical_failures.append("Metric contains NaN/Inf")
        
        # Dask-safe Bxy positivity check (avoid ds["Bxy"].values)
        if "Bxy" in ds:
            _B_min_verdict = float(to_scalar(ds["Bxy"].min()))
            if _B_min_verdict <= 0:
                critical_failures.append("Magnetic field non-positive")
        
        # Jacobian boundary singularity:
        # IMPORTANT:
        #   J_boundary is an xarray.DataArray (lazy / possibly dask-backed), not a numpy array.
        #   In the singularity check section above, we already reduced the boundary minimum
        #   to a real Python float: `_jb_min`.
        #
        #   Reuse that scalar here so we don't:
        #     - trigger accidental full-array loads
        #     - get ambiguous truth-value errors from xarray objects
        #
        # NOTE:
        #   `_jb_min` is only defined if the "=== Singularity check ===" ran.
        #   Guard with `locals()` to keep behavior robust if code paths change.
        if "_jb_min" in locals():
            if float(_jb_min) <= jacobian_singularity_min:
                critical_failures.append("Jacobian singularity at boundaries")
        else:
            # If for some reason boundary min wasn't computed, be conservative and
            # don't invent a failure here. (You already record boundary checks above.)
            pass

        #shiftAngle / zShift warnings (non-critical)
        #These get reported in the final verdict + HTML report.
        try:
            if "shift_name" in locals() and shift_name is not None:
                if "shift_wrap_artifact" in locals() and shift_wrap_artifact:
                    warnings.append(f"{shift_name} appears wrapped in y (monotonic after phase unwrap)")
                elif "shift_mono_used" in locals() and (not shift_mono_used):
                    if shift_name == "shiftAngle":
                        warnings.append(f"{shift_name} not monotonic in y (even after phase unwrap)")
                    else:
                        warnings.append(f"{shift_name} not monotonic in y")
        except Exception as e:
            print(f"  [INFO] Could not evaluate shift field warning state: {e}")
        
        # Check for warnings
        if min_metric > 0:
            metric_ratio = max_metric / min_metric
            if metric_ratio >= metric_cond_warn:
                warnings.append(f"Poor metric conditioning (ratio={metric_ratio:.2e})")

        # surfvol is optional, but missing it should still warn (cannot verify surfvol/J consistency)
        if "surfvol" not in ds.variables or surf is None:
            warnings.append("surfvol missing (optional) - surfvol/J consistency check skipped")

        if aspect_ratio < aspect_ratio_warn_min:
            warnings.append(f"Low aspect ratio (A={aspect_ratio:.2f})")
        
        if ny < ny_warn_min:
            warnings.append(f"Low poloidal resolution (ny={ny})")

        if a_R > 0 and a_R / nx > 0.1:
            warnings.append(f"Coarse radial resolution ({a_R/nx*1000:.1f} mm/point)")
        
        # Print summary
        if len(critical_failures) == 0:
            print("Status: ✓ GRID FILE VALID")
            print("\nThis grid file passes all critical validation checks and is")
            print("suitable for BOUT++ 5.x simulations.")
            
            print(f"\nGrid Type: {grid_type}")
            print(f"Configuration: {config_type}")
            print(f"Dimensions: nx={nx}, ny={ny}, nz={nz}")
            print(f"Elongation: κ={kappa:.2f}")
            print(f"Aspect Ratio: A={aspect_ratio:.2f}")
            
            if len(warnings) > 0:
                print(f"\n⚠ Warnings ({len(warnings)}):")
                for w in warnings:
                    print(f"  • {w}")
                print("\nWarnings indicate potential issues but do not prevent usage.")
            else:
                print("\n✓ No warnings - grid quality is good")
            
        else:
            print("Status: ✗ GRID FILE INVALID")
            print(f"\n{len(critical_failures)} critical failure(s) detected:")
            for i, fail in enumerate(critical_failures, 1):
                print(f"  {i}. {fail}")
            print("\nThis grid CANNOT be used for BOUT++ simulations.")
            print("Fix the issues listed above before proceeding.")
        
        print("="*60)

        # ----------------------------------------------------------
        # Collect validation results and generate HTML report
        # ----------------------------------------------------------
        validation_results = {
            'nx': nx,
            'ny': ny,
            'nz': nz,
            'grid_type': grid_type,
            'grid_format': grid_format,
            'dimensionality': dimensionality,
            'config_type': config_type,
            'kappa': kappa,
            'aspect_ratio': aspect_ratio,
            'critical_failures': critical_failures,
            'warnings': warnings,
            'checks': checks,

            #record actual output image paths (may be None if --no-plots)
            'render_img': viz_path,
            'bfield_img': bfield_path,

            #Metric/J determinant interpretation (autodetect)
            'metric_interpretation': (metric_interpretation if 'metric_interpretation' in locals() else None),
            'metric_det_relation': (metric_det_relation if 'metric_det_relation' in locals() else None),
            'metric_det_mean_rel_err': (metric_det_mean_rel_err if 'metric_det_mean_rel_err' in locals() else None),
            'metric_det_alt_mean_rel_err': (metric_det_alt_mean_rel_err if 'metric_det_alt_mean_rel_err' in locals() else None),
            'metric_det_max_rel_err': (metric_det_max_rel_err if 'metric_det_max_rel_err' in locals() else None),

            # IMPORTANT (Dask-/xarray-safe):
            # xarray reductions like .min()/.max() return a 0-D DataArray, not a Python float.
            # The HTML report formats these with "{val:.4e}" which will crash if val isn't a float.
            # Use `to_scalar()` (already defined in this script) to:
            #   - compute lazily if dask-backed
            #   - return a true Python scalar
            #
            # Also use skipna=True so NaNs don't poison reductions.
            'g11_min': float(to_scalar(g11.min(skipna=True))),
            'g11_max': float(to_scalar(g11.max(skipna=True))),
            'g22_min': float(to_scalar(g22.min(skipna=True))),
            'g22_max': float(to_scalar(g22.max(skipna=True))),
            'g33_min': float(to_scalar(g33_full.min(skipna=True))),
            'g33_max': float(to_scalar(g33_full.max(skipna=True))),
            'J_min': float(to_scalar(J.min(skipna=True))),
            'J_max': float(to_scalar(J.max(skipna=True))),

            # surfvol may be missing
            'surf_min': (float(to_scalar(surf.min(skipna=True))) if surf is not None else None),
            'surf_max': (float(to_scalar(surf.max(skipna=True))) if surf is not None else None),
        }


        if B is not None:
            # IMPORTANT (Dask-/xarray-safe):
            # `B` is kept as a DataArray alias (not `.values`) to avoid loading big 3D fields.
            # Therefore reductions still return 0-D DataArrays unless we scalarize.
            # Convert to Python floats so HTML formatting "{val:.4e}" works reliably.
            validation_results['B_min'] = float(to_scalar(B.min(skipna=True)))
            validation_results['B_max'] = float(to_scalar(B.max(skipna=True)))

        
        # ----------------------------------------------------------
        # Reports: HTML and/or JSON
        # JSON path is derived from the HTML base name for consistency.
        # ----------------------------------------------------------
        html_path = _outpath("_report.html")

        if make_html and (not json_only):
            generate_html_report(path, ds, validation_results, html_path)
        else:
            print("\n=== HTML Report ===")
            print("  (skipped)")

        # Always allow JSON unless someone explicitly disables it (we don't add that flag here).
        write_json_report(html_path, validation_results)


        # Exit code contract:
        # 0 = valid, 1 = invalid (critical failures), 2 = partial grid (handled earlier)
        if strict and (len(warnings) > 0) and (len(critical_failures) == 0):
            # Strict mode: warnings are treated as failures
            return 1

        return 0 if len(critical_failures) == 0 else 1

def get_version(pkg_name: str) -> str:
    try:
        return version(pkg_name)
    except PackageNotFoundError:
        return __version__

# Back-compat wrapper (CLI still calls main(...))
def main(path, outdir=None, make_plots=True, make_html=True, json_only=False, strict=False, det_relerr_mean_max=1e-3, det_relerr_max_max=1e-2,
    metric_cond_warn=1e6, jacobian_singularity_min=1e-10, aspect_ratio_warn_min=2.0, ny_warn_min=32, force_chunk=False, chunks=None):

    return DiagnosticsRunner.main(
        path,
        outdir=outdir,
        make_plots=make_plots,
        make_html=make_html,
        json_only=json_only,
        strict=strict,
        det_relerr_mean_max=det_relerr_mean_max,
        det_relerr_max_max=det_relerr_max_max,
        metric_cond_warn=metric_cond_warn,
        jacobian_singularity_min=jacobian_singularity_min,
        aspect_ratio_warn_min=aspect_ratio_warn_min,
        ny_warn_min=ny_warn_min,

        # Chunking controls
        force_chunk=force_chunk,
        chunks=chunks,
    )
      
# --------------------------------------------------------------
# Script / CLI entrypoint
# --------------------------------------------------------------
def main_cli(argv=None) -> int:
    """
    CLI wrapper used by:
      1) direct execution: python bout_tokamak_grid_diagnostics.py 
      2) console script entrypoint (pip install): bout-grid-diagnostics  

    This keeps the current HPC-friendly behavior (single-file, argparse, sys.exit codes),
    while making the tool pip-installable via a [project.scripts] entrypoint.
    """
    parser = argparse.ArgumentParser(
        prog="bout_tokamak_grid_diagnostics.py",
        description="BOUT++ 4.x/5.x grid diagnostics: metrics/Jacobian/geometry sanity checks + HTML/JSON reports."
    )

    # Keep it scientist-simple:
    # - If they pass a path: use it.
    # - If they pass nothing: try a sensible default ("grid.nc") in the current directory.
    parser.add_argument(
        "grid",
        nargs="?",
        default="grid.nc",
        help="Path to BOUT++ grid netCDF file. Default: grid.nc (in current directory)."
    )

    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s " + get_version("bout-tokamak-grid-diagnostics"),
    )

    # Output control
    parser.add_argument(
        "--outdir",
        default=None,
        help="Output directory for reports/PNGs. Default: same directory as the grid file."
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable PNG visualization outputs (geometry/B-field)."
    )
    parser.add_argument(
        "--no-html",
        action="store_true",
        help="Disable HTML report generation (JSON can still be written)."
    )
    parser.add_argument(
        "--json-only",
        action="store_true",
        help="Write JSON report only (implies --no-html and --no-plots)."
    )

    # Verdict policy
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as failures (exit code 1 if any warnings)."
    )

    # -------------------------
    # Threshold knobs (defaults match current hard-coded behavior)
    # -------------------------
    # Metric determinant consistency: existing logic marks CRITICAL if:
    #   rel_error.mean() > 1e-3 AND rel_error.max() > 1e-2
    parser.add_argument(
        "--det-relerr-mean-max",
        type=float,
        default=1e-3,
        help="CRITICAL if mean(det(g) vs J consistency relative error) exceeds this AND max exceeds --det-relerr-max-max. Default: 1e-3"
    )
    parser.add_argument(
        "--det-relerr-max-max",
        type=float,
        default=1e-2,
        help="CRITICAL if max(det(g) vs J consistency relative error) exceeds this AND mean exceeds --det-relerr-mean-max. Default: 1e-2"
    )

    # Poor metric conditioning warning: existing logic warns if ratio >= 1e6
    parser.add_argument(
        "--metric-cond-warn",
        type=float,
        default=1e6,
        help="WARNING if (max_metric / min_metric) >= this. Default: 1e6"
    )

    # Jacobian boundary singularity critical: existing logic critical if min(|J_boundary|) <= 1e-10
    parser.add_argument(
        "--jacobian-singularity-min",
        type=float,
        default=1e-10,
        help="CRITICAL if min(|J| at boundaries) <= this. Default: 1e-10"
    )

    # Low aspect ratio warning: existing logic warns if A < 2
    parser.add_argument(
        "--aspect-ratio-warn-min",
        type=float,
        default=2.0,
        help="WARNING if aspect ratio A < this. Default: 2.0"
    )

    # Low poloidal resolution warning: existing logic warns if ny < 32
    parser.add_argument(
        "--ny-warn-min",
        type=int,
        default=32,
        help="WARNING if ny < this. Default: 32"
    )

    parser.add_argument(
        "--force-chunk",
        action="store_true",
        help=(
            "Force dask chunked open regardless of the internal memory heuristic. "
            "Useful for very large grids or unknown auxiliary fields that can undercount the estimate."
        )
    )
    parser.add_argument(
        "--chunks",
        default=None,
        help=(
            "Explicit chunk sizes as a comma-separated list, e.g. 'x=256,y=256,z=1'. "
            "Only dimensions present in the dataset will be used. "
            "If omitted, the tool uses its internal default chunk sizes when chunking is enabled."
        )
    )   

    # Logging controls
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only show warnings/errors."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose (DEBUG) output."
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Write log output to this file."
    )

    # Parse argv if provided (console_scripts will call us this way),
    # otherwise argparse will use sys.argv as usual.
    args = parser.parse_args(argv)

    # Configure logging before any meaningful output
    setup_logging(verbose=args.verbose, quiet=args.quiet, log_file=args.log_file)

    # json-only implies no-html and no-plots (consistent, simple behavior)
    if args.json_only:
        args.no_html = True
        args.no_plots = True

    # Keep the "no args" experience friendly:
    # If the default grid.nc isn't there, don't pretend things are fine.
    if args.grid == "grid.nc" and (not os.path.exists(args.grid)):
        print("Usage: bout_tokamak_grid_diagnostics.py grid.nc")
        print("  (No arguments provided and default 'grid.nc' not found in current directory.)")
        return 1

    # Parse chunk spec once here so downstream code stays clean.
    # If the user types something malformed, fail fast with a clear message.
    try:
        user_chunks = parse_chunks_spec(args.chunks)
    except Exception as e:
        print(f"[ERROR] {e}")
        return 1

    # ----------------------------------------------------------
    # Run main diagnostics with CLI-friendly error handling.
    #
    # Rationale:
    # - This is a CLI tool; by default we want "clean fail" messages
    #   rather than Python tracebacks for common user errors like:
    #     * missing file
    #     * wrong extension / not a NetCDF file
    #
    # Notes:
    # - We still return non-zero exit codes for automation/HPC scripts.
    # - Verbose/debug tracebacks can be added later if desired, but
    #   we keep behavior consistent with the rest of this script's
    #   scientist-friendly print style.
    # ----------------------------------------------------------
    try:
        return main(
            args.grid,
            outdir=args.outdir,
            make_plots=(not args.no_plots),
            make_html=(not args.no_html),
            json_only=args.json_only,
            strict=args.strict,
            det_relerr_mean_max=args.det_relerr_mean_max,
            det_relerr_max_max=args.det_relerr_max_max,
            metric_cond_warn=args.metric_cond_warn,
            jacobian_singularity_min=args.jacobian_singularity_min,
            aspect_ratio_warn_min=args.aspect_ratio_warn_min,
            ny_warn_min=args.ny_warn_min,

            # Chunking controls
            force_chunk=bool(args.force_chunk),
            chunks=user_chunks,
        )

    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        return 1

    except IsADirectoryError as e:
        print(f"[ERROR] {e}")
        return 1

    except ValueError as e:
        # xarray often raises ValueError for "can't guess engine" /
        # "no backend match". Turn that into a clearer CLI message.
        msg = str(e)
        if "did not find a match in any of xarray's currently installed IO backends" in msg:
            print(
                "[ERROR] Could not open grid file as NetCDF with the installed xarray backends.\n"
                f"        Path: {args.grid}\n"
                "        Possible causes:\n"
                "          - File is not a NetCDF file (wrong format)\n"
                "          - File extension is unusual and engine can't be guessed\n"
                "          - Missing optional backend dependency for this file type\n"
                "        Tip: try renaming to .nc if it's NetCDF, or install the needed backend."
            )
            return 1

        # Unknown ValueError: still keep it clean, but show the message.
        print(f"[ERROR] {e}")
        return 1

if __name__ == "__main__":
    raise SystemExit(main_cli())