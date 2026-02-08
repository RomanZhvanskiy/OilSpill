# grid_builder.py
"""
Stretched grid generator for the oil–wave model.

We parameterise the grid by:
    - domain bounds: Xmin, Xmax, Ymin, Ymax
    - refinement centre: x0, y0 (injection point)
    - smallest cell sizes at (x0, y0): dx_min, dy_min
    - refinement ratios in x and y directions:
        grid_refinement_ratio_x, grid_refinement_ratio_y

The user-specified grid_refinement_ratio_* is interpreted as the maximum
local relative difference between two successive cell sizes:

    R = max( |dx1 - dx2| / dx1, |dx1 - dx2| / dx2 )

For a geometric progression dx_{i+1} = r * dx_i with r > 1, this reduces to:
    R = r - 1

So internally we use:
    r_init_x = 1 + grid_refinement_ratio_x
    r_init_y = 1 + grid_refinement_ratio_y

For each axis, the algorithm is:

  1. For a given geometric factor r, build a one-sided sequence of cell
     widths starting from dx_min and factor r, until adding the next
     geometric cell would overshoot the distance from the refinement
     centre to the boundary. Then add a final "remainder" cell so the
     total hits the boundary exactly. This gives an effective last-cell
     ratio r_eff = dx_last / dx_prev.

  2. Perform a TWO-STAGE search for r in [1+eps, r_init]:

        Stage 1 (coarse):
          - Start at r_init and step DOWN by 0.000001 in r.
          - For each r, compute r_eff and the relative mismatch
                err = |r_eff - r| / r.
          - As soon as err <= tol_coarse (e.g. 0.1), accept this r as the
            coarse candidate and stop the coarse search.
          - If nothing meets tol_coarse, keep the r that gives the smallest
            err as the coarse fallback.

        Stage 2 (fine):
          - Start from the coarse candidate and step DOWN by 0.0000001.
          - Use a tighter tolerance tol_fine (e.g. 0.1%) to accept the
            first r that satisfies err <= tol_fine.
          - If nothing meets tol_fine, use the r with the smallest err
            encountered in the fine stage.

     This ensures we always prefer larger r among acceptable candidates,
     and never exceed the user-specified maximum refinement ratio.

  3. Use the resulting r (possibly different on + and - sides) to construct
     coordinates from the centre to each bound, combine them into a single
     axis, and then set the first and last coordinate exactly to Xmin/Xmax.

This gives:
  - a grid that satisfies the refinement constraint r <= 1 + R,
  - smoothly graded spacing,
  - a last cell whose effective ratio is close to the nominal geometric
    factor (O(0.1%) mismatch, controlled by tol_fine).
"""

from __future__ import annotations
from typing import Tuple

import numpy as np


class StretchedGridBuilder:
    def __init__(
        self,
        x0: float,
        y0: float,
        Xmin: float,
        Xmax: float,
        Ymin: float,
        Ymax: float,
        dx_min: float,
        dy_min: float,
        grid_refinement_ratio_x: float,
        grid_refinement_ratio_y: float,
    ):
        """
        Parameters
        ----------
        x0, y0 : float
            Refinement centre (injection point). Must lie strictly inside
            the domain: Xmin < x0 < Xmax, Ymin < y0 < Ymax.

        Xmin, Xmax, Ymin, Ymax : float
            Domain bounds.

        dx_min, dy_min : float
            Smallest cell sizes at (x0, y0).

        grid_refinement_ratio_x, grid_refinement_ratio_y : float
            User's refinement parameter R in each direction, corresponding
            to a geometric factor r_init = 1 + R. Must be > 0.
        """
        if not (Xmin < x0 < Xmax):
            raise ValueError(f"x0={x0} must lie strictly inside [{Xmin}, {Xmax}]")
        if not (Ymin < y0 < Ymax):
            raise ValueError(f"y0={y0} must lie strictly inside [{Ymin}, {Ymax}]")

        if grid_refinement_ratio_x <= 0.0:
            raise ValueError("grid_refinement_ratio_x must be > 0.")
        if grid_refinement_ratio_y <= 0.0:
            raise ValueError("grid_refinement_ratio_y must be > 0.")

        self.x0 = x0
        self.y0 = y0
        self.Xmin = Xmin
        self.Xmax = Xmax
        self.Ymin = Ymin
        self.Ymax = Ymax
        self.dx_min = dx_min
        self.dy_min = dy_min
        self.Rx = grid_refinement_ratio_x
        self.Ry = grid_refinement_ratio_y

        # internal geometric factors
        self.r_x_init = 1.0 + self.Rx
        self.r_y_init = 1.0 + self.Ry

    # ---------- public API ----------

    def build(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build and return the stretched x and y coordinate arrays.

        Returns
        -------
        x, y : np.ndarray
            1D coordinate arrays for x and y.
        """
        x = self._build_axis(
            centre=self.x0,
            Xmin=self.Xmin,
            Xmax=self.Xmax,
            dx_min=self.dx_min,
            r_init=self.r_x_init,
        )
        y = self._build_axis(
            centre=self.y0,
            Xmin=self.Ymin,
            Xmax=self.Ymax,
            dx_min=self.dy_min,
            r_init=self.r_y_init,
        )
        return x, y

    # ---------- internal helpers ----------

    @staticmethod
    def _build_axis_one_side(L: float, dx_min: float, r: float):
        """
        Build one-sided geometric cell widths from dx_min out to length L.

        Starting with dx_0 = dx_min, then dx_{k+1} = r * dx_k for k >= 0,
        append cells until adding the next cell would exceed L. The final
        cell is then taken as a "remainder" so that the sum of widths
        exactly equals L.

        Returns
        -------
        widths : list[float]
            List of all cell widths including the remainder cell.
        r_eff : float
            Effective ratio dx_last / dx_prev for the last cell (if there
            is only one cell, dx_prev is taken as dx_min).
        """
        widths = []
        dx = dx_min
        total = 0.0
        prev_dx = None

        # geometric cells that fit fully
        while total + dx < L and dx > 0.0:
            widths.append(dx)
            total += dx
            prev_dx = dx
            dx *= r

        # remainder cell
        remainder = L - total
        if remainder <= 0.0:
            # degenerate: domain smaller than dx_min
            widths = [L]
            prev_dx = dx_min
        else:
            widths.append(remainder)
            if prev_dx is None:
                prev_dx = dx_min

        r_eff = widths[-1] / prev_dx if prev_dx > 0.0 else 1.0
        return widths, r_eff

    @classmethod
    def _search_ratio_for_side(
        cls,
        L: float,
        dx_min: float,
        r_init: float,
        coarse_step: float = 0.000001,
        fine_step: float = 0.0000001,
        tol_coarse: float = 0.1,
        tol_fine: float = 1e-3,
    ):
        """
        Two-stage search for geometric factor r on one side:

        Stage 1 (coarse):
            - Start from r_init and go DOWN in steps of `coarse_step` (e.g. 0.01).
            - For each r, build widths with _build_axis_one_side().
            - Compute:
                  err = |r_eff - r| / r
              where r_eff = dx_last / dx_prev (effective last-cell ratio).
            - As soon as err <= tol_coarse (e.g. 0.5%), ACCEPT this r as
              coarse candidate and proceed to fine stage.

        Stage 2 (fine):
            - Starting from r_coarse (the accepted coarse candidate) and
              going DOWN in steps of `fine_step` (e.g. 0.001), repeat the
              same evaluation with a tighter tolerance tol_fine (e.g. 0.1%).
            - As soon as err <= tol_fine, ACCEPT this r and return.

        Fallbacks:
            - If no r in coarse stage meets tol_coarse, we pick the r in
              [1+eps, r_init] with the smallest err and use that as r_coarse.
            - If no r in fine stage meets tol_fine, we return the best
              (smallest-err) r encountered during fine stage (or r_coarse
              itself if nothing better).

        This implements:
            "start from peak refinement ratio and go down in small steps
             until domain size / last-cell ratio is very close",
        while always preferring larger r among acceptable candidates.
        """
        eps = 1e-6
        if L <= 0.0:
            return 1.0 + eps, []

        def eval_r(r):
            widths, r_eff = cls._build_axis_one_side(L, dx_min, r)
            err = abs(r_eff - r) / r if r > 0.0 else 0.0
            return err, widths

        # ---------- Stage 1: coarse search ----------
        best_r_coarse = r_init
        best_err_coarse = float("inf")
        best_widths_coarse = []

        r = r_init
        r_min = 1.0 + eps

        while r >= r_min:
            err, widths = eval_r(r)

            # track best (for fallback)
            if err < best_err_coarse:
                best_err_coarse = err
                best_r_coarse = r
                best_widths_coarse = widths

            # EARLY ACCEPT coarse candidate
            if err <= tol_coarse:
                # use this as coarse base for fine search
                break

            r -= coarse_step

        # if loop finished without triggering break, r is now < r_min
        # so we restore best coarse r/widths
        if r < r_min:
            r_coarse = best_r_coarse
            widths_coarse = best_widths_coarse
        else:
            r_coarse = r
            widths_coarse = widths  # from last eval_r(r) inside loop

        # ---------- Stage 2: fine search around r_coarse ----------
        # We search from r_coarse downward in finer steps.
        best_r_fine = r_coarse
        best_err_fine = float("inf")
        best_widths_fine = widths_coarse

        r = r_coarse
        while r >= r_min:
            err, widths = eval_r(r)

            # track best in fine stage
            if err < best_err_fine:
                best_err_fine = err
                best_r_fine = r
                best_widths_fine = widths

            # EARLY ACCEPT fine candidate
            if err <= tol_fine:
                return r, widths

            r -= fine_step

        # If no fine candidate met the tight tolerance, return best in fine stage
        return best_r_fine, best_widths_fine



    @classmethod
    def _build_axis(
        cls,
        centre: float,
        Xmin: float,
        Xmax: float,
        dx_min: float,
        r_init: float,
    ) -> np.ndarray:
        """
        Build a 1D stretched axis for either x or y.

        Parameters
        ----------
        centre : float
            Refinement centre (x0 or y0).
        Xmin, Xmax : float
            Domain bounds.
        dx_min : float
            Smallest cell size at the centre.
        r_init : float
            Initial geometric factor (1 + grid_refinement_ratio).

        Returns
        -------
        axis : np.ndarray
            Monotonic coordinate array from Xmin to Xmax.
        """
        eps = 1e-6

        if not (Xmin < centre < Xmax):
            raise ValueError(f"centre={centre} must lie strictly inside [{Xmin}, {Xmax}]")

        # distances from centre to bounds
        L_pos = Xmax - centre
        L_neg = centre - Xmin

        # If distances are tiny relative to dx_min, just do two cells
        if L_pos <= dx_min * (1.0 + eps) and L_neg <= dx_min * (1.0 + eps):
            return np.array([Xmin, centre, Xmax], dtype=float)

        # positive side (centre -> Xmax)
        r_pos, dx_pos = cls._search_ratio_for_side(L_pos, dx_min, r_init)

        # negative side (centre -> Xmin)
        r_neg, dx_neg = cls._search_ratio_for_side(L_neg, dx_min, r_init)

        # build coordinates on positive side
        coords_pos = [centre]
        current = centre
        for dx in dx_pos:
            current += dx
            coords_pos.append(current)

        # build coordinates on negative side
        coords_neg = [centre]
        current = centre
        for dx in dx_neg:
            current -= dx
            coords_neg.append(current)

        # sort negative side, combine, replace ends with exact bounds
        coords_neg_sorted = sorted(coords_neg)

        # avoid duplicating centre
        axis = coords_neg_sorted[:-1] + coords_pos

        # enforce exact domain bounds
        axis[0] = Xmin
        axis[-1] = Xmax

        return np.array(axis, dtype=float)

# =========================
#  SELF-TEST / DEMO
# =========================

if __name__ == "__main__":
    """
    Test driver:

    - Domain:
        Xmin = -1000, Xmax = 1000
        Ymin = -1500, Ymax = 1500
        refinement centre at (0, 0)
        dx_min = dy_min = 1 m

    - Refinement ratios:
        R from 0.10 to 0.30 in steps of 0.01

    For each R:
      - Build grid
      - Compute achieved max local refinement ratio in x and y

    For R = 0.20:
      - Print detailed info and ALL grid nodes

    For other R:
      - Print only target and achieved ratios
    """

    Xmin, Xmax = -1000.0, 1000.0
    Ymin, Ymax = -1500.0, 1500.0
    x0, y0 = 0.0, 0.0
    dx_min = 1.0
    dy_min = 1.0

    def max_ratio_from_coords(coords: np.ndarray) -> float:
        d = np.diff(coords)
        if len(d) < 2:
            return 0.0
        r_loc = np.maximum(
            np.abs((d[1:] - d[:-1]) / d[1:]),
            np.abs((d[1:] - d[:-1]) / d[:-1]),
        )
        return float(r_loc.max())

    print("\n========== STRETCHED GRID REFINEMENT SWEEP ==========\n")

    # Sweep refinement ratios from 0.10 to 0.30 in steps of 0.01
    R_values = np.arange(0.10, 0.30 + 1e-9, 0.01)

    for R in R_values:
        builder = StretchedGridBuilder(
            x0=x0,
            y0=y0,
            Xmin=Xmin,
            Xmax=Xmax,
            Ymin=Ymin,
            Ymax=Ymax,
            dx_min=dx_min,
            dy_min=dy_min,
            grid_refinement_ratio_x=R,
            grid_refinement_ratio_y=R,
        )

        x, y = builder.build()
        R_x_ach = max_ratio_from_coords(x)
        R_y_ach = max_ratio_from_coords(y)

        if abs(R - 0.20) < 5e-4:
            # Detailed case for R = 0.20
            print("--------------------------------------------------")
            print(f"DETAILED CASE: target refinement ratio R = {R:.2f}")
            print(f"  nx = {len(x)}, ny = {len(y)}")
            print(f"  x in [{x[0]:.6f}, {x[-1]:.6f}]")
            print(f"  y in [{y[0]:.6f}, {y[-1]:.6f}]")
            print(f"  achieved max local refinement ratio in x ≈ {R_x_ach:.6f}")
            print(f"  achieved max local refinement ratio in y ≈ {R_y_ach:.6f}")

            print("\n  X nodes:")
            for i, xv in enumerate(x):
                print(f"    x[{i:4d}] = {xv:12.6f}")

            print("\n  Y nodes:")
            for j, yv in enumerate(y):
                print(f"    y[{j:4d}] = {yv:12.6f}")

            print("--------------------------------------------------\n")
        else:
            # Summary only
            print(
                f"R = {R:.2f}  |  "
                f"max ratio x ≈ {R_x_ach:.6f}, "
                f"max ratio y ≈ {R_y_ach:.6f}"
            )

    print("\nGrid refinement sweep completed.\n")
