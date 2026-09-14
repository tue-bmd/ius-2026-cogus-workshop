"""
Three additional zea ops for scan-line beamforming.

The beamformer in ``s5-1.yaml`` is an ordinary :class:`zea.Pipeline`::

    select_scanlines -> cast -> band_pass_filter -> apply_window -> demodulate
                     -> map_scanlines[ tof_correction -> delay_and_sum
                                       -> reshape_grid -> envelope_detect ]
                     -> stack_scanlines -> normalize -> log_compress
"""

from __future__ import annotations

from typing import ClassVar

from keras import ops
from zea.func.tensor import vmap
from zea.internal.core import DataTypes
from zea.internal.registry import ops_registry
from zea.ops import Map, Operation, Pipeline

#: What :class:`SelectScanlines` emits per line, and hence what
#: :class:`MapScanlines` maps over. Anything not listed -- ``probe_geometry``, say --
#: is the same for every line and flows into the mapped pipeline unchanged.
SCANLINE_KEYS = (
    "data",
    "grid",
    "flatgrid",
    "t0_delays",
    "tx_apodizations",
    "focus_distances",
    "polar_angles",
    "initial_times",
    "t_peak",
    "transmit_origins",
)


@ops_registry("select_scanlines")
class SelectScanlines(Operation):
    """Reduce a frame to the queried scan lines, one transmit each.

    Stacks the per-line inputs along a leading ``n_lines`` axis, keeping a length-1
    transmit axis so each line is an ordinary single-transmit problem for the ops
    inside :class:`MapScanlines`.
    """

    ADD_OUTPUT_KEYS: ClassVar[list[str]] = list(SCANLINE_KEYS)

    def __init__(self, **kwargs):
        super().__init__(
            input_data_type=DataTypes.RAW_DATA,
            output_data_type=DataTypes.RAW_DATA,
            **kwargs,
        )

    def call(
        self,
        line_indices,
        grid,
        t0_delays,
        tx_apodizations,
        transmit_origins,
        focus_distances,
        polar_angles,
        initial_times,
        t_peak,
        **kwargs,
    ):
        """
        Args:
            line_indices: ``(n_lines,)`` image columns to reconstruct. Column ``n``
                is beamformed from transmit ``n``, so the pipeline must be prepared
                on a scan-line grid (``parameters.enable_scanline``).
            grid: ``(n_z, n_tx, 3)`` pixel positions of the full frame.
        """
        data = kwargs[self.key]  # (n_tx, n_ax, n_el, n_ch)
        idx = ops.cast(line_indices, "int32")
        per_tx = lambda arr: ops.take(arr, idx, axis=0)[:, None]

        # Each line's grid column on its own leading axis: (n_lines, n_z, 1, 3).
        line_grid = ops.transpose(ops.take(grid, idx, axis=1), (1, 0, 2))[:, :, None]

        return {
            "data": per_tx(data),  # (n_lines, 1, n_ax, n_el, n_ch)
            "grid": line_grid,
            "flatgrid": line_grid[:, :, 0],
            "t0_delays": per_tx(t0_delays),
            "tx_apodizations": per_tx(tx_apodizations),
            "transmit_origins": per_tx(transmit_origins),
            "focus_distances": per_tx(focus_distances),
            "polar_angles": per_tx(polar_angles),
            "initial_times": per_tx(initial_times),
            "t_peak": per_tx(t_peak),
        }


@ops_registry("map_scanlines")
class MapScanlines(Map):
    """Run the enclosed operations on one scan line at a time (a ``vmap``).

    :meth:`call_item` is :meth:`zea.ops.Map.call_item` with one flag flipped: ``Map``
    passes ``fn_supports_batch=True`` because it exists to *chunk* an axis its
    operations already handle (``PatchedGrid`` splitting the pixel axis). Here the
    mapped axis is a new one, so the enclosed pipeline has to be vectorized over it.

    Args:
        batch_size: Lines per chunk. ``None`` maps every line in one ``vmap``; set it
            to bound peak memory on a whole frame.
    """

    def __init__(self, operations, batch_size=None, **kwargs):
        kwargs.pop("argnames", None)
        super().__init__(
            operations,
            argnames=list(SCANLINE_KEYS),
            batch_size=batch_size,
            # One chunk = map every line at once. Passing it explicitly also keeps
            # Map from warning that neither chunks nor batch_size was set.
            chunks=None if batch_size else 1,
            **kwargs,
        )

    def call_item(self, **inputs):
        mapped_args = [inputs.pop(name, None) for name in self.argnames]

        def one_line(*args):
            outputs = Pipeline.call(self, **dict(zip(self.argnames, args)), **inputs)
            return outputs[self.output_key]

        return vmap(
            one_line,
            in_axes=self.in_axes,
            out_axes=self.out_axes,
            chunks=self.chunks,
            batch_size=self.batch_size,
            disable_jit=not bool(self.jit_options) and not self._inside_outer_jit,
        )(*mapped_args)

    def get_dict(self, compact=True):
        config = super().get_dict(compact=compact)
        config["name"] = "map_scanlines"
        params = config.get("params", {})
        params.pop("argnames", None)
        params.pop("chunks", None)
        return config


@ops_registry("stack_scanlines")
class StackScanlines(Operation):
    """``(n_lines, n_z, 1, ...)`` -> ``(n_z, n_lines, ...)``.

    The ``1`` is each line's own single-column grid width; dropping it and swapping
    the mapped axis into its place gives the usual image layout.
    """

    def call(self, **kwargs):
        data = kwargs[self.key]
        return {self.output_key: ops.swapaxes(ops.squeeze(data, axis=2), 0, 1)}
