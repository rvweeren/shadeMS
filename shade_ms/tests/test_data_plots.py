from types import SimpleNamespace

import dask.dataframe as dd
import datashader
import matplotlib.figure
from matplotlib.scale import SymmetricalLogTransform
import numpy as np
import pandas as pd
from PIL import Image
import pytest

from shade_ms.__main__ import cli
from shade_ms import data_plots


def test_axis_scale_defaults():
    parser, _ = cli()
    options = parser.parse_args(["test.ms"])
    assert (options.xscale, options.yscale, options.linthresh) == (
        "linear", "linear", 1.0)


@pytest.mark.parametrize("value", [
    "0", "-1", "nan", "inf", "-inf", "not-a-number",
])
def test_invalid_linthresh(value, capsys):
    parser, _ = cli()
    with pytest.raises(SystemExit) as error:
        parser.parse_args(["test.ms", f"--linthresh={value}"])
    assert error.value.code == 2
    assert "--linthresh" in capsys.readouterr().err


@pytest.mark.parametrize("option", ["--xscale", "--yscale"])
def test_invalid_scale(option):
    parser, _ = cli()
    with pytest.raises(SystemExit) as error:
        parser.parse_args(["test.ms", option, "log"])
    assert error.value.code == 2


@pytest.mark.parametrize("scales", [
    ("linear", "linear"), ("symlog", "linear"),
    ("linear", "symlog"), ("symlog", "symlog"),
])
@pytest.mark.parametrize("mode", [
    "density", "alpha", "category", "continuous",
])
def test_render_axis_scales(scales, mode, tmp_path, monkeypatch):
    parser, _ = cli()
    options = parser.parse_args([
        "test.ms", "--xscale", scales[0], "--yscale", scales[1],
        "--linthresh", "10", "--xcanvas", "64", "--ycanvas", "48",
    ])
    values = np.array([-1000., -100., -10., -5., 0., 5., 10., 100., 1000.])
    frame = pd.DataFrame({"u": values, "v": values[::-1],
                          "category": np.arange(9) % 2, "colour": values})
    ddf = dd.from_pandas(frame, npartitions=2)

    def datum(label, minmax, discrete=False):
        return SimpleNamespace(
            label=label, minmax=minmax, is_discrete=discrete,
            subset_indices=None, discretized_labels=None,
            subset_remapper=None, nlevels=2, columns=())

    # Exercise auto limits and caching in original units.
    xdatum = datum("u", (None, None))
    ydatum = datum("v", (None, None))
    adatum = xdatum if mode == "alpha" else None
    if mode == "category":
        cdatum = datum("category", (0, 1), discrete=True)
    elif mode == "continuous":
        cdatum = datum("colour", (-1000, 1000))
    else:
        cdatum = None
    rasters = []
    figures = []
    points = datashader.Canvas.points
    savefig = matplotlib.figure.Figure.savefig

    def record_points(canvas, source, x, y, **kwargs):
        raster = points(canvas, source, x, y, **kwargs)
        rasters.append((canvas, source.compute(), raster))
        return raster

    def record_savefig(figure, *args, **kwargs):
        figures.append(figure)
        return savefig(figure, *args, **kwargs)

    monkeypatch.setattr(datashader.Canvas, "points", record_points)
    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", record_savefig)
    png = tmp_path / "coverage.png"
    cache = {}
    colors = ["#0000ff", "#ff0000"]
    assert data_plots.create_plot(
        ddf, {}, xdatum, ydatum, adatum, "mean", cdatum,
        cmap=colors, bmap=colors, dmap=colors, normalize="linear",
        xlabel="u (wavelengths)", ylabel="v (wavelengths)",
        title="uv coverage", pngname=str(png), options=options,
        minmax_cache=cache,
        extra_markup=[("axvline", [5], {"color": "black"})],
    ) == str(png)
    with Image.open(png) as image:
        assert image.format == "PNG"
        assert image.width > 0 and image.height > 0

    assert cache == {"u": (-1000, 1000), "v": (-1000, 1000)}
    canvas, rendered_frame, raster = rasters[0]
    np.testing.assert_array_equal(rendered_frame["u"], frame["u"])
    np.testing.assert_array_equal(rendered_frame["v"], frame["v"])
    ax = figures[0].axes[0]
    assert (ax.get_xscale(), ax.get_yscale()) == scales
    np.testing.assert_array_equal(ax.lines[0].get_xdata(), [5, 5])

    transformed = []
    edges = []
    for dimension, scale, size, bounds in zip(
            ("x", "y"), scales, (64, 48), (canvas.x_range, canvas.y_range)):
        original = frame["u" if dimension == "x" else "v"].to_numpy()
        transform = (
            SymmetricalLogTransform(10, 10, 1) if scale == "symlog" else None)
        mapped = (
            transform.transform(original) if transform is not None
            else original)
        transformed.append(mapped)
        expected_bounds = (
            transform.transform([-1000, 1000]) if transform is not None
            else [-1000, 1000])
        np.testing.assert_allclose(bounds, expected_bounds)
        axis_edges = np.linspace(*bounds, size + 1)
        edges.append(
            transform.inverted().transform(axis_edges) if transform is not None
            else axis_edges)
        if transform is not None:
            actual_transform = getattr(ax, f"{dimension}axis").get_transform()
            np.testing.assert_allclose(
                actual_transform.transform(original), mapped)
            delta = (bounds[1] - bounds[0]) / 100
            np.testing.assert_allclose(
                actual_transform.transform(
                    getattr(ax, f"get_{dimension}lim")()),
                [bounds[0] - delta, bounds[1] + delta])

    expected, _, _ = np.histogram2d(
        transformed[1], transformed[0], bins=(48, 64),
        range=[canvas.y_range, canvas.x_range])
    if mode == "alpha":
        sums, _, _ = np.histogram2d(
            transformed[1], transformed[0], bins=(48, 64),
            range=[canvas.y_range, canvas.x_range], weights=values)
        means = np.full_like(sums, np.nan)
        np.divide(sums, expected, out=means, where=expected > 0)
        np.testing.assert_allclose(raster.data, means)
    else:
        counts = raster.data.sum(axis=2) if cdatum is not None else raster.data
        np.testing.assert_array_equal(counts, expected)

    if "symlog" in scales:
        mesh = ax.collections[0]
        coordinates = mesh.get_coordinates()
        np.testing.assert_allclose(coordinates[0, :, 0], edges[0])
        np.testing.assert_allclose(coordinates[:, 0, 1], edges[1])
        assert not ax.images
    else:
        assert len(ax.images) == 1
        np.testing.assert_allclose(
            ax.images[0].get_extent(), [-1000, 1000, -1000, 1000])
