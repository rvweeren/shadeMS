from types import SimpleNamespace

import dask.dataframe as dd
import datashader
import matplotlib.figure
from matplotlib.scale import SymmetricalLogTransform
import numpy as np
import pandas as pd
from PIL import Image
import pytest

from shade_ms.__main__ import cli, parse_plot_spec
from shade_ms import data_plots


def _datum(label, minmax=(None, None), discrete=False, unit="wavelengths"):
    return SimpleNamespace(
        label=label, fullname=label, minmax=minmax, is_discrete=discrete,
        mapper=SimpleNamespace(unit=unit), subset_indices=None,
        discretized_labels=None, subset_remapper=None, nlevels=2, columns=())


def _render(frame, tmp_path, monkeypatch, scale=None, linthresh=10,
            mode="density", limits=(None, None)):
    parser, _ = cli()
    args = [
        "test.ms", "-x", "u", "-y", "v", "--polar",
        "--linthresh", str(linthresh), "--xcanvas", "72", "--ycanvas", "64",
    ]
    if scale is not None:
        args.extend(["--rscale", scale])
    options = parser.parse_args(args)
    parse_plot_spec(parser, options)
    ddf = dd.from_pandas(frame, npartitions=2)
    xdatum = _datum("u", limits)
    ydatum = _datum("v", limits)
    adatum = xdatum if mode == "alpha" else None
    if mode == "category":
        cdatum = _datum("category", (0, 1), discrete=True)
    elif mode == "continuous":
        cdatum = _datum("colour", (0, 2))
    else:
        cdatum = None
    rasters = []
    figures = []
    points = datashader.Canvas.points
    savefig = matplotlib.figure.Figure.savefig

    def record_points(canvas, source, x, y, **kwargs):
        raster = points(canvas, source, x, y, **kwargs)
        rasters.append((canvas, source.compute(), x, y, raster))
        return raster

    def record_savefig(figure, *args, **kwargs):
        figures.append(figure)
        return savefig(figure, *args, **kwargs)

    monkeypatch.setattr(datashader.Canvas, "points", record_points)
    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", record_savefig)
    png = tmp_path / "polar.png"
    cache = {}
    colors = ["#0000ff", "#ff0000"]
    output = data_plots.create_plot(
        ddf, {}, xdatum, ydatum, adatum, "mean", cdatum,
        cmap=colors, bmap=colors, dmap=colors, normalize="linear",
        xlabel="u (wavelengths)", ylabel="v (wavelengths)",
        title="Polar uv coverage", pngname=str(png), options=options,
        minmax_cache=cache,
        extra_markup=[("axhline", [10], {"color": "black"}),
                      ("axvline", [np.pi / 2], {"color": "black"})],
    )
    return SimpleNamespace(
        output=output, png=png, cache=cache, rasters=rasters, figures=figures)


def test_polar_defaults():
    parser, _ = cli()
    options = parser.parse_args(["test.ms", "--polar", "-x", "u", "-y", "v"])
    parse_plot_spec(parser, options)
    assert options.polar
    assert options.rscale is None
    assert (options.xscale, options.yscale) == ("linear", "linear")


@pytest.mark.parametrize("args, message", [
    (["--polar", "--xscale", "symlog"], "uses --rscale"),
    (["--polar", "--yscale", "symlog"], "uses --rscale"),
    (["--rscale", "linear"], "requires --polar"),
    (["--rscale", "symlog"], "requires --polar"),
])
def test_invalid_polar_options(args, message, capsys):
    parser, _ = cli()
    options = parser.parse_args(["test.ms", *args])
    with pytest.raises(SystemExit) as error:
        parse_plot_spec(parser, options)
    assert error.value.code == 2
    assert message in capsys.readouterr().err


def test_invalid_radial_scale():
    parser, _ = cli()
    with pytest.raises(SystemExit) as error:
        parser.parse_args(["test.ms", "--polar", "--rscale", "log"])
    assert error.value.code == 2


@pytest.mark.parametrize("xdatum, ydatum, message", [
    (_datum("u", discrete=True), _datum("v"), "continuous"),
    (_datum("u"), _datum("v", discrete=True), "continuous"),
    (_datum("u"), _datum("time", unit="s"), "matching units"),
])
def test_invalid_polar_axes(xdatum, ydatum, message):
    with pytest.raises(ValueError, match=message):
        data_plots.validate_plot_axes(
            SimpleNamespace(polar=True), xdatum, ydatum)
    data_plots.validate_plot_axes(
        SimpleNamespace(polar=False), xdatum, ydatum)


@pytest.mark.parametrize("scale", [None, "linear", "symlog"])
@pytest.mark.parametrize("linthresh", [1, 10, 100])
@pytest.mark.parametrize("mode", [
    "density", "alpha", "category", "continuous",
])
def test_polar_render_and_circular_geometry(
        scale, linthresh, mode, tmp_path, monkeypatch):
    theta = np.arange(16) * (2 * np.pi / 16)
    radii = np.array([1., 10., 100., 1000.])
    theta = np.tile(theta, len(radii))
    radii = np.repeat(radii, 16)
    frame = pd.DataFrame({
        "u": radii * np.cos(theta), "v": radii * np.sin(theta),
        "category": np.arange(len(theta)) % 2,
        "colour": np.ones(len(theta)),
    })
    frame.loc[len(frame)] = [0, 0, 0, 1]
    result = _render(
        frame, tmp_path, monkeypatch, scale, linthresh, mode)
    assert result.output == str(result.png)
    with Image.open(result.png) as image:
        assert image.format == "PNG"
        assert image.width > 0 and image.height > 0
    for axis in ("u", "v"):
        np.testing.assert_allclose(
            result.cache[axis], [frame[axis].min(), frame[axis].max()])
    canvas, source, x, y, raster = result.rasters[0]
    pd.testing.assert_frame_equal(source[frame.columns], frame)
    angles = np.mod(np.arctan2(frame["v"], frame["u"]), 2 * np.pi)
    radii = np.hypot(frame["u"], frame["v"])
    transform = (
        SymmetricalLogTransform(10, linthresh, 1)
        if scale != "linear" else None)
    mapped = transform.transform(radii) if transform is not None else radii
    np.testing.assert_allclose(source[x], angles)
    np.testing.assert_allclose(source[y], mapped)
    np.testing.assert_allclose(canvas.x_range, [0, 2 * np.pi])
    np.testing.assert_allclose(canvas.y_range, [0, np.max(mapped)])
    expected, _, _ = np.histogram2d(
        mapped, angles, bins=(64, 72),
        range=[canvas.y_range, canvas.x_range])
    if mode == "alpha":
        sums, _, _ = np.histogram2d(
            mapped, angles, bins=(64, 72),
            range=[canvas.y_range, canvas.x_range], weights=frame["u"])
        means = np.full_like(sums, np.nan)
        np.divide(sums, expected, out=means, where=expected > 0)
        np.testing.assert_allclose(raster.data, means, atol=1e-12)
    else:
        counts = raster.data.sum(axis=2) if raster.ndim == 3 else raster.data
        np.testing.assert_array_equal(counts, expected)
        assert counts.sum() == len(frame)

    fig = result.figures[0]
    ax = fig.axes[0]
    assert ax.name == "polar"
    assert ax.get_xscale() == "linear"
    assert ax.get_yscale() == ("linear" if scale == "linear" else "symlog")
    assert ax.get_rorigin() == 0
    assert ax.get_ylim()[0] == 0
    assert ax.get_ylabel() == "Radius (wavelengths)"
    assert "Angle from u toward v" in ax.get_xlabel()
    np.testing.assert_array_equal(ax.lines[0].get_ydata(), [10, 10])
    np.testing.assert_array_equal(
        ax.lines[1].get_xdata(), [np.pi / 2, np.pi / 2])
    mesh = ax.collections[0].get_coordinates()
    np.testing.assert_allclose(
        mesh[0, :, 0], np.linspace(0, 2 * np.pi, 73))
    radial_edges = np.linspace(*canvas.y_range, 65)
    if transform is not None:
        radial_edges = transform.inverted().transform(radial_edges)
    np.testing.assert_allclose(mesh[:, 0, 1], radial_edges)
    fig.canvas.draw()
    center = ax.transData.transform([[0, 0]])[0]
    display_radii = []
    circle_angles = np.linspace(0, 2 * np.pi, 361)
    for radius in (0, 1, 10, 100, 1000):
        positions = ax.transData.transform(np.column_stack((
            circle_angles, np.full_like(circle_angles, radius))))
        distances = np.linalg.norm(positions - center, axis=1)
        assert np.ptp(distances) < 1e-8
        display_radii.append(np.mean(distances))
    assert display_radii[0] == 0
    assert np.all(np.diff(display_radii) > 0)
    physical_radii = np.array([0., 1., 10., 100., 1000.])
    expected_radii = (
        transform.transform(physical_radii) if transform is not None
        else physical_radii)
    np.testing.assert_allclose(
        np.array(display_radii) / display_radii[-1],
        expected_radii / expected_radii[-1], atol=1e-12)


def test_polar_clipping_nan_and_column_collisions(tmp_path, monkeypatch):
    frame = pd.DataFrame({
        "u": [0., 3., -3., 100., np.nan, np.inf],
        "v": [0., 4., -4., 0., 1., 1.],
        "__shadems_theta": [42] * 6,
        "__shadems_radius": [43] * 6,
    })
    result = _render(
        frame, tmp_path, monkeypatch, limits=(-5, 5))
    assert result.cache == {}
    canvas, source, x, y, raster = result.rasters[0]
    pd.testing.assert_frame_equal(source[frame.columns], frame.iloc[:3])
    assert x != "__shadems_theta"
    assert y != "__shadems_radius"
    transform = SymmetricalLogTransform(10, 10, 1)
    np.testing.assert_allclose(canvas.y_range, transform.transform([0, 5]))
    assert raster.data.sum() == 3
    np.testing.assert_allclose(source[y], transform.transform([0, 5, 5]))
    assert source[x].iloc[2] - source[x].iloc[1] == pytest.approx(np.pi)


@pytest.mark.parametrize("scale", ["linear", "symlog"])
def test_polar_origin_only(scale, tmp_path, monkeypatch):
    result = _render(
        pd.DataFrame({"u": [0.], "v": [0.]}),
        tmp_path, monkeypatch, scale=scale, limits=(-1, 1))
    assert result.output == str(result.png)
    assert result.rasters[0][-1].data.sum() == 1
    ax = result.figures[0].axes[0]
    ax.figure.canvas.draw()
    assert np.isfinite(ax.transData.transform([[0, 0]])).all()


@pytest.mark.parametrize("frame", [
    pd.DataFrame({"u": [np.nan], "v": [np.nan]}),
    pd.DataFrame({"u": [100.], "v": [100.]}),
])
def test_polar_no_selected_points(frame, tmp_path, monkeypatch):
    messages = []
    monkeypatch.setattr(data_plots.log, "info", messages.append)
    result = _render(
        frame, tmp_path, monkeypatch, limits=(-1, 1))
    assert result.output is None
    assert not result.png.exists()
    assert not result.rasters
    assert any("no valid data" in message for message in messages)
