from types import SimpleNamespace

import dask.dataframe as dd
import datashader
import matplotlib.figure
from matplotlib.scale import SymmetricalLogTransform
import numpy as np
import pandas as pd
from PIL import Image
import pytest

from shade_ms.__main__ import cli, parse_plot_spec, parse_radial_minimum
from shade_ms import data_plots


def _datum(label, minmax=(None, None), discrete=False, unit="wavelengths"):
    return SimpleNamespace(
        label=label, fullname=label, function=label, minmax=minmax,
        is_discrete=discrete,
        mapper=SimpleNamespace(unit=unit), subset_indices=None,
        discretized_labels=None, subset_remapper=None, nlevels=2, columns=())


def _render(frame, tmp_path, monkeypatch, scale=None, linthresh=10,
            mode="density", limits=(None, None), r_min=None,
            unit="wavelengths"):
    parser, _ = cli()
    args = [
        "test.ms", "-x", "u", "-y", "v", "--polar",
        "--linthresh", str(linthresh), "--xcanvas", "72", "--ycanvas", "64",
    ]
    if scale is not None:
        args.extend(["--rscale", scale])
    if r_min is not None:
        args.extend(["--r_min", r_min])
    options = parser.parse_args(args)
    parse_plot_spec(parser, options)
    if "wavelength" in frame.columns:
        options.polar_wavelength_label = "wavelength"
    ddf = dd.from_pandas(frame, npartitions=2)
    xdatum = _datum("u", limits, unit=unit)
    ydatum = _datum("v", limits, unit=unit)
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
    assert options.r_min is None
    assert (options.xscale, options.yscale) == ("linear", "linear")


@pytest.mark.parametrize("args, message", [
    (["--polar", "--xscale", "symlog"], "uses --rscale"),
    (["--polar", "--yscale", "symlog"], "uses --rscale"),
    (["--rscale", "linear"], "requires --polar"),
    (["--rscale", "symlog"], "requires --polar"),
    (["--r_min", "0"], "requires --polar"),
    (["--r-min", "5km"], "requires --polar"),
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


@pytest.mark.parametrize("value, number, unit", [
    ("0", 0, ""), ("100", 100, ""), ("2e3", 2000, ""),
    ("5000m", 5000, "m"), ("5km", 5000, "m"),
    ("5 km", 5000, "m"), ("5KM", 5000, "m"), ("0m", 0, "m"),
])
def test_parse_minimum_radius(value, number, unit):
    assert parse_radial_minimum(value) == (number, unit)


@pytest.mark.parametrize("value", [
    "-1", "-5km", "nan", "inf", "-inf", "nanm", "infkm",
    "1e308km", "5cm", "km", "", "five",
])
def test_invalid_minimum_radius(value, capsys):
    parser, _ = cli()
    with pytest.raises(SystemExit) as error:
        parser.parse_args(["test.ms", "--polar", f"--r_min={value}"])
    assert error.value.code == 2
    assert "finite nonnegative radius" in capsys.readouterr().err


def test_minimum_radius_aliases():
    parser, _ = cli()
    for name in ("--r_min", "--r-min"):
        options = parser.parse_args(["test.ms", "--polar", name, "5km"])
        assert options.r_min == (5000, "m")


@pytest.mark.parametrize("scale", ["linear", "symlog"])
def test_minimum_radius_filter_and_geometry(scale, tmp_path, monkeypatch):
    frame = pd.DataFrame({"u": [0., 4., 5., 10., 100.], "v": [0.] * 5})
    result = _render(
        frame, tmp_path, monkeypatch, scale=scale, r_min="5")
    canvas, source, _, _, raster = result.rasters[0]
    pd.testing.assert_frame_equal(source[frame.columns], frame.iloc[2:])
    assert raster.data.sum() == 3
    transform = (
        SymmetricalLogTransform(10, 10, 1) if scale == "symlog" else None)
    expected_bounds = (
        transform.transform([5, 100]) if transform is not None else [5, 100])
    np.testing.assert_allclose(canvas.y_range, expected_bounds)
    ax = result.figures[0].axes[0]
    assert ax.get_ylim()[0] == pytest.approx(5)
    assert ax.get_rorigin() == 5
    ax.figure.canvas.draw()
    center = ax.transData.transform([[0, 5]])[0]
    theta = np.linspace(0, 2 * np.pi, 361)
    display_radii = []
    for radius in (5, 10, 100):
        positions = ax.transData.transform(np.column_stack((
            theta, np.full_like(theta, radius))))
        distances = np.linalg.norm(positions - center, axis=1)
        assert np.isfinite(distances).all()
        assert np.ptp(distances) < 1e-8
        display_radii.append(np.mean(distances))
    assert display_radii[0] == 0
    assert np.all(np.diff(display_radii) > 0)
    mesh = ax.collections[0].get_coordinates()
    assert mesh[0, 0, 1] == pytest.approx(5)
    expected_radii = (
        transform.transform([5, 10, 100]) if transform is not None
        else np.array([5, 10, 100]))
    expected_radii -= expected_radii[0]
    np.testing.assert_allclose(
        np.array(display_radii) / display_radii[-1],
        expected_radii / expected_radii[-1], atol=1e-12)


@pytest.mark.parametrize("mode", [
    "density", "alpha", "category", "continuous",
])
def test_physical_radius_per_channel(mode, tmp_path, monkeypatch):
    frame = pd.DataFrame({
        "u": [4999., 5000., 6000., 2499.5, 2500., 3000.],
        "v": [0.] * 6, "wavelength": [1., 1., 1., 2., 2., 2.],
        "category": [0, 1, 0, 1, 0, 1], "colour": [1.] * 6,
    })
    result = _render(
        frame, tmp_path, monkeypatch, r_min="5km", mode=mode)
    canvas, source, _, _, raster = result.rasters[0]
    pd.testing.assert_frame_equal(
        source[frame.columns], frame.iloc[[1, 2, 4, 5]])
    projected_metres = np.hypot(source["u"], source["v"])
    projected_metres *= source["wavelength"]
    assert projected_metres.min() == 5000
    assert (projected_metres >= 5000).all()
    transform = SymmetricalLogTransform(10, 10, 1)
    np.testing.assert_allclose(
        canvas.y_range, transform.transform([2500, 6000]))
    assert result.figures[0].axes[0].get_rorigin() == 2500
    if mode == "alpha":
        np.testing.assert_allclose(
            np.sort(raster.data[np.isfinite(raster.data)]),
            [2500, 3000, 5000, 6000])
    else:
        assert raster.data.sum() == 4


@pytest.mark.parametrize("unit, values, minimum", [
    ("m", [4999., 5000., 6000.], 5000),
    ("km", [4.999, 5., 6.], 5),
])
def test_physical_radius_on_metric_axes(
        unit, values, minimum, tmp_path, monkeypatch):
    result = _render(
        pd.DataFrame({"u": values, "v": [0.] * 3}),
        tmp_path, monkeypatch, r_min="5km", unit=unit)
    assert len(result.rasters[0][1]) == 2
    assert result.figures[0].axes[0].get_rorigin() == minimum


@pytest.mark.parametrize("xdatum, ydatum", [
    (_datum("real", unit=""), _datum("imag", unit="")),
    (_datum("u"), _datum("w")),
])
def test_physical_cutoff_requires_distance_axes(xdatum, ydatum):
    with pytest.raises(ValueError, match="--r_min"):
        data_plots.validate_plot_axes(
            SimpleNamespace(polar=True, r_min=parse_radial_minimum("5km")),
            xdatum, ydatum)


def test_physical_cutoff_requires_channel_wavelengths(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="requires channel wavelengths"):
        _render(
            pd.DataFrame({"u": [5000.], "v": [0.]}),
            tmp_path, monkeypatch, r_min="5km")


@pytest.mark.parametrize("scale", ["linear", "symlog"])
def test_only_points_at_minimum_radius(scale, tmp_path, monkeypatch):
    result = _render(
        pd.DataFrame({"u": [5.], "v": [0.]}),
        tmp_path, monkeypatch, scale=scale, r_min="5")
    assert result.output == str(result.png)
    assert result.rasters[0][-1].data.sum() == 1
    ax = result.figures[0].axes[0]
    assert ax.get_ylim()[1] > ax.get_ylim()[0]
    ax.figure.canvas.draw()
    assert np.isfinite(ax.transData.transform([[0, 5]])).all()


@pytest.mark.parametrize("minimum", ["10", "5km"])
def test_no_points_above_minimum(minimum, tmp_path, monkeypatch):
    messages = []
    monkeypatch.setattr(data_plots.log, "info", messages.append)
    result = _render(
        pd.DataFrame({"u": [1.], "v": [0.], "wavelength": [1.]}),
        tmp_path, monkeypatch, r_min=minimum)
    assert result.output is None
    assert not result.png.exists()
    assert any("no valid data" in message for message in messages)
