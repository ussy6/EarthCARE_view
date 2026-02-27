#!/usr/bin/env python3
"""EarthCARE AC__TC__2B (Synergetic Target Classification) カーテンプロット生成

軌道トラック地図は plot_map.py で生成する。

Usage:
    python plot_ac_tc.py <h5_file> [--outdir DIR] [--height H_MIN-H_MAX] [--lat-min LAT] [--lat-max LAT]

Example:
    python plot_ac_tc.py ../data/ECA_EXBC_AC__TC__2B_20251223T000721Z_20251223T012326Z_08916G.h5
    python plot_ac_tc.py ../data/ECA_EXBC_AC__TC__2B_20251223T000721Z_20251223T012326Z_08916G.h5 --height 0-15 --lat-min -85 --lat-max -60
"""

import argparse
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np

def cumulative_distance(lat, lon):
    """沿軌道の累積距離 [km] を計算する"""
    R = 6371.0
    dist = np.zeros(len(lat))
    for i in range(1, len(lat)):
        dlat = np.radians(lat[i] - lat[i - 1])
        dlon = np.radians(lon[i] - lon[i - 1])
        a = (np.sin(dlat / 2) ** 2
             + np.cos(np.radians(lat[i - 1])) * np.cos(np.radians(lat[i])) * np.sin(dlon / 2) ** 2)
        dist[i] = dist[i - 1] + R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return dist


def load_data(h5_path):
    """HDF5 ファイルからデータを読み込む"""
    f = h5py.File(h5_path, "r")
    sd = f["ScienceData"]
    hdr = f["HeaderData/VariableProductHeader/MainProductHeader"]

    data = {
        "lat": sd["latitude"][:],
        "lon": sd["longitude"][:],
        "height": sd["height"][:],
        "tc": sd["synergetic_target_classification"][:],
        "elevation": sd["elevation"][:],
        "tropopause": sd["tropopause_height"][:],
        "frame_start": hdr["frameStartTime"][()].decode(),
        "frame_stop": hdr["frameStopTime"][()].decode(),
        "orbit": int(hdr["orbitNumber"][()]),
        "frame_id": hdr["frameID"][()].decode(),
    }

    tc_ds = sd["synergetic_target_classification"]
    data["definition"] = tc_ds.attrs["definition"].decode()
    data["plot_colors"] = tc_ds.attrs["plot_colors"].decode()

    f.close()
    return data


def parse_classification(definition_str, colors_str):
    """分類定義とカラーをパースする"""
    labels = {}
    for line in definition_str.strip().split("\n"):
        parts = line.strip().split(": ", 1)
        if len(parts) == 2:
            labels[int(parts[0])] = parts[1]

    color_list = [c.strip() for c in colors_str.strip().split("\n")]
    return labels, color_list


def make_colormap(color_list):
    """分類用カラーマップを作成する"""
    cmap = ListedColormap(color_list)
    bounds = np.arange(-1.5, -1.5 + len(color_list) + 1, 1)
    norm = BoundaryNorm(bounds, cmap.N)
    return cmap, norm


def make_legend_patches(unique_vals, labels, color_list):
    """凡例パッチを作成する"""
    patches = []
    for v in sorted(unique_vals):
        v = int(v)
        if v in labels and (v + 1) < len(color_list):
            patches.append(
                mpatches.Patch(color=color_list[v + 1], label=f"{v}: {labels[v]}")
            )
    return patches


def add_lat_lon_axes(ax, dist_km, lat, lon):
    """X 軸に緯度、上部軸に経度を表示する"""
    # 下側: 緯度ラベル（等間隔に数箇所）
    n_ticks = 8
    tick_indices = np.linspace(0, len(dist_km) - 1, n_ticks, dtype=int)
    tick_positions = dist_km[tick_indices]
    lat_labels = [f"{lat[i]:.1f}°" for i in tick_indices]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(lat_labels)
    ax.set_xlabel("Latitude [°]", fontsize=14)

    # 上側: 経度ラベル
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(tick_positions)
    lon_labels = [f"{lon[i]:.1f}°E" if lon[i] >= 0 else f"{abs(lon[i]):.1f}°W" for i in tick_indices]
    ax2.set_xticklabels(lon_labels, fontsize=10)
    ax2.set_xlabel("Longitude", fontsize=12)
    return ax2


def plot_curtain(data, cmap, norm, patches, outdir,
                 height_range=(0, 20), lat_range=None):
    """カーテンプロット（along-track、上軸に経度表示）

    Parameters
    ----------
    height_range : (float, float)
        表示する高度範囲 [km]。デフォルト (0, 20)。
    lat_range : (float, float) or None
        表示する緯度範囲 [°]。None のとき全域を表示。
    """
    lat = data["lat"]
    lon = data["lon"]
    dist_km = cumulative_distance(lat, lon)

    if lat_range is not None:
        lat_min, lat_max = lat_range
        mask = (lat >= lat_min) & (lat <= lat_max)
        dist_km = dist_km[mask]
        height_km = data["height"][mask, :] / 1000.0
        tc_masked = np.ma.masked_where(data["tc"][mask, :] == -1, data["tc"][mask, :])
        elevation = data["elevation"][mask]
        tropopause = data["tropopause"][mask]
        lat = lat[mask]
        lon = lon[mask]
    else:
        height_km = data["height"] / 1000.0
        tc_masked = np.ma.masked_where(data["tc"] == -1, data["tc"])
        elevation = data["elevation"]
        tropopause = data["tropopause"]

    h_min, h_max = height_range

    fig, ax = plt.subplots(figsize=(18, 6))

    ax.pcolormesh(
        np.broadcast_to(dist_km[:, np.newaxis], height_km.shape),
        height_km,
        tc_masked,
        cmap=cmap, norm=norm,
        shading="nearest", rasterized=True,
    )

    ax.fill_between(dist_km, h_min, elevation / 1000, color="black", alpha=0.8)

    valid_t = tropopause > 0
    ax.plot(
        dist_km[valid_t], tropopause[valid_t] / 1000,
        "k--", linewidth=0.5, alpha=0.5,
    )

    ax.set_xlim(dist_km[0], dist_km[-1])
    ax.set_ylim(h_min, h_max)
    ax.set_ylabel("Height [km]", fontsize=14)

    add_lat_lon_axes(ax, dist_km, lat, lon)

    orbit = data["orbit"]
    frame_id = data["frame_id"]
    start = data["frame_start"]
    stop = data["frame_stop"]
    fig.suptitle(
        f"EarthCARE AC__TC__2B Synergetic Target Classification\n"
        f"Orbit {orbit:05d} Frame {frame_id} ({start} — {stop})",
        fontsize=14, y=1.02,
    )

    ax.legend(
        handles=patches, loc="upper left", fontsize=9, ncol=1,
        framealpha=0.9, bbox_to_anchor=(1.05, 1.0),
    )

    subdir = outdir / "curtain"
    subdir.mkdir(parents=True, exist_ok=True)
    out = subdir / f"curtain_TC_{orbit:05d}{frame_id}.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def parse_height_range(s):
    """'H_MIN-H_MAX' 形式の文字列を (float, float) にパースする"""
    try:
        a, b = s.split("-", 1)
        return float(a), float(b)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"--height の形式が不正です: '{s}'  例: --height 0-15"
        )


def main():
    parser = argparse.ArgumentParser(
        description="EarthCARE AC__TC__2B (Synergetic Target Classification) 可視化"
    )
    parser.add_argument("h5_file", help="AC__TC__2B の HDF5 ファイルパス")
    parser.add_argument(
        "--outdir", default=None,
        help="出力ディレクトリ（デフォルト: data/ の親ディレクトリ直下の figures/）",
    )
    parser.add_argument(
        "--height", default=None, metavar="H_MIN-H_MAX",
        help="表示する高度範囲 [km]（例: --height 0-15）。デフォルト: 0-20",
    )
    parser.add_argument(
        "--lat-min", dest="lat_min", type=float, default=None, metavar="LAT",
        help="表示する緯度範囲の下限 [°]（例: --lat-min -85）",
    )
    parser.add_argument(
        "--lat-max", dest="lat_max", type=float, default=None, metavar="LAT",
        help="表示する緯度範囲の上限 [°]（例: --lat-max -60）",
    )
    args = parser.parse_args()

    h5_path = Path(args.h5_file)
    outdir = Path(args.outdir) if args.outdir else h5_path.parent.parent / "figures"

    height_range = parse_height_range(args.height) if args.height else (0, 20)
    lat_range = (args.lat_min, args.lat_max) if (args.lat_min is not None and args.lat_max is not None) else None

    print(f"Reading: {h5_path}")
    data = load_data(h5_path)

    orbit = data["orbit"]
    frame_id = data["frame_id"]
    print(f"Orbit: {orbit:05d}, Frame: {frame_id}")
    print(f"Time:  {data['frame_start']} — {data['frame_stop']}")
    print(f"Lat:   {data['lat'].min():.2f} to {data['lat'].max():.2f}")
    print(f"Lon:   {data['lon'].min():.2f} to {data['lon'].max():.2f}")

    labels, color_list = parse_classification(data["definition"], data["plot_colors"])
    cmap, norm = make_colormap(color_list)

    unique_vals = np.unique(data["tc"][data["tc"] >= 0])
    patches = make_legend_patches(unique_vals, labels, color_list)

    plot_curtain(data, cmap, norm, patches, outdir,
                 height_range=height_range, lat_range=lat_range)

    print("\nDone!")


if __name__ == "__main__":
    main()
