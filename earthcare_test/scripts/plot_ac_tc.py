#!/usr/bin/env python3
"""EarthCARE AC__TC__2B (Synergetic Target Classification) 可視化スクリプト

3種類の図を生成:
  1. カーテンプロット（along-track 全域、上軸に経度表示）
  2. カーテンプロット（最接近点周辺の拡大）
  3. 南極正距方位図法の軌道トラック地図

Usage:
    python plot_ac_tc.py <h5_file> [--outdir <output_directory>]

Example:
    python plot_ac_tc.py ../ECA_EXBC_AC__TC__2B_20251223T000721Z_20251223T012326Z_08916G.h5
"""

import argparse
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np

# 昭和基地・ドームふじの座標
SYOWA = {"name": "Syowa", "lat": -69.0, "lon": 39.6, "alt": 29}
DOME_FUJI = {"name": "Dome Fuji", "lat": -77.3, "lon": 39.7, "alt": 3810}


def haversine_km(lat1, lon1, lat2, lon2):
    """2点間の大圏距離 [km]"""
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def cumulative_distance(lat, lon):
    """沿軌道の累積距離 [km] を計算する"""
    dist = np.zeros(len(lat))
    for i in range(1, len(lat)):
        dist[i] = dist[i - 1] + haversine_km(lat[i - 1], lon[i - 1], lat[i], lon[i])
    return dist


def find_closest_point(lat, lon, site):
    """軌道上で地上サイトに最も近い点のインデックスと距離を返す"""
    dists = haversine_km(lat, lon, site["lat"], site["lon"])
    idx = np.argmin(dists)
    return idx, dists[idx]


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
    ax.set_xlabel("Latitude [°]", fontsize=12)

    # 上側: 経度ラベル
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(tick_positions)
    lon_labels = [f"{lon[i]:.1f}°E" if lon[i] >= 0 else f"{abs(lon[i]):.1f}°W" for i in tick_indices]
    ax2.set_xticklabels(lon_labels, fontsize=8)
    ax2.set_xlabel("Longitude", fontsize=10)
    return ax2


def plot_curtain_full(data, cmap, norm, patches, outdir):
    """カーテンプロット（全域、X軸=沿軌道距離）"""
    fig, ax = plt.subplots(figsize=(18, 6))

    lat = data["lat"]
    lon = data["lon"]
    dist_km = cumulative_distance(lat, lon)
    height_km = data["height"] / 1000.0
    tc_masked = np.ma.masked_where(data["tc"] == -1, data["tc"])

    ax.pcolormesh(
        np.broadcast_to(dist_km[:, np.newaxis], height_km.shape),
        height_km,
        tc_masked,
        cmap=cmap, norm=norm,
        shading="nearest", rasterized=True,
    )

    ax.fill_between(dist_km, 0, data["elevation"] / 1000, color="black", alpha=0.8)

    valid_t = data["tropopause"] > 0
    ax.plot(
        dist_km[valid_t], data["tropopause"][valid_t] / 1000,
        "k--", linewidth=0.5, alpha=0.5,
    )

    # 昭和基地・ドームふじの最接近点をマーク
    for site, color, marker in [(SYOWA, "red", "*"), (DOME_FUJI, "blue", "^")]:
        idx, d = find_closest_point(lat, lon, site)
        ax.axvline(dist_km[idx], color=color, linewidth=1.5, alpha=0.8, linestyle="--")
        label = f"{site['name']}\n({d:.0f} km away)"
        ax.text(
            dist_km[idx], 19, label, color=color, fontsize=8,
            fontweight="bold", ha="center", va="top",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, alpha=0.8),
        )

    ax.set_xlim(dist_km[0], dist_km[-1])
    ax.set_ylim(0, 20)
    ax.set_ylabel("Height [km]", fontsize=12)

    add_lat_lon_axes(ax, dist_km, lat, lon)

    orbit = data["orbit"]
    frame_id = data["frame_id"]
    start = data["frame_start"]
    stop = data["frame_stop"]
    fig.suptitle(
        f"EarthCARE AC__TC__2B Synergetic Target Classification\n"
        f"Orbit {orbit:05d} Frame {frame_id} ({start} — {stop})",
        fontsize=13, y=1.02,
    )

    ax.legend(
        handles=patches, loc="upper left", fontsize=6, ncol=2,
        framealpha=0.9, bbox_to_anchor=(1.01, 1.0),
    )

    out = outdir / f"curtain_TC_{orbit:05d}{frame_id}.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def plot_curtain_zoom(data, cmap, norm, patches, outdir,
                      margin_km=500, height_max=15):
    """カーテンプロット（昭和基地・ドームふじ最接近域を拡大）"""
    lat = data["lat"]
    lon = data["lon"]
    dist_km = cumulative_distance(lat, lon)

    # 両サイトの最接近点を求める
    idx_s, d_s = find_closest_point(lat, lon, SYOWA)
    idx_d, d_d = find_closest_point(lat, lon, DOME_FUJI)

    # 表示範囲: 両最接近点を含む ± margin_km
    idx_min = min(idx_s, idx_d)
    idx_max = max(idx_s, idx_d)
    d_min = max(0, dist_km[idx_min] - margin_km)
    d_max = min(dist_km[-1], dist_km[idx_max] + margin_km)
    mask = (dist_km >= d_min) & (dist_km <= d_max)

    dist_z = dist_km[mask]
    height_z = data["height"][mask, :] / 1000.0
    tc_z = np.ma.masked_where(data["tc"][mask, :] == -1, data["tc"][mask, :])
    elev_z = data["elevation"][mask]
    tropo_z = data["tropopause"][mask]
    lat_z = lat[mask]
    lon_z = lon[mask]

    fig, ax = plt.subplots(figsize=(16, 6))

    ax.pcolormesh(
        np.broadcast_to(dist_z[:, np.newaxis], height_z.shape),
        height_z,
        tc_z,
        cmap=cmap, norm=norm,
        shading="nearest", rasterized=True,
    )

    ax.fill_between(dist_z, 0, elev_z / 1000, color="black", alpha=0.8)

    valid_t = tropo_z > 0
    ax.plot(
        dist_z[valid_t], tropo_z[valid_t] / 1000,
        "k--", linewidth=0.8, alpha=0.5,
    )

    for site, color, idx, d in [
        (SYOWA, "red", idx_s, d_s),
        (DOME_FUJI, "blue", idx_d, d_d),
    ]:
        ax.axvline(dist_km[idx], color=color, linewidth=2, alpha=0.8, linestyle="--")
        label = f"{site['name']}\n({lat[idx]:.1f}°, {lon[idx]:.1f}°E)\n{d:.0f} km away"
        ax.text(
            dist_km[idx], height_max - 0.5, label, color=color, fontsize=8,
            fontweight="bold", ha="center", va="top",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, alpha=0.8),
        )

    ax.set_xlim(d_min, d_max)
    ax.set_ylim(0, height_max)
    ax.set_ylabel("Height [km]", fontsize=12)

    add_lat_lon_axes(ax, dist_z, lat_z, lon_z)

    orbit = data["orbit"]
    frame_id = data["frame_id"]
    fig.suptitle(
        f"EarthCARE AC__TC__2B — Nearest approach to Syowa & Dome Fuji\n"
        f"Orbit {orbit:05d} Frame {frame_id}  |  "
        f"Syowa: {d_s:.0f} km, Dome Fuji: {d_d:.0f} km",
        fontsize=12, y=1.02,
    )

    ax.legend(
        handles=patches, loc="upper left", fontsize=6, ncol=2,
        framealpha=0.9, bbox_to_anchor=(1.01, 1.0),
    )

    out = outdir / f"curtain_TC_{orbit:05d}{frame_id}_zoom.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def plot_map(data, outdir):
    """南極正距方位図法の軌道トラック地図"""
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        has_cartopy = True
    except ImportError:
        has_cartopy = False

    lat = data["lat"]
    lon = data["lon"]
    orbit = data["orbit"]
    frame_id = data["frame_id"]
    start = data["frame_start"]

    if has_cartopy:
        proj = ccrs.SouthPolarStereo()
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"projection": proj})
        ax.set_extent([-180, 180, -90, -60], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND, facecolor="#d2b48c", edgecolor="gray", linewidth=0.5)
        ax.add_feature(cfeature.OCEAN, facecolor="#e6f2ff")
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)

        transform = ccrs.PlateCarree()
        ax.plot(lon, lat, "b-", linewidth=1.5, alpha=0.7, label="EarthCARE track", transform=transform)

        # 軌道上の方向を示す矢印
        mid = len(lat) // 2
        ax.annotate(
            "", xy=(lon[mid + 50], lat[mid + 50]), xytext=(lon[mid], lat[mid]),
            arrowprops=dict(arrowstyle="->", color="blue", lw=1.5),
            transform=transform,
        )

        # 昭和基地・ドームふじ
        for site, color, marker, ms in [(SYOWA, "red", "*", 200), (DOME_FUJI, "blue", "^", 150)]:
            ax.scatter(
                site["lon"], site["lat"], c=color, marker=marker, s=ms,
                zorder=5, transform=transform, edgecolors="black", linewidths=0.5,
            )
            ax.text(
                site["lon"] + 3, site["lat"] + 0.5, site["name"],
                fontsize=10, fontweight="bold", color=color, transform=transform,
            )

            # 最接近点を表示
            idx, d = find_closest_point(lat, lon, site)
            ax.plot(
                [site["lon"], lon[idx]], [site["lat"], lat[idx]],
                color=color, linewidth=1, linestyle=":", alpha=0.8, transform=transform,
            )
            mid_lon = (site["lon"] + lon[idx]) / 2
            mid_lat = (site["lat"] + lat[idx]) / 2
            ax.text(
                mid_lon, mid_lat, f"{d:.0f} km",
                fontsize=8, color=color, ha="center", transform=transform,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7),
            )
    else:
        # cartopy なしのフォールバック
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.plot(lon, lat, "b-", linewidth=1, alpha=0.6, label="EarthCARE track")
        ax.plot(SYOWA["lon"], SYOWA["lat"], "r*", markersize=15, label="Syowa Station", zorder=5)
        ax.plot(DOME_FUJI["lon"], DOME_FUJI["lat"], "b^", markersize=12, label="Dome Fuji", zorder=5)
        ax.set_xlabel("Longitude [°]", fontsize=12)
        ax.set_ylabel("Latitude [°]", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-180, 180)
        ax.set_ylim(-85, -60)

    ax.set_title(
        f"EarthCARE Orbit {orbit:05d} Frame {frame_id}\n{start[:10]}",
        fontsize=13,
    )
    ax.legend(fontsize=10, loc="lower left")

    out = outdir / f"map_track_{orbit:05d}{frame_id}.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def main():
    parser = argparse.ArgumentParser(
        description="EarthCARE AC__TC__2B (Synergetic Target Classification) 可視化"
    )
    parser.add_argument("h5_file", help="AC__TC__2B の HDF5 ファイルパス")
    parser.add_argument(
        "--outdir", default=None,
        help="出力ディレクトリ（デフォルト: 入力ファイルと同じディレクトリ）",
    )
    args = parser.parse_args()

    h5_path = Path(args.h5_file)
    outdir = Path(args.outdir) if args.outdir else h5_path.parent
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Reading: {h5_path}")
    data = load_data(h5_path)

    orbit = data["orbit"]
    frame_id = data["frame_id"]
    print(f"Orbit: {orbit:05d}, Frame: {frame_id}")
    print(f"Time:  {data['frame_start']} — {data['frame_stop']}")
    print(f"Lat:   {data['lat'].min():.2f} to {data['lat'].max():.2f}")
    print(f"Lon:   {data['lon'].min():.2f} to {data['lon'].max():.2f}")

    # 最接近距離を表示
    for site in [SYOWA, DOME_FUJI]:
        idx, d = find_closest_point(data["lat"], data["lon"], site)
        print(f"{site['name']:10s}: nearest {d:.0f} km "
              f"(track at {data['lat'][idx]:.2f}°, {data['lon'][idx]:.2f}°E)")

    labels, color_list = parse_classification(data["definition"], data["plot_colors"])
    cmap, norm = make_colormap(color_list)

    unique_vals = np.unique(data["tc"][data["tc"] >= 0])
    patches = make_legend_patches(unique_vals, labels, color_list)

    plot_curtain_full(data, cmap, norm, patches, outdir)
    plot_curtain_zoom(data, cmap, norm, patches, outdir)
    plot_map(data, outdir)

    print("\nDone!")


if __name__ == "__main__":
    main()
