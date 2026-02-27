#!/usr/bin/env python3
"""EarthCARE AC__TC__2B 軌道トラック地図プロット

南極正距方位図法（cartopy）で EarthCARE の軌道トラックと
昭和基地・ドームふじへの最接近点を描画する。

Usage:
    python plot_map.py <h5_file> [--outdir <output_directory>]

Example:
    python plot_map.py ../data/ECA_EXBC_AC__TC__2B_20251223T000721Z_20251223T012326Z_08916G.h5
"""

import argparse
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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


def find_closest_point(lat, lon, site):
    """軌道上で地上サイトに最も近い点のインデックスと距離を返す"""
    dists = haversine_km(lat, lon, site["lat"], site["lon"])
    idx = np.argmin(dists)
    return idx, dists[idx]


def load_data(h5_path):
    """HDF5 ファイルから地図描画に必要なデータを読み込む"""
    f = h5py.File(h5_path, "r")
    sd = f["ScienceData"]
    hdr = f["HeaderData/VariableProductHeader/MainProductHeader"]

    data = {
        "lat": sd["latitude"][:],
        "lon": sd["longitude"][:],
        "frame_start": hdr["frameStartTime"][()].decode(),
        "frame_stop": hdr["frameStopTime"][()].decode(),
        "orbit": int(hdr["orbitNumber"][()]),
        "frame_id": hdr["frameID"][()].decode(),
    }

    f.close()
    return data


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
        for site, color, marker, ms in [(SYOWA, "black", ".", 200), (DOME_FUJI, "black", ".", 150)]:
            ax.scatter(
                site["lon"], site["lat"], c=color, marker=marker, s=ms,
                zorder=5, transform=transform, edgecolors="black", linewidths=0.5,
            )
            ax.text(
                site["lon"] + 3, site["lat"] + 0.5, site["name"],
                fontsize=12, fontweight="bold", color=color, transform=transform,
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
                fontsize=10, color=color, ha="center", transform=transform,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7),
            )
    else:
        # cartopy なしのフォールバック
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.plot(lon, lat, "b-", linewidth=1, alpha=0.6, label="EarthCARE track")
        ax.plot(SYOWA["lon"], SYOWA["lat"], "r*", markersize=15, label="Syowa Station", zorder=5)
        ax.plot(DOME_FUJI["lon"], DOME_FUJI["lat"], "b^", markersize=12, label="Dome Fuji", zorder=5)
        ax.set_xlabel("Longitude [°]", fontsize=14)
        ax.set_ylabel("Latitude [°]", fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-180, 180)
        ax.set_ylim(-85, -60)

    ax.set_title(
        f"EarthCARE Orbit {orbit:05d} Frame {frame_id}\n{start[:10]}",
        fontsize=14,
    )
    ax.legend(fontsize=12, loc="lower left")

    subdir = outdir / "map"
    subdir.mkdir(parents=True, exist_ok=True)
    out = subdir / f"map_track_{orbit:05d}{frame_id}.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def main():
    parser = argparse.ArgumentParser(
        description="EarthCARE AC__TC__2B 軌道トラック地図（南極正距方位図法）"
    )
    parser.add_argument("h5_file", help="AC__TC__2B の HDF5 ファイルパス")
    parser.add_argument(
        "--outdir", default=None,
        help="出力ディレクトリ（デフォルト: 入力ファイルと同じディレクトリ）",
    )
    args = parser.parse_args()

    h5_path = Path(args.h5_file)
    outdir = Path(args.outdir) if args.outdir else h5_path.parent.parent / "figures"

    print(f"Reading: {h5_path}")
    data = load_data(h5_path)

    orbit = data["orbit"]
    frame_id = data["frame_id"]
    print(f"Orbit: {orbit:05d}, Frame: {frame_id}")
    print(f"Time:  {data['frame_start']} — {data['frame_stop']}")
    print(f"Lat:   {data['lat'].min():.2f} to {data['lat'].max():.2f}")
    print(f"Lon:   {data['lon'].min():.2f} to {data['lon'].max():.2f}")

    for site in [SYOWA, DOME_FUJI]:
        idx, d = find_closest_point(data["lat"], data["lon"], site)
        print(f"{site['name']:10s}: nearest {d:.0f} km "
              f"(track at {data['lat'][idx]:.2f}°, {data['lon'][idx]:.2f}°E)")

    plot_map(data, outdir)
    print("\nDone!")


if __name__ == "__main__":
    main()
