# EarthCARE 描画スクリプト

EarthCAREのSynergetic Target Classification (AC__TC__2B) データを可視化するスクリプト．

## ディレクトリ構成

```
earthcare/
├── data/        衛星データ（HDF5, HDR, ZIP）
├── figures/
│   ├── curtain/  カーテンプロット
│   └── map/      軌道トラック地図（南極正射方位図法）
└── scripts/
    ├── plot_ac_tc.py  カーテンプロット生成
    └── plot_map.py    軌道トラック地図生成
```

## 依存ライブラリ

```
h5py, matplotlib, numpy, cartopy
```

## 使い方

```bash
# カーテンプロット
python scripts/plot_ac_tc.py data/<file>.h5

# 高度・緯度範囲を指定する場合
python scripts/plot_ac_tc.py data/<file>.h5 --height 0-15 --lat-min -85 --lat-max -60

# 軌道トラック地図
python scripts/plot_map.py data/<file>.h5
```

出力先を省略すると `figures/` 以下の各サブディレクトリに保存される。

## データ

| 変数 | 内容 |
|------|------|
| `synergetic_target_classification` | ターゲット分類（カーテンプロット） |
| `latitude` / `longitude` | 軌道上の緯度・経度 |
| `height` | 高度 [m] |
| `elevation` | 地表高度 [m] |
| `tropopause_height` | 対流圏界面高度 [m] |
