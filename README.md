# transforms_tools

把多次采集、由 SfM(Metashape / VGGT 等)得到的相机位姿 `transforms.json`,
**标定到真实物理尺度、统一对齐到同一套坐标系**的命令行工具。

适用于「相机绕中心转、被摄物体不动」的多相机转台/机械臂式采集设备:
先用相机齐全的少数几次采集**联合标定**出设备的机械常数,之后**任意一次采集
(相机数任意、可缺相机、可改过径向距离/俯仰)都能被正确缩放并摆正**,输出的点云
即为真实毫米尺度、坐标轴统一。

## 能做什么

- `calibrate_rig`:用好日期联合标定机械常数(电机 0 位偏移 `r`、杠杆臂 `L`)。
- `canonicalize`:用标定结果把各次 `transforms.json` 缩放到真实 mm 并对齐到统一坐标系。
- `finalize`:整理产物,把统一坐标系版本提升为 `transforms.json`,其余归档。
- `set_aabb`:批量写入训练用的包围盒 `aabb` / `aabb_scale`。
- `use_existed_trans`:把一份已算好的位姿复用到一批新目录(按图片过滤)。

## 快速开始

```bash
# 1. 安装(见 docs/USAGE.md 获取完整环境说明)
pip install -e .

# 2. 运行(安装后可直接用命令行)
transforms_tools --help
```

## 文档

请阅读 `docs/` 下的两份文档:

- **[docs/USAGE.md](docs/USAGE.md)** —— 环境安装、命令行运行方式、每个子命令的参数与完整示例。
- **[docs/PRINCIPLE.md](docs/PRINCIPLE.md)** —— 数学原理:标定模型、未知量与方程、坐标系对齐的推导。

## 依赖

Python ≥ 3.8,`numpy`、`scipy`(安装时自动拉取)。

## 目录结构

```
transforms_tools/
├── README.md                     # 本文件
├── pyproject.toml                # 打包/安装配置(提供 transforms_tools 命令)
├── docs/
│   ├── USAGE.md                  # 安装与使用说明
│   └── PRINCIPLE.md              # 原理与公式说明
└── scripts/
    └── transforms_tools/         # Python 包(核心代码)
        ├── cli.py                # 命令行入口
        ├── calibrate_rig.py      # 联合标定 r、L
        ├── canonicalize.py       # 解尺度 + 对齐到 canonical
        ├── finalize.py           # 整理产物
        ├── set_aabb.py           # 写 aabb / aabb_scale
        └── use_existed_trans.py  # 复用已有位姿
```
