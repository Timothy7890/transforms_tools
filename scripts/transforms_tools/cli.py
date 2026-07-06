#!/usr/bin/env python3
"""
transforms_tools —— 相机位姿(transforms.json)标定与统一坐标系工具集(命令行入口)

安装后可直接用 `transforms_tools <子命令> ...`;未安装时用 `python -m transforms_tools ...`。

四步主流程
----------
# 第一步:用相机齐全的好日期联合标定机械常数 r、L(导出 rig_calib.json)
transforms_tools calibrate_rig -i <根> -L 2 -o <标定输出目录>

# 第二步:用标定结果对齐所有日期(任意相机数只解尺度)
transforms_tools canonicalize -i <根> -L 2 -rl <标定输出目录>

# 第三步:把 transforms_canonical.json 提升为 transforms.json,其余 json 归档到 trans/
transforms_tools finalize -i <根> -L 2

# 第四步:写入/修改 aabb 与 aabb_scale
transforms_tools set_aabb -i <根> -L 2 --aabb -200 -200 -200 200 200 500 --aabb-scale 1

辅助
----
# 复用已有 transforms.json 到新目录(按目标 images/ 过滤 frames)
transforms_tools use_existed_trans -i <新目录根> -t <来源/transforms.json> -L 1

# 重复性诊断(只读):判断各次采集位姿能否直接复用
transforms_tools check -i <根> -L 2

写盘类子命令都支持 --dry-run,先预览不写盘。

另外本包还附带两个独立命令(见 __init__/pyproject):
    transforms_viz        —— 网页版 3D 位姿可视化(需可选依赖 [viz])
    copy_first_images     —— 抽取每次采集的代表图
"""

import argparse

from .use_existed_trans import use_existed_trans
from .canonicalize import canonicalize
from .calibrate_rig import calibrate_rig
from .finalize import finalize
from .set_aabb import set_aabb
from .check_repeatability import check_repeatability


def _cmd_use_existed_trans(args):
    use_existed_trans(
        input_root=args.input,
        src_transforms=args.transforms,
        level=args.level,
        method=args.method,
        images_subdir=args.images_subdir,
        dry_run=args.dry_run,
    )


def _cmd_canonicalize(args):
    canonicalize(
        input_root=args.input,
        level=args.level,
        camera_ids=args.camera_ids,
        plane_cam=args.plane_cam,
        azimuth_cam=args.azimuth_cam,
        flip_z=args.flip_z,
        transforms_name=args.transforms_name,
        output_name=args.output_name,
        dry_run=args.dry_run,
        calib_path=args.calib,
    )


def _cmd_calibrate_rig(args):
    calibrate_rig(
        input_root=args.input,
        output_dir=args.output,
        level=args.level,
        camera_ids=args.camera_ids,
        min_cams=args.min_cams,
        transforms_name=args.transforms_name,
        output_name=args.output_name,
        dry_run=args.dry_run,
    )


def _cmd_finalize(args):
    finalize(
        input_root=args.input,
        level=args.level,
        canonical_name=args.canonical_name,
        final_name=args.final_name,
        archive_dir=args.archive_dir,
        pattern=args.pattern,
        dry_run=args.dry_run,
    )


def _cmd_set_aabb(args):
    set_aabb(
        input_root=args.input,
        level=args.level,
        aabb=args.aabb,
        aabb_scale=args.aabb_scale,
        transforms_name=args.transforms_name,
        dry_run=args.dry_run,
    )


def _cmd_check(args):
    check_repeatability(
        input_root=args.input,
        level=args.level,
        reference=args.reference,
        mm_per_unit=args.mm_per_unit,
        camera_ids=args.camera_ids,
        transforms_name=args.transforms_name,
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="transforms_tools",
        description="相机位姿 transforms.json 标定与统一坐标系工具集",
    )
    sub = p.add_subparsers(dest="command", required=True)

    # ── use_existed_trans ──
    sp = sub.add_parser(
        "use_existed_trans",
        help="复用已有 transforms.json 到新目录,并按目标 images/ 过滤 frames",
    )
    sp.add_argument("-i", "--input", required=True,
                    help="新目录根路径(在其下按 -L 深度寻找目标目录)")
    sp.add_argument("-t", "--transforms", required=True,
                    help="来源 transforms.json 路径")
    sp.add_argument("-L", "--level", type=int, default=1,
                    help="处理层级(1-based,最小 1):1=输入目录本身,2=其直接子目录,3=孙目录,... (默认: 1)")
    sp.add_argument("-m", "--method", default="Metashape",
                    help="写入被保留的原 transforms.json 的 method 字段名 (默认: Metashape)")
    sp.add_argument("--images-subdir", default="images",
                    help="目标目录中存放图片的子目录名 (默认: images)")
    sp.add_argument("--dry-run", action="store_true",
                    help="只打印将要进行的操作,不写盘")
    sp.set_defaults(func=_cmd_use_existed_trans)

    # ── canonicalize ──
    sp = sub.add_parser(
        "canonicalize",
        help="各目录 transforms.json 各自解尺度,缩放并对齐到统一 canonical 坐标系",
    )
    sp.add_argument("-i", "--input", required=True,
                    help="输入根路径(在其下按 -L 层级寻找含 transforms 的目标目录)")
    sp.add_argument("-L", "--level", type=int, default=1,
                    help="处理层级(1-based,最小 1):1=输入目录本身,2=其直接子目录,... (默认: 1)")
    sp.add_argument("--camera-ids", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6],
                    dest="camera_ids", help="参与拟合/求解的相机编号 (默认: 1-6)")
    sp.add_argument("--plane-cam", type=int, default=1, dest="plane_cam",
                    help="该相机环心->原点、环面->XOY 平面 (默认: 1)")
    sp.add_argument("--azimuth-cam", type=int, default=1, dest="azimuth_cam",
                    help="用该相机的 angle0 光心定 +X 方位 (默认: 1)")
    sp.add_argument("--flip-z", action="store_true", dest="flip_z",
                    help="翻转 +Z 朝向(默认 cam06 在上、cam01 在下;不满意用此反向)")
    sp.add_argument("--transforms-name", default="transforms.json", dest="transforms_name",
                    help="源 transforms 文件名 (默认: transforms.json)")
    sp.add_argument("--output-name", default="transforms_canonical.json", dest="output_name",
                    help="输出文件名,不覆盖原文件 (默认: transforms_canonical.json)")
    sp.add_argument("-rl", "--calib", default=None,
                    help="标定目录或 rig_calib.json 文件;给定则只解尺度(任意相机数可用)")
    sp.add_argument("--dry-run", action="store_true",
                    help="只打印将要进行的操作,不写盘")
    sp.set_defaults(func=_cmd_canonicalize)

    # ── calibrate_rig ──
    sp = sub.add_parser(
        "calibrate_rig",
        help="用相机齐全的好日期联合标定机械常数 r、L,导出 rig_calib.json",
    )
    sp.add_argument("-i", "--input", required=True,
                    help="输入根路径(在其下按 -L 层级寻找标定日目录)")
    sp.add_argument("-L", "--level", type=int, default=1,
                    help="处理层级(1-based,最小 1):1=输入目录本身,2=其直接子目录,... (默认: 1)")
    sp.add_argument("-o", "--output", required=True,
                    help="标定输出目录(写入 rig_calib.json)")
    sp.add_argument("--camera-ids", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6],
                    dest="camera_ids", help="参与的相机编号 (默认: 1-6)")
    sp.add_argument("--min-cams", type=int, default=6, dest="min_cams",
                    help="标定日所需最少相机数 (默认: 6)")
    sp.add_argument("--transforms-name", default="transforms.json", dest="transforms_name",
                    help="源 transforms 文件名 (默认: transforms.json)")
    sp.add_argument("--output-name", default="rig_calib.json", dest="output_name",
                    help="标定文件名 (默认: rig_calib.json)")
    sp.add_argument("--dry-run", action="store_true",
                    help="只打印,不写盘")
    sp.set_defaults(func=_cmd_calibrate_rig)

    # ── finalize ──
    sp = sub.add_parser(
        "finalize",
        help="收尾:把 transforms_canonical.json 提升为 transforms.json,其余 transforms json 归档到 trans/",
    )
    sp.add_argument("-i", "--input", required=True,
                    help="输入根路径(在其下按 -L 层级寻找目标目录)")
    sp.add_argument("-L", "--level", type=int, default=1,
                    help="处理层级(1-based,最小 1):1=输入目录本身,2=其直接子目录,... (默认: 1)")
    sp.add_argument("--canonical-name", default="transforms_canonical.json", dest="canonical_name",
                    help="要保留并提升的文件名 (默认: transforms_canonical.json)")
    sp.add_argument("--final-name", default="transforms.json", dest="final_name",
                    help="提升后的最终文件名 (默认: transforms.json)")
    sp.add_argument("--archive-dir", default="trans", dest="archive_dir",
                    help="归档子目录名 (默认: trans)")
    sp.add_argument("--pattern", default="transforms*.json",
                    help="要归档的文件匹配(canonical 自身除外) (默认: transforms*.json)")
    sp.add_argument("--dry-run", action="store_true",
                    help="只打印将要进行的操作,不动盘")
    sp.set_defaults(func=_cmd_finalize)

    # ── set_aabb ──
    sp = sub.add_parser(
        "set_aabb",
        help="批量写入/修改 transforms.json 的 aabb 与 aabb_scale 字段",
    )
    sp.add_argument("-i", "--input", required=True,
                    help="输入根路径(在其下按 -L 层级寻找目标目录)")
    sp.add_argument("-L", "--level", type=int, default=1,
                    help="处理层级(1-based,最小 1):1=输入目录本身,2=其直接子目录,... (默认: 1)")
    sp.add_argument("--aabb", type=float, nargs=6,
                    metavar=("MINX", "MINY", "MINZ", "MAXX", "MAXY", "MAXZ"),
                    default=[-200, -200, -200, 200, 200, 500],
                    help="包围盒 6 个数:min xyz + max xyz (默认: -200 -200 -200 200 200 500)")
    sp.add_argument("--aabb-scale", type=float, default=1, dest="aabb_scale",
                    help="aabb_scale 值 (默认: 1)")
    sp.add_argument("--transforms-name", default="transforms.json", dest="transforms_name",
                    help="目标文件名 (默认: transforms.json)")
    sp.add_argument("--dry-run", action="store_true",
                    help="只打印将要进行的操作,不写盘")
    sp.set_defaults(func=_cmd_set_aabb)

    # ── check(重复性诊断,只读) ──
    sp = sub.add_parser(
        "check",
        help="重复性诊断(只读):扣掉全局相似变换后各次采集之间还剩多少残差",
    )
    sp.add_argument("-i", "--input", required=True,
                    help="输入根路径(在其下按 -L 层级寻找含 transforms 的目标目录)")
    sp.add_argument("-L", "--level", type=int, default=1,
                    help="处理层级(1-based,最小 1):1=输入目录本身,2=其直接子目录,... (默认: 1)")
    sp.add_argument("-r", "--reference", default=None,
                    help="参考目录名(按目录名匹配);缺省用排序后的第一个")
    sp.add_argument("--mm-per-unit", type=float, default=1.0, dest="mm_per_unit",
                    help="把残差换算成 mm 的系数(参考单位->mm),仅用于展示 (默认: 1.0)")
    sp.add_argument("--camera-ids", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6],
                    dest="camera_ids", help="参与配对的相机编号 (默认: 1-6)")
    sp.add_argument("--transforms-name", default="transforms.json", dest="transforms_name",
                    help="源 transforms 文件名 (默认: transforms.json)")
    sp.set_defaults(func=_cmd_check)

    return p


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
