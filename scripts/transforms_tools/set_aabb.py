#!/usr/bin/env python3
"""
set_aabb —— 批量写入/修改 transforms.json 的 aabb 与 aabb_scale(第四步)

场景:
    finalize 之后每个目录里只剩一份 transforms.json。训练前常需要给它设定
    场景包围盒 aabb 和 aabb_scale,例如:
        "aabb": [[-200, -200, -200], [200, 200, 500]],
        "aabb_scale": 1

对每个目标目录的处理:
    1. 没有 transforms.json 的目录跳过。
    2. 读入 transforms.json,写入(覆盖)aabb 与 aabb_scale 字段,其余保持不变。

本模块只提供核心函数 set_aabb();命令行入口见同包 cli.py。
"""

import json
import os

from .use_existed_trans import list_dirs_at_depth

DEFAULT_AABB = [[-200, -200, -200], [200, 200, 500]]
DEFAULT_AABB_SCALE = 1


def _tidy_num(x):
    """整数值的 float 转成 int,JSON 更干净(200.0 -> 200)。"""
    f = float(x)
    return int(f) if f.is_integer() else f


def _normalize_aabb(aabb):
    """接受 [minx,miny,minz,maxx,maxy,maxz] 或 [[...],[...]],统一成 [[3],[3]]。"""
    flat = []
    for item in aabb:
        if isinstance(item, (list, tuple)):
            flat.extend(item)
        else:
            flat.append(item)
    if len(flat) != 6:
        raise ValueError(f"aabb 需要 6 个数(min xyz + max xyz),收到 {len(flat)} 个: {aabb}")
    nums = [_tidy_num(v) for v in flat]
    return [nums[:3], nums[3:]]


def set_aabb(input_root, level=1, aabb=None, aabb_scale=DEFAULT_AABB_SCALE,
             transforms_name="transforms.json", dry_run=False):
    """给 input_root 第 level 层(1-based)各目录的 transforms.json 写入 aabb/aabb_scale。

    参数:
        input_root      : 输入根路径
        level           : 处理层级(1-based,1=输入目录本身,2=直接子目录,...)
        aabb            : [minx,miny,minz,maxx,maxy,maxz] 或 [[..],[..]];None 用默认
        aabb_scale      : aabb_scale 值
        transforms_name : 目标文件名(默认 transforms.json)
        dry_run         : 只打印,不写盘
    """
    aabb = _normalize_aabb(aabb if aabb is not None else DEFAULT_AABB)
    aabb_scale = _tidy_num(aabb_scale)

    level = max(int(level), 1)
    targets = list_dirs_at_depth(input_root, level - 1)
    print(f"[INFO] 在 {input_root} 第 {level} 层(1=自身)找到 {len(targets)} 个目标目录")
    print(f"[INFO] aabb={aabb}  aabb_scale={aabb_scale}  "
          f"文件={transforms_name}  dry_run={dry_run}\n")

    processed = 0
    for tgt in targets:
        name = os.path.basename(os.path.normpath(tgt))
        tpath = os.path.join(tgt, transforms_name)
        if not os.path.isfile(tpath):
            print(f"[SKIP] {name}: 无 {transforms_name}")
            continue

        with open(tpath, "r") as f:
            data = json.load(f)

        old_aabb = data.get("aabb", "(无)")
        old_scale = data.get("aabb_scale", "(无)")
        data["aabb"] = aabb
        data["aabb_scale"] = aabb_scale

        if dry_run:
            print(f"[DRY ] {name}: aabb {old_aabb} -> {aabb} | "
                  f"aabb_scale {old_scale} -> {aabb_scale}")
        else:
            with open(tpath, "w") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"[OK ] {name}: aabb={aabb}  aabb_scale={aabb_scale}")
        processed += 1

    print(f"\n[DONE] 处理 {processed} 个目录"
          f"{' (dry-run, 未写盘)' if dry_run else ''}")
    return processed
