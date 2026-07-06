#!/usr/bin/env python3
"""
finalize —— canonicalize 之后的收尾整理(第三步)

场景:
    canonicalize 跑完后,每个目录里通常会有一堆 transforms 相关的 json:
        transforms.json                (原始/复用的位姿)
        transforms_<时间戳>.json        (use_existed_trans 备份的原文件)
        transforms_canonical.json      (canonicalize 输出的统一坐标系版本)
    我们最终只想留下「统一坐标系」那一份,并让它就叫 transforms.json,
    其余 transforms 相关 json 全部归档进子目录,保留可追溯但不碍事。

对每个目标目录的处理:
    1. 若目录里没有 transforms_canonical.json,则跳过(没有可提升的成果)。
    2. 建立归档子目录 trans/(可用 --archive-dir 改名)。
    3. 把匹配 --pattern(默认 transforms*.json)的所有 json 移入 trans/,
       但 **排除 transforms_canonical.json 本身**;若有同名冲突则自动加 _1/_2…。
    4. 把 transforms_canonical.json 改名为 transforms.json(留在目录下)。

本模块只提供核心函数 finalize();命令行入口见同包 cli.py。
"""

import glob
import os
import shutil

from .use_existed_trans import list_dirs_at_depth, unique_path


def finalize(input_root, level=1,
             canonical_name="transforms_canonical.json",
             final_name="transforms.json",
             archive_dir="trans",
             pattern="transforms*.json",
             dry_run=False):
    """把各目标目录里 canonicalize 的成果提升为 transforms.json,其余归档到 trans/。

    参数:
        input_root    : 输入根路径
        level         : 处理层级(1-based,1=输入目录本身,2=直接子目录,...)
        canonical_name: 要保留并提升的文件名(默认 transforms_canonical.json)
        final_name    : 提升后的最终文件名(默认 transforms.json)
        archive_dir   : 归档子目录名(默认 trans)
        pattern       : 要归档的文件匹配(默认 transforms*.json)
        dry_run       : 只打印,不动盘
    """
    level = max(int(level), 1)
    targets = list_dirs_at_depth(input_root, level - 1)
    print(f"[INFO] 在 {input_root} 第 {level} 层(1=自身)找到 {len(targets)} 个目标目录")
    print(f"[INFO] 保留并提升: {canonical_name} -> {final_name}")
    print(f"[INFO] 其余匹配 '{pattern}' 的 json 归档到 ./{archive_dir}/  dry_run={dry_run}\n")

    processed = 0
    for tgt in targets:
        name = os.path.basename(os.path.normpath(tgt))
        canon = os.path.join(tgt, canonical_name)
        if not os.path.isfile(canon):
            print(f"[SKIP] {name}: 无 {canonical_name},跳过")
            continue

        archive = os.path.join(tgt, archive_dir)

        # 收集待归档文件:匹配 pattern 的 json,排除 canonical 本身
        candidates = sorted(glob.glob(os.path.join(tgt, pattern)))
        to_move = [p for p in candidates
                   if os.path.isfile(p) and os.path.basename(p) != canonical_name]

        # 若最终名(transforms.json)当前已存在且未被 pattern 命中,也一并归档,
        # 否则下面改名时会覆盖它
        final_path = os.path.join(tgt, final_name)
        if (os.path.isfile(final_path)
                and os.path.basename(final_path) != canonical_name
                and final_path not in to_move):
            to_move.append(final_path)

        # 1) 归档其余 transforms 相关 json
        moved = 0
        for src in to_move:
            dst = unique_path(os.path.join(archive, os.path.basename(src)))
            if dry_run:
                print(f"[DRY ] {name}: 移动 {os.path.basename(src)} "
                      f"-> {archive_dir}/{os.path.basename(dst)}")
            else:
                os.makedirs(archive, exist_ok=True)
                shutil.move(src, dst)
            moved += 1

        # 2) 提升 canonical -> final
        if dry_run:
            print(f"[DRY ] {name}: 重命名 {canonical_name} -> {final_name} "
                  f"(归档 {moved} 个)\n")
        else:
            shutil.move(canon, final_path)
            print(f"[OK ] {name}: {canonical_name} -> {final_name}  "
                  f"(归档 {moved} 个到 {archive_dir}/)")
        processed += 1

    print(f"\n[DONE] 处理 {processed} 个目录"
          f"{' (dry-run, 未动盘)' if dry_run else ''}")
    return processed
