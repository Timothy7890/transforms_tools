#!/usr/bin/env python3
"""
use_existed_trans —— 复用已有的 transforms.json

场景:
    某个目录(-t 来源)已经算好了 transforms.json(例如 Metashape 解出的位姿),
    想把它复用到一批新目录(-i 下、深度 -L 处,每个目录里有自己的 images/)。

对每个目标目录的处理:
    1. 若目标目录已存在 transforms.json,则把它「更名保留」为
       transforms_<method>.json,并在其内部写入:
           "method":    <-m 指定,默认 "Metashape">
           "come_from":  目标目录自身的上级文件夹名称
    2. 读取来源 transforms.json(-t),按目标目录 images/ 里实际存在的图片
       过滤 frames(删除指向不存在图片的帧),并写入:
           "come_from":  来源所在的文件夹名称(它来自哪里)
       然后写成目标目录的新 transforms.json。

本模块只提供核心函数 use_existed_trans();命令行入口见同包 cli.py。
"""

import json
import os
from datetime import datetime


def list_dirs_at_depth(root, depth):
    """返回 root 下「正好」位于给定深度的所有目录(depth=0 即 root 本身)。"""
    root = os.path.abspath(root)
    if depth <= 0:
        return [root]
    current = [root]
    for _ in range(depth):
        nxt = []
        for d in current:
            try:
                for name in sorted(os.listdir(d)):
                    sub = os.path.join(d, name)
                    if os.path.isdir(sub):
                        nxt.append(sub)
            except OSError:
                continue
        current = nxt
    return current


def image_basenames(images_dir):
    """目标 images/ 目录下所有图片文件的 basename 集合(小写后缀判断)。"""
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    if not os.path.isdir(images_dir):
        return set()
    return {
        f for f in os.listdir(images_dir)
        if f.lower().endswith(exts)
    }


def unique_path(path):
    """若 path 已存在,追加 _1/_2... 直到不冲突。"""
    if not os.path.exists(path):
        return path
    stem, ext = os.path.splitext(path)
    k = 1
    while os.path.exists(f"{stem}_{k}{ext}"):
        k += 1
    return f"{stem}_{k}{ext}"


def filter_frames(src_data, image_set):
    """只保留 file_path 的 basename 出现在 image_set 里的 frame。"""
    frames = src_data.get("frames", [])
    kept, dropped = [], []
    for fr in frames:
        base = os.path.basename(fr.get("file_path", ""))
        if base in image_set:
            kept.append(fr)
        else:
            dropped.append(base)
    return kept, dropped


def use_existed_trans(input_root, src_transforms, level=1, method="Metashape",
                      images_subdir="images", dry_run=False):
    """把 src_transforms 复用到 input_root 第 level 层(1-based,1=自身)的各目标目录。"""
    if not os.path.isfile(src_transforms):
        raise FileNotFoundError(f"来源 transforms.json 不存在: {src_transforms}")

    with open(src_transforms, "r") as f:
        src_data = json.load(f)

    # 来源「它来自哪里」= 来源 transforms.json 的完整路径
    src_come_from = os.path.abspath(src_transforms)

    # 本次运行的时间戳,用于被保留原文件的命名(method 已写在 JSON 内部)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # level 为 1-based:1=输入目录本身,2=其直接子目录,...(内部深度 = level-1)
    level = max(int(level), 1)
    targets = list_dirs_at_depth(input_root, level - 1)
    print(f"[INFO] 来源: {src_transforms} (come_from={src_come_from})")
    print(f"[INFO] 在 {input_root} 第 {level} 层(1=自身)找到 {len(targets)} 个目标目录")
    print(f"[INFO] method={method}  dry_run={dry_run}\n")

    processed = 0
    for tgt in targets:
        images_dir = os.path.join(tgt, images_subdir)
        img_set = image_basenames(images_dir)
        if not img_set:
            print(f"[SKIP] 无 {images_subdir}/ 或无图片: {tgt}")
            continue

        tgt_name = os.path.basename(os.path.normpath(tgt))
        old_path = os.path.join(tgt, "transforms.json")

        # 1) 更名保留已有的 transforms.json
        if os.path.isfile(old_path):
            with open(old_path, "r") as f:
                old_data = json.load(f)
            old_come_from = os.path.abspath(old_path)  # 自身完整路径
            old_data["method"] = method
            old_data["come_from"] = old_come_from
            preserved = unique_path(os.path.join(tgt, f"transforms_{stamp}.json"))
            if dry_run:
                print(f"[DRY ] 保留原文件 -> {os.path.relpath(preserved, tgt)} "
                      f"(+method={method}, +come_from={old_come_from})")
            else:
                with open(preserved, "w") as f:
                    json.dump(old_data, f, indent=2, ensure_ascii=False)
                print(f"[KEEP] 原 transforms.json -> {os.path.basename(preserved)} "
                      f"(+method={method}, +come_from={old_come_from})")
        else:
            print(f"[NOTE] 目标无原 transforms.json,跳过保留步骤: {tgt}")

        # 2) 过滤来源 frames 后写入新的 transforms.json
        new_data = dict(src_data)
        kept, dropped = filter_frames(src_data, img_set)
        new_data["frames"] = kept
        new_data["come_from"] = src_come_from

        n_src = len(src_data.get("frames", []))
        msg = (f"frames: 来源 {n_src} -> 保留 {len(kept)} (删除 {len(dropped)}) "
               f"| 目标图片 {len(img_set)} 张")
        if dry_run:
            print(f"[DRY ] 写 transforms.json @ {tgt_name}  {msg}\n")
        else:
            with open(old_path, "w") as f:
                json.dump(new_data, f, indent=2, ensure_ascii=False)
            print(f"[WRITE] transforms.json @ {tgt_name}  {msg}")
            print(f"        (+come_from={src_come_from})\n")
        processed += 1

    print(f"[DONE] 处理完成,共 {processed} 个目标目录"
          f"{' (dry-run, 未写盘)' if dry_run else ''}")
    return processed
