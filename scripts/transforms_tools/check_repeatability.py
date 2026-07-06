#!/usr/bin/env python3
"""
check_repeatability —— 采集重复性诊断(只读)

背景
----
设备为「相机动、植物不动」,各相机绕中心走固定轨迹。若机械完全可复用,则两次
拍摄的 SfM 位姿之间应「只差一个全局相似变换」(R, t, s);而 NeRF 重建对全局相似
变换不变 —— 即位姿可直接复用。

本命令以一个目录为参考,对其余每个目录用 Umeyama 相似变换对齐(按文件名配对相机),
然后报告扣掉该相似变换后的残差:
    - 位置残差 RMS(参考单位 / 估算 mm / 占环半径百分比)
    - 朝向残差(每台相机旋转角差,度)
    - 目录间尺度比 s(SfM 尺度漂移)

判读
----
位置残差(占比)很小 + 朝向残差零点几度  -> 纯 gauge 差异,位姿可放心复用。
残差明显                                  -> 真实非重复(碰相机/对焦/轨迹漂),需各自重算。

本模块只读,不修改任何文件;命令行入口见同包 cli.py(子命令 check)。
"""

import os
import re

import numpy as np

from .use_existed_trans import list_dirs_at_depth

CAM_RE = re.compile(r"cam0*(\d+)", re.IGNORECASE)


def load_pose_map(json_path, camera_ids):
    """返回 {basename: (center(3,), R(3,3))},只保留 camera_ids 里的相机。"""
    import json
    with open(json_path, "r") as f:
        data = json.load(f)
    out = {}
    for fr in data.get("frames", []):
        base = os.path.basename(fr.get("file_path", ""))
        m = CAM_RE.search(base)
        if not m or int(m.group(1)) not in camera_ids:
            continue
        M = np.array(fr["transform_matrix"], dtype=np.float64)
        out[base] = (M[:3, 3], M[:3, :3])
    return out


def umeyama(src, dst):
    """求 s,R,t 使 s*R*src + t ≈ dst。src,dst: (N,3)。返回 (s,R,t)。"""
    n = src.shape[0]
    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    src_c = src - mu_s
    dst_c = dst - mu_d
    cov = (dst_c.T @ src_c) / n
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1
    R = U @ S @ Vt
    var_s = (src_c ** 2).sum() / n
    s = np.trace(np.diag(D) @ S) / var_s
    t = mu_d - s * R @ mu_s
    return s, R, t


def rot_angle_deg(R):
    """旋转矩阵对应的旋转角(度)。"""
    c = (np.trace(R) - 1.0) / 2.0
    c = max(-1.0, min(1.0, c))
    return float(np.degrees(np.arccos(c)))


def check_repeatability(input_root, level=1, reference=None, mm_per_unit=1.0,
                        camera_ids=(1, 2, 3, 4, 5, 6),
                        transforms_name="transforms.json"):
    """对 input_root 第 level 层(1-based,1=自身)各目录的 transforms.json 做重复性诊断。"""
    camera_ids = tuple(camera_ids)
    unit = "mm"

    level = max(int(level), 1)
    dirs = list_dirs_at_depth(input_root, level - 1)
    pairs = []  # (date_name, transforms_path)
    for d in dirs:
        p = os.path.join(d, transforms_name)
        if os.path.isfile(p):
            pairs.append((os.path.basename(os.path.normpath(d)), p))
    pairs.sort(key=lambda x: x[0])

    if len(pairs) < 2:
        print(f"[ERR] 至少需要 2 个含 {transforms_name} 的目录,找到 {len(pairs)} 个")
        return []

    # 选参考目录
    ref_idx = 0
    if reference:
        for i, (name, _) in enumerate(pairs):
            if name == reference:
                ref_idx = i
                break
        else:
            print(f"[WARN] 未找到参考目录 '{reference}',改用排序后的第一个 '{pairs[0][0]}'")
    ref_name, ref_path = pairs[ref_idx]

    ref_map = load_pose_map(ref_path, camera_ids)
    if len(ref_map) < 3:
        print(f"[ERR] 参考 {ref_name} 可用相机不足({len(ref_map)})")
        return []
    ref_centers = np.array([c for c, _ in ref_map.values()])
    ref_radius = float(np.linalg.norm(ref_centers - ref_centers.mean(axis=0), axis=1).mean())

    print(f"[INFO] 参考: {ref_name}  ({len(ref_map)} 帧, 环特征半径={ref_radius:.4f} 参考单位)")
    print(f"[INFO] 换算: 1 参考单位 ≈ {mm_per_unit} {unit}\n")
    print(f"{'目录':<12} {'配对':>4} {'尺度比s':>9} {'位置RMS(单位)':>13} "
          f"{'位置RMS(mm)':>12} {'占环半径%':>9} {'朝向(度)均/最大':>16}")
    print("-" * 84)

    rows = []
    for name, p in pairs:
        if p == ref_path:
            continue
        m = load_pose_map(p, camera_ids)
        common = sorted(set(ref_map) & set(m))
        if len(common) < 3:
            print(f"{name:<12} 仅 {len(common)} 个公共帧,跳过")
            continue

        src = np.array([m[k][0] for k in common])
        dst = np.array([ref_map[k][0] for k in common])
        s, R, t = umeyama(src, dst)

        aligned = (s * (R @ src.T).T) + t
        pos_res = np.linalg.norm(aligned - dst, axis=1)
        rms = float(np.sqrt(np.mean(pos_res ** 2)))
        rms_mm = rms * mm_per_unit
        pct = 100.0 * rms / ref_radius

        ang = [rot_angle_deg(R @ m[k][1] @ ref_map[k][1].T) for k in common]
        ang_mean, ang_max = float(np.mean(ang)), float(np.max(ang))

        rows.append((name, rms_mm, pct, ang_mean, ang_max, s))
        print(f"{name:<12} {len(common):>4} {s:>9.4f} {rms:>13.5f} "
              f"{rms_mm:>12.2f} {pct:>9.2f} {ang_mean:>7.3f}/{ang_max:<7.3f}")

    if rows:
        rms_all = np.array([r[1] for r in rows])
        pct_all = np.array([r[2] for r in rows])
        ang_all = np.array([r[3] for r in rows])
        s_all = np.array([r[5] for r in rows])
        print("-" * 84)
        print(f"[汇总] {len(rows)} 个目录 vs 参考 {ref_name}")
        print(f"  位置残差:  中位 {np.median(rms_all):.2f}{unit}  最大 {rms_all.max():.2f}{unit}  "
              f"(占环半径 中位 {np.median(pct_all):.2f}%  最大 {pct_all.max():.2f}%)")
        print(f"  朝向残差:  中位 {np.median(ang_all):.3f}°  最大 {np.array([r[4] for r in rows]).max():.3f}°")
        print(f"  SfM尺度比: {s_all.min():.4f} ~ {s_all.max():.4f} (越接近常数=尺度越稳)")
        print()
        print(f"[判读] 残差占比很小(亚%)+朝向零点几度 => 纯gauge,位姿可复用;")
        print(f"       残差明显 => 真实非重复(碰相机/对焦/轨迹漂),建议各目录重算。")
    return rows
