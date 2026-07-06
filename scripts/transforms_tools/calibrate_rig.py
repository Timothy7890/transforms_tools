#!/usr/bin/env python3
"""
calibrate_rig —— 用多个「好日期」(相机齐全)联合标定机械常数 r、L,并导出每台相机
光心到转轴的真实半径 R_opt_mm(机械常数,可被 canonicalize 复用)。

模型(与 canonicalize 一致)
----
每个标定日 d、每台相机 i:
    r_sfm(d,i) = s_d * sqrt( (r + disp_i + a_(d,i)·L)^2 + (b_(d,i)·L)^2 )
其中:
    r, L=(dx,dy,dz) : 机械常数(全体日期共享)        <- 标定目标
    s_d             : 第 d 天的 SfM 任意尺度(每天一个)<- 顺带解
    a,b,disp,r_sfm  : 各日各相机的已知量(几何拟合得到)

联合最小二乘:未知量 = 4 + D(D=标定天数),方程 = Σ 各天相机数,通常远超定。
解完后用各天均值的 a_i,b_i 计算 R_opt_mm[i],写入 rig_calib.json。

只把「相机数 >= --min-cams(默认 6)」的日期用于标定。
"""

import glob
import json
import os
from datetime import datetime

import numpy as np

from .use_existed_trans import list_dirs_at_depth
from .canonicalize import load_disp, load_frames, per_camera_geometry, r_opt_mm


def calibrate_rig(input_root, output_dir, level=1, camera_ids=(1, 2, 3, 4, 5, 6),
                  min_cams=6, transforms_name="transforms.json",
                  output_name="rig_calib.json", dry_run=False):
    """联合标定 r、L,并导出 R_opt_mm 到 output_dir/output_name。"""
    from scipy.optimize import least_squares

    camera_ids = list(camera_ids)
    level = max(int(level), 1)
    targets = list_dirs_at_depth(input_root, level - 1)

    # 收集合格标定日的几何
    days = []
    for tgt in targets:
        tpath = os.path.join(tgt, transforms_name)
        if not os.path.isfile(tpath):
            continue
        name = os.path.basename(os.path.normpath(tgt))
        disp, disp_src = load_disp(tgt, camera_ids)
        _, groups = load_frames(tpath, camera_ids)
        present = [c for c in camera_ids if c in groups]
        if len(present) < min_cams:
            print(f"[skip] {name}: cam={len(present)} < {min_cams}")
            continue
        geo = per_camera_geometry(groups, disp, camera_ids)
        days.append({"name": name, "geo": geo, "disp_src": disp_src})
        print(f"[use ] {name}: cam={len(present)} disp={disp_src}")

    if len(days) < 1:
        raise SystemExit(f"没有相机数 >= {min_cams} 的标定日")

    D = len(days)
    print(f"\n[INFO] 用 {D} 个标定日联合求解 r, L\n")

    # 组装残差:未知量 x = [r, dx, dy, dz, s_1..s_D]
    def residuals(x):
        r_base = x[0]
        L = x[1:4]
        s = x[4:4 + D]
        res = []
        for di, day in enumerate(days):
            geo = day["geo"]
            for cid in geo["ids"]:
                R = r_opt_mm(geo["disp"][cid], geo["a"][cid], geo["b"][cid], r_base, L)
                res.append(s[di] * R - geo["radius"][cid])
        return np.array(res)

    # 初值:每天线性拟合 r_sfm = s*(r+disp) 取 s0,r0;L=0
    s0_list, r0_list = [], []
    for day in days:
        geo = day["geo"]
        disp_arr = np.array([geo["disp"][c] for c in geo["ids"]], dtype=float)
        r_obs = np.array([geo["radius"][c] for c in geo["ids"]])
        A_lin = np.vstack([disp_arr, np.ones_like(disp_arr)]).T
        (sd, bd), *_ = np.linalg.lstsq(A_lin, r_obs, rcond=None)
        s0_list.append(sd if abs(sd) > 1e-12 else 1e-3)
        r0_list.append(bd / sd if abs(sd) > 1e-12 else 50.0)
    x0 = np.array([float(np.median(r0_list)), 0.0, 0.0, 0.0] + s0_list)

    sol = least_squares(residuals, x0, method="lm")
    r_base = float(sol.x[0])
    L = sol.x[1:4]
    s_days = sol.x[4:4 + D]
    res = residuals(sol.x)

    # 各天均值的 a_i,b_i -> R_opt_mm[i]
    all_ids = sorted({c for day in days for c in day["geo"]["ids"]})
    R_opt = {}
    a_mean, b_mean, disp_mean = {}, {}, {}
    for cid in all_ids:
        a_acc = np.mean([day["geo"]["a"][cid] for day in days if cid in day["geo"]["a"]], axis=0)
        b_acc = np.mean([day["geo"]["b"][cid] for day in days if cid in day["geo"]["b"]], axis=0)
        d_acc = np.mean([day["geo"]["disp"][cid] for day in days if cid in day["geo"]["disp"]])
        a_mean[cid], b_mean[cid], disp_mean[cid] = a_acc, b_acc, d_acc
        R_opt[cid] = r_opt_mm(d_acc, a_acc, b_acc, r_base, L)

    # 残差(按天换算 mm:用该天 s_d)
    rms_unit = float(np.sqrt(np.mean(res ** 2)))
    rms_mm = float(np.sqrt(np.mean((res / np.median(s_days)) ** 2)))

    print(f"[SOLVE] success={sol.success}")
    print(f"  r(电机0位偏移) = {r_base:.2f} mm")
    print(f"  L(杠杆臂)      = ({L[0]:+.2f}, {L[1]:+.2f}, {L[2]:+.2f}) mm  |L|={np.linalg.norm(L):.2f}")
    print(f"  残差 RMS       ≈ {rms_mm:.3f} mm")
    print(f"  各天 1/s       = " + ", ".join(f"{d['name']}:{1/sd:.2f}" for d, sd in zip(days, s_days)))
    print(f"\n  R_opt_mm(光心到轴真实半径,mm):")
    for cid in all_ids:
        print(f"    cam{cid:02d}: disp={disp_mean[cid]:.0f}  R_opt={R_opt[cid]:.2f} mm")

    calib = {
        "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "r_base_mm": r_base,
        "L_mm": [float(L[0]), float(L[1]), float(L[2])],
        "R_opt_mm": {str(cid): R_opt[cid] for cid in all_ids},
        "disp": {str(cid): disp_mean[cid] for cid in all_ids},
        "calib_days": [d["name"] for d in days],
        "rms_mm": rms_mm,
        "units": "mm",
        "note": "R_opt_mm 为光心到转轴真实半径,机械常数;canonicalize --calib 复用之只解尺度。",
    }

    if dry_run:
        print(f"\n[DRY ] 将写: {os.path.join(output_dir, output_name)}(未写盘)")
        return calib

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, output_name)
    with open(out_path, "w") as f:
        json.dump(calib, f, indent=2, ensure_ascii=False)
    print(f"\n[DONE] 标定已写入: {out_path}")
    return calib
