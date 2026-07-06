#!/usr/bin/env python3
"""
canonicalize —— 把多个 transforms.json 各自「解尺度 -> 缩放 -> 旋转平移到统一 canonical 坐标系」。

设备:相机动、植物不动,各相机绕中心走固定圆轨迹。
对每个目标目录(各含一份 transforms.json + 含 CameraDisplacement 的配置 json):
    1) 用杠杆臂方程解出该目录自己的尺度 s(SfM单位/mm),缩放因子 c = 1/s。
    2) 拟合各相机圆 -> 转轴方向 a、基准相机环心 O1。
    3) 构造相似变换,把位姿摆到 canonical:
         - 转轴 a            -> 世界 +Z
         - 基准相机环心 O1   -> 原点(转轴落在 Z 轴上、基准相机圆面落在 XOY z=0)
         - 基准相机 angle0 光心 -> +X 方向(绕 Z 定方位)
       位姿变换(只缩放/旋转/平移,不改内参,不加杠杆臂 L):
         new_R = R · M_R
         new_t = c · R · (M_t - O1)   (单位变 mm)
    4) 写出 canonical transforms(默认 transforms_canonical.json,不覆盖原文件)。

依赖: numpy, scipy
"""

import glob
import json
import os
import re

import numpy as np

from .use_existed_trans import list_dirs_at_depth

CAM_RE = re.compile(r"cam0*(\d+)", re.IGNORECASE)
ANGLE_RE = re.compile(r"angle0*(\d+)", re.IGNORECASE)

DISP_DEFAULT = {1: 540.0, 2: 470.0, 3: 400.0, 4: 300.0, 5: 200.0, 6: 0.0}


def load_disp(folder, camera_ids):
    """从目录里含 CameraParametersList 的配置 json 读 disp;失败则用默认。"""
    for jp in sorted(glob.glob(os.path.join(folder, "*.json"))):
        base = os.path.basename(jp).lower()
        if base.startswith("transforms") or base.startswith("plantsegnerf"):
            continue
        try:
            with open(jp, "r") as f:
                data = json.load(f)
        except Exception:
            continue
        cams = data.get("CameraParametersList")
        if isinstance(cams, list) and cams:
            disp = {}
            for idx, cam in enumerate(cams):
                cid = idx + 1
                if cid in camera_ids and "CameraDisplacement" in cam:
                    disp[cid] = float(cam["CameraDisplacement"])
            if all(c in disp for c in camera_ids):
                return disp, os.path.basename(jp)
    return {c: DISP_DEFAULT.get(c, 0.0) for c in camera_ids}, "(default)"


def load_frames(json_path, camera_ids):
    """返回 (data, groups),groups[cid] = list of (basename, angle, C(3,), Q(3,3))。"""
    with open(json_path, "r") as f:
        data = json.load(f)
    groups = {cid: [] for cid in camera_ids}
    for fr in data.get("frames", []):
        base = os.path.basename(fr.get("file_path", ""))
        m = CAM_RE.search(base)
        if not m:
            continue
        cid = int(m.group(1))
        if cid not in groups:
            continue
        am = ANGLE_RE.search(base)
        angle = int(am.group(1)) if am else 0
        M = np.array(fr["transform_matrix"], dtype=np.float64)
        groups[cid].append((base, angle, M[:3, 3], M[:3, :3]))
    return data, {cid: g for cid, g in groups.items() if g}


def fit_circle_3d(points):
    """3D 圆拟合,返回 (半径, 圆心3D, 法向)。"""
    centroid = points.mean(axis=0)
    centered = points - centroid
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    bx, by, normal = vh[0], vh[1], vh[2]
    x = centered @ bx
    y = centered @ by
    sol, *_ = np.linalg.lstsq(np.c_[2 * x, 2 * y, np.ones_like(x)],
                              x ** 2 + y ** 2, rcond=None)
    cx, cy, c = sol
    r = float(np.sqrt(max(c + cx ** 2 + cy ** 2, 0.0)))
    center3d = centroid + cx * bx + cy * by
    return r, center3d, normal


def _axis_from_normals(ids, normals):
    ref = normals[ids[0]]
    axis = np.zeros(3)
    for cid in ids:
        n = normals[cid]
        axis += -n if np.dot(n, ref) < 0 else n
    return axis / np.linalg.norm(axis)


def per_camera_geometry(groups, disp, camera_ids):
    """逐相机几何量(都是机械可复用量)。

    返回 dict: ids, radius{cid}, a{cid}, b{cid}, disp{cid}。
        radius : SfM 圆半径
        a, b   : 杠杆臂 L 在「径向/切向」上的投影系数(p_rad=a·L, p_tan=b·L)
    """
    ids = [c for c in camera_ids if c in groups]
    radius, centers, normals = {}, {}, {}
    for cid in ids:
        C = np.array([t[2] for t in groups[cid]])
        r, ctr, nrm = fit_circle_3d(C)
        radius[cid], centers[cid], normals[cid] = r, ctr, nrm

    axis = _axis_from_normals(ids, normals)

    a_vec, b_vec = {}, {}
    for cid in ids:
        Cs = np.array([t[2] for t in groups[cid]])
        Qs = np.array([t[3] for t in groups[cid]])
        O = centers[cid]
        a_acc = np.zeros(3)
        b_acc = np.zeros(3)
        cnt = 0
        for ck, Qk in zip(Cs, Qs):
            v = ck - O
            v_perp = v - np.dot(v, axis) * axis
            nrm = np.linalg.norm(v_perp)
            if nrm < 1e-9:
                continue
            u = v_perp / nrm
            tg = np.cross(axis, u)
            a_acc += Qk.T @ u
            b_acc += Qk.T @ tg
            cnt += 1
        a_vec[cid] = a_acc / max(cnt, 1)
        b_vec[cid] = b_acc / max(cnt, 1)

    return {"ids": ids, "radius": radius, "a": a_vec, "b": b_vec,
            "disp": {c: float(disp[c]) for c in ids}}


def r_opt_mm(disp_i, a_i, b_i, r_base, L):
    """由机械常数(r,L)与已知 disp 算光心到轴的真实半径(mm)。"""
    R_conn = r_base + disp_i
    p_rad = float(np.dot(a_i, L))
    p_tan = float(np.dot(b_i, L))
    return float(np.sqrt((R_conn + p_rad) ** 2 + p_tan ** 2))


def solve_scale(groups, disp, camera_ids):
    """自解:用杠杆臂方程解尺度 s(SfM单位/mm)。返回 (s, info)。失败回退线性拟合。"""
    from scipy.optimize import least_squares

    geo = per_camera_geometry(groups, disp, camera_ids)
    ids = geo["ids"]
    A = np.array([geo["a"][c] for c in ids])
    B = np.array([geo["b"][c] for c in ids])
    r_obs = np.array([geo["radius"][c] for c in ids])
    disp_arr = np.array([geo["disp"][c] for c in ids], dtype=float)

    A_lin = np.vstack([disp_arr, np.ones_like(disp_arr)]).T
    (s0, b0), *_ = np.linalg.lstsq(A_lin, r_obs, rcond=None)
    if abs(s0) < 1e-12:
        return None, {"ok": False, "reason": "线性初值退化"}
    r0 = b0 / s0

    def residuals(x):
        s, r_base, L = x[0], x[1], x[2:5]
        R_conn = r_base + disp_arr
        p_rad = A @ L
        p_tan = B @ L
        R_opt = np.sqrt((R_conn + p_rad) ** 2 + p_tan ** 2)
        return s * R_opt - r_obs

    try:
        sol = least_squares(residuals, np.array([s0, r0, 0.0, 0.0, 0.0]), method="lm")
        s = float(sol.x[0])
        rms = float(np.sqrt(np.mean(residuals(sol.x) ** 2)))
        return s, {"ok": True, "linear": False,
                   "rms_mm": rms / s if s else float("nan"), "n_cam": len(ids)}
    except Exception as e:
        return float(s0), {"ok": True, "linear": True, "reason": str(e), "n_cam": len(ids)}


def solve_scale_with_calib(groups, calib, disp, camera_ids):
    """复用标定只解尺度 s,任意相机数可用。

    优先用标定的机械常数 r_base、L,配合「当天实际 disp + 当天朝向」逐相机重算
    R_opt(这样改过 disp/俯仰的日子也能正确缩放);若标定文件缺 r_base/L,则回退
    到按相机号查冻结的 R_opt_mm 表(此回退假设当天为标准 disp 配置)。
    """
    geo = per_camera_geometry(groups, disp, camera_ids)
    r_base = calib.get("r_base_mm")
    L = calib.get("L_mm")
    ratios, used = [], []

    if r_base is not None and L is not None:
        L = np.array(L, dtype=float)
        for cid in geo["ids"]:
            R = r_opt_mm(geo["disp"][cid], geo["a"][cid], geo["b"][cid],
                         float(r_base), L)
            if R <= 0:
                continue
            ratios.append(geo["radius"][cid] / R)
            used.append(cid)
        mode = "r,L+当天disp"
    else:
        R_opt = calib.get("R_opt_mm", {})
        for cid in geo["ids"]:
            Rmm = R_opt.get(str(cid), R_opt.get(cid))
            if Rmm is None or Rmm <= 0:
                continue
            ratios.append(geo["radius"][cid] / float(Rmm))
            used.append(cid)
        mode = "R_opt表"

    if not ratios:
        return None, {"ok": False, "reason": "标定缺少可用机械常数/相机"}
    s = float(np.median(ratios))
    spread = float(np.std(ratios) / np.mean(ratios)) if len(ratios) > 1 else 0.0
    return s, {"ok": True, "calib": True, "n_cam": len(used),
               "scale_spread": spread, "mode": mode}


def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def rotation_align(a, b):
    """返回 R 使 R@a = b(a,b 单位向量)。"""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if c > 1 - 1e-12:
        return np.eye(3)
    if c < -1 + 1e-12:
        perp = np.array([1.0, 0.0, 0.0])
        if abs(a[0]) > 0.9:
            perp = np.array([0.0, 1.0, 0.0])
        perp = perp - np.dot(perp, a) * a
        perp /= np.linalg.norm(perp)
        return 2 * np.outer(perp, perp) - np.eye(3)
    vx = skew(v)
    return np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))


def rot_z(angle_rad):
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def build_canonical(groups, camera_ids, plane_cam, azimuth_cam, flip_z):
    """返回 (R, O1):canonical 旋转、基准相机环心(SfM)。"""
    ids = [c for c in camera_ids if c in groups]
    centers, normals = {}, {}
    for cid in ids:
        C = np.array([t[2] for t in groups[cid]])
        _, ctr, nrm = fit_circle_3d(C)
        centers[cid], normals[cid] = ctr, nrm

    axis = _axis_from_normals(ids, normals)

    pcam = plane_cam if plane_cam in centers else ids[0]
    last_cam = ids[-1]
    O1 = centers[pcam]
    # 默认方向:编号大的相机(如 cam06)在上、基准相机(cam01)在下
    # => +Z 指向 plane_cam -> last_cam 方向(其余相机落在 +Z 侧)
    if np.dot(axis, centers[last_cam] - O1) < 0:
        axis = -axis
    if flip_z:
        axis = -axis

    R1 = rotation_align(axis, np.array([0.0, 0.0, 1.0]))

    acam = azimuth_cam if azimuth_cam in groups else ids[0]
    frames = sorted(groups[acam], key=lambda t: t[1])
    C0 = frames[0][2]
    w = R1 @ (C0 - O1)
    phi = np.arctan2(w[1], w[0])
    R2 = rot_z(-phi)

    return R2 @ R1, O1


def load_calib(calib_path, name="rig_calib.json"):
    """从目录或文件加载标定。返回 dict 或 None。"""
    if not calib_path:
        return None
    path = calib_path
    if os.path.isdir(calib_path):
        path = os.path.join(calib_path, name)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"标定文件不存在: {path}")
    with open(path, "r") as f:
        return json.load(f)


def canonicalize_one(transforms_path, camera_ids, plane_cam, azimuth_cam,
                     flip_z, output_name, dry_run, calib=None):
    folder = os.path.dirname(transforms_path)
    name = os.path.basename(os.path.normpath(folder))
    disp, disp_src = load_disp(folder, camera_ids)
    data, groups = load_frames(transforms_path, camera_ids)
    present = [c for c in camera_ids if c in groups]
    # 有标定时只需 >=1 台;自解时仍需 >=3 台
    min_cam = 1 if calib else 3
    if len(present) < min_cam:
        print(f"[SKIP] {name}: 仅 {len(present)} 台相机,跳过")
        return False

    if calib:
        s, info = solve_scale_with_calib(groups, calib, disp, camera_ids)
    else:
        s, info = solve_scale(groups, disp, camera_ids)
    if s is None or s == 0:
        print(f"[SKIP] {name}: 尺度求解失败 ({info.get('reason')})")
        return False
    c = 1.0 / s

    R, O1 = build_canonical(groups, camera_ids, plane_cam, azimuth_cam, flip_z)

    new_data = dict(data)
    new_frames = []
    for fr in data.get("frames", []):
        M = np.array(fr["transform_matrix"], dtype=np.float64)
        M_R, M_t = M[:3, :3], M[:3, 3]
        Mn = np.eye(4)
        Mn[:3, :3] = R @ M_R
        Mn[:3, 3] = c * (R @ (M_t - O1))
        nf = dict(fr)
        nf["transform_matrix"] = Mn.tolist()
        new_frames.append(nf)
    new_data["frames"] = new_frames
    new_data["canonical"] = True
    new_data["mm_per_unit"] = c
    new_data["disp_source"] = disp_src
    new_data["scale_method"] = "calib" if calib else "self"
    new_data["units"] = "mm"

    if info.get("calib"):
        tag = (f"标定[{info.get('mode', '?')}] "
               f"spread={info.get('scale_spread', 0.0) * 100:.1f}%")
    elif info.get("linear"):
        tag = "线性"
    else:
        tag = f"RMS={info.get('rms_mm', float('nan')):.2f}mm"
    out_path = os.path.join(folder, output_name)
    if dry_run:
        print(f"[DRY ] {name}: cam={len(present)} disp={disp_src} "
              f"1/s={c:.3f}mm/单位 {tag} -> {output_name}")
        return True

    with open(out_path, "w") as f:
        json.dump(new_data, f, indent=2, ensure_ascii=False)
    print(f"[OK ] {name}: cam={len(present)} disp={disp_src} "
          f"1/s={c:.3f}mm/单位 {tag} -> {output_name}")
    return True


def canonicalize(input_root, level=1, camera_ids=(1, 2, 3, 4, 5, 6),
                 plane_cam=1, azimuth_cam=1, flip_z=False,
                 transforms_name="transforms.json",
                 output_name="transforms_canonical.json", dry_run=False,
                 calib_path=None):
    """在 input_root 第 level 层(1-based)的各目录中,对 transforms 做 canonical 对齐。

    calib_path 给定时(目录或 rig_calib.json 文件),只解尺度(任意相机数可用);
    否则每目录自解杠杆臂模型(需 >=6 台才稳)。
    """
    camera_ids = list(camera_ids)
    level = max(int(level), 1)
    calib = load_calib(calib_path)
    targets = list_dirs_at_depth(input_root, level - 1)
    print(f"[INFO] 在 {input_root} 第 {level} 层(1=自身)找到 {len(targets)} 个目录")
    print(f"[INFO] camera_ids={camera_ids} plane_cam={plane_cam} "
          f"azimuth_cam={azimuth_cam} flip_z={flip_z} dry_run={dry_run}")
    print(f"[INFO] 尺度方法: {'复用标定 ' + str(calib_path) if calib else '每目录自解'}\n")

    done = 0
    for tgt in targets:
        tpath = os.path.join(tgt, transforms_name)
        if not os.path.isfile(tpath):
            print(f"[MISS] {os.path.basename(os.path.normpath(tgt))}: 无 {transforms_name}")
            continue
        try:
            if canonicalize_one(tpath, camera_ids, plane_cam, azimuth_cam,
                                 flip_z, output_name, dry_run, calib=calib):
                done += 1
        except Exception as e:
            print(f"[ERR] {tgt}: {e}")

    print(f"\n[DONE] 处理 {done} 个目录"
          f"{' (dry-run, 未写盘)' if dry_run else ''},输出 {output_name}(原文件未改)。")
    return done
