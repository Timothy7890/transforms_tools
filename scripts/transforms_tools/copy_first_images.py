#!/usr/bin/env python3
"""
copy_first_images —— 把每个 <日期>/<sample>/images 下的第 n 张图汇集到一个目录

用「日期目录名」作前缀防止重名,例如 20250928__auto_cam01_angle0.png。
常用于快速抽取每次采集的代表图做预览/汇报。

数据结构假设:  <root>/<日期>/<sample>/images/*.png
命令(安装后):  copy_first_images --root <root> --sample K04 -n 1 -o <输出目录>
"""

import argparse
import glob
import os
import shutil

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="copy_first_images",
        description="复制每个 <root>/<日期>/<sample>/images 下的第 n 张图到一个目录(日期前缀防重名)。",
    )
    p.add_argument("--root", default=os.getcwd(),
                   help="数据集根目录,其下结构为 <日期>/<sample>/images (默认: 当前目录)")
    p.add_argument("--sample", "-s", default="K04",
                   help="样本/相机文件夹名 (默认: K04)")
    p.add_argument("-n", "--index", type=int, default=1,
                   help="取排序后的第几张图,从 1 开始 (默认: 1)")
    p.add_argument("-o", "--output", default=os.getcwd(),
                   help="保存目录 (默认: 当前目录)")
    return p.parse_args(argv)


def copy_first_images(root, sample="K04", index=1, output="."):
    """核心逻辑:从 root/*/sample/images 各取第 index 张图复制到 output。返回复制张数。"""
    if index < 1:
        raise ValueError("index 必须 >= 1")
    os.makedirs(output, exist_ok=True)

    image_dirs = sorted(glob.glob(os.path.join(root, "*", sample, "images")))
    print(f"found {len(image_dirs)} {sample}/images dirs under {root}")

    copied = 0
    for img_dir in image_dirs:
        imgs = sorted(f for f in os.listdir(img_dir) if f.lower().endswith(IMG_EXTS))
        if len(imgs) < index:
            print(f"  [skip] only {len(imgs)} image(s) in {img_dir}")
            continue

        chosen = imgs[index - 1]
        src = os.path.join(img_dir, chosen)
        date = os.path.basename(os.path.dirname(os.path.dirname(img_dir)))
        dst_name = f"{date}__{chosen}"
        dst = os.path.join(output, dst_name)

        stem, ext = os.path.splitext(dst_name)
        k = 1
        while os.path.exists(dst):
            dst = os.path.join(output, f"{stem}_{k}{ext}")
            k += 1

        shutil.copy2(src, dst)
        print(f"  {src}  ->  {os.path.basename(dst)}")
        copied += 1

    print(f"done. copied {copied} images to {output}")
    return copied


def main(argv=None):
    args = parse_args(argv)
    copy_first_images(root=args.root, sample=args.sample,
                      index=args.index, output=args.output)


if __name__ == "__main__":
    main()
