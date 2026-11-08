#!/usr/bin/env python3
"""扫描 RimWorld 1.7(Waylith) 存档（未压缩 XML），统计指定建筑 ID 是否存在。

用途：验证 "Perspective.CompOffsetter" 在载入时抛 MissingMethodException
导致建筑被丢弃的结论，并确认旧存档中建筑是否仍然完整（能否救回）。
"""
import mmap
import os
import sys

# 日志中 "Exception while saving <名字>" 列出的 12 个建筑
TARGETS = [
    "Door121568", "Door121738", "Door121752", "Door121781", "Door121796",
    "ElectricTailoringBench121804",
    "RGK_bedSingle121618", "RGK_bedSingle121626", "RGK_bedSingle121685",
    "RGK_bedSingle121694", "RGK_bedSingle121766", "RGK_bedSingle121861",
]

NEEDLES = [b"Perspective.CompOffsetter", b"<thing ", b"MapMeshFlag"]


def scan(path):
    size = os.path.getsize(path)
    with open(path, "rb") as fh:
        if size == 0:
            return None
        data = fh.read()
        found = {t: data.find(b"Thing_" + t.encode()) != -1 for t in TARGETS}
        counts = {n.decode(): data.count(n) for n in NEEDLES}
    return found, counts, size


def main():
    paths = sys.argv[1:]
    if not paths:
        print("usage: rw_save_scan.py <save.rws> ...")
        return 1
    for p in paths:
        res = scan(p)
        name = os.path.basename(p)
        if res is None:
            print(f"{name:<22} EMPTY")
            continue
        found, counts, size = res
        have = [t for t, ok in found.items() if ok]
        missing = [t for t, ok in found.items() if not ok]
        print(f"{name:<22} {size/1048576:7.1f} MB  "
              f"目标建筑 {len(have):2}/{len(TARGETS)}  "
              f"CompOffsetter×{counts['Perspective.CompOffsetter']}  "
              f"thing节点={counts['<thing ']}  "
              f"MapMeshFlag×{counts['MapMeshFlag']}")
        if missing:
            print("      缺失: " + ", ".join(missing))
    return 0


if __name__ == "__main__":
    sys.exit(main())
