#!/usr/bin/env -S python -S
"""CREST 的 `method = "generic"` 客户端 —— 把一份 xyz 换成一份 `.engrad`.

PRODUCTION. Not a driver but a runtime component: CREST starts it once per gradient
call. Remove it and no CREST run with the composite calculator can finish.

**这个脚本每次梯度调用都会被 CREST 起一次**, 所以它只 import `os/socket/sys`,
不碰 numpy、不碰 torch。实测启动 13.6 ms; 真正的力计算在常驻服务端里
(`python -m openqha.potentials.mace_server`)。

CREST 侧的约定 (`src/calculator/generic_sc.f90`):
  * CREST 写 `genericinp.xyz` 到 calcspace, 然后执行
    `cd <calcspace> && <binary> genericinp.xyz [flags] > generic.out`
  * 算完后 CREST 读同目录下的 `genericinp.engrad`, 格式是 xtb/ORCA 的 `.engrad`:
    能量 Eh, 梯度 Eh/a0, 每行一个分量, `#` 开头是注释

用法 (由 CREST 调用, 一般不手工跑)::

    s0_mace_engrad.py genericinp.xyz
"""
import os
import socket
import sys

# CODATA 2018, 与 openqha.thermo 同一套
EV_PER_HARTREE = 27.211386245988
BOHR_PER_ANGSTROM = 1.0 / 0.529177210903

SOCKET = os.environ.get("S0_MACE_SOCKET", "/tmp/s0_mace_engrad.sock")


def die(msg):
    sys.stderr.write("s0_mace_engrad: {}\n".format(msg))
    sys.exit(1)


def main():
    if len(sys.argv) < 2:
        die("用法: s0_mace_engrad.py <xyz>")
    xyz = sys.argv[1]
    try:
        with open(xyz, "r") as fh:
            lines = fh.read().split("\n")
    except OSError as exc:
        die("读不了 {}: {}".format(xyz, exc))
    try:
        n = int(lines[0].split()[0])
    except (IndexError, ValueError):
        die("{} 的第一行不是原子数".format(xyz))
    body = []
    for line in lines[2:2 + n]:
        f = line.split()
        if len(f) < 4:
            die("坐标行解析失败: {!r}".format(line))
        body.append("{} {} {} {}".format(f[0], f[1], f[2], f[3]))
    if len(body) != n:
        die("声明 {} 个原子, 只读到 {} 行".format(n, len(body)))

    req = "{}\n{}\nEND\n".format(n, "\n".join(body)).encode("utf-8")
    try:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(600.0)
        s.connect(SOCKET)
        s.sendall(req)
        buf = b""
        while b"\nEND\n" not in buf:
            chunk = s.recv(65536)
            if not chunk:
                break
            buf += chunk
        s.close()
    except OSError as exc:
        die("连不上常驻服务端 {} ({}) —— 先跑 "
            "`python -m openqha.potentials.mace_server --socket {}`".format(SOCKET, exc, SOCKET))

    out = buf.decode("utf-8").strip().split("\n")
    if not out or out[0].startswith("ERROR"):
        die("服务端报错: {}".format(out[0] if out else "(空回复)"))
    if len(out) < n + 2:
        die("回复不完整: {} 行, 应为 {}".format(len(out), n + 2))

    energy_eh = float(out[0]) / EV_PER_HARTREE
    grad = []
    for line in out[1:1 + n]:
        fx, fy, fz = (float(v) for v in line.split())
        # 力 (eV/Å) -> 梯度 (Eh/a0): 取负, 换单位
        grad.append([-fx / EV_PER_HARTREE / BOHR_PER_ANGSTROM,
                     -fy / EV_PER_HARTREE / BOHR_PER_ANGSTROM,
                     -fz / EV_PER_HARTREE / BOHR_PER_ANGSTROM])

    stem = xyz[:-4] if xyz.endswith(".xyz") else xyz
    with open(stem + ".engrad", "w") as fh:
        fh.write("#\n# Atoms\n#\n{:5d}\n".format(n))
        fh.write("#\n# Energy ( Eh )\n#\n{:25.15f}\n".format(energy_eh))
        fh.write("#\n# Gradient ( Eh/a0 )\n#\n")
        for row in grad:
            for v in row:
                fh.write("{:25.15f}\n".format(v))


if __name__ == "__main__":
    main()
