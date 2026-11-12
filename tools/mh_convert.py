"""Pack MakeHuman CC0 base mesh + default rig + weights + macro targets into assets/makehuman/body.bin.gz.
usage: python -I mh_convert.py <makehuman/data dir> <out file>"""
import gzip, json, os, struct, sys

data, out = sys.argv[1], sys.argv[2]
S = 0.1  # MakeHuman decimetres -> metres
Q = 20000.0  # delta quantisation (1/Q metres)

verts, faces = [], []
group = None
for line in open(os.path.join(data, "3dobjs/base.obj"), encoding="utf-8"):
    if line.startswith("v "):
        _, x, y, z = line.split()[:4]
        verts.append((float(x) * S, float(y) * S, float(z) * S))
    elif line.startswith("g "):
        group = line.split()[1]
    elif line.startswith("f ") and group == "body":
        idx = [int(t.split("/")[0]) - 1 for t in line.split()[1:]]
        for k in range(1, len(idx) - 1):
            faces.append((idx[0], idx[k], idx[k + 1]))

sk = json.load(open(os.path.join(data, "rigs/default.mhskel")))
wt = json.load(open(os.path.join(data, "rigs/default_weights.mhw")))["weights"]
joint_names = sorted(sk["joints"].keys())
jidx = {n: i for i, n in enumerate(joint_names)}
bone_names = list(sk["bones"].keys())
# parents before children
order, placed = [], set()
while len(order) < len(bone_names):
    for b in bone_names:
        if b in placed:
            continue
        p = sk["bones"][b]["parent"]
        if p is None or p in placed:
            order.append(b)
            placed.add(b)
bidx = {b: i for i, b in enumerate(order)}

def s(buf, txt):
    e = txt.encode("utf-8")
    buf += struct.pack("<H", len(e)) + e

buf = bytearray(b"MHB1")
buf += struct.pack("<I", len(verts))
for v in verts:
    buf += struct.pack("<3f", *v)
buf += struct.pack("<I", len(faces))
for f in faces:
    buf += struct.pack("<3I", *f)
buf += struct.pack("<I", len(joint_names))
for n in joint_names:
    s(buf, n)
    vs = sk["joints"][n]
    buf += struct.pack("<I", len(vs)) + struct.pack(f"<{len(vs)}I", *vs)
buf += struct.pack("<I", len(order))
for b in order:
    bd = sk["bones"][b]
    s(buf, b)
    p = bd["parent"]
    buf += struct.pack("<iII", bidx[p] if p else -1, jidx[bd["head"]], jidx[bd["tail"]])
    ws = wt.get(b, [])
    buf += struct.pack("<I", len(ws))
    for v, w in ws:
        buf += struct.pack("<If", v, w)

tdir = os.path.join(data, "targets/macrodetails")
names = sorted(n for n in os.listdir(tdir) if n.endswith(".target") and (n.startswith("universal-") or n.split("-")[0] in ("african", "asian", "caucasian")))
bdir = os.path.join(data, "targets/breast")
bnames = sorted(n for n in os.listdir(bdir) if n.endswith(".target") and "averagemuscle-averageweight" in n and "-young-" in n)
targets = [(n[:-7], os.path.join(tdir, n)) for n in names] + [("breast/" + n[:-7], os.path.join(bdir, n)) for n in bnames]
buf += struct.pack("<I", len(targets))
for name, path in targets:
    rows = []
    for line in open(path, encoding="utf-8"):
        if not line.strip() or line.startswith("#"):
            continue
        i, x, y, z = line.split()[:4]
        q = [int(round(float(c) * S * Q)) for c in (x, y, z)]
        if any(q):
            rows.append((int(i), max(-32767, min(32767, q[0])), max(-32767, min(32767, q[1])), max(-32767, min(32767, q[2]))))
    s(buf, name)
    buf += struct.pack("<I", len(rows))
    for r in rows:
        buf += struct.pack("<I3h", *r)
os.makedirs(os.path.dirname(out), exist_ok=True)
with gzip.open(out, "wb", compresslevel=9) as f:
    f.write(bytes(buf))
ys = [v[1] for v in verts]
print(f"verts {len(verts)} tris {len(faces)} joints {len(joint_names)} bones {len(order)} targets {len(targets)} height {max(ys)-min(ys):.3f} raw {len(buf)/1e6:.1f}MB -> {os.path.getsize(out)/1e6:.1f}MB")
