#!/usr/bin/env python3
"""evt.py prof.txt [prof2 ...]: per-event max time (s), count, and the GPU
transfer columns (CpuToGpu count/MB, GpuToCpu count/MB) of a PETSc -log_view,
summed over stages; events listed in EV, one column block per file."""
import sys, re
EV = ["KSPSolve","PCApply","PhysPC_Apply","PhysPC_SolveW","PhysPC_SolveRhoT","GMG_VCycle","GMG_Lines","GMG_AxSolve",
      "GMG_Coarse","GMG3_VCycle","GMG4_VCycle","GMG3_Lines","GMG4_Lines","GMG3_AxSolve","GMG4_AxSolve",
      "MatMult","MatMultAdd","MatMultTranspose","MatSolve","VecScatterEnd","VecMDot","VecNorm",
      "PhysPC_Extract","PhysPC_FactW","PhysPC_FactRhoT","BLK_Band","BLK_Dense","SPK_GathScat","SPK_Interior","SPK_Sep","SPK_Update","GMG_PtAP","GMG_SmSetup","GMG_DevOps","GMG_BlkNum","GMG_AxNum"]
def parse(fn):
    d = {}
    for ln in open(fn, errors="replace"):
        t = ln.split()
        if len(t) < 10 or t[0] not in EV: continue
        try:
            cnt = int(t[1]); tm = float(t[3]) if t[3] != "n/a" else 0.
        except ValueError: continue
        gpu = [0, 0., 0, 0.]
        if len(t) >= 27:   # ... GpuMflops CpuToGpu(count size) GpuToCpu(count size) GPU%
            try: gpu = [int(float(t[-5])), float(t[-4]), int(float(t[-3])), float(t[-2])]
            except ValueError: pass
        a = d.setdefault(t[0], [0, 0., 0, 0., 0, 0.])
        a[0] += cnt; a[1] += tm
        for i in range(4): a[2+i] += gpu[i]
    return d
fs = sys.argv[1:]; D = [parse(f) for f in fs]
print("%-18s" % "event" + "".join("| %8s %7s %7s %8s %7s %8s " % ("t[s]","n","H2D n","H2D MB","D2H n","D2H MB") for _ in fs))
for e in EV:
    if not any(e in d for d in D): continue
    s = "%-18s" % e
    for d in D:
        a = d.get(e, [0,0,0,0,0,0])
        s += "| %8.2f %7d %7d %8.0f %7d %8.0f " % (a[1], a[0], a[2], a[3], a[4], a[5])
    print(s)
