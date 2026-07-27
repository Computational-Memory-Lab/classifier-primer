#!/usr/bin/env python3
"""Rasterise the 3D head exactly as assets/sereega.js draws it, to a PNG.

WHY. Headless Chromium and headless Chrome both hang in this environment, so the
page cannot be screenshotted, and every visual bug in this figure shipped
unnoticed as a result: a camera looking down the top of the skull, a mesh
truncated below the ears, electrodes at the wrong latitudes, a colour scale so
compressed the topography was invisible. This replicates the renderer -- same
rotation, projection, auto-fit, backface culling, shading and colour mapping --
so the output can actually be looked at.

It is a CHECK, not the renderer. If you change headView(), change this too, or
it stops telling you the truth.

    python3 tools_preview_head.py     # writes /tmp/head_dark.png, /tmp/head_light.png
"""
import json, re, numpy as np
from PIL import Image, ImageDraw

meta = json.load(open('data/head_mesh.json')); buf = open('data/head_mesh.bin','rb').read()
nV, nF = meta['nVerts'], meta['nFaces']
V = np.frombuffer(buf,'<f4',nV*3).reshape(-1,3).astype(float)
N = np.frombuffer(buf,'<f4',nV*3,nV*3*4).reshape(-1,3).astype(float)
F = np.frombuffer(buf,'<u2',nF*3,nV*3*4*2).reshape(-1,3).astype(int)
EL = [(n, np.array([float(x),float(y),float(z)])) for n,x,y,z in re.findall(
    r'\{ name: "([^"]+)", pos: \[([-\d.]+), ([-\d.]+), ([-\d.]+)\] \}', open('assets/electrodes.js').read())]
M19 = json.loads('[' + re.search(r'\n  19: \[([^\]]+)\]', open('assets/electrodes.js').read()).group(1) + ']')

W,H,CAM,PAD = 520,430,4.2,14
SS = 2
def rotate(P, yaw, pitch):
    cw,sw = np.cos(yaw),np.sin(yaw); cp,sp = np.cos(pitch),np.sin(pitch)
    xr = P[...,0]*cw - P[...,1]*sw; ya = P[...,0]*sw + P[...,1]*cw
    return np.stack([xr, ya*sp + P[...,2]*cp, ya*cp - P[...,2]*sp], -1)

def diverging(v, dark):
    v = np.clip(v,-1,1)
    neg = np.array([42,120,214.]); pos = np.array([235,104,52.])
    mid = np.array([58,58,62.]) if dark else np.array([238,238,234.])
    g = abs(v)**0.6
    t = neg if v < 0 else pos
    return mid + (t-mid)*g

def dip(P, pos, mom):
    r = P - pos; d2 = (r*r).sum(-1); d = np.sqrt(d2)
    return (r*mom).sum(-1)/np.maximum(d2*d,1e-12)

def render(yaw, pitch, dark, pos, mom, fname):
    RV = rotate(V,yaw,pitch); RN = rotate(N,yaw,pitch)
    PV = dip(V,pos,mom); maxAbs = max(np.sort(abs(PV))[int(len(PV)*0.97)],1e-9)
    k = CAM/(CAM-RV[:,2]); px = RV[:,0]*k; py = RV[:,1]*k
    sc = min((W-2*PAD)/(px.max()-px.min()), (H-2*PAD)/(py.max()-py.min()))
    cx = W/2-((px.min()+px.max())/2)*sc; cy = H/2+((py.min()+py.max())/2)*sc
    sx = cx+px*sc; sy = cy-py*sc
    fn = (RN[F[:,0]]+RN[F[:,1]]+RN[F[:,2]])/3
    fn /= np.linalg.norm(fn,axis=1,keepdims=True).clip(1e-9)
    depth = RV[F,2].mean(1)
    img = Image.new('RGB',(W*SS,H*SS),(13,13,13) if dark else (249,249,247))
    dr = ImageDraw.Draw(img)
    drawn = 0
    for f in np.argsort(depth):
        nz = fn[f,2]
        if nz <= 0.02: continue
        lam = max(0.0, fn[f,0]*-0.28 + fn[f,1]*0.34 + nz*0.90)
        shade = 0.46+0.54*lam
        v = PV[F[f]].mean()/maxAbs
        col = tuple(int(c) for c in np.clip(diverging(v,dark)*shade,0,255))
        dr.polygon([(sx[i]*SS,sy[i]*SS) for i in F[f]], fill=col, outline=col)
        drawn += 1
    for gi in M19:
        nm,p = EL[gi]
        r = rotate(p[None,:],yaw,pitch)[0]
        if r[2] < -0.02: continue
        kk = CAM/(CAM-r[2]); ex = (cx+r[0]*kk*sc)*SS; ey = (cy-r[1]*kk*sc)*SS
        v = dip(p[None,:],pos,mom)[0]/maxAbs
        col = tuple(int(c) for c in np.clip(diverging(v,dark),0,255))
        dr.ellipse([ex-5*SS,ey-5*SS,ex+5*SS,ey+5*SS], fill=col,
                   outline=(255,255,255) if dark else (0,0,0), width=SS)
    img.resize((W,H), Image.LANCZOS).save(fname)
    return drawn, sc

pos = np.array([0,-0.30,0.34]); mom = np.array([0,0,1.0])
d1,s1 = render(-0.62,0.10,True,  pos,mom,'/tmp/head_dark.png')
d2,s2 = render(1.30,0.15, False, pos,mom,'/tmp/head_light.png')
print(f'dark  view: {d1} faces drawn, scale {s1:.0f}')
print(f'light view: {d2} faces drawn, scale {s2:.0f}')
