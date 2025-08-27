# -*- coding: utf-8 -*-
"""
image_generation_tauhat.py
  - 1000 grayscale images, each 100*100 pixels  
  - Tiny foreground: 1-4% -> class weights w0*w1 very small -> Otsu fails  
  - Large background σ_b (20-35), small foreground σ_s (4-7) -> large intra-class variance difference  
  - Bimodal distance Δ ∈ [45, 65], ensuring basic separability  
  - Valley point m = (μ_b + μ_s)/2 allowed near center - no need to avoid it  
Output directory: C:/Users/charlietommy/Desktop/picture_tauhat/
"""

import os, random, cv2
import numpy as np
OUT = r"C:/Users/charlietommy/Desktop/picture_tauhat/"
N, SZ = 1000, 100
os.makedirs(OUT, exist_ok=True)

def tiny_mask():
    """1-4% tiny shape masks, randomly chosen from circle, square, or polygon"""
    goal = random.randint(int(0.01*SZ*SZ), int(0.04*SZ*SZ))
    m = np.zeros((SZ,SZ), np.uint8)
    while m.sum() < goal:
        m[:] = 0
        kind = random.choice(["circle","square","poly"])
        if kind == "circle":
            r = max(int(np.sqrt(goal/np.pi)), 2)
            cx = random.randint(r, SZ-r); cy = random.randint(r, SZ-r)
            cv2.circle(m, (cx,cy), r, 1, -1)
        elif kind == "square":
            s = max(int(np.sqrt(goal)), 3)
            x = random.randint(0, SZ-s); y = random.randint(0, SZ-s)
            cv2.rectangle(m, (x,y), (x+s,y+s), 1, -1)
        else:
            k  = random.randint(4,6)
            pts= np.random.uniform(0.1*SZ,0.9*SZ,(k,2)).astype(int)
            cv2.fillPoly(m,[pts.reshape(-1,1,2)],1)
    return m

log=[]
for idx in range(N):
    # --- Background: wide peak + large area ---  
    μ_b  = random.uniform(80,140)                 # Medium gray  
    σ_b  = random.uniform(20,35)                  # Very wide  
    bg   = np.random.normal(μ_b, σ_b, (SZ,SZ))

    # --- Foreground: narrow peak + tiny area ---  
    Δ    = random.uniform(45,65)
    bright = random.random()<0.5
    μ_s  = μ_b + Δ if bright else μ_b - Δ         # Can be bright or dark  
    μ_s  = np.clip(μ_s, 5, 250)
    σ_s  = random.uniform(4,7)                    # Very narrow  
    fg   = np.random.normal(μ_s, σ_s, (SZ,SZ))

    mask = tiny_mask().astype(bool)
    img  = np.where(mask, fg, bg)
    img  = np.clip(img, 0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(OUT, f"img_{idx:04d}.png"), img)
    cv2.imwrite(os.path.join(OUT, f"img_{idx:04d}_mask.png"), (mask*255).astype(np.uint8))

    log.append(f"{idx:04d},μ_b={μ_b:.1f},σ_b={σ_b:.1f},μ_s={μ_s:.1f},σ_s={σ_s:.1f},area={mask.mean():.3f}")
with open(os.path.join(OUT,"log.txt"),"w") as f: f.write("\n".join(log))
print("[Generation Done] 1000 τ‑hat friendly images saved.")

