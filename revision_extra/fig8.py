import os
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import spearmanr
R = os.environ.get("RESULTS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")); S=[11,22,33,44,55]
L=["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]; C=["NORM","MI","STTC","CD","HYP"]
Y=np.load(f"{R}/y_test.npy")
A=np.stack([np.load(f"{R}/b1_ci/m4_mh_se_seed{s}.npz")["A"] for s in S])
plt.rcParams.update({"font.size":7,"font.family":"serif"})
fig,ax=plt.subplots(1,5,figsize=(6.25,1.75),sharey=True)
meds=[]
for c in range(5):
    pos=Y[:,c]>0.5; Ac=A[:,pos,c,:]; Am=Ac.mean(0)
    for s in range(5):
        mu=Ac[s].mean(0); meds.append((C[c],S[s],np.median([spearmanr(Ac[s,i],mu)[0] for i in range(Ac.shape[1])])))
    ax[c].boxplot([Am[:,l] for l in range(12)],tick_labels=L,showfliers=False,widths=0.55,
                  medianprops=dict(color="k",lw=0.8),boxprops=dict(lw=0.6),whiskerprops=dict(lw=0.6),capprops=dict(lw=0.6))
    ax[c].set_title(f"{C[c]} (n = {pos.sum()})",fontsize=7); ax[c].tick_params(axis='x',labelrotation=90,labelsize=6); ax[c].tick_params(axis='y',labelsize=6)
ax[0].set_ylabel("Gate weight",fontsize=7)
plt.tight_layout(pad=0.3,w_pad=0.3); plt.savefig("fig8_gate_records.png",dpi=300)
v=np.array([m[2] for m in meds]); print("median spearman range",v.min().round(3),v.max().round(3))
for cn in C: print(cn, np.round([m[2] for m in meds if m[0]==cn],3))
