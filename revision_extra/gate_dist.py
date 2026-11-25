import os
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, json
R = os.environ.get("RESULTS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")); S=[11,22,33,44,55]
L=["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]; C=["NORM","MI","STTC","CD","HYP"]
Y=np.load(f"{R}/y_test.npy")
A=np.stack([np.load(f"{R}/b1_ci/m4_mh_se_seed{s}.npz")["A"] for s in S])   # (5,N,5,12)
stats={}
fig,ax=plt.subplots(1,5,figsize=(15,3.4),sharey=True)
for c in range(5):
    pos=Y[:,c]>0.5
    Ac=A[:,pos,c,:]                       # (5, npos, 12)
    Am=Ac.mean(0)                         # seed-averaged per record
    ax[c].boxplot([Am[:,l] for l in range(12)],tick_labels=L,showfliers=False,widths=0.6)
    ax[c].set_title(f"{C[c]} (n = {pos.sum()})",fontsize=10); ax[c].tick_params(axis='x',labelrotation=90,labelsize=8)
    iqr=np.percentile(Am,75,0)-np.percentile(Am,25,0)
    # record-level agreement: Spearman between each record's gate vector and the class-mean vector, per seed
    from scipy.stats import spearmanr
    rho=[np.nanmedian([spearmanr(Ac[s,i],Ac[s].mean(0))[0] for i in range(Ac.shape[1])]) for s in range(5)]
    stats[C[c]]=dict(n=int(pos.sum()),median_IQR=float(np.median(iqr)),max_IQR=float(iqr.max()),
                     median_record_vs_class_mean_spearman=[round(float(r),3) for r in rho])
ax[0].set_ylabel("gate weight (per record, mean of 5 seeds)")
plt.suptitle("Multi-Head SE: distribution of class-specific gate weights over test records positive for each class",fontsize=10)
plt.tight_layout(); plt.savefig("fig_gate_distribution_m4.png",dpi=300)
print(json.dumps(stats,indent=1))
