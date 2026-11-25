import os
import sys, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code"))
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from make_figures import setup
setup()
R = os.environ.get("RESULTS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")); S=[11,22,33,44,55]
Z=[np.load(f'{R}/phase2a/cnn_lstm/seed{s}/lead_perturb.npz',allow_pickle=True) for s in S]
G=[np.load(f'{R}/ig_sens/cnn_lstm/seed{s}.npz') for s in S]
names=[str(n) for n in Z[0]['del_names']]; rnd=[i for i,n in enumerate(names) if n.startswith('random')]
D=np.stack([z['del_curves'] for z in Z]); IGv=np.stack([g['curve_val_ig'] for g in G]); k=np.arange(12)
fig, ax = plt.subplots(figsize=(6.4, 3.8))
ax.plot(k, D[:,names.index('occlusion_val_most_first')].mean(0), color="#e34948", lw=1.8, label="validation-ranked, most first")
ax.plot(k, IGv.mean(0), color="#eb6834", ls="--", lw=1.8, label="IG-ranked (validation records), most first")
ax.plot(k, D[:,names.index('occlusion_val_least_first')].mean(0), color="#2a78d6", lw=1.8, label="validation-ranked, least first")
ax.plot(k, D[:,rnd].mean(1).mean(0), color="#52514e", ls=":", lw=1.8, label="random order (mean of 10)")
ax.set_xlabel("Number of leads removed (zeroed)"); ax.set_ylabel("Test macro-AUROC"); ax.legend(fontsize=7.5)
ax.set_title("CNN-LSTM: deletion curves (mean of 5 seeds)", fontsize=9.5, loc="left")
fig.savefig('fig3_deletion_valIG.png', dpi=200, bbox_inches='tight')
print('IG val k3,k6', IGv[:,3].mean().round(4), IGv[:,6].mean().round(4), ' old IG k3', D[:,2,3].mean().round(4))
print('seeds IG below random k3/k6', (IGv[:,3]<D[:,rnd,3].mean(1)).sum(), (IGv[:,6]<D[:,rnd,6].mean(1)).sum())
