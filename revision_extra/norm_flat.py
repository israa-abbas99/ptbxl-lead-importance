import os
import numpy as np, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code"))
from leadlib import importance_from_preds, mean_pairwise_spearman
R = os.environ.get("RESULTS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")); S=[11,22,33,44,55]; C=["NORM","MI","STTC","CD","HYP"]
Y=np.load(f"{R}/y_test.npy")
for m in ["cnn_lstm","m4_mh_se"]:
    imp=np.stack([importance_from_preds(Y,*(lambda z:(z["P_base_test"],z["P_zero_test"]))(np.load(f"{R}/phase2a/{m}/seed{s}/lead_perturb.npz"))) for s in S])
    print(m)
    for c in range(5):
        v=imp[:,:,c]; sm=v.mean(0)
        print(f"  {C[c]:5s} stability {mean_pairwise_spearman(v):.3f}  seed-avg drop range {sm.min():.4f}..{sm.max():.4f}  spread(max-min) {sm.max()-sm.min():.4f}  mean within-lead seed SD {v.std(0,ddof=1).mean():.4f}")
