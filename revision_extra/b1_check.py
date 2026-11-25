import os
import numpy as np, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code"))
from leadlib import mean_pairwise_spearman
R = os.environ.get("RESULTS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")); S=[11,22,33,44,55]
pid=np.load(f"{R}/test_patient_ids.npy"); uniq,inv=np.unique(pid,return_inverse=True)
groups=[np.where(inv==k)[0] for k in range(len(uniq))]
rng=np.random.default_rng(2026)
draws=[np.concatenate([groups[k] for k in rng.integers(0,len(uniq),len(uniq))]) for _ in range(1000)]
for m in ["m1_early_se","m3_cs_se","m4_mh_se"]:
    A=[np.load(f"{R}/b1_ci/{m}_seed{s}.npz")["A"] for s in S]   # (N,5,12)
    pt=mean_pairwise_spearman(np.stack([a.mean(0).mean(0) for a in A]))
    b=np.array([mean_pairwise_spearman(np.stack([a[i].mean(0).mean(0) for a in A])) for i in draws])
    vals,cnt=np.unique(np.round(b,4),return_counts=True)
    print(m, "point %.4f"%pt, "CI", np.round(np.percentile(b,[2.5,97.5]),3), "share==point %.2f"%np.mean(np.isclose(b,pt,atol=1e-6)),
          "share>point %.3f"%np.mean(b>pt+1e-9), "distinct", len(vals), "top", list(zip(vals[np.argsort(-cnt)][:4],np.sort(cnt)[::-1][:4])))
