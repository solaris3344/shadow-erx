#!/usr/bin/env python3
"""V10. Same-prior, single intervention, short-horizon ERX counterfactual diagnostics.

At decision line t, an online anchor predicts line t using its pre-decision
state, then selects a candidate update prepared from line t-1. We SNAPSHOT the
same candidate states ('skip', 'trim05', 'trim10') before the anchor's decision.
Each branch commits exactly one alternative at t. Branches then use an IDENTICAL
followup policy: freeze the accepted state, or delayed fixed Trim10. Outcomes
are scored on lines t+1..t+H. No evaluation label enters action or followup.

Counterfactual comparisons are finite-horizon model interventions on a fixed
observed sequence, NOT randomized causal estimates or held-out validation.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, time
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from threadpoolctl import threadpool_limits
from baseline_v3 import stable_md, shared_projector, reference_repo, normalise_line
from run_shadow import ShadowPolicy, DelayedPolicy, prepare_warmup, proposal

ACTIONS=('skip','trim05','trim10')
FOLLOWUPS=('frozen','delayed_trim10')


def save_csv(path, rows):
    if not rows: return
    path=Path(path)
    with path.open('w', newline='', encoding='utf-8') as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def core_shift(z,mu,cov,fraction):
    # Unlabeled diagnostic. Do not conflate this proxy with proven scene drift.
    raw=stable_md(z,mu,cov)
    k=min(len(raw),max(z.shape[1]+2,int(round(len(raw)*fraction))))
    ids=np.argsort(raw,kind='stable')[:k]
    return float(np.linalg.norm(z[ids].mean(axis=0)-mu)/
                 np.sqrt(max(float(np.trace(cov)),1e-12)))


def make_state(prior):
    return (np.asarray(prior[0]).copy(), np.asarray(prior[1]).copy())


def make_candidates(z_prev,prior,momentum):
    return {'skip':make_state(prior),
            'trim05':proposal(z_prev,*prior,.05,momentum),
            'trim10':proposal(z_prev,*prior,.10,momentum)}


def roll_forward(z,decision_idx,committed_state,momentum,horizon,followup,normalise=False):
    """Return [H,pixels] scores starting strictly AFTER decision_idx.

    On decision row j: a pending update from j-1 is committed; then a proposal
    trained on row j is staged. At row j+1 prediction uses the committed state;
    only AFTER that prediction is the staged j proposal committed.
    """
    if followup not in FOLLOWUPS: raise ValueError(followup)
    mu,cov=make_state(committed_state)
    if followup=='delayed_trim10':
        pending=proposal(z[decision_idx],mu,cov,.10,momentum)
    scores=np.empty((horizon,z.shape[1]),dtype=np.float64)
    for h in range(horizon):
        line=z[decision_idx+1+h]
        val=stable_md(line,mu,cov)
        scores[h]=normalise_line(val) if normalise else val
        if followup=='delayed_trim10' and h+1<horizon:
            mu,cov=pending
            pending=proposal(line,mu,cov,.10,momentum)
    return scores


def window_metrics(gt,scores,h):
    """Pooled H-line metrics; labels are used here and nowhere upstream."""
    # z windows are [line,pixel]; ground truth is also supplied [line,pixel].
    truth=np.asarray(gt[:h],dtype=bool).ravel()
    pred=np.asarray(scores[:h],dtype=np.float64).ravel()
    if not np.isfinite(pred).all(): raise ValueError('nonfinite model scores')
    p=int(truth.sum()); n=len(truth)-p
    if p==0 or n==0:
        return {'ap':float('nan'),'auroc':float('nan'),'fpr90':float('nan'),
                'n_positive':p,'n_negative':n,'valid':0}
    fpr,tpr,_=roc_curve(truth,pred)
    ix=min(int(np.searchsorted(tpr,.90,side='left')),len(fpr)-1)
    return {'ap':float(average_precision_score(truth,pred)),
            'auroc':float(roc_auc_score(truth,pred)),
            'fpr90':float(fpr[ix]), 'n_positive':p,'n_negative':n,'valid':1}


def density_bin(v):
    if v==0:return '0'
    if v<=.01:return '(0,1%]'
    if v<=.05:return '(1%,5%]'
    if v<=.10:return '(5%,10%]'
    if v<=.20:return '(10%,20%]'
    return '>20%'


def source_check(anchor,anchor_name,normalise,z,j,original_score=None):
    # no side effects; anchor step will be called by the caller, afterwards
    if original_score is not None:
        val=stable_md(z[j],anchor.mu,anchor.cov)
        if normalise:val=normalise_line(val)
        if not np.array_equal(val,original_score):
            raise AssertionError('Source scoring changed before intervention')


def run_seed(x, mask, seed, warmup, dims, momentum,core_fraction,penalty,
             anchor_name, horizon, horizons, stride, max_windows,
             normalise=False, n=None, check_reference=True):
    """Run source policy serially, sample interventions by predetermined stride."""
    n=x.shape[1] if n is None else min(n,x.shape[1])
    projector,mu0,cov0=prepare_warmup(x,seed,warmup,dims,momentum)
    z=np.empty((n-warmup,x.shape[0],dims),dtype=np.float64)
    for t in range(warmup,n):
        z[t-warmup]=projector.transform(np.asarray(x[:,t,:],dtype=np.float64))
    if anchor_name=='shadow_full':
        anchor=ShadowPolicy(mu0,cov0,momentum,core_fraction,penalty)
    elif anchor_name=='delayed_trim10':
        anchor=DelayedPolicy(mu0,cov0,momentum,.10)
    else:raise ValueError(anchor_name)
    all_rows=[]; actions=[]; cases=0
    prior_source_scores=[]
    expected_next = None
    last_proposal_prior = None
    for j,line in enumerate(z):
        t=warmup+j
        active_prior=(anchor.mu.copy(),anchor.cov.copy())
        pending = None if anchor.pending is None else {k:make_state(v) for k,v in anchor.pending.items()} if isinstance(anchor.pending,dict) else None
        # DelayedPolicy holds only the fixed update candidate; construct other
        # candidates with the SAME state used for that line's proposal at j-1.
        # This exact proposal base is stored explicitly at end of previous step.
        if anchor_name=='delayed_trim10' and j>0:
            if last_proposal_prior is None:raise AssertionError('Missing proposal prior')
            pending=make_candidates(z[j-1],last_proposal_prior,momentum)
            if not np.array_equal(pending['trim10'][0],anchor.pending[0]) or not np.array_equal(pending['trim10'][1],anchor.pending[1]):
                raise AssertionError('Delayed Trim10 prior/proposal reconstruction mismatch')
        expect=stable_md(line,*active_prior)
        if normalise: expect=normalise_line(expect)
        if expected_next is not None:
            expected_j, expected_scores = expected_next
            if j != expected_j or not np.array_equal(expect, expected_scores):
                raise AssertionError('Selected branch next-line forecast does not match source online model')
            expected_next=None
        emit=anchor.step(line,t,normalise) if anchor_name=='shadow_full' else anchor.step(line,normalise)
        if check_reference and not np.array_equal(expect,emit):
            raise AssertionError('Anchor output is not prequential')
        prior_source_scores.append(emit)
        if anchor_name=='delayed_trim10':
            # Anchor proposed this row's candidate AFTER applying prior pending.
            last_proposal_prior=(anchor.mu.copy(),anchor.cov.copy())
        if j==0 or j+horizon>=len(z) or ((j-1)%stride!=0):continue
        if max_windows is not None and cases>=max_windows:continue
        cases+=1
        assert pending is not None and set(pending)==set(ACTIONS)
        selected = (anchor.decision_rows[-1]['selected'] if anchor_name=='shadow_full' else 'trim10')
        if anchor_name=='shadow_full' and selected not in ACTIONS:
            raise AssertionError('Unrecognized source action')
        # Important: GT is retrieved only AFTER all branches are independently
        # forecast. Source policy has already advanced, and cannot be changed.
        trajectories={}
        for followup in FOLLOWUPS:
            for action in ACTIONS:
                trajectories[(followup,action)]=roll_forward(
                    z,j,pending[action],momentum,horizon,followup,normalise)
        if selected in ACTIONS:
            expected_next=(j+1,trajectories[('frozen',selected)][0].copy())
        gt_future=np.asarray(mask[:,t+1:t+1+horizon]!=0,dtype=bool).T
        gt_previous=np.asarray(mask[:,t-1]!=0,dtype=bool)
        gt_decision=np.asarray(mask[:,t]!=0,dtype=bool)
        context={'seed':seed,'anchor':anchor_name,'decision_line':t,
                 'candidate_update_line':t-1,'source_action':selected,
                 'source_core_shift':core_shift(line,*active_prior,core_fraction),
                 'candidate_gt_fraction_POSTHOC':float(gt_previous.mean()),
                 'candidate_density_bin_POSTHOC':density_bin(float(gt_previous.mean())),
                 'decision_gt_fraction_POSTHOC':float(gt_decision.mean()),
                 'stride':stride}
        for followup in FOLLOWUPS:
            for h in horizons:
                for action in ACTIONS:
                    s=trajectories[(followup,action)]
                    metrics=window_metrics(gt_future,s,h)
                    all_rows.append({**context,'followup':followup,'horizon':h,
                                     'action':action,'is_source_action':int(action==selected),
                                     **metrics})
        actions.append(context)
    return all_rows,actions,np.stack(prior_source_scores),cases


def aggregate(rows):
    """Paired differences within each same-state intervention; no IID claims."""
    # keyed by (seed,anchor,decision_line,followup,horizon)
    groups={}
    for r in rows:
        k=tuple(r[x] for x in ('seed','anchor','decision_line','followup','horizon'))
        groups.setdefault(k,{})[r['action']]=r
    comparisons=[]
    for (seed,anchor,t,pol,h),s in groups.items():
        if set(s)!=set(ACTIONS):raise AssertionError('Incomplete paired branches')
        common={k:s['skip'][k] for k in ('candidate_gt_fraction_POSTHOC','candidate_density_bin_POSTHOC','source_action','source_core_shift')}
        r={'seed':seed,'anchor':anchor,'decision_line':t,'followup':pol,'horizon':h,**common,
           'valid':int(all(v['valid'] for v in s.values()))}
        for action in ACTIONS:
            for m in ('ap','auroc','fpr90'):
                r[f'{m}_{action}']=s[action][m]
        for m in ('ap','auroc','fpr90'):
            for action in ('trim05','trim10'):
                r[f'delta_{m}_{action}_minus_skip']=s[action][m]-s['skip'][m]
            r[f'delta_{m}_trim10_minus_trim05']=s['trim10'][m]-s['trim05'][m]
        comparisons.append(r)
    summaries=[]
    for anchor in sorted(set(r['anchor'] for r in comparisons)):
        for policy in FOLLOWUPS:
            for h in sorted(set(r['horizon'] for r in comparisons)):
                sub=[r for r in comparisons if r['anchor']==anchor and r['followup']==policy and r['horizon']==h and r['valid']]
                row={'anchor':anchor,'followup':policy,'horizon':h,'valid_seed_windows':len(sub),
                     'unique_projection_seeds':len({v['seed'] for v in sub})}
                for m in ('ap','auroc','fpr90'):
                    for action in ACTIONS:
                        vals=np.asarray([v[f'{m}_{action}'] for v in sub],dtype=float)
                        row[f'{m}_{action}_window_mean']=float(vals.mean()) if len(vals) else float('nan')
                    for action in ('trim05','trim10'):
                        vals=np.asarray([v[f'delta_{m}_{action}_minus_skip'] for v in sub],dtype=float)
                        row[f'delta_{m}_{action}_minus_skip_mean']=float(vals.mean()) if len(vals) else float('nan')
                summaries.append(row)
    strata=[]
    for anchor in sorted(set(r['anchor'] for r in comparisons)):
      for pol in FOLLOWUPS:
       for h in sorted(set(r['horizon'] for r in comparisons)):
        for decision in ACTIONS:
         for bin_name in ('0','(0,1%]','(1%,5%]','(5%,10%]','(10%,20%]','>20%'):
          sub=[r for r in comparisons if r['anchor']==anchor and r['followup']==pol and r['horizon']==h and r['source_action']==decision and r['candidate_density_bin_POSTHOC']==bin_name and r['valid']]
          if not sub:continue
          strata.append({'anchor':anchor,'followup':pol,'horizon':h,'source_action':decision,
                         'gt_bin_POSTHOC':bin_name,'n_seed_windows':len(sub),
                         'delta_ap_trim05_minus_skip':float(np.mean([r['delta_ap_trim05_minus_skip'] for r in sub])),
                         'delta_ap_trim10_minus_skip':float(np.mean([r['delta_ap_trim10_minus_skip'] for r in sub])),
                         'delta_ap_trim10_minus_trim05':float(np.mean([r['delta_ap_trim10_minus_trim05'] for r in sub]))})
    return comparisons,summaries,strata


def validate_args(args,x,mask):
    if x.ndim!=3 or mask.shape!=x.shape[:2]:raise ValueError(f'image={x.shape},mask={mask.shape}; expected H,W,B and H,W')
    if args.warmup<2 or args.warmup+max(args.horizons)+2>=x.shape[1]:raise ValueError('Warmup or horizon invalid')
    if args.projected_dims<2 or args.projected_dims>x.shape[2]:raise ValueError('Bad projected dims')
    if not (0<args.core_fraction<1) or args.penalty<0 or not (0<args.momentum<=1):raise ValueError('Invalid method parameters')
    if args.stride<1 or any(h<1 for h in args.horizons) or len(set(args.horizons))!=len(args.horizons):raise ValueError('Invalid horizons or stride')
    if len(set(args.seeds))!=len(args.seeds):raise ValueError('Seeds must be unique')
    if args.max_windows is not None and args.max_windows<1:raise ValueError('max_windows must be >=1')
    if args.max_lines is not None and args.max_lines<args.warmup+max(args.horizons)+3:raise ValueError('max_lines too short')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--hyperad-root',required=True)
    p.add_argument('--hsi',required=True);p.add_argument('--mask',required=True)
    p.add_argument('--dataset-name',default='Beach')
    p.add_argument('--warmup',type=int,default=99)
    p.add_argument('--projected-dims',type=int,default=5)
    p.add_argument('--momentum',type=float,default=.10)
    p.add_argument('--core-fraction',type=float,default=.80)
    p.add_argument('--penalty',type=float,default=.05)
    p.add_argument('--seeds',type=int,nargs='+',default=[0,1,2])
    p.add_argument('--anchors',nargs='+',choices=['shadow_full','delayed_trim10'],default=['shadow_full','delayed_trim10'])
    p.add_argument('--horizons',type=int,nargs='+',default=[1,5,10])
    p.add_argument('--stride',type=int,default=10,help='Predetermined window sampling; do not select by labels')
    p.add_argument('--max-windows',type=int,default=None,help='Smoke/quick runs only: first N sampled windows')
    p.add_argument('--max-lines',type=int,default=None,help='Quick tests: first N lines incl warmup')
    p.add_argument('--normalise-md',action='store_true')
    p.add_argument('--output',default='results_v10')
    args=p.parse_args()
    root=reference_repo(args.hyperad_root)
    x=np.load(args.hsi, mmap_mode='r',allow_pickle=False)
    mask=np.load(args.mask,mmap_mode='r',allow_pickle=False)
    if mask.ndim==3 and mask.shape[-1]==1:mask=mask[...,0]
    validate_args(args,x,mask)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    erx_source=root/'detectors'/'erx.py'
    meta={k:v for k,v in vars(args).items()}
    meta.update({'shape':list(x.shape),'mask_shape':list(mask.shape),
                 'reference_erx_sha256':hashlib.sha256(erx_source.read_bytes()).hexdigest(),
                 'design':'Same pending states at validation t; isolated 1-action intervention; fixed followup without GT; t+1..t+H metrics; diagnostic, NOT external test',
                 'gt_policy':'Mask accessed only AFTER counterfactual forecasts are produced. Never used for action selection.',
                 'note':'SNP and Beach already used for method development; windows overlap and projection seeds are not independent scenes.'})
    (out/'run_config.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False),encoding='utf-8')
    records=[]; source=[]
    print(f'V10 dataset={args.dataset_name}, shape={x.shape}, seeds={args.seeds}, anchors={args.anchors}, horizons={args.horizons}')
    print('NO future labels enter intervention or followup; source line t is scored before pending t-1 proposal is accepted.')
    with threadpool_limits(limits=1):
      for seed in args.seeds:
       for anchor in args.anchors:
        started=time.perf_counter()
        rs,src,scores,count=run_seed(x,mask,seed,args.warmup,args.projected_dims,
          args.momentum,args.core_fraction,args.penalty,anchor,
          max(args.horizons),sorted(args.horizons),args.stride,args.max_windows,
          args.normalise_md,args.max_lines)
        records.extend(rs); source.extend(src)
        save_csv(out/'windows_raw.csv',records)
        save_csv(out/'source_decisions.csv',source)
        print(f'  seed={seed} anchor={anchor} windows={count} records={len(rs)} elapsed={time.perf_counter()-started:.1f}s',flush=True)
    paired,summary,strata=aggregate(records)
    save_csv(out/'paired_window_effects.csv',paired)
    save_csv(out/'summary_by_horizon.csv',summary)
    save_csv(out/'stratified_effects_POSTHOC.csv',strata)
    for r in summary:
        print(f"  {r['anchor']:15s} {r['followup']:14s} H={r['horizon']:2d} valid_windows={r['valid_seed_windows']:4d} "
              f"AP(skip/05/10)={r['ap_skip_window_mean']:.4f}/{r['ap_trim05_window_mean']:.4f}/{r['ap_trim10_window_mean']:.4f}")
    print(f'Saved: {out.resolve()}')

if __name__=='__main__':main()
