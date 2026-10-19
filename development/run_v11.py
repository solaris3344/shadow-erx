#!/usr/bin/env python3
"""HyperAD v11: event-clustered, equal-state follow-up diagnostics.

Identical Shadow/DelayedTrim10 source logic to v10; label-free model evolution.
Labels are used *only after* predictions to define event clusters, normal-only
strata, and performance metrics. Events/lines/seeds aren't IID replications.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,time
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from baseline_v3 import stable_md, normalise_line, reference_repo
from run_shadow import ShadowPolicy,DelayedPolicy,prepare_warmup,proposal
from run_v10 import (ACTIONS,FOLLOWUPS,make_state,make_candidates,core_shift,
                      window_metrics,aggregate,save_csv,roll_forward)


def build_event_map(mask,warmup,gap):
    """POST-HOC GT event clusters: join positive lines separated by <= gap clean lines."""
    active=np.any(np.asarray(mask[:,warmup:]!=0),axis=0)
    indices=np.flatnonzero(active)+warmup
    mapping=np.full(mask.shape[1],-1,dtype=np.int32)
    events=[]
    if len(indices)==0:return mapping,events
    starts=[int(indices[0])];ends=[]
    prev=int(indices[0])
    for line in indices[1:]:
        line=int(line)
        if line-prev-1>gap:
            ends.append(prev);starts.append(line)
        prev=line
    ends.append(prev)
    for k,(a,b) in enumerate(zip(starts,ends)):
        mapping[a:b+1]=k
        sub=np.asarray(mask[:,a:b+1]!=0)
        events.append({'event_id':k,'start_line':a,'end_line':b,
                       'span_lines':b-a+1,'anomalous_lines':int(np.any(sub,axis=0).sum()),
                       'positive_pixels':int(sub.sum())})
    return mapping,events


def score_and_nll(z,j,state,momentum,horizon,policy,normalise=False):
    """Same update order as v10.roll_forward, plus Gaussian proper log-score.

    For non-physical projected z; comparisons are within the SAME feature space.
    NLL is a diagnostic; not an external real-world likelihood claim.
    """
    mu,cov=make_state(state)
    if policy=='delayed_trim10':pending=proposal(z[j],mu,cov,.10,momentum)
    scores=np.empty((horizon,z.shape[1]),dtype=np.float64)
    nll=np.empty_like(scores)
    const=z.shape[2]*np.log(2*np.pi)
    for h in range(horizon):
        line=z[j+1+h]
        raw=stable_md(line,mu,cov)
        scores[h]=normalise_line(raw) if normalise else raw
        sign,logdet=np.linalg.slogdet(cov+1e-5*np.eye(cov.shape[0]))
        if sign<=0:raise ArithmeticError('Nonpositive covariance determinant')
        nll[h]=0.5*(raw**2+logdet+const)
        if policy=='delayed_trim10' and h+1<horizon:
            mu,cov=pending
            pending=proposal(line,mu,cov,.10,momentum)
    return scores,nll


def run_seed(x,mask,seed,warmup,dims,momentum,core_fraction,penalty,
             anchor_name,horizons,stride,max_windows=None,normalise=False,max_lines=None):
    n=x.shape[1] if max_lines is None else min(max_lines,x.shape[1])
    H=max(horizons)
    p,mu0,cov0=prepare_warmup(x,seed,warmup,dims,momentum)
    z=np.empty((n-warmup,x.shape[0],dims),dtype=np.float64)
    for t in range(warmup,n):z[t-warmup]=p.transform(np.asarray(x[:,t,:],dtype=np.float64))
    if anchor_name=='shadow_full':anchor=ShadowPolicy(mu0,cov0,momentum,core_fraction,penalty)
    elif anchor_name=='delayed_trim10':anchor=DelayedPolicy(mu0,cov0,momentum,.10)
    else:raise ValueError(anchor_name)
    rows=[];decisions=[];count=0;prev_prior=None;source_scores=[]
    next_expected=None
    for j,line in enumerate(z):
        t=warmup+j
        prior=make_state((anchor.mu,anchor.cov))
        pending=None
        if anchor_name=='shadow_full' and anchor.pending is not None:
            pending={k:make_state(v) for k,v in anchor.pending.items()}
        if anchor_name=='delayed_trim10' and j>0:
            pending=make_candidates(z[j-1],prev_prior,momentum)
            assert np.array_equal(pending['trim10'][0],anchor.pending[0])
            assert np.array_equal(pending['trim10'][1],anchor.pending[1])
        expected=stable_md(line,*prior)
        if normalise:expected=normalise_line(expected)
        if next_expected is not None:
            if j!=next_expected[0] or not np.array_equal(expected,next_expected[1]):
                raise AssertionError('Selected next-line scoring differs from source policy')
            next_expected=None
        emit=anchor.step(line,t,normalise) if anchor_name=='shadow_full' else anchor.step(line,normalise)
        assert np.array_equal(expected,emit)
        source_scores.append(emit)
        if anchor_name=='delayed_trim10':prev_prior=make_state((anchor.mu,anchor.cov))
        if j==0 or j+H>=len(z) or ((j-1)%stride!=0):continue
        if max_windows is not None and count>=max_windows:continue
        count+=1
        chosen=anchor.decision_rows[-1]['selected'] if anchor_name=='shadow_full' else 'trim10'
        shift=core_shift(line,*prior,core_fraction)
        trajectories={}
        for followup in FOLLOWUPS:
            for action in ACTIONS:
                scores,nll=score_and_nll(z,j,pending[action],momentum,H,followup,normalise)
                trajectories[(followup,action)]=(scores,nll)
        next_expected=(j+1,trajectories[('frozen',chosen)][0][0].copy())
        # Only now open labels; no intervention, projection, or policy reads GT.
        gt_cand=np.asarray(mask[:,t-1]!=0,dtype=bool)
        gt_now=np.asarray(mask[:,t]!=0,dtype=bool)
        gt_future=np.asarray(mask[:,t+1:t+1+H]!=0,dtype=bool).T
        context={'seed':seed,'anchor':anchor_name,'decision_line':t,'candidate_update_line':t-1,
                 'source_action':chosen,'source_core_shift':shift,
                 'candidate_gt_fraction_POSTHOC':float(gt_cand.mean()),
                 'decision_gt_fraction_POSTHOC':float(gt_now.mean()),'stride':stride}
        # Metrics are appended only AFTER all candidate predictions are calculated.
        for followup in FOLLOWUPS:
            for h in horizons:
                truth=gt_future[:h]
                normal=~truth
                for action in ACTIONS:
                    score,nll=trajectories[(followup,action)]
                    metric=window_metrics(gt_future,score,h)
                    # Mean proper NLL on annotated-negative pixels; for normal-only
                    # windows it is an unbiased within-scene *descriptive* proxy.
                    normal_nll=float(nll[:h][normal].mean()) if normal.any() else float('nan')
                    all_nll=float(nll[:h].mean())
                    rows.append({**context,'followup':followup,'horizon':h,
                                 'action':action,'is_source_action':int(action==chosen),
                                 **metric,'nll_annotated_normal_POSTHOC':normal_nll,
                                 'nll_all':all_nll,'future_has_gt_positive_POSTHOC':int(truth.any()),
                                 'future_gt_positive_fraction_POSTHOC':float(truth.mean())})
        decisions.append(context)
    return rows,decisions,count


def paired_v11(rows):
    groups={}
    for r in rows:
        key=(r['seed'],r['anchor'],r['decision_line'],r['followup'],r['horizon'])
        groups.setdefault(key,{})[r['action']]=r
    paired=[]
    for (seed,anchor,t,pol,h),s in groups.items():
        if set(s)!=set(ACTIONS):raise ValueError('Missing branch')
        base=s['skip']
        r={k:base[k] for k in ('seed','anchor','decision_line','candidate_update_line',
                             'source_action','source_core_shift','candidate_gt_fraction_POSTHOC',
                             'decision_gt_fraction_POSTHOC','stride')}
        r.update(followup=pol,horizon=h,valid_ap=int(all(v['valid'] for v in s.values())),
                 future_has_gt_positive_POSTHOC=base['future_has_gt_positive_POSTHOC'],
                 future_gt_positive_fraction_POSTHOC=base['future_gt_positive_fraction_POSTHOC'])
        for a in ACTIONS:
            for k in ('ap','auroc','fpr90','nll_annotated_normal_POSTHOC','nll_all'):
                r[f'{k}_{a}']=s[a][k]
        for k in ('ap','auroc','fpr90','nll_annotated_normal_POSTHOC','nll_all'):
            r[f'delta_{k}_skip_minus_trim10']=s['skip'][k]-s['trim10'][k]
            r[f'delta_{k}_trim05_minus_trim10']=s['trim05'][k]-s['trim10'][k]
        paired.append(r)
    return paired


def grouped_mean(rows,keys,valkeys):
    groups={}
    for r in rows:
        key=tuple(r[k] for k in keys)
        groups.setdefault(key,[]).append(r)
    result=[]
    for group,sub in sorted(groups.items(),key=lambda t:tuple(map(str,t[0]))):
        out=dict(zip(keys,group));out['n_windows']=len(sub)
        for val in valkeys:
            values=np.array([float(v[val]) for v in sub],dtype=float)
            values=values[np.isfinite(values)]
            out['n_valid_'+val]=len(values)
            out['mean_'+val]=float(values.mean()) if len(values) else float('nan')
            out['median_'+val]=float(np.median(values)) if len(values) else float('nan')
        result.append(out)
    return result


def analyze(paired,events,event_map,horizons,shift_quantile=.75):
    # Assign events only POST-HOC, never to proposal generation or source policy.
    for r in paired:
        cand=r['candidate_update_line'];r['event_id_POSTHOC']=int(event_map[cand])
    event_rows=[r for r in paired if r['event_id_POSTHOC']>=0 and r['valid_ap']]
    keys=['event_id_POSTHOC','seed','anchor','followup','horizon']
    event_seed=grouped_mean(event_rows,keys,
        ['delta_ap_skip_minus_trim10','delta_ap_trim05_minus_trim10',
         'delta_fpr90_skip_minus_trim10'])
    # First average repeated windows WITHIN each event and within seed; then
    # aggregate seed event means (NOT independent scene replication).
    event_group=grouped_mean(event_seed,['event_id_POSTHOC','anchor','followup','horizon'],
                             ['mean_delta_ap_skip_minus_trim10',
                              'mean_delta_ap_trim05_minus_trim10'])
    event_summary=grouped_mean(event_group,['anchor','followup','horizon'],
                               ['mean_mean_delta_ap_skip_minus_trim10',
                                'mean_mean_delta_ap_trim05_minus_trim10'])
    # Event leave-one-out sensitivity of the event-equal summary.
    loo=[]
    settings=sorted({(r['anchor'],r['followup'],r['horizon']) for r in event_group})
    for anchor,pol,h in settings:
        subset=[r for r in event_group if (r['anchor'],r['followup'],r['horizon'])==(anchor,pol,h)]
        values=np.array([r['mean_mean_delta_ap_skip_minus_trim10'] for r in subset],dtype=float)
        if len(values)<2:continue
        for k,r in enumerate(subset):
            loo.append({'anchor':anchor,'followup':pol,'horizon':h,
                        'held_out_event_id_POSTHOC':r['event_id_POSTHOC'],
                        'event_count':len(values),
                        'mean_with_all_events':float(values.mean()),
                        'mean_without_event':float(np.delete(values,k).mean())})
    # High drift threshold from unlabeled shift scores only, without GT.
    # Calculate within each anchor+seed to handle different model histories.
    shifts={}
    for r in paired:
        key=(r['anchor'],r['seed']);shifts.setdefault(key,[]).append(r['source_core_shift'])
    cutoffs={k:float(np.quantile(v,shift_quantile)) for k,v in shifts.items()}
    clean=[]
    for r in paired:
        # Strictly label-negative candidate, decision and all future H lines;
        # in addition exclude windows near any labeled event (gap +/- H).
        t=r['candidate_update_line'];h=r['horizon']; n=len(event_map)
        lo=max(0,t-h);hi=min(n,t+h+2)
        far_from_events=not np.any(event_map[lo:hi]>=0)
        if (r['candidate_gt_fraction_POSTHOC']>0 or
            r['decision_gt_fraction_POSTHOC']>0 or
            r['future_has_gt_positive_POSTHOC'] or not far_from_events):
            continue
        high=r['source_core_shift']>=cutoffs[(r['anchor'],r['seed'])]
        clean.append({**r,'drift_group_POSTHOC':'high_shift' if high else 'lower_shift',
                      'unlabeled_shift_quantile_cutoff':cutoffs[(r['anchor'],r['seed'])]})
    clean_summary=grouped_mean(clean,['anchor','followup','horizon','drift_group_POSTHOC'],
        ['delta_nll_annotated_normal_POSTHOC_skip_minus_trim10',
         'delta_nll_annotated_normal_POSTHOC_trim05_minus_trim10',
         'nll_annotated_normal_POSTHOC_skip','nll_annotated_normal_POSTHOC_trim05',
         'nll_annotated_normal_POSTHOC_trim10'])
    return event_seed,event_group,event_summary,loo,clean,clean_summary,cutoffs


def write_report(out,events,paired,event_group,event_summary,clean_summary,loo):
    lines=['HyperAD v11 — OLAY BAZLI VE NORMAL SAHNE TANISI',
           'SONUCLAR yalnızca gelişim sahnesinin POST-HOC analizidir.',
           'GT etiketleri hiçbir karar/güncellemeye girmez.',
           f'Birleştirilmiş GT olay sayısı: {len(events)}',
           f'Sabit adımla örneklenen eşleştirilmiş karar-ufuk kayıtları: {len(paired)}',
           f'Geçerli AP kayıtları: {sum(int(r["valid_ap"]) for r in paired)}',
           f'Olay atanmış geçerli AP kayıtları: {sum(int(r["valid_ap"] and r["event_id_POSTHOC"]>=0) for r in paired)}',
           '', 'OLAY DÜZEYİNDE ÖZET (Skip-Trim10 AP; pozitif Skip lehine):']
    for r in event_summary:
        lines.append(f"{r['anchor']} / {r['followup']} / H{r['horizon']} : "
                     f"n_events={r['n_windows']} event_mean="
                     f"{r['mean_mean_mean_delta_ap_skip_minus_trim10']:+.6f}, "
                     f"event_median={r['median_mean_mean_delta_ap_skip_minus_trim10']:+.6f}")
    lines+=['','ANOTASYONDA ANOMALİ OLMAYAN, OLAYLARDAN UZAK PENCERELER:',
            'Gaussian NLL düşük daha iyi; etiketli normal referansı POST-HOC filtreleme.',
            'Trim05-Trim10 delta NLL <0 ise Trim05 lehine.']
    for r in clean_summary:
        lines.append(f"{r['anchor']} / {r['followup']} / H{r['horizon']} / {r['drift_group_POSTHOC']}: "
                     f"n={r['n_windows']} deltaNLL(05-10)="
                     f"{r['mean_delta_nll_annotated_normal_POSTHOC_trim05_minus_trim10']:+.6f}")
    lines+=['','UYARILAR:',
            'Aynı olayın çok sayıda penceresi ve projeksiyon tohumu bağımsız değildir.',
            'Olaylar anotasyondan türetilmiştir ve etiketli test olarak değerlendirilmez.',
            'Etiketlerde işaretlenmeyen gerçek anomaliler normal grubuna karışabilir.',
            'Küçük olay sayısında güven aralığı veya genelleme iddiası kurulmaz.',
            'Sabit stride, event-gap, ufuklar ve tüm öğrenme ayarları kayıt altındadır.',
            'SNP ve Beach geliştirme sahneleridir; yöntem tasarlandıktan sonra yeni sahnede sınanmalı.',
            'Bu paket YENİ algoritma değil, kaynak V10 modelinin olay düzeyi tanısıdır.']
    (Path(out)/'diagnostic_report_TR.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines[:min(len(lines),20)]))


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
    p.add_argument('--stride',type=int,default=10)
    p.add_argument('--event-gap',type=int,default=10,
                   help='POST-HOC: group annotated-positive lines separated by <= gap negatives')
    p.add_argument('--high-shift-quantile',type=float,default=.75)
    p.add_argument('--max-windows',type=int,default=None,help='Quick tests only')
    p.add_argument('--max-lines',type=int,default=None,help='Quick tests only')
    p.add_argument('--normalise-md',action='store_true')
    p.add_argument('--output',default='results_beach_v11')
    a=p.parse_args()
    root=reference_repo(a.hyperad_root)
    x=np.load(a.hsi,mmap_mode='r',allow_pickle=False)
    mask=np.load(a.mask,mmap_mode='r',allow_pickle=False)
    if mask.ndim==3 and mask.shape[-1]==1:mask=mask[...,0]
    if x.ndim!=3 or mask.shape!=x.shape[:2]:raise ValueError(f'shape mismatch image {x.shape}, mask {mask.shape}')
    if a.warmup<2 or max(a.horizons)>=x.shape[1]-a.warmup-2:raise ValueError('Bad warmup/horizon')
    if a.projected_dims<2 or a.projected_dims>x.shape[2]:raise ValueError('Bad projected dims')
    if a.stride<1 or a.event_gap<0 or not 0<a.high_shift_quantile<1:raise ValueError('Invalid protocol args')
    if not 0<a.core_fraction<1 or a.penalty<0 or not 0<a.momentum<=1:raise ValueError('Invalid method settings')
    if len(set(a.seeds))!=len(a.seeds) or len(set(a.horizons))!=len(a.horizons):raise ValueError('Duplicate seeds/horizons')
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    event_map,events=build_event_map(mask,a.warmup,a.event_gap)
    save_csv(out/'event_registry_POSTHOC.csv',events)
    meta=vars(a).copy();meta.update(shape=list(x.shape),mask_shape=list(mask.shape),
       reference_erx_sha256=hashlib.sha256((root/'detectors'/'erx.py').read_bytes()).hexdigest(),
       protocol='Same prior, single action; GT ONLY posthoc grouping and outcome reporting')
    (out/'run_config.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    paired=[];decisions=[]
    print(f"V11 DATA={a.dataset_name} shape={x.shape} GT_events_POSTHOC={len(events)} seeds={a.seeds} anchors={a.anchors}",flush=True)
    with threadpool_limits(limits=1):
      for seed in a.seeds:
       for anchor in a.anchors:
        tic=time.perf_counter()
        rows,d,count=run_seed(x,mask,seed,a.warmup,a.projected_dims,a.momentum,a.core_fraction,
            a.penalty,anchor,sorted(a.horizons),a.stride,a.max_windows,a.normalise_md,a.max_lines)
        paired.extend(paired_v11(rows));decisions.extend(d)
        save_csv(out/'paired_window_effects.csv',paired)
        save_csv(out/'source_decisions.csv',decisions)
        print(f'seed={seed} anchor={anchor} windows={count} elapsed={time.perf_counter()-tic:.1f}s',flush=True)
    eg,ev,es,loo,clean,cs,cut=analyze(paired,events,event_map,sorted(a.horizons),a.high_shift_quantile)
    save_csv(out/'paired_window_effects.csv',paired)
    save_csv(out/'event_effects_per_seed_POSTHOC.csv',eg)
    save_csv(out/'event_effects_POSTHOC.csv',ev)
    save_csv(out/'event_summary_POSTHOC.csv',es)
    save_csv(out/'event_leave_one_out_POSTHOC.csv',loo)
    save_csv(out/'clean_drift_windows_POSTHOC.csv',clean)
    save_csv(out/'clean_drift_summary_POSTHOC.csv',cs)
    write_report(out,events,paired,ev,es,cs,loo)
    print('Outputs saved:',out.resolve())

if __name__=='__main__':main()
