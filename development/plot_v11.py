#!/usr/bin/env python3
"""Small descriptive figures for v11 outputs (no seaborn)."""
import argparse,csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def rows(p):
    if not p.exists():return []
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--results',required=True)
    p.add_argument('--anchor',default='shadow_full')
    p.add_argument('--followup',default='frozen')
    p.add_argument('--horizon',type=int,default=10)
    a=p.parse_args();root=Path(a.results)
    events=rows(root/'event_effects_POSTHOC.csv')
    ev=[r for r in events if r['anchor']==a.anchor and r['followup']==a.followup and int(r['horizon'])==a.horizon]
    if ev:
        ev=sorted(ev,key=lambda r:int(r['event_id_POSTHOC']))
        xs=[int(r['event_id_POSTHOC']) for r in ev]
        ys=[float(r['mean_mean_delta_ap_skip_minus_trim10']) for r in ev]
        fig,ax=plt.subplots(figsize=(max(7,len(xs)*.25),4.5))
        ax.bar(np.arange(len(xs)),ys);ax.axhline(0,linewidth=1,color='gray')
        ax.set_xticks(np.arange(len(xs)),list(map(str,xs)),rotation=90 if len(xs)>15 else 0)
        ax.set_xlabel('GT event ID (POST-HOC; adjacent lines grouped)')
        ax.set_ylabel('Event mean AP(skip) − AP(trim10)')
        ax.set_title(f'{a.anchor} / {a.followup} / H{a.horizon}: equal-event descriptive effects')
        fig.tight_layout();dest=root/'event_ap_by_event.png';fig.savefig(dest,dpi=170);plt.close(fig)
        print('Saved',dest)
    clean=rows(root/'clean_drift_summary_POSTHOC.csv')
    c=[r for r in clean if r['anchor']==a.anchor and r['followup']==a.followup and int(r['horizon'])==a.horizon]
    if c:
        labels=['lower_shift','high_shift'];ys=[];names=[]
        for key in labels:
            rr=next((r for r in c if r['drift_group_POSTHOC']==key),None)
            if rr:
                names.append(key+f" (n={rr['n_windows']})")
                ys.append(float(rr['mean_delta_nll_annotated_normal_POSTHOC_trim05_minus_trim10']))
        if ys:
            fig,ax=plt.subplots(figsize=(7,4))
            ax.bar(names,ys);ax.axhline(0,color='gray',linewidth=1)
            ax.set_ylabel('Mean Gaussian NLL(Trim05) − NLL(Trim10)')
            ax.set_title('Annotated-negative, event-remote windows (POST-HOC)')
            fig.tight_layout();dest=root/'clean_shift_nll.png';fig.savefig(dest,dpi=170);plt.close(fig)
            print('Saved',dest)
if __name__=='__main__':main()
