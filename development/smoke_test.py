"""V11 invariants and v10 outcome compatibility (no temp file locks)."""
import numpy as np
from run_v11 import build_event_map,score_and_nll,run_seed,paired_v11,analyze,write_report
from run_v10 import run_seed as reference_run_seed,roll_forward,make_candidates
from run_shadow import prepare_warmup
from pathlib import Path
from tempfile import TemporaryDirectory

def main():
    rng=np.random.default_rng(20261018)
    pixels,lines,bands,warmup,dims=36,91,9,13,5
    x=rng.normal(size=(pixels,lines,bands)).astype(np.float32)
    y=np.zeros((pixels,lines),dtype=np.int8)
    for a,b in [(26,32),(42,50),(68,72)]:
        x[4:13,a:b+1,:4]+=2.8;y[4:13,a:b+1]=1
    x[:,:,4]+=np.arange(lines)[None,:]*.01
    map_,events=build_event_map(y,warmup,3)
    assert len(events)==3 and map_[27]==0 and map_[41]==-1
    proj,mu,cov=prepare_warmup(x,0,warmup,dims,.1)
    z=np.stack([proj.transform(x[:,t].astype('float64')) for t in range(warmup,lines)])
    c=make_candidates(z[4],(mu,cov),.1)
    for act in c:
        for follow in ('frozen','delayed_trim10'):
            pred,nll=score_and_nll(z,5,c[act],.1,10,follow)
            expected=roll_forward(z,5,c[act],.1,10,follow)
            assert np.array_equal(pred,expected) and np.isfinite(nll).all()
    allpaired=[]
    for anchor in ('shadow_full','delayed_trim10'):
        rows,dec,n=run_seed(x,y,0,warmup,dims,.1,.8,.05,anchor,[1,5,10],4)
        ref,refdec,_,refn=reference_run_seed(x,y,0,warmup,dims,.1,.8,.05,
            anchor,10,[1,5,10],4,None)
        assert n==refn and len(rows)==len(ref)
        for r,s in zip(rows,ref):
            for name in ('ap','auroc','fpr90'):
                assert (np.isnan(r[name]) and np.isnan(s[name])) or r[name]==s[name],(name,r,s)
        y2=np.zeros_like(y)
        _,dec2,_=run_seed(x,y2,0,warmup,dims,.1,.8,.05,anchor,[1,5,10],4)
        assert [(r['source_action'],r['decision_line']) for r in dec] == [(r['source_action'],r['decision_line']) for r in dec2]
        paired=paired_v11(rows)
        assert len(paired)==n*2*3
        allpaired.extend(paired)
    es,eg,sm,loo,clean,cs,_=analyze(allpaired,events,map_,[1,5,10])
    assert es and eg and sm and clean and cs and loo
    with TemporaryDirectory() as p:
        write_report(p,events,allpaired,eg,sm,cs,loo)
        assert (Path(p)/'diagnostic_report_TR.txt').exists()
    print('PASS: event clustering separates disjoint abnormal intervals')
    print('PASS: all 3 actions, 2 followups and 3 horizons match V10 predictions/metrics')
    print('PASS: decisions unchanged when GT masks change')
    print('PASS: event-level grouping, leave-one-event-out and clean-shift NLL summaries')
    print('PASS: Turkish report generated')
if __name__=='__main__':main()
