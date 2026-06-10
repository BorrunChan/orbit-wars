"""Step 1 of direction-3: offline (state -> outcome) dataset + learnability gate.

For each replay, at sampled game-progress fractions, compute cheap per-player
features (all computable at inference from a single obs) for every ALIVE player,
labeled with that player's final outcome (won = rank 1). Then test whether a
simple model predicts win mid-game, and how that compares to a flow-diff-style
myopic proxy (production share alone). If the learned model adds little over the
myopic proxy, direction-3 won't help.
"""
import json, os, glob, math, statistics as S

REPLAYS = glob.glob('replays/2p/episode-*.json') + glob.glob('replays/4p/episode-*.json')

def feats(obs, p, nP, step):
    pl = obs['planets']
    def agg(pred):
        c=sh=pr=0.0
        for x in pl:
            if len(x)<7: continue
            if pred(x): c+=1; sh+=x[5]; pr+=x[6]
        return c,sh,pr
    mc,msh,mpr = agg(lambda x:x[1]==p)
    if mc==0: return None
    oc,osh,opr = agg(lambda x:x[1]>=0 and x[1]!=p)
    nc,nsh,npr = agg(lambda x:x[1]<0)  # neutral
    # per-opponent max planet count (leader gap)
    opp_pc={}
    for x in pl:
        if len(x)<7: continue
        if x[1]>=0 and x[1]!=p: opp_pc[x[1]]=opp_pc.get(x[1],0)+1
    leadpc=max(opp_pc.values()) if opp_pc else 0
    tot_pr=mpr+opr+1e-6; tot_sh=msh+osh+1e-6
    return {
        'step_frac': step/500.0,
        'my_pc': mc, 'my_sh': msh, 'my_pr': mpr,
        'prod_share': mpr/tot_pr, 'ship_share': msh/tot_sh,
        'pc_minus_leader': mc-leadpc,
        'neutral_pc': nc, 'neutral_share_left': nc/(len(pl)+1e-6),
        'nP': nP,
    }

def main():
    rows=[]  # (features dict, won)
    n_games=0
    for path in REPLAYS:
        try: r=json.load(open(path))
        except: continue
        st=r.get('steps');
        if not st: continue
        rew=r.get('rewards') or []
        nP=len(st[0])
        if not rew or len(rew)!=nP: continue
        maxr=max(x for x in rew if x is not None)
        n_games+=1
        N=len(st)
        for ph in (0.25,0.4,0.55,0.7):
            t=min(N-1,int(ph*(N-1)))
            obs=st[t][0]['observation']
            for p in range(nP):
                f=feats(obs,p,nP,obs.get('step',t))
                if f is None: continue
                won = 1 if (rew[p] is not None and rew[p]==maxr and [x for x in rew].count(maxr)==1) else 0
                rows.append((f,won))
    print(f'games={n_games}  samples={len(rows)}  win_rate={sum(w for _,w in rows)/len(rows):.2f}')
    json.dump([{**f,'won':w} for f,w in rows], open('rl/value_data.json','w'))
    print('wrote rl/value_data.json')
    # quick gate: AUC of single features vs won, by mid-game subset
    keys=['prod_share','ship_share','pc_minus_leader','my_pc','my_pr','neutral_share_left']
    def auc(key, subset):
        pos=[f[key] for f,w in subset if w]; neg=[f[key] for f,w in subset if not w]
        if not pos or not neg: return None
        # Mann-Whitney AUC estimate
        import random
        c=0; tot=0
        for _ in range(20000):
            a=random.choice(pos); b=random.choice(neg)
            c+= 1 if a>b else (0.5 if a==b else 0); tot+=1
        return c/tot
    import random as _r; _r.seed(0)
    mid=[(f,w) for f,w in rows if 0.4<=f['step_frac']<=0.6]
    print(f'\nmid-game (40-60%) single-feature AUC vs win  (n={len(mid)}):')
    for k in keys:
        a=auc(k,mid)
        if a: print(f'  {k:22s} AUC={a:.3f}')
main()
