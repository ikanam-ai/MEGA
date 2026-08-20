from __future__ import annotations
import argparse
import glob
import itertools
import json
import os
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import krippendorff

def parse_args():
    repo_dir = Path(__file__).resolve().parents[1]
    default_data = os.environ.get("REGARD_DATA_ROOT", str(repo_dir))
    parser = argparse.ArgumentParser(
        description="Reproduce the statistics and figures reported in the REGARD paper."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(default_data),
        help="Path containing the downloaded REGARD data layout (default: REGARD_DATA_ROOT or this repository).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_dir / "analysis" / "output",
        help="Directory for CSV, JSON, and figure outputs.",
    )
    parser.add_argument(
        "--copy-figures-to",
        type=Path,
        default=None,
        help="Optional directory to receive the three publication figures.",
    )
    return parser.parse_args()

ARGS = parse_args()
ROOT = ARGS.data_root.expanduser().resolve()
OUT = ARGS.output_dir.expanduser().resolve()
OUT.mkdir(parents=True, exist_ok=True)
required = [
    ROOT / "data/generations/generations_raw.jsonl",
    ROOT / "data/scores/judge_scores_raw.jsonl",
    ROOT / "data/targets/aist_cis_targets_final.csv",
    ROOT / "vad_annotation_package/data/annotation_items.jsonl",
    ROOT / "vad_annotation_package/data/assignments.csv",
]
missing = [str(x) for x in required if not x.exists()]
if missing:
    raise FileNotFoundError("Missing required REGARD data files:\n" + "\n".join(missing))

SEED=20260713
rng=np.random.default_rng(SEED)
axes=['valence','arousal','dominance']

def boot_ci(x, n=10000, alpha=.05):
    x=np.asarray(pd.Series(x).dropna(),float)
    idx=rng.integers(0,len(x),(n,len(x)))
    means=x[idx].mean(axis=1)
    return float(np.quantile(means,alpha/2)),float(np.quantile(means,1-alpha/2))

def exact_perm(model_df, axis, role_col='model_role'):
    # model_df one row per model
    vals=model_df.set_index('model_id')[axis].to_dict()
    roles=model_df.set_index('model_id')[role_col].to_dict()
    mods=list(vals)
    local=[m for m in mods if roles[m]=='local']
    obs=np.mean([vals[m] for m in local])-np.mean([vals[m] for m in mods if m not in local])
    perm=[]
    for localset in itertools.combinations(mods,len(local)):
        ls=set(localset)
        perm.append(np.mean([vals[m] for m in ls])-np.mean([vals[m] for m in mods if m not in ls]))
    perm=np.array(perm)
    p2=float(np.mean(np.abs(perm)>=abs(obs)-1e-15))
    p_lower=float(np.mean(perm<=obs+1e-15))
    return float(obs),p2,p_lower,perm

def model_boot_ci(model_df,axis,n=10000):
    loc=model_df.loc[model_df.model_role=='local',axis].to_numpy(float)
    intl=model_df.loc[model_df.model_role=='international',axis].to_numpy(float)
    li=rng.integers(0,len(loc),(n,len(loc))); ii=rng.integers(0,len(intl),(n,len(intl)))
    vals=loc[li].mean(axis=1)-intl[ii].mean(axis=1)
    return float(np.quantile(vals,.025)),float(np.quantile(vals,.975))

gen=pd.read_json(ROOT/'data/generations/generations_raw.jsonl',lines=True)
sc=pd.read_json(ROOT/'data/scores/judge_scores_raw.jsonl',lines=True)
# filter each judge score by coverage, then average available judges
sf=sc[sc.target_coverage>=0.5].copy()
agg=sf.groupby('generation_id').agg(
    valence=('lm_valence','mean'), arousal=('lm_arousal','mean'), dominance=('lm_dominance','mean'),
    n_judges=('judge_model','nunique'),
    all_valid=('valid_for_vad_analysis','all'),
    any_refusal=('refusal_flag','any'), any_generic=('generic_answer_flag','any'),
    any_mismatch=('hallucination_or_mismatch_flag','any')
).reset_index()
df=gen.merge(agg,on='generation_id',how='left')

# Main target deltas and model means
main=df[df.prompt_id=='evaluative_stance'].copy()
rt=main.groupby(['target_id','target_family','model_role'])[axes].mean().reset_index()
piv=rt.pivot(index=['target_id','target_family'],columns='model_role',values=axes)
deltas=pd.DataFrame(index=piv.index)
for a in axes: deltas[a]=piv[(a,'local')]-piv[(a,'international')]
mm=main.groupby(['model_id','model_role'])[axes].mean().reset_index()

records=[]
for a in axes:
    x=deltas[a].dropna()
    obs,p2,p1,perm=exact_perm(mm,a)
    records.append(dict(axis=a,target_n=len(x),target_delta=x.mean(),target_ci_low=boot_ci(x)[0],target_ci_high=boot_ci(x)[1],target_sd=x.std(ddof=1),target_d=x.mean()/x.std(ddof=1),model_delta=obs,model_perm_p2=p2,model_perm_p1=p1,model_boot_low=model_boot_ci(mm,a)[0],model_boot_high=model_boot_ci(mm,a)[1]))
main_stats=pd.DataFrame(records)
main_stats.to_csv(OUT/'main_stats.csv',index=False)

# Main-prompt per-model diagnostics used in the manuscript table.
per_model_main = main.groupby(['model_id','model_role']).agg(
    valence=('valence','mean'), arousal=('arousal','mean'),
    dominance=('dominance','mean'), words=('response_word_len','mean'),
    generic_rate=('any_generic','mean'), mismatch_rate=('any_mismatch','mean'),
    refusal_rate=('any_refusal','mean'), valid_rate=('all_valid','mean'),
    n_generations=('generation_id','size'),
).reset_index()
per_model_main.to_csv(OUT/'per_model_main.csv',index=False)

# Axis decomposition of the target-level VAD profile distance.
d2=(deltas[axes]**2).sum(axis=1)
decomposition=[]
for a in axes:
    decomposition.append(dict(
        axis=a,
        signed_mean=float(deltas[a].mean()),
        mean_abs_delta=float(deltas[a].abs().mean()),
        mean_share_d2=float(((deltas[a]**2)/d2.replace(0,np.nan)).mean()),
        n_nonzero_distance=int((d2>0).sum()),
    ))
pd.DataFrame(decomposition).to_csv(OUT/'axis_decomposition.csv',index=False)

# Leave-one-model-out descriptive sensitivity for main-prompt arousal.
lomo=[]
for excluded in sorted(main.model_id.unique()):
    sub=main[main.model_id!=excluded]
    rr=sub.groupby(['target_id','model_role']).arousal.mean().unstack()
    x=(rr['local']-rr['international']).dropna()
    lo,hi=boot_ci(x)
    lomo.append(dict(excluded_model=excluded,n_targets=len(x),delta=x.mean(),ci_low=lo,ci_high=hi))
pd.DataFrame(lomo).to_csv(OUT/'lomo_arousal.csv',index=False)

# model-clustered target fixed-effects models via within-target demeaning
cluster=[]
for a in axes:
    d=main.dropna(subset=[a]).copy(); d['local_ind']=(d.model_role=='local').astype(float)
    d['y_dm']=d[a]-d.groupby('target_id')[a].transform('mean')
    d['x_dm']=d.local_ind-d.groupby('target_id').local_ind.transform('mean')
    res=smf.ols('y_dm ~ 0 + x_dm',d).fit(cov_type='cluster',cov_kwds={'groups':d.model_id,'use_correction':True},use_t=True)
    ci=res.conf_int().loc['x_dm']
    cluster.append(dict(axis=a,beta=res.params['x_dm'],se=res.bse['x_dm'],t=res.tvalues['x_dm'],p=res.pvalues['x_dm'],ci_low=ci[0],ci_high=ci[1]))
pd.DataFrame(cluster).to_csv(OUT/'cluster_target_fe.csv',index=False)

# prompt-specific, pooled, and interaction exact randomization
prompt_records=[]
for prompt,sub in df.groupby('prompt_id'):
    rt2=sub.groupby(['target_id','model_role'])[axes].mean().reset_index().pivot(index='target_id',columns='model_role',values=axes)
    mm2=sub.groupby(['model_id','model_role'])[axes].mean().reset_index()
    for a in axes:
        x=(rt2[(a,'local')]-rt2[(a,'international')]).dropna()
        obs,p2,p1,perm=exact_perm(mm2,a)
        lo,hi=boot_ci(x)
        prompt_records.append(dict(prompt=prompt,axis=a,target_n=len(x),target_delta=x.mean(),target_ci_low=lo,target_ci_high=hi,model_delta=obs,model_perm_p2=p2,model_perm_p1=p1))
# pooled equal over all generations/prompts, model mean
rpt=df.groupby(['prompt_id','target_id','model_role'])[axes].mean().reset_index()
pooled_target=rpt.groupby(['target_id','model_role'])[axes].mean().reset_index().pivot(index='target_id',columns='model_role',values=axes)
pooled_mm=df.groupby(['model_id','model_role'])[axes].mean().reset_index()
for a in axes:
    x=(pooled_target[(a,'local')]-pooled_target[(a,'international')]).dropna()
    obs,p2,p1,perm=exact_perm(pooled_mm,a)
    lo,hi=boot_ci(x)
    prompt_records.append(dict(prompt='pooled_equal_prompts',axis=a,target_n=len(x),target_delta=x.mean(),target_ci_low=lo,target_ci_high=hi,model_delta=obs,model_perm_p2=p2,model_perm_p1=p1))
prompt_stats=pd.DataFrame(prompt_records)
prompt_stats.to_csv(OUT/'prompt_stats.csv',index=False)

# exact prompt interaction statistic for arousal: model-fixed OLS incremental SSE/F
mp=df.groupby(['model_id','model_role','prompt_id']).arousal.mean().reset_index()
# Use difference of group effects heterogeneity: variance across prompt-specific group deltas around pooled, weighted equally
obs_d=[]
for p,g in mp.groupby('prompt_id'):
    obs_d.append(g.loc[g.model_role=='local','arousal'].mean()-g.loc[g.model_role=='international','arousal'].mean())
obs_d=np.array(obs_d)
obs_stat=float(np.sum((obs_d-obs_d.mean())**2))
mods=sorted(mp.model_id.unique()); local_n=4
perm_stats=[]
for localset in itertools.combinations(mods,local_n):
    ls=set(localset); ds=[]
    for p,g in mp.groupby('prompt_id'):
        ds.append(g.loc[g.model_id.isin(ls),'arousal'].mean()-g.loc[~g.model_id.isin(ls),'arousal'].mean())
    ds=np.array(ds); perm_stats.append(np.sum((ds-ds.mean())**2))
perm_stats=np.array(perm_stats)
interaction_p=float(np.mean(perm_stats>=obs_stat-1e-15))
json.dump({'prompt_order':sorted(mp.prompt_id.unique()),'observed_deltas':obs_d.tolist(),'heterogeneity_stat':obs_stat,'exact_perm_p':interaction_p},open(OUT/'prompt_interaction.json','w'),indent=2)

# category stats, consistent category labels
cat=[]
for fam,g in deltas.reset_index().groupby('target_family'):
    x=g.arousal.dropna(); lo,hi=boot_ci(x)
    tt=stats.ttest_1samp(x,0)
    cat.append(dict(category=fam,n=len(x),mean=x.mean(),sd=x.std(ddof=1),ci_low=lo,ci_high=hi,target_t=tt.statistic,target_p=tt.pvalue))
pd.DataFrame(cat).sort_values('mean').to_csv(OUT/'category_arousal.csv',index=False)

# quality/balanced sensitivity
roles=main[['model_id','model_role']].drop_duplicates().set_index('model_id').model_role.to_dict()
allmods=sorted(roles); locmods=[m for m in allmods if roles[m]=='local']; intmods=[m for m in allmods if roles[m]=='international']
sens=[]
def add_sens(label,sub,require_complete=False):
    if require_complete:
        w=sub.pivot(index='target_id',columns='model_id',values='arousal').dropna(subset=allmods)
        x=w[locmods].mean(axis=1)-w[intmods].mean(axis=1)
        n_responses=int(w[allmods].notna().sum().sum())
    else:
        r=sub.groupby(['target_id','model_role']).arousal.mean().unstack()
        x=(r['local']-r['international']).dropna()
        n_responses=int(sub.arousal.notna().sum())
    lo,hi=boot_ci(x)
    sens.append(dict(spec=label,n_responses=n_responses,n_targets=len(x),delta=x.mean(),ci_low=lo,ci_high=hi))
add_sens('baseline',main)
add_sens('all_8_models_complete',main,True)
add_sens('all_judges_valid',main[main.all_valid.astype('boolean').fillna(False)])
add_sens('no_refusal_or_mismatch',main[~(main.any_refusal.astype('boolean').fillna(True) | main.any_mismatch.astype('boolean').fillna(True))])
add_sens('no_generic_refusal_mismatch',main[~(main.any_generic.astype('boolean').fillna(True) | main.any_refusal.astype('boolean').fillna(True) | main.any_mismatch.astype('boolean').fillna(True))])
pd.DataFrame(sens).to_csv(OUT/'sensitivity_arousal.csv',index=False)

# Judge-specific effects
jrecs=[]
for j,sj0 in sc[(sc.prompt_id=='evaluative_stance')&(sc.target_coverage>=.5)].groupby('judge_model'):
    sj=sj0.merge(gen[['generation_id','model_role']],on='generation_id',how='left')
    for a in axes:
        r=sj.groupby(['target_id','model_role'])[f'lm_{a}'].mean().unstack()
        x=(r['local']-r['international']).dropna(); lo,hi=boot_ci(x)
        mmj=sj.groupby(['model_id','model_role'])[f'lm_{a}'].mean().reset_index().rename(columns={f'lm_{a}':a})
        obs,p2,p1,perm=exact_perm(mmj,a)
        jrecs.append(dict(judge=j,axis=a,n_targets=len(x),target_delta=x.mean(),ci_low=lo,ci_high=hi,model_delta=obs,model_perm_p2=p2))
pd.DataFrame(jrecs).to_csv(OUT/'judge_specific.csv',index=False)

# Human clean expected assignments only
items=pd.read_json(ROOT/'vad_annotation_package/data/annotation_items.jsonl',lines=True)
assign=pd.read_csv(ROOT/'vad_annotation_package/data/assignments.csv')
expected=set(zip(assign.username,assign.item_id))
rows=[]
for f in glob.glob(str(ROOT/'vad_annotation_package/annotations/*/*.json')):
    z=json.load(open(f))
    if (z.get('username'),z.get('item_id')) in expected:
        rows.append(z)
ad=pd.DataFrame(rows)
assert len(ad)==900 and ad.item_id.nunique()==300 and ad.groupby('item_id').size().eq(3).all()
hm=ad.groupby('item_id')[[f'human_{a}' for a in axes]].mean().reset_index().merge(items[['item_id','item_metadata','model_id','target_id','lm_valence','lm_arousal','lm_dominance']],on='item_id')
hm['group']=hm.item_metadata.map(lambda x:x['group'])
hrecs=[]
for a in axes:
    h=hm[f'human_{a}']; l=hm[f'lm_{a}']
    r,p=stats.pearsonr(l,h)
    mae=np.abs(l-h).mean()
    # build reliability matrix annotators x items, NaN missing
    mat=ad.pivot(index='username',columns='item_id',values=f'human_{a}').to_numpy(float)
    alpha=float(krippendorff.alpha(reliability_data=mat,level_of_measurement='interval'))
    ru=h[hm.group=='ru']; intl=h[hm.group=='intl']
    delta=ru.mean()-intl.mean(); welch=stats.ttest_ind(ru,intl,equal_var=False)
    # bootstrap groups independently
    bri=rng.integers(0,len(ru),(10000,len(ru))); bii=rng.integers(0,len(intl),(10000,len(intl)))
    b=ru.to_numpy()[bri].mean(axis=1)-intl.to_numpy()[bii].mean(axis=1)
    hlo,hhi=np.quantile(b,[.025,.975])
    lru=l[hm.group=='ru']; lint=l[hm.group=='intl']; ldelta=lru.mean()-lint.mean(); lp=stats.ttest_ind(lru,lint,equal_var=False).pvalue
    hrecs.append(dict(axis=a,pearson_r=r,pearson_p=p,mae=mae,kripp_alpha=alpha,human_group_delta=delta,human_ci_low=hlo,human_ci_high=hhi,human_welch_p=welch.pvalue,lm_group_delta_same_sample=ldelta,lm_welch_p=lp))
human_stats=pd.DataFrame(hrecs)
human_stats.to_csv(OUT/'human_stats.csv',index=False)

# Group comparison of human-minus-judge residuals requested as an additional
# calibration diagnostic. Tests are exploratory; Holm adjustment covers the
# three VAD axes.
residual_records=[]
for a in axes:
    residual=hm[f'human_{a}']-hm[f'lm_{a}']
    ru=residual[hm.group=='ru']
    intl=residual[hm.group=='intl']
    welch=stats.ttest_ind(ru,intl,equal_var=False)
    residual_records.append(dict(
        axis=a,
        n_ru=len(ru),
        n_comparison=len(intl),
        mean_residual_ru=ru.mean(),
        mean_residual_comparison=intl.mean(),
        residual_delta=ru.mean()-intl.mean(),
        welch_t=welch.statistic,
        welch_p=welch.pvalue,
    ))
residual_df=pd.DataFrame(residual_records)
residual_df['holm_p']=multipletests(residual_df.welch_p,method='holm')[1]
residual_df.to_csv(OUT/'human_residual_by_group.csv',index=False)
# matched target subset both groups (target group average)
matched=[]
for a in axes:
    t=hm.groupby(['target_id','group'])[[f'human_{a}',f'lm_{a}']].mean().unstack()
    t=t.dropna()
    hx=t[(f'human_{a}','ru')]-t[(f'human_{a}','intl')]
    lx=t[(f'lm_{a}','ru')]-t[(f'lm_{a}','intl')]
    matched.append(dict(axis=a,n_targets=len(t),human_delta=hx.mean(),human_p=stats.ttest_1samp(hx,0).pvalue,lm_delta=lx.mean(),lm_p=stats.ttest_1samp(lx,0).pvalue))
pd.DataFrame(matched).to_csv(OUT/'human_matched_targets.csv',index=False)

# Data/source counts
targets=pd.read_csv(ROOT/'data/targets/aist_cis_targets_final.csv')
summary={
 'n_targets':len(targets),
 'countries':targets.country.value_counts().sort_index().to_dict(),
 'sources':targets.source_dataset.value_counts().to_dict(),
 'qid_nonmissing':int(targets.wikidata_qid.notna().sum()),
 'qid_unique':int(targets.wikidata_qid.nunique()),
 'boundary_cases':int(targets.final_needs_review.sum()),
 'generation_min':str(gen.timestamp.min()),'generation_max':str(gen.timestamp.max()),
 'n_generations':len(gen),'n_scores':len(sc),'n_paired_scored':int(sc.groupby('generation_id').judge_model.nunique().eq(2).sum()),
}
json.dump(summary,open(OUT/'data_summary.json','w'),indent=2,ensure_ascii=False)

# Figure 1: per-model main VAD, correct data
labels={'avibe':'AVIBE','yandexgpt':'YandexGPT','gigachat':'GigaChat','tpro':'T-pro','glm':'GLM','gemma2_27b':'Gemma 4','qwen25_14b':'Qwen2.5','ministral_14b':'Ministral 3'}
order=['yandexgpt','gigachat','tpro','avibe','glm','gemma2_27b','qwen25_14b','ministral_14b']
plot=mm.set_index('model_id').loc[order].reset_index()
fig,axs=plt.subplots(1,3,figsize=(11.2,3.5),sharey=False)
for ax,a in zip(axs,axes):
    y=np.arange(len(plot)); vals=plot[a]
    ax.barh(y,vals)
    ax.set_yticks(y, [labels[x] for x in plot.model_id] if a=='valence' else [])
    ax.invert_yaxis(); ax.set_xlim(0,1); ax.set_title(a.capitalize()); ax.set_xlabel('Mean score')
    ax.axhline(3.5,linewidth=.8)
fig.suptitle('Per-model VAD means, evaluative-stance prompt (500 targets)',y=1.02)
fig.tight_layout()
fig.savefig(OUT/'fig_permodel_vad.png',dpi=220,bbox_inches='tight')
fig.savefig(OUT/'fig_permodel_vad.pdf',bbox_inches='tight')
plt.close(fig)

# Figure prompt effects + permutation distribution
ps=prompt_stats[(prompt_stats.axis=='arousal') & (prompt_stats.prompt!='pooled_equal_prompts')].copy()
po=['evaluative_stance','neutral_descriptive','evaluative_paraphrase']; ps=ps.set_index('prompt').loc[po].reset_index()
fig,axs=plt.subplots(1,2,figsize=(11.2,3.8))
ax=axs[0]
y=np.arange(len(ps)); x=ps.target_delta.to_numpy(); lo=x-ps.target_ci_low.to_numpy(); hi=ps.target_ci_high.to_numpy()-x
ax.errorbar(x,y,xerr=np.vstack([lo,hi]),fmt='o',capsize=4)
ax.axvline(0,linewidth=.8); ax.set_yticks(y,['Evaluative stance','Neutral descriptive','Evaluative paraphrase']); ax.invert_yaxis(); ax.set_xlabel('Target-bank mean Δ arousal (RU - comparison)'); ax.set_title('(a) Prompt-specific target-bank estimates')
obs,p2,p1,perm=exact_perm(mm,'arousal')
ax=axs[1]; bins=np.linspace(perm.min()-.005,perm.max()+.005,15); ax.hist(perm,bins=bins,edgecolor='black'); ax.axvline(obs,linewidth=2,label=f'Observed Δ={obs:.3f}'); ax.axvline(-obs,linestyle='--',linewidth=1); ax.set_xlabel('Δ arousal under 4-vs-4 model label assignment'); ax.set_ylabel('Number of assignments'); ax.set_title(f'(b) Exact model-label permutation (p={p2:.2f})'); ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(OUT/'fig_inference.png',dpi=220,bbox_inches='tight'); fig.savefig(OUT/'fig_inference.pdf',bbox_inches='tight'); plt.close(fig)

# Distribution of target-level VAD contrasts for the decomposition section.
fig,ax=plt.subplots(figsize=(7.2,3.2))
ax.boxplot([deltas[a].dropna().to_numpy() for a in axes],
           tick_labels=['Valence','Arousal','Dominance'],showfliers=False)
ax.axhline(0,linewidth=1,linestyle='--')
ax.set_ylabel('Russian-developed minus comparison')
ax.set_title('Per-target VAD contrasts under the main prompt')
fig.tight_layout()
fig.savefig(OUT/'fig_target_deltas.png',dpi=220,bbox_inches='tight')
fig.savefig(OUT/'fig_target_deltas.pdf',bbox_inches='tight')
plt.close(fig)

# Human validation: scatter with calibration and group delta inset-style as 2 panels
fig,axs=plt.subplots(1,3,figsize=(11.2,3.5))
for ax,a in zip(axs,axes):
    x=hm[f'lm_{a}']; y=hm[f'human_{a}'];
    ax.scatter(x,y,s=10,alpha=.45); ax.plot([0,1],[0,1],linestyle='--',linewidth=.8)
    slope,intercept=np.polyfit(x,y,1); xx=np.array([0,1]); ax.plot(xx,intercept+slope*xx,linewidth=1.2)
    rec=human_stats.set_index('axis').loc[a]
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.set_title(f'{a.capitalize()}  r={rec.pearson_r:.2f}, α={rec.kripp_alpha:.2f}')
    ax.set_xlabel('LM mean');
axs[0].set_ylabel('Human mean')
fig.tight_layout(); fig.savefig(OUT/'fig_human_validation.png',dpi=220,bbox_inches='tight'); fig.savefig(OUT/'fig_human_validation.pdf',bbox_inches='tight'); plt.close(fig)

print(main_stats.to_string(index=False))
print('\nCluster:\n',pd.DataFrame(cluster).to_string(index=False))
print('\nPrompt:\n',prompt_stats[prompt_stats.axis=='arousal'].to_string(index=False))
print('\nInteraction',interaction_p,obs_d)
print('\nJudge:\n',pd.DataFrame(jrecs).query("axis=='arousal'").to_string(index=False))
print('\nHuman:\n',human_stats.to_string(index=False))
print('\nSensitivity:\n',pd.DataFrame(sens).to_string(index=False))

if ARGS.copy_figures_to is not None:
    dest = ARGS.copy_figures_to.expanduser().resolve()
    dest.mkdir(parents=True, exist_ok=True)
    for name in ["fig_permodel_vad.png", "fig_permodel_vad.pdf", "fig_inference.png", "fig_inference.pdf", "fig_target_deltas.png", "fig_target_deltas.pdf", "fig_human_validation.png", "fig_human_validation.pdf"]:
        shutil.copy2(OUT / name, dest / name)
    print(f"Copied publication figures to {dest}")

print(f"Wrote reproducibility outputs to {OUT}")
