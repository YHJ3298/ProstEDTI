# -*- coding: utf-8 -*-
"""
Final CV training code: SHAP Top320 + Pseudo-WkNNIR + PU reliability + ENN + XGB/LGBM.
Compact outputs: CV metrics, OOF predictions, fold bundles, process summaries, and two SHAP figures.
"""
import argparse, hashlib, json, warnings
from pathlib import Path
from collections import Counter, defaultdict

import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_curve, auc, matthews_corrcoef, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from imblearn.under_sampling import EditedNearestNeighbours
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import shap

POSITIVE_LABEL = 1
NEGATIVE_LABEL = 0
PRIOR_EPS = 1e-8
PRIOR_SIM_CLIP_ZERO = True

XGB_PARAMS = dict(n_estimators=600, max_depth=7, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9,
                  min_child_weight=1, gamma=0, reg_alpha=0, reg_lambda=1, objective='binary:logistic',
                  eval_metric='logloss', tree_method='hist', n_jobs=-1)
LGBM_PARAMS = dict(n_estimators=600, learning_rate=0.05, num_leaves=127, max_depth=-1, min_child_samples=20,
                   subsample=0.9, subsample_freq=1, colsample_bytree=0.9, reg_alpha=0, reg_lambda=1,
                   objective='binary', metric='binary_logloss', n_jobs=-1, verbosity=-1)
SHAP_XGB_PARAMS = dict(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                       min_child_weight=1, gamma=0, reg_alpha=0, reg_lambda=1, objective='binary:logistic',
                       eval_metric='logloss', tree_method='hist', n_jobs=-1)

def str2bool(v):
    if isinstance(v, bool): return v
    v = str(v).lower().strip()
    if v in ['true','1','yes','y']: return True
    if v in ['false','0','no','n']: return False
    raise argparse.ArgumentTypeError('Boolean expected: true/false')

def make_json_safe(obj):
    if isinstance(obj, dict): return {str(k): make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)): return [make_json_safe(x) for x in obj]
    if isinstance(obj, np.ndarray): return make_json_safe(obj.tolist())
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, np.floating): return None if np.isnan(obj) else float(obj)
    if isinstance(obj, np.bool_): return bool(obj)
    if isinstance(obj, float) and np.isnan(obj): return None
    return obj

def save_json(obj, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(make_json_safe(obj), f, ensure_ascii=False, indent=2)

def cdict(c): return {str(k): int(v) for k, v in c.items()}
def safe_div(a,b): return a/b if b != 0 else np.nan

def calc_metrics(y_true, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    sn = safe_div(tp, tp+fn); sp = safe_div(tn, tn+fp)
    acc = accuracy_score(y_true, y_pred); mcc = matthews_corrcoef(y_true, y_pred)
    if len(np.unique(y_true)) == 2:
        auc_score = roc_auc_score(y_true, y_prob)
        p, r, _ = precision_recall_curve(y_true, y_prob, pos_label=1)
        aupr_score = auc(r, p)
    else:
        auc_score = np.nan; aupr_score = np.nan
    return dict(SN=float(sn), SP=float(sp), ACC=float(acc), MCC=float(mcc), AUC=float(auc_score), AUPR=float(aupr_score),
                TP=int(tp), FP=int(fp), FN=int(fn), TN=int(tn))

def load_dataset(csv_path, label_is_last_column=True, label_column='label'):
    df = pd.read_csv(csv_path)
    df = df.drop(columns=[c for c in df.columns if str(c).startswith('Unnamed')], errors='ignore')
    if label_is_last_column:
        feature_df = df.iloc[:, :-1].copy(); y = df.iloc[:, -1].values; label_col = df.columns[-1]
    else:
        feature_df = df.drop(columns=[label_column]).copy(); y = df[label_column].values; label_col = label_column
    non_numeric = feature_df.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric: raise ValueError(f'Non-numeric feature columns found: {non_numeric}')
    if feature_df.isnull().any().any(): raise ValueError('Missing values found.')
    X = feature_df.values.astype(np.float32); feature_names = feature_df.columns.tolist()
    info = dict(path=str(csv_path), n_samples=int(X.shape[0]), n_features=int(X.shape[1]), label_column=str(label_col), label_distribution=cdict(Counter(y)))
    return X, y.astype(int), feature_names, info

def get_feature_pools(feature_names):
    mp = {n:i for i,n in enumerate(feature_names)}
    mol = [mp[f'mol2vec_{i:03d}'] for i in range(64) if f'mol2vec_{i:03d}' in mp]
    pro = [mp[f'feature_{i}'] for i in range(1024) if f'feature_{i}' in mp]
    return mol, pro

def norm_shap_values(vals):
    if isinstance(vals, list): vals = vals[1] if len(vals) == 2 else vals[0]
    vals = np.asarray(vals)
    if vals.ndim == 3: vals = vals[:,:,1] if vals.shape[-1] >= 2 else vals[:,:,0]
    if vals.ndim != 2: raise ValueError(f'Unable to recognize SHAP output dimensions: {vals.shape}')
    return vals

def build_shap_model(seed):
    p = SHAP_XGB_PARAMS.copy(); p['random_state'] = int(seed)
    return XGBClassifier(**p)

def shap_select(X_scaled, y, feature_names, top_protein_k=320, shap_max_samples=5000, random_state=42):
    mol_idx, protein_idx = get_feature_pools(feature_names)
    if not mol_idx or not protein_idx: raise ValueError('Unable to identify mol2vec or ProstT5 feature columns.')
    if shap_max_samples and len(y) > shap_max_samples:
        rng = np.random.default_rng(int(random_state)); sample_idx = rng.choice(np.arange(len(y)), size=int(shap_max_samples), replace=False)
    else:
        sample_idx = np.arange(len(y))
    Xcand = X_scaled[:, protein_idx]
    model = build_shap_model(random_state); model.fit(Xcand, y)
    explainer = shap.TreeExplainer(model)
    vals = norm_shap_values(explainer.shap_values(Xcand[sample_idx]))
    imp = np.abs(vals).mean(axis=0)
    imp_df = pd.DataFrame({'feature':[feature_names[i] for i in protein_idx], 'original_index':protein_idx, 'shap_importance':imp})
    imp_df = imp_df.sort_values('shap_importance', ascending=False).reset_index(drop=True)
    selected_protein = imp_df.head(int(top_protein_k))['original_index'].astype(int).tolist()
    selected = []
    for idx in mol_idx + selected_protein:
        if int(idx) not in selected: selected.append(int(idx))
    return selected, [feature_names[i] for i in selected], imp_df, dict(shap_explained_samples=int(len(sample_idx)), n_selected_base_features=len(selected))

def save_shap_figures(X_scaled, y, feature_names, out_dir, top_protein_k=320, shap_max_samples=5000, random_state=42, max_display=30):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    _, protein_idx = get_feature_pools(feature_names)
    if shap_max_samples and len(y) > shap_max_samples:
        rng = np.random.default_rng(int(random_state)); sample_idx = rng.choice(np.arange(len(y)), size=int(shap_max_samples), replace=False)
    else:
        sample_idx = np.arange(len(y))
    Xcand = X_scaled[:, protein_idx]; Xs = Xcand[sample_idx]
    names = [feature_names[i] for i in protein_idx]
    model = build_shap_model(random_state); model.fit(Xcand, y)
    explainer = shap.TreeExplainer(model)
    vals = norm_shap_values(explainer.shap_values(Xs))
    imp_df = pd.DataFrame({'feature':names, 'original_index':protein_idx, 'shap_importance':np.abs(vals).mean(axis=0)})
    imp_df = imp_df.sort_values('shap_importance', ascending=False).reset_index(drop=True)
    imp_df.to_csv(out_dir/'final_shap_protein_importance.csv', index=False, encoding='utf-8-sig')
    imp_df.head(int(top_protein_k)).to_csv(out_dir/f'final_shap_top{top_protein_k}_protein_features.csv', index=False, encoding='utf-8-sig')
    Xdf = pd.DataFrame(Xs, columns=names)
    plt.figure(); shap.summary_plot(vals, Xdf, show=False, max_display=int(max_display)); plt.title('SHAP Feature Summary (Beeswarm)'); plt.tight_layout(); plt.savefig(out_dir/'SHAP_Feature_Summary_Beeswarm.png', dpi=300, bbox_inches='tight'); plt.close()
    plt.figure(); shap.summary_plot(vals, Xdf, plot_type='bar', show=False, max_display=int(max_display)); plt.title('SHAP Feature Importance (Bar)'); plt.tight_layout(); plt.savefig(out_dir/'SHAP_Feature_Importance_Bar.png', dpi=300, bbox_inches='tight'); plt.close()
    return imp_df

def hash_vector(vec, decimals=6):
    arr = np.round(np.asarray(vec, dtype=np.float32), decimals=int(decimals))
    return hashlib.sha1(arr.tobytes()).hexdigest()

def vector_hash_ids(X, decimals=6): return np.array([hash_vector(row, decimals) for row in X], dtype=object)

def build_prototypes(ids, vecs):
    groups = defaultdict(list)
    for i,g in enumerate(ids): groups[g].append(i)
    pids, pvecs = [], []
    for g, idxs in groups.items(): pids.append(g); pvecs.append(np.mean(vecs[idxs], axis=0))
    return np.array(pids, dtype=object), np.asarray(pvecs, dtype=np.float32)

def fit_nn(proto_vecs, k):
    if len(proto_vecs) == 0: return None
    nn = NearestNeighbors(n_neighbors=max(1, min(int(k), len(proto_vecs))), metric='cosine', algorithm='auto')
    nn.fit(proto_vecs); return nn

def cosine_neighbors(nn, proto_ids, query_vecs):
    if nn is None or len(proto_ids) == 0: return [], []
    d, idx = nn.kneighbors(query_vecs, return_distance=True); sims = 1.0 - d
    if PRIOR_SIM_CLIP_ZERO: sims = np.maximum(sims, 0.0)
    return [[proto_ids[j] for j in row] for row in idx], sims

class PseudoWkNNIRPrior:
    def __init__(self, k_drug=10, k_target=10, hash_decimals=6):
        self.k_drug=int(k_drug); self.k_target=int(k_target); self.hash_decimals=int(hash_decimals)
    def fit(self, drug_vecs, target_vecs, y):
        self.drug_ids_ = vector_hash_ids(drug_vecs, self.hash_decimals)
        self.target_ids_ = vector_hash_ids(target_vecs, self.hash_decimals)
        self.proto_drug_ids_, self.proto_drug_vecs_ = build_prototypes(self.drug_ids_, drug_vecs)
        self.proto_target_ids_, self.proto_target_vecs_ = build_prototypes(self.target_ids_, target_vecs)
        self.drug_nn_ = fit_nn(self.proto_drug_vecs_, self.k_drug)
        self.target_nn_ = fit_nn(self.proto_target_vecs_, self.k_target)
        pv = defaultdict(list)
        for d,t,label in zip(self.drug_ids_, self.target_ids_, y): pv[(d,t)].append(float(label))
        self.pair_score_ = {k: float(np.mean(v)) for k,v in pv.items()}
        self.train_positive_rate_ = float(np.mean(y))
        self.meta_ = dict(n_train_samples_for_prior=int(len(y)), unique_drug_count=int(len(self.proto_drug_ids_)), unique_target_count=int(len(self.proto_target_ids_)), unique_pair_count=int(len(self.pair_score_)), train_positive_rate=self.train_positive_rate_, k_drug=self.k_drug, k_target=self.k_target, hash_decimals=self.hash_decimals, drug_repeat_ratio=float(1.0-len(self.proto_drug_ids_)/max(len(y),1)), target_repeat_ratio=float(1.0-len(self.proto_target_ids_)/max(len(y),1)))
        return self
    def transform(self, drug_vecs, target_vecs):
        qd = vector_hash_ids(drug_vecs, self.hash_decimals); qt = vector_hash_ids(target_vecs, self.hash_decimals)
        dn, ds = cosine_neighbors(self.drug_nn_, self.proto_drug_ids_, drug_vecs)
        tn, ts = cosine_neighbors(self.target_nn_, self.proto_target_ids_, target_vecs)
        rows=[]
        for i in range(len(drug_vecs)):
            d_id, t_id = qd[i], qt[i]
            d_num=d_den=0.0; d_known=d_pos=d_neg=0
            for nd,w in zip(dn[i], ds[i]):
                key=(nd,t_id)
                if key in self.pair_score_:
                    val=self.pair_score_[key]; d_num += float(w)*val; d_den += float(w); d_known += 1; d_pos += int(val>=0.5); d_neg += int(val<0.5)
            drug_score = d_num/(d_den+PRIOR_EPS) if d_known>0 else self.train_positive_rate_
            t_num=t_den=0.0; t_known=t_pos=t_neg=0
            for nt,w in zip(tn[i], ts[i]):
                key=(d_id,nt)
                if key in self.pair_score_:
                    val=self.pair_score_[key]; t_num += float(w)*val; t_den += float(w); t_known += 1; t_pos += int(val>=0.5); t_neg += int(val<0.5)
            target_score = t_num/(t_den+PRIOR_EPS) if t_known>0 else self.train_positive_rate_
            dual_num=dual_den=0.0; dual_known=dual_pos=dual_neg=0
            for nd,wd in zip(dn[i], ds[i]):
                for nt,wt in zip(tn[i], ts[i]):
                    key=(nd,nt)
                    if key in self.pair_score_:
                        val=self.pair_score_[key]; w=float(wd)*float(wt); dual_num += w*val; dual_den += w; dual_known += 1; dual_pos += int(val>=0.5); dual_neg += int(val<0.5)
            dual_score = dual_num/(dual_den+PRIOR_EPS) if dual_known>0 else self.train_positive_rate_
            known = d_known + t_known + dual_known; pos = d_pos+t_pos+dual_pos; neg=d_neg+t_neg+dual_neg
            max_score = max(drug_score, target_score, dual_score); mean_score=float(np.mean([drug_score,target_score,dual_score]))
            rows.append([drug_score,target_score,dual_score,max_score,mean_score,pos/max(known,1),neg/max(known,1),known/max(len(dn[i])+len(tn[i])+len(dn[i])*len(tn[i]),1),float(d_known),float(t_known),float(dual_known)])
        cols=['prior_drug_neighbor_score','prior_target_neighbor_score','prior_dual_neighbor_score','prior_max_score','prior_mean_score','prior_local_positive_density','prior_local_negative_density','prior_local_coverage','prior_drug_known_count','prior_target_known_count','prior_dual_known_count']
        return np.asarray(rows,dtype=np.float32), cols

def oof_prior(drug, target, y, k_drug, k_target, decimals, inner_folds, seed):
    y=np.asarray(y); n=len(y); pf=None; cols=None; metas=[]
    n_inner=max(2, min(int(inner_folds), min(Counter(y).values())))
    skf=StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=int(seed))
    for i,(fi,hi) in enumerate(skf.split(drug,y), start=1):
        m=PseudoWkNNIRPrior(k_drug,k_target,decimals).fit(drug[fi], target[fi], y[fi])
        part, cols = m.transform(drug[hi], target[hi])
        if pf is None: pf=np.zeros((n, part.shape[1]), dtype=np.float32)
        pf[hi]=part; meta=m.meta_.copy(); meta['inner_fold']=i; metas.append(meta)
    return pf, cols, pd.DataFrame(metas)

def fit_prior_transform(drug, target, y, qdrug, qtarget, k_drug, k_target, decimals):
    m=PseudoWkNNIRPrior(k_drug,k_target,decimals).fit(drug,target,y)
    pf, cols = m.transform(qdrug,qtarget)
    return m, pf, cols, m.meta_

def pu_weights(y, prior, cols, use=True, threshold=0.75, min_weight=0.5, continuous=True):
    y=np.asarray(y); w=np.ones(len(y),dtype=np.float32); score=pd.DataFrame(prior,columns=cols)['prior_max_score'].values.astype(float)
    neg=y==0; sus=neg & (score>=float(threshold))
    if use:
        if continuous:
            scaled=np.clip((score-float(threshold))/max(1.0-float(threshold), PRIOR_EPS),0,1)
            nw=1.0-scaled*(1.0-float(min_weight)); w[neg]=np.minimum(w[neg], nw[neg])
        else:
            w[sus]=float(min_weight)
    return w, dict(use_pu_reliability=bool(use), pu_suspicious_threshold=float(threshold), pu_min_neg_weight=float(min_weight), pu_continuous_downweight=bool(continuous), n_negative=int(np.sum(neg)), n_suspicious_negative=int(np.sum(sus)), suspicious_negative_ratio_in_negative=float(np.sum(sus)/max(np.sum(neg),1)), weight_mean=float(np.mean(w)), weight_min=float(np.min(w)), weight_max=float(np.max(w)))

def apply_enn(X, y, w, k=5, kind='mode'):
    enn=EditedNearestNeighbours(sampling_strategy='majority', n_neighbors=int(k), kind_sel=str(kind), n_jobs=-1)
    Xr, yr = enn.fit_resample(X,y)
    if hasattr(enn,'sample_indices_'):
        kept=np.asarray(enn.sample_indices_, dtype=int); wr=w[kept]
    else:
        kept=None; wr=np.ones(len(yr), dtype=np.float32)
    info=dict(before_enn_counter=cdict(Counter(y)), after_enn_counter=cdict(Counter(yr)), before_enn_n=int(len(y)), after_enn_n=int(len(yr)), enn_n_neighbors=int(k), enn_kind_sel=str(kind), has_sample_indices=bool(kept is not None), weight_after_enn_mean=float(np.mean(wr)), weight_after_enn_min=float(np.min(wr)), weight_after_enn_max=float(np.max(wr)))
    return Xr, yr, wr, kept, info

def fit_softvote(X,y,w=None,seed=42):
    px=XGB_PARAMS.copy(); px['random_state']=int(seed)
    pl=LGBM_PARAMS.copy(); pl['random_state']=int(seed)
    xgb=XGBClassifier(**px); lgbm=LGBMClassifier(**pl)
    if w is None: xgb.fit(X,y); lgbm.fit(X,y)
    else: xgb.fit(X,y,sample_weight=w); lgbm.fit(X,y,sample_weight=w)
    return {'xgb':xgb,'lgbm':lgbm}

def pred_softvote(model, X, wx=0.5, wl=0.5):
    p1=model['xgb'].predict_proba(X)[:,1]; p2=model['lgbm'].predict_proba(X)[:,1]
    return (float(wx)*p1+float(wl)*p2)/max(float(wx)+float(wl), PRIOR_EPS)

def run(args):
    warnings.filterwarnings('ignore')
    out=Path(args.output_root); models=out/'models'; shapdir=out/'shap_explainability'
    out.mkdir(parents=True, exist_ok=True); models.mkdir(exist_ok=True); shapdir.mkdir(exist_ok=True)
    X,y,feature_names,info=load_dataset(args.train_input,args.label_is_last_column,args.label_column)
    if set(pd.unique(y)) != {0,1}: raise ValueError('The training-set labels must contain both 0 and 1.')
    drug_idx,target_idx=get_feature_pools(feature_names)
    if not drug_idx or not target_idx: raise ValueError('Unable to identify mol2vec or ProstT5 feature columns.')
    save_json({'args':vars(args),'data_info':info,'pipeline':'SHAP320_PseudoWkNNIR_PU_ENN_XGBLGBM','drug_dim':len(drug_idx),'target_dim':len(target_idx)}, out/'run_config.json')
    print('='*100); print('Final CV training'); print(f'Train shape={X.shape}, labels={Counter(y)}'); print(f'Top_Protein_K={args.top_protein_k}, threshold={args.threshold}'); print('='*100)
    skf=StratifiedKFold(n_splits=int(args.n_splits), shuffle=True, random_state=int(args.random_state))
    fold_rows=[]; oof=[]; sel_records=[]; samp=[]; priorm=[]; pur=[]; manifest=[]
    for fold,(tr,va) in enumerate(tqdm(skf.split(X,y), total=int(args.n_splits), desc='10-fold CV'), start=1):
        Xtr_raw,Xva_raw=X[tr],X[va]; ytr,yva=y[tr],y[va]
        scaler=StandardScaler(); Xtr=scaler.fit_transform(Xtr_raw); Xva=scaler.transform(Xva_raw)
        selected_idx, selected_names, imp_df, shap_meta=shap_select(Xtr,ytr,feature_names,args.top_protein_k,args.shap_max_samples,int(args.random_state)+fold)
        if args.save_fold_shap_importance:
            d=out/'fold_shap_importance'; d.mkdir(exist_ok=True); imp_df.to_csv(d/f'fold_{fold:02d}_protein_shap_importance.csv',index=False,encoding='utf-8-sig')
        for rank,idx in enumerate(selected_idx,1): sel_records.append({'Fold':fold,'rank':rank,'feature':feature_names[idx],'original_index':int(idx)})
        Xtr_base=Xtr[:,selected_idx]; Xva_base=Xva[:,selected_idx]
        tr_drug,tr_target=Xtr_raw[:,drug_idx],Xtr_raw[:,target_idx]; va_drug,va_target=Xva_raw[:,drug_idx],Xva_raw[:,target_idx]
        tr_prior,prior_cols,_=oof_prior(tr_drug,tr_target,ytr,args.prior_k_drug,args.prior_k_target,args.prior_hash_decimals,args.prior_inner_folds,int(args.random_state)+fold)
        prior_model,va_prior,va_cols,pm=fit_prior_transform(tr_drug,tr_target,ytr,va_drug,va_target,args.prior_k_drug,args.prior_k_target,args.prior_hash_decimals)
        assert prior_cols==va_cols
        pm['Fold']=fold; priorm.append(pm)
        Xtr_aug=np.hstack([Xtr_base,tr_prior]).astype(np.float32); Xva_aug=np.hstack([Xva_base,va_prior]).astype(np.float32)
        prew,pu=pu_weights(ytr,tr_prior,prior_cols,args.use_pu_reliability,args.pu_suspicious_threshold,args.pu_min_neg_weight,args.pu_continuous_downweight)
        pu['Fold']=fold; pur.append(pu)
        Xenn,yenn,wenn,kept,ei=apply_enn(Xtr_aug,ytr,prew,args.enn_n_neighbors,args.enn_kind_sel); ei['Fold']=fold; samp.append(ei)
        model=fit_softvote(Xenn,yenn,wenn,int(args.random_state)+fold)
        prob=pred_softvote(model,Xva_aug,args.xgb_weight,args.lgbm_weight); pred=(prob>=float(args.threshold)).astype(int); met=calc_metrics(yva,pred,prob)
        row={'Fold':fold,'Top_Protein_K':int(args.top_protein_k),'Threshold':float(args.threshold),'N_Base_Selected_Features':len(selected_idx),'N_Prior_Features':len(prior_cols),'N_Total_Features':Xtr_aug.shape[1],**met,'Train_Before_ENN':str(ei['before_enn_counter']),'Train_After_ENN':str(ei['after_enn_counter']),'Suspicious_Negative_N':pu['n_suspicious_negative'],'Weight_Mean_Before_ENN':pu['weight_mean'],'Weight_Mean_After_ENN':ei['weight_after_enn_mean']}
        fold_rows.append(row)
        for si,yt,yp,pp in zip(va,yva,pred,prob): oof.append({'sample_index':int(si),'Fold':fold,'y_true':int(yt),'y_pred':int(yp),'y_prob':float(pp)})
        mpath=models/f'fold_{fold:02d}_model_bundle.pkl'
        bundle=dict(fold=fold,scaler=scaler,selected_idx=selected_idx,selected_names=selected_names,prior_cols=prior_cols,prior_model=prior_model,model=model,feature_names=feature_names,drug_idx=drug_idx,target_idx=target_idx,threshold=float(args.threshold),xgb_weight=float(args.xgb_weight),lgbm_weight=float(args.lgbm_weight),top_protein_k=int(args.top_protein_k),cv_metrics=met,train_indices=tr.astype(int),valid_indices=va.astype(int),shap_meta=shap_meta,prior_meta=pm,pu_info=pu,enn_info=ei,pipeline='SHAP320_PseudoWkNNIR_PU_ENN_XGBLGBM')
        joblib.dump(bundle,mpath,compress=int(args.model_compress)); manifest.append({'Fold':fold,'model_path':str(mpath),**met})
    fdf=pd.DataFrame(fold_rows); fdf.to_csv(out/'cv_fold_metrics.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(oof).sort_values('sample_index').to_csv(out/'cv_oof_predictions.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(sel_records).to_csv(out/'cv_selected_features_by_fold.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(samp).to_csv(out/'cv_sampling_distribution.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(priorm).to_csv(out/'cv_prior_meta.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(pur).to_csv(out/'cv_pu_reliability_summary.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(manifest).to_csv(out/'model_manifest.csv',index=False,encoding='utf-8-sig')
    summary=[]; wide={}
    for m in ['SN','SP','ACC','MCC','AUC','AUPR']:
        vals=fdf[m].values.astype(float); mean=float(np.mean(vals)); std=float(np.std(vals,ddof=1)); summary.append({'Metric':m,'mean':mean,'std':std}); wide[f'{m}_mean']=mean; wide[f'{m}_std']=std
    pd.DataFrame(summary).to_csv(out/'cv_summary_metrics.csv',index=False,encoding='utf-8-sig'); pd.DataFrame([wide]).to_csv(out/'cv_summary_metrics_wide.csv',index=False,encoding='utf-8-sig')
    sc=StandardScaler(); Xfull=sc.fit_transform(X); save_shap_figures(Xfull,y,feature_names,shapdir,args.top_protein_k,args.shap_max_samples,args.random_state,args.shap_max_display)
    print('\nCompleted. Key files:'); print(out/'cv_summary_metrics_wide.csv'); print(out/'cv_fold_metrics.csv'); print(out/'cv_oof_predictions.csv'); print(out/'model_manifest.csv'); print(shapdir/'SHAP_Feature_Summary_Beeswarm.png'); print(shapdir/'SHAP_Feature_Importance_Bar.png')

def build_parser():
    p=argparse.ArgumentParser(description='Final 10-fold CV training: SHAP320 + Pseudo-WkNNIR + PU + ENN + XGB/LGBM')
    p.add_argument('--train-input',type=str,default='./mol2vec_ProstT5_train.csv'); p.add_argument('--output-root',type=str,default='./final_shap320_cv_training')
    p.add_argument('--label-is-last-column',type=str2bool,default=True); p.add_argument('--label-column',type=str,default='label')
    p.add_argument('--top-protein-k',type=int,default=320); p.add_argument('--shap-max-samples',type=int,default=5000); p.add_argument('--shap-max-display',type=int,default=30); p.add_argument('--save-fold-shap-importance',type=str2bool,default=False)
    p.add_argument('--enn-n-neighbors',type=int,default=5); p.add_argument('--enn-kind-sel',type=str,default='mode',choices=['all','mode'])
    p.add_argument('--prior-k-drug',type=int,default=10); p.add_argument('--prior-k-target',type=int,default=10); p.add_argument('--prior-hash-decimals',type=int,default=6); p.add_argument('--prior-inner-folds',type=int,default=3)
    p.add_argument('--use-pu-reliability',type=str2bool,default=True); p.add_argument('--pu-suspicious-threshold',type=float,default=0.75); p.add_argument('--pu-min-neg-weight',type=float,default=0.5); p.add_argument('--pu-continuous-downweight',type=str2bool,default=True)
    p.add_argument('--threshold',type=float,default=0.5); p.add_argument('--n-splits',type=int,default=10); p.add_argument('--random-state',type=int,default=42); p.add_argument('--xgb-weight',type=float,default=0.5); p.add_argument('--lgbm-weight',type=float,default=0.5); p.add_argument('--model-compress',type=int,default=3)
    return p

if __name__=='__main__':
    run(build_parser().parse_args())
