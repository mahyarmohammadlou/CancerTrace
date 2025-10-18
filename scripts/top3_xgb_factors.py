"""
Top-3 risk factors per cancer type (no-arg version).
Save results to Excel and print per-cancer performance (AUC for classification, R2 for regression).
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score, r2_score
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings("ignore")


INPUT_CSV = os.path.abspath("../data/raw/cancer-risk-factors.csv")
OUTPUT_XLSX = os.path.abspath("../data/processed/XLSX")


# candidate features you listed
CANDIDATE_FEATURES = [
    'Smoking','Alcohol_Use','Obesity','Family_History','Diet_Red_Meat',
    'Diet_Salted_Processed','Fruit_Veg_Intake','Physical_Activity',
    'Air_Pollution','Occupational_Hazards','BRCA_Mutation',
    'H_Pylori_Infection','Calcium_Intake','Overall_Risk_Score','BMI','Physical_Activity_Level'
]

# Try to import xgboost; fallback to sklearn histogram GB
USE_XGB = True
try:
    import xgboost as xgb
except Exception:
    USE_XGB = False
    from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier

def detect_target_and_cancer_column(df):
    # common target names
    for t in ['Overall_Risk_Score','Risk_Score','RiskScore','Score','Label','Has_Cancer','Cancer_Present']:
        if t in df.columns:
            return t
    # fallback: choose a numeric column with "score" substring if present
    for c in df.select_dtypes(include=[np.number]).columns:
        if 'score' in c.lower():
            return c
    return None

def detect_cancer_type_column(df):
    for cand in ['Cancer_Type','CancerType','Type','Cancer','Cancer_Name']:
        if cand in df.columns:
            return cand
    obj_cols = df.select_dtypes(include=['object']).columns
    for c in obj_cols:
        if 1 < df[c].nunique() <= 50:
            return c
    return None

def build_and_rank(X, y, use_xgb=USE_XGB):
    # X: numeric DataFrame, y: Series
    # returns combined importance DataFrame, model, perf dict
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # determine task
    is_classification = False
    if pd.api.types.is_numeric_dtype(y):
        uniq = pd.Series(y).dropna().unique()
        if set(uniq).issubset({0,1}):
            is_classification = True

    # create model
    if use_xgb:
        if is_classification:
            model = xgb.XGBClassifier(n_estimators=150, random_state=42, use_label_encoder=False, eval_metric='logloss')
        else:
            model = xgb.XGBRegressor(n_estimators=150, random_state=42)
    else:
        if is_classification:
            model = HistGradientBoostingClassifier(random_state=42, max_iter=100)
        else:
            model = HistGradientBoostingRegressor(random_state=42, max_iter=100)

    model.fit(X_train, y_train)

    perf = {}
    try:
        if is_classification:
            probs = model.predict_proba(X_test)[:,1]
            perf['auc'] = float(roc_auc_score(y_test, probs))
        else:
            preds = model.predict(X_test)
            perf['r2'] = float(r2_score(y_test, preds))
    except Exception:
        pass

    # permutation importance (n_repeats small for speed)
    try:
        r = permutation_importance(model, X_test_s, y_test, n_repeats=10, random_state=42, n_jobs=1)
        perm_df = pd.DataFrame({'feature': X.columns, 'perm_mean': r.importances_mean, 'perm_std': r.importances_std})
    except Exception:
        try:
            imps = model.feature_importances_
            perm_df = pd.DataFrame({'feature': X.columns, 'perm_mean': imps, 'perm_std': 0.0})
        except Exception:
            perm_df = pd.DataFrame({'feature': X.columns, 'perm_mean': 0.0, 'perm_std': 0.0})

    perm_df = perm_df.sort_values('perm_mean', ascending=False).reset_index(drop=True)

    # Spearman
    spearman_rows = []
    for feat in X.columns:
        try:
            coef, p = spearmanr(X[feat], y, nan_policy='omit')
        except Exception:
            coef, p = np.nan, np.nan
        spearman_rows.append({'feature': feat, 'spearman': coef, 'pval': p})
    spearman_df = pd.DataFrame(spearman_rows).sort_values('spearman', key=lambda s: s.abs(), ascending=False).reset_index(drop=True)

    # combine ranks
    perm_df['perm_rank'] = perm_df['perm_mean'].rank(ascending=False, method='min')
    spearman_df['sp_rank'] = spearman_df['spearman'].abs().rank(ascending=False, method='min')
    combined = perm_df.merge(spearman_df, on='feature', how='outer')
    combined['perm_rank'] = combined['perm_rank'].fillna(combined['perm_rank'].max()+1)
    combined['sp_rank'] = combined['sp_rank'].fillna(combined['sp_rank'].max()+1)
    combined['combined_rank'] = combined['perm_rank'] + combined['sp_rank']
    combined = combined.sort_values('combined_rank').reset_index(drop=True)

    return combined, model, perf, is_classification, (X_train, X_test, y_train, y_test)

def main(input_csv, output_xlsx):
    input_csv = os.path.abspath("cancer-risk-factors.csv")
    output_xlsx = os.path.abspath(output_xlsx)
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")
    df = pd.read_csv(input_csv)
    print("Loaded:", input_csv, "| shape:", df.shape)

    features_present = [c for c in CANDIDATE_FEATURES if c in df.columns]
    print("Candidate features present:", features_present)

    target = detect_target_and_cancer_column(df)
    cancer_col = detect_cancer_type_column(df)
    print("Detected target:", target, "| cancer column:", cancer_col)

    if target is None or cancer_col is None:
        raise RuntimeError("Couldn't detect target or cancer-type column. Please ensure CSV has 'Overall_Risk_Score' and 'Cancer_Type' or similar.")

    results = []
    all_imps = []

    for cancer_type, group in df.groupby(cancer_col):
        sub = group.dropna(subset=[target]).copy()
        n = len(sub)
        if n < 15:
            print(f"Skipping {cancer_type} (n={n}) — too few samples.")
            continue

        X = sub[features_present].drop(columns=[target], errors='ignore')
        X = X.select_dtypes(include=[np.number])
        if X.shape[1] == 0:
            print(f"No numeric candidate features for {cancer_type}, skipping.")
            continue
        y = sub[target]

        combined, model, perf, is_classification, splits = build_and_rank(X, y)
        top3 = combined['feature'].iloc[:3].tolist()

        # Additional: if regression, compute binary-AUC by thresholding at median (informative)
        extra_auc = None
        if not is_classification:
            try:
                median = np.nanmedian(y)
                y_bin = (y > median).astype(int)
                X_train, X_test, y_train, y_test = splits
                try:
                    proba = model.predict_proba(X_test)[:,1]
                except Exception:
                    # for regressors, use predictions and treat as score
                    proba = model.predict(X_test)
                extra_auc = float(roc_auc_score(y_bin.loc[y_test.index] if hasattr(y_test, 'index') else y_bin.iloc[y_test.index], proba))
            except Exception:
                extra_auc = None

        # Print results for this cancer type
        print("\n" + "="*60)
        print(f"Cancer type: {cancer_type} | samples: {n}")
        if is_classification:
            print(f"Task: Classification | AUC = {perf.get('auc', 'NA'):.4f}" if 'auc' in perf else "AUC: NA")
        else:
            print(f"Task: Regression | R2 = {perf.get('r2', 'NA'):.4f}" if 'r2' in perf else "R2: NA")
            if extra_auc is not None:
                print(f"(Aux) AUC after median-binarization = {extra_auc:.4f}")
        print("Top-3 risk factors (combined rank):", top3)
        print("Top importance table (first 8 rows):")
        print(combined.head(8).to_string(index=False))
        results.append({
            'Cancer_Type': cancer_type,
            'n_samples': n,
            **perf,
            'aux_auc_median_binarized': extra_auc,
            'top1': top3[0] if len(top3)>0 else None,
            'top2': top3[1] if len(top3)>1 else None,
            'top3': top3[2] if len(top3)>2 else None
        })

        tmp = combined.copy()
        tmp['Cancer_Type'] = cancer_type
        all_imps.append(tmp)

    res_df = pd.DataFrame(results).sort_values('Cancer_Type') if results else pd.DataFrame()
    imp_df = pd.concat(all_imps, ignore_index=True) if all_imps else pd.DataFrame()

    # Save to Excel
    with pd.ExcelWriter(output_xlsx) as writer:
        res_df.to_excel(writer, sheet_name='top3_summary', index=False)
        imp_df.to_excel(writer, sheet_name='all_importances', index=False)
        df.to_excel(writer, sheet_name='raw_input', index=False)

if __name__ == "__main__":
    main(INPUT_CSV, OUTPUT_XLSX)
