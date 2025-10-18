import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.nonparametric.smoothers_lowess import lowess
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from functools import lru_cache

# ---------------------------
#  Config
# ---------------------------
st.set_page_config(layout="wide", page_title="Cancer Risk Viz", initial_sidebar_state="expanded")


# ---------------------------
#  Caching data load / prep
# ---------------------------
@st.cache_data
def load_and_prep(path="../data/raw/cancer-risk-factors.csv"):
    df = pd.read_csv(path)
    # normalize numerical cols for certain viz
    numeric_cols = ["Age", "Smoking", "Alcohol_Use", "BMI", "Fruit_Veg_Intake", "Diet_Red_Meat",
                    "Diet_Salted_Processed", "Physical_Activity_Level", "Air_Pollution", "Occupational_Hazards",
                    "Overall_Risk_Score", "Calcium_Intake"]
    numeric_cols = [c for c in numeric_cols if c in df.columns]
    df = df.copy()
    # create HighRiskFlag once on original df (avoid SettingWithCopy later)
    if "Risk_Level" in df.columns:
        df["HighRiskFlag"] = (df["Risk_Level"] == "High").astype(int)
    # z-score scaled copy for numeric visualization if needed
    df_z = df.copy()
    if numeric_cols:
        scaler = StandardScaler()
        try:
            df_z[[c + "_z" for c in numeric_cols]] = scaler.fit_transform(df[numeric_cols])
        except Exception:
            # fallback: if scaling fails, just copy numeric columns
            for c in numeric_cols:
                df_z[c + "_z"] = df[c]
    return df, df_z


df, df_z = load_and_prep()

# ---------------------------
#  Sidebar filters and params
# ---------------------------
st.sidebar.title("Filters & Parameters")

# Cancer filter
cancer_options = ["All"] + sorted(df["Cancer_Type"].unique().tolist())
sel_cancers = st.sidebar.multiselect("Cancer Type (multi)", options=cancer_options, default=["All"])

# Gender filter (map 0 → Female, 1 → Male)
gender_map = {0: "Female", 1: "Male"}
df["Gender_Label"] = df["Gender"].map(gender_map)

gender_opt = ["All"] + sorted(df["Gender_Label"].dropna().unique().tolist())
sel_gender = st.sidebar.selectbox("Gender", gender_opt, index=0)

# Risk filter
risk_levels = ["All"] + sorted(df["Risk_Level"].unique().tolist())
sel_risk = st.sidebar.selectbox("Risk Level", risk_levels, index=0)

def apply_filters(df_in):
    d = df_in.copy()
    if sel_cancers and "All" not in sel_cancers:
        d = d[d["Cancer_Type"].isin(sel_cancers)]
    if sel_gender != "All":
        d = d[d["Gender_Label"] == sel_gender]
    if sel_risk != "All":
        d = d[d["Risk_Level"] == sel_risk]
    return d


df_f = apply_filters(df)


# ---------------------------
#  Cached statistical helpers
# ---------------------------
@st.cache_data
def compute_corr_and_pvals(data, cols):
    """Compute Pearson correlation matrix and p-value matrix for given columns.
       Uses point-biserial when one variable is binary.
    """
    corr = data[cols].corr()
    pvals = pd.DataFrame(np.nan, index=cols, columns=cols)
    for i in cols:
        xi = data[i].dropna()
        for j in cols:
            if i == j:
                pvals.loc[i, j] = 0.0
                continue
            xj = data[j].dropna()
            # align indices
            common_idx = xi.index.intersection(xj.index)
            xi_al = xi.loc[common_idx]
            xj_al = xj.loc[common_idx]
            try:
                # if one is binary (0/1), use pointbiserial
                if set(xi_al.unique()) <= {0, 1} and len(xi_al.unique()) > 1:
                    r, p = stats.pointbiserialr(xi_al, xj_al)
                elif set(xj_al.unique()) <= {0, 1} and len(xj_al.unique()) > 1:
                    r, p = stats.pointbiserialr(xj_al, xi_al)
                else:
                    r, p = stats.pearsonr(xi_al, xj_al)
                pvals.loc[i, j] = p
            except Exception:
                pvals.loc[i, j] = np.nan
    return corr, pvals


@st.cache_data
def compute_loess_with_ci(x, y, frac=0.3, n_boot=200, seed=0):
    """Compute LOWESS line + approximate CI via bootstrap of residuals (lightweight)."""
    np.random.seed(seed)
    xy = pd.DataFrame({"x": x, "y": y}).dropna().sort_values("x")
    xs = xy["x"].values
    ys = xy["y"].values
    # base loess
    base = lowess(ys, xs, frac=frac, return_sorted=True)
    fitted = base[:, 1]
    # bootstrap simple: resample pairs and recompute lowess on sample (lightweight approximation)
    boot_lines = []
    n = len(xs)
    if n < 5:
        return base[:, 0], base[:, 1], None
    for _ in range(min(n_boot, 500)):
        idx = np.random.choice(np.arange(n), size=n, replace=True)
        try:
            b = lowess(ys[idx], xs[idx], frac=frac, return_sorted=True)
            # interpolate b onto original xs
            boot_lines.append(np.interp(xs, b[:, 0], b[:, 1]))
        except Exception:
            continue
    if len(boot_lines) >= 10:
        boot_arr = np.vstack(boot_lines)
        lower = np.percentile(boot_arr, 2.5, axis=0)
        upper = np.percentile(boot_arr, 97.5, axis=0)
    else:
        lower, upper = None, None
    return xs, fitted, (lower, upper)


@st.cache_data
def compute_logistic_or(df_in, exposure_col, outcome_col="HighRiskFlag"):
    df_ = df_in[[exposure_col, outcome_col]].dropna()
    try:
        X = sm.add_constant(df_[exposure_col])
        model = sm.Logit(df_[outcome_col], X).fit(disp=0, maxiter=100)
        params = model.params
        conf = model.conf_int()
        or_ = np.exp(params[exposure_col])
        ci_low, ci_high = np.exp(conf.loc[exposure_col])
        pval = model.pvalues[exposure_col]
        return {"or": or_, "ci": (ci_low, ci_high), "p": pval, "n": df_.shape[0]}
    except Exception:
        return None


# ---------------------------
#  Utility tests
# ---------------------------
def ttest_annotation(group_a, group_b):
    try:
        t, p = stats.ttest_ind(group_a, group_b, equal_var=False, nan_policy='omit')
        return t, p
    except Exception:
        return np.nan, np.nan


# ---------------------------
#  Overview Section
# ---------------------------
st.title("Cancer Risk — Exploratory & Statistical Visualization")
with st.expander("Population Overview", expanded=True):
    # KPI row
    k1, k2, k3, k4 = st.columns([1, 1, 1, 1])
    mean_age = df_f["Age"].mean() if "Age" in df_f.columns else np.nan
    Total_population = df_f["Patient_ID"].count() if "Patient_ID" in df_f.columns else np.nan
    Male = (df_f["Gender"] == 1).sum() if "Gender" in df_f.columns else np.nan
    Female = (df_f["Gender"] == 0).sum() if "Gender" in df_f.columns else np.nan
    k1.metric("Mean Age:", f"{mean_age:.1f}" if not np.isnan(mean_age) else "N/A")
    k2.metric("Total population:", f"{Total_population}" if not np.isnan(Total_population) else "N/A")
    k3.metric("Male:", f"{Male}" if not np.isnan(Male) else "N/A")
    k4.metric("Female:", f"{Female}" if not np.isnan(Female) else "N/A")

    # Bar chart cancer types (Plotly) with hovertemplate
    st.markdown("**Distribution of Cancer Types**")
    c_counts = df_f["Cancer_Type"].value_counts().reset_index()
    c_counts.columns = ["Cancer_Type", "Count"]
    fig_bar = px.bar(c_counts, x="Cancer_Type", y="Count", text="Count", template="simple_white")
    fig_bar.update_traces(marker_line_width=0.5, hovertemplate="<b>%{x}</b><br>Count: %{y}<extra></extra>")
    fig_bar.update_layout(margin=dict(t=10, b=10), xaxis_title="", yaxis_title="Count")
    st.plotly_chart(fig_bar, use_container_width=True)

    # Stacked bar: Risk_Level by Cancer_Type
    st.markdown("**Risk Level by Cancer Type**")
    fig_stack = px.histogram(df_f, x="Cancer_Type", color="Risk_Level", barmode="stack",
                             category_orders={"Risk_Level": ["Low", "Medium", "High"]}, template="simple_white",
                             color_discrete_map={
                                 "Low": "#f4faff",
                                 "Medium": "#c8eaff",
                                 "High": "#2c4166"
                             })
    fig_stack.update_traces(hovertemplate="Cancer: %{x}<br>Risk Level: %{legendgroup}<br>Count: %{y}<extra></extra>")
    fig_stack.update_layout(margin=dict(t=10, b=10), yaxis_title="Count")
    st.plotly_chart(fig_stack, use_container_width=True)

    # Histogram Risk_Score
    if "Overall_Risk_Score" in df_f.columns:
        st.markdown("**Overall Risk Score distribution**")
        fig_hist = px.histogram(df_f, x="Overall_Risk_Score", nbins=30, marginal="box", template="simple_white")
        fig_hist.update_traces(hovertemplate="Risk Score: %{x}<br>Count: %{y}<extra></extra>")
        st.plotly_chart(fig_hist, use_container_width=True)

# ---------------------------
#  Risk Factors Deep Dive
# ---------------------------
with st.expander("Risk Factors — Correlations & Interactions", expanded=False):
    st.markdown("### Correlation heatmap")
    # list of target behaviorally relevant columns
    cols_of_interest = [c for c in ["Smoking", "Alcohol_Use", "BMI", "Obesity",
    "Diet_Red_Meat", "Diet_Salted_Processed", "Fruit_Veg_Intake",
    "Physical_Activity", "Physical_Activity_Level",
    "Air_Pollution", "Occupational_Hazards",
    "Family_History", "BRCA_Mutation", "H_Pylori_Infection",
    "Overall_Risk_Score"] if c in df_f.columns]
    if len(cols_of_interest) >= 2:
        corr_matrix, pvals = compute_corr_and_pvals(df_f, cols_of_interest)
        # annot with stars
        annot = corr_matrix.round(2).astype(str)
        for r in corr_matrix.index:
            for c in corr_matrix.columns:
                p = pvals.loc[r, c]
                if not np.isnan(p) and p < 0.001:
                    annot.loc[r, c] += " ***"
                elif not np.isnan(p) and p < 0.01:
                    annot.loc[r, c] += " **"
                elif not np.isnan(p) and p < 0.05:
                    annot.loc[r, c] += " *"
        fig_heat = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            text=annot.values,
            hoverinfo="text",
            colorscale="RdBu",
            zmid=0
        ))
        fig_heat.update_layout(height=450, template="simple_white", margin=dict(t=10, b=10))
        st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.markdown("Not enough numeric columns to compute correlation matrix.")

    # Top 5 correlated with Overall_Risk_Score
    st.markdown("### Top features correlated with Overall Risk Score")
    if "Overall_Risk_Score" in df_f.columns and len(cols_of_interest) >= 2:
        corrs_with_target = corr_matrix["Overall_Risk_Score"].abs().sort_values(ascending=False)
        top5 = corrs_with_target.index[1:6]  # skip itself
        top5_vals = corr_matrix.loc[top5, "Overall_Risk_Score"]
        df_top5 = pd.DataFrame({"feature": top5, "r": top5_vals.values})
        fig_top5 = px.bar(df_top5, x="feature", y="r", text=df_top5["r"].round(2), template="simple_white")
        fig_top5.update_traces(hovertemplate="Feature: %{x}<br>r: %{y}<extra></extra>")
        st.plotly_chart(fig_top5, use_container_width=True)

    # Showing top 3 risk factors radar chart per cancer
    st.markdown(f"**Risk Factor Patterns for each cancer (Top 3 Correlations)**")
    file_path = "../data/processed/xgb_top3_risks.xlsx"
    df_summary = pd.read_excel(file_path, sheet_name="top3_summary")
    df_importances = pd.read_excel(file_path, sheet_name="all_importances")
    df_top = pd.DataFrame(columns=["Cancer_Type", "Feature", "Correlation"])

    for _, row in df_summary.iterrows():
        cancer = row["Cancer_Type"]
        top_features = [row["top1"], row["top2"], row["top3"]]
        sub_df = df_importances[
            (df_importances["Cancer_Type"] == cancer) & (df_importances["feature"].isin(top_features))]

        temp_df = pd.DataFrame({
            "Cancer_Type": [cancer] * len(top_features),
            "Feature": sub_df["feature"],
            "Correlation": sub_df["spearman"]
        })
        df_top = pd.concat([df_top, temp_df], ignore_index=True)

    max_corr = df_top["Correlation"].abs().max()
    df_top["Normalized_Correlation"] = df_top["Correlation"] / max_corr

    cancer_types = df_top["Cancer_Type"].unique().tolist()
    selected_cancer = st.selectbox("Select cancer type:", cancer_types)
    fig = go.Figure()

    sub = df_top[df_top["Cancer_Type"] == selected_cancer]
    fig.add_trace(go.Scatterpolar(
        r=sub["Normalized_Correlation"],
        theta=sub["Feature"],
        fill='toself',
        name=selected_cancer,
        hovertemplate="Factor: %{theta}<br>Correlation: %{r:.2f}<extra></extra>",
        line=dict(color=px.colors.qualitative.Plotly[cancer_types.index(selected_cancer)])
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[-1, 1],
                tickvals=[-1, -0.5, 0, 0.5, 1],
                tickmode='array'
            ),
        ),
        title=f"Top Risk Factors for {selected_cancer}",
        template="plotly_dark",
        height=700,
        showlegend=True
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Key Insights")
    sub = df_top[df_top["Cancer_Type"] == selected_cancer]
    top_factor = sub.iloc[0]["Feature"]
    top_corr = sub.iloc[0]["Correlation"]
    st.write(f"{selected_cancer}: Strongest risk factor is **{top_factor}** with correlation {top_corr:.2f}.")

    # BRCA / Family History boxplot — show only if relevant (avoid misleading when not Breast)
    st.markdown("### Genetic: Family History")
    show_brca = ("BRCA_Mutation" in df_f.columns) and (("All" in sel_cancers) or ("Breast" in sel_cancers))
    if "Family_History" in df_f.columns:
        fig_fh = px.box(df_f, x="Family_History", y="Overall_Risk_Score", points="outliers", template="simple_white")
        fig_fh.update_traces(hovertemplate="Family History: %{x}<br>Risk: %{y}<extra></extra>")
        st.plotly_chart(fig_fh, use_container_width=True)

# ---------------------------
#  Exploratory Insights
# ---------------------------
with st.expander("Exploratory Insights (Subgroup analyses)", expanded=False):
    st.markdown("### Gender differences in Risk Score")
    if "Gender" in df_f.columns and "Overall_Risk_Score" in df_f.columns:
        df_f["Gender"] = df_f["Gender"].map({0: "Female", 1: "Male"})
        fig_gender = px.box(df_f, x="Gender", y="Overall_Risk_Score", points="outliers", template="simple_white")
        fig_gender.update_traces(hovertemplate="Gender: %{x}<br>Risk: %{y}<extra></extra>")
        st.plotly_chart(fig_gender, use_container_width=True)


    st.markdown("### BMI vs Overall Risk Score")
    if "BMI" in df_f.columns and "Overall_Risk_Score" in df_f.columns:
        xy = df_f[["BMI", "Overall_Risk_Score"]].dropna().sort_values("BMI")
        xs, fitted, ci = compute_loess_with_ci(xy["BMI"], xy["Overall_Risk_Score"], frac=0.3, n_boot=200)
        fig_loess = go.Figure()
        fig_loess.add_trace(
            go.Scatter(x=xy["BMI"], y=xy["Overall_Risk_Score"], mode="markers", marker=dict(opacity=0.5),
                       name="points"))
        fig_loess.add_trace(go.Scatter(x=xs, y=fitted, mode="lines", line=dict(color="black"), name="LOESS"))
        if ci and ci[0] is not None:
            lower, upper = ci
            fig_loess.add_traces([
                go.Scatter(x=xs, y=upper, mode="lines", line=dict(width=1, dash="dash"), showlegend=False,
                           marker_color='lightgrey'),
                go.Scatter(x=xs, y=lower, mode="lines", line=dict(width=1, dash="dash"), showlegend=False,
                           marker_color='lightgrey')
            ])
        fig_loess.update_layout(template="simple_white", xaxis_title="BMI", yaxis_title="Overall Risk Score")
        fig_loess.update_traces(hovertemplate="BMI: %{x}<br>Risk: %{y}<extra></extra>")
        st.plotly_chart(fig_loess, use_container_width=True)

    st.markdown("### Smoking vs Risk by Cancer Type")
    if "Smoking" in df_f.columns and "Overall_Risk_Score" in df_f.columns:
        fig_facet = px.scatter(df_f, x="Smoking", y="Overall_Risk_Score", color="Risk_Level", facet_col="Cancer_Type",
                               template="simple_white", height=400)
        fig_facet.update_traces(hovertemplate="Smoking: %{x}<br>Risk: %{y}<extra></extra>")
        st.plotly_chart(fig_facet, use_container_width=True)

    st.markdown("### Physical Activity distribution by Risk Level")
    if "Physical_Activity_Level" in df_f.columns:
        fig_violin = px.violin(df_f, x="Risk_Level", y="Physical_Activity_Level", box=True, points="all",
                               template="simple_white")
        fig_violin.update_traces(hovertemplate="Risk Level: %{x}<br>Activity: %{y}<extra></extra>")
        st.plotly_chart(fig_violin, use_container_width=True)
