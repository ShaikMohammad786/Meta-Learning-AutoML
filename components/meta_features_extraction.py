## For Classification : n_instances,n_features,n_num_features,n_cat_features,missing_values_pct,class_entropy,n_classes,mean_skewness,mean_kurtosis,avg_correlation,max_correlation,mean_mutual_info,max_mutual_info,pca_fraction_95,feature_to_instance_ratio,best_model

## For Regression : n_instances,n_features,n_num_features,n_cat_features,missing_values_pct,mean_skewness,mean_kurtosis,avg_correlation,max_correlation,pca_fraction_95,var_mean,var_std,mean_feature_entropy,feature_to_instance_ratio,task_type,best_model

# For Clustering : n_instances,n_features,n_num_features,missing_values_pct,mean_skewness,mean_kurtosis,avg_correlation,max_correlation,pca_fraction_95,silhouette_kmeans,davies_bouldin,calinski_harabasz,feature_to_instance_ratio,task_type,best_model

import pandas as pd
import numpy as np
import os
from sklearn.feature_selection import mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import entropy
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from constants import *



def mean_feature_entropy_auto(numeric_df):
    if numeric_df.shape[1] == 0:
        return np.nan
    entropies = []
    for col in numeric_df.columns:
        vals = numeric_df[col].dropna()
        if vals.nunique() > 1:
            if np.issubdtype(vals.dtype, np.integer) and vals.nunique() < 20:
                probs = vals.value_counts(normalize=True)
                entropies.append(entropy(probs))
            else:
                hist, _ = np.histogram(vals, bins=10, density=True)
                hist = hist[hist > 0]
                entropies.append(entropy(hist))
    return np.mean(entropies) if entropies else np.nan


def meta_features_extract_reg(X_train, y_train, best_model = None, raw_df=None):

    # Ensure DataFrame
    if not isinstance(X_train, pd.DataFrame):
        X_train = pd.DataFrame(X_train)

    n_instances = X_train.shape[0]
    n_features = X_train.shape[1]

    numeric_df = X_train.copy()
    n_num_features = n_features
    if raw_df is not None:
        raw_no_target = raw_df.drop(columns=[raw_df.columns[-1]], errors="ignore")
        original_numeric = raw_no_target.select_dtypes(include=['int64', 'float64']).shape[1]
        original_categorical = raw_no_target.shape[1] - original_numeric
        n_cat_features = original_categorical
    else:
        n_cat_features = 0

    # Missing % (if raw dataset available)
    if raw_df is not None:
        missing_values_pct = float(raw_df.isnull().mean().mean() * 100)
    else:
        missing_values_pct = float(X_train.isnull().mean().mean() * 100)

    # Skewness & Kurtosis
    mean_skewness = numeric_df.skew().mean()
    mean_kurtosis = numeric_df.kurtosis().mean()

    # ========= SAFE CORRELATION =========
    clean_df = numeric_df.replace([np.inf, -np.inf], np.nan).fillna(numeric_df.median())
    clean_df = clean_df.loc[:, clean_df.std() > 0]  # remove constant columns

    if clean_df.shape[1] > 1:
        corr_matrix = clean_df.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        corr_vals = upper.stack()

        avg_correlation = float(corr_vals.mean()) if len(corr_vals) > 0 else 0.0
        max_correlation = float(corr_vals.max()) if len(corr_vals) > 0 else 0.0
    else:
        avg_correlation = 0.0
        max_correlation = 0.0

    # ========= CORRELATION WITH TARGET =========
    try:
        corrs_t = clean_df.corrwith(pd.Series(y_train)).abs()
        avg_corr_with_target = float(corrs_t.mean())
        max_corr_with_target = float(corrs_t.max())
    except:
        avg_corr_with_target = 0.0
        max_corr_with_target = 0.0

    # ========= PCA =========
    pca_fraction_95 = np.nan
    if clean_df.shape[1] > 1:
        try:
            scaled = StandardScaler().fit_transform(clean_df)
            pca = PCA().fit(scaled)
            cumvar = np.cumsum(pca.explained_variance_ratio_)
            ncomp = np.argmax(cumvar >= 0.95) + 1 if np.any(cumvar >= 0.95) else clean_df.shape[1]
            pca_fraction_95 = float(ncomp / clean_df.shape[1])
        except:
            pca_fraction_95 = np.nan

    # Variance
    var_mean = float(clean_df.mean().var())
    var_std = float(clean_df.std().var())

    # Entropy
    mean_feature_entropy = mean_feature_entropy_auto(clean_df)

    # Ratio
    feature_to_instance_ratio = float(n_features / max(1, n_instances))

    # Final meta-features
    meta_features = {
        "n_instances": n_instances,
        "n_features": n_features,
        "n_num_features": n_num_features,
        "n_cat_features": n_cat_features,
        "missing_values_pct": missing_values_pct,
        "mean_skewness": mean_skewness,
        "mean_kurtosis": mean_kurtosis,
        "avg_correlation": avg_correlation,
        "max_correlation": max_correlation,
        "mean_corr_with_target": avg_corr_with_target,
        "max_corr_with_target": max_corr_with_target,
        "pca_fraction_95": pca_fraction_95,
        "var_mean": var_mean,
        "var_std": var_std,
        "mean_feature_entropy": mean_feature_entropy,
        "feature_to_instance_ratio": feature_to_instance_ratio,
        "task_type": "regression",
        "best_model": best_model
    }

    os.makedirs(os.path.dirname(META_REGRESSION_DATASET), exist_ok=True)
    save_path = META_REGRESSION_DATASET
    if best_model is not None:
        if os.path.exists(save_path):
            df = pd.read_csv(save_path)
            df = pd.concat([df, pd.DataFrame([meta_features])], ignore_index=True)
        else:
            df = pd.DataFrame([meta_features])

        df.to_csv(save_path, index=False)
        print("✅ Regression meta-features saved.")
    return pd.DataFrame([meta_features])


def meta_features_extract_class(X_train, y_train, best_model = None, raw_df=None, save=True):

    if not isinstance(X_train, pd.DataFrame):
        X_train = pd.DataFrame(X_train)

    n_instances = X_train.shape[0]
    n_features = X_train.shape[1]

    numeric_df = X_train.copy()
    n_num_features = n_features
    if raw_df is not None:
        raw_no_target = raw_df.drop(columns=[raw_df.columns[-1]], errors="ignore")
        original_numeric = raw_no_target.select_dtypes(include=['int64', 'float64']).shape[1]
        original_categorical = raw_no_target.shape[1] - original_numeric
        n_cat_features = original_categorical
    else:
        n_cat_features = 0

    # Missing %
    if raw_df is not None:
        missing_values_pct = float(raw_df.isnull().mean().mean() * 100)
    else:
        missing_values_pct = float(X_train.isnull().mean().mean() * 100)

    mean_skewness = numeric_df.skew().mean()
    mean_kurtosis = numeric_df.kurtosis().mean()

    # Safe correlation
    clean_df = numeric_df.replace([np.inf, -np.inf], np.nan).fillna(numeric_df.median())
    clean_df = clean_df.loc[:, clean_df.std() > 0]

    if clean_df.shape[1] > 1:
        corr_matrix = clean_df.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        corr_vals = upper.stack()

        avg_correlation = float(corr_vals.mean()) if len(corr_vals) > 0 else 0.0
        max_correlation = float(corr_vals.max()) if len(corr_vals) > 0 else 0.0
    else:
        avg_correlation = 0.0
        max_correlation = 0.0

    # Target correlations
    try:
        corrs_t = clean_df.corrwith(pd.Series(y_train)).abs()
        avg_corr_with_target = float(corrs_t.mean())
        max_corr_with_target = float(corrs_t.max())
    except:
        avg_corr_with_target = 0.0
        max_corr_with_target = 0.0

    # Mutual information
    try:
        mi = mutual_info_classif(X_train, y_train)
        mean_mi = float(mi.mean())
        max_mi = float(mi.max())
    except:
        mean_mi = 0.0
        max_mi = 0.0

    # PCA
    pca_fraction_95 = 0.0
    if clean_df.shape[1] > 1:
        try:
            scaled = StandardScaler().fit_transform(clean_df)
            pca = PCA().fit(scaled)
            cumvar = np.cumsum(pca.explained_variance_ratio_)
            ncomp = np.argmax(cumvar >= 0.95) + 1 if np.any(cumvar >= 0.95) else clean_df.shape[1]
            pca_fraction_95 = float(ncomp / clean_df.shape[1])
        except:
            pca_fraction_95 = 0.0

    feature_to_instance_ratio = float(n_features / max(1, n_instances))

    class_entropy = entropy(pd.Series(y_train).value_counts(normalize=True), base=2)
    n_classes = pd.Series(y_train).nunique()

    meta_features = {
        "n_instances": n_instances,
        "n_features": n_features,
        "n_num_features": n_num_features,
        "n_cat_features": n_cat_features,
        "missing_values_pct": missing_values_pct,
        "class_entropy": class_entropy,
        "n_classes": n_classes,
        "mean_skewness": mean_skewness,
        "mean_kurtosis": mean_kurtosis,
        "avg_correlation": avg_correlation,
        "max_correlation": max_correlation,
        "mean_corr_with_target": avg_corr_with_target,
        "max_corr_with_target": max_corr_with_target,
        "mean_mutual_info": mean_mi,
        "max_mutual_info": max_mi,
        "pca_fraction_95": pca_fraction_95,
        "feature_to_instance_ratio": feature_to_instance_ratio,
        "task_type": "classification",
        "best_model": best_model
    }


    os.makedirs(os.path.dirname(META_CLASSIFICATION_DATASET), exist_ok=True)
    save_path = META_CLASSIFICATION_DATASET
    if best_model is not None:
        if os.path.exists(save_path):
            df = pd.read_csv(save_path)
            df = pd.concat([df, pd.DataFrame([meta_features])], ignore_index=True)
        else:
            df = pd.DataFrame([meta_features])

        df.to_csv(save_path, index=False)
        print("✅ Classification meta-features saved.")

    return pd.DataFrame([meta_features])





def meta_features_extract_clust(X_train, best_model="", raw_df=None):

    if not isinstance(X_train, pd.DataFrame):
        X_train = pd.DataFrame(X_train)

    numeric_df = X_train.copy()
    n_instances = numeric_df.shape[0]
    n_features = numeric_df.shape[1]
    n_num_features = n_features
    n_cat_features = 0

    # Missing %
    if raw_df is not None:
        missing_values_pct = float(raw_df.isnull().mean().mean() * 100)
    else:
        missing_values_pct = float(numeric_df.isnull().mean().mean() * 100)

    # Skew / Kurtosis
    mean_skewness = numeric_df.skew().mean()
    mean_kurtosis = numeric_df.kurtosis().mean()

    # Safe correlation
    clean_df = numeric_df.replace([np.inf, -np.inf], np.nan).fillna(numeric_df.median())
    clean_df = clean_df.loc[:, clean_df.std() > 0]

    if clean_df.shape[1] > 1:
        corr_matrix = clean_df.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        corr_vals = upper.stack()

        avg_correlation = float(corr_vals.mean()) if len(corr_vals) > 0 else 0.0
        max_correlation = float(corr_vals.max()) if len(corr_vals) > 0 else 0.0
    else:
        avg_correlation = 0.0
        max_correlation = 0.0

    # PCA
    pca_fraction_95 = 0.0
    if clean_df.shape[1] > 1:
        try:
            scaled = StandardScaler().fit_transform(clean_df)
            pca = PCA().fit(scaled)
            cumvar = np.cumsum(pca.explained_variance_ratio_)
            ncomp = np.argmax(cumvar >= 0.95) + 1 if np.any(cumvar >= 0.95) else clean_df.shape[1]
            pca_fraction_95 = float(ncomp / clean_df.shape[1])
        except:
            pca_fraction_95 = 0.0

    # Clustering metrics
    try:
        X_scaled = StandardScaler().fit_transform(clean_df)
        km = KMeans(n_clusters=3, random_state=0)
        labels = km.fit_predict(X_scaled)

        silhouette_kmeans = silhouette_score(X_scaled, labels)
        davies_bouldin = davies_bouldin_score(X_scaled, labels)
        calinski_harabasz = calinski_harabasz_score(X_scaled, labels)
    except:
        silhouette_kmeans = np.nan
        davies_bouldin = np.nan
        calinski_harabasz = np.nan

    # Final
    feature_to_instance_ratio = float(n_features / max(1, n_instances))

    meta_features = {
        "n_instances": n_instances,
        "n_features": n_features,
        "n_num_features": n_num_features,
        "missing_values_pct": missing_values_pct,
        "mean_skewness": mean_skewness,
        "mean_kurtosis": mean_kurtosis,
        "avg_correlation": avg_correlation,
        "max_correlation": max_correlation,
        "pca_fraction_95": pca_fraction_95,
        "silhouette_kmeans": silhouette_kmeans,
        "davies_bouldin": davies_bouldin,
        "calinski_harabasz": calinski_harabasz,
        "feature_to_instance_ratio": feature_to_instance_ratio,
        "task_type": "clustering",
        "best_model": best_model
    }

    os.makedirs("meta_model/meta_learning/meta_clustering", exist_ok=True)
    save_path = "meta_model/meta_learning/meta_clustering/meta_features_clustering.csv"

    if os.path.exists(save_path):
        df = pd.read_csv(save_path)
        df = pd.concat([df, pd.DataFrame([meta_features])], ignore_index=True)
    else:
        df = pd.DataFrame([meta_features])

    df.to_csv(save_path, index=False)
    print("✅ Clustering meta-features saved.")