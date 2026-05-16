import os
import pandas as pd
from sklearn.cluster import (
    KMeans, DBSCAN, OPTICS, MeanShift, SpectralClustering,
    AgglomerativeClustering, Birch
)
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


class Analyzer:

    # ======================================================
    # ===================== LOAD DATA =======================
    # ======================================================

    @staticmethod
    def load_data(path, separator=",", encoding_list=("utf-8", "latin1", "utf-16")):
        if not os.path.exists(path):
            raise FileNotFoundError(f"File tidak ditemukan: {path}")

        for enc in encoding_list:
            try:
                df = pd.read_csv(path, sep=separator, encoding=enc)
                return df
            except UnicodeDecodeError:
                continue

        raise UnicodeDecodeError("Gagal membaca file: semua encoding gagal.")

    # ======================================================
    # ================== CLEAN COLUMN NAMES =================
    # ======================================================

    @staticmethod
    def clean_columns(df):
        df = df.copy()

        cleaned = (
            df.columns
            .str.strip()
            .str.lower()
            .str.replace(r"\s+", "_", regex=True)
            .str.replace(r"[^a-z0-9_]", "", regex=True)
        )

        df.columns = cleaned
        return df

    # ======================================================
    # ============== REMOVE INCOMPLETE ROWS =================
    # ======================================================

    @staticmethod
    def remove_incomplete_rows(df):
        df = df.copy()

        df = df.replace(r"^\s*$", pd.NA, regex=True)
        df = df.dropna(axis=0, how='any')

        return df

    # ======================================================
    # =================== RANDOM SAMPLE =====================
    # ======================================================

    @staticmethod
    def random_sample(df, percentage=10, random_state=42):
        if percentage <= 0 or percentage > 100:
            raise ValueError("percentage harus 1-100")

        sample_size = int(len(df) * (percentage / 100))
        return df.sample(n=sample_size, replace=False, random_state=random_state)

    # ======================================================
    # ================= DETECT COLUMN TYPES =================
    # ======================================================

    @staticmethod
    def detect_column_types(df, text_threshold=30):
        numeric_cols = []
        categorical_cols = []
        boolean_cols = []
        text_cols = []

        for col in df.columns:
            series = df[col]

            if pd.api.types.is_numeric_dtype(series):
                numeric_cols.append(col)
                continue

            if pd.api.types.is_bool_dtype(series):
                boolean_cols.append(col)
                continue

            if pd.api.types.is_object_dtype(series):
                avg_len = series.astype(str).str.len().mean()
                if avg_len <= text_threshold:
                    categorical_cols.append(col)
                else:
                    text_cols.append(col)
                continue

        return {
            "numeric": numeric_cols,
            "categorical": categorical_cols,
            "boolean": boolean_cols,
            "text": text_cols
        }

    # ======================================================
    # =================== PREPARE FEATURES ==================
    # ======================================================

    @staticmethod
    def prepare_features(df, numeric_cols, categorical_cols):
        df_numeric = df[numeric_cols].copy()

        df_categorical = pd.get_dummies(
            df[categorical_cols],
            drop_first=False
        ) if categorical_cols else pd.DataFrame()

        df_final = pd.concat([df_numeric, df_categorical], axis=1)
        return df_final

    # ======================================================
    # =================== SCALE FEATURES ====================
    # ======================================================

    @staticmethod
    def scale_features(df):
        scaler = StandardScaler()
        arr = scaler.fit_transform(df)
        df_scaled = pd.DataFrame(arr, columns=df.columns, index=df.index)
        return df_scaled

    # ======================================================
    # ==================== RUN CLUSTERING ===================
    # ======================================================

    @staticmethod
    def run_kmeans(X_scaled, n_clusters=3, n_init=10, random_state=42):
        model = KMeans(n_clusters=n_clusters, n_init=n_init, random_state=random_state)
        return model.fit_predict(X_scaled)

    @staticmethod
    def run_dbscan(X_scaled, eps=0.5, min_samples=5):
        return DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X_scaled)

    @staticmethod
    def run_optics(X_scaled, min_samples=5, xi=0.05, min_cluster_size=0.05):
        return OPTICS(min_samples=min_samples, xi=xi, min_cluster_size=min_cluster_size).fit_predict(X_scaled)

    @staticmethod
    def run_meanshift(X_scaled, bandwidth=None):
        return MeanShift(bandwidth=bandwidth).fit_predict(X_scaled)

    @staticmethod
    def run_spectral(X_scaled, n_clusters=3, gamma=1.0, assign_labels="kmeans"):
        return SpectralClustering(
            n_clusters=n_clusters,
            gamma=gamma,
            assign_labels=assign_labels,
            random_state=42
        ).fit_predict(X_scaled)

    @staticmethod
    def run_agglomerative(X_scaled, n_clusters=3, linkage="ward"):
        return AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage).fit_predict(X_scaled)

    @staticmethod
    def run_birch(X_scaled, n_clusters=3, threshold=0.5, branching_factor=50):
        return Birch(n_clusters=n_clusters, threshold=threshold, branching_factor=branching_factor).fit_predict(X_scaled)

    @staticmethod
    def run_gmm(X_scaled, n_clusters=3, covariance_type="full"):
        model = GaussianMixture(n_components=n_clusters, covariance_type=covariance_type, random_state=42)
        model.fit(X_scaled)
        return model.predict(X_scaled)

    # ======================================================
    # ================== PROFILE CLUSTERS ===================
    # ======================================================

    @staticmethod
    def profile_clusters(X_scaled, cluster_labels, feature_names, top_n=5):
        df = X_scaled.copy()
        df["cluster"] = cluster_labels

        cluster_means = df.groupby("cluster")[feature_names].mean()

        profiles = {}

        for cid in cluster_means.index:
            row = cluster_means.loc[cid]
            top_f = row.sort_values(ascending=False).head(top_n)
            bot_f = row.sort_values(ascending=True).head(top_n)

            profiles[cid] = {
                "top": top_f.to_dict(),
                "bottom": bot_f.to_dict()
            }

        # hitung silhouette score
        if len(set(cluster_labels)) > 1:
            sil = silhouette_score(X_scaled, cluster_labels)
        else:
            sil = None

        return profiles, sil

    # ======================================================
    # ================== VISUALIZE CLUSTERS =================
    # ======================================================

    @staticmethod
    def visualize_clusters(X_scaled, cluster_labels, save_path, title="Cluster Visualization (PCA 2D)"):

        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)

        plt.figure(figsize=(10, 7))
        SC = plt.scatter(
            X_pca[:, 0],
            X_pca[:, 1],
            c=cluster_labels,
            cmap="viridis",
            alpha=0.85,
            s=18
        )

        plt.title(title)
        plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.2f}%)")
        plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.2f}%)")
        plt.grid(True, alpha=0.3)
        plt.legend(*SC.legend_elements(), title="Cluster")

        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()

        return save_path