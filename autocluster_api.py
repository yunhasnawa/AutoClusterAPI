# autocluster_api.py

import os
from typing import List, Optional, Dict, Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from analyzer import Analyzer  # pastikan analyzer.py berisi versi stateless


# ===================== Pydantic Models =====================

class PrepareFeaturesRequest(BaseModel):
    numeric: List[str]
    categorical: List[str] = []


class RunClusteringRequest(BaseModel):
    algorithm: str  # kmeans, dbscan, optics, meanshift, spectral, agglomerative, birch, gmm
    # parameter opsional, dipakai sesuai algoritma
    n_clusters: Optional[int] = None
    eps: Optional[float] = None
    min_samples: Optional[int] = None
    xi: Optional[float] = None
    min_cluster_size: Optional[float] = None
    bandwidth: Optional[float] = None
    gamma: Optional[float] = None
    assign_labels: Optional[str] = None
    linkage: Optional[str] = None
    threshold: Optional[float] = None
    branching_factor: Optional[int] = None
    covariance_type: Optional[str] = None


# ===================== AutoClusterAPI ======================

class AutoClusterAPI:
    """
    Wrapper untuk membuat FastAPI server berbasis pipeline clustering
    menggunakan class Analyzer (stateless).
    """

    def __init__(self, data: str, pipeline_root: str = "pipeline"):
        """
        Parameters
        ----------
        data : str
            Path ke CSV utama yang akan dipakai di endpoint /load-data
        pipeline_root : str
            Folder root untuk menyimpan semua file pipeline per session_id
        """
        self.data_path = data
        self.pipeline_root = pipeline_root

        os.makedirs(self.pipeline_root, exist_ok=True)

        self.app = FastAPI(
            title="AutoClusterAPI",
            description="Backend API for auomated clustering pipeline",
            version="1.0.0"
        )

        # Mount static untuk akses file (termasuk gambar visualisasi)
        self.app.mount(
            "/pipeline",
            StaticFiles(directory=self.pipeline_root),
            name="pipeline"
        )

        self._setup_routes()

    # ------------------------------------------------------
    # Helper: Session & File Paths
    # ------------------------------------------------------

    def _get_session_dir(self, session_id: str) -> str:
        path = os.path.join(self.pipeline_root, session_id)
        os.makedirs(path, exist_ok=True)
        return path

    def _file_path(self, session_id: str, filename: str) -> str:
        return os.path.join(self._get_session_dir(session_id), filename)

    def _ensure_file_exists(self, path: str, desc: str):
        if not os.path.exists(path):
            raise HTTPException(
                status_code=400,
                detail=f"{desc} tidak ditemukan: {path}"
            )

    # ------------------------------------------------------
    # Public: start server
    # ------------------------------------------------------

    def serve(self, host: str = "0.0.0.0", port: int = 8000):
        import uvicorn
        uvicorn.run(self.app, host=host, port=port)

    # ------------------------------------------------------
    # Route definitions
    # ------------------------------------------------------

    def _setup_routes(self):

        # ------------- Endpoint-1: /load-data -------------
        @self.app.get("/load-data")
        def load_data(
            session_id: str = Query(..., description="ID sesi (multi-user safe)"),
            separator: str = Query(",", description="Separator CSV, misal ',' atau ';'"),
            limit: Optional[int] = Query(20, description="Jumlah baris preview yang ditampilkan")
        ):
            """
            Membaca CSV utama (self.data_path) -> step-1-loaded-data.csv
            """
            try:
                df = Analyzer.load_data(self.data_path, separator=separator)
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))

            out_path = self._file_path(session_id, "step-1-loaded-data.csv")
            df.to_csv(out_path, index=False)

            preview = df.head(limit) if limit is not None else df

            return {
                "session_id": session_id,
                "file": out_path,
                "rows": len(df),
                "cols": len(df.columns),
                "columns": list(df.columns),
                "preview": preview.to_dict(orient="records")
            }

        # ------------- Endpoint-2: /clean-columns ----------
        @self.app.get("/clean-columns")
        def clean_columns(
            session_id: str = Query(..., description="ID sesi")
        ):
            """
            Membaca step-1-loaded-data.csv, membersihkan nama kolom,
            menyimpan ke step-2-cleaned-columns.csv
            """
            in_path = self._file_path(session_id, "step-1-loaded-data.csv")
            self._ensure_file_exists(in_path, "Data step-1")

            df = Analyzer.load_data(in_path, separator=",")
            df_clean = Analyzer.clean_columns(df)

            out_path = self._file_path(session_id, "step-2-cleaned-columns.csv")
            df_clean.to_csv(out_path, index=False)

            return {
                "session_id": session_id,
                "file": out_path,
                "rows": len(df_clean),
                "cols": len(df_clean.columns),
                "columns": list(df_clean.columns),
                "preview": df_clean.head(20).to_dict(orient="records")
            }

        # ------------- Endpoint-3: /remove-incomplete-rows -
        @self.app.get("/remove-incomplete-rows")
        def remove_incomplete_rows(
            session_id: str = Query(..., description="ID sesi")
        ):
            """
            Membaca step-2-cleaned-columns.csv, menghapus baris dengan nilai kosong,
            menyimpan ke step-3-removed-incomplete-rows.csv
            """
            in_path = self._file_path(session_id, "step-2-cleaned-columns.csv")
            self._ensure_file_exists(in_path, "Data step-2")

            df = Analyzer.load_data(in_path, separator=",")
            df_clean = Analyzer.remove_incomplete_rows(df)

            out_path = self._file_path(session_id, "step-3-removed-incomplete-rows.csv")
            df_clean.to_csv(out_path, index=False)

            return {
                "session_id": session_id,
                "file": out_path,
                "rows": len(df_clean),
                "cols": len(df_clean.columns),
                "columns": list(df_clean.columns),
                "preview": df_clean.head(20).to_dict(orient="records")
            }

        # ------------- Endpoint-4: /set-data-to-cluster ----
        @self.app.get("/set-data-to-cluster")
        def set_data_to_cluster(
            session_id: str = Query(..., description="ID sesi"),
            step: int = Query(..., ge=1, le=3, description="1, 2, atau 3")
        ):
            """
            Menyalin hasil step-1/2/3 menjadi data-to-cluster.csv
            """
            if step == 1:
                src_name = "step-1-loaded-data.csv"
            elif step == 2:
                src_name = "step-2-cleaned-columns.csv"
            else:  # step == 3
                src_name = "step-3-removed-incomplete-rows.csv"

            src_path = self._file_path(session_id, src_name)
            self._ensure_file_exists(src_path, f"Data step-{step}")

            df = Analyzer.load_data(src_path, separator=",")
            out_path = self._file_path(session_id, "data-to-cluster.csv")
            df.to_csv(out_path, index=False)

            return {
                "session_id": session_id,
                "source_step": step,
                "file": out_path,
                "rows": len(df),
                "cols": len(df.columns),
                "columns": list(df.columns)
            }

        # ------------- Endpoint-5: /detect-column-types ----
        @self.app.get("/detect-column-types")
        def detect_column_types(
            session_id: str = Query(..., description="ID sesi")
        ):
            """
            Mendeteksi tipe kolom pada data-to-cluster.csv
            """
            in_path = self._file_path(session_id, "data-to-cluster.csv")
            self._ensure_file_exists(in_path, "data-to-cluster")

            df = Analyzer.load_data(in_path, separator=",")
            info = Analyzer.detect_column_types(df)

            out_path = self._file_path(session_id, "step-4-detected-column-types.json")
            pd.Series(info).to_json(out_path)  # simple save

            return {
                "session_id": session_id,
                "file": out_path,
                "column_types": info
            }

        # ------------- Endpoint-6: /prepare-features -------
        @self.app.post("/prepare-features")
        def prepare_features(
            req: PrepareFeaturesRequest,
            session_id: str = Query(..., description="ID sesi")
        ):
            """
            Menyiapkan fitur untuk clustering berdasarkan kolom numeric & categorical.
            Menyimpan ke step-5-prepared-features.csv
            """
            in_path = self._file_path(session_id, "data-to-cluster.csv")
            self._ensure_file_exists(in_path, "data-to-cluster")

            df = Analyzer.load_data(in_path, separator=",")
            all_cols = set(df.columns)

            unknown_numeric = sorted(set(req.numeric) - all_cols)
            unknown_categorical = sorted(set(req.categorical) - all_cols)

            if unknown_numeric or unknown_categorical:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "message": "Beberapa kolom tidak ditemukan di data-to-cluster.",
                        "unknown_numeric": unknown_numeric,
                        "unknown_categorical": unknown_categorical
                    }
                )

            X = Analyzer.prepare_features(df, req.numeric, req.categorical)

            out_path = self._file_path(session_id, "step-5-prepared-features.csv")
            X.to_csv(out_path, index=False)

            return {
                "session_id": session_id,
                "file": out_path,
                "rows": len(X),
                "cols": len(X.columns),
                "columns": list(X.columns),
                "preview": X.head(20).to_dict(orient="records")
            }

        # ------------- Endpoint-7: /run-clustering ---------
        @self.app.post("/run-clustering")
        def run_clustering(
            req: RunClusteringRequest,
            session_id: str = Query(..., description="ID sesi")
        ):
            """
            Menjalankan scaling + clustering pada step-5-prepared-features.csv
            dan menyimpan hasilnya ke step-6-cluster-result-<algorithm>.csv
            """
            algo = req.algorithm.lower()
            valid_algos = {
                "kmeans", "dbscan", "optics", "meanshift",
                "spectral", "agglomerative", "birch", "gmm"
            }
            if algo not in valid_algos:
                raise HTTPException(
                    status_code=400,
                    detail=f"Algoritma tidak dikenal. Pilihan: {sorted(valid_algos)}"
                )

            in_path = self._file_path(session_id, "step-5-prepared-features.csv")
            self._ensure_file_exists(in_path, "step-5-prepared-features")

            X = Analyzer.load_data(in_path, separator=",")
            X_scaled = Analyzer.scale_features(X)

            # Pilih algoritma
            if algo == "kmeans":
                n_clusters = req.n_clusters or 3
                labels = Analyzer.run_kmeans(X_scaled, n_clusters=n_clusters)
            elif algo == "dbscan":
                eps = req.eps or 0.5
                min_samples = req.min_samples or 5
                labels = Analyzer.run_dbscan(X_scaled, eps=eps, min_samples=min_samples)
            elif algo == "optics":
                min_samples = req.min_samples or 5
                xi = req.xi or 0.05
                min_cluster_size = req.min_cluster_size or 0.05
                labels = Analyzer.run_optics(
                    X_scaled,
                    min_samples=min_samples,
                    xi=xi,
                    min_cluster_size=min_cluster_size
                )
            elif algo == "meanshift":
                labels = Analyzer.run_meanshift(X_scaled, bandwidth=req.bandwidth)
            elif algo == "spectral":
                n_clusters = req.n_clusters or 3
                gamma = req.gamma or 1.0
                assign_labels = req.assign_labels or "kmeans"
                labels = Analyzer.run_spectral(
                    X_scaled,
                    n_clusters=n_clusters,
                    gamma=gamma,
                    assign_labels=assign_labels
                )
            elif algo == "agglomerative":
                n_clusters = req.n_clusters or 3
                linkage = req.linkage or "ward"
                labels = Analyzer.run_agglomerative(
                    X_scaled,
                    n_clusters=n_clusters,
                    linkage=linkage
                )
            elif algo == "birch":
                n_clusters = req.n_clusters or 3
                threshold = req.threshold or 0.5
                branching_factor = req.branching_factor or 50
                labels = Analyzer.run_birch(
                    X_scaled,
                    n_clusters=n_clusters,
                    threshold=threshold,
                    branching_factor=branching_factor
                )
            else:  # gmm
                n_clusters = req.n_clusters or 3
                covariance_type = req.covariance_type or "full"
                labels = Analyzer.run_gmm(
                    X_scaled,
                    n_clusters=n_clusters,
                    covariance_type=covariance_type
                )

            # Hitung silhouette score
            labels_set = set(labels)
            if len(labels_set) > 1:
                try:
                    sil = float(
                        Analyzer.profile_clusters(
                            X_scaled,
                            labels,
                            feature_names=list(X_scaled.columns),
                            top_n=1
                        )[1]
                    )
                except Exception:
                    sil = None
            else:
                sil = None

            # Simpan hasil cluster (X_scaled + cluster)
            df_out = X_scaled.copy()
            df_out["cluster"] = labels

            out_name = f"step-6-cluster-result-{algo}.csv"
            out_path = self._file_path(session_id, out_name)
            df_out.to_csv(out_path, index=False)

            counts = pd.Series(labels).value_counts().to_dict()

            return {
                "session_id": session_id,
                "algorithm": algo,
                "file": out_path,
                "rows": len(df_out),
                "cols": len(df_out.columns),
                "cluster_counts": counts,
                "silhouette_score": sil
            }

        # ------------- Endpoint-8: /profile-clusters -------
        @self.app.get("/profile-clusters")
        def profile_clusters(
            algorithm: str = Query(..., description="Nama algoritma (kmeans, birch, dsb.)"),
            session_id: str = Query(..., description="ID sesi"),
            top_n: int = Query(5, description="Jumlah fitur top/bottom")
        ):
            """
            Membaca step-6-cluster-result-<algo>.csv dan membuat profil cluster.
            Menyimpan ke TXT, JSON, dan HTML.
            """
            algo = algorithm.lower()
            in_name = f"step-6-cluster-result-{algo}.csv"
            in_path = self._file_path(session_id, in_name)
            self._ensure_file_exists(in_path, "Hasil cluster step-6")

            df = Analyzer.load_data(in_path, separator=",")
            if "cluster" not in df.columns:
                raise HTTPException(
                    status_code=400,
                    detail="Kolom 'cluster' tidak ditemukan pada file hasil cluster."
                )

            labels = df["cluster"].values
            X_scaled = df.drop(columns=["cluster"])

            profiles, sil = Analyzer.profile_clusters(
                X_scaled,
                labels,
                feature_names=list(X_scaled.columns),
                top_n=top_n
            )

            # ---- Build text report ----
            lines: List[str] = []
            lines.append("=" * 60)
            lines.append(f"Cluster Profiling Report - {algo.upper()}")
            lines.append("=" * 60)
            lines.append("")
            lines.append(f"Jumlah Cluster: {len(profiles.keys())}")
            lines.append(f"Silhouette Score: {sil if sil is not None else 'N/A'}")
            lines.append("")

            for cid, prof in profiles.items():
                lines.append(f"\n--- Cluster {cid} ---\n")
                lines.append("> Ciri Paling Dominan (nilai tinggi):")
                for feat, val in prof["top"].items():
                    lines.append(f"  + {feat}: {val:.2f}")
                lines.append("\n> Ciri Paling Rendah (nilai rendah):")
                for feat, val in prof["bottom"].items():
                    lines.append(f"  - {feat}: {val:.2f}")

            lines.append("\n" + "=" * 60 + "\n")
            report_text = "\n".join(lines)

            # ---- Save TXT, JSON, HTML ----
            base_name = f"step-7-profiled-clusters-{algo}"
            txt_path = self._file_path(session_id, base_name + ".txt")
            json_path = self._file_path(session_id, base_name + ".json")
            html_path = self._file_path(session_id, base_name + ".html")

            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(report_text)

            import json
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "algorithm": algo,
                        "silhouette_score": sil,
                        "profiles": profiles
                    },
                    f,
                    ensure_ascii=False,
                    indent=2
                )

            # HTML sederhana
            html_lines = [
                "<html><head><meta charset='utf-8'><title>Cluster Profiling</title></head><body>",
                f"<h1>Cluster Profiling Report - {algo.upper()}</h1>",
                f"<p><strong>Jumlah Cluster:</strong> {len(profiles.keys())}</p>",
                f"<p><strong>Silhouette Score:</strong> {sil if sil is not None else 'N/A'}</p>",
            ]
            for cid, prof in profiles.items():
                html_lines.append(f"<h2>Cluster {cid}</h2>")
                html_lines.append("<h3>Ciri Paling Dominan (nilai tinggi)</h3><ul>")
                for feat, val in prof["top"].items():
                    html_lines.append(f"<li>{feat}: {val:.2f}</li>")
                html_lines.append("</ul>")
                html_lines.append("<h3>Ciri Paling Rendah (nilai rendah)</h3><ul>")
                for feat, val in prof["bottom"].items():
                    html_lines.append(f"<li>{feat}: {val:.2f}</li>")
                html_lines.append("</ul>")

            html_lines.append("</body></html>")
            with open(html_path, "w", encoding="utf-8") as f:
                f.write("\n".join(html_lines))

            return {
                "session_id": session_id,
                "algorithm": algo,
                "silhouette_score": sil,
                "txt_file": txt_path,
                "json_file": json_path,
                "html_file": html_path,
                "profiles": profiles
            }

        # ------------- Endpoint-9: /visualize-clusters -----
        @self.app.get("/visualize-clusters")
        def visualize_clusters(
            algorithm: str = Query(..., description="Nama algoritma (kmeans, birch, dsb.)"),
            session_id: str = Query(..., description="ID sesi")
        ):
            """
            Membaca step-6-cluster-result-<algo>.csv dan membuat visualisasi PCA 2D.
            Menyimpan PNG ke step-8-visualized-cluster-<algo>.png
            """
            algo = algorithm.lower()
            in_name = f"step-6-cluster-result-{algo}.csv"
            in_path = self._file_path(session_id, in_name)
            self._ensure_file_exists(in_path, "Hasil cluster step-6")

            df = Analyzer.load_data(in_path, separator=",")
            if "cluster" not in df.columns:
                raise HTTPException(
                    status_code=400,
                    detail="Kolom 'cluster' tidak ditemukan pada file hasil cluster."
                )

            labels = df["cluster"].values
            X_scaled = df.drop(columns=["cluster"])

            img_name = f"step-8-visualized-cluster-{algo}.png"
            img_path = self._file_path(session_id, img_name)

            Analyzer.visualize_clusters(
                X_scaled,
                labels,
                save_path=img_path,
                title=f"Cluster Visualization ({algo.upper()})"
            )

            # URL relatif agar bisa diakses via /pipeline/...
            image_url = f"/pipeline/{session_id}/{img_name}"

            return {
                "session_id": session_id,
                "algorithm": algo,
                "image_path": img_path,
                "image_url": image_url
            }

        # ------------- Endpoint-10: /random-sample ---------
        @self.app.get("/random-sample")
        def random_sample(
            session_id: str = Query(..., description="ID sesi"),
            step: int = Query(..., ge=1, le=3, description="Ambil dari step-1/2/3"),
            percent: float = Query(..., gt=0, le=100, description="Persentase baris yang diambil")
        ):
            """
            Menampilkan sampel acak dari salah satu step (1/2/3).
            Tidak wajib menyimpan ke file, hanya dikembalikan sebagai JSON.
            """
            if step == 1:
                src_name = "step-1-loaded-data.csv"
            elif step == 2:
                src_name = "step-2-cleaned-columns.csv"
            else:
                src_name = "step-3-removed-incomplete-rows.csv"

            in_path = self._file_path(session_id, src_name)
            self._ensure_file_exists(in_path, f"Data step-{step}")

            df = Analyzer.load_data(in_path, separator=",")

            try:
                df_sample = Analyzer.random_sample(df, percentage=percent)
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))

            return {
                "session_id": session_id,
                "step": step,
                "percent": percent,
                "total_rows": len(df),
                "sample_rows": len(df_sample),
                "columns": list(df.columns),
                "sample": df_sample.to_dict(orient="records")
            }