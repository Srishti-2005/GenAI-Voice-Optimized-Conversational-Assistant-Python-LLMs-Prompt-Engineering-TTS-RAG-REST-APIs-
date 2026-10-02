import os
import json
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import mlflow
from mlflow.entities import ViewType
from rag_assistant.tracking.evaluator import EvaluationResult


class MLflowTracker:
    """
    MLflow tracking integration for RAG parameter tuning,
    prompt experiment logging, and retrieval performance evaluation.
    """

    def __init__(self, tracking_uri: str, experiment_name: str = "Enterprise-Document-RAG"):
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self._setup_mlflow()

    def _setup_mlflow(self):
        try:
            mlflow.set_tracking_uri(self.tracking_uri)
            # Create or set experiment
            exp = mlflow.get_experiment_by_name(self.experiment_name)
            if exp is None:
                self.experiment_id = mlflow.create_experiment(self.experiment_name)
            else:
                self.experiment_id = exp.experiment_id
            mlflow.set_experiment(self.experiment_name)
        except Exception as e:
            # Fallback to local default file store
            fallback_dir = Path("./mlruns").resolve()
            fallback_dir.mkdir(parents=True, exist_ok=True)
            mlflow.set_tracking_uri(f"file:///{fallback_dir.as_posix()}")
            mlflow.set_experiment(self.experiment_name)
            self.experiment_id = mlflow.get_experiment_by_name(self.experiment_name).experiment_id

    def log_rag_run(
        self,
        run_name: str,
        parameters: Dict[str, Any],
        metrics: Dict[str, float],
        query: str,
        response_text: str,
        retrieved_chunks: List[Dict[str, Any]],
        eval_result: Optional[EvaluationResult] = None,
    ) -> str:
        """Log a complete RAG execution trace, parameters, metrics, and artifacts to MLflow."""
        with mlflow.start_run(experiment_id=self.experiment_id, run_name=run_name) as run:
            run_id = run.info.run_id

            # 1. Log Hyperparameters & Experiment Config
            safe_params = {}
            for k, v in parameters.items():
                safe_params[k] = str(v) if not isinstance(v, (int, float, bool, str)) else v
            mlflow.log_params(safe_params)

            # 2. Log Quantitative Retrieval & Generation Metrics
            safe_metrics = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
            if eval_result:
                safe_metrics["eval_avg_similarity"] = eval_result.avg_similarity
                safe_metrics["eval_context_relevance"] = eval_result.context_relevance
                safe_metrics["eval_mrr_score"] = eval_result.mrr_score
                safe_metrics["eval_precision_at_k"] = eval_result.precision_at_k
                safe_metrics["eval_citation_coverage"] = eval_result.citation_coverage

            mlflow.log_metrics(safe_metrics)

            # 3. Log Informative Tags
            doc_sources = list({c.get("filename", "unknown") for c in retrieved_chunks})
            mlflow.set_tag("query_preview", query[:100])
            mlflow.set_tag("doc_sources", ", ".join(doc_sources))
            mlflow.set_tag("retrieval_mode", parameters.get("retrieval_mode", "hybrid"))

            # 4. Log Rich Artifacts
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_path = Path(tmp_dir)

                # Artifact 1: Query & Response Markdown report
                report_file = tmp_path / "rag_report.md"
                report_content = (
                    f"# RAG Execution Report\n\n"
                    f"**Run ID:** `{run_id}`\n"
                    f"**Query:** {query}\n"
                    f"**Model:** {parameters.get('llm_model')}\n"
                    f"**Retrieval Mode:** {parameters.get('retrieval_mode')}\n\n"
                    f"## Generated Response\n{response_text}\n\n"
                    f"## Retrieved Context Chunks ({len(retrieved_chunks)})\n"
                )
                for idx, c in enumerate(retrieved_chunks):
                    report_content += (
                        f"### Snippet {idx + 1}: {c.get('filename')} (ID: {c.get('chunk_id')})\n"
                        f"- Score: {c.get('final_score', c.get('similarity_score', 0)):.4f}\n"
                        f"- Content:\n> {c.get('content', '')}\n\n"
                    )
                report_file.write_text(report_content, encoding="utf-8")
                mlflow.log_artifact(str(report_file))

                # Artifact 2: Evaluation JSON
                if eval_result:
                    eval_file = tmp_path / "eval_metrics.json"
                    eval_data = {
                        "avg_similarity": eval_result.avg_similarity,
                        "context_relevance": eval_result.context_relevance,
                        "mrr_score": eval_result.mrr_score,
                        "precision_at_k": eval_result.precision_at_k,
                        "citation_coverage": eval_result.citation_coverage,
                        "details": eval_result.details,
                    }
                    eval_file.write_text(json.dumps(eval_data, indent=2), encoding="utf-8")
                    mlflow.log_artifact(str(eval_file))

            return run_id

    def get_runs_dataframe(self, max_results: int = 50) -> pd.DataFrame:
        """Retrieve recent MLflow runs as a structured pandas DataFrame."""
        try:
            runs = mlflow.search_runs(
                experiment_ids=[self.experiment_id],
                run_view_type=ViewType.ACTIVE_ONLY,
                max_results=max_results,
                order_by=["start_time DESC"],
            )
            return runs
        except Exception as e:
            return pd.DataFrame()
