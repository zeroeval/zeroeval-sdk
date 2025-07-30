from typing import Any, Callable, Optional, Union, List, Dict
from .writer import ExperimentResultBackendWriter
from .evaluator_class import Evaluator, Evaluation as EvaluationResult
from .evaluation import Evaluation, EvaluationMode, get_evaluation
from .metrics import ColumnMetric, RunMetric, get_column_metric, get_run_metric
import os
import numpy as np
import requests


class Run:
    """
    Represents the result of running a task on a dataset.
    Contains the results and can be evaluated with evaluators.
    """
    
    def __init__(
        self, 
        dataset_name: str,
        dataset_id: str,
        dataset_version_id: str,
        task_name: str,
        task_code: str,
        rows: list[dict[str, Any]],
        outputs: list[str],
        run_number: int = 1,
        total_runs: int = 1,
        task_func: Optional[Callable] = None,
        dataset_ref: Optional[Any] = None,
        experiment_id: Optional[str] = None  # Share experiment ID across runs
    ):
        self.dataset_name = dataset_name
        self.dataset_id = dataset_id
        self.dataset_version_id = dataset_version_id
        self.task_name = task_name
        self.task_code = task_code
        self.rows = rows  # Contains original data + task outputs
        self.outputs = outputs
        self.run_number = run_number
        self.total_runs = total_runs
        self._task_func = task_func  # Store the task function for repeat()
        self._dataset_ref = dataset_ref  # Store dataset reference for repeat()
        self._experiment_id = experiment_id  # Can be shared across runs
        self._writer = ExperimentResultBackendWriter()
        self._evaluators = []
        self._evaluations = []
        self._evaluator_funcs = []  # Store evaluator functions for repeat()
        self._experiment_created = False
        self._result_ids = {}  # Map row_id to experiment_result_id
        self.metrics = {}  # Store dataset-level metrics
        self._all_runs = None  # For run-level evaluations
        self._results_written = False  # Track if results have been written
        
    def _ensure_experiment_exists(self):
        """Create experiment if it doesn't exist yet."""
        if self._experiment_created or self._experiment_id:
            return
            
        # Check if we have valid dataset version ID
        if not self.dataset_version_id:
            print(f"Warning: Cannot create experiment without dataset version ID")
            return
            
        # Create experiment (only once for all runs)
        import requests
        
        self._writer._ensure_auth_setup()
        
        # Create experiment with base name (no run number)
        exp_payload = {
            "dataset_version_id": self.dataset_version_id,
            "name": self.task_name,
            "description": f"Experiment with {self.total_runs} runs" if self.total_runs > 1 else "",
            "alias": self.task_name,
            "workspace_id": ""  # Will be overridden by API key workspace
        }
        
        try:
            exp_response = requests.post(
                f"{self._writer.api_url}/v1/experiments",
                json=exp_payload,
                headers=self._writer._headers,
            )
            exp_response.raise_for_status()
            exp_data = exp_response.json()
            self._experiment_id = exp_data["id"]
            self._experiment_created = True
            print(f"Created experiment: {self._experiment_id}")
        except Exception as e:
            print(f"Warning: Could not create experiment in backend: {e}")
            if hasattr(e, 'response') and e.response:
                print(f"Response: {e.response.text}")
                print(f"Request payload was: {exp_payload}")
                
    def _write_results(self):
        """Write results for this run to the backend."""
        print(f"🔍 _write_results called - experiment_id: {self._experiment_id}, results_written: {self._results_written}")
        print(f"🔍 Number of rows to write: {len(self.rows) if hasattr(self, 'rows') and self.rows else 0}")
        
        if not self._experiment_id or self._results_written:
            print(f"🚫 Skipping _write_results - experiment_id: {self._experiment_id}, results_written: {self._results_written}")
            return
            
        import requests
        
        # Ensure writer is set up
        self._writer._ensure_auth_setup()
        
        # Write each result with the current run_number
        print(f"🔍 Starting _write_results loop with {len(self.rows)} rows")
        for i, row in enumerate(self.rows):
            print(f"🔍 Processing result row {i+1}/{len(self.rows)} - keys: {list(row.keys())}")
            print(f"🔍 Row data: {row}")
            if "row_id" not in row:
                print(f"🚫 Skipping row {i} - no row_id found")
                continue
            print(f"🔍 Row {i} has row_id: {row['row_id']}")
                
            # Extract the task outputs
            result_data = {k: row.get(k) for k in self.outputs if k in row}
            
            # Create experiment result
            result_payload = {
                "dataset_row_id": row["row_id"],
                "result": str(result_data),
                "result_type": "text",
                "trace_id": None,  # NULL instead of empty string
                "run_number": self.run_number,
            }
            
            try:
                result_url = f"{self._writer.api_url}/v1/experiments/{self._experiment_id}/results"
                print(f"🔗 SDK Request: POST {result_url}")
                print(f"📦 Result for row: {row['row_id']}")
                
                result_response = requests.post(
                    result_url,
                    json=result_payload,
                    headers=self._writer._headers,
                )
                print(f"📊 Response: {result_response.status_code} - {result_response.reason}")
                result_response.raise_for_status()
                result_data = result_response.json()
                self._result_ids[row["row_id"]] = result_data["id"]
            except Exception as e:
                print(f"Warning: Could not create result for row {row['row_id']}: {e}")
                if hasattr(e, 'response') and e.response:
                    print(f"Response: {e.response.text}")
        
        # Mark results as written
        self._results_written = True
        
    def eval(self, evaluators: List[Union[Callable, Evaluation, str]], **kwargs) -> "Run":
        """
        Apply evaluators to the run results.
        
        Args:
            evaluators: List of evaluator functions, Evaluation objects, or names
            **kwargs: Additional arguments for evaluators (e.g., gt="answer", pred="prediction")
            
        Returns:
            self for method chaining
            
        Examples:
            # Using decorator-registered evaluations
            run.eval([exact_match, f1_score])
            
            # Using evaluation names
            run.eval(["exact_match", "f1_score"])
            
            # With custom column mappings
            run.eval([exact_match], gt="ground_truth", pred="model_output")
        """
        # Store evaluator functions for potential repeat()
        self._evaluator_funcs.extend(evaluators)
        
        # Normalize evaluators to Evaluation objects
        eval_objects = []
        for evaluator in evaluators:
            if isinstance(evaluator, str):
                # Look up by name
                eval_obj = get_evaluation(evaluator)
                if not eval_obj:
                    raise ValueError(f"Evaluation '{evaluator}' not found in registry")
                eval_objects.append(eval_obj)
            elif isinstance(evaluator, Evaluation):
                eval_objects.append(evaluator)
            elif callable(evaluator):
                # Legacy function - treat as row evaluator
                if hasattr(evaluator, '_is_evaluation'):
                    # It's a decorated evaluation
                    eval_obj = evaluator if isinstance(evaluator, Evaluation) else get_evaluation(evaluator.__name__)
                    eval_objects.append(eval_obj)
                else:
                    # Plain function - wrap as row evaluator
                    from .evaluation import evaluation
                    eval_decorator = evaluation(mode="row", outputs=[evaluator.__name__])
                    eval_obj = eval_decorator(evaluator)
                    eval_objects.append(eval_obj)
            else:
                raise TypeError(f"Invalid evaluator type: {type(evaluator)}")
        
        # Group evaluators by mode
        row_evals = [e for e in eval_objects if e.mode == EvaluationMode.ROW]
        column_evals = [e for e in eval_objects if e.mode == EvaluationMode.COLUMN]
        run_evals = [e for e in eval_objects if e.mode == EvaluationMode.RUN]
        
        # Ensure experiment exists before creating evaluators
        self._ensure_experiment_exists()
        
        # Write results for this run
        self._write_results()
        
        # Process row-level evaluations
        if row_evals:
            print(f"🔍 Processing {len(row_evals)} row evaluations: {[e.name for e in row_evals]}")
            self._process_row_evaluations(row_evals, **kwargs)
        else:
            print(f"🔍 No row evaluations to process")
            
        # Process column-level evaluations
        if column_evals:
            self._process_column_evaluations(column_evals, **kwargs)
            
        # Process run-level evaluations (only if we have access to all runs)
        if run_evals and self._all_runs:
            self._process_run_evaluations(run_evals, self._all_runs, **kwargs)
        elif run_evals:
            print("Warning: Run-level evaluations require multiple runs. Use .repeat() first.")
            
        return self
    
    def _process_row_evaluations(self, evaluators: List[Evaluation], **kwargs):
        """Process row-level evaluations."""
        print(f"🔍 _process_row_evaluations called with {len(evaluators)} evaluators")
        print(f"🔍 Experiment ID: {self._experiment_id}, Run number: {self.run_number}")
        print(f"🔍 Number of rows: {len(self.rows) if hasattr(self, 'rows') and self.rows else 0}")
        
        # Create Evaluator objects if needed (only once per experiment)
        if self._experiment_id and self.run_number == 1:
            for eval_obj in evaluators:
                print(f"🔍 Creating evaluator DB object for: {eval_obj.name}")
                evaluator_db = Evaluator(
                    name=eval_obj.name,
                    code=eval_obj._code,
                    description=eval_obj.description,
                    experiment_id=self._experiment_id
                )
                print(f"🔗 Writing evaluator to backend: {eval_obj.name}")
                evaluator_db._write()
                self._evaluators.append(evaluator_db)
                print(f"✅ Evaluator created and added: {eval_obj.name}")
        
        # Run evaluators on each row
        print(f"🔍 Starting row evaluation loop - {len(self.rows)} rows, {len(evaluators)} evaluators")
        for i, row in enumerate(self.rows):
            print(f"🔍 Processing row {i+1}/{len(self.rows)} - row_id: {row.get('row_id', 'N/A')}")
            # Call each evaluator with the full row
            for j, eval_obj in enumerate(evaluators):
                try:
                    print(f"🔍 Running evaluator '{eval_obj.name}' on row {i}")
                    # Pass the entire row to the evaluator
                    eval_result = eval_obj(row)
                    print(f"🔍 Evaluator result: {eval_result}")
                    
                    # Add evaluation result to the row
                    row.update(eval_result)
                    
                    # Save to database
                    if self._experiment_id and self._evaluators and j < len(self._evaluators):
                        print(f"🔍 Saving evaluation to DB - evaluator index: {j}, available evaluators: {len(self._evaluators)}")
                        self._save_evaluation_result(self._evaluators[j], eval_result, row, i)
                    else:
                        print(f"🚫 Not saving evaluation - experiment_id: {self._experiment_id}, evaluators: {len(self._evaluators) if self._evaluators else 0}, index: {j}")
                        
                except Exception as e:
                    print(f"❌ Error running evaluator {eval_obj.name} on row {i}: {e}")
                    import traceback
                    traceback.print_exc()
    
    def _process_column_evaluations(self, evaluators: List[Evaluation], **kwargs):
        """Process column-level evaluations."""
        # Pass all rows to column evaluators
        for eval_obj in evaluators:
            try:
                # Column evaluator gets access to all rows
                eval_result = eval_obj(self.rows)
                
                # Store metrics at the dataset level
                self.metrics.update(eval_result)
                
                # Save column evaluation to database if needed
                if self._experiment_id:
                    self._save_column_evaluation(eval_obj, eval_result)
                    
            except Exception as e:
                print(f"Error running column evaluator {eval_obj.name}: {e}")
    
    def _process_run_evaluations(self, evaluators: List[Evaluation], all_runs: List["Run"], **kwargs):
        """Process run-level evaluations across multiple runs."""
        # Run each run-level evaluator
        for eval_obj in evaluators:
            try:
                # Pass all runs to the evaluator
                eval_result = eval_obj(all_runs, **kwargs)
                
                # Store in the first run's metrics
                self.metrics.update(eval_result)
                
                # Save run evaluation to database if needed
                if self._experiment_id:
                    self._save_run_evaluation(eval_obj, eval_result, all_runs)
                    
            except Exception as e:
                print(f"Error running run evaluator {eval_obj.name}: {e}")
    
    def _save_evaluation_result(self, evaluator_db: Evaluator, eval_result: dict, row: dict, row_idx: int):
        """Save evaluation result to database."""
        print(f"🔍 _save_evaluation_result called for evaluator: {evaluator_db.name if evaluator_db else 'None'}")
        print(f"🔍 Row ID: {row.get('row_id', str(row_idx))}, Eval result: {eval_result}")
        
        experiment_result_id = self._result_ids.get(row.get("row_id", str(row_idx)))
        print(f"🔍 Found experiment_result_id: {experiment_result_id}")
        
        if not experiment_result_id:
            print(f"🚫 No experiment_result_id found for row {row.get('row_id', str(row_idx))}")
            return
            
        evaluation = EvaluationResult(
            evaluator=evaluator_db,
            result=eval_result,
            experiment_result_id=experiment_result_id,
            dataset_row_id=row.get("row_id", str(row_idx))
        )
        print(f"🔗 About to write evaluation to backend")
        evaluation._write()
        self._evaluations.append(evaluation)
        print(f"✅ Evaluation saved and added to list")
    
    def _save_column_evaluation(self, eval_obj: Evaluation, eval_result: dict):
        """Save column-level evaluation to database."""
        try:
            # Create evaluator if needed
            evaluator_db = Evaluator(
                name=eval_obj.name,
                code=eval_obj._code,
                description=eval_obj.description,
                experiment_id=self._experiment_id,
                evaluation_mode="column"
            )
            evaluator_db._write()
            evaluator_id = evaluator_db._backend_id
            
            payload = {
                "evaluator_id": evaluator_id,
                "experiment_id": self._experiment_id,
                "run_number": self.run_number,
                "evaluation_results": eval_result
            }
            
            eval_url = f"{self._writer.api_url}/v1/column-evaluations"
            print(f"🔗 SDK Request: POST {eval_url}")
            print(f"📦 Column evaluation for experiment: {self._experiment_id}")
            
            response = requests.post(
                eval_url,
                json=payload,
                headers=self._writer._headers
            )
            print(f"📊 Response: {response.status_code} - {response.reason}")
            
            if response.status_code not in [200, 201]:
                print(f"Warning: Failed to save column evaluation: {response.text}")
                print(f"Request payload: {payload}")
                
        except Exception as e:
            print(f"Error saving column evaluation: {e}")
    
    def _save_run_evaluation(self, eval_obj: Evaluation, eval_result: dict, all_runs: List["Run"]):
        """Save run-level evaluation to database."""
        try:
            # Create evaluator if needed
            evaluator_db = Evaluator(
                name=eval_obj.name,
                code=eval_obj._code,
                description=eval_obj.description,
                experiment_id=self._experiment_id,
                evaluation_mode="run"
            )
            evaluator_db._write()
            evaluator_id = evaluator_db._backend_id
            
            payload = {
                "evaluator_id": evaluator_id,
                "experiment_id": self._experiment_id,
                "total_runs": len(all_runs),
                "evaluation_results": eval_result
            }
            
            response = requests.post(
                f"{self._writer.api_url}/workspaces/{self._writer.workspace_id}/experiments/{self._experiment_id}/run-evaluations",
                json=payload,
                headers=self._writer._headers
            )
            
            if response.status_code not in [200, 201]:
                print(f"Warning: Failed to save run evaluation: {response.text}")
                
        except Exception as e:
            print(f"Error saving run evaluation: {e}")
        
    def repeat(self, n: int) -> list["Run"]:
        """
        Repeat the experiment n times total (including this run).
        
        Args:
            n: Total number of runs (must be at least 1)
            
        Returns:
            List of all runs (including this one)
        """
        if not self._task_func or not self._dataset_ref:
            raise ValueError("Cannot repeat without task function and dataset reference. Use dataset.run() instead.")
            
        if n < 1:
            raise ValueError("Number of repeats must be at least 1")
            
        # Update total runs for all runs
        self.total_runs = n
            
        # Include the current run
        all_runs = [self]
        
        # Create additional runs with the same experiment ID
        for run_num in range(2, n + 1):
            print(f"Running experiment (run {run_num}/{n})...")
            
            # Re-run the task on the dataset with proper run number and shared experiment ID
            new_run = self._dataset_ref.run(
                self._task_func, 
                run_number=run_num, 
                total_runs=n,
                experiment_id=self._experiment_id  # Share the experiment ID
            )
            
            # Share the evaluator objects from the first run
            new_run._evaluators = self._evaluators
            
            # Apply the same evaluators
            if self._evaluator_funcs:
                new_run.eval(self._evaluator_funcs)
                
            all_runs.append(new_run)
            
        self._all_runs = all_runs # Store all runs for run-level evaluations
        return all_runs
        
    def score(self, evaluators: list[Callable]) -> "Run":
        """
        Alias for eval() to match the user's preferred API.
        
        Example:
            dataset.run(task).score([exact_match]).repeat(8)
        """
        return self.eval(evaluators)
        
    def __repr__(self):
        run_info = f"run {self.run_number}/{self.total_runs}" if self.total_runs > 1 else "single run"
        return f"Run(dataset='{self.dataset_name}', task='{self.task_name}', rows={len(self.rows)}, evaluators={len(self._evaluators)}, {run_info})"
        
    def __len__(self):
        return len(self.rows)
        
    def __getitem__(self, idx):
        return self.rows[idx] 

    def column_metrics(self, metrics: List[Union[Callable, ColumnMetric, str]]) -> "Run":
        """Apply column-level metrics to the dataset.
        
        Args:
            metrics: List of column metrics (functions, ColumnMetric objects, or names)
        """
        # Normalize to ColumnMetric objects
        metric_objects = []
        for metric in metrics:
            if isinstance(metric, str):
                metric_obj = get_column_metric(metric)
                if not metric_obj:
                    print(f"Warning: Column metric '{metric}' not found")
                    continue
            elif isinstance(metric, ColumnMetric):
                metric_obj = metric
            elif callable(metric):
                # Treat as direct function
                metric_obj = ColumnMetric(
                    name=metric.__name__,
                    func=metric,
                    outputs=[]
                )
            else:
                print(f"Warning: Invalid metric type: {type(metric)}")
                continue
            metric_objects.append(metric_obj)
        
        # Apply each column metric
        for metric_obj in metric_objects:
            try:
                # Column metric gets all rows
                result = metric_obj(self.rows)
                
                # Store in run metrics
                self.metrics.update(result)
                
                # Save to database
                if self._experiment_id:
                    self._save_column_metric(metric_obj, result)
                    
            except Exception as e:
                print(f"Error running column metric {metric_obj.name}: {e}")
        
        return self

    def run_metrics(self, metrics: List[Union[Callable, RunMetric, str]], all_runs: List["Run"]) -> "Run":
        """Apply run-level metrics across multiple runs.
        
        Args:
            metrics: List of run metrics (functions, RunMetric objects, or names)
            all_runs: List of all Run objects to analyze
        """
        # Normalize to RunMetric objects
        metric_objects = []
        for metric in metrics:
            if isinstance(metric, str):
                metric_obj = get_run_metric(metric)
                if not metric_obj:
                    print(f"Warning: Run metric '{metric}' not found")
                    continue
            elif isinstance(metric, RunMetric):
                metric_obj = metric
            elif callable(metric):
                # Treat as direct function
                metric_obj = RunMetric(
                    name=metric.__name__,
                    func=metric,
                    outputs=[]
                )
            else:
                print(f"Warning: Invalid metric type: {type(metric)}")
                continue
            metric_objects.append(metric_obj)
        
        # Apply each run metric
        for metric_obj in metric_objects:
            try:
                # Run metric gets all runs
                result = metric_obj(all_runs)
                
                # Store in run metrics
                self.metrics.update(result)
                
                # Save to database
                if self._experiment_id:
                    self._save_run_metric(metric_obj, result, all_runs)
                    
            except Exception as e:
                print(f"Error running run metric {metric_obj.name}: {e}")
        
        return self

    def _save_column_metric(self, metric_obj: ColumnMetric, result: dict):
        """Save column metric to database via simple API call."""
        try:
            payload = {
                "metric_name": metric_obj.name,
                "experiment_id": self._experiment_id,
                "run_number": self.run_number,
                "results": result
            }
            
            metric_url = f"{self._writer.api_url}/v1/column-metrics"
            print(f"🔗 SDK Request: POST {metric_url}")
            print(f"📦 Column metric: {metric_obj.name} for experiment: {self._experiment_id}")
            
            response = requests.post(
                metric_url,
                json=payload,
                headers=self._writer._headers
            )
            print(f"📊 Response: {response.status_code} - {response.reason}")
            
            if response.status_code not in [200, 201]:
                print(f"Warning: Failed to save column metric: {response.text}")
                print(f"Request payload: {payload}")
                
        except Exception as e:
            print(f"Error saving column metric: {e}")

    def _save_run_metric(self, metric_obj: RunMetric, result: dict, all_runs: List["Run"]):
        """Save run metric to database via simple API call."""
        try:
            payload = {
                "metric_name": metric_obj.name,
                "experiment_id": self._experiment_id,
                "total_runs": len(all_runs),
                "results": result
            }
            
            run_metric_url = f"{self._writer.api_url}/v1/run-metrics"
            print(f"🔗 SDK Request: POST {run_metric_url}")
            print(f"📦 Run metric: {metric_obj.name} for experiment: {self._experiment_id}")
            
            response = requests.post(
                run_metric_url,
                json=payload,
                headers=self._writer._headers
            )
            print(f"📊 Response: {response.status_code} - {response.reason}")
            
            if response.status_code not in [200, 201]:
                print(f"Warning: Failed to save run metric: {response.text}")
                print(f"Request payload: {payload}")
                
        except Exception as e:
            print(f"Error saving run metric: {e}") 