# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from pathlib import Path
import numpy as np
from tqdm import tqdm
from sdp.processors.base_processor import BaseProcessor
from typing import List, Dict, Union, Optional, Tuple
from . import metrics_computation as metrics

class BootstrapProcessor(BaseProcessor):
    """
    Performs bootstrapped metric computation (WER, CER, WMR, etc.) on speech predictions
    and computes the probability of improvement (POI) between models if `calculate_pairwise`
    is set to True.

    Args:
        manifest_files (List[str]): List of file paths to manifest (JSONL) files for metric calculation
        num_bootstraps (int): Number of bootstrap iterations for metric computation
        dataset_size (float): Proportion of dataset size for each bootstrap sample
        output_file (str): Path to the output JSON file to save results
        calculate_pairwise (bool): Whether to calculate pairwise metric difference and POI (default: True)
        metric_type (str): The type of metric to calculate, options include 'wer', 'cer', 'wmr', 'charrate', 'wordrate' (default: 'wer')
        text_key (str): The key in the manifest that contains the ground truth text (default: 'text')
        pred_text_key (str): The key in the manifest that contains the predicted text (default: 'pred_text')
        ci_lower (float): Lower bound percentile for confidence intervals (default: 2.5)
        ci_upper (float): Upper bound percentile for confidence intervals (default: 97.5)
    """

    def __init__(
        self,
        manifest_files: List[str],
        num_bootstraps: int = 1000,
        dataset_size: float = 1.0,
        output_file: Optional[str] = None,
        calculate_pairwise: bool = True,  # Option for pairwise comparison
        metric_type: str = 'wer',  # Default metric is WER
        text_key: str = 'text',  # Default key for ground truth text in the manifest
        pred_text_key: str = 'pred_text',  # Default key for predicted text in the manifest
        ci_lower: float = 2.5,  # Lower percentile for confidence intervals
        ci_upper: float = 97.5,  # Upper percentile for confidence intervals
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.manifest_files = manifest_files
        self.num_bootstraps = num_bootstraps
        self.dataset_size = dataset_size
        self.output_file = output_file
        self.calculate_pairwise = calculate_pairwise  # Store the option to calculate pairwise metrics
        self.metric_type = metric_type.lower()  # Store metric type and convert to lowercase for consistency
        self.text_key = text_key  # Store the key for ground truth text
        self.pred_text_key = pred_text_key  # Store the key for predicted text
        self.ci_lower = ci_lower  # Store the lower percentile for confidence intervals
        self.ci_upper = ci_upper  # Store the upper percentile for confidence intervals

        # Validate metric type
        if self.metric_type not in ['wer', 'cer', 'wmr', 'charrate', 'wordrate']:
            raise ValueError(f"Invalid metric_type '{self.metric_type}'! Must be one of ['wer', 'cer', 'wmr', 'charrate', 'wordrate']")

    def read_manifest(self, manifest_path: Path) -> List[Dict[str, Union[str, float]]]:
        """
        Read a manifest file in JSONL format and return a list of dictionaries.

        Args:
            manifest_path (Path): Path to the manifest file (.jsonl)

        Returns:
            List[Dict[str, Union[str, float]]]: A list of dictionaries where each dictionary corresponds to an entry in the manifest.
        """
        manifest_data = []
        with manifest_path.open('r', encoding='utf-8') as f:
            for line in f:
                # Each line in the manifest file is a JSON object
                data = json.loads(line.strip())  # Parse the JSON object
                manifest_data.append(data)

        return manifest_data

    def calculate_metric(self, text: str, pred_text: str, duration: Optional[float] = None) -> float:
        """
        Calculate the specified metric between ground truth and predicted text.

        Args:
            text (str): Ground truth text
            pred_text (str): Predicted text
            duration (Optional[float]): Duration of the audio (used for charrate and wordrate)

        Returns:
            float: Computed metric value
        """
        if self.metric_type == 'wer':
            return metrics.get_wer(text, pred_text)
        elif self.metric_type == 'cer':
            return metrics.get_cer(text, pred_text)
        elif self.metric_type == 'wmr':
            return metrics.get_wmr(text, pred_text)
        elif self.metric_type == 'charrate':
            if duration is None:
                raise ValueError("Duration is required for calculating character rate.")
            return metrics.get_charrate(text, duration)
        elif self.metric_type == 'wordrate':
            if duration is None:
                raise ValueError("Duration is required for calculating word rate.")
            return metrics.get_wordrate(text, duration)
        else:
            raise ValueError(f"Unsupported metric_type: {self.metric_type}")

    def bootstrap_metric(self, hypotheses: List[str], references: List[str], durations: Optional[List[float]] = None) -> np.ndarray:
        """
        Bootstraps metric computation (WER, CER, etc.) to calculate confidence intervals.

        Args:
            hypotheses (List[str]): Predicted transcriptions
            references (List[str]): Ground truth transcriptions
            durations (Optional[List[float]]): Duration for each transcription, required for charrate and wordrate

        Returns:
            np.ndarray: Bootstrapped metric values
        """
        n = len(hypotheses)
        sample_size = int(n * self.dataset_size)

        metric_bootstrap = []
        for _ in tqdm(range(self.num_bootstraps), desc=f"Bootstrapping {self.metric_type.upper()}"):
            indices = np.random.choice(n, size=sample_size, replace=True)
            sampled_hypotheses = [hypotheses[i] for i in indices]
            sampled_references = [references[i] for i in indices]
            if durations:
                sampled_durations = [durations[i] for i in indices]
                metric = [self.calculate_metric(sampled_references[i], sampled_hypotheses[i], sampled_durations[i])
                          for i in range(sample_size)]
            else:
                metric = [self.calculate_metric(sampled_references[i], sampled_hypotheses[i]) for i in range(sample_size)]
            metric_bootstrap.append(np.mean(metric))

        return np.array(metric_bootstrap)

    def bootstrap_wer_difference(self, predictions1: List[str], predictions2: List[str], references: List[str], durations: Optional[List[float]] = None) -> Tuple[np.ndarray, float]:
        """
        Calculates the bootstrapped difference in metrics between two sets of predictions and the probability of improvement.

        Args:
            predictions1 (List[str]): Predictions from the first model
            predictions2 (List[str]): Predictions from the second model
            references (List[str]): Ground truth references
            durations (Optional[List[float]]): Durations for each sample, if required for the metric

        Returns:
            Tuple[np.ndarray, float]: A tuple containing:
                - np.ndarray: Bootstrapped differences in metric
                - float: Probability of Improvement (POI)
        """
        n = len(references)
        sample_size = int(n * self.dataset_size)
        delta_metric_bootstrap = []

        for _ in tqdm(range(self.num_bootstraps), desc=f"Bootstrapping {self.metric_type.upper()} difference"):
            indices = np.random.choice(n, size=sample_size, replace=True)
            sampled_pred1 = [predictions1[i] for i in indices]
            sampled_pred2 = [predictions2[i] for i in indices]
            sampled_refs = [references[i] for i in indices]

            if durations:
                sampled_durations = [durations[i] for i in indices]
                metric1 = [self.calculate_metric(sampled_refs[i], sampled_pred1[i], sampled_durations[i]) for i in range(sample_size)]
                metric2 = [self.calculate_metric(sampled_refs[i], sampled_pred2[i], sampled_durations[i]) for i in range(sample_size)]
            else:
                metric1 = [self.calculate_metric(sampled_refs[i], sampled_pred1[i]) for i in range(sample_size)]
                metric2 = [self.calculate_metric(sampled_refs[i], sampled_pred2[i]) for i in range(sample_size)]

            delta_metric = np.mean(metric1) - np.mean(metric2)
            delta_metric_bootstrap.append(delta_metric)

        poi = np.mean(np.array(delta_metric_bootstrap) > 0)
        return np.array(delta_metric_bootstrap), poi

    def process(self):
        """
        Main processing function that loads data, performs metric bootstrapping and optionally 
        pairwise comparison, and saves the results to a JSON file.
        """
        results = {}

        # Load ground truth and predictions
        manifest_files = [Path(f) for f in self.manifest_files]
        ground_truth = []
        predicted_texts = []
        durations = []  # Optional durations for charrate and wordrate

        for manifest_file in manifest_files:
            manifest_data = self.read_manifest(manifest_file)
            # Use text_key and pred_text_key to extract ground truth and predictions
            gt_texts = [entry[self.text_key] for entry in manifest_data]
            pred_texts = [entry[self.pred_text_key] for entry in manifest_data]
            if 'duration' in manifest_data[0]:  # Check if duration is available
                file_durations = [entry['duration'] for entry in manifest_data]
                durations.append(file_durations)

            if not ground_truth:
                ground_truth = gt_texts  # Ground truth is assumed to be the same for all models
            predicted_texts.append(pred_texts)

        # Bootstrapping individual metric for each model
        results["individual_results"] = {}
        for idx, predicted in enumerate(predicted_texts):
            if durations:
                metric_conf_intervals = self.bootstrap_metric(predicted, ground_truth, durations[idx])
            else:
                metric_conf_intervals = self.bootstrap_metric(predicted, ground_truth)

            ci_lower_value = np.percentile(metric_conf_intervals, self.ci_lower)
            ci_upper_value = np.percentile(metric_conf_intervals, self.ci_upper)
            mean_metric = np.mean(metric_conf_intervals)

            results["individual_results"][manifest_files[idx].name] = {
                f"mean_{self.metric_type}": mean_metric,
                "ci_lower": ci_lower_value,
                "ci_upper": ci_upper_value
            }

        # Pairwise comparison between models (only if calculate_pairwise is True)
        if self.calculate_pairwise:
            results["pairwise_comparisons"] = []
            num_files = len(predicted_texts)
            for i in range(num_files):
                for j in range(i + 1, num_files):
                    if durations:
                        delta_metric_bootstrap, poi = self.bootstrap_wer_difference(predicted_texts[i], predicted_texts[j], ground_truth, durations[i])
                    else:
                        delta_metric_bootstrap, poi = self.bootstrap_wer_difference(predicted_texts[i], predicted_texts[j], ground_truth)

                    mean_delta_metric = np.mean(delta_metric_bootstrap)
                    ci_lower_value = np.percentile(delta_metric_bootstrap, self.ci_lower)
                    ci_upper_value = np.percentile(delta_metric_bootstrap, self.ci_upper)

                    results["pairwise_comparisons"].append({
                        "file_1": manifest_files[i].name,
                        "file_2": manifest_files[j].name,
                        f"delta_{self.metric_type}_mean": mean_delta_metric,
                        "ci_lower": ci_lower_value,
                        "ci_upper": ci_upper_value,
                        "poi": poi
                    })

        # Save results to output file
        if self.output_file:
            output_path = Path(self.output_file)
            output_path.parent.mkdir(exist_ok=True, parents=True)
            with output_path.open('w') as out_file:
                json.dump(results, out_file, indent=4)

            print(f"Results saved to {self.output_file}")

