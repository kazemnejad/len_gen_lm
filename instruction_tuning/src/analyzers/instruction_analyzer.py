import copy
import logging
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, Deque

import jsonlines
import wandb
from nltk import edit_distance
from tqdm import tqdm

from analyzers import Analyzer, Seq2SeqAnalyzer
from common import ExperimentStage
from data import Seq2SeqDataLoaderFactory

from datasets import load_metric

logger = logging.getLogger("app")


@Analyzer.register("instruction")
class InstructionAnalyzer(Seq2SeqAnalyzer):
    dl_factory: Seq2SeqDataLoaderFactory

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert isinstance(self.dl_factory, Seq2SeqDataLoaderFactory)

    def analyze(self):
        predictions_path = self.exp_root / f"pred_out_{self.split}.jsonl"
        if predictions_path.exists():
            (
                evaluation_table,
                accuracies,
                distances,
                generation_ref_pairs,
            ) = self._analyze_prediction(predictions_path)
            self.logger.log({f"evaluated_acc/{self.split}/table": evaluation_table})

            self.log_accuracies_and_distances(
                copy.deepcopy(accuracies), copy.deepcopy(distances), copy.deepcopy(generation_ref_pairs)
            )

    def _analyze_prediction(self, predictions_path: Path):

        assert (
            predictions_path.exists()
        ), f"Prediction file not found: {predictions_path}"

        pred_objs = []
        with jsonlines.open(str(predictions_path)) as reader:
            for obj in reader:
                pred_objs.append(obj)

        ds_path = self.dl_factory.get_ds_file_path(
            ExperimentStage.from_split(self.split)
        )
        logger.info(f"Evaluating against split: {self.split} at {ds_path}")

        dataset_objs = self.dl_factory.get_dataset(
            ExperimentStage.PREDICTION, path=ds_path
        )
        assert len(dataset_objs) == len(pred_objs)

        accuracies: Dict[str, Deque] = defaultdict(deque)
        distances: Dict[str, Deque] = defaultdict(deque)
        generation_ref_pairs: Dict[str, Deque] = defaultdict(deque)

        scratchpad_metrics: Dict[str, Dict[str, Deque]] = defaultdict(
            lambda: defaultdict(deque)
        )

        evaluation_table = wandb.Table(
            columns=[
                "idx",
                "prediction",
                "gold_answer",
                "is_correct",
                "edit_distance",
                "parse_error",
            ]
        )
        for idx, (pred_obj, ds_obj) in tqdm(
                enumerate(zip(pred_objs, dataset_objs)), total=len(pred_objs)
        ):
            category = ds_obj["category"]
            prediction = pred_obj["prediction"]

            if hasattr(self.dl_factory.instance_processor, "_create_answer"):
                gold_answer = self.dl_factory.instance_processor._create_answer(ds_obj)
            else:
                gold_answer = ds_obj["answer"]
                
#             breakpoint()

            try:
                parsed_pred = (
                    self.dl_factory.instance_processor.extract_answer_from_prediction(
                        prediction
                    )
                )
                ed = 0

                is_correct = self.dl_factory.instance_processor.is_prediction_correct(
                    prediction, ds_obj
                )

                if hasattr(self.dl_factory.instance_processor, "evaluate_scratchpad"):
                    eval_result = (
                        self.dl_factory.instance_processor.evaluate_scratchpad(
                            prediction, ds_obj
                        )
                    )
                else:
                    eval_result = None

                exp_str = ""
            except Exception as exp:
                logger.warning(f"Couldn't parse the model's prediction {exp}")
                is_correct = False
                exp_str = str(exp)
                parsed_pred = prediction
                ed = 100
                eval_result = None

            accuracies[category].append(is_correct)
            distances[category].append(ed)
            generation_ref_pairs[category].append((str(parsed_pred), gold_answer))

            if eval_result is not None:
                for k, v in eval_result.items():
                    scratchpad_metrics[k][category].append(v)

            evaluation_table.add_data(
                idx, str(parsed_pred), gold_answer, is_correct, ed, exp_str
            )

        return evaluation_table, accuracies, distances, generation_ref_pairs

    def log_accuracies_and_distances(self, accuracies, distances, generation_ref_pairs, prefix: str = ""):
        # Compute BLEU and ROUGE
        bleu = load_metric("sacrebleu")
        # rouge = load_metric("rouge")
        for cat, pairs in generation_ref_pairs.items():
            predictions = [pair[0] for pair in pairs]
            references = [[pair[1]] for pair in pairs]
            result = bleu.compute(predictions=predictions, references=references)
            for key, value in result.items():
                self.logger.log({f"pred/{self.split}_{prefix}bleu_{cat}_{key}": value})
                self.log({f"pred/{self.split}_{prefix}bleu_{cat}_{key}": value})

            # result = rouge.compute(predictions=predictions, references=references)
            # for key, value in result.items():
            #     self.logger.log({f"pred/{self.split}_{prefix}rouge_{cat}_{key}": value})
            #     self.log({f"pred/{self.split}_{prefix}rouge_{cat}_{key}": value})

        stats = []
        for key, acc_lst in accuracies.items():
            acc = sum(acc_lst) / len(acc_lst)
            acc = round(acc, 4)
            stats.append((f"{key}", acc))
            self.logger.log({f"pred/{self.split}_{prefix}acc_{key}": acc})
            self.log({f"pred/{self.split}_{prefix}acc_{key}": acc})

        all_predictions = [
            is_correct for acc_lst in accuracies.values() for is_correct in acc_lst
        ]
        overall_acc = sum(all_predictions) / len(all_predictions)
        overall_acc = round(overall_acc, 4)
        stats.append(("overall", overall_acc))
        self.logger.log({f"pred/{self.split}_{prefix}acc_overall": overall_acc})
        self.log({f"pred/{self.split}_{prefix}acc_overall": overall_acc})

        plot = wandb.plot.bar(
            wandb.Table(data=stats, columns=["split", "eAcc"]),
            label="split",
            value="eAcc",
            title=f"Evaluated {prefix} accuracy in split: {self.split}",
        )
        self.logger.log({f"evaluated_acc/{self.split}/{prefix}plot": plot})
        stats = []
        for key, dist_lst in distances.items():
            dist = sum(dist_lst) / len(dist_lst)
            dist = round(dist, 4)
            stats.append((f"{key}", dist))
            self.logger.log({f"pred/{self.split}_{prefix}editDistance_{key}": dist})
            self.log({f"pred/{self.split}_{prefix}editDistance_{key}": dist})
        distances = [dist for dist_lst in distances.values() for dist in dist_lst]
        overall_dist = sum(distances) / len(distances)
        overall_dist = round(overall_dist, 4)
        stats.append(("overall", overall_dist))
        self.logger.log(
            {f"pred/{self.split}_{prefix}editDistance_overall": overall_dist}
        )
        self.log({f"pred/{self.split}_{prefix}editDistance_overall": overall_dist})

        plot = wandb.plot.bar(
            wandb.Table(data=stats, columns=["split", "edist"]),
            label="split",
            value="edist",
            title=f"Evaluated {prefix} Edit Distance (editDistance) in split: {self.split}",
        )
        self.logger.log({f"evaluated_acc/{self.split}/{prefix}editDistance_plot": plot})