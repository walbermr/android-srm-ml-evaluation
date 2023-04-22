import os
import json
import ensemble_classifiers

from copy import deepcopy
from tqdm import tqdm

from utils.latex import (
    print_latex_table_header,
    print_latex_table_entry,
    print_latex_table_footer,
)
from utils.logging import print_metrics, save_metrics

from experiments.base_experiments import BasicExperiments, ClassMetric

from ensemble_classifiers import PoolManager


class StratificationExperiment(BasicExperiments):
    def __init__(
        self,
        models,
        dataset,
        samples_path="./datasetutils/sampled_db/",
        names=None,
        reduction_algorithm=None,
        use_latex=True,
        labels=["source", "sink", "neithernor"],
        experiment_name="default",
        train_classifiers=True,
    ):
        self._train_classifiers = train_classifiers
        super().__init__(
            models,
            dataset,
            samples_path,
            names,
            reduction_algorithm,
            use_latex,
            labels,
            experiment_name,
        )

    def run_experiment(self):

        (
            recursive_directories,
            path_is_recursive,
        ) = self.get_recursive_directories(self.samples_path)

        if isinstance(self.models, list):
            model_it = self.models
        elif isinstance(self.models, dict):
            if isinstance(list(self.models.values())[0], dict):
                table_indexer = self.models
                model_it = None
            else:
                model_it = self.models.keys()
                table_indexer = {"Classifiers": self.models}
        else:
            raise ValueError("Model iterator must be list or dictionary.")

        models_predictions = {}
        models_accuracies = {}

        for t in table_indexer:
            model_it = table_indexer[t]
            models_predictions[t] = {}
            models_accuracies[t] = {}

            if path_is_recursive:
                prefix = "%s_%s" % (t, self._experiment_name)
            else:
                prefix = self._experiment_name

            # execute the experiment
            for sub_directory in recursive_directories:
                PoolManager().clear()
                for n_components in sub_directory:
                    # open table
                    if self.use_latex:
                        if path_is_recursive:
                            aux = list(n_components.keys())[0].split("_")
                            # get embedding algo name and num dims
                            reduction_algorithm_name = aux[0]
                            n_components_name = aux[1]
                            latex_table = open(
                                os.path.join(
                                    self.get_output_directory(),
                                    "%s_latex_table_%s-%s.tex"
                                    % (
                                        t.lower().replace(" ", "_"),
                                        reduction_algorithm_name,
                                        n_components_name,
                                    ),
                                ),
                                "w+",
                            )

                            print_latex_table_header(latex_table)
                        else:
                            latex_table = open(
                                os.path.join(
                                    self.get_output_directory(),
                                    "%s_results.tex"
                                    % (t.lower().replace(" ", "_"),),
                                ),
                                "w+",
                            )
                            print_latex_table_header(latex_table)

                    n_components_name = list(n_components.keys())[0]
                    n_components_path = n_components[n_components_name]

                    self.dataset.load_samples(
                        n_components_path, recursive_reading=path_is_recursive
                    )

                    for m in model_it:
                        source = ClassMetric("source")
                        sink = ClassMetric("sink")
                        neithernor = ClassMetric("neithernor")
                        global_metrics = ClassMetric("global")
                        name = ""
                        predictions = []

                        classes_metrics = [source, sink, neithernor]

                        if isinstance(self.models, list):
                            if m.__class__.__name__ == "Ensemble":
                                name = m.__name__
                            else:
                                name = m.__class__.__name__

                            name = self.beautify_name(name)

                        elif isinstance(m, str):
                            name = m
                            model = self.models[t][m]

                        else:
                            raise ValueError(
                                "Model iterator must be list or dictionary."
                            )

                        if path_is_recursive:
                            name = n_components_name + "_" + name

                        print("\n%s" % (name))

                        for i in tqdm(
                            range(0, len(self.dataset.dataframe["samples"]))
                        ):
                            prediction = []
                            m = deepcopy(model)
                            (
                                X_train,
                                y_train,
                                X_test,
                                y_test,
                            ) = self.dataset.get_sampled(i)

                            x_train, x_test = X_train.values, X_test.values
                            if self._train_classifiers:
                                try:
                                    m.fit(x_train, y_train)
                                except:
                                    m.fit(
                                        x_train,
                                        self._encoder.inverse_transform(
                                            y_train
                                        ),
                                    )

                                if "oracle" in name.lower():
                                    prediction = m.predict(x_test, y=y_test)
                                else:
                                    prediction = m.predict(x_test)
                            else:
                                prediction = self._get_saved_model_prediction(
                                    prefix, name, i
                                )

                            report, prediction = self.get_metrics(
                                y_test, prediction
                            )
                            for c in classes_metrics:
                                c.add_values(report[c.class_label])
                            global_metrics.add_values(report["global"])

                            predictions.append(
                                self.to_label(prediction).tolist()
                            )

                        print_metrics(classes_metrics, global_metrics)
                        save_metrics(name, classes_metrics)

                        if self.use_latex:
                            print_latex_table_entry(
                                latex_table,
                                name,
                                global_metrics,
                                new_mode=True,
                            )

                        models_predictions[t][name] = predictions
                        models_accuracies[t][name] = list(
                            global_metrics.accuracies
                        )

                    if self.use_latex:
                        print_latex_table_footer(
                            latex_table,
                            t,
                            new_mode=True,
                            colorize_oracle=False,
                        )
                        latex_table.close()

        print("Writing to: %s prefix." % (prefix))
        predictions_file_path = os.path.join(
            self._base_predictions_file % (prefix)
        )
        accuracies_file_path = os.path.join(
            self._base_accuracies_file % (prefix)
        )

        predictions_file = open(predictions_file_path, "w+")
        predictions_file.write(json.dumps(models_predictions))

        accuracies_file = open(accuracies_file_path, "w+")
        accuracies_file.write(json.dumps(models_accuracies))

        predictions_file.close()
        accuracies_file.close()
