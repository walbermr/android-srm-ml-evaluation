# colocar nos detalhes da legenda, o resultado para o teste estatístico
# colocar diagrama cd para cada um das classes de modelos

from scipy.stats import wilcoxon


METRICS_PER_MODEL = {}


def print_latex_table_header(f):
    f.write("\\begin{table} \n\\centering \n\t\\renewcommand{\\arraystretch}{1.8} \n")
    f.write(
        "\t\\begin{tabular}{ p{3cm}p{2.8cm}p{2.8cm}p{2.8cm}p{2.8cm} } \n\t\t\\toprule \n"
    )
    f.write(
        "\t\tModel & Precision & Recall & F1 Score & Accuracy \\\\ \n\t\t\\midrule\n\n\t"
    )


def colorize(colorize_oracle=False):
    global METRICS_PER_MODEL
    aux = {}
    colors = ["red", "dark-green", "blue"]
    best_models = []

    # i and i+1 are mean and standard deviation for each model
    for i in range(0, 8, 2):
        sorted_dict = {
            k: v.to_list()
            for k, v in sorted(
                METRICS_PER_MODEL.items(),
                key=lambda item: item[1].to_list()[i],
                reverse=True,
            )
        }
        c = 0
        for j, k in enumerate(sorted_dict.keys()):
            mean, std = sorted_dict[k][i], sorted_dict[k][i + 1]

            if c < 3 and (
                (colorize_oracle and "oracle" in k.lower())
                or ("oracle" not in k.lower())
            ):
                data = "\\textcolor{%s}{%.4f(%.4f)} " % (colors[c], mean, std)
                c += 1
            else:
                data = "%.4f(%.4f) " % (mean, std)

            if k not in aux:
                aux[k] = data
            else:
                aux[k] += data

            if i < 5:
                aux[k] += "& "
            else:
                aux[k] += " \\\\\n\t"

        best_models.append(list(sorted_dict.keys())[:4])

    return {k: aux[k] for k in METRICS_PER_MODEL}, best_models


def print_latex_table_entry(
    f, name, global_metrics, new_mode=False, commit_print=False, colorize_oracle=False
):
    """
    Return a list with the ordered 3 best models. If do not commit the print, returns None.
    The None value must be discarded.
    """
    best_models = None

    if new_mode:
        global METRICS_PER_MODEL
        if not commit_print:
            METRICS_PER_MODEL[name] = global_metrics
        else:
            data, best_models = colorize(colorize_oracle=colorize_oracle)
            for model_name in data:
                f.write("\t\t%s & %s" % (model_name, data[model_name]))
            f.write("\n\t")
    else:
        global_list = global_metrics.to_list()
        f.write("\t\t%s & " % (name))
        for i in range(0, len(global_list), 2):
            mean, std = global_list[i], global_list[i + 1]
            f.write("%.4f(%.4f) " % (mean, std))
            if i < 5:
                f.write("& ")
            else:
                f.write(" \\\\")

        f.write("\n\t")

    return best_models


def get_statistical_report(metrics_models, statistical_test):
    global METRICS_PER_MODEL
    tests = {"wilcoxon": wilcoxon}
    if statistical_test not in tests:
        raise ValueError("%s not implemented." % (statistical_test))

    test_function = tests[statistical_test]
    metrics_list = ["precisions", "recalls", "f1s", "accuracies"]
    string_result = ""

    best_idx = 0
    for i, models in enumerate(metrics_models):
        best_model = models[best_idx]
        if "Oracle" in best_model:
            best_idx += 1
            best_model = models[best_idx]

        if i == 0:
            string_result += "for"
        else:
            string_result += "For"

        string_result += " %s, comparing %s with " % (
            metrics_list[i].replace("accuracies", "accuracy ").capitalize()[:-1],
            best_model,
        )

        for model in models[best_idx + 1 : best_idx + 3]:
            # TODO: colocar o nome das metricas
            best_results = getattr(METRICS_PER_MODEL[best_model], metrics_list[i])
            pair_results = getattr(METRICS_PER_MODEL[model], metrics_list[i])
            ranks, p_value = test_function(best_results, pair_results)

            if p_value < 0.001:
                notation_result = "%.3e" % (p_value)
            else:
                notation_result = "%.3f" % (p_value)

            if model != models[-1]:
                string_result += "%s (%s), " % (model, notation_result)
            else:
                string_result = string_result[:-2]
                string_result += " and %s (%s). " % (model, notation_result)

    return string_result


def print_latex_table_footer(
    f, label, new_mode=False, colorize_oracle=False, statistical_test="wilcoxon"
):
    global METRICS_PER_MODEL

    if new_mode:
        best_models = print_latex_table_entry(
            f,
            "",
            None,
            new_mode=True,
            commit_print=True,
            colorize_oracle=colorize_oracle,
        )
        statistical_analysis = get_statistical_report(best_models, statistical_test)

    f.write(
        """\\end{tabular}
    \\caption{Results for %s, where red indicates the best model, green the second-best model, and blue the third-best. We used %s to compare the results of each metrics statistically. The observed p-values %s}
    \\label{table:}
\\end{table}
        """
        % (label, statistical_test.capitalize(), statistical_analysis[:-1])
    )

    METRICS_PER_MODEL = {}
