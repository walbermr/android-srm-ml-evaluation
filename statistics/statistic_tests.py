import pandas as pd

from scipy.stats import wilcoxon, kruskal
from scikit_posthocs import posthoc_nemenyi


import json, argparse


def print_latex_table_header(f):
    f.write(
        "\\begin{table} \n\\centering \n\t\\renewcommand{\\arraystretch}{1.8} \n\t\\begin{tabular}{ p{8.5cm}p{2.8cm} } \n\t\t\\toprule \n\t\tModels & Wilcoxon \\  \\\\ \n\t\t\\midrule\n\t"
    )


def print_latex_table_entry(f, name, p):

    f.write("\t\t%s & " % (name))
    if p < 0.0001:
        f.write("%.2e " % (p))
    else:
        f.write("%.4f " % (p))

    if p < 0.001:
        f.write("$\\bullet\\bullet\\bullet$ ")
    elif p < 0.01:
        f.write("$\\bullet\\bullet$ ")
    elif p < 0.1:
        f.write("$\\bullet$ ")

    f.write(" \\\\")
    f.write("\n\t")


def print_latex_table_footer(f):
    f.write(
        "\\end{tabular} \n\t\\caption{} \n\t\\label{table:} \n\\end{table}"
    )


def clear_key(k):
    filter_keywords = [
        " Bagging",
        " Boosting",
        " Boost",
        " Decision Tree",
        " Perceptron",
        " Random Forest",
    ]

    for f in filter_keywords:
        k = k.replace(f, "")

    return k


def main(input, output, r_o):

    f = open(input, "r")
    crude_data = f.readlines()
    crude_data = json.loads(crude_data[0])

    filter_keys = [
        # "OLP",
        # "Triplet OLP",
        # "META-DES Random Forest",
        "KNORA-U with Triplet RoC",
        "KNORA-E with Triplet RoC",
    ]  # classifiers to compare

    all_classifiers = {}

    for model_group in crude_data.keys():
        data = crude_data[model_group]

        # clear data keys
        for k in list(data.keys()):
            new_k = clear_key(k)
            if new_k != k:
                data[clear_key(k)] = data[k]
                del data[k]

        for k in list(data.keys()):
            if (k in filter_keys) or (r_o and k == "Oracle"):
                del data[k]

        table_file = open(output + "/statistical-tests.tex", "w+")
        # print_latex_table_header(table_file)

        l = list(data.values())
        print(model_group)
        print(kruskal(*l)[1])
        # result_df = posthoc_nemenyi(l, dist="tukey")

        # modularizar
        result_arr = []
        for column in data.keys():
            all_classifiers[model_group + " " + column] = data[column]
            result_arr.append([])
            for row in data.keys():
                p_value = kruskal(data[column], data[row])[1]
                rounded = round(p_value, 3)
                str_p_value = (
                    "%.3f" % (rounded)
                    if rounded > 0.001
                    else "%.3e" % (p_value)
                )
                result_arr[-1].append(str_p_value)

        result_df = pd.DataFrame(
            result_arr, index=data.keys(), columns=data.keys()
        )
        print(result_df)
        print("\n\n")

    # modularizar
    result_arr = []
    for k1 in all_classifiers.keys():
        result_arr.append([])
        for k2 in all_classifiers.keys():
            p_value = kruskal(all_classifiers[k1], all_classifiers[k2])[1]
            rounded = round(p_value, 3)
            str_p_value = (
                "%.3f" % (rounded) if rounded > 0.001 else "%.3e" % (p_value)
            )
            result_arr[-1].append(str_p_value)
    result_df = pd.DataFrame(
        result_arr,
        index=all_classifiers.keys(),
        columns=all_classifiers.keys(),
    )
    print(result_df)
    with open("stats.csv", "w+") as f:
        result_df.to_csv(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--file",
        action="store",
        dest="input_file",
        help="Input accuracy file.",
    )
    parser.add_argument(
        "-o",
        "--output",
        action="store",
        dest="output_path",
        help="Define output path for files",
    )

    args = parser.parse_args()

    if args.input_file is None:
        input_file = "default_models_accuracies.txt"
    else:
        input_file = args.input_file

    if args.output_path is None:
        output_path = "./"
    else:
        output_path = args.output_path

    main(input_file, output_path, True)
