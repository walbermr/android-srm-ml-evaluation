import csv


def print_metrics(classes, globals):
    print("\t\tmean precision\t\tmean recall\t\tmean f1 score\t\tmean accuracy")
    for c in classes:
        r = c.report()
        print("%s\t" % (c.class_label), end="")
        if c.class_label != "neithernor":
            print("\t", end="")
        for l1 in ["precision", "recall", "f1", "accuracy"]:
            for l2 in ["mean", "std"]:
                if l2 == "std":
                    print("(+/-%0.4f)" % (r[l1][l2]), end="")
                else:
                    print("%0.4f" % (r[l1][l2]), end="")
            print("\t", end="")
        print("\n", end="")

    print("\n\t\t", end="")
    global_list = globals.to_list()
    for i in range(0, len(global_list), 2):
        mean, std = global_list[i], global_list[i + 1]
        print("%.4f(+/-%.4f) " % (mean, std), end="\t")
    print("\n")


def save_metrics(name, classes):
    with open(("reports/metrics/%s.csv" % (name)), "w") as f:
        wr = csv.writer(f)
        for metric_list in [c.to_list() for c in classes]:
            wr.writerow(metric_list)
