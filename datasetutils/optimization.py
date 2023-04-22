from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import matplotlib


def pca_search(dataset, graph=True, variance_lim=0.8):
    # variance per component, PCA: >=90%: 63, >=80%: 39
    variances = []
    colors = []
    i = 0
    while True:
        pca = PCA(i)
        pca.fit(dataset.dataframe["original"][dataset.feature_keys])
        variances.append(sum(pca.explained_variance_ratio_))

        if variances[i] >= 0.9:
            colors.append("r")
            print("variance(%.4f) - iteration %d" % (variances[i], i))
            break

        if variances[i] >= 0.8:
            colors.append("g")
            print("variance(%.4f) - iteration %d" % (variances[i], i))
        else:
            colors.append("b")

        i += 1

    fig, ax = plt.subplots(1, 1)
    ax.set_xlabel("Components")
    ax.set_ylabel("Total Variance")
    ax.set_xlim(0, len(variances))
    barlist = ax.bar(range(len(variances)), variances)

    for i in range(0, len(variances)):
        barlist[i].set_color(colors[i])

    if graph:
        plt.show(fig)

    return [x for x in variances if x >= variance_lim]
