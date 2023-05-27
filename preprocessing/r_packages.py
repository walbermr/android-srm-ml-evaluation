import rpy2
import rpy2.robjects.numpy2ri

from rpy2.robjects.vectors import StrVector
import rpy2.robjects.packages as rpackages


def install():
    rutils = rpackages.importr("utils")
    rutils.chooseCRANmirror(ind=1)
    packnames = ("logisticPCA",)
    names_to_install = [x for x in packnames if not rpackages.isinstalled(x)]
    if len(names_to_install) > 0:
        print(names_to_install)
        rutils.install_packages(StrVector(names_to_install))
    from rpy2.robjects.packages import importr

    rpy2.robjects.numpy2ri.activate()



if __name__ == "__main__":
    install()