
import os
base_dir = os.path.dirname(__file__)

import sys
sys.path.extend([os.path.join(base_dir, '../..')])

from dryadic.features.cohorts import BaseTransferMutationCohort
from dryadic.features.mutations import MuType
from dryadic.learning.pipelines import TransferPipe, PresencePipe
from dryadic.learning.selection import SelectMeanVar
from dryadic.learning.kbtl.multi_domain import MultiDomain

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class KBTL_test(TransferPipe, PresencePipe):

    tune_priors = (
        ('fit__margin', (0.5, 0.75, 1.0, 1.5)),
        )

    feat_inst = SelectMeanVar(mean_perc=90, var_perc=90)
    norm_inst = StandardScaler()
    fit_inst = MultiDomain(latent_features=2, max_iter=40)

    def __init__(self):
        super().__init__([('feat', self.feat_inst), ('norm', self.norm_inst),
                          ('fit', self.fit_inst)])


def main():
    data_dir = os.path.join(base_dir, "resources")

    expr_data = pd.read_csv(os.path.join(data_dir, "expr.txt.gz"),
                            sep='\t', index_col=0)
    mut_data = pd.read_csv(os.path.join(data_dir, "variants.txt.gz"),
                           sep='\t', index_col=0)

    expr_dict = {'C1': expr_data.iloc[::2, :], 'C2': expr_data.iloc[1::2, :]}
    mut_dict = {coh: mut_data.loc[mut_data.Sample.isin(expr.index), :].copy()
                for coh, expr in expr_dict.items()}

    cdata = BaseTransferMutationCohort(
        expr_dict=expr_dict, variant_dict=mut_dict,
        mut_genes=['TP53'], cv_prop=0.7, cv_seed=101
        )
    test_mtype = MuType({('Gene', 'TP53'): None})

    clf = KBTL_test()
    clf.tune_coh(cdata, test_mtype,
                 test_count=4, tune_splits=2, parallel_jobs=1)
    clf.fit_coh(cdata, test_mtype)

    train_auc = clf.eval_coh(cdata, test_mtype, use_train=True)
    print("KBTL model training AUC: {:.3f}".format(train_auc))
    assert train_auc >= 0.6, (
        "KBTL model did not obtain a training AUC of at least 0.6!"
        )

    test_auc = clf.eval_coh(cdata, test_mtype, use_train=False)
    print("KBTL model testing AUC: {:.3f}".format(test_auc))
    assert test_auc >= 0.6, (
        "KBTL model did not obtain a testing AUC of at least 0.6!"
        )

    print("All transfer learning tests passed successfully!")


if __name__ == '__main__':
    main()

