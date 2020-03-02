
import os
base_dir = os.path.dirname(__file__)

import sys
sys.path.extend([os.path.join(base_dir, '../..')])

from dryadic.features.cohorts import *
from dryadic.features.mutations import MuType
from dryadic.learning.pipelines import PresencePipe, TransferPipe
from dryadic.learning.pipelines.transfer import MultiPipe
from dryadic.learning.selection import SelectMeanVar

from dryadic.learning.kbtl.single_domain import SingleDomain
from dryadic.learning.kbtl.multi_domain import MultiDomain

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class SingleTransfer(MultiPipe, PresencePipe):

    tune_priors = (
        ('fit__margin', (2./3, 24./23)),
        ('fit__sigma_h', (1./11, 1./7)),
        )

    feat_inst = SelectMeanVar(mean_perc=80, var_perc=90)
    norm_inst = StandardScaler()
    fit_inst = SingleDomain(latent_features=3, max_iter=50)

    def __init__(self):
        super().__init__([('feat', self.feat_inst), ('norm', self.norm_inst),
                          ('fit', self.fit_inst)])


class MultiTransfer(TransferPipe, PresencePipe):

    tune_priors = (
        ('fit__margin', (2./3, 24./23)),
        ('fit__sigma_h', (1./11, 1./7)),
        )

    feat_inst = SelectMeanVar(mean_perc=80, var_perc=90)
    norm_inst = StandardScaler()
    fit_inst = MultiDomain(latent_features=3, max_iter=50)

    def __init__(self):
        super().__init__([('feat', self.feat_inst), ('norm', self.norm_inst),
                          ('fit', self.fit_inst)])


class MultiMultiTransfer(MultiPipe, MultiTransfer):
    pass


def main():
    data_dir = os.path.join(base_dir, "resources")

    expr_data = pd.read_csv(os.path.join(data_dir, "expr.txt.gz"),
                            sep='\t', index_col=0)
    mut_data = pd.read_csv(os.path.join(data_dir, "variants.txt.gz"),
                           sep='\t', index_col=0)

    expr_dict = {'C1': expr_data.iloc[::2, :], 'C2': expr_data.iloc[1::2, :]}
    mut_dict = {coh: mut_data.loc[mut_data.Sample.isin(expr.index), :].copy()
                for coh, expr in expr_dict.items()}

    sing_clf = SingleTransfer()
    mult_clf = MultiTransfer()
    multmult_clf = MultiMultiTransfer()

    sing_mtype = MuType({('Gene', 'TP53'): None})
    mult_mtypes = [MuType({('Gene', 'TP53'): None}),
                   MuType({('Gene', 'GATA3'): None})]

    uni_cdata = BaseMutationCohort(expr_data, mut_data, mut_levels=[['Gene']],
                                   mut_genes=['TP53', 'GATA3'],
                                   cv_seed=None, test_prop=0.3)
    uni_cdata.update_split(new_seed=101)

    sing_clf.tune_coh(uni_cdata, mult_mtypes,
                      test_count=4, tune_splits=2, parallel_jobs=1)
    print(sing_clf)
    sing_clf.fit_coh(uni_cdata, mult_mtypes)

    train_auc = sing_clf.eval_coh(uni_cdata, mult_mtypes, use_train=True)
    print("Multi-pheno single-domain "
          "KBTL model training AUC: {:.3f}".format(train_auc))
    assert train_auc >= 0.6, (
        "KBTL model did not obtain a training AUC of at least 0.6!"
        )

    test_auc = sing_clf.eval_coh(uni_cdata, mult_mtypes, use_train=False)
    print("Multi-pheno single-domain "
          "KBTL model testing AUC: {:.3f}".format(test_auc))
    assert test_auc >= 0.6, (
        "KBTL model did not obtain a testing AUC of at least 0.6!"
        )

    trs_cdata = BaseTransferMutationCohort(expr_dict, mut_dict,
                                           mut_levels=[['Gene']],
                                           mut_genes=['TP53', 'GATA3'],
                                           cv_seed=None, test_prop=0.3)
    trs_cdata.update_split(new_seed=101)

    mult_clf.tune_coh(trs_cdata, sing_mtype,
                      test_count=4, tune_splits=2, parallel_jobs=1)
    print(mult_clf)
    mult_clf.fit_coh(trs_cdata, sing_mtype)

    train_auc = mult_clf.eval_coh(trs_cdata, sing_mtype, use_train=True)
    print("Single-pheno multi-domain "
          "KBTL model training AUC: {:.3f}".format(train_auc))
    assert train_auc >= 0.6, (
        "KBTL model did not obtain a training AUC of at least 0.6!"
        )

    test_auc = mult_clf.eval_coh(trs_cdata, sing_mtype, use_train=False)
    print("Single-pheno multi-domain "
          "KBTL model testing AUC: {:.3f}".format(test_auc))
    assert test_auc >= 0.6, (
        "KBTL model did not obtain a testing AUC of at least 0.6!"
        )

    multmult_clf.tune_coh(trs_cdata, mult_mtypes,
                          test_count=4, tune_splits=2, parallel_jobs=1)
    print(multmult_clf)
    multmult_clf.fit_coh(trs_cdata, mult_mtypes)

    train_auc = multmult_clf.eval_coh(trs_cdata, mult_mtypes, use_train=True)
    print("Multi-pheno multi-domain "
          "KBTL model training AUC: {:.3f}".format(train_auc))
    assert train_auc >= 0.6, (
        "KBTL model did not obtain a training AUC of at least 0.6!"
        )

    test_auc = multmult_clf.eval_coh(trs_cdata, mult_mtypes, use_train=False)
    print("Multi-pheno multi-domain "
          "KBTL model testing AUC: {:.3f}".format(test_auc))
    assert test_auc >= 0.6, (
        "KBTL model did not obtain a testing AUC of at least 0.6!"
        )

    print("All transfer learning tests passed successfully!")


if __name__ == '__main__':
    main()

