
import os
import sys
base_dir = os.path.dirname(__file__)
sys.path.extend([os.path.join(base_dir, '../..')])

from dryadic.features.cohorts import BaseMutationCohort
from dryadic.features.mutations import MuType
from dryadic.learning.classifiers import Lasso, RidgeWhite, SVCrbf
from dryadic.learning.regressors import OmicRidge

import numpy as np
import pandas as pd


def main():
    data_dir = os.path.join(base_dir, "resources")

    expr_data = pd.read_csv(os.path.join(data_dir, "expr.txt.gz"),
                            sep='\t', index_col=0)
    mut_data = pd.read_csv(os.path.join(data_dir, "variants.txt.gz"),
                           sep='\t', index_col=0)

    cdata = BaseMutationCohort(expr_data, mut_data,
                               mut_levels=['Gene'], mut_genes=['GATA3'],
                               cv_seed=101, test_prop=0.3)
    test_mtype = MuType({('Gene', 'GATA3'): None})

    clf = Lasso()
    clf, cvs = clf.tune_coh(cdata, test_mtype,
                            test_count=16, tune_splits=4, parallel_jobs=1)

    best_indx = np.argmax(cvs['mean_test_score'] - cvs['std_test_score'])
    for param, _ in clf.tune_priors:
        assert clf.get_params()[param] == cvs['params'][best_indx][param]

    clf.fit_coh(cdata, test_mtype)
    tuned_coefs = np.floor(expr_data.shape[1]
                           * (clf.named_steps['feat'].mean_perc / 100))
    assert tuned_coefs == len(clf.named_steps['fit'].coef_[0]), (
        "Tuned feature selection step does not match number of features that "
        "were fit over!"
        )

    assert len(clf.get_coef()) <= len(clf.expr_genes), (
        "Pipeline produced more gene coefficients than genes "
        "it was originally given!"
        )

    train_auc = clf.eval_coh(cdata, test_mtype, use_train=True)
    print("Lasso model training AUC: {:.3f}".format(train_auc))
    assert train_auc >= 0.6, (
        "Lasso model did not obtain a training AUC of at least 0.6!"
        )

    test_auc = clf.eval_coh(cdata, test_mtype, use_train=False)
    print("Lasso model testing AUC: {:.3f}".format(test_auc))
    assert test_auc >= 0.6, (
        "Lasso model did not obtain a testing AUC of at least 0.6!"
        )

    infer_mat = clf.infer_coh(cdata, test_mtype,
                              infer_splits=8, infer_folds=4, parallel_jobs=1)

    assert len(infer_mat) == 143, (
        "Pipeline inference did not produce scores for each sample!"
        )

    clf = RidgeWhite()
    clf.tune_coh(cdata, test_mtype,
                 test_count=4, tune_splits=2, parallel_jobs=1)
    clf.fit_coh(cdata, test_mtype)

    clf = SVCrbf()
    clf.tune_coh(cdata, test_mtype,
                 test_count=4, tune_splits=2, parallel_jobs=1)
    clf.fit_coh(cdata, test_mtype)

    infer_mat = clf.infer_coh(cdata, test_mtype,
                              infer_splits=4, infer_folds=4, parallel_jobs=1)

    print("All pipeline tests passed successfully!")


if __name__ == '__main__':
    main()

