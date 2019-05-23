
import os
base_dir = os.path.dirname(__file__)

import sys
sys.path.extend([os.path.join(base_dir, '../..')])

from dryadic.features.cohorts import BaseMutationCohort
from dryadic.features.mutations import MuType
from dryadic.learning.pipelines import PresencePipe

from dryadic.learning.stan.base import StanOptimizing
from dryadic.learning.stan.logistic import *
from dryadic.learning.stan.margins import GaussLabels
from dryadic.learning.stan.margins import gauss_model as overlap_model

import pandas as pd
from sklearn.preprocessing import StandardScaler


class OptimLogistic(BaseLogistic, StanOptimizing):
 
    def run_model(self, **fit_params):
        super().run_model(**{**fit_params, **{'iter': 1e2}})


class OptimOverlap(GaussLabels, StanOptimizing):
 
    def run_model(self, **fit_params):
        super().run_model(**{**fit_params, **{'iter': 1e3}})


class StanLogistic(PresencePipe):
    
    norm_inst = StandardScaler()
    fit_inst = OptimLogistic(alpha=1./13, model_code=gauss_model)

    def __init__(self):
        super().__init__([('norm', self.norm_inst), ('fit', self.fit_inst)])


class StanOverlap(PresencePipe):
    
    norm_inst = StandardScaler()
    fit_inst = OptimOverlap(alpha=1./23, model_code=overlap_model)

    def __init__(self):
        super().__init__([('norm', self.norm_inst), ('fit', self.fit_inst)])


def main():
    data_dir = os.path.join(base_dir, "resources")

    expr_data = pd.read_csv(os.path.join(data_dir, "expr.txt.gz"),
                            sep='\t', index_col=0)
    mut_data = pd.read_csv(os.path.join(data_dir, "variants.txt.gz"),
                           sep='\t', index_col=0)

    cdata = BaseMutationCohort(expr_data, mut_data, mut_levels=['Gene'],
                               mut_genes=['TP53'], cv_seed=987, test_prop=0.8)
    test_mtype = MuType({('Gene', 'TP53'): None})

    log_clf = StanLogistic()
    log_clf.fit_coh(cdata, test_mtype)

    train_auc = log_clf.eval_coh(cdata, test_mtype, use_train=True)
    print("Stan logistic model training AUC: {:.3f}".format(train_auc))
    assert train_auc >= 0.7, (
        "Stan logistic model did not obtain a training AUC of at least 0.7!"
        )

    test_auc = log_clf.eval_coh(cdata, test_mtype, use_train=False)
    print("Stan logistic model testing AUC: {:.3f}".format(test_auc))
    assert test_auc >= 0.7, (
        "Stan logistic model did not obtain a testing AUC of at least 0.7!"
        )

    ovp_clf = StanOverlap()
    ovp_clf.fit_coh(cdata, test_mtype)

    train_auc = ovp_clf.eval_coh(cdata, test_mtype, use_train=True)
    print("Stan overlap model training AUC: {:.3f}".format(train_auc))
    assert train_auc >= 0.7, (
        "Stan overlap model did not obtain a training AUC of at least 0.7!"
        )

    test_auc = ovp_clf.eval_coh(cdata, test_mtype, use_train=False)
    print("Stan overlap model testing AUC: {:.3f}".format(test_auc))
    assert test_auc >= 0.7, (
        "Stan overlap model did not obtain a testing AUC of at least 0.7!"
        )

    print("All Stan learning tests passed successfully!")


if __name__ == '__main__':
    main()

