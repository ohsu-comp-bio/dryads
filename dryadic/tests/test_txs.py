
import os
import sys
base_dir = os.path.dirname(__file__)
sys.path.extend([os.path.join(base_dir, '../..')])

from dryadic.features.cohorts import BaseMutationCohort
from dryadic.features.mutations import MuType
from dryadic.learning.pipelines import PresencePipe
from dryadic.learning.stan.base import StanOptimizing
from dryadic.learning.stan.transcripts import *

import pandas as pd
from sklearn.preprocessing import StandardScaler


class OptimTranscripts(BaseTranscripts, StanOptimizing):
 
    def run_model(self, **fit_params):
        super().run_model(**{**fit_params, **{'iter': 8e3}})


class StanTranscripts(PresencePipe):

    tune_priors = (
        ('fit__alpha', (1., 2.)),
        ('fit__gamma', (3./7, 37./13)),
        )
 
    norm_inst = StandardScaler()
    fit_inst = OptimTranscripts(model_code=base_model)

    def __init__(self):
        super().__init__([('norm', self.norm_inst), ('fit', self.fit_inst)])


def main():
    data_dir = os.path.join(base_dir, "resources")

    expr_data = pd.read_csv(os.path.join(data_dir, "expr_txs.txt.gz"),
                            sep='\t', index_col=0, header=[0, 1])
    mut_data = pd.read_csv(os.path.join(data_dir, "variants.txt.gz"),
                           sep='\t', index_col=0)

    cdata = BaseMutationCohort(expr_data, mut_data, mut_levels=['Gene'],
                               mut_genes=['TP53'], cv_seed=987, test_prop=0.2)
    test_mtype = MuType({('Gene', 'TP53'): None})

    tx_clf = StanTranscripts()
    tx_clf.tune_coh(cdata, test_mtype,
                    test_count=4, tune_splits=1, parallel_jobs=1)

    print(tx_clf)
    tx_clf.fit_coh(cdata, test_mtype)

    train_auc = tx_clf.eval_coh(cdata, test_mtype, use_train=True)
    print("Stan transcript model training AUC: {:.3f}".format(train_auc))
    assert train_auc >= 0.7, (
        "Stan transcript model did not obtain a training AUC of at least 0.7!"
        )

    test_auc = tx_clf.eval_coh(cdata, test_mtype, use_train=False)
    print("Stan transcript model testing AUC: {:.3f}".format(test_auc))
    assert test_auc >= 0.7, (
        "Stan transcript model did not obtain a testing AUC of at least 0.7!"
        )

    print("All Stan transcript tests passed successfully!")


if __name__ == '__main__':
    main()

