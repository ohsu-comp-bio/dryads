
from .base import UniCohort, PresenceCohort, TransferCohort
from ..mutations import *
import numpy as np


class BaseMutationCohort(PresenceCohort, UniCohort):

    def __init__(self,
                 expr, variants,
                 mut_genes=None, mut_levels=('Gene', 'Form'), top_genes=100,
                 samp_cutoff=None, cv_prop=2.0/3, cv_seed=None):

        if mut_genes is None:
            self.path = None

            var_df = variants.loc[
                (variants['Scale'] == 'Point')
                | ((variants['Scale'] == 'Copy')
                   & variants['Copy'].isin(['HomDel', 'HomGain'])),
                :]

            # find how many unique samples each gene is mutated in, filter for
            # genes that appear in the annotation data
            gn_counts = var_df.groupby(by='Gene').Sample.nunique()
            gn_counts = gn_counts.loc[gn_counts.index.isin(expr.columns)]

            if samp_cutoff is None:
                gn_counts = gn_counts.sort_values(ascending=False)
                cutoff_mask = ([True] * min(top_genes, len(gn_counts))
                               + [False] * max(len(gn_counts) - top_genes, 0))

            elif isinstance(samp_cutoff, int):
                cutoff_mask = gn_counts >= samp_cutoff

            elif isinstance(samp_cutoff, float):
                cutoff_mask = gn_counts >= samp_cutoff * expr.shape[0]

            elif hasattr(samp_cutoff, '__getitem__'):
                if isinstance(samp_cutoff[0], int):
                    cutoff_mask = ((samp_cutoff[0] <= gn_counts)
                                   & (samp_cutoff[1] >= gn_counts))

                elif isinstance(samp_cutoff[0], float):
                    cutoff_mask = (
                            (samp_cutoff[0] * expr.shape[0] <= gn_counts)
                            & (samp_cutoff[1] * expr.shape[0] >= gn_counts)
                        )

            else:
                raise ValueError("Unrecognized `samp_cutoff` argument!")

            gn_counts = gn_counts[cutoff_mask]
            variants = variants.loc[variants['Gene'].isin(gn_counts.index), :]

        else:
            variants = variants.loc[variants['Gene'].isin(mut_genes), :]

        # gets subset of samples to use for training, and split the expression
        # and variant datasets accordingly into training/testing cohorts
        train_samps, test_samps = self.split_samples(
            cv_seed, cv_prop, expr.index)

        # if the cohort is to have a testing cohort, build the tree with info
        # on which testing samples have which types of mutations
        if test_samps:
            self.test_mut = MuTree(
                muts=variants.loc[variants['Sample'].isin(test_samps), :],
                levels=mut_levels
                )

        else:
            test_samps = None

        # likewise, build a representation of mutation types across
        # training cohort samples
        self.train_mut = MuTree(
            muts=variants.loc[variants['Sample'].isin(train_samps), :],
            levels=mut_levels
            )

        self.mut_genes = mut_genes
        self.cv_prop = cv_prop
        super().__init__(expr, train_samps, test_samps, cv_seed)

    def cna_pheno(self, cna_dict, samps):
        if self.copy_data is None:
            raise CohortError("Cannot retrieve copy number alteration "
                              "phenotypes from a cohort not loaded with "
                              "continuous CNA scores!")

        copy_use = self.copy_data.loc[samps, cna_dict['Gene']]
        
        if cna_dict['CNA'] == 'Gain':
            stat_list = copy_use > cna_dict['Cutoff']

        elif cna_dict['CNA'] == 'Loss':
            stat_list = copy_use < cna_dict['Cutoff']

        elif cna_dict['CNA'] == 'Range':
            stat_list = copy_use.between(*cna_dict['Cutoff'], inclusive=False)
        
        else:
            raise ValueError(
                'A dictionary representing a CNA phenotype must have "Gain", '
                '"Loss", or "Range" as its `CNA` entry!'
                )

        return stat_list

    def train_pheno(self, pheno, samps=None):
        """Gets the mutation status of samples in the training cohort.

        Args:
            mtype (:obj:`MuType` or :obj:`list` of :obj:`MuType`)
                A particular type of mutation or list of types.
            samps (:obj:`list` of :obj:`str`, optional)
                A list of samples, of which those not in the training cohort
                will be ignored. Defaults to using all the training samples.

        Returns:
            stat_list (:obj:`list` of :obj:`bool`)

        """

        # use all the training samples if no list of samples is provided
        if samps is None:
            samps = sorted(self.train_samps)

        # otherwise, filter out the provided samples
        # not in the training cohort
        else:
            samps = sorted(set(samps) & self.train_samps)
 
        if isinstance(pheno, MuType) or isinstance(pheno, MutComb):
            stat_list = self.train_mut.status(samps, pheno)

        elif isinstance(pheno, dict):
            stat_list = self.cna_pheno(pheno, samps)

        elif isinstance(tuple(pheno)[0], MuType):
            stat_list = [self.train_mut.status(samps, phn) for phn in pheno]

        elif isinstance(tuple(pheno)[0], dict):
            stat_list = [self.cna_pheno(phn, samps) for phn in pheno]

        else:
            raise TypeError(
                "A VariantCohort accepts only MuTypes, CNA dictionaries, or "
                "lists thereof as training phenotypes!"
                )

        return stat_list

    def test_pheno(self, pheno, samps=None):
        """Gets the mutation status of samples in the testing cohort.

        Args:
            mtype (:obj:`MuType` or :obj:`list` of :obj:`MuType`)
                A particular type of mutation or list of types.
            samps (:obj:`list` of :obj:`str`, optional)
                A list of samples, of which those not in the testing cohort
                will be ignored. Defaults to using all the testing samples.

        Returns:
            stat_list (:obj:`list` of :obj:`bool`)

        """

        # use all the testing samples if no list of samples is provided
        if samps is None:
            samps = sorted(self.test_samps)

        # otherwise, filter out the provided samples not in the testing cohort
        else:
            samps = sorted(set(samps) & self.test_samps)
        
        if isinstance(pheno, MuType) or isinstance(pheno, MutComb):
            stat_list = self.test_mut.status(samps, pheno)

        elif isinstance(pheno, dict):
            stat_list = self.cna_pheno(pheno, samps)

        elif isinstance(tuple(pheno)[0], MuType):
            stat_list = [self.test_mut.status(samps, phn) for phn in pheno]

        elif isinstance(tuple(pheno)[0], dict):
            stat_list = [self.cna_pheno(phn, samps) for phn in pheno]

        else:
            raise TypeError(
                "A VariantCohort accepts only MuTypes, CNA dictionaries, or "
                "lists thereof as testing phenotypes!"
                )

        return stat_list


class BaseTransferMutationCohort(PresenceCohort, TransferCohort):
    """Mutiple datasets used to predict mutations using transfer learning.

    Args:
        expr_dict (:obj:`dict` of :obj:`pd.DataFrame`)
        variant_dict (:obj:`dict` of :obj:`pd.DataFrame`)

    """

    def __init__(self,
                 expr_dict, variant_dict,
                 mut_genes=None, mut_levels=('Gene', 'Form'), top_genes=100,
                 samp_cutoff=None, cv_prop=2.0/3, cv_seed=None):

        if mut_genes is None:
            self.path = None

            var_df = {coh: var.loc[(var['Scale'] == 'Point')
                                   | ((var['Scale'] == 'Copy')
                                      & var['Copy'].isin(
                                          ['HomDel', 'HomGain'])), :]
                      for coh, var in variant_dict.items()}

            gn_counts = {coh: var.groupby(by='Gene').Sample.nunique()
                         for coh, var in var_df.items()}

            if samp_cutoff is None:
                use_counts = {
                    coh: gn_cnts.sort_values(ascending=False)[:top_genes]
                    for coh, gn_cnts in gn_counts.items()
                    }

            elif isinstance(samp_cutoff, int):
                use_counts = {coh: gn_cnts[gn_cnts >= samp_cutoff]
                              for coh, gn_cnts in gn_counts.items()}

            elif isinstance(samp_cutoff, float):
                use_counts = {
                    coh: gn_cnts[
                        gn_cnts >= samp_cutoff * len(use_samples[coh])]
                    for coh, gn_cnts in gn_counts.items()
                    }

            elif hasattr(samp_cutoff, '__getitem__'):
                if isinstance(samp_cutoff[0], int):
                    use_counts = {
                        coh: gn_cnts[(samp_cutoff[0] <= gn_cnts)
                                     & (samp_cutoff[1] >= gn_cnts)]
                        for coh, gn_cnts in gn_counts.items()
                        }

                elif isinstance(samp_cutoff[0], float):
                    use_counts = {
                        coh: gn_cnts[(samp_cutoff[0]
                                      * len(use_samples[coh]) <= gn_cnts)
                                     & (samp_cutoff[1]
                                        * len(use_samples[coh]) >= gn_cnts)]
                        for coh, gn_cnts in gn_counts.items()
                        }

            else:
                raise ValueError("Unrecognized `samp_cutoff` argument!")

            use_gns = reduce(and_,
                             [cnts.index for cnts in use_counts.values()])
            variants = {coh: var.loc[var['Gene'].isin(use_gns), :]
                        for coh, var in variants.items()}

        else:
            variants = {coh: var.loc[var['Gene'].isin(mut_genes), :]
                        for coh, var in variant_dict.items()}

        split_samps = {coh: self.split_samples(cv_seed, cv_prop, expr.index)
                       for coh, expr in expr_dict.items()}

        train_samps = {coh: samps[0] for coh, samps in split_samps.items()}
        test_samps = {coh: samps[1] for coh, samps in split_samps.items()}

        self.train_mut = dict()
        self.test_mut = dict()
        for coh in expr_dict:

            if test_samps[coh]:
                self.test_mut[coh] = MuTree(
                    muts=variants[coh].loc[
                         variants[coh]['Sample'].isin(test_samps[coh]), :],
                    levels=mut_levels
                    )

            else:
                test_samps[coh] = None

            self.train_mut[coh] = MuTree(
                muts=variants[coh].loc[
                     variants[coh]['Sample'].isin(train_samps[coh]), :],
                levels=mut_levels
                )

        self.mut_genes = mut_genes
        self.cv_prop = cv_prop
        super().__init__(expr_dict, train_samps, test_samps, cv_seed)

    @classmethod
    def combine_cohorts(cls, *cohorts, **named_cohorts):
        new_cohort = TransferCohort.combine_cohorts(*cohorts, **named_cohorts)
        new_cohort.__class__ = cls

        new_cohort.train_mut = dict()
        new_cohort.test_mut = dict()

        for cohort in cohorts:
            new_cohort.train_mut[cohort.cohort] = cohort.train_mut

            if hasattr(cohort, "test_mut"):
                new_cohort.test_mut[cohort.cohort] = cohort.test_mut

        for lbl, cohort in named_cohorts.items():
            new_cohort.train_mut[lbl] = cohort.train_mut

            if hasattr(cohort, "test_mut"):
                new_cohort.test_mut[lbl] = cohort.test_mut

        return new_cohort

    def train_pheno(self, pheno, samps=None):
        """Gets the mutation status of samples in the training cohort.

        Args:
            pheno (:obj:`MuType` or :obj:`list` of :obj:`MuType`)
                A particular type of mutation or list of types.
            samps (:obj:`list` of :obj:`str`, optional)
                A list of samples, of which those not in the training cohort
                will be ignored. Defaults to using all the training samples.

        Returns:
            stat_list (:obj:`list` of :obj:`bool`)

        """

        # use all the training samples if no list of samples is provided
        if samps is None:
            samps = {coh: sorted(smps)
                     for coh, smps in self.train_samps.items()}

        # otherwise, filter out the provided samples
        # not in the training cohort
        else:
            samps = {coh: sorted(set(samps[coh]) & self.train_samps[coh])
                     for coh in self.train_samps}
 
        if isinstance(pheno, MuType) or isinstance(pheno, MutComb):
            stat_list = {coh: self.train_mut[coh].status(samps[coh], pheno)
                         for coh in self.train_samps}

        elif isinstance(tuple(pheno)[0], MuType):
            stat_list = {coh: [self.train_mut[coh].status(samps[coh], phn)
                               for phn in pheno]
                         for coh in self.train_samps}

        else:
            raise TypeError(
                "A MutationCohort accepts only MuTypes, CNA dictionaries, or "
                "lists thereof as training phenotypes!"
                )

        return stat_list

    def test_pheno(self, pheno, samps=None):
        """Gets the mutation status of samples in the testing cohort.

        Args:
            pheno (:obj:`MuType` or :obj:`list` of :obj:`MuType`)
                A particular type of mutation or list of types.
            samps (:obj:`list` of :obj:`str`, optional)
                A list of samples, of which those not in the testing cohort
                will be ignored. Defaults to using all the testing samples.

        Returns:
            stat_list (:obj:`list` of :obj:`bool`)

        """

        # use all the testing samples if no list of samples is provided
        if samps is None:
            samps = {coh: sorted(smps)
                     for coh, smps in self.test_samps.items()}

        # otherwise, filter out the provided samples
        # not in the testing cohort
        else:
            samps = {coh: sorted(set(samps[coh]) & self.test_samps[coh])
                     for coh in self.test_samps}
 
        if isinstance(pheno, MuType) or isinstance(pheno, MutComb):
            stat_list = {coh: self.test_mut[coh].status(samps[coh], pheno)
                         for coh in self.test_samps}

        elif isinstance(tuple(pheno)[0], MuType):
            stat_list = {coh: [self.test_mut[coh].status(samps[coh], phn)
                               for phn in pheno]
                         for coh in self.test_samps}

        else:
            raise TypeError(
                "A MutationCohort accepts only MuTypes, CNA dictionaries, or "
                "lists thereof as testing phenotypes!"
                )

        return stat_list

