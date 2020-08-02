
from .base import (UniCohort, PresenceCohort, ValueCohort,
                   TransferCohort, CohortError)
from ..mutations import *
import pandas as pd


class BaseMutationCohort(PresenceCohort, UniCohort):
    """Base class for -omic datasets predicting binary genomic phenotypes.

    Args:
        expr_mat (pd.DataFrame, shape = [n_samps, n_features])
            -Omic dataset that will be used as input features for prediction.
        var_df (pd.DataFrame, shape = [n_muts, n_fields])
            A list of mutations present in the samples, with various fields
            corresponding to mutation attributes.

        mut_levels (iterable of list-like), optional
            Which combinations of mutation attributes to use when creating
            hierarchical representations of mutation data. Default is to
            initialize with one tree that only sorts mutations by gene.

        cv_seed (int), optional: Seed used for random sampling.
        test_prop (float), optional: Proportion of cohort's samples that will
                                     be used for testing. Default is to not
                                     have a testing sub-cohort.

    Attributes:
        mtrees (:obj:`dict` of :obj:`MuTree`)
            Hierarchical representations of the mutations present in the
            dataset, ordered according to combinations of mutation attributes.

    """

    def __init__(self,
                 expr_mat, var_df, mut_levels, copy_df=None, gene_annot=None,
                 leaf_annot=('PolyPhen', ), cv_seed=None, test_prop=0):

        self.muts = var_df
        self.gene_annot = gene_annot
        self.leaf_annot = leaf_annot
        self.mtrees = dict()

        # reconciles attribute levels specific to copy number mutations with
        # those associated with point mutations
        if copy_df is not None:
            self.muts = pd.concat([self.muts, copy_df], sort=True)

            for i in range(len(mut_levels)):
                if 'Scale' not in mut_levels[i]:
                    if 'Gene' in mut_levels[i]:
                        scale_lvl = mut_levels[i].index('Gene') + 1
                    else:
                        scale_lvl = 0

                    mut_levels[i] = (tuple(mut_levels[i][:scale_lvl])
                                     + ('Scale', 'Copy')
                                     + tuple(mut_levels[i][scale_lvl:]))

        # initialize mutation tree(s) according to specified mutation
        # attribute combinations
        for lvls in mut_levels:
            self.add_mut_lvls(lvls)

        super().__init__(expr_mat, cv_seed, test_prop)

    def add_mut_lvls(self, lvls):
        """Adds a hierarchical representation of mutations.

        This method adds (or replaces an existing) tree of mutations based
        on a given combination of mutation attributes.

        Args:
            lvls (list-like of :obj:`str`)


        """
        self.mtrees[tuple(lvls)] = MuTree(self.muts, levels=lvls,
                                          leaf_annot=self.leaf_annot)

    def find_pheno(self, mut):
        """Finds the tree that matches a given mutation object."""
        mut_lvls = mut.get_sorted_levels()

        if not mut_lvls:
            mtree_lvls = tuple(self.mtrees)[0]
        elif mut_lvls in self.mtrees:
            mtree_lvls = mut_lvls

        else:
            for lvls, mtree in self.mtrees.items():
                if mtree.match_levels(mut):
                    mtree_lvls = lvls
                    break

            else:
                self.add_mut_lvls(mut_lvls)
                mtree_lvls = mut_lvls

        return mtree_lvls

    def get_pheno(self, mut, samps=None):
        if not isinstance(mut, (MuType, MutComb)):
            raise TypeError("Unrecognized class of phenotype `{}`!".format(
                type(mut)))

        pheno_samps = mut.get_samples(self.mtrees)

        if samps is None:
            samps = sorted(self.get_samples())

        elif set(samps) - set(self.get_samples()):
            raise ValueError("Cannot retrieve phenotypes for samples "
                             "not in this cohort!")

        return [s in pheno_samps for s in samps]

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
        if samps is None:
            samps = self.get_train_samples()
        else:
            samps = sorted(set(samps) & set(self.get_train_samples()))

        return self.get_pheno(pheno, samps)

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
        if samps is None:
            samps = self.get_test_samples()
        else:
            samps = sorted(set(samps) & set(self.get_test_samples()))

        return self.get_pheno(pheno, samps)

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

    def get_cis_genes(self, cis_lbl, mtype=None, cur_genes=None):
        """Identifies the genes located in the proximity of a given mutation.

        This is a utility method that applies the genomic annotation data
        loaded with the cohort to find genes located on the same chromosome
        as a given mutation, usually for the purpose of removing the
        expression features of such genes from a machine learning pipeline so
        as to prevent the direct cis-effects of a mutation from influencing
        the pipeline output.

        Args:
            cis_lbl (str), one of {'None', 'Self', 'Chrm'}
                Return an empty set (None), the gene associated with the
                mutation (Self), or all the genes on the same chromosome.

            mtype (MuType), optional
            cur_genes (list-like of :obj:`str`), optional
                Instead of specifying a mutation, a set of genes can be given.
                This function will then list the genes proximal to this set.
                If both are given this function will ignore this argument.

        Returns:
            ex_genes (:obj:`set` of :obj:`str`)

        """

        if self.gene_annot is None:
            raise CohortError("Cannot use this cohort to retrieve "
                              "cis-affected genes for a mutation type "
                              "without loading an annotation dataset during "
                              "instantiation!")

        if mtype is not None:
            if mtype.cur_level != 'Gene':
                raise ValueError("Cannot retrieve cis-affected genes for "
                                 "a mutation type not representing a "
                                 "subgrouping of a gene or set of genes!")

            cur_genes = mtype.get_labels()

        elif cur_genes is None:
            raise ValueError("One of `mtype` or `cur_genes` has to be "
                             "specified to identify local genes!")

        if cis_lbl == 'None':
            ex_genes = set()
        elif cis_lbl == 'Self':
            ex_genes = set(cur_genes)

        elif cis_lbl == 'Chrm':
            mtype_chrs = {self.gene_annot[gene]['Chr'] for gene in cur_genes}
            ex_genes = {gene for gene, annot in self.gene_annot.items()
                        if annot['Chr'] in mtype_chrs}

        else:
            raise ValueError("Unrecognized value for `cis_lbl`!")

        return ex_genes

    def data_hash(self):
        """Generates a unique tag of the expression and mutation datasets."""
        return super().data_hash(), hash(tuple(sorted(self.mtrees.items())))


class BaseCopyCohort(ValueCohort, UniCohort):
    """Base class for -omic datasets predicting continuous copy number scores.

    Args:
        expr_mat (pd.DataFrame, shape = [n_samps, n_features])
        copy_mat (pd.DataFrame, shape = [n_samps, n_features])
        cv_seed (int), optional: Seed used for random sampling.
        test_prop (float), optional: Proportion of cohort's samples that will
                                     be used for testing. Default is to not
                                     have a testing sub-cohort.

    """

    def __init__(self,
                 expr_mat, copy_mat, copy_genes=None,
                 cv_seed=None, test_prop=0):
        if copy_genes is not None:
            copy_mat = copy_mat[copy_genes]

        self.copy_mat = copy_mat
        self.mut_genes = copy_genes

        super().__init__(expr_mat, cv_seed, test_prop)

    def train_pheno(self, pheno, samps=None):
        """Gets the mutation status of samples in the training cohort.

        Args:
            samps (:obj:`list` of :obj:`str`, optional)
                A list of samples, of which those not in the training cohort
                will be ignored. Defaults to using all the training samples.

        Returns:
            stat_list (:obj:`list` of :obj:`float`)

        """

        # use all the training samples if no list of samples is provided
        if samps is None:
            samps = self.get_train_samples()

        # otherwise, filter out the provided samples
        # not in the training cohort
        else:
            samps = sorted(set(samps) & set(self.get_train_samples()))

        if isinstance(pheno, str):
            stat_list = self.copy_mat.loc[samps, pheno]

        else:
            raise TypeError("A copy phenotype must be a string!")

        return stat_list

    def test_pheno(self, pheno, samps=None):
        """Gets the mutation status of samples in the testing cohort.

        Args:
            samps (:obj:`list` of :obj:`str`, optional)
                A list of samples, of which those not in the testing cohort
                will be ignored. Defaults to using all the testing samples.

        Returns:
            stat_list (:obj:`list` of :obj:`float`)

        """

        # use all the testing samples if no list of samples is provided
        if samps is None:
            samps = self.get_test_samples()

        # otherwise, filter out the provided samples not in the testing cohort
        else:
            samps = sorted(set(samps) & set(self.get_test_samples()))
       
        if isinstance(pheno, str):
            stat_list = self.copy_mat.loc[samps, pheno]

        else:
            raise TypeError("A copy phenotype must be a string!")

        return stat_list


class BaseTransferMutationCohort(PresenceCohort, TransferCohort):
    """Mutiple datasets used to predict mutations using transfer learning.

    Args:
        expr_dict (:obj:`dict` of :obj:`pd.DataFrame`)
        mut_dict (:obj:`dict` of :obj:`pd.DataFrame`)

    Attributes:
        mtree_dict (:obj:`dict` of :obj:`MuTree`)
            A hierarchical representation of the mutations
            present in each of the datasets.

    """

    def __init__(self,
                 expr_dict, mut_dict, mut_levels, mut_genes=None,
                 domain_dir=None, leaf_annot=('PolyPhen', ),
                 cv_seed=None, test_prop=0):

        if mut_genes is None:
            mut_dict = {coh: mut_df.loc[mut_df.Gene.isin(mut_genes)]
                        for coh, mut_df in mut_dict.items()}

        self.mut_genes = mut_genes
        self.muts = mut_dict
        self.domain_dir = domain_dir
        self.leaf_annot = leaf_annot
        self.mtrees_dict = dict()

        if mut_levels is None:
            self.add_mut_lvls(('Gene', ))

        else:
            for lvls in mut_levels:
                self.add_mut_lvls(lvls)

        super().__init__(expr_dict, cv_seed, test_prop)

    def add_mut_lvls(self, lvls):
        for lvl_k in lvls:
            if tuple(lvl_k) not in self.mtrees_dict:
                self.mtrees_dict[tuple(lvl_k)] = dict()

            for coh, muts in self.muts.items():
                self.mtrees_dict[tuple(lvl_k)][coh] = MuTree(
                    muts, levels=lvls,
                    domain_dir=self.domain_dir, leaf_annot=self.leaf_annot
                    )

    def choose_mtree(self, pheno, coh):
        if isinstance(pheno, MuType):
            phn_lvls = pheno.get_sorted_levels()

            if not phn_lvls:
                mtree_lvls = tuple(self.mtrees_dict)[0]
            elif phn_lvls in self.mtrees_dict:
                mtree_lvls = phn_lvls

            else:
                for mut_lvls, mtrees in self.mtrees_dict.items():
                    if mtrees[coh].match_levels(pheno):
                        mtree_lvls = mut_lvls
                        break

                else:
                    self.add_mut_lvls(phn_lvls)
                    mtree_lvls = phn_lvls

        elif isinstance(pheno, MutComb):
            mtree_lvls = self.choose_mtree(list(pheno.mtypes)[0], coh)

        return mtree_lvls

    @classmethod
    def combine_cohorts(cls, *cohorts, **named_cohorts):
        new_cohort = TransferCohort.combine_cohorts(*cohorts, **named_cohorts)
        new_cohort.__class__ = cls

        for cohort in cohorts:
            new_cohort.mtree_dict[cohort.cohort] = cohort.mtree

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
            stat_dict (dict)

        """

        # use all the training samples if no list of samples is provided
        if samps is None:
            samps = self.get_train_samples()

        # otherwise, filter out the provided samples
        # not in the training cohort
        else:
            samps = {coh: sorted(set(samps[coh])
                                 & set(self.get_train_samples()[coh]))
                     for coh in self._omic_data}
 
        if isinstance(pheno, MuType) or isinstance(pheno, MutComb):
            stat_dict = {
                coh: self.mtrees_dict[self.choose_mtree(pheno, coh)][
                    coh].status(samps[coh], pheno)
                for coh in self._omic_data
                }

        elif isinstance(tuple(pheno)[0], MuType):
            stat_dict = {
                coh: [
                    self.mtrees_dict[self.choose_mtree(phn, coh)][
                        coh].status(samps[coh], phn)
                    for phn in pheno
                    ]
                for coh in self._omic_data
                }

        else:
            raise TypeError(
                "A MutationCohort accepts only MuTypes, CNA dictionaries, or "
                "lists thereof as training phenotypes!"
                )

        return stat_dict

    def test_pheno(self, pheno, samps=None):
        """Gets the mutation status of samples in the testing cohort.

        Args:
            pheno (:obj:`MuType` or :obj:`list` of :obj:`MuType`)
                A particular type of mutation or list of types.
            samps (:obj:`list` of :obj:`str`, optional)
                A list of samples, of which those not in the testing cohort
                will be ignored. Defaults to using all the testing samples.

        Returns:
            stat_dict

        """

        # use all the testing samples if no list of samples is provided
        if samps is None:
            samps = self.get_test_samples()

        # otherwise, filter out the provided samples
        # not in the testing cohort
        else:
            samps = {coh: sorted(set(samps[coh])
                                 & set(self.get_test_samples()[coh]))
                     for coh in self._omic_data}
 
        if isinstance(pheno, MuType) or isinstance(pheno, MutComb):
            stat_dict = {
                coh: self.mtrees_dict[coh][
                    self.choose_mtree(pheno, coh)].status(samps[coh], pheno)
                for coh in self._omic_data
                }

        elif isinstance(tuple(pheno)[0], MuType):
            stat_dict = {
                coh: [
                    self.mtrees_dict[coh][
                        self.choose_mtree(phn, coh)].status(samps[coh], phn)
                    for phn in pheno
                    ]
                for coh in self._omic_data
                }

        else:
            raise TypeError(
                "A MutationCohort accepts only MuTypes, CNA dictionaries, or "
                "lists thereof as testing phenotypes!"
                )

        return stat_dict

