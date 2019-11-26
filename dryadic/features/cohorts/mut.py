
from .base import UniCohort, PresenceCohort, ValueCohort, TransferCohort
from ..mutations import *


class BaseMutationCohort(PresenceCohort, UniCohort):
    """Base class for -omic datasets predicting binary genomic phenotypes.

    Args:
        expr_mat (pandas.DataFrame, shape = [n_samps, n_features])
        mut_df (pandas.DataFrame, shape = [n_muts, n_fields])
        cv_seed (int), optional: Seed used for random sampling.
        test_prop (float), optional: Proportion of cohort's samples that will
                                     be used for testing. Default is to not
                                     have a testing sub-cohort.

    Attributes:
        mtree (MuTree): A hierarchical representation of the mutations present
                        in the dataset.

    """

    def __init__(self,
                 expr_mat, mut_df, mut_levels=None, mut_genes=None,
                 domain_dir=None, leaf_annot=('PolyPhen', ),
                 cv_seed=None, test_prop=0):
        if mut_genes is not None:
            mut_df = mut_df.loc[mut_df.Gene.isin(mut_genes)]

        self.mut_genes = mut_genes
        self.muts = mut_df
        self.domain_dir = domain_dir
        self.leaf_annot = leaf_annot
        self.mtrees = dict()

        if mut_levels is None:
            self.add_mut_lvls(('Gene', ))

        else:
            for lvls in mut_levels:
                self.add_mut_lvls(lvls)

        super().__init__(expr_mat, cv_seed, test_prop)

    def add_mut_lvls(self, lvls):
        self.mtrees[tuple(lvls)] = MuTree(self.muts, levels=lvls,
                                          domain_dir=self.domain_dir,
                                          leaf_annot=self.leaf_annot)

    def choose_mtree(self, pheno):
        if isinstance(pheno, MuType):
            phn_lvls = pheno.get_sorted_levels()

            if not phn_lvls:
                mtree_lvls = tuple(self.mtrees)[0]
            elif phn_lvls in self.mtrees:
                mtree_lvls = phn_lvls

            else:
                for mut_lvls, mtree in self.mtrees.items():
                    if mtree.match_levels(pheno):
                        mtree_lvls = mut_lvls
                        break

                else:
                    self.add_mut_lvls(phn_lvls)
                    mtree_lvls = phn_lvls

        elif isinstance(pheno, MutComb):
            mtree_lvls = self.choose_mtree(list(pheno.mtypes)[0])

        return mtree_lvls

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
            samps = self.get_train_samples()

        # otherwise, filter out the provided samples
        # not in the training cohort
        else:
            samps = sorted(set(samps) & set(self.get_train_samples()))
 
        if isinstance(pheno, MuType) or isinstance(pheno, MutComb):
            mtree_lvls = self.choose_mtree(pheno)
            stat_list = self.mtrees[mtree_lvls].status(samps, pheno)

        elif isinstance(pheno, dict):
            stat_list = self.cna_pheno(pheno, samps)

        elif isinstance(tuple(pheno)[0], MuType):
            stat_list = [
                self.mtrees[self.choose_mtree(phn)].status(samps, phn)
                for phn in pheno
                ]

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
            samps = self.get_test_samples()

        # otherwise, filter out the provided samples not in the testing cohort
        else:
            samps = sorted(set(samps) & set(self.get_test_samples()))
        
        if isinstance(pheno, MuType) or isinstance(pheno, MutComb):
            mtree_lvls = self.choose_mtree(pheno)
            stat_list = self.mtrees[mtree_lvls].status(samps, pheno)

        elif isinstance(pheno, dict):
            stat_list = self.cna_pheno(pheno, samps)

        elif isinstance(tuple(pheno)[0], MuType):
            stat_list = [
                self.mtrees[self.choose_mtree(phn)].status(samps, phn)
                for phn in pheno
                ]

        elif isinstance(tuple(pheno)[0], dict):
            stat_list = [self.cna_pheno(phn, samps) for phn in pheno]

        else:
            raise TypeError(
                "A VariantCohort accepts only MuTypes, CNA dictionaries, or "
                "lists thereof as testing phenotypes!"
                )

        return stat_list

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

    def data_hash(self):
        return super().data_hash(), hash(tuple(sorted(self.mtrees.items())))


class BaseCopyCohort(ValueCohort, UniCohort):
    """Base class for -omic datasets predicting continuous copy number scores.

    Args:
        expr_mat (pandas.DataFrame, shape = [n_samps, n_features])
        copy_mat (pandas.DataFrame, shape = [n_samps, n_features])
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

