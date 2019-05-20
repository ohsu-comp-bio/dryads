
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

import random
from abc import abstractmethod


class CohortError(Exception):
    pass


class Cohort(object):
    """Abstract class for -omic datasets used in machine learning.

    This class consists of a dataset of -omic measurements collected for a
    collection of samples over a set of genetic features. The samples are
    divided into training and testing sub-cohorts for use in the evaluation of
    machine learning models. The specific nature of these -omic measurements
    and the phenotypes the models will be used to predict are defined by
    children classes.

    Args:
        omic_data : An -omic dataset or collection thereof.
        cv_seed (int): A seed used for random sampling from the datasets.
        test_prop: The proportion of samples in each dataset used for testing.

    """

    def __init__(self, omic_data, cv_seed, test_prop):
        self._omic_data = omic_data
        self._cv_seed = cv_seed
        self._train_samps, self._test_samps = self._split_samples(test_prop)

    @abstractmethod
    def get_samples(self, include_samps=None, exclude_samps=None):
        """Retrieves the samples for which -omic data is available."""

    @abstractmethod
    def _split_samples(self, test_prop):
        """Splits a list of samples into training and testing sub-cohorts."""

    def get_seed(self):
        """Retrieves the seed used for random sampling from the cohort."""
        return self._cv_seed

    def update_seed(self, new_seed, test_prop=None):
        """Updates the sampling seed, and optionally the train/test split.

        This method is used to change the random sampling seed used by this
        cohort for learning tasks. The default behaviour is to leave the split
        of the cohort's samples into training and testing sub-cohorts as is,
        but if a proportion is given, a new split will be created according to
        the new sampling seed regardless of whether the new proportion is the
        same as the old one.

        Args:
            new_seed (int): A seed for random sampling from the datasets.
            test_prop, optional
                The new proportion of samples to use for testing.

        """
        self._cv_seed = new_seed

        if test_prop is not None:
            self._train_samps, self._test_samps = self._split_samples(
                test_prop)

    @abstractmethod
    def get_train_samples(self, include_samps=None, exclude_samps=None):
        return self._train_samps

    @abstractmethod
    def get_test_samples(self, include_samps=None, exclude_samps=None):
        return self._test_samps

    @abstractmethod
    def get_features(self, include_feats=None, exclude_feats=None):
        """Retrieves features over which -omic measurements were made."""

    def train_data(self,
                   pheno=None,
                   include_samps=None, exclude_samps=None,
                   include_feats=None, exclude_feats=None):
        """Retrieval of the training cohort from the -omic dataset."""

        samps = self.get_train_samples(include_samps, exclude_samps)
        feats = self.get_features(include_feats, exclude_feats)

        # only get training phenotypic data if phenotype is specified
        if pheno is not None:
            pheno_data, samps = self.parse_pheno(
                self.train_pheno(pheno, samps), samps)

        else:
            pheno_data = None

        return self.get_omic_data(samps, feats), pheno_data

    def test_data(self,
                  pheno=None,
                  include_samps=None, exclude_samps=None,
                  include_feats=None, exclude_feats=None):
        """Retrieval of the testing cohort from the -omic dataset."""

        samps = self.get_test_samples(include_samps, exclude_samps)
        feats = self.get_features(include_feats, exclude_feats)

        # only get testing phenotypic data if phenotype is specified
        if pheno is not None:
            pheno_data, samps = self.parse_pheno(
                self.test_pheno(pheno, samps), samps)

        return self.get_omic_data(samps, feats), pheno_data

    @abstractmethod
    def get_omic_data(self, samps=None, feats=None):
        """Retrieval of a subset of the -omic dataset."""

        return self._omic_data

    @abstractmethod
    def train_pheno(self, pheno, samps=None):
        """Returns the values for a phenotype from the training sub-cohort."""

    @abstractmethod
    def test_pheno(self, pheno, samps=None):
        """Returns the values for a phenotype from the testing sub-cohort."""

    def parse_pheno(self, pheno, samps):
        pheno = np.array(pheno)

        if pheno.ndim == 1:
            pheno_mat = pheno.reshape(-1, 1)

        elif pheno.shape[1] == len(samps):
            pheno_mat = np.transpose(pheno)

        elif pheno.shape[0] != len(samps):
            raise ValueError("Given phenotype(s) do not return a valid "
                             "matrix of values to predict!")

        else:
            pheno_mat = pheno.copy()

        nan_stat = np.any(~np.isnan(pheno_mat), axis=1)
        samps_use = np.array(samps)[nan_stat]

        if pheno_mat.shape[1] == 1:
            pheno_mat = pheno_mat.ravel()

        return pheno_mat[nan_stat], samps_use


class UniCohort(Cohort):
    """Abstract class for predicting phenotypes using a single dataset.

    This class consists of a dataset of -omic measurements collected for a
    collection of samples coming from a single context, such as TCGA-BRCA
    or ICGC PACA-AU.

    Args:
        omic_mat (:obj:`pd.DataFrame`, shape = [n_samps, n_feats])
        cv_seed (int): A seed used for random sampling from the dataset.
        test_prop (float): The proportion of samples in the dataset
                           used for testing.

    """

    def __init__(self, omic_mat, cv_seed, test_prop):
        if not isinstance(omic_mat, pd.DataFrame):
            raise TypeError("`omic_mat` must be a pandas DataFrame, found "
                            "{} instead!".format(type(omic_mat)))

        super().__init__(omic_mat, cv_seed, test_prop)

    def get_samples(self):
        """Retrieves all samples in the -omic dataset.

        Returns:
            samps (list)

        """
        return self._omic_data.index.tolist()

    def _split_samples(self, test_prop):
        """Splits the dataset's samples into training and testing subsets.

        Args:
            test_prop (float): The proportion of samples which will be in the
                               testing subset. Must be non-negative and
                               strictly less than one.

        Returns:
            train_samps (set): The samples for the training sub-cohort.
            test_samps (set): The samples for the testing sub-cohort.

        """
        if test_prop < 0 or test_prop >= 1:
            raise ValueError("Improper testing sample ratio that is not at "
                             "least zero and less than one!")

        # get the full list of samples in the cohort, fix the seed that will
        # be used for random sampling
        samps = self.get_samples()
        if self._cv_seed is not None:
            random.seed(a=self._cv_seed)

        # if not all samples are to be in the training sub-cohort, randomly
        # choose samples for the testing cohort...
        if test_prop > 0:
            train_samps = set(random.sample(population=sorted(tuple(samps)),
                                            k=int(round(len(samps)
                                                        * (1 - test_prop)))))
            test_samps = set(samps) - train_samps

        # ...otherwise, copy the sample list to create the training cohort
        else:
            train_samps = set(samps)
            test_samps = set()

        return train_samps, test_samps

    def get_train_samples(self, include_samps=None, exclude_samps=None):
        """Gets a subset of the samples in the cohort used for training.

        This is a utility function whereby a list of samples to be included
        and/or excluded in a given analysis can be specified. This list is
        checked against the samples actually available in the training
        cohort, and the samples that are both available and match the
        inclusion/exclusion criteria are returned.

        Note that exclusion takes precedence over inclusion, that is, if a
        sample is asked to be both included and excluded it will be excluded.
        Returned samples are sorted to ensure that subsetted datasets with the
        same samples will be identical.

        Args:
            include_samps (:obj:`iterable` of :obj: `str`, optional)
            exclude_samps (:obj:`iterable` of :obj: `str`, optional)

        Returns:
            train_samps (:obj:`list` of :obj:`str`)

        See Also:
            :method:`get_features`
                A similar method but for the genetic features of the dataset.

        """
        train_samps = self._train_samps.copy()

        # decide which training samples to use based on exclusion and
        # inclusion criteria
        if include_samps is not None:
            train_samps &= set(include_samps)
        if exclude_samps is not None:
            train_samps -= set(exclude_samps)

        return sorted(train_samps)

    def get_test_samples(self, include_samps=None, exclude_samps=None):
        """Gets a subset of the samples in the cohort used for testing.

        See :method:`get_train_samples` above for an explanation of how this
        method is used.

        Args:
            include_samps (:obj:`iterable` of :obj: `str`, optional)
            exclude_samps (:obj:`iterable` of :obj: `str`, optional)

        Returns:
            test_samps (:obj:`list` of :obj:`str`)

        """
        test_samps = self._test_samps.copy()

        if include_samps is not None:
            test_samps &= set(include_samps)
        if exclude_samps is not None:
            test_samps -= set(exclude_samps)

        return sorted(test_samps)

    def get_features(self, include_feats=None, exclude_feats=None):
        """Gets a subset of the features over which -omics were measured.

        This is a utility function whereby a list of genetic features to be
        included and/or excluded in a given analysis can be specified. This
        list is checked against the features actually available in the
        dataset, and the genes that are both available and match the
        inclusion/exclusion criteria are returned.

        Note that exclusion takes precedence over inclusion, that is, if a
        feature is asked to be both included and excluded it will be excluded.
        Returned features are sorted to ensure that subsetted datasets
        with the same genes will be identical.

        Args:
            include_feats (:obj:`iterable` of :obj: `str`), optional
            exclude_feats (:obj:`iterable` of :obj: `str`), optional

        Returns:
            feats (:obj:`list` of :obj:`str`)

        See Also:
            :method:`get_train_samples`, :method:`get_test_samples`
                Similar functions but for the samples in the dataset.

        """
        feats = self._omic_data.columns.tolist()

        if isinstance(feats[0], str):
            feats = set(feats)

        elif isinstance(feats[0], tuple):
            feats = set(x[0] for x in feats)

        # decide which genetic features to use based on
        # inclusion/exclusion criteria
        if include_feats is not None:
            feats &= set(include_feats)
        if exclude_feats is not None:
            feats -= set(exclude_feats)

        return sorted(feats)

    def get_omic_data(self, samps=None, feats=None):
        """Retrieves a subset of the -omic dataset."""

        if samps is None:
            samps = self.get_samples()
        if feats is None:
            feats = self.get_features()

        return self._omic_data.loc[samps, feats]

    @abstractmethod
    def train_pheno(self, pheno, samps=None):
        return np.array([])

    @abstractmethod
    def test_pheno(self, pheno, samps=None):
        return np.array([])

    def data_hash(self):
        return tuple(dict(self._omic_data.sum().round(5)).items())


class PresenceCohort(Cohort):
    """Abstract class for -omic datasets predicting binary phenotypes.
    
    This class is used to predict features such as the presence of a
    particular type of variant or copy number alteration, the presence of a
    binarized drug response, etc.

    """

    @abstractmethod
    def train_pheno(self, pheno, samps=None):
        """Returns the binary labels corresponding to the presence of
           a phenotype for each of the samples in the training sub-cohort.

        Returns:
            pheno_vec (:obj:`list` of :obj:`bool`)
        """

    @abstractmethod
    def test_pheno(self, pheno, samps=None):
        """Returns the binary labels corresponding to the presence of
           a phenotype for each of the samples in the testing sub-cohort.

        Returns:
            pheno_vec (:obj:`list` of :obj:`bool`)
        """

    def mutex_test(self, pheno1, pheno2):
        """Tests the mutual exclusivity of two phenotypes.

        Args:
            pheno1, pheno2: A pair of phenotypes stored in this cohort.

        Returns:
            pval (float): The p-value given by a Fisher's one-sided exact test
                          on the pair of phenotypes in the training cohort.

        Examples:
            >>> from dryadic.features.mutations.branches import MuType
            >>>
            >>> self.mutex_test(MuType({('Gene', 'TP53'): None}),
            >>>                 MuType({('Gene', 'CDH1'): None}))
            >>>
            >>> self.mutex_test(MuType({('Gene', 'PIK3CA'): None}),
            >>>                 MuType({('Gene', 'BRAF'): {
            >>>                             ('Location', '600'): None
            >>>                        }}))

        """
        pheno1_vec = self.train_pheno(pheno1)
        pheno2_vec = self.train_pheno(pheno2)
        conting_df = pd.DataFrame({'ph1': pheno1_vec, 'ph2': pheno2_vec})

        return fisher_exact(
            table=pd.crosstab(conting_df['ph1'], conting_df['ph2']),
            alternative='less'
            )[1]


class TransferCohort(Cohort):
    """Abstract class for predicting phenotypes using multiple datasets.

    This class consists of multiple datasets of -omic measurements, each for a
    different set of samples coming from a unique context, which will
    nevertheless be used to predict phenotypes common between the contexts.

    Args:
        omic_mats (:obj:`dict` of :obj:`pd.DataFrame`)
        cv_seed (int): A random seed used for sampling from the datasets.
        test_prop (float or :obj:`dict` of :obj:`float`)
            The proportion of samples in each dataset used for testing.

    """
 
    def __init__(self, omic_mats, cv_seed, test_prop):
        if not isinstance(omic_mats, dict):
            raise TypeError("`omic_mats` must be a dictionary, found {} "
                            "instead!".format(type(omic_mats)))

        for coh, omic_mat in omic_mats.items():
            if not isinstance(omic_mat, pd.DataFrame):
                raise TypeError("`omic_mats` must have pandas DataFrames as "
                                "values, found {} instead for "
                                "cohort {}!".format(type(omic_mat), coh))

        super().__init__(omic_mats, cv_seed, test_prop)

    def get_samples(self):
        """Retrieves all samples from each -omic dataset.

        Returns:
            samps (:obj:`dict` of :obj:`list`)

        """
        return {coh: omic_mat.index.tolist()
                for coh, omic_mat in self._omic_data.items()}

    def _split_samples(self, test_prop):
        """Splits each dataset's samples into training and testing subsets.

        Args:
            test_prop (float or :obj:`dict` of :obj:`float`)
                The proportion of samples in each dataset which will be used
                for the testing subset. Must be non-negative and strictly
                less than one.

        Returns:
            train_samps, test_samps (:obj:`dict` of :obj:`set`)

        """
        samp_dict = self.get_samples()
        train_samps = {coh: set() for coh in samp_dict}
        test_samps = {coh: set() for coh in samp_dict}

        if not isinstance(test_prop, dict):
            test_prop = {coh: test_prop for coh in samp_dict}
        if self._cv_seed is not None:
            random.seed(a=self._cv_seed)

        for coh, samps in samp_dict.items():
            if test_prop[coh] < 0 or test_prop[coh] >= 1:
                raise ValueError("Improper testing sample ratio for cohort "
                                 "{} that is not at least zero and less than "
                                 "one!".format(coh))

            if test_prop[coh] > 0:
                train_samps[coh] = set(
                    random.sample(population=sorted(tuple(samps)),
                                  k=int(round(len(samps)
                                              * (1 - test_prop[coh]))))
                    )
                test_samps[coh] = set(samps) - train_samps[coh]

            else:
                train_samps[coh] = set(samps)

        return train_samps, test_samps

    def get_train_samples(self, include_samps=None, exclude_samps=None):
        train_samps = self._train_samps.copy()

        # decide what samples to use based on exclusion and inclusion criteria
        if include_samps is not None:
            train_samps = {coh: samps & set(include_samps[coh])
                           for coh, samps in train_samps.items()}

        if exclude_samps is not None:
            train_samps = {coh: samps - set(exclude_samps[coh])
                           for coh, samps in train_samps.items()}

        return {coh: sorted(samps) for coh, samps in train_samps.items()}

    def get_test_samples(self, include_samps=None, exclude_samps=None):
        test_samps = self._test_samps.copy()

        # decide what samples to use based on exclusion and inclusion criteria
        if include_samps is not None:
            test_samps = {coh: samps & set(include_samps[coh])
                          for coh, samps in test_samps.items()}

        if exclude_samps is not None:
            test_samps = {coh: samps - set(exclude_samps[coh])
                          for coh, samps in test_samps.items()}

        return {coh: sorted(samps) for coh, samps in test_samps.items()}

    def get_features(self, include_feats=None, exclude_feats=None):
        """Gets a subset of the -omic features from each of the datasets.

        This is a utility function whereby a list of genetic features to be
        included and/or excluded in a given analysis can be specified. This
        list is checked against the features actually measured in each of the
        datasets, and the features that are both available and match the
        inclusion/exclusion criteria are returned.

        Note that exclusion takes precedence over inclusion, that is, if a
        feature is asked to be both included and excluded it will be excluded.
        Returned features are sorted to ensure that lists of subsetted
        datasets with the same features will be identical.

        Also note that the features to be included and excluded can be given
        as lists specific to each dataset in the cohort, or as a single list
        used to subset all the -omic datasets.

        Args:
            include_feats (:obj:`list` or :obj:`iterable` of :obj: `str`,
                           optional)
            exclude_feats (:obj:`list` or :obj:`iterable` of :obj: `str`,
                           optional)

        Returns:
            feat_dict (:obj:`list` of :obj:`str`)

        See Also:
            :method:`get_train_samples`, :method:`get_test_samples`
                Similar functions but for the samples in each dataset.

        """
        feats = dict()

        for coh, omic_mat in self._omic_data.items():
            coh_feats = omic_mat.columns.tolist()

            if isinstance(coh_feats[0], str):
                feats[coh] = frozenset(coh_feats)
            elif isinstance(coh_feats[0], tuple):
                feats[coh] = frozenset(coh_feat[0] for coh_feat in coh_feats)

        # adds features to the list of features to retrieve
        if include_feats is not None:
            if isinstance(list(include_feats)[0], str):
                feats = {coh: fts & set(include_feats)
                         for coh, fts in feats.items()}

            else:
                feats = {coh: fts & set(in_fts)
                         for (coh, fts), in_fts in zip(feats.items(),
                                                       include_feats)}

        # removes features from the list of features to retrieve
        if exclude_feats is not None:
            if isinstance(list(exclude_feats)[0], str):
                feats = {coh: fts - set(exclude_feats)
                         for coh, fts in feats.items()}

            else:
                feats = {coh: fts - set(ex_fts)
                         for (coh, fts), ex_fts in zip(feats.items(),
                                                       exclude_feats)}

        return {coh: sorted(fts) for coh, fts in feats.items()}

    def get_omic_data(self, samps=None, feats=None):
        """Retrieves a subset of each -omic dataset."""

        if samps is None:
            samps = self.get_samples()
        if feats is None:
            feats = self.get_features()

        return {coh: omic_mat.loc[samps[coh], feats[coh]]
                for coh, omic_mat in self._omic_data.items()}

    @abstractmethod
    def train_pheno(self, pheno, samps=None):
        return {coh: np.array([]) for coh in self._omic_data}

    @abstractmethod
    def test_pheno(self, pheno, samps=None):
        return {coh: np.array([]) for coh in self._omic_data}

    def parse_pheno(self, pheno, samps):

        parse_dict = {
            coh: super(TransferCohort, self).parse_pheno(
                pheno[coh], samps[coh])
            for coh in pheno
            }

        return ({coh: phn for (coh, (phn, _)) in parse_dict.items()},
                {coh: smps for (coh, (_, smps)) in parse_dict.items()})

    @classmethod
    def combine_cohorts(cls, *cohorts, **named_cohorts):
        cohort_dict = dict()
        omic_mats = dict()
        cv_seed = None

        for cohort in cohorts:
            if not hasattr(cohort, 'cohort'):
                raise CohortError("Unnamed cohorts must have a `cohort` "
                                  "attribute so that they can be "
                                  "automatically labelled during combining!")

            if cohort.cohort in cohort_dict:
                raise CohortError("Cannot pass two cohorts with the same "
                                  "`cohort` label, pass these as unique "
                                  "keyword arguments instead!")

            cohort_dict[cohort.cohort] = cohort

        for lbl, cohort in named_cohorts.items():
            if lbl in cohort_dict:
                raise CohortError("Cannot use custom cohort label <{}>, "
                                  "which is the label of another cohort "
                                  "already used!".format(lbl))

            cohort_dict[lbl] = cohort

        for lbl, cohort in cohort_dict.items():
            omic_mats[lbl] = cohort.omic_data
            test_prop[lbl] = len(cohort.get_test_samples())
            test_prop[lbl] /= len(cohort.get_samples())

            if cohort.cv_seed is not None:
                if cv_seed is None or cv_seed > cohort.cv_seed:
                    cv_seed = cohort.cv_seed

        return cls(omic_mats, cv_seed, test_prop)

