
from .branches import MuType

import numpy as np
import pandas as pd
import os

from functools import reduce
from itertools import combinations as combn
from itertools import product
from operator import or_
from re import sub as gsub

from math import log10, floor
from sklearn.cluster import MeanShift


class MuTree(object):
    """A hierarchy of samples organized by mutation annotation levels.

    A MuTree stores variant mutant data for a set of samples in a tree-like
    data structure. Each level in the tree corresponds to a particular
    mutation annotation hierarchy level, such as Gene, Form, Exon, Protein,
    etc. Each node in the tree corresponds to a particular value of the
    annotation level present in at least one of the samples stored in the
    tree, thus representing a mutation sub-type such as 'TP53' for the Gene
    level, 'Missense_Mutation' for the Form level, 'R34K' for the Protein
    level, and so on.

    A node N* at the ith level of the tree has children nodes for each of
    the mutation types present at the (i+1)th annotation level for the samples
    also having mutations of type represented by N*. Thus in a tree
    containing the levels Gene, Form, and Exon, a node representing the ACT1
    gene will have a child representing missense mutations of ACT1, but only
    if at least one of the samples in the tree has this type of missense
    mutations. Similarly, this ACT1 - missense node may have children
    corresponding further sub-types of this mutation located on the 3rd, 5th,
    or 8th exon of ACT1.

    Every node in a MuTree is also a MuTree, except for the leaf nodes, which
    are frozensets of the samples which the mutation sub-type with all of the
    annotation level values of the parent nodes. Thus in the above example,
    the node representing the missense mutations of the ACT1 gene located on
    its 5th exon would simply be the samples with this mutation sub-type,
    since 'Exon' is the final annotation level contained in this MuTree.

    Levels can either be fields in the 'muts' DataFrame, in which case the
    tree will have a branch for each unique value in the field, or one of the
    keys of the MuTree.mut_fields object, in which case they will be defined
    by the corresponding MuType.muts_<level> method.

    Args:
        muts (:obj:`pd.DataFrame`, shape = [n_muts, n_annot_fields]
            Input mutation data, each record is a mutation occurring in
            a sample to be included in the tree.
            Must contain a 'Sample' column.

        levels (:obj:`tuple` of :obj:`str`)
            A list of mutation annotation levels to be included in the tree.

    Attributes:
        depth (int): How many mutation levels are above the tree
                     in the hierarchy.
        mut_level (str): The mutation annotation level described by the top
                         level of the tree.

    Examples:
        >>> mut_data = pd.DataFrame(
        >>>     {'Sample': ['S1', 'S2', 'S3', 'S4'],
        >>>      'Gene': ['TP53', 'TP53', 'KRAS', 'TP53'],
        >>>      'Exon': ['3', '3', '2', '7'],
        >>>      'Protein': ['H3R', 'S7T', 'E1R', 'Y11R']}
        >>>     )
        >>> mtree = MuTree(mut_data, levels=['Gene', 'Exon', 'Protein'])
        >>> print(mtree)
            Gene IS TP53 AND
                Exon is 3 AND
                    Protein is H3R: S1
                    Protein is S7T: S2
                Exon is 7 AND
                    Protein is Y11R: S4
            Gene is KRAS AND
                Exon is 2 AND
                    Protein is E1R: S3
    """

    # mapping between fields in an input mutation table and
    # custom mutation levels
    mut_fields = {
        'Type': ('Consequence', 'Protein'),
        'Location': ('Protein', ),
        }

    @classmethod
    def split_muts(cls, muts, lvl_name, **kwargs):
        """Splits mutations into tree branches for a given level.

        Args:
            muts (pd.DataFrame), shape = [n_muts, n_annot_fields]
                A list of mutations to be split according to the given
                annotation level, where each row corresponds to a mutation
                in a particular sample. Must contain the annotation fields
                needed by the given level.
            lvl_name (str)
                An annotation level, must be either a column in the mutation
                dataframe, a parsed variation thereof, or a custom annotation
                level listed in `MuTree.mut_fields`.

        Returns:
            split_muts (:obj:`dict` of :obj:`pd.DataFrame`)
        """

        # level names have to consist of a base level name and an al
        # parsing label separated by an underscore
        lvl_info = lvl_name.split('_')
        if len(lvl_info) > 2:
            raise ValueError(
                "Invalid level name {} with more than two fields!".format(
                    lvl_name)
                )

        # if a parsing label is present, add the parsed level
        # to the table of mutations
        elif len(lvl_info) == 2:
            parse_lbl = lvl_info[1].lower()
            parse_fx = 'parse_{}'.format(parse_lbl)

            if lvl_info[0] not in muts:
                lvl_func = 'cls.muts_{}'.format(lvl_info[0].lower())
                muts_dict = eval(lvl_func)(muts, **kwargs)
                muts = pd.DataFrame([])

                for lvl, muts_df in muts_dict.items():
                    muts = pd.concat([
                        muts, muts_df.join(pd.Series(lvl, index=muts_df.index,
                                                     name=lvl_info[0]))
                        ])
                
            if parse_fx in cls.__dict__:
                muts = eval('cls.{}'.format(parse_fx))(muts, lvl_info[0])
            
            else:
                raise ValueError(
                    "Custom parse label {} must have a corresponding <{}> "
                    "method defined in the MuTree class!".format(
                        parse_lbl, parse_fx)
                    )

        # splits mutations according to values of the specified level
        if isinstance(muts, tuple):
            if np.all(pd.isnull(val) for _, val in muts):
                split_muts = {}

            else:
                split_muts = muts

        elif lvl_name in muts:
            split_muts = dict(tuple(muts.groupby(lvl_name)))

        else:
            split_fx = 'muts_{}'.format(lvl_info[0].lower())

            if split_fx in cls.__dict__:
                split_muts = eval(
                    'cls.{}'.format(split_fx))(muts)

            else:
                raise ValueError(
                    "Custom mutation level {} must have a corresponding <{}> "
                    "method defined in the MuTree class!".format(
                        lvl_name, split_fx)
                    )

        return split_muts

    """Functions for defining custom mutation levels.

    Args:
        muts (pd.DataFrame), shape = [n_muts, n_annot_fields]
            Mutations to be split according to the given level.
            Must contain a 'Sample' field as well as the fields defined in
            MuTree.mut_fields for each custom level.

    Returns:
        new_muts (:obj:`dict` of :obj:`pd.DataFrame`)
    """

    @staticmethod
    def muts_type(muts):
        """Parses mutations according to Type, which can be 'CNV' (Gain or
           Loss), 'Point' (missense and silent mutations), or 'Frame' (indels,
           frameshifts, nonsense mutations).

        """
        new_muts = {}

        cnv_indx = muts['Form'].isin(
            ['HomDel', 'HetDel', 'HetGain', 'HomGain'])
        point_indx = muts['Protein'].str.match(
            pat='^p\\.[A-Z][0-9]+[A-Z]$', as_indexer=True, na=False)
        frame_indx = muts['Protein'].str.match(
            pat='^p\\..*(?:\\*|(?:ins|del))', as_indexer=True, na=False)
        other_indx = ~(cnv_indx | point_indx | frame_indx)

        if any(cnv_indx):
            new_muts['CNV'] = muts.loc[cnv_indx, :]
        if any(point_indx):
            new_muts['Point'] = muts.loc[point_indx, :]
        if any(frame_indx):
            new_muts['Frame'] = muts.loc[frame_indx, :]
        if any(other_indx):
            new_muts['Other'] = muts.loc[other_indx, :]

        return new_muts

    @staticmethod
    def muts_location(muts):
        """Parses mutation according to protein location."""
        new_muts = {}

        loc_tbl = muts['Protein'].str.extract('(^p\\.[A-Z|*])([0-9]+)',
                                              expand=False)
        none_indx = pd.isnull(loc_tbl.iloc[:, 1])
        loc_tbl.loc[none_indx, 1] = muts['Protein'][none_indx]

        for loc, grp in loc_tbl.groupby(by=1):
            new_muts[loc] = muts.loc[grp.index, :]

        return new_muts

    """Functions for custom parsing of mutation levels.

    Args:
        muts (pd.DataFrame), shape = [n_muts, n_annot_fields]
            Mutations whose properties are to be parsed.

    Returns:
        new_muts (pd.DataFrame), shape = [n_muts, n_annot_fields]
            The given list of mutations with the given mutation levels
            altered according to the corresponding parsing rule.
    """

    @staticmethod
    def parse_base(muts, parse_lvl):
        """Removes trailing _Del and _Ins, merging insertions and deletions
           of the same type together.
        """

        new_lvl = '{}_base'.format(parse_lvl)

        new_muts = muts.assign(**{new_lvl: muts.loc[:, parse_lvl]})
        new_muts.replace(to_replace={new_lvl: {'_(Del|Ins)$': ''}},
                         regex=True, inplace=True)

        return new_muts

    @staticmethod
    def parse_bins(muts, parse_lvl):
        """Arranges continuous mutation properties such as Exons and MutSig
           scores into equally-spaced discrete bins.
        """

        bin_vals = muts[parse_lvl].replace(['.'], np.nan)
        skip_indx = pd.isnull(bin_vals)
        bin_vals = pd.to_numeric(bin_vals[~skip_indx])

        new_muts = muts.copy()
        if len(bin_vals):

            new_lvl = '{}_bins'.format(parse_lvl)
            new_muts[new_lvl] = 'None'
            new_muts.loc[~skip_indx, new_lvl] = pd.cut(
                bin_vals, bins=10).astype('str')

        return new_muts

    @staticmethod
    def parse_clust(muts, parse_lvl):
        """Clusters continuous mutation properties such as PolyPhen scores
           and genomic locations into discrete levels.
        """

        # find the property values that can be interpreted as a numeric value
        clust_vals = muts[parse_lvl].replace(['.'], np.nan)
        skip_indx = pd.isnull(clust_vals)
        clust_vals = pd.to_numeric(clust_vals[~skip_indx])

        new_muts = muts.copy()
        if len(clust_vals):

            bw_val = (clust_vals.max() - clust_vals.min()) / 40
            if bw_val > 4:
                round_val = 0
            else:
                round_val = 1 - floor(log10(bw_val))

            # find the clusters of property values
            mshift = MeanShift(bandwidth=bw_val)
            mshift.fit(clust_vals.values.reshape(-1, 1))
            
            # parse the clusters into a label for each mutation
            clust_vec = [
                '{}__{}'.format(
                    parse_lvl,
                    mshift.cluster_centers_[lbl, 0].round(round_val)
                    )
                for lbl in mshift.labels_
                ]
            
            # add a new column to the mutations table with the cluster labels
            new_muts['{}_clust'.format(parse_lvl)] = 'None'
            new_muts.loc[~skip_indx, '{}_clust'.format(parse_lvl)] = clust_vec

        return new_muts

    def __new__(cls, muts=None, levels=('Gene', 'Form'), **kwargs):
        """Given a list of mutations and a set of mutation levels, determines
           whether a mutation tree should be built, or a frozenset returned,
           presumably as a branch of another MuTree.
        """

        # used for eg. copying
        if muts is None:
            return super(MuTree, cls).__new__(cls)

        # checks mutation table for proper format
        if not isinstance(muts, pd.DataFrame):
            raise TypeError("Mutation table must be a pandas DataFrame!")

        if 'Sample' not in muts:
            raise ValueError("Mutation table must have a 'Sample' field!")

        # initializes branch search variables
        muts_left = False
        lvls_left = list(levels)

        # look for a level at which MuTree branches can be sprouted until we
        # are either out of levels or we have found such a level
        while lvls_left and not muts_left:
            cur_lvl = lvls_left.pop(0).split('_')[0]

            # if the level is a field in the mutation DataFrame, check if any
            # mutations have non-null values...
            if cur_lvl in muts:
                muts_left = not np.all(pd.isnull(muts[cur_lvl]))

            # ...otherwise, check if the fields corresponding to the custom
            # level have any non-null values...
            elif cur_lvl in cls.mut_fields:
                if not np.all([x in muts for x in cls.mut_fields[cur_lvl]]):
                    raise ValueError("For mutation level " + cur_lvl + ", "
                                     + str(cls.mut_fields[cur_lvl])
                                     + " need to be provided as fields.")

                else:
                    muts_left = not np.all(pd.isnull(
                        muts.loc[:, cls.mut_fields[cur_lvl]]))

            else:
                raise ValueError("Unknown mutation level " + cur_lvl
                                 + " which is not in the given mutation data"
                                 + " frame and not a custom-defined level!")

        # if we have found a level at which branches can be built,
        # continue with instantiating the MuTree...
        if muts_left:
            return super(MuTree, cls).__new__(cls)

        # ...otherwise, return a set of samples as a leaf node
        else:
            return {samp: ({fld: group[fld].tolist()
                            for fld in kwargs['leaf_annot']}
                           if 'leaf_annot' in kwargs and kwargs['leaf_annot']
                           else None)
                    for samp, group in muts.groupby('Sample')}

    def __init__(self, muts, levels=('Gene', 'Form'), **kwargs):
        if 'depth' in kwargs:
            self.depth = kwargs['depth']
            del(kwargs['depth'])

        else:
            self.depth = 0

        # intializes mutation hierarchy construction variables
        lvls_left = list(levels)
        self._child = {}
        rel_depth = 0

        # look for a mutation level at which we can create branches until we
        # have found such a level, note that we know such a level exists
        # because of the check performed in the __new__ method
        while lvls_left and not self._child:

            # get the split of the mutations given the current level
            cur_lvl = lvls_left.pop(0)
            splat_muts = self.split_muts(muts, cur_lvl, **kwargs)

            # if the mutations can be split, set the current mutation
            # level of the tree...
            if splat_muts:
                self.mut_level = levels[rel_depth]

                # ...and also set up the children nodes of the tree, which can
                # either all be frozensets corresponding to leaf nodes...
                if isinstance(splat_muts, tuple):
                    self._child = dict(splat_muts)

                # ...or a mixture of further MuTrees and leaf nodes
                else:
                    self._child = {
                        nm: MuTree(mut, lvls_left, depth=self.depth + 1,
                                   **kwargs)
                        for nm, mut in splat_muts.items()
                        }

            # if the mutations cannot be split at this level, move on to the
            # next level and keep track of how many levels we have skipped
            else:
                rel_depth += 1

    def __iter__(self):
        """Allows iteration over mutation categories at the current level, or
           the samples at the current level if we are at a leaf node."""

        return iter(self._child.items())

    def __getitem__(self, key):
        """Gets a particular category of mutations at the current level."""

        if not key:
            key_item = self

        elif isinstance(key, str):
            key_item = self._child[key]

        elif hasattr(key, '__getitem__'):
            sub_item = self._child[key[0]]

            if isinstance(sub_item, MuTree):
                key_item = sub_item[key[1:]]
            elif key[1:]:
                raise KeyError("Key has more levels than this MuTree!")
            else:
                key_item = sub_item

        else:
            raise TypeError("Unsupported key type {} !".format(type(key)))

        return key_item

    def __eq__(self, other):
        if not isinstance(other, MuTree):
            return False

        return self._child == other._child

    def __str__(self):
        """Printing a MuTree shows each of the branches of the tree and
           the samples at the end of each branch."""

        new_str = self.mut_level

        for nm, mut in self:
            new_str += ' IS {}'.format(nm)

            if isinstance(mut, MuTree):
                new_str += (' AND ' + '\n'
                            + '\t' * (self.depth + 1) + str(mut))

            # if we have reached a root node, print the samples
            elif len(mut) > 8:
                    new_str += ': ({} samples)'.format(str(len(mut)))
            else:
                    new_str += ': {}'.format(
                        reduce(lambda x, y: '{},{}'.format(x, y), mut))

            new_str += ('\n' + '\t' * self.depth)
        new_str = gsub('\n$', '', new_str)

        return new_str

    def __len__(self):
        """Returns the number of unique samples this MuTree contains."""

        return len(self.get_samples())

    def sort_iter(self):
        """Iterates through the branches of the tree, ordering mutation 
           attributes where possible."""

        if self.mut_level in ['Exon', 'Location']:
            return iter(sorted(
                [("0", branch) if lbl == '.' else (lbl, branch)
                 for lbl, branch in self._child.items()],
                key=lambda x: int(gsub('[^0-9]', '0', x[0].split('/')[0]))
                ))

        else:
            return self.__iter__()

    def get_newick(self):
        """Get the Newick tree format representation of this MuTree."""
        newick_str = ''

        for nm, mut in self.sort_iter():

            if isinstance(mut, MuTree):
                newick_str += '(' + gsub(',$', '', mut.get_newick()) + ')'

            if nm == "0":
                newick_str += '{*none*},'
            else:
                newick_str += '{' + nm + '},'

        if self.depth == 0:
            newick_str = gsub(',$', '', newick_str) + ';'

        return newick_str

    def __hash__(self):
        mut_str = self.get_newick()
        return hash(tuple(mut_str.count(k) for k in sorted(set(mut_str))))

    def get_levels(self):
        """Gets all the levels present in this tree and its children."""

        levels = {self.mut_level}

        for _, mut in self:
            if isinstance(mut, MuTree):
                levels |= mut.get_levels()

        return levels

    def get_samples(self):
        """Gets the set of unique samples contained within the tree."""

        samps = set()

        for nm, mut in self:
            if isinstance(mut, MuTree):
                samps |= mut.get_samples()
            elif isinstance(mut, dict):
                samps |= set(mut)
            else:
                raise ValueError

        return samps

    def get_leaf_annot(self, ant_flds):
        ant_dict = dict()

        for nm, mut in self:
            if isinstance(mut, MuTree):
                lf_ant = mut.get_leaf_annot(ant_flds)
            elif isinstance(mut, dict):
                lf_ant = {samp: {fld: ant[fld] for fld in ant_flds}
                          for samp, ant in mut.items()}

            else:
                raise ValueError

            ant_dict = {**{samp: lf_ant[samp]
                           for samp in lf_ant.keys() - ant_dict.keys()},
                        **{samp: ant_dict[samp]
                           for samp in ant_dict.keys() - lf_ant.keys()},
                        **{samp: {ant_fld: (ant_dict[samp][ant_fld]
                                            + lf_ant[samp][ant_fld])
                                  for ant_fld in ant_flds}
                           for samp in ant_dict.keys() & lf_ant.keys()}}

        return ant_dict

    def subtree(self, samps):
        """Modifies the MuTree in place so that it only has the given samples.

        Args:
            samps (list or set)

        Returns:
            self

        Examples:
            >>> # remove a sample from the tree
            >>> mtree = MuTree(...)
            >>> new_tree = mtree.subtree(mtree.get_samples() - {'TCGA-04'})

        """
        new_child = self._child.copy()
        for nm, mut in self:

            if isinstance(mut, MuTree):
                new_samps = mut.get_samples() & set(samps)

                if new_samps:
                    new_child[nm] = mut.subtree(new_samps)
                else:
                    del(new_child[nm])

            elif isinstance(mut, frozenset):
                new_samps = mut & frozenset(samps)

                if new_samps:
                    new_child[nm] = new_samps
                else:
                    del(new_child[nm])

            else:
                pass

        self._child = new_child
        return self

    def allkey(self, levels=None):
        """Gets the key corresponding to the MuType with all the branches.

        A convenience function that makes it easier to list all of the
        possible branches present in the tree, and to instantiate MuType
        objects that correspond to all of the possible mutation types.

        Args:
            levels (list)

        Returns:
            dict

        """
        new_key = None

        # use all levels if no levels to filter on are provided
        if levels is None:
            levels = self.get_levels()

        if self.mut_level in levels:
            new_key = {
                (self.mut_level, nm): (
                    branch.allkey(branch.get_levels() & set(levels))
                    if (isinstance(branch, MuTree)
                        and branch.get_levels() & set(levels))
                    else None
                    )
                for nm, branch in self
                }

        elif set(levels) & self.get_levels():
            new_key = reduce(
                lambda x, y: dict(
                    tuple(x.items()) + tuple(y.items())
                    + tuple((k, None) if x[k] is None
                            else (k, {**x[k], **y[k]})
                            for k in set(x) & set(y))),
                [branch.allkey(branch.get_levels() & set(levels))
                 for nm, branch in self
                 if (isinstance(branch, MuTree)
                     and branch.get_levels() & set(levels))]
                )

        return new_key

    def branchtypes(self, mtype=None, sub_levels=None, min_size=1):
        """Gets all MuTypes corresponding to one branch of the MuTree.

        Args:
            mtype (MuType), optional
                A set of mutations of which the returned MuTypes must be a
                subset. The default is to use all MuTypes within this MuTree.
            sub_levels (list of str), optional
                The levels of the leaf nodes of the returned MuTypes. The
                default is to use all levels of the MuTree.
            min_size (int), optional
                The minimum number of samples in each returned MuType. The
                default is not to do filtering based on MuType sample count.

        Returns:
            sub_mtypes (set of MuType)

        Examples:
            >>> # get all possible single-branch MuTypes
            >>> mtree = MuTree(...)
            >>> mtree.branchtypes()
            >>>
            >>> # get all possible MuTypes with at least five samples
            >>> mtree.branchtypes(min_size=5)
            >>>
            >>> # use different filters on the MuTypes returned for a given
            >>> # MuTree based on mutation type and mutation level
            >>> mtree.branchtypes(sub_levels=['Gene'])
                {MuType({('Gene', 'TP53'): None}),
                 MuType({('Gene', 'TTN'): None})}
            >>> mtree.branchtypes(sub_levels=['Gene', 'Type'])
                {MuType({('Gene', 'TP53'): {('Type', 'Point'): None}}),
                 MuType({('Gene', 'TP53'): {('Type', 'Frame'): None}}),
                 MuType({('Gene', 'TTN'): {('Type', 'Point'): None}})}
            >>> mtree.branchtypes(mtype=MuType({('Gene', 'TTN'): None}),
            >>>               sub_levels=['Gene', 'Type'])
                {MuType({('Gene', 'TTN'): {('Type', 'Point'): None}})}

        """
        sub_mtypes = set()

        # use all mutation property levels if none are given
        if sub_levels is None:
            sub_levels = self.get_levels()

        # if not mutation type is given, use all the types in this tree
        if mtype is None:
            mtype_dict = {lbl: None for lbl in dict(self)}
        else:
            mtype_dict = dict(mtype.subtype_list())

        # check if we are at one of the property levels specified in the
        # query and if this level is also that of the given mutation type
        if self.mut_level in sub_levels:
            if mtype is None or self.mut_level == mtype.cur_level:

                # for each level label specified by the mutation type that is
                # also in the tree, check if there are enough samples in the
                # corresponding tree branch to satisfy the query
                for lbl in dict(self).keys() & mtype_dict.keys():
                    if len(self[lbl]) >= min_size:

                        # if we are at a leaf node of the mutation type or the
                        # tree, or if there are no further property levels to
                        # recurse into, return the mutation type of the branch
                        if (mtype_dict[lbl] is None
                                or not isinstance(self[lbl], MuTree)
                                or not (set(sub_levels)
                                        & self[lbl].get_levels())):
                            sub_mtypes.update(
                                {MuType({(self.mut_level, lbl): None})})

                        # recurse deeper into this branch if possible
                        if (isinstance(self[lbl], MuTree)
                                and set(sub_levels) & self[lbl].get_levels()):
                            sub_mtypes.update(
                                {MuType({(self.mut_level, lbl): rec_mtype})
                                 for rec_mtype in self[lbl].branchtypes(
                                     mtype_dict[lbl], sub_levels, min_size)}
                                )

            # if we are at one of the property levels specified in the query
            # but the levels of the mutation type and tree do not match...
            else:
                for lbl, branch in self:
                    if (isinstance(branch, MuTree)
                            and len(mtype.get_samples(branch)) >= min_size):

                        # ...for each branch at this level return its
                        # sub-branches that satisfy the query
                        if set(sub_levels) & branch.get_levels():
                            sub_mtypes.update(
                                {MuType({(self.mut_level, lbl): rec_mtype})
                                 for rec_mtype in branch.branchtypes(
                                     mtype, sub_levels, min_size)}
                                )

                        else:
                            sub_mtypes.update(
                                {MuType({(self.mut_level, lbl): None})})

        # if we are not one of the property levels specified in the query but
        # the levels of the mutation type and tree do match...
        else:
            if mtype is None or self.mut_level == mtype.cur_level:
                rec_dict = {}

                # ...group together the identical sub-branches at deeper
                # levels across branches that do satisfy the query
                for lbl in dict(self).keys() & mtype_dict.keys():
                    rec_mtypes = self[lbl].branchtypes(
                        mtype_dict[lbl], sub_levels, min_size=1)

                    for rec_mtype in rec_mtypes:
                        rec_dict[rec_mtype] = (
                            rec_mtype.get_samples(self[lbl]) |
                            rec_dict.setdefault(rec_mtype, set())
                            )

                sub_mtypes.update(
                    {mtype for mtype, samps in rec_dict.items()
                     if len(samps) >= min_size}
                    )

            # if we are not at a level matching that of the mutation type or
            # specified in the query, recurse into each branch, only
            # considering those branches that could possibly satisfy the query
            else:
                for lbl, branch in self:
                    if (isinstance(branch, MuTree)
                            and set(sub_levels) & branch.get_levels()
                            and len(branch) >= min_size):

                        sub_mtypes.update(
                            branch.branchtypes(mtype, sub_levels, min_size))

        return sub_mtypes

    def combtypes(self,
                  mtype=None, sub_levels=None, comb_sizes=(1, 2),
                  min_type_size=10, min_branch_size='auto'):
        """Gets all MuTypes that combine multiple branches of the tree.

        Args:
            mtype (MuType), optional
                A set of mutations of which the returned MuTypes must be a
                subset. The default is to use all MuTypes within this MuTree.
            sub_levels (list of str), optional
                The levels of the leaf nodes of the returned MuTypes. The
                default is to use all levels of the MuTree.
            comb_sizes (list of int), optional
                The number of branches that each returned MyType can combine.
                The default is to consider combinations of up to two branches.
            min_type_size (int), optional
                The minimum number of samples in each returned MuType. The
                default is each returned MuType having at least ten samples.
            min_branch_size (:obj:`int` or :obj:`str`), optional
                The minimum number of samples in each of the MuTypes combined
                to compose each of the returned MuTypes. The default is to
                divide min_type_size by the number of MuTypes in each returned
                combination MuType.

        Returns:
            comb_mtypes (:obj:`set` of :obj:`MuType`)

        Examples:
            >>> # get all possible MuTypes that combine three branches
            >>> mtree = MuTree(...)
            >>> mtree.combtypes(comb_sizes=(3,))
            >>>
            >>> # get all possible MuTypes that combine two 'Type' branches
            >>> # that have at least twenty samples in this tree
            >>> mtree.combtypes(min_size=20, sub_levels=['Type'])

        """
        branch_mtypes = set()
        comb_mtypes = set()
        #TODO: more error-checking for input values, e.g. sizes can't be zero

        if not isinstance(min_branch_size, str):
            branch_mtypes = self.branchtypes(
                mtype, sub_levels, min_size=min_branch_size)

        for csize in comb_sizes:
            if min_branch_size == 'auto':
                branch_mtypes = self.branchtypes(
                    mtype, sub_levels, min_size=(min_type_size / csize))

            if branch_mtypes:
                for mtype_combs in combn(branch_mtypes, csize):

                    if (csize == 1
                            or all([not (mtype1.is_supertype(mtype2)
                                         or mtype2.is_supertype(mtype1))
                                    for mtype1, mtype2
                                    in combn(mtype_combs, 2)])):
                        new_mtype = reduce(or_, mtype_combs)

                        if (min_branch_size == 'auto'
                                or (len(new_mtype.get_samples(self))
                                    >= min_type_size)):
                            comb_mtypes |= {new_mtype}

        return comb_mtypes

    def find_unique_subtypes(self,
                             max_types=1000, max_combs=5, verbose=0,
                             **comb_args):
        use_mtypes = set()
        use_sampsets = set()
        use_combs = 0

        while len(use_mtypes) <= max_types and use_combs < max_combs:
            use_combs += 1

            cur_mtypes = self.combtypes(comb_sizes=(use_combs, ), **comb_args)
            cur_sampsets = {mtype: frozenset(mtype.get_samples(self))
                            for mtype in cur_mtypes - use_mtypes}

            if verbose:
                print("Found {} new sub-types!".format(len(cur_sampsets)))

            if len(cur_sampsets) <= max_types:
                for i, (mtype, sampset) in enumerate(cur_sampsets.items()):
                    if verbose > 1:
                        if (i % (max_types // 10)) == (max_types // 20):
                            print("\t...checked {} sub-types...".format(i))
                
                    if sampset in use_sampsets:
                        if verbose:
                            print("\t\tRemoving functionally "
                                  "duplicate MuType {}".format(mtype))

                    else:
                        use_mtypes.update({mtype})
                        use_sampsets.update({sampset})

            else:
                break

        return use_mtypes

    def status(self, samples, mtype=None):
        """Finds if each sample has a mutation of this type in the tree.

        Args:
            samples (:obj:`list` of :obj:`str`)
                Which samples' mutation status is to be retrieved.
            mtype (MuType, optional)
                A set of mutations whose membership we want to test.
                The default is to check against any mutation
                contained in the tree.

        Returns:
            stat_list (:obj:`list` of :obj:`bool`)
                For each input sample, whether or not it has a mutation
                in the given set.

        """
        if mtype is None:
            samp_list = self.get_samples()
        else:
            samp_list = mtype.get_samples(self)

        return [s in samp_list for s in samples]

    def match_levels(self, mtype):
        if self.mut_level == mtype.cur_level:
            mtype_mtch = all(
                False if not isinstance(muts, MuTree)
                else muts.match_levels(tp)
                for (nm, muts), (lbl, tp) in product(self,
                                                     mtype.subtype_list())
                if tp is not None and (nm == lbl or lbl not in self._child)
                )

        else:
            mtype_mtch = any(muts.match_levels(mtype)
                             for muts in self._child.values()
                             if isinstance(muts, MuTree))

        return mtype_mtch

