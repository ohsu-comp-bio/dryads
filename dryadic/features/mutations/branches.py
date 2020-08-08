
from functools import reduce
from itertools import product
from itertools import combinations as combn
from operator import and_
import re


class MuType(object):
    """A set of properties recursively defining a particular type of mutation.

    This class corresponds to a mutation type defined through a list of
    properties, each possibly linked to a further mutation sub-type. Used in
    conjunction with the above MuTree class to represent and navigate the
    space of possible mutation subsets in a given cohort. While a MuTree is
    linked to a particular set of samples, a MuType represents a mutation
    type abstract of any samples that may or may not have it.

    MuTypes are initialized via recursively structured type dictionaries of
    the form type_dict={(Level, Label1): <None or subtype_dict>,
                        (Level, Label2): <None or subtype_dict>, ...}

    The keys of this type dictionary are thus 2-tuples composed of
        1) `Level`: anything that defines categories that a mutation can
        belong to, such as Gene, Exon, PolyPhen
        2) `Label`: one or more of these categories, such as TP53, 7/11, 0.67

    The value for a given key in a type_dict can be either `None`, indicating
    that all mutations in this category are represented this MuType, or
    another type_dict, indicating that only the given subset of mutations
    within this category are represented by this MuType. A MuType can thus
    contain children MuTypes, and the set of mutations the MuType stands for
    is the intersection of the children and their parent(s), and the union of
    parents and their siblings.

    For the sake of convenience, a `Label` can itself be a tuple of mutation
    property level categories, thus indicating that the corresponding subtype
    value applies to all the listed categories. Type dictionaries are
    automatically rationalized to group together identical subtype values when
    they are being parsed in order to reduce the memory footprint of each
    MuType object.

    Note that subtypes can be already-instantiated MuType objects instead of
    type dictionaries. Explicitly passing the `None` object by itself as a
    type dictionary creates a MuType corresponding to the empty null set of
    mutations, as does passing any empty iterable such as [] or (,).

    Arguments:
        type_dict (dict, list, tuple, None, or MuType)

    Attributes:
        cur_level (str): The mutation property level whose categories are
                         listed in this type.

    Examples:
        >>> # mutations of the KRAS gene
        >>> mtype1 = MuType({('Gene', 'KRAS'): None})
        >>>
        >>> # missense mutations of the KRAS gene
        >>> mtype2 = MuType({('Gene', 'KRAS'):
        >>>             {('Form', 'Missense_Mutation'): None}})
        >>>
        >>> # mutations of the BRAF or RB1 genes
        >>> mtype3 = MuType({('Gene', ('BRAF', 'RB1')): None})
        >>>
        >>> # frameshift mutations of the BRAF or RB1 genes and nonsense
        >>> # mutations of the TP53 gene occuring on its 8th exon
        >>> mtype4 = MuType({('Gene', ('BRAF', 'RB1')):
        >>>                     {('Type', 'Frame_Shift'): None},
        >>>                 {('Gene', 'TP53'):
        >>>                     {('Form', 'Nonsense_Mutation'):
        >>>                         {('Exon', '8/33'): None}}})

    """

    def __init__(self, type_dict):

        # ensures the type dictionary is in the proper format, parses out the
        # mutation property level from its keys
        if not type_dict:
            type_dict = {}
            self.cur_level = None

        elif isinstance(type_dict, dict):
            levels = set(lvl for lvl, _ in type_dict)

            if len(levels) > 1:
                raise ValueError("Improperly defined set key with multiple"
                                 "mutation levels!")

            else:
                self.cur_level = tuple(levels)[0]

        else:
            raise TypeError("MuType type dictionary must be a dict object!")

        # parses out the category labels listed for the given property level
        level_lbls = [(lbls, ) if isinstance(lbls, str) else lbls
                      for _, lbls in type_dict]

        # creates an expanded type dictionary where category labels that were
        # originally grouped together by subtype are given separate keys
        full_pairs = [
            (lbl, sub_type)
            for lbls, sub_type in zip(level_lbls, type_dict.values())
            if not (isinstance(sub_type, MuType) and sub_type.is_empty())
            for lbl in lbls
            ]

        # merges identical labels according to the union of their subtypes
        # i.e. silent:Exon7, silent:Exon8 => silent:(Exon7 or Exon8)
        full_dict = {}
        for lbl, sub_type in full_pairs:

            if lbl in full_dict:
                if sub_type is None or full_dict[lbl] is None:
                    full_dict[lbl] = None

                elif isinstance(sub_type, dict):
                    full_dict[lbl] |= MuType(sub_type)

                else:
                    full_dict[lbl] |= sub_type

            elif isinstance(sub_type, dict):
                full_dict[lbl] = MuType(sub_type)

            else:
                full_dict[lbl] = sub_type

        # collapses category labels with the same subtype into one key:subtype
        # pair, i.e. silent:None, frameshift:None => (silent, frameshift):None
        uniq_vals = [(frozenset(sorted(k for k, v in full_dict.items()
                                       if v == sub_type)),
                      sub_type)
                     for sub_type in set(full_dict.values())]

        # merges the subtypes of type dictionary entries with the same
        # category label, i.e. silent: <Exon IS 7/11>, silent: <Exon IS 10/11>
        # => silent: <Exon IS 7/11 or 10/11> to create the final dictionary
        self._child = {}
        for lbls, sub_type in uniq_vals:

            if lbls in self._child:
                if sub_type is None or self._child[lbls] is None:
                    self._child[lbls] = None

                else:
                    self._child[lbls] |= sub_type

            else:
                self._child[lbls] = sub_type

    def __getstate__(self):
        """Defines instantiation parameters for unpickling."""
        state_dict = dict()

        for lbls, tp in self._child.items():
            lbls_key = self.cur_level, tuple(sorted(lbls))

            if tp is None:
                state_dict[lbls_key] = None
            else:
                state_dict[lbls_key] = tp.__getstate__()

        return state_dict

    def __setstate__(self, state):
        self.__init__(state)

    def is_empty(self):
        """Checks if this MuType corresponds to the null mutation set."""
        return self._child == {}

    def get_levels_tree(self):
        return (self.cur_level, ) + tuple({
            tp.get_levels_tree() for tp in self._child.values()
            if tp is not None
            })

    def get_levels(self):
        """Gets the property levels present in this type and its subtypes."""
        levels = {self.cur_level}

        for tp in self._child.values():
            if tp is not None:
                levels |= set(tp.get_levels())

        return levels

    def get_sorted_levels(self):
        """Gets sorted list of the properties present throughout this type."""
        child_levels = set()

        for tp in self._child.values():
            if tp is not None:
                child_levels |= {tp.get_sorted_levels()}

        if child_levels:
            sorted_levels = tuple([self.cur_level]
                                  + list(sorted(child_levels, key=len)[-1]))

        elif self.cur_level:
            sorted_levels = self.cur_level,
        else:
            sorted_levels = None

        return sorted_levels

    def __hash__(self):
        """MuType hashes are defined in an analagous fashion to those of
           tuples, see for instance http://effbot.org/zone/python-hash.htm"""
        value = 0x163125 ^ len(self._child)

        for lbls, tp in sorted(self._child.items(), key=lambda x: list(x[0])):
            value += eval(hex((int(value) * 1000007) & 0xFFFFFFFF)[:-1])
            value ^= hash(lbls) ^ hash(tp)

        if value == -1:
            value = -2

        return value

    def child_iter(self):
        """Returns an iterator over the collapsed (labels):subtype pairs."""
        return iter(self._child.items())

    def subtype_iter(self):
        """Returns an iterator over all unique label:subtype pairs."""
        return iter((lbl, tp) for lbls, tp in self._child.items()
                    for lbl in lbls)

    def label_iter(self):
        """Returns an iterator over the unique labels at this level."""
        return iter(lbl for lbls in self._child for lbl in lbls)

    def leaves(self):
        """Gets all of the possible subsets of this MuType that contain
           exactly one of the leaf properties."""
        mkeys = []

        for lbls, tp in self._child.items():
            if tp is None:
                mkeys += [{(self.cur_level, lbl): None} for lbl in lbls]

            else:
                mkeys += [{(self.cur_level, lbl): sub_tp}
                          for lbl in lbls for sub_tp in tp.leaves()]

        return mkeys

    def __eq__(self, other):
        """Checks if one MuType is equal to another."""

        # if the other object is not a MuType they are not equal
        if not isinstance(other, MuType):
            eq = False

        # MuTypes with different mutation property levels are not equal
        elif self.cur_level != other.cur_level:
            eq = False

        # MuTypes with the same mutation levels are equal if and only if
        # they have the same subtypes for the same level category labels
        else:
            eq = self._child == other._child

        return eq

    def __lt__(self, other):
        """Defines a sort order for MuTypes."""
        if isinstance(other, MutComb):
            return True

        if not isinstance(other, MuType):
            return NotImplemented

        # handles case where at least one of the MuTypes is empty
        if self.is_empty() and not other.is_empty():
            return True
        if other.is_empty():
            return False

        # we first compare the mutation property levels of the two MuTypes...
        if self.cur_level == other.cur_level:

            # sort label:subtype pairs according to label such that pairwise
            # invariance is preserved
            self_pairs = sorted(self.subtype_iter(), key=lambda x: x[0])
            other_pairs = sorted(other.subtype_iter(), key=lambda x: x[0])

            # ...then compare how many (label:subtype) pairs they have...
            if len(self_pairs) == len(other_pairs):
                self_lbls = [lbl for lbl, _ in self_pairs]
                other_lbls = [lbl for lbl, _ in other_pairs]

                # ...then compare the labels themselves...
                if self_lbls == other_lbls:
                    self_lvls = self.get_levels()
                    other_lvls = other.get_levels()

                    # ...then compare how deep the subtypes recurse...
                    if len(self_lvls) == len(other_lvls):
                        self_subtypes = [tp for _, tp in self_pairs]
                        other_subtypes = [tp for _, tp in other_pairs]

                        # ...then compare the subtypes for each pair of
                        # matching labels...
                        for tp1, tp2 in zip(self_subtypes, other_subtypes):
                            if tp1 != tp2:

                                # for the first pair of subtypes that are not
                                # equal (always the same pair because entries
                                # are sorted), we recursively compare the pair
                                if tp1 is None:
                                    return False

                                elif tp2 is None:
                                    return True

                                else:
                                    return tp1 < tp2

                        # if all subtypes are equal, the two MuTypes are equal
                        else:
                            return False

                    # MuTypes with fewer subtype levels are sorted first
                    else:
                        return len(self_lvls) < len(other_lvls)

                # MuTypes with different labels are sorted according to the
                # order defined by the sorted label lists
                else:
                    return self_lbls < other_lbls

            # MuTypes with fewer mutation entries are sorted first
            else:
                return len(self_pairs) < len(other_pairs)

        # MuTypes with differing mutation property levels are sorted according
        # to the sort order of the property strings
        else:
            return self.cur_level < other.cur_level

    # remaining methods necessary to define rich comparison for MuTypes
    def __ne__(self, other):
        return not self == other

    def __le__(self, other):
        return self < other or self == other

    def __gt__(self, other):
        return not (self < other or self == other)

    def __ge__(self, other):
        return not (self < other)

    def __repr__(self):
        """Shows the hierarchy of mutation properties within the MuType."""
        new_str = ''

        # iterate over all mutation types at this level separately
        # regardless of their children
        for lbl, tp in self.subtype_iter():
            new_str += self.cur_level + ' IS ' + lbl

            if tp is not None:
                new_str += ' WITH ' + repr(tp)

            new_str += ' OR '

        return re.sub(' OR $', '', new_str)

    def __str__(self):
        """Gets a condensed label for the MuType."""
        new_str = ''
        self_iter = sorted(
            self._child.items(),
            key=lambda x: sorted(list(x[0]))
            )

        # if there aren't too many types to list at this mutation level...
        if len(self_iter) <= 10:

            # ...iterate over the types, grouping together those with the
            # same children to produce a more concise label
            for lbls, tp in self_iter:

                if len(lbls) > 1:

                    #TODO: find a more elegant way of dealing with this
                    if self.cur_level == 'Gene' and tp is None:
                        new_str += '|'.join(sorted(lbls))

                    else:
                        new_str += '(' + '|'.join(sorted(lbls)) + ')'

                else:
                    new_str += list(lbls)[0]

                if tp is not None:
                    new_str += ':' + str(tp)

                new_str += '|'

        # ...otherwise, show how many types there are and move on to the
        # levels further down if they exist
        else:
            new_str += "({} {}s)".format(
                self.__len__(), self.cur_level.lower())

            # condense sub-types at the further levels
            for lbls, tp in self_iter:
                new_str += "-(>= {} sub-types at level(s): {})".format(
                    len(tp),
                    reduce(lambda x, y: x + ', ' + y,
                           self.get_levels() - {self.cur_level})
                    )

        return re.sub('\\|+$', '', new_str)

    def get_filelabel(self):
        return "".join([c if re.match(r'\w', c) else '-' for c in str(self)])

    def __or__(self, other):
        """Returns the union of two MuTypes."""
        if not isinstance(other, MuType):
            return NotImplemented

        if self.is_empty():
            return other
        if other.is_empty():
            return self

        # gets the unique label:subtype pairs in each MuType
        self_dict = dict(self.subtype_iter())
        other_dict = dict(other.subtype_iter())

        if self.cur_level == other.cur_level:
            new_key = {}

            # adds the subtypes paired with the labels in the symmetric
            # difference of the labels in the two MuTypes
            new_key.update({(self.cur_level, lbl): self_dict[lbl]
                            for lbl in self_dict.keys() - other_dict.keys()})
            new_key.update({(self.cur_level, lbl): other_dict[lbl]
                            for lbl in other_dict.keys() - self_dict.keys()})

            # finds the union of the subtypes paired with each of the labels
            # appearing in both MuTypes
            new_key.update(
                {(self.cur_level, lbl): (
                    None if self_dict[lbl] is None or other_dict[lbl] is None
                    else self_dict[lbl] | other_dict[lbl]
                    )
                 for lbl in self_dict.keys() & other_dict.keys()}
                )

        else:
            raise ValueError(
                "Cannot take the union of two MuTypes with "
                "mismatching mutation levels {} and {}!".format(
                    self.cur_level, other.cur_level)
                )

        return MuType(new_key)

    def __and__(self, other):
        """Finds the intersection of two MuTypes."""
        if not isinstance(other, MuType):
            return NotImplemented

        if self.is_empty() or other.is_empty():
            return MuType({})

        # gets the unique label:subtype pairs in each MuType
        self_dict = dict(self.subtype_iter())
        other_dict = dict(other.subtype_iter())

        new_key = {}
        if self.cur_level == other.cur_level:
            for lbl in self_dict.keys() & other_dict.keys():

                if self_dict[lbl] is None:
                    new_key.update({(self.cur_level, lbl): other_dict[lbl]})

                elif other_dict[lbl] is None:
                    new_key.update({(self.cur_level, lbl): self_dict[lbl]})

                else:
                    new_ch = self_dict[lbl] & other_dict[lbl]

                    if not new_ch.is_empty():
                        new_key.update({(self.cur_level, lbl): new_ch})

        #TODO: better rules mismatching annotation levels
        elif other.cur_level in self.get_levels():
            for lbl in self_dict.keys():

                if self_dict[lbl] is None:
                    new_key.update({(self.cur_level, lbl): other})

                else:
                    new_ch = self_dict[lbl] & other

                    if not new_ch.is_empty():
                        new_key.update({(self.cur_level, lbl): new_ch})

        else:
            for lbl in self_dict.keys():

                if self_dict[lbl] is None:
                    new_key.update({(self.cur_level, lbl): other})

                else:
                    new_ch = other & self_dict[lbl]

                    if not new_ch.is_empty():
                        new_key.update({(self.cur_level, lbl): new_ch})

        return MuType(new_key)

    def __add__(self, other):
        """The sum of MuTypes yields the type where both MuTypes appear."""
        if not isinstance(other, MuType):
            return NotImplemented

        return MutComb([self, other])

    def is_supertype(self, other):
        """Checks if one MuType (non-strictly) contains another MuType."""
        if not isinstance(other, MuType):
            return NotImplemented

        # the empty null set cannot be the supertype of any other MuType
        if self.is_empty():
            return False

        # the empty null set is a subtype of every other non-empty MuType
        if other.is_empty():
            return True

        # gets the unique label:subtype pairs in each MuType
        self_dict = dict(self.subtype_iter())
        other_dict = dict(other.subtype_iter())

        # a MuType cannot be a supertype of another MuType unless they are on
        # the same mutation property level and its category labels are a
        # superset of the others'
        if self.cur_level == other.cur_level:
            if self_dict.keys() >= other_dict.keys():

                for k in (self_dict.keys() & other_dict.keys()):
                    if self_dict[k] is not None:

                        if other_dict[k] is None:
                            return False
                        elif not self_dict[k].is_supertype(other_dict[k]):
                            return False

            else:
                return False

        else:
            return False

        return True

    def __sub__(self, other):
        """Subtracts one MuType from another."""
        if not isinstance(other, MuType):
            return NotImplemented

        if self.is_empty() or other.is_empty():
            return self

        # gets the unique label:subtype pairs in each MuType
        self_dict = dict(self.subtype_iter())
        other_dict = dict(other.subtype_iter())

        if self.cur_level == other.cur_level:
            new_key = {(self.cur_level, lbl): self_dict[lbl]
                       for lbl in self_dict.keys() - other_dict.keys()}

            for lbl in self_dict.keys() & other_dict.keys():
                if other_dict[lbl] is not None:

                    if self_dict[lbl] is not None:
                        sub_val = self_dict[lbl] - other_dict[lbl]

                        if not sub_val.is_empty():
                            new_key.update({(self.cur_level, lbl): sub_val})

                    else:
                        new_key.update({(self.cur_level, lbl): None})

        else:
            raise ValueError("Cannot subtract MuType with mutation level {} "
                             "from MuType with mutation level {}!".format(
                                other.cur_level, self.cur_level))

        return MuType(new_key)

    def get_samples(self, *mtrees):
        """Finds the samples carrying this mutation in a collection of trees.

        Returns:
             samps (set)

        """
        use_indx = 0

        # if more than one mutation tree is given, find the first tree that
        # has all of the annotation levels contained in this mutation type
        if len(mtrees) > 1:
            for i, mtree in enumerate(mtrees):
                if mtree.match_levels(self):
                    use_indx = i
                    break

            else:
                raise ValueError("No trees found in this collection whose "
                                 "levels match `{}` !".format(self))

        # if this MuType has the same mutation level as the MuTree...
        samps = set()
        if self.cur_level == mtrees[use_indx].mut_level:
            tree_nms = {nm for nm, _ in mtrees[use_indx]}

            # ...find the mutation entries in the MuTree that match the
            # mutation entries in the MuType
            for lbl, tp in self.subtype_iter():
                if lbl in tree_nms:

                    # if both the subtype and the subtree for this entry are
                    # leaf nodes, add the samples on tree leaf...
                    if isinstance(mtrees[use_indx][lbl], dict):
                        if tp is None:
                            samps |= set(mtrees[use_indx][lbl])

                        else:
                            raise ValueError(
                                "The given tree cannot be used to retrieve "
                                "samples for `{}` !".format(tp)
                                )

                    # ...otherwise, recurse farther down the tree
                    elif tp is None:
                        samps |= mtrees[use_indx][lbl].get_samples()
                    else:
                        samps |= tp.get_samples(mtrees[use_indx][lbl])

        else:
            for _, branch in mtrees[use_indx]:
                if (not isinstance(branch, dict)
                        and branch.match_levels(self)):
                    samps |= self.get_samples(branch)

        return samps

    def get_leaf_annot(self, mtree, ant_flds):
        ant_dict = dict()

        if self.cur_level == mtree.mut_level:
            for (nm, mut), (lbl, tp) in product(mtree, self.subtype_iter()):
                if lbl == nm:

                    if hasattr(mut, 'get_samples'):
                        if tp is None:
                            lf_ant = mut.get_leaf_annot(ant_flds)
                        else:
                            lf_ant = tp.get_leaf_annot(mut, ant_flds)

                    elif isinstance(mut, dict):
                        lf_ant = {samp: {fld: ant[fld] for fld in ant_flds}
                                  for samp, ant in mut.items()}

                    else:
                        raise ValueError

                    ant_dict = {
                        **{samp: lf_ant[samp]
                           for samp in lf_ant.keys() - ant_dict.keys()},
                        **{samp: ant_dict[samp]
                           for samp in ant_dict.keys() - lf_ant.keys()},
                        **{samp: {ant_fld: (ant_dict[samp][ant_fld]
                                            + lf_ant[samp][ant_fld])
                                  for ant_fld in ant_flds}
                           for samp in ant_dict.keys() & lf_ant.keys()}
                        }

        else:
            for _, mut in mtree:
                if (hasattr(mut, 'get_levels')
                        and mut.get_levels() & self.get_levels()):
                    lf_ant = self.get_leaf_annot(mut, ant_flds)

                    ant_dict = {
                        **{samp: lf_ant[samp]
                           for samp in lf_ant.keys() - ant_dict.keys()},
                        **{samp: ant_dict[samp]
                           for samp in ant_dict.keys() - lf_ant.keys()},
                        **{samp: {ant_fld: (ant_dict[samp][ant_fld]
                                            + lf_ant[samp][ant_fld])
                                  for ant_fld in ant_flds}
                           for samp in ant_dict.keys() & lf_ant.keys()}
                        }

        return ant_dict

    def invert(self, mtree):
        """Gets the MuType of mutations in a MuTree but not in this MuType.

        Args:
            mtree (MuTree): A hierarchy of mutations present in a cohort.

        Returns:
            inv_mtype (MuType)

        """
        return MuType(mtree.allkey()) - self


class MutComb(object):
    """A combination of mutations simultaenously present in a sample.

    """

    def __new__(cls, *mtypes, not_mtype=None):
        if not all(isinstance(mtype, MuType) for mtype in mtypes):
            raise TypeError(
                "A MutComb object must be a combination of MuTypes!")

        mtypes = list(mtypes)
        obj = super().__new__(cls)

        # removes overlap between the given mutations
        for i, j in combn(range(len(mtypes)), r=2):
            if mtypes[i] is not None and mtypes[j] is not None:
                if mtypes[i].is_supertype(mtypes[j]):
                    mtypes[i] = None
                elif mtypes[j].is_supertype(mtypes[i]):
                    mtypes[j] = None

        # removes mutations that are covered by other given mutations
        mtypes = [mtype for mtype in mtypes if mtype is not None]
        intrx_mtype = reduce(and_, mtypes)

        if not_mtype is not None:
            mtypes = [mtype - not_mtype
                      if mtype.get_levels() == not_mtype.get_levels()
                      else mtype for mtype in mtypes]

            if not_mtype.get_levels() == intrx_mtype.get_levels():
                not_mtype -= intrx_mtype

            if not_mtype.is_empty():
                not_mtype = None

        # removes mutations that are covered by other given mutations
        mtypes = [mtype for mtype in mtypes if not mtype.is_empty()]

        # if only one unique mutation was given, return that mutation...
        if mtypes:
            if len(mtypes) == 1 and not_mtype is None:
                return mtypes[0]

            # ...otherwise, return the combination of the given mutations
            else:
                obj.mtypes = frozenset(mtypes)
                obj.not_mtype = not_mtype
                return obj

        else:
            return MuType({})

    def is_empty(self):
        return False

    def __eq__(self, other):
        """Checks if one MutComb is equal to another."""

        if not isinstance(other, MutComb):
            eq = False

        else:
            eq = self.mtypes == other.mtypes
            eq &= self.not_mtype == other.not_mtype

        return eq

    def mtype_apply(self, each_fx, comb_fx):
        each_list = [each_fx(mtype) for mtype in self.mtypes]
        return reduce(comb_fx, each_list)

    def __repr__(self):
        out_str = self.mtype_apply(repr, lambda x, y: x + ' AND ' + y)
        if self.not_mtype is not None:
            out_str += ' WITHOUT ' + repr(self.not_mtype)

        return out_str

    def __str__(self):
        out_str = self.mtype_apply(str, lambda x, y: x + ' & ' + y)
        if self.not_mtype is not None:
            out_str += ' ~ ' + str(self.not_mtype)

        return out_str

    def __hash__(self):
        value = 0x213129

        return value + self.mtype_apply(
            hash,
            lambda x, y: (x ^ y + eval(hex((int(value) * 1003)
                                           & 0xFFFFFFFF)[:-1]))
            )

    def __and__(self, other):
        ovlp_mcomb = MuType({})

        if len(self.mtypes) == len(other.mtypes):
            ovlp_mat = [[not (mtype & other_mtype).is_empty()
                         for other_mtype in other.mtypes]
                        for mtype in self.mtypes]

            if all(any(ovlp) for ovlp in ovlp_mat):
                ovlp_mtypes = []
                sort_ovlp = sorted(zip(self.mtypes, ovlp_mat),
                                   key=lambda x: sum(x[1]))

                while sort_ovlp:
                    ovlp_mtypes += [[]]
                    cur_mtype, cur_ovlp = sort_ovlp.pop(0)

                    for i, (other_mtype, ovlp_stat) in enumerate(
                            zip(other.mtypes, cur_ovlp)):
                        if ovlp_stat:
                            ovlp_mtypes[-1] += [cur_mtype & other_mtype]

                            for j in range(len(sort_ovlp)):
                                sort_ovlp[j][1][i] = False

                comb_prod = tuple(product(*ovlp_mtypes))
                if len(comb_prod) == 1:
                    ovlp_mcomb = MutComb(*comb_prod[0])
                elif len(comb_prod) > 1:
                    ovlp_mcomb = tuple(MutComb(*prod) for prod in comb_prod)

        return ovlp_mcomb

    def __sub__(self, other):
        if isinstance(other, MuType):
            return MutComb([mtype - other for mtype in self.mtypes])
        else:
            return self

    def get_samples(self, *mtrees):
        samps = self.mtype_apply(
            lambda mtype: mtype.get_samples(*mtrees), and_)

        if self.not_mtype is not None:
            samps -= self.not_mtype.get_samples(*mtrees)

        return samps

