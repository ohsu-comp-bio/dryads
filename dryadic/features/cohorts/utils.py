
import numpy as np
import pandas as pd

from re import sub as gsub
from functools import reduce
from operator import and_


def match_tcga_samples(*samples):
    """Finds the tumour samples common between lists of TCGA barcodes.

    Args:
        samples (:obj:`list` of :obj:`str`)

    Returns:
        samps_match (:obj:`list` of :obj:`tuple`)

    """
    samp_lists = [sorted(set(samps)) for samps in samples]

    # parse the sample barcodes into their constituent parts
    parsed_samps = [[samp.split('-') for samp in samps]
                    for samps in samp_lists]

    # get the names of the individuals associated with each sample
    partics = [['-'.join(prs[:3]) for prs in parsed]
               for parsed in parsed_samps]

    # get the type of each sample (tumour, control, etc.) and the
    # vial it was tested in
    types = [np.array([int(prs[3][:2]) for prs in parsed])
             for parsed in parsed_samps]
    vials = [[prs[3][-1] for prs in parsed] for parsed in parsed_samps]

    # find the individuals with primary tumour samples in both lists
    partic_use = sorted(reduce(
        and_,
        [set(prt for prt, tp in zip(partic, typ) if tp < 10)
         for partic, typ in zip(partics, types)]
        ))

    # find the positions of the samples associated with these shared
    # individuals in the original lists of samples
    partics_indx = [[[i for i, prt in enumerate(partic) if prt == use_prt]
                     for partic in partics]
                    for use_prt in partic_use]

    # match the samples of the individuals with only one sample in each list
    samps_match = [
        (partics[0][indx[0][0]],
         tuple(samp_list[ix[0]] for samp_list, ix in zip(samp_lists, indx)))
         for indx in partics_indx if all(len(ix) == 1 for ix in indx)
        ]

    # for individuals with more than one sample in at least one of the two
    # lists, find the sample in each list closest to the primary tumour type
    if len(partics_indx) > len(samps_match):
        choose_indx = [
            tuple(ix[np.argmin(typ)] for ix, typ in zip(indx, types))
            for indx in partics_indx if any(len(ix) > 1 for ix in indx)
            ]

        samps_match += [
            (partics[0][chs[0]],
             tuple(samp_list[ix] for samp_list, ix in zip(samp_lists, chs)))
            for chs in choose_indx
            ]

    match_dict = [{old_samps[i]: new_samp
                   for new_samp, old_samps in samps_match}
                  for i in range(len(samples))]

    return match_dict


def get_gencode(annot_file):
    """Gets annotation data for protein-coding genes on non-sex
       chromosomes from a Gencode file.

    Returns
    -------
    annot : dict
        Dictionary with keys corresponding to Ensembl gene IDs and values
        consisting of dicts with annotation fields.
    """
    annot = pd.read_csv(annot_file, usecols=[0, 2, 3, 4, 8],
                        names=['Chr', 'Type', 'Start', 'End', 'Info'],
                        sep='\t', header=None, comment='#')

    # filter out annotation records that aren't
    # protein-coding genes on non-sex chromosomes
    chroms_use = ['chr' + str(i+1) for i in range(22)]
    annot = annot.loc[annot['Type'] == 'gene', ]
    chr_indx = np.array([chrom in chroms_use for chrom in annot['Chr']])
    annot = annot.loc[chr_indx, ]

    # parse the info field to get each gene's annotation data
    gn_annot = {gsub('\.[0-9]+', '', z['gene_id']).replace('"', ''): z
                for z in [dict([['chr', an[0]]]
                               + [['Start', an[2]]] + [['End', an[3]]] +
                               [y for y in [x.split(' ')
                                            for x in an[4].split('; ')]
                                if len(y) == 2])
                          for an in annot.values]
                if z['gene_type'] == '"protein_coding"'}

    gn_annot = {g: {k: v.replace('"', '') if isinstance(v, str) else v
                    for k, v in annt.items()}
                for g, annt in gn_annot.items()}

    return gn_annot


def log_norm(data_mat):
    """Log-normalizes a dataset, usually RNA-seq expression.

    Puts a matrix of continuous values into log-space after adding
    a constant derived from the smallest non-zero value.

    Args:
        data_mat (:obj:`np.array` of :obj:`float`,
                  shape = [n_samples, n_features])

    Returns:
        norm_mat (:obj:`np.array` of :obj:`float`,
                  shape = [n_samples, n_features])

    Examples:
        >>> norm_expr = log_norm(np.array([[1.0, 0], [2.0, 8.0]]))
        >>> print(norm_expr)
                [[ 0.5849625 , -1.],
                 [ 1.32192809,  3.08746284]]

    """
    log_add = np.nanmin(data_mat[data_mat > 0]) * 0.5
    norm_mat = np.log2(data_mat + log_add)

    return norm_mat

