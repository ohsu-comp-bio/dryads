
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
            tuple(ix[np.argmin([typ[i] for i in ix])]
                  for ix, typ in zip(indx, types))
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

    Args:
        annot_file (str): A .gtf file, downloaded from eg.
                          www.gencodegenes.org/releases/22.html

    Returns:
        gn_annot (dict): Dictionary with keys corresponding to Ensembl gene
                         IDs and values consisting of dicts with
                         annotation fields.

    """
    annot = pd.read_csv(annot_file, usecols=[0, 2, 3, 4, 6, 8],
                        names=['Chr', 'Type', 'Start', 'End',
                               'Strand', 'Info'],
                        sep='\t', header=None, comment='#')

    # remove annotation records that are non-relevant or on sex chromosomes
    chroms_use = ['chr' + str(i+1) for i in range(22)]
    annot = annot.loc[annot['Type'].isin(['gene', 'transcript',
                                          'exon', 'UTR'])
                      & annot['Chr'].isin(chroms_use), :]

    # parse the annotation field of each record into a table of fields
    info_flds = pd.DataFrame.from_records(
        annot['Info'].str.split('; ').apply(
            lambda flds: dict(fld.split(' ') for fld in flds))
        ).applymap(lambda x: x.replace('"', '') if isinstance(x, str) else x)

    # remove version numbers from Ensembl IDs of genes and transcripts
    info_flds['gene_id'] = info_flds.gene_id.str.replace('\.[0-9]+', '')
    info_flds['transcript_id'] = info_flds.transcript_id.str.replace(
        '\.[0-9]+', '')
    info_flds['exon_number'] = pd.to_numeric(info_flds['exon_number'],
                                             downcast='integer')

    # merge record annotations, remove non-protein-coding genes and
    # transcripts, create table containing just the unique gene records
    info_df = pd.concat([annot.iloc[:, :5].reset_index(drop=True),
                         info_flds.reset_index(drop=True)],
                        axis=1)
    info_df = info_df[info_df.transcript_type == 'protein_coding']
    gene_df = info_df[info_df.Type == 'gene'].set_index('gene_id')

    # group transcript records according to parent gene, transform gene
    # records into a dictionary
    tx_groups = info_df[(info_df.Type == 'transcript')
                        & info_df.gene_id.isin(gene_df.index)].groupby(
                            ['gene_id'])
    gn_annot = {gn: dict(recs[['Chr', 'Start', 'End', 'Strand', 'gene_name']])
                for gn, recs in gene_df.iterrows()}

    # insert the transcripts for each gene into the gene record dictionary
    for gn, tx_df in tx_groups:
        gn_annot[gn]['Transcripts'] = {
            tx: dict(recs[['Start', 'End', 'transcript_name']])
            for tx, recs in tx_df.set_index('transcript_id').iterrows()
            }

    # likewise, group exon records according to parent gene
    exn_groups = info_df[info_df.Type.isin(['exon', 'UTR'])
                         & info_df.gene_id.isin(gene_df.index)].groupby(
                             ['gene_id'])

    # insert the exons for each gene into the gene dictionary, using the first
    # transcript of each gene as the genomic reference
    for gn, exn_df in exn_groups:
        use_tx = '{}-001'.format(gn_annot[gn]['gene_name'])
        use_exns = exn_df[exn_df.transcript_name == use_tx].sort_values(
            by=['Start', 'Type'], ascending=[True, False])

        use_exns = use_exns.reset_index(drop=True)
        gn_annot[gn]['Exons'] = use_exns[
            use_exns.Type == 'exon'].sort_values(by='exon_number')[
                ['Start', 'End', 'exon_id']].apply(dict, axis=1).tolist()

        for i in range(use_exns.shape[0]):
            if use_exns['Type'][i] == 'UTR':
                utr_exn = int(use_exns['exon_number'][cur_exon]) - 1

                if 'UTR' in gn_annot[gn]['Exons'][utr_exn]:
                    gn_annot[gn]['Exons'][utr_exn]['UTR'] += [use_exns[
                        ['Start', 'End']].iloc[i].to_dict()]

                else:
                    gn_annot[gn]['Exons'][utr_exn]['UTR'] = [use_exns[
                        ['Start', 'End']].iloc[i].to_dict()]

            else:
                cur_exon = i

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

