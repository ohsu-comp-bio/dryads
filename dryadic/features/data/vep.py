
import os
import subprocess
from io import StringIO
import pandas as pd


field_map = {
    'Uploaded_variation': "Sample", 'Allele': "VarAllele", 'Gene': "ENSGene",
    'Protein_position': "Position",
    'SYMBOL': "Gene", 'EXON': "Exon", 'INTRON': "Intron",
    'POLYPHEN': "PolyPhen", 'DOMAINS': "Domains", 'VARIANT_CLASS': "Class",
    }

# default pick order copied from uswest.ensembl.org/info/genome/variation
# /prediction/predicted_data.html#consequences on April 21, 2020
consequences = [
    'transcript_ablation', 'splice_acceptor_variant', 'splice_donor_variant',
    'stop_gained', 'frameshift_variant', 'stop_lost', 'start_lost',
    'transcript_amplification', 'inframe_insertion', 'inframe_deletion',
    'missense_variant', 'protein_altering_variant', 'splice_region_variant',
    'incomplete_terminal_codon_variant', 'start_retained_variant',
    'stop_retained_variant', 'synonymous_variant', 'coding_sequence_variant',
    'mature_miRNA_variant', '5_prime_UTR_variant', '3_prime_UTR_variant',
    'non_coding_transcript_exon_variant', 'intron_variant',
    'NMD_transcript_variant', 'non_coding_transcript_variant',
    'upstream_gene_variant', 'downstream_gene_variant', 'TFBS_ablation',
    'TFBS_amplification', 'TF_binding_site_variant',
    'regulatory_region_ablation', 'regulatory_region_amplification',
    'feature_elongation', 'regulatory_region_variant', 'feature_truncation',
    'intergenic_variant'
    ]


class VariantEffectPredictorError(Exception):
    pass


def process_variants(var_df, out_fields=None, cache_dir=None, temp_dir=None,
                     species='homo_sapiens', assembly='GRCh37',
                     distance=5000, flag_pick=False,
                     consequence_choose='sort', forks=None,
                     buffer_size=1e4, vep_version=99, update_cache=False):

    if out_fields is None:
        out_fields = ["Gene"]

    if cache_dir is None:
        cache_dir = os.path.join(os.getcwd(), "vep-cache")
        print("no cache directory specified for VEP, defaulting "
              "to:\n\t{}".format(cache_dir))

    if temp_dir is None:
        temp_dir = os.path.join(os.getcwd(), "vep-temp")
        print("no directory for intermediate files specified, defaulting "
              "to:\n\t{}".format(temp_dir))

    if isinstance(distance, int):
        distance = str(distance)

    if update_cache:
        subprocess.run(["mkdir", "-p", cache_dir], check=True)

        subprocess.run(
            ["vep_install", "-c", cache_dir, "-a", "cf", "-s", species,
             "-y", assembly, "--VERSION", str(vep_version), "--CONVERT"],
            check=True
            )

    var_list = pd.DataFrame({
        'Chr': var_df.Chr, 'Start': var_df.Start, 'End': var_df.End,
        'Allele': ['/'.join([ref, tmr])
                   for ref, tmr in zip(var_df.RefAllele.values,
                                       var_df.VarAllele.values)],
        'Strand': var_df.Strand, 'Sample': var_df.Sample
        }).sort_values(by=['Chr', 'Start', 'End'])

    ins_indx = var_df.RefAllele == '-'
    pos_dummy = var_list.loc[ins_indx, 'Start']
    var_list.loc[ins_indx, 'Start'] = var_list.loc[ins_indx, 'End']
    var_list.loc[ins_indx, 'End'] = pos_dummy

    subprocess.run(["mkdir", "-p", temp_dir], check=True)
    var_file = os.path.join(temp_dir, "vars.txt")
    var_list.to_csv(var_file, sep='\t', index=False, header=False)

    vep_subm = ["vep", "-i", var_file, "-o", "STDOUT", "-a", assembly,
                "-s", species, "--cache", "--dir_cache", cache_dir,
                "--distance", distance, "--no_stats", "--tab",
                "--buffer_size", format(buffer_size, '.0f'), "--offline"]

    if forks is not None:
        if isinstance(forks, int):
            vep_subm += ["--fork", str(forks)]

        else:
            raise ValueError("`forks` argument must be an integer describing "
                             "a number of compute cores!")

    # these fields are both in the default VEP output and are to be
    # returned by this wrapper no matter what arguments are given
    vep_fields = ["Uploaded_variation", "Feature"]

    # these fields are in the default VEP output but are only returned by this
    # wrapper if they are explicitly requested
    if 'Location' in out_fields:
        vep_fields += ["Location"]
    if 'VarAllele' in out_fields:
        vep_fields += ["Allele"]
    if 'ENSGene' in out_fields:
        vep_fields += ["Gene"]
    if 'Consequence' in out_fields:
        vep_fields += ["Consequence"]
    if 'Position' in out_fields:
        vep_fields += ["Protein_position"]

    # these fields are also optionally returned but do not appear
    # in the default VEP output
    if 'Gene' in out_fields:
        vep_subm += ["--symbol"]
        vep_fields += ["SYMBOL"]
    if 'Canonical' in out_fields:
        vep_subm += ["--canonical"]
        vep_fields += ["CANONICAL"]

    if 'Exon' in out_fields or 'Intron' in out_fields:
        vep_subm += ["--numbers"]
    if 'Exon' in out_fields:
        vep_fields += ["EXON"]
    if 'Intron' in out_fields:
        vep_fields += ["INTRON"]

    if 'HGVSc' in out_fields or 'HGVSp' in out_fields:
        vep_subm += ["--hgvs"]
    if 'HGVSp' in out_fields:
        vep_fields += ["HGVSp"]
    if 'HGVSc' in out_fields:
        vep_fields += ["HGVSc"]

    if 'Domains' in out_fields:
        vep_subm += ["--domains"]
        vep_fields += ["DOMAINS"]

    if 'Class' in out_fields:
        vep_subm += ["--variant_class"]
        vep_fields += ["VARIANT_CLASS"]

    if any('PolyPhen' in fld for fld in out_fields):
        vep_fields += ["POLYPHEN"]
    if 'PolyPhen_Score' in out_fields:
        vep_subm += ["--polyphen", "s"]
    elif 'PolyPhen_Term' in out_fields:
        vep_subm += ["--polyphen", "p"]
    elif 'PolyPhen' in out_fields:
        vep_subm += ["--polyphen", "b"]

    if 'SIFT_Score' in out_fields:
        vep_subm += ["--sift", "s"]
    elif 'SIFT_Term' in out_fields:
        vep_subm += ["--sift", "p"]
    elif 'SIFT' in out_fields:
        vep_subm += ["--sift", "b"]

    if flag_pick:
        vep_subm += ["--flag_pick_allele_gene"]
        vep_fields += ["PICK"]

    # run the VEP command line tool using the given parameters
    vep_subm += ["--fields", ','.join(vep_fields)]
    vep_resp = subprocess.run(vep_subm,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # handle errors thrown by VEP, other than Perl flow operator issues
    vep_err = str(vep_resp.stderr, 'utf-8').split('\n')
    if len(vep_err) > 1 and any(len(e) > 0 for e in vep_err[1:]):
        raise VariantEffectPredictorError('\n'.join(vep_err[1:]))

    # parse the output returned by VEP, find where the output header ends
    vep_out = str(vep_resp.stdout, 'utf-8').split('\n')
    for i, s in enumerate(vep_out):
        if s[:2] != '##':
            break

    vep_out[i] = vep_out[i].split('#')[1]
    mut_df = pd.read_csv(StringIO('\n'.join(vep_out[i:])), sep='\t').rename(
        columns=field_map)

    if consequence_choose is not None and 'Consequence' in out_fields:
        conseq_lists = mut_df.Consequence.str.split(',')

        if consequence_choose == 'sort':
            mut_df = mut_df.assign(Consequence=conseq_lists.apply(
                lambda vals: ','.join(sorted(
                    vals, key=lambda val: consequences.index(val)))
                ))

        elif consequence_choose == 'pick':
            mut_df = mut_df.assign(Consequence=conseq_lists.apply(
                lambda vals: sorted(
                    vals, key=lambda val: consequences.index(val))[0]
                ))

        else:
            raise ValueError("Unrecognized value for `consequence_choose`, "
                             "must be one of {'sort', 'pick'}!")

    if 'Domains' in out_fields:
        domn_data = mut_df.Domains.str.split(',').apply(
            lambda dmns: [dmn.split(':') for dmn in dmns])

        mut_df = pd.concat([
            mut_df, pd.DataFrame(domn_data.apply(
                lambda dmns: {dmn[0]: ':'.join(dmn[1:])
                              for dmn in dmns if dmn[0] != '-'}
                ).tolist()).fillna('none').rename(
                    columns=lambda dmn: dmn.replace('_', '-'))
            ], axis=1)

    return mut_df

