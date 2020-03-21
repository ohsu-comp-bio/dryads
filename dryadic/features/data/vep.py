
import os
import subprocess
from io import StringIO
import pandas as pd


field_map = {
    'Uploaded_variation': "Sample", 'SYMBOL': "Gene", 'Gene': "ENSGene",
    'Protein_position': "Position", 'EXON': "Exon", 'INTRON': "Intron",
    'POLYPHEN': "PolyPhen", 'DOMAINS': "Domains", 'VARIANT_CLASS': "Class",
    }


class VariantEffectPredictorError(Exception):
    pass


def process_variants(var_df, out_fields=None, cache_dir=None, temp_dir=None,
                     species='homo_sapiens', assembly='GRCh37',
                     flag_pick=False, forks=None, buffer_size=1e4,
                     vep_version=99, distance=5000, update_cache=False):

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
                "--buffer_size", format(buffer_size, '.0f'),
                "--distance", distance, "--no_stats", "--tab",
                "--symbol", "--canonical", "--offline"]

    if forks is not None and isinstance(forks, int):
        vep_subm += ["--fork", str(forks)]

    vep_fields = ["Uploaded_variation", "Feature", "Gene", "Consequence",
                  "Protein_position", "CANONICAL"]

    if 'Gene' in out_fields:
        vep_subm += ["--symbol"]
        vep_fields += ["SYMBOL"]

    if 'Exon' in out_fields or 'Intron' in out_fields:
        vep_subm += ["--numbers"]
    if 'Exon' in out_fields:
        vep_fields += ["EXON"]
    if 'Intron' in out_fields:
        vep_fields += ["INTRON"]

    if 'HGVSc' in out_fields or 'HGVSp' in out_fields:
        vep_subm += ["--hgvs"]
    if 'HGVSp' in out_fields:
        vep_fields += ['HGVSp']
    if 'HGVSc' in out_fields:
        vep_fields += ['HGVSc']

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
    vep_resp = subprocess.run(vep_subm + ["--numbers"],
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

