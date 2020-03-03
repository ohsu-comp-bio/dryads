
import os
import subprocess
from io import StringIO
import pandas as pd


field_map = {
    'SYMBOL': 'Gene', 'Gene': 'ENSGene', 'Protein_position': 'Position',
    "EXON": "Exon", "INTRON": "Intron", "POLYPHEN": "PolyPhen",
    }


def process_variants(mut_list, out_fields=None, species='homo_sapiens',
                     assembly='GRCh38', cache_dir=None, overwrite=False,
                     buffer_size=1e4, vep_version=99):

    if out_fields is None:
        out_fields = ["Gene"]

    if cache_dir is None:
        cache_dir = os.path.join(os.getcwd(), "vep-cache")
        print("no cache directory specified for VEP, defaulting "
              "to:\n\t{}".format(cache_dir))

    subprocess.run(["mkdir", "-p", cache_dir], check=True)
    cache_path = os.path.join(cache_dir, species,
                              '_'.join([str(vep_version), assembly]))

    if overwrite or not os.path.isdir(cache_path):
        subprocess.run(
            ["vep_install", "-c", cache_dir, "-a", "c", "-s", species,
             "-y", assembly, "--VERSION", str(vep_version), "--CONVERT"],
            check=True
            )

    vep_subm = ["vep", "--id", '\n'.join(mut_list), "-s", species,
                "-a", assembly, "-o", "STDOUT",
                "--cache", "--dir_cache", cache_dir,
                "--buffer_size", format(buffer_size, '.0f'),
                "--no_stats", "--tab", "--symbol"]

    vep_fields = ["Gene", "Consequence", "Protein_position"]

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

    # run the VEP command line tool using the given parameters
    vep_subm += ["--fields", ','.join(vep_fields)]
    vep_resp = subprocess.run(vep_subm + ["--numbers"],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # parse the output returned by VEP, find where the output header ends
    vep_out = str(vep_resp.stdout, 'utf-8').split('\n')
    for i, s in enumerate(vep_out):
        if s[:2] != '##':
            break

    vep_out[i] = vep_out[i].split('#')[1]
    mut_df = pd.read_csv(StringIO('\n'.join(vep_out[i:])), sep='\t')

    return mut_df.rename(columns=field_map)

