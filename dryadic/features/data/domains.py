
import os
import pandas as pd


def get_protein_domains(domain_dir, domain_lbl):
    domain_file = os.path.join(domain_dir,
                               '{}_to_gene.txt.gz'.format(domain_lbl))
    
    domain_data = pd.read_csv(domain_file, sep='\t')
    use_columns = pd.Series(["Gene", "Transcript", "", "", ""])

    use_columns[domain_data.columns.str.contains(
        'domain[s]? ID')] = "DomainID"
    use_columns[domain_data.columns.str.contains(
        'domain[s]? start')] = "DomainStart"
    use_columns[domain_data.columns.str.contains(
        'domain[s]? end')] = "DomainEnd"

    assert not (use_columns == "").any()
    domain_data.columns = use_columns

    return domain_data

