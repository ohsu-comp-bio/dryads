
base_model = '''
    data {
        int<lower=1> N;         // number of samples
        int<lower=1> T;         // number of transcript features
        int<lower=1> G;         // number of genetic features
 
        matrix[N, T] expr;      // RNA-seq expression values
        int<lower=0, upper=1> mut[N];       // mutation status
        int<lower=1, upper=T> tx_indx[G];   // transcripts per gene
 
        real<lower=0> alpha;    // regularization coefficient
        real<lower=0> gamma;    // Dirichlet distribution prior
    }
 
    parameters {
        real intercept;

        vector[G] gn_wghts;
        vector[T] gn_wghts_use;
        vector<lower=0, upper=1>[T] tx_wghts;
    }

    model {
        int pos = 1;

        for (g in 1:G) {
            vector[tx_indx[g]] pi = segment(tx_wghts, pos, tx_indx[g]);
            vector[tx_indx[g]] tx_gn = segment(gn_wghts_use, pos, tx_indx[g]);

            pi = pi / sum(pi);
            for (t in 1:tx_indx[g]) {
                tx_gn[t] = gn_wghts[g];
            }

            pos = pos + tx_indx[g];
        }

        intercept ~ normal(0, 1.0);
        gn_wghts ~ normal(0, alpha);
        target += exponential_lpdf(tx_wghts | gamma);

        mut ~ bernoulli_logit(intercept + expr * (tx_wghts .* gn_wghts_use));
    }
'''


cauchy_model = '''
    data {
        int<lower=1> N;         // number of samples
        int<lower=1> T;         // number of transcript features
        int<lower=1> G;         // number of genetic features
 
        matrix[N, T] expr;      // RNA-seq expression values
        int<lower=0, upper=1> mut[N];       // mutation status
        int<lower=1, upper=T> tx_indx[G];   // transcripts per gene
 
        real<lower=0> alpha;    // regularization coefficient
        real<lower=0> gamma;    // Dirichlet distribution prior
    }
 
    parameters {
        real intercept;

        vector[G] gn_wghts;
        vector[T] gn_wghts_use;
        vector<lower=0, upper=1>[T] tx_wghts;
    }

    model {
        int pos = 1;

        for (g in 1:G) {
            vector[tx_indx[g]] pi = segment(tx_wghts, pos, tx_indx[g]);
            vector[tx_indx[g]] tx_gn = segment(gn_wghts_use, pos, tx_indx[g]);

            pi = pi / sum(pi);
            for (t in 1:tx_indx[g]) {
                tx_gn[t] = gn_wghts[g];
            }

            pos = pos + tx_indx[g];
        }

        intercept ~ normal(0, 1.0);
        gn_wghts ~ cauchy(0, alpha);
        target += exponential_lpdf(tx_wghts | gamma);

        mut ~ bernoulli_logit(intercept + expr * (tx_wghts .* gn_wghts_use));
    }
'''

