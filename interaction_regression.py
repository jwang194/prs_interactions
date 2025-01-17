import os
import numpy as np
import pandas as pd

from functools import reduce
from itertools import combinations
from scipy.special import binom
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split,cross_validate

# Let this be the working directory

wd = ''
os.chdir(wd)


# Let this be the name of a tab-delimited data file (with header) containing the following columns:
#   ID: individual level identifiers
#   ER+: PRS for ER+
#   ER-: PRS for ER-
#   TPN: PRS for triple-negative
#   Unknown: PRS for unknown subtype
#   Outcome: Case/control status

df_path = ''
predictors = ['ER+','ER-','TPN','Unknown']

df = pd.read_csv(df_path,sep='\t')
full_X = df[predictors]
y = df['Outcome']                                                

# We'll need test data for the boundary problem later
test_prop = 0.2
full_X,test_X,y,test_y = train_test_split(full_X,y,test_size=test_prop)


# We'll test every combination of subtype PRS.
# This code produces the powerset of subtypes

from itertools import chain, combinations

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(1,len(s)+1))

subsets = list(powerset(predictors))


accuracies = []
models = []
# For each subset, fit a 5-fold CV model with dependent variables:
# the subset and all pairwise interactions

nfolds = 5
for s in subsets:
    s = list(s)
    X = full_X[s]
    X.insert(0,'Intercept',np.ones(X.shape[0]))
    pairs = list(combinations(s,2))
    columns = ['x'.join(p) for p in pairs]

    # Compute the pairwise products of all subtypes in the subset
    X[columns] = pd.DataFrame([X[a]*X[b] for a,b in pairs]).T

    # We'll use the L2 penalty
    lr = LogisticRegression(penalty='l2',C=1)
    results = cross_validate(lr,
                             X,
                             y,
                             cv=nfolds,
                             scoring='roc_auc',
                             return_estimator=True)
    scores = results['test_score']

    # Compute the average ROC and standard deviations across all folds
    accuracies.append([scores.mean(),scores.std()])
    best_C = results['estimator'][np.argmax(scores)].C

    # Fit a new model to the full data set using the best \lambda
    lr = LogisticRegression(penalty='l2',C=best_C)
    lr.fit(X,y)
    models.append(lr)

    
# Let's save accuracies and model weights;
# Model weights will be in lexicographic order

accuracies = np.array(accuracies)
index_order = ['Intercept',
               'ER+',
               'ER-',
               'ER+xER-',
               'TPN',
               'ER+xTPN',
               'ER-xTPN',
               'Unknown',
               'ER+xUnknown',
               'ER-xUnknown',
               'TPNxUnknown']

index_dict = {s:i for s,i in zip(index_order,range(len(index_order)))}

# The weight matrix is of dimension (2^#predictors-1, #predictors + (#predictors choose 2))
weights = np.zeros((2**len(predictors)-1,len(index_order)))
for i,s,m in zip(range(len(subsets)),subsets,models):
    pairs = ['x'.join(p) for p in combinations(s,2)]
    variables = list(s) + pairs
    for w,v in zip(m.coef_[0],variables):
        weights[i,index_dict[v]] = w

output = np.hstack((accuracies,weights))
output = pd.DataFrame(output,columns = ['Mean AUC','Standard Deviation of AUC'] + index_order)

# Let this be the output directory for the accuracies and weights
odir = '' 
output.to_csv(odir,sep='\t',index=False)


# Now that we have fitted models, we can produce:
# 1. Per-SNP and per-SNP-pair odds ratios
# 2. Decision boundaries 

# 1. Per-SNP and per-SNP-pair odds ratios

# Note that this is quite memory- and time-intensive, since we have a quadratic number of SNP-pairs
# I include an option to restrict the joint PRS to SNPs only, but this will decrease performance
# proportionally to the coefficients of the interaction terms
include_pairs = True

# Let this be a path to a tab-delimited data file (with header and index) containing:
#   1. One column for each SNP
#   2. One row for for each subtype PRS
#   3. In each entry, effect sizes (log-odds ratios) for the corresponding SNP-PRS pair

snp_dir = ''
snps = pd.read_csv(snp_dir,sep='\t',index_col=0)
M = snps.shape[1]
flatten_indices = np.triu_indices(M)

prs_pairs = ['x'.join(p) for p in combinations(snps.index,2)]
snp_pairs = np.char.add(np.char.add(np.array(snps.columns)[:, np.newaxis], "x"), np.array(snps.columns))
snp_pairs = snp_pairs[flatten_indices]
# Compute a matrix of SNP-pair effects for each product of PRSs then merge effect matrices...
if include_pairs:
    pairs = pd.DataFrame(np.zeros((len(prs_pairs),len(snp_pairs))))
    pairs = pd.DataFrame(np.zeros((len(prs_pairs),len(snp_pairs))))
    pairs.index = prs_pairs
    pairs.columns = snp_pairs
    for pp in prs_pairs:
        prs1,prs2 = pp.split('x')
        pairs.loc[pp] = np.outer(snps.loc[prs1],snps.loc[prs2])[flatten_indices]
    # Collect SNP-pair and single-SNP effects into block matrix form
    A = np.block([
            [snps,                       np.zeros((snps.shape[0],pairs.shape[1]))],
            [np.zeros((pairs.shape[0],snps.shape[1])),                      pairs]
    ])
    snp_weights = pd.DataFrame(A)
    snp_weights.index = list(snps.index) + list(pairs.index)
    snp_weights.columns = list(snps.columns) + list(pairs.columns)
# ...Or simply use single-SNP effects
else:
    snp_weights = snps

# This function will compute effect sizes (log-odds ratios) for each SNP on a single ensemble PRS 
# It uses the trivial application of distributivity to \alpha_1(\sum \beta_{1,i}*x_i) + 
# \alpha_2(\sum \beta_{2,i}*x_i) + \alpha_3(\sum \sum \beta_{1,i}*\beta_{2,j}*x_i*x_j)
def distribute_coefficients(weights):
    if include_pairs:
        return snp_weights.T @ weights
    else:
        return snp_weights.T @ weights[:len(predictors)]
    

# 2. Decision boundaries

# Recall that test_X and test_y were pulled from the data above
expanded_X = test_X.copy(deep=True)
for i in index_order:
    expanded_X[i] = reduce(lambda x,y: x*y,[X[c] for c in i.split('x')])

expanded_X.insert(0,'Intercept',np.ones(expanded_X.shape[0]))

# This function will determine a level set for one set of weights 
def boundary(weights,percent):
    # Compute the correct level set to select top percent% of women
    scores = expanded_X @ weights
    level_set = np.quantile(scores,1-percent)
    return level_set 

