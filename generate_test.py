import numpy as np
import pandas as pd

from itertools import combinations

# Let N and M be the numbers of individuals and SNPs, respectively
N = 10**5
M = 10**4
# We'll use the same names as in the real data
names = ['ER+','ER-','TPN','Unknown']
P = len(names)

# We simulate genotypes based on the assumption of independent SNPs
alleles = 2
mafs = np.random.uniform(0,0.5,M)
genotypes = np.zeros((N,M))
for i in range(alleles):
    genotypes += np.random.rand(N,M) < mafs

genotypes -= genotypes.mean(0)
genotypes /= np.array([s if s!= 0 else 1 for s in genotypes.std(0)])

# Simulate effect sizes and compute PRS
betas = np.random.randn(M*P).reshape(M,P)
prs = pd.DataFrame(genotypes @ betas, columns=names)
prs /= prs.std()
prs -= prs.min()
pairs = list(combinations(names,2))
columns = ['x'.join(p) for p in pairs]
prs[columns] = pd.DataFrame([prs[a]*prs[b] for a,b in pairs]).T

# Simulate the outcome under a liability model with various marginal and joint thresholds
predictors = names + columns
liabilities = prs.copy(deep=True)
# We'll use substantially twice as much noise as signal in our liability
liabilities += np.sqrt(2)*np.random.randn(N*liabilities.shape[1]).reshape(liabilities.shape)
# Let's choose ER+,ER-,TPN,ER+xER-,ER+xTPN as our true thresholded liabilities
# We'll use 5% thresholds for the marginals, and pairwise 30% thresholds for the joints
thresholds = np.array([np.quantile(liabilities['ER+'],0.95),
                       np.quantile(liabilities['ER-'],0.95),
                       np.quantile(liabilities['TPN'],0.95),
                       np.quantile(liabilities['ER+xER-'],0.70),
                       np.quantile(liabilities['ER+xTPN'],0.70)])
outcome = ((liabilities[['ER+','ER-','TPN','ER+xER-','ER+xTPN']]>thresholds).sum(1)>0).astype(int)

df = prs[names]
df.insert(0,'ID',range(1,N+1))
df['Outcome'] = outcome

betas = pd.DataFrame(betas).T
betas.index = names
betas.columns = ['SNP%i'%i for i in range(M)]
betas.to_csv('example_betas.txt',sep='\t')
df.to_csv('example_prs.txt',sep='\t',index=False)
