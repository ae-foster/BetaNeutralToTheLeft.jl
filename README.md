# NTL.jl

For accompanying overleaf file, see https://www.overleaf.com/13236006mvbvdwgpwnxq

## TODO

- Make it scale
 - Why global parameter update is slow ??
 - Preallocate matrices ?
 - Faster log likelihood of Dirichlet-Multinomial distribution ?
- Mean field + taylor expension (1st order) approximation, for model with prior on Geometric parameter
- 2nd order Taylor expension ?
- predictive log-likelihood metric ?

## Data

Obtain training data as follows

0. Download http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Musical_Instruments_5.json.gz
0. Run `gunzip reviews_Musical_Instruments_5.json.gz`
0. Run `sed '1s/^/[/;$!s/$/,/;$s/$/]/' reviews_Musical_Instruments_5.json > reviews.json`

## Julia packages

Add the following packages:
 - JSON
 - ProgressMeter
 - Distributions

For the TextAnalysis package, use:

    Pkg.checkout("TextAnalysis")

rather than the conventional `Pkg.add` to get the master branch
