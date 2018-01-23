# NTL.jl

For accompanying overleaf file, see https://www.overleaf.com/13236006mvbvdwgpwnxq

## TODO
### Priority
 - [ ] Improved estimator for K_{n-1}
 - [ ] For positive alpha, epsilon = alpha. Accumulate discarded probability
 - [ ] Synthetic data with Gaussian emissions
 - [ ] Train on bigger Amazon data on ziz
 - [ ] Compare (synthetic Gaussian data) with a NRM model using Alex Tank's VI
 - Other estimates of q^pr
   - [x] MC estimate
   - [ ] 1st order Taylor expansions
   - [ ] 2nd order Taylor expansions
 - [ ] Case of negative alpha
 - [ ] Instantiate clusters randomly
 - [ ] Metrics for coclustering matrix
 - [ ] predictive log-likelihood metric ?

### Secondary
 - Preallocate matrices ?
 - (possible) Faster log likelihood of Dirichlet-Multinomial distribution ?

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
