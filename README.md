# NTL.jl

For accompanying overleaf file, see https://www.overleaf.com/13236006mvbvdwgpwnxq

## TODO
### Priority
 - [x] Improved estimator for K_{n-1}
 - [x] For positive alpha, epsilon = alpha. Accumulate discarded probability
 - [x] Synthetic data with Gaussian emissions
 - [x] Compare (synthetic Gaussian data) with a NRM model using Alex Tank's VI
 - Other estimates of q^pr
   - [x] NRM model
   - [x] MC estimate
   - [ ] 1st order Taylor expansions
   - [ ] 2nd order Taylor expansions
 - [ ] Case of negative alpha (what does this mean?)
 - [ ] Tune alpha using CV
 - [ ] Instantiate clusters randomly
 - [ ] Metrics for coclustering matrix
 - [x] predictive log-likelihood metric
 - [ ] Expectation propagation
 - [x] Coclustering and Gaussian plots
 
### Data sources
 - [ ] Earthquake data
 - [ ] Kaggle movies
 - [ ] Malicious activities

### Secondary
 - [ ] Train on bigger Amazon data on ziz
 - [ ] Preallocate matrices ?
 - (possible) Faster log likelihood of Dirichlet-Multinomial distribution ?
 
### Ben
 - [ ] Gibbs sampler for partition- and graph-valued data
 - [ ] Gibbs sampler for mixture model

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
