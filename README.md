# NTL.jl

For accompanying overleaf file, see https://www.overleaf.com/13236006mvbvdwgpwnxq

## Data

Obtain training data as follows

0. Download http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Musical_Instruments_5.json.gz
0. Run `gunzip reviews_Musical_Instruments_5.json.gz`
0. Run `sed '1s/^/[/;$!s/$/,/;$s/$/]/' reviews_Musical_Instruments_5.json > reviews.json`

## Julia packages

Add the following packages:
 - JSON
 - ProgressMeter

For the TextAnalysis package, use:

    Pkg.checkout("TextAnalysis")

rather than the conventional `Pkg.add`
