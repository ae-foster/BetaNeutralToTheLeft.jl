using JSON
using  DataStructures
using TextAnalysis

# Download http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Musical_Instruments_5.json.gz
# run(`sed '1s/^/[/;$!s/$/,/;$s/$/]/' reviews_Musical_Instruments_5.json > reviews.json`)
filename = "reviews.json"
f = JSON.parsefile(filename; dicttype=OrderedDict, inttype=Int64, use_mmap=true)

N = length(f)
z = Array{String}(N)
reviews = Array{String}(N)

for index, review in enumerate(f)
    z[index] = f["asin"]
    reviews[index] = StringDocument(f["reviewText"])
end

function clean_corpus!(crps::Corpus)
    remove_corrupt_utf8!(crps)
    remove_punctuation!(crps)
    remove_numbers!(crps)
    remove_case!(crps)
    remove_stop_words!(crps)
    remove_prepositions!(crps)
    remove_pronouns!(crps)
    remove_articles!(crps)
end

crps = Corpus(reviews)
clean_corpus!(crps)

# hash_dtm(crps) # avoid having to compute the lexicon
update_lexicon!(crps)
m = DocumentTermMatrix(crps)
m = tf_idf(m)
x = dtm(m) # dtm(m, :dense)
