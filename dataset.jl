using JSON
using DataStructures
using TextAnalysis # Pkg.checkout("TextAnalysis")
using Distributions

function prepare2!(crps::Corpus, flags::UInt32; skip_patterns = Set{AbstractString}(), skip_words = Set{AbstractString}())
    ((flags & strip_sparse_terms) > 0) && union!(skip_words, sparse_terms(crps))
    ((flags & strip_frequent_terms) > 0) && union!(skip_words, frequent_terms(crps))

    ((flags & strip_corrupt_utf8) > 0) && remove_corrupt_utf8!(crps)
    ((flags & strip_case) > 0) && remove_case!(crps)
    ((flags & strip_html_tags) > 0) && remove_html_tags!(crps)

    lang = language(crps.documents[1])   # assuming all documents are of the same language - practically true
    r = TextAnalysis._build_regex(lang, flags, skip_patterns, skip_words)
    !isempty(r.pattern) && remove_patterns!(crps, r)

    ((flags & tag_part_of_speech) > 0) && tag_pos!(crps)
    nothing
end

function clean_corpus!(crps::Corpus)
    remove_case!(crps)
    remove_corrupt_utf8!(crps)
    prepare2!(crps, strip_whitespace)
    prepare2!(crps, strip_punctuation)
    prepare2!(crps, strip_non_letters)
    prepare2!(crps, strip_numbers)
    prepare2!(crps, strip_prepositions)
    prepare2!(crps, strip_pronouns)
    prepare2!(crps, strip_articles)
    prepare2!(crps, strip_stopwords)
end

function getDocumentTermMatrixFromReviewsJson(filename::String)
    f = JSON.parsefile(filename; dicttype=OrderedDict, inttype=Int64, use_mmap=true)
    N = length(f)
    z = Array{String}(N)
    reviews = Array{StringDocument}(N)
    timestamps = Array{Int}(N)

    index = 1
    for review in f
        z[index] = review["asin"]
        timestamps[index] = review["unixReviewTime"]
        reviews[index] = StringDocument(review["reviewText"])
        index += 1
    end

    timestamp_ordering = sortperm(timestamps)
    z = z[timestamp_ordering]
    reviews = reviews[timestamp_ordering]

    crps = Corpus(Any[reviews...])
    clean_corpus!(crps)

    # hash_dtm(crps) # avoid having to compute the lexicon
    update_lexicon!(crps)
    m = DocumentTermMatrix(crps)
    # m = tf_idf(m)
    z, dtm(m)
end

function generateDataset(N::Int, D::Int, n_x::Int, a::Float64, alpha::Float64, dir_prior_param::Vector)
    # N: number of observations/documents
    # D: size of vocabulary
    # n_x: number of word per document
    # a: Geometric parameter for inter-arrival times
    # alpha: Neutral to the left parameter
    # dir_prior_param: Dirichlet's parameter
    @assert length(dir_prior_param) == D

    z = zeros(Int, N)
    X = zeros(Int, N, D)

    # Sample arrival times
    T = 1
    Ts = [T]
    while T < N
        geo = rand(Geometric(a)) + 1
        T += geo
        push!(Ts, T)
    end

    # Sample observations/documents
    thetas = zeros(Float64, length(Ts)+1, D)
    K = 1
    nk = zeros(Int, K)
    for n in 1:N
        if Ts[K] == n
            # Sample and assign to a new cluster
            K += 1
            thetas[K, :] = rand(Dirichlet(dir_prior_param))
            push!(nk, 0)
            z[n] = K
        else
            # Choose an existing cluster
            w = (nk - alpha) ./ (n - 1 - alpha*K)
            z[n] = wsample(1:K, w)
        end
        nk[z[n]] += 1
        X[n, :] = rand(Multinomial(n_x, thetas[K, :]))
    end

    return z, X
end
