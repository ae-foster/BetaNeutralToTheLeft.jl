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

function generateDataset(N::Int, D::Int, n_x::Int, a::Float64, alpha::Vector)
    @assert length(alpha) == D
    z = zeros(Int, N)
    X = zeros(Int, N, D)
    Ts = [1]
    while Ts[end] < N
        geo = rand(Geometric(a)) + 1
        T = Int(Ts[end]) + rand(Geometric(a))
        push!(Ts, T)
    end
    K = 1
    theta = rand(Dirichlet(alpha))
    for n in 1:N
        println("Ts[K]: ", Ts[K])
        if Ts[K] == n
            K += 1
            theta = rand(Dirichlet(alpha))
        end
        z[n] = K
        X[n, :] = rand(Multinomial(n_x, theta))
    end

    perm = randperm(N)
    return z[perm], X[perm, :]
end
