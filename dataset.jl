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

function generateDataset(N::Int, D::Int, a::Float64, alpha::Float64,
        cluster_creator::Function, emission::Function, etype::Type)
    """
    - `N`: number of observations/documents
    - `D`: emission dimension
    - `a`: Geometric parameter for inter-arrival times
    - `cluster_creator`: function generating random clusters
    - `emission`: function generating random emission given a cluster
    """
    z = zeros(Int, N)
    X = zeros(etype, N, D)

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
            thetas[K, :] = cluster_creator()
            push!(nk, 0)
            z[n] = K
        else
            # Choose an existing cluster
            w = (nk - alpha) ./ (n - 1 - alpha*K)
            z[n] = wsample(1:K, w)
        end
        nk[z[n]] += 1
        X[n, :] = emission(thetas[z[n], :])
    end

    return z, X
end

function generateDirDataset(N::Int, D::Int, n_x::Int, a::Float64,
        alpha::Float64, dir_prior_param::Vector)
    # N: number of observations/documents
    # D: size of vocabulary
    # n_x: number of word per document
    # a: Geometric parameter for inter-arrival times
    # alpha: Neutral to the left parameter
    # dir_prior_param: Dirichlet's parameter
    @assert length(dir_prior_param) == D

    cluster_creator = () -> rand(Dirichlet(dir_prior_param))
    emission = (cluster) -> rand(Multinomial(n_x, cluster))

    z, X = generateDataset(N, D, a, alpha, cluster_creator, emission, Int)

    return z, sparse(X)
end

function generateGaussianDataset(N::Int, D::Int, a::Float64, alpha::Float64,
        sigma2::Float64, sigma2_observe::Float64)
    # N: number of observations/documents
    # D: size of vocabulary
    # a: Geometric parameter for inter-arrival times
    # alpha: Neutral to the left parameter
    # sigma: Variance of cluster creation
    # sigma_observe: Variance of emission

    cluster_creator = () -> rand(MvNormal(zeros(D), sqrt(sigma2)))
    emission = (cluster) -> rand(MvNormal(cluster, sqrt(sigma2_observe)))

    return generateDataset(N, D, a, alpha, cluster_creator, emission, Float64)
end

function generateDriftingGaussian(N::Int, D::Int, a::Float64, alpha::Float64,
        sigma2::Float64, sigma2_observe::Float64, drift::Vector{Float64})
    # N: number of observations/documents
    # D: size of vocabulary
    # a: Geometric parameter for inter-arrival times
    # alpha: Neutral to the left parameter
    # sigma: Variance of cluster creation
    # sigma_observe: Variance of emission
    # drift: The drift in the mean (starts at 0)

    cluster_mean = zeros(D)
    cluster_creator = function make_drift_cluster()
        cluster_mean += drift
        return rand(MvNormal(cluster_mean, sqrt(sigma2)))
    end
    emission = (cluster) -> rand(MvNormal(cluster, sqrt(sigma2_observe)))

    return generateDataset(N, D, a, alpha, cluster_creator, emission, Float64)
end
