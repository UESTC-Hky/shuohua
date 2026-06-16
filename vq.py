import numpy as np


def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2, axis=1))


def lbg_train(features, target_size=16, epsilon=0.01, max_iter=50):
    num_vectors, dim = features.shape

    codebook = np.mean(features, axis=0, keepdims=True)
    distortion = np.mean(euclidean_distance(features, codebook) ** 2)

    while codebook.shape[0] < target_size:
        new_codebook = np.zeros((codebook.shape[0] * 2, dim))
        for i in range(codebook.shape[0]):
            new_codebook[2 * i] = codebook[i] * (1.0 + epsilon)
            new_codebook[2 * i + 1] = codebook[i] * (1.0 - epsilon)
        codebook = new_codebook

        prev_distortion = distortion
        for iteration in range(max_iter):
            distances = np.zeros((num_vectors, codebook.shape[0]))
            for c in range(codebook.shape[0]):
                diff = features - codebook[c]
                distances[:, c] = np.sqrt(np.sum(diff ** 2, axis=1))

            nearest = np.argmin(distances, axis=1)

            new_codebook = np.zeros_like(codebook)
            counts = np.zeros(codebook.shape[0], dtype=int)
            for c in range(codebook.shape[0]):
                assigned = features[nearest == c]
                if len(assigned) > 0:
                    new_codebook[c] = np.mean(assigned, axis=0)
                    counts[c] = len(assigned)
                else:
                    new_codebook[c] = codebook[c]

            distortion = 0.0
            for i in range(num_vectors):
                distortion += np.sum((features[i] - new_codebook[nearest[i]]) ** 2)
            distortion /= num_vectors

            if prev_distortion > 0:
                relative_change = (prev_distortion - distortion) / prev_distortion
                if relative_change < 0.001:
                    break

            codebook = new_codebook
            prev_distortion = distortion

    return codebook


def compute_distortion(features, codebook):
    num_vectors = features.shape[0]
    distances = np.zeros((num_vectors, codebook.shape[0]))
    for c in range(codebook.shape[0]):
        diff = features - codebook[c]
        distances[:, c] = np.sqrt(np.sum(diff ** 2, axis=1))
    min_distances = np.min(distances, axis=1)
    return np.mean(min_distances)
