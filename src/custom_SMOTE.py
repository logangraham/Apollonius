from sklearn.preprocessing import MinMaxScaler
import numpy as np
import time
from scipy.spatial import distance_matrix


def normalize(data):
    scaler = MinMaxScaler()
    scaler.fit(data)
    return scaler.transform(data)


def custom_SMOTE(X,
                 match_columns,
                 smote_columns,
                 num_smote,
                 y=None,
                 K=5):
    """
    Custom implementation of SMOTE (Chawla et al., 2002).
    SMOTE generates synthetic examples by way of a linear interpolation
    between a positive-class observation (the "celebrity") and another
    positive-class observation (the "fan") randomly chosen out of `K`
    nearest-neighbour positive-class observations.
    More details here: https://arxiv.org/pdf/1106.1813.pdf

    Args:
        X (pd DataFrame): matrix to generate synthetic samples from
        match_columns (list): non-continuous columns not to interpolate.
        smote_columns (list): continous columns for interpolation.
        y (pd Series) (optional): y-values to subset X by class
        K (int): number of nearest neighbour interpolation partner candidates.

    Returns:
        `final_synthetics` (pd DataFrame): a dataframe of synthetic samples.
    """
    print("Starting SMOTE...")
    if y is not None:
        X_sm = X[y == 1]
    else:
        X_sm = X

    synthetics = []
    celebrities = []
    fans = []
    print("    1. Creating different data types")
    # create data types
    smote_data = np.array(X_sm[smote_columns])  # d_smote < d_match
    binary_columns = list(set(match_columns).difference(set(smote_columns)))
    binary_data = np.array(X_sm[binary_columns])

    unused_columns = list(set(X_sm.columns).difference(set(match_columns)))
    unused_data = np.array(X_sm[unused_columns])

    print("    2. Retaining orderings")
    # retain ordering
    original_ordering = X_sm.columns
    new_ordering = smote_columns + binary_columns + unused_columns
    reorder_mask = [new_ordering.index(x) for x in original_ordering]

    print("    3. Preparing data")
    # normalize the data
    # normalized_data = normalize(smote_data)
    match_data = np.hstack((smote_data, binary_data))

    # Randomly select celebrity
    celeb_indices = sorted(np.random.choice(range(len(match_data)), num_smote))
    fan_indices = []
    print("    4. Generating samples")

    # compute distance matrix
    print("    5. Creating Distance Matrix")
    s = time.time()
    D = distance_matrix(match_data, match_data)
    D = np.argsort(D)[:, 1:K + 1]
    e = time.time()
    print("Completed in {} seconds".format(round(e - s, 2)))
    print("    6. Generating Synthetics")
    s = time.time()
    for i, index in enumerate(celeb_indices):
        if i % 1000 == 0:
            print("Creating sample {}".format(i))
        s = time.time()
        celebrity_match = match_data[index]
        celebrity_smote = smote_data[index]
        celebrities.append(celebrity_match)

        # choose distance matrixrow
        d_row = D[index, :]

        # choose one random one (the fan)
        fan = np.random.choice(d_row)
        fan_indices.append(fan)
        fan = smote_data[fan]
        fans.append(fan)

        # create synthetic example
        coefficient = np.random.uniform(size=smote_data.shape[1])
        difference = fan - celebrity_smote
        synthetic = celebrity_smote + (coefficient * difference)

        # add to synthetics
        synthetics.append(synthetic)
    celebrity_binaries = binary_data[celeb_indices, :]
    celebrity_unuseds = unused_data[celeb_indices, :]
    final_synthetics = np.array(synthetics)
    final_synthetics = np.hstack((final_synthetics,
                                  celebrity_binaries,
                                  celebrity_unuseds))
    final_synthetics = np.array(final_synthetics)[:, reorder_mask]
    e = time.time()
    print("Completed in {} seconds".format(round(e - s, 2)))
    return final_synthetics