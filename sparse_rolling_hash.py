from random import randint as rdi

maximum = 114 # integer range
sparcity = 3 # percent
length = 2048 # vector length
count = 1000 # number of vectors

sparse_vectors = [[rdi(0,maximum) if (rdi(0,100) < sparcity) else 0 for _ in range(length)] for _ in range(count)]
print("done :)")

def sparse_rolling_hash(spv, val_prime, dim_prime): # my contribution :)
    ret_val = [0 for _ in range(dim_prime)]
    for idx in range(len(spv)):
        ret_val[idx % dim_prime] += spv[idx]
        ret_val[idx % dim_prime] %= val_prime
    return ret_val

# compressing dimensions by taking advantage of sparcity
val_prime = 17 # just because
dim_prime = 61 # nearest next prime to length * sparcity should have been larger than 61

compressed = []
for idx in range(len(sparse_vectors)): # compressing sparse vectors with hash
    compressed.append(sparse_rolling_hash(sparse_vectors[idx], val_prime, dim_prime))

def KNN(vectors, K, v_in): # KNN search algo
    nearests = [None for _ in range(K)]
    distances = [maximum**2 for _ in range(K)]
    for vidx in range(len(vectors)):
        for idx in range(len(v_in)):
            distance = (v_in[idx]**2 + vectors[vidx][idx]**2)**0.5
        for idx in range(K):
            if distance < distances[idx]:
                nearests[idx] = vidx
                distances[idx] = distance
                break
    return nearests

for _ in range(20): # running tests

    # looking for K nearest neighbours in compressed format
    K = 10
    starter = rdi(0, count)
    print(starter)
    compressed_start = sparse_rolling_hash(sparse_vectors[starter], val_prime, dim_prime)
    compressed_near = KNN(compressed, K, compressed_start)
    print(compressed_near)

    # testing for really nearest K neighbours of original dataset
    compressed_sheer = KNN(sparse_vectors, K, sparse_vectors[starter])
    print(compressed_sheer)

    # comparing results by total distances
    total = 0
    for compressed_near_ones in compressed_near:
        for idx in range(len(sparse_vectors[starter])):
            total += (sparse_vectors[starter][idx]**2 + sparse_vectors[compressed_near_ones][idx]**2)**0.5 
    print(total)

    total = 0
    for near_ones in compressed_sheer:
        for idx in range(len(sparse_vectors[starter])):
            total += (sparse_vectors[starter][idx]**2 + sparse_vectors[near_ones][idx]**2)**0.5 
    print(total)

