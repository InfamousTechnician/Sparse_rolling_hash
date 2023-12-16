# Sparse rolling hash

A very simple rolling hash is a one that takes moduluses and sums them up. I twisted this by not only adding up modulus values, but also taking the moduluses of the respective dimensions, and created compressed vectors so, with lower dimensionality. I also assume and take advantage of the vectors being sparse. Afterwards, the K nearest neighbour problem is being solved and although the suggested neighbours mostly differ, the total distances are quite similar!
