function G = G_matrix(N)
F = [1 0;1 1];
G = [1 0;1 1];
n = log2(N);
for i = 1:n-1
    G=kron(G,F);
end
end