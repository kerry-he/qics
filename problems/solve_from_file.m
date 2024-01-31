clear; close all; clc;

file_name = "ncm_50_eye.hdf5";

% Read data from file
description = h5readatt(file_name, '/', 'description');
offset = h5readatt(file_name, '/', 'offset');
fprintf("Now solving: %s\n", description)


% Objective and constraint matrices
c_DDS = h5read(file_name, '/data/c');
b = h5read(file_name, '/data/b')';
h = h5read(file_name, '/data/h')';

is_sparse = h5readatt(file_name, '/data/A', 'sparse');
if strcmp(is_sparse{1}, 'TRUE')
    A_vij   = h5read(file_name, '/data/A');
    A_v     = A_vij(:, 1);
    A_i     = A_vij(:, 2) + 1;
    A_j     = A_vij(:, 3) + 1;
    A_shape = h5readatt(file_name, '/data/A', 'shape');

    A       = sparse(A_i, A_j, A_v, A_shape(1), A_shape(2));
    A       = full(A);
else
    A = h5read(file_name, '/data/A');
end

is_sparse = h5readatt(file_name, '/data/G', 'sparse');
if strcmp(is_sparse{1}, 'TRUE')
    G_vij   = h5read(file_name, '/data/G');
    G_v     = G_vij(:, 1);
    G_i     = G_vij(:, 2) + 1;
    G_j     = G_vij(:, 3) + 1;
    G_shape = h5readatt(file_name, '/data/G', 'shape');

    G       = sparse(G_i, G_j, G_v, G_shape(1), G_shape(2));
    G       = full(G);
else
    G = h5read(file_name, '/data/G');
end


% List of cones
total_dim = 1;
for i = 1:length(h5info(file_name, '/cones').Datasets)
    cone_name = strcat('/cones/', string(i - 1));
    cone_type = h5read(file_name, cone_name);

    if strcmp(cone_type, 'qre')
        n       = h5readatt(file_name, cone_name, 'n');
        dim     = h5readatt(file_name, cone_name, 'dim');
        complex = h5readatt(file_name, cone_name, 'complex');

        vn = n * (n + 1) / 2;
        cons_DDS{i, 1}  = 'QRE';
        cons_DDS{i, 2}  = [double(n)];
        A_DDS{i, 1} = -[          G(total_dim, :);
                       expand_vec(G(total_dim+1    : total_dim+vn, :));
                       expand_vec(G(total_dim+vn+1 : dim, :))];
        b_DDS{i, 1} =  [          h(total_dim);
                       expand_vec(h(total_dim+1    : total_dim+vn));
                       expand_vec(h(total_dim+vn+1 : dim))];
    end

    total_dim = total_dim + dim;
end

cons{i + 1, 1}  = 'EQ';
cons{i + 1, 2}  = [length(b)];
A_DDS{i + 1, 1} = A;
b_DDS{i + 1, 1} = b;

[x, y, info] = DDS(c_DDS, A_DDS, b_DDS, cons_DDS);



%% Functions
function mat = vec_to_mat(vec)
    [vn, ~] = size(vec);
    n = (sqrt(1 + 8 * vn) - 1) / 2;
    mat = zeros(n, n);
    t = 1;
    for i = 1:n
        for j = 1:i-1
            mat(i, j) = vec(t) * sqrt(0.5);
            mat(j, i) = vec(t) * sqrt(0.5);
            t = t + 1;
        end
        mat(i, i) = vec(t);
        t = t + 1;
    end
end

function new_vecs = expand_vec(vecs)
    [vn, p] = size(vecs);
    n = (sqrt(1 + 8 * vn) - 1) / 2;

    new_vecs = zeros(n*n, p);

    for i = 1:p
        vec = vecs(:, i);
        mat = vec_to_mat(vec);
        new_vecs(:, i) = mat(:);
    end
end