clear; close all; clc;

file_name = "qkd_dprBB84_4_02_15.mat";

% Read data from file
load(file_name)
fprintf("Now solving: %s\n", description)

% Objective and constraint matrices
A = full(A);

if complex
    G = full(G_alt);
    h = h_alt;
    cones = cones_alt;
else
    G = full(G);
end

% List of cones
total_dim = 1;
for i = 1:length(cones)
    cone_k = cones{i};
    cone_type = cone_k.type;

    if strcmp(cone_type, 'qre')
        n    = cone_k.n;
        vn   = n * (n + 1) / 2;
        dim  = cone_k.dim;     

        cons{i, 1}  = 'QRE';
        cons{i, 2}  = [double(n)];
        A_DDS{i, 1} = -[          G(total_dim, :);
                       expand_vec(G(total_dim+1    : total_dim+vn, :));
                       expand_vec(G(total_dim+vn+1 : total_dim+dim-1, :))];
        b_DDS{i, 1} =  [          h(total_dim);
                       expand_vec(h(total_dim+1    : total_dim+vn));
                       expand_vec(h(total_dim+vn+1 : total_dim+dim-1))];

    elseif strcmp(cone_type, 'nn')
        dim         = cone_k.dim;
        cons{i, 1}  = 'LP';
        cons{i, 2}  = [double(dim)];
        A_DDS{i, 1} = -G(total_dim:total_dim+dim-1, :);
        b_DDS{i, 1} = h(total_dim:total_dim+dim-1);

    elseif strcmp(cone_type, 'psd')
        n    = cone_k.n;
        vn   = n * (n + 1) / 2;
        dim  = cone_k.dim;     

        cons{i, 1}  = 'SDP';
        cons{i, 2}  = [double(n)];
        A_DDS{i, 1} = -expand_vec(G(total_dim : total_dim+dim-1, :));
        b_DDS{i, 1} =  expand_vec(h(total_dim : total_dim+dim-1));    
    end

    total_dim = total_dim + dim;
end

cons{i + 1, 1}  = 'EQ';
cons{i + 1, 2}  = [length(b)];
A_DDS{i + 1, 1} = A;
b_DDS{i + 1, 1} = b;


[x, y, info] = DDS(c, A_DDS, b_DDS, cons);

fprintf("Opt value: %.10f \t\n", c'*x + offset);
fprintf("Solve time: %.10f \t\n", info.time);

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