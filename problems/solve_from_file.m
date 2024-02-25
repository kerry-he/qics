clear; close all; clc;

file_name = "qkd_dprBB84_4_02_15.mat";

% Read data from file
load(file_name)
fprintf("Now solving: %s\n", description)

% Objective and constraint matrices
A = full(A);
G = full(G);

% List of cones
total_dim = 1;
for i = 1:length(cones)
    cone_k = cones{i};
    cone_type = cone_k.type;

    if strcmp(cone_type, 'qre')
        n           = cone_k.n;
        dim         = cone_k.dim;
        hermitian   = cone_k.complex;

        if ~hermitian
            vn          = n * (n + 1) / 2;
            cons{i, 1}  = 'QRE';
            cons{i, 2}  = [double(n)];
            A_DDS{i, 1} = -[          G(total_dim, :);
                           expand_vec(G(total_dim+1    : total_dim+vn, :));
                           expand_vec(G(total_dim+vn+1 : total_dim+dim-1, :))];
            b_DDS{i, 1} =  [          h(total_dim);
                           expand_vec(h(total_dim+1    : total_dim+vn));
                           expand_vec(h(total_dim+vn+1 : total_dim+dim-1))];
        else
            vn = n * n;
            cons{i, 1}  = 'QRE';
            cons{i, 2}  = [2 * double(n)];
            A_DDS{i, 1} = -[          G(total_dim, :);
                           complex_to_real(G(total_dim+1    : total_dim+vn, :), n);
                           complex_to_real(G(total_dim+vn+1 : total_dim+dim-1, :), n)];
            b_DDS{i, 1} =  [          h(total_dim);
                           complex_to_real(h(total_dim+1    : total_dim+vn), n);
                           complex_to_real(h(total_dim+vn+1 : total_dim+dim-1), n)];
        end

    elseif strcmp(cone_type, 'nn')
        dim         = cone_k.dim;
        cons{i, 1}  = 'LP';
        cons{i, 2}  = [double(dim)];
        A_DDS{i, 1} = -G(total_dim:total_dim+dim-1, :);
        b_DDS{i, 1} = h(total_dim:total_dim+dim-1);

    elseif strcmp(cone_type, 'psd')
        n           = cone_k.n;
        dim         = cone_k.dim;
        hermitian   = cone_k.complex;

        if ~hermitian
            cons{i, 1}  = 'SDP';
            cons{i, 2}  = [double(n)];
            A_DDS{i, 1} = -expand_vec(G(total_dim : total_dim+dim-1, :));
            b_DDS{i, 1} =  expand_vec(h(total_dim : total_dim+dim-1));    
        else
            cons{i, 1}  = 'SDP';
            cons{i, 2}  = [2 * double(n)];
            A_DDS{i, 1} = -complex_to_real(G(total_dim : total_dim+dim-1, :), n);
            b_DDS{i, 1} =  complex_to_real(h(total_dim : total_dim+dim-1), n);    
        end
    end

    total_dim = total_dim + dim;
end

cons{i + 1, 1}  = 'EQ';
cons{i + 1, 2}  = [length(b)];
A_DDS{i + 1, 1} = A;
b_DDS{i + 1, 1} = b;


[x, y, info] = DDS(c, A_DDS, b_DDS, cons);

fprintf("Opt value: %.10f \t\n", c*x + offset);
fprintf("Solve time: %.10f \t\n", info.time);

%% Functions
function new_vecs = expand_vec(vecs)
    [vn, p] = size(vecs);
    n = (sqrt(1 + 8 * vn) - 1) / 2;

    new_vecs = zeros(n*n, p);

    for i = 1:p
        vec = vecs(:, i);
        mat = vec_to_mat(vec, false);
        new_vecs(:, i) = mat(:);
    end
end

function vec = mat_to_vec(mat, hermitian)
    if hermitian
        [n, ~] = size(mat);
        vn = n * n;
        vec = zeros(vn, 1);
        t = 1;
        for j = 1:n
            for i = 1:j-1
                vec(t)     = real(mat(i, j)) * sqrt(2.0);
                vec(t + 1) = imag(mat(i, j)) * sqrt(2.0);
                t = t + 2;
            end
            vec(t) = real(mat(j, j));
            t = t + 1;
        end
    else
        [n, ~] = size(mat);
        vn = n * (n + 1) / 2;
        vec = zeros(vn, 1);
        t = 1;
        for j = 1:n
            for i = 1:j-1
                vec(t) = mat(i, j) * sqrt(2.0);
                t = t + 1;
            end
            vec(t) = mat(j, j);
            t = t + 1;
        end
    end
end

function mat = vec_to_mat(vec, hermitian)
    if hermitian
        [vn, ~] = size(vec);
        n = sqrt(vn);
        mat = zeros(n, n);
        t = 1;
        for j = 1:n
            for i = 1:j-1
                mat(i, j) = complex(vec(t),  vec(t + 1)) * sqrt(0.5);
                mat(j, i) = complex(vec(t), -vec(t + 1)) * sqrt(0.5);
                t = t + 2;
            end
            mat(j, j) = vec(t);
            t = t + 1;
        end
    else
        [vn, ~] = size(vec);
        n = (sqrt(1 + 8*vn) - 1) / 2;
        mat = zeros(n, n);
        t = 1;
        for j = 1:n
            for i = 1:j-1
                mat(i, j) = vec(t) * sqrt(0.5);
                t = t + 1;
            end
            mat(j, j) = vec(t);
            t = t + 1;
        end    
    end
end

function Gout = complex_to_real(Gin, n)
    m = size(Gin, 2);
    Gout = zeros(4*n*n, m);

    for k = 1:m
        Gk = vec_to_mat(Gin(:, k), true);
        Gk_real = real(Gk);
        Gk_imag = imag(Gk);
        Gk = [Gk_real, -Gk_imag;
              Gk_imag,  Gk_real];
        Gout(:, k) = Gk(:);
    end
end