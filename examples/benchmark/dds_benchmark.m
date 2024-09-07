clear; close all; clc;


output = "data.csv";
heading = ["problem", "solver", "status", "optval", "time", "iter", "gap", "pfeas", "dfeas"];
writematrix(heading, output);

folder = './qreps';
files = dir(folder);

for i = 1:length(files)
    file = files(i);
    if file.isdir
        continue
    end
    filename = file.name;
    
    % Read data from file
    f = fopen([folder, '/', filename], 'r');
    k = 1;
    use_G = false;
    offset = 0.0;
    
    tline = fgetl(f);
    while ischar(tline)
        %%%%%%%%%%%%%%%%%%%%%%
        %% File information
        %%%%%%%%%%%%%%%%%%%%%%
        if strcmp(tline, "VER")
            ver = str2double(fgetl(f));
            assert(ver == 4)
        end
        
        %%%%%%%%%%%%%%%%%%%%%%
        %% Model structure
        %%%%%%%%%%%%%%%%%%%%%%
        if strcmp(tline, "OBJSENSE")
            line = fgetl(f);
            if strcmp(line, "MIN")
                objsense = 1;
            elseif strcmp(line, "MAX")
                objsense = -1;
            end
        end
    
        if strcmp(tline, "VAR")
            line = split(fgetl(f), ' ');
            nx = str2double(line{1});
            lines = str2double(line{2});
            total_dim = 1;
    
            for i = 1:lines
                line = split(fgetl(f), ' ');
                cone = line{1};
                dim = str2double(line{2});
    
                if strcmp(cone, "F")
                    assert(lines == 1)
                    assert(nx == dim)
                    use_G = true;
                elseif strcmp(cone, "L+")
                    cons{k, 1}  = 'LP';
                    cons{k, 2}  = [dim];
                    A_DDS{k, 1} = zeros(dim, nx);
                    A_DDS{k, 1}(:, total_dim:total_dim+dim-1) = eye(dim);
                    b_DDS{k, 1} = zeros(dim, 1);
                    k = k + 1;
                elseif strcmp(cone, "SVECPSD")
                    n           = floor(sqrt(1 + 8 * dim) / 2);        
                    cons{k, 1}  = 'SDP';
                    cons{k, 2}  = [n];
                    A_DDS{k, 1} = zeros(n*n, nx);
                    A_DDS{k, 1}(:, total_dim:total_dim+dim-1) = seye(n);
                    b_DDS{k, 1} = zeros(n*n, 1);
                    k = k + 1;
                elseif strcmp(cone, "SVECQE")
                    vn          = dim - 2;
                    n           = floor(sqrt(1 + 8 * vn) / 2);        
                    cons{k, 1}  = 'QE';
                    cons{k, 2}  = [n];
                    A_DDS{k, 1} = zeros(1 + n*n, nx);
                    A_DDS{k, 1}(:, total_dim:total_dim+dim-1) = [
                                    1,   zeros(1, vn)
                        zeros(n*n, 1),        seye(n)
                    ];
                    b_DDS{k, 1} = zeros(1 + 2*n*n, 1);
                    k = k + 1;
                elseif strcmp(cone, "SVECQRE")
                    vn          = (dim - 1) / 2;
                    n           = floor(sqrt(1 + 8 * vn) / 2);        
                    cons{k, 1}  = 'QRE';
                    cons{k, 2}  = [n];
                    A_DDS{k, 1} = zeros(1 + 2*n*n, nx);
                    A_DDS{k, 1}(:, total_dim:total_dim+dim-1) = [
                                    1,   zeros(1, vn),   zeros(1, vn);
                        zeros(n*n, 1),        seye(n), zeros(n*n, vn);
                        zeros(n*n, 1), zeros(n*n, vn),        seye(n);
                    ];
                    b_DDS{k, 1} = zeros(1 + 2*n*n, 1);
                    k = k + 1;
                end
                total_dim = total_dim + dim;
            end
        end
    
         if strcmp(tline, "CON")
            line = split(fgetl(f), ' ');
            ng = str2double(line{1});
            lines = str2double(line{2});
    
            for i = 1:lines
                line = split(fgetl(f), ' ');
                cone = line{1};
                dim = str2double(line{2});
    
                if strcmp(cone, "L=")
                    cons{k, 1}  = 'EQ';
                    cons{k, 2}  = [dim];
                    k = k + 1;
                elseif strcmp(cone, "L+")
                    cons{k, 1}  = 'LP';
                    cons{k, 2}  = [dim];
                    k = k + 1;
                elseif strcmp(cone, "SVECPSD")
                    n           = floor(sqrt(1 + 8 * dim) / 2);        
                    cons{k, 1}  = 'SDP';
                    cons{k, 2}  = [n];
                    k = k + 1;
                elseif strcmp(cone, "SVECQE")
                    vn          = dim - 2;
                    n           = floor(sqrt(1 + 8 * vn) / 2);        
                    cons{k, 1}  = 'QE';
                    cons{k, 2}  = [n];
                    k = k + 1;
                elseif strcmp(cone, "SVECQRE")
                    vn          = (dim - 1) / 2;
                    n           = floor(sqrt(1 + 8 * vn) / 2);        
                    cons{k, 1}  = 'QRE';
                    cons{k, 2}  = [n];
                    k = k + 1;
                end
            end
         end
    
        %%%%%%%%%%%%%%%%%%%%%%
        %% Problem data
        %%%%%%%%%%%%%%%%%%%%%%
         if strcmp(tline, "OBJACOORD")
            c = zeros(nx, 1);
            nnz = str2double(fgetl(f));
            for i = 1:nnz
                line = split(fgetl(f), ' ');
                idx = str2double(line{1}) + 1;
                val = str2double(line{2});
                c(idx) = val;
            end
         end
    
         if strcmp(tline, "OBJBCOORD")
            offset = str2double(fgetl(f));
         end
    
         if strcmp(tline, "ACOORD")
            A = zeros(ng, nx);
            nnz = str2double(fgetl(f));
            for i = 1:nnz
                line = split(fgetl(f), ' ');
                idx = str2double(line{1}) + 1;
                jdx = str2double(line{2}) + 1;
                val = str2double(line{3});
                A(idx, jdx) = val;
            end
         end
    
        if strcmp(tline, "BCOORD")
            b = zeros(ng, 1);
            nnz = str2double(fgetl(f));
            for i = 1:nnz
                line = split(fgetl(f), ' ');
                idx = str2double(line{1}) + 1;
                val = str2double(line{2});
                b(idx) = val;
            end
        end
    
        tline = fgetl(f);
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%
    %% Process problem data
    %%%%%%%%%%%%%%%%%%%%%%%%
    if use_G
        total_dim = 1;
        for i = 1:k-1
            if strcmp(cons{i, 1}, "EQ")
                dim = cons{i, 2};
                A_DDS{i, 1} = A(total_dim:total_dim+dim-1, :);
                b_DDS{i, 1} = b(total_dim:total_dim+dim-1, :);
            elseif strcmp(cons{i, 1}, "LP")
                dim = cons{i, 2};
                A_DDS{i, 1} =  A(total_dim:total_dim+dim-1, :);
                b_DDS{i, 1} = -b(total_dim:total_dim+dim-1, :);
            elseif strcmp(cons{i, 1}, "SDP")
                n = cons{i, 2};
                dim = n * (n + 1) / 2;
                A_DDS{i, 1} =  expand_vec(A(total_dim:total_dim+dim-1, :));
                b_DDS{i, 1} = -expand_vec(b(total_dim:total_dim+dim-1, :));
            elseif strcmp(cons{i, 1}, "QE")
                n = cons{i, 2};
                vn = n * (n + 1) / 2;
                dim = 2 + vn;
                A_DDS{i, 1} =  [          A(total_dim, :);
                               expand_vec(A(total_dim+2 : total_dim+dim-1, :))];
                b_DDS{i, 1} = -[          b(total_dim);
                               expand_vec(b(total_dim+2 : total_dim+dim-1))];
            elseif strcmp(cons{i, 1}, "QRE")
                n = cons{i, 2};
                vn = n * (n + 1) / 2;
                dim = 1 + 2*vn;
                A_DDS{i, 1} =  [          A(total_dim, :);
                               expand_vec(A(total_dim+1    : total_dim+vn, :));
                               expand_vec(A(total_dim+vn+1 : total_dim+dim-1, :))];
                b_DDS{i, 1} = -[          b(total_dim);
                               expand_vec(b(total_dim+1    : total_dim+vn));
                               expand_vec(b(total_dim+vn+1 : total_dim+dim-1))];
            end
            total_dim = total_dim + dim;
        end
    else
        assert(strcmp(cons{k-1, 1}, "EQ"))
        A_DDS{k-1, 1} = A;
        b_DDS{k-1, 1} = b;
    end
    
    fclose(f);
    
    [x, y, info] = DDS(c, A_DDS, b_DDS, cons);

end

%% Aux functions
function vec = mat_to_vec(mat)
    n = size(mat, 1);
    vn = n * (n + 1) / 2;
    vec = zeros(vn, 1);
    k = 1;
    for j = 1:n
        for i = 1:j-1
            vec(k) = mat(i, j) * sqrt(2);
            k = k + 1;
        end
        vec(k) = mat(j, j);
        k = k + 1;
    end
end

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

function eye = seye(n)
    vn = n * (n + 1) / 2;
    eye = zeros(n*n, vn);
    for k = 1:vn
        H = zeros(vn, 1);
        H(k) = 1;
        H = vec_to_mat(H);
        eye(:, k) = H(:);
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