import SparseArrays
using LinearAlgebra

import Hypatia
import Hypatia.Cones
import Hypatia.Solvers

T = Float64
R = Complex{T}

function _swap_rows(G, cones)
    total_dim = 1
    for (k, cone) in enumerate(cones)
        if cone isa Cones.EpiTrRelEntropyTri
            for k in 1:cone.vw_dim
                Base.swaprows!(G, total_dim + k, total_dim + cone.vw_dim + k)
            end
        elseif cone isa Cones.EpiRelEntropy
            for k in 1:cone.w_dim
                Base.swaprows!(G, total_dim + k, total_dim + cone.w_dim + k)
            end
        end
        total_dim += Cones.dimension(cone)
    end
    return G
end

function read_problem(filename)
    # Read data from file
    fd = open(filename, "r")
    cones = Cones.Cone{T}[]
    obj_offset = 0.0
    use_G = false
    totalvars = 0
    totalconstr = 0
    objsense = 1

    function _read_cone(cone, sz)
        if cone == "L+"
            return Cones.Nonnegative{T}(sz)
        elseif cone == "Q"
            return Cones.EpiNormEucl{T}(sz)
        elseif cone == "SVECPSD"
            return Cones.PosSemidefTri{T, T}(sz)
        elseif cone == "HVECPSD"
            return Cones.PosSemidefTri{T, R}(sz)
        elseif cone == "CRE"
            return Cones.EpiRelEntropy{T}(sz)
        elseif cone == "SVECQE"
            n = div(isqrt(1 + 8 * (sz - 2)), 2)
            return Cones.EpiPerSepSpectral{Cones.MatrixCSqr{T, T}, T}(
                Cones.NegEntropySSF(),
                n
            )
        elseif cone == "HVECQE"
            n = isqrt(sz - 2)
            return Cones.EpiPerSepSpectral{Cones.MatrixCSqr{T, R}, T}(
                Cones.NegEntropySSF(),
                n
            )
        elseif cone == "SVECQRE"
            return Cones.EpiTrRelEntropyTri{T, T}(sz)
        elseif cone == "HVECQRE"
            return Cones.EpiTrRelEntropyTri{T, R}(sz)
        end
    end

    while !eof(fd)
        line = readline(fd)
        startswith(line,"#") && continue # comments
        length(line) == 0 && continue # blank lines

        # new block

        if startswith(line,  "VER")
            nextline = readline(fd)
            @assert startswith(nextline, "4")
            continue
        end

        if startswith(line, "OBJSENSE")
            nextline = readline(fd)
            if strip(nextline) == "MIN"
                objsense = 1
            else
                objsense = -1
            end
            continue
        end

        if startswith(line, "VAR")
            nextline = readline(fd)
            totalvars, lines = split(nextline)
            totalvars = parse(Int, strip(totalvars))
            lines = parse(Int, strip(lines))

            for k in 1:lines
                nextline = readline(fd)
                cone, sz = split(nextline)
                sz = parse(Int,strip(sz))
                if cone == "F"
                    @assert lines == 1
                    @assert totalvars = sz
                    use_G = true
                else
                    push!(cones, _read_cone(cone, sz))
                    use_G = false
                end
            end
            
            continue
        end

        if startswith(line, "CON")
            nextline = readline(fd)
            totalconstr, lines = split(nextline)
            totalconstr = parse(Int,strip(totalconstr))
            lines = parse(Int,strip(lines))
            A_idxs = 0:0
            total_cone_dim = 0

            for k in 1:lines
                nextline = readline(fd)
                cone, sz = split(nextline)
                sz = parse(Int,strip(sz))
                if cone == "L="
                    A_idxs = total_cone_dim:total_cone_dim+sz
                else
                    @assert (~use_G)
                    push!(cones, _read_cone(cone, sz))
                end
                total_cone_dim += sz
            end

            continue
        end

        if startswith(line, "OBJACOORD")
            c = zeros(T, totalvars, 1)
            nextline = readline(fd)
            lines = parse(Int, strip(nextline))
            for k in 1:lines
                nextline = readline(fd)
                i, v = split(nextline)
                i = parse(Int,strip(i)) + 1
                v = parse(Float64,strip(v))
                c[i] = v
            end
        end

        if startswith(line, "OBJBCOORD")
            nextline = readline(fd)
            obj_offset = parse(Float64, strip(nextline))
        end

        if startswith(line, "ACOORD")
            A = zeros(T, totalconstr, totalvars)
            nextline = readline(fd)
            lines = parse(Int, strip(nextline))
            for k in 1:lines
                nextline = readline(fd)
                i, j, v = split(nextline)
                i = parse(Int,strip(i)) + 1
                j = parse(Int,strip(j)) + 1
                v = parse(Float64,strip(v))
                A[i, j] = v
            end
        end

        if startswith(line, "BCOORD")
            b = zeros(T, totalconstr, 1)
            nextline = readline(fd)
            lines = parse(Int, strip(nextline))
            for k in 1:lines
                nextline = readline(fd)
                i, v = split(nextline)
                i = parse(Int,strip(i)) + 1
                v = parse(Float64,strip(v))
                b[i] = v
            end
        end
    end

    if use_G
        # Need to split A into [-G; -A] and b into [-h; -b]
        # and swap G, h
        c *= objsense
        G_idxs = np.delete(np.arange(A.shape[0]), A_idxs)
        G = -_swap_rows(A[G_idxs], cones)
        h = -_swap_rows(b[G_idxs], cones)
        A = A[A_idxs]
        b = b[A_idxs]
    else
        # No G, just need to swap c and A
        c = _swap_rows(c, cones) * objsense
        A = _swap_rows(A', cones)'
        G = -one(T) * I
        h = zeros(T, totalvars)
    end
    c = c[:, 1]
    b = b[:, 1]
    h = h[:, 1]

    return Hypatia.Models.Model{T}(c, A, b, G, h, cones, obj_offset=obj_offset)

end

# Input into model and solve
file_name = "test.cbf"

model = read_problem(file_name)
solver = Solvers.Solver{T}(verbose = true)
Solvers.load(solver, model)
Solvers.solve(solver)