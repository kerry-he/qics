import SparseArrays

import Hypatia
import Hypatia.Cones
import Hypatia.Solvers

T = Float64

function read_problem(filename)
    # Read data from file
    fd = open(filename, "r")

    while !eof(fd)
        line = readline(fd)
        startswith(line,"#") && continue # comments
        length(line) == 1 && continue # blank lines

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
            varcnt = 0

            for k in 1:lines
                nextline = readline(fd)
                cone, sz = split(nextline)
                sz = parse(Int,strip(sz))

                if cone == "L+"
                    
                elseif cone == "Q"
                    
                elseif cone == "SVECPSD"
                    
                elseif cone == "HVECPSD"
                    
                elseif cone == "CE"
                    
                elseif cone == "CRE"
                    
                elseif cone == "SVECQE"
                    
                elseif cone == "HVECQE"
                    
                elseif cone == "SVECQRE"
                    
                elseif cone == "HVECQRE"
                    
                end

                varcnt += sz
            end
            
            continue
        end

        if startswith(line, "CON")
            nextline = readline(fd)
            totalconstr, lines = split(nextline)
            totalconstr = parse(Int,strip(totalconstr))
            lines = parse(Int,strip(lines))
            constrcnt = 0

            for k in 1:lines
                nextline = readline(fd)
                cone, sz = split(nextline)
                sz = parse(Int,strip(sz))
                push!(dat.con, (cone, sz))
                constrcnt += sz
            end
            @assert totalconstr == constrcnt
            dat.nconstr = constrcnt
            continue
        end

        if startswith(line, "OBJACOORD")
            parse_matblock(fd,dat.objacoord,1)
        end

        if startswith(line, "OBJBCOORD")
            nextline = readline(fd)
            dat.objoffset = parse(Float64, strip(nextline))
            @warn "Instance has objective offset"
        end

        if startswith(line, "ACOORD")
            parse_matblock(fd,dat.acoord,2)
        end

        if startswith(line, "BCOORD")
            parse_matblock(fd,dat.bcoord,1)
        end
    end

    return dat

end

# Input into model and solve
file_name = "test.cbf"

model = read_problem(file_name)
