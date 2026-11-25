##################################################################################
# This file is part of ModelBaseEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2025, Bank of Canada
# All rights reserved.
##################################################################################

# Package quality checks, https://github.com/JuliaTesting/Aqua.jl
using Aqua

@testset "Aqua" begin
    Aqua.test_all(ModelBaseEcon;
        # get(d, i) returns the i-th value of an ordered dictionary (misc.jl);
        # user code relies on it, so it stays until the maintainers decide otherwise
        piracies=(treat_as_own=[ModelBaseEcon.OrderedDict, ModelBaseEcon.LittleDict],),
    )
end
