
"""
    A version of E7 for testing linearization
"""
module E7A

using ModelBaseEcon

model = Model()
model.options.substitutions = false

@parameters model begin
    delta = 0.1000000000000000 
    p_dlc_ss = 0.0040000000000000 
    p_dlinv_ss = 0.0040000000000000 
    p_growth = 0.0040000000000000 
end

@variables model begin
    dlc; dlinv; dly; lc; linv;
    lk; ly;
end

@shocks model begin
    dlc_shk; dlinv_shk;
end

@autoexogenize model begin
    lc = dlc_shk
    linv = dlinv_shk
end

@equations model begin
    dlc[t] = (1 - 0.2 - 0.2) * p_dlc_ss + 0.2 * dlc[t - 1] + 0.2 * dlc[t + 1] + dlc_shk[t]
    dlinv[t] = (1 - 0.5) * p_dlinv_ss + 0.1 * dlinv[t - 2] + 0.1 * dlinv[t - 1] + 0.1 * dlinv[t + 1] + 0.1 * dlinv[t + 2] + 0.1 * dlinv[t + 3] + dlinv_shk[t]
    lc[t] = lc[t - 1] + dlc[t]
    linv[t] = linv[t - 1] + dlinv[t]
    ly[t] = log(exp(lc[t]) + exp(linv[t]))
    dly[t] = ly[t] - ly[t - 1]
    lk[t] = log((1 - delta) * exp(lk[t - 1]) + exp(linv[t]))
end

@initialize model

@steadystate model linv = lc - 7;
@steadystate model lc = 14;

end