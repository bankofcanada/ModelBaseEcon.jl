
"""
    Model for testing steady state solver with linear growth variables.
"""
module E6

using ModelBaseEcon

model = Model()

@parameters model begin
    p_dlp = 0.0050000000000000 
    p_dly = 0.0045000000000000 
end

@variables model begin
    dlp; dly; dlyn; lp; ly; lyn
end

@shocks model begin
    dlp_shk; dly_shk
end

@autoexogenize model begin
    ly = dly_shk
    lp = dlp_shk
end

@equations model begin
    dly[t]=(1-0.2-0.2)*p_dly+0.2*dly[t-1]+0.2*dly[t+1]+dly_shk[t]
    dlp[t]=(1-0.5)*p_dlp+0.1*dlp[t-2]+0.1*dlp[t-1]+0.1*dlp[t+1]+0.1*dlp[t+2]+0.1*dlp[t+3]+dlp_shk[t]
    dlyn[t]=dly[t]+dlp[t]
    ly[t]=ly[t-1]+dly[t]
    lp[t]=lp[t-1]+dlp[t]
    lyn[t]=lyn[t-1]+dlyn[t]
 end

@initialize model

end
