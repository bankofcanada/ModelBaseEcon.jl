


export SteadyStateEquation

"""
    struct SteadyStateEquation <: AbstractEquation

Data structure representing an individual steady state equation.
"""
struct SteadyStateEquation <: AbstractEquation
    type::Symbol
    vinds::Vector{Int64}
    vsyms::Vector{Symbol}
    expr::ExtExpr
    eval_resid::Function
    eval_RJ::Function
    # The default constructor
    SteadyStateEquation(type, vinds, vsyms, expr, eval_resid, eval_RJ) = 
            new(type, vinds, vsyms, expr, eval_resid, eval_RJ)
    # # Constructor that automatically adds `eval_RJ` from `eval_resid`
    # SteadyStateEquation(type, vinds, vsyms, expr, eval_resid) = 
    #         new(type, vinds, vsyms, expr, eval_resid, make_eval_RJ(eval_resid, length(vinds)))
end


########################################################

export SteadyStateData

"""
    SteadyStateData

Data structure that holds information about the steady state solution of the Model.
"""
struct SteadyStateData
    "List of steady state variables."
    vars::Vector{Symbol}
    "Steady state solution vector."
    values::Vector{Float64}
    "`mask[i] == true if and only if `values[i]` holds the steady state value."
    mask::Vector{Bool}
    "Steady state equations derived from the dynamic system."
    equations::Vector{SteadyStateEquation}
    "Steady state equations explicitly added separately from the dynamic system."
    constraints::Vector{SteadyStateEquation}
    # constructor
    SteadyStateData() = new([], [], [], [], [])
end

export alleqn
"""
    alleqn(ssd::SteadyStateData)

Return a list of all steady state equations. 

The list contains all equations derived from the dynamic system and all explicitly added steady state constraints.
"""
@inline alleqn(ssd::SteadyStateData) = vcat(ssd.equations, ssd.constraints)

Base.show(io::IO, ssd::SteadyStateData) = print(io, length(ssd.constraints), " Steady State Constraints:\n    ", join(ssd.constraints, "\n    "))


#####
# These are used for indexing in the values vector

@inline makesym(::Val{:level}, var::Symbol) = Symbol("$(var)#lvl")
@inline makesym(::Val{:slope}, var::Symbol) = Symbol("$(var)#slp")
@inline makeind(::Val{:level}, ind::Int64) = 2*ind-1
@inline makeind(::Val{:slope}, ind::Int64) = 2*ind

#########
# Implement access to steady state values using index notation ([])

struct SSMissingVariableError <: ModelErrorBase
    var::Symbol
end
msg(me::SSMissingVariableError) = "Steady state variable $(me.var) not found."

# ssvarindex() - return the index of a steady state value given its Symbol
# If the symbol might end in #lvl or #slp, if not #lvl is assumed
function ssvarindex(v::Symbol, vars::AbstractArray{Symbol})
    ind = indexin([v], vars)[1]
    if ind === nothing
        ind = indexin([makesym(Val(:level), v)], vars)[1]
        if ind === nothing
            throw(SSMissingVariableError(v))
        end
    end
    return ind
end

@inline Base.getindex(sstate::SteadyStateData, var::String) = getindex(sstate, Symbol(var))
Base.getindex(sstate::SteadyStateData, var::Symbol) = getindex(getfield(sstate, :values), ssvarindex(var, sstate.vars))

Base.setindex!(sstate::SteadyStateData, val, var::String) = setindex!(sstate, val, Symbol(var))
Base.setindex!(sstate::SteadyStateData, val, var::Symbol) = setindex!(getfield(sstate,:values), val, ssvarindex(var, sstate.vars))

########
# Implement access to steady state values using dot notation

function Base.getproperty(ssd::SteadyStateData, name::Symbol)
    if name ∈ fieldnames(SteadyStateData)
        return getfield(ssd, name)
    else
        return getindex(ssd, name)
    end
end

function Base.setproperty!(ssd::SteadyStateData, name::Symbol, val)
    if name ∈ fieldnames(SteadyStateData)
        return setfield!(ssd, name, val)
    else
        return setindex!(ssd, val, name)
    end
end

function Base.propertynames(ssd::SteadyStateData, private=false)
    if private
        return ((Symbol(split("$v", "#")[1]) for v in ssd.vars[1:2:end])..., fieldnames(SteadyStateData)...)
    else
        return ((Symbol(split("$v", "#")[1]) for v in ssd.vars[1:2:end])..., )
    end
end

#########################
#

export printsstate

"""
    printsstate([io::IO,] ssd::SteadyStateData)

Display steady state solution.

Steady state solution is presented in a table, where the first column is
the name of the variable, the second and third columns are the corresponding
values of the level and the slope. If the value is not determined
(as per its `mask` value) then it is displayed as "*".
"""
function printsstate(io::IO, ssd::SteadyStateData)
    @printf io "  % 30s   %-15s \t%-15s\n" "" "level" "slope"
    printmv(m,v) = m ? @sprintf("%- 15g", v) : @sprintf("%15s", " * ")
    for i = 1:2:length(ssd.values)
        var = split(string(ssd.vars[i]), '#')[1]
        val1 = printmv(ssd.mask[i], ssd.values[i])
        val2 = printmv(ssd.mask[i+1], ssd.values[i+1])
        @printf io "  % 30s = %s \t%s\n" var val1 val2
    end
end
printsstate(ssd::SteadyStateData) = printsstate(Base.stdout, ssd)

# printsstate(io::IO, m::Model) = printsstate(io, m.sstate)
# printsstate(m::Model) = printsstate(m.sstate)

###########################
# Make steady state equation from dynamic equation

# Idea:
#   in the steady state equation, we assume that the variable y_ss
#   follows a linear motion expressed as y_ss[t] = y_ss#lvl + t * y_ss#slp
#   where y_ss#lvl and y_ss#slp are two unknowns we solve for.
#
#   The dynamic equation has mentions of lags and leads. We replace those
#   with the above expression.
#
#   Since we have two parameters to determine, we need two steady state equations
#   from each dynamic equation. We get this by writing the dynamic equation at
#   two different values of `t` - 0 and another one we call `shift`. 
#
#   Shift is a an option in the model object, which the user can set to any integer
#   other than 0. The default is 10.

"""
    SSEqnData

Internal structure used for evaluation of the residual of the steady state equation
derived from a dynamic equation.
"""
struct SSEqnData
    "Value of `shift` for the current steady state equation"
    shift::Int64
    "A matrix which translates the values of the steady state into the lagged(lead) values for the dynamic equation"
    JT::Array{Float64,2}
    "The dynamic equation instance"
    eqn::Equation
end

function sseqn_resid_RJ(s::SSEqnData)::Function
    function _resid(pt::AbstractArray{Float64,1})
        return s.eqn.eval_resid(s.JT*pt)
    end
    function _RJ(pt::AbstractArray{Float64,1})
        R, jj = s.eqn.eval_RJ(s.JT*pt)
        return R, vec(jj'*s.JT)
    end
    return _resid, _RJ
end

"""
    make_sseqn(model::AbstractModel, eqn::Equation; shift::Int64=0)

Create a steady state equation from the given dynamic equation for the given model.

!!! note
    Internal function, do not call directly.
"""
function make_sseqn(model::AbstractModel, eqn::Equation; shift::Int64=0)
    vinds = Int64[]
    nvariables = length(model.variables)
    nshocks = length(model.shocks)
    nauxvars = length(model.auxvars)
    # ssind converts the dynamic index (t, v) into 
    # the corresponding indexes of steady state unknowns. 
    # Returned value is a list of length 0, 1, or 2.
    function ssind((ti,vi),)::Array{Int64,1}
        if nvariables < vi <= nvariables+nshocks
            # The mentioned variable is a shock. No steady state unknown for it.
            return []
        else
            # In the dynamic equation the indexing goes - variables, shocks, auxvars
            # In the steady state equation the indexing goes - variables, auxvars
            # So, if the dynamic variable is an auxvar, we have to subtract nshocks.
            if vi > nvariables
                vi -= nshocks
            end
            # The level unknown has index 2*vi-1.
            # The slope unknown has index 2*vi, but it in the equation only if its coefficient is not 0.
            if ti+shift == 0
                return [2vi-1]
            else
                return [2vi-1, 2vi]
            end
        end
    end
    # The steady state indexes.
    vinds = unique(vcat(map(ssind, eqn.vinds)...))
    # The corresponding steady state symbols
    vsyms = model.sstate.vars[vinds]
    # In the next loop we build the matrix JT which transforms
    # from the steady state values to the dynamic point values. 
    JT = zeros(length(eqn.vinds), length(vinds))
    for (i, (ti,vi)) in enumerate(eqn.vinds)
        # for each index in the dynamic equation, we have a row
        # with all zeros, except at the columns of the level and slope.
        # The coefficient for the level is 1.0 and for the slope is ti+shift.
        # Note that the level is always an unknown, but the slope may not be, 
        # depending if ti+shift is 0.
        if nvariables < vi <= nvariables+nshocks
            continue
        else
            if vi > nvariables
                vi -= nshocks
            end
            vi_lvl = indexin([vi*2-1], vinds)[1]
            vi_slp = indexin([vi*2], vinds)[1]
            JT[i,vi_lvl] = 1.0
            if vi_slp !== nothing
                JT[i,vi_slp] = ti+shift
            end
        end
    end
    type = shift==0 ? :tzero : :tshift
    let sseqndata = SSEqnData(shift, JT, eqn)
        return SteadyStateEquation(type, vinds, vsyms, :($type => $(eqn.expr)), 
            sseqn_resid_RJ(sseqndata)...)
    end
end
