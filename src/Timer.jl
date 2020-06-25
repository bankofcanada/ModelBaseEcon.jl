"""
    Timer

A sub-module of ModelBaseEcon, although it can be used independently.
Provides functionality for measuring the aggregate time spent
in individual lines or blocks of code.

!!! tip

    The implementation here is quick-and-dirty and is intended for getting
    a rough idea of where the bottlenecks are. It is best used for timing
    blocks of code that are passed through relatively few times and each pass
    takes a relatively long time. In the opposite extreme case (fast code that
    is called many times), the current implementation of `@timer` might add
    extreme overhead.

# Contents

  * [`inittimer`](@ref)       - Enable collection of timer data.
  * [`stoptimer`](@ref)       - Disable collection of timer data.
  * [`printtimer`](@ref)      - Display timer data.
  * [`@timer`](@ref)          - Measure the runtime taken by the given code.

# Example
```jldoctest
julia> true
[...]
```
"""
module Timer

using Printf

export @timer, inittimer, printtimer, stoptimer

"""
    timerData 
    
Stores timing data.
    
!!! note

    For internal use. Do not modify directly.

If equal to `nothing`, timing is disabled.
Otherwise, contains a "database" of timimng data in the 
form of a Dict.
"""
global timerData = nothing

"""
    inittimer()

Enable the collection of timing data. Existing timing data is lost.
By default, collection of timing data is disabled.

See also: [`stoptimer`](@ref), [`@timer`](@ref), [`printtimer`](@ref)
"""
function inittimer()
    global timerData = Dict()
    return nothing
end

"""
    stoptimer()

Disable the collection of timing data. Existing data is lost.
By default, collection of timing data is disabled.

See also: [`inittimer`](@ref), [`@timer`](@ref), [`printtimer`](@ref)
"""
function stoptimer()
    global timerData = nothing
    return nothing
end

"""
    printtimer(io::IO=Base.stdout)

Display timing data.

Timing data is displayed in a table with each row containing the
number of calls, total time in seconds, and the source line or block tag.
Rows are sorted in order of decreasing total time.

See also: [`@timer`](@ref)
"""
function printtimer(io::IO=Base.stdout)
    global timerData    
    if timerData !== nothing
        @printf(io, "%10s  %10s  %s\n", "Calls", "Seconds", "Source")
        items = collect(timerData)
        items = sort(items, lt=(l,r)->r[2][:seconds]<l[2][:seconds])
        for (src, val) ∈ items
            @printf(io, "% 10d  %10.3f  %s\n", val[:calls], val[:seconds], src)
        end
    end
    return nothing
end

"""
    @timer(code)
    @timer(tag::String, code)

Measure the number of calls and the total time taken by the given code.

If a `tag` string is not provided, one is generated from the source file and line.
The return value of this macro call is the return value of the code.

!!! warning
    Important limitation is that the code must not contain a `return`, `break`,
    `continue`, or any other jump out of it. If it does, the program would run
    correctly, but the timing data collected would be incorrect.

See also: [`inittimer`](@ref), [`stoptimer`](@ref), [`printtimer`](@ref)

# Example

```jldoctest
julia> inittimer()

julia> @timer Base.sleep(1.0)

julia> printtimer()
[...]
```
"""
macro timer(args...)
    if length(args) == 1
        name = QuoteNode(__source__)
        code = args[1]
    elseif length(args) == 2
        name = string(args[1])
        code = args[2]
    else
        error("Too many arguments")
    end
    return quote
        global timerData
        local val
        if timerData === nothing
            val = $(esc(code))
        else
            local t = time()
            val = $(esc(code))
            t = time() - t
            local td = get!(timerData, $name, Dict(:calls=>0, :seconds=>0.0))
            td[:calls] += 1
            td[:seconds] += t
            timerData[$name] = td
        end
        val
    end
end

end # module