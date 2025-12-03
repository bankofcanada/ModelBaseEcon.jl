
ST = ModelBaseEcon.SimpleTensors

@testset "simpletensors" begin

    for N = 0:5, d = 1:5, kind = (:dense, :sparse)
        A = ST.Tensor(N, d, kind)
        @test A isa AbstractArray{Float64,N}
        @test A isa ST.AbstractTensor{Float64,N}
        @test ndims(A) == N
        @test size(A) == ntuple((_) -> d, N)
        @test length(A) == d^N
        @test A[(1 for i = 1:N)...,] == 0
    end

    for N = 0:5, d = 1:5, kind = (:dense, :sparse)
        A = ST.SymmetricTensor(N, d, kind)
        @test A isa ST.AbstractSymmetricTensor
        @test ndims(A) == N
        @test size(A) == ntuple((_) -> d, N)
        @test length(A) == d^N
        @test A[(1 for i = 1:N)...,] == 0
    end

    A = randn(1000, 40)
    symA = A'A

    @test ST.SymmetricTensor(symA) isa ST.SymmetricTensor
    @test ST.SymmetricTensor(symA, :sparse) isa ST.SparseSymmetricTensor
    @test ST.SparseSymmetricTensor(symA) isa ST.SparseSymmetricTensor
    @test_throws r"Not supported kind=.*"i ST.SymmetricTensor(symA, :abcd) 
    @test ST.SymmetricTensor(symA, :dense) isa ST.DenseSymmetricTensor
    @test ST.DenseSymmetricTensor(symA) isa ST.DenseSymmetricTensor
    @test ST.SymmetricTensor(symA, :dense, check=true) isa ST.DenseSymmetricTensor
    symA[1,2] += 1e-7
    @test_throws r"Not symmetric.*"i ST.SymmetricTensor(symA, :dense, check=true) isa ST.DenseSymmetricTensor
    @test ST.SymmetricTensor(symA, :dense, check=true, tol=1e-6) isa ST.DenseSymmetricTensor
    @test ST.SymmetricTensor(symA, :dense, check=false) isa ST.DenseSymmetricTensor

    stA = ST.SymmetricTensor(symA)
    @test (stA[1,2] = 7; true)
    @test stA[1,2] == stA[2,1]
    @test stA[CartesianIndex((1,2))] == stA[2,1]
    @test ST.n_store(stA) == size(A,2)*(size(A,2)+1)//2


    @test ST.Tensor(symA) isa ST.Tensor
    @test ST.Tensor(symA, :dense) isa ST.DenseTensor
    @test ST.DenseTensor(symA) isa ST.DenseTensor
    @test ST.Tensor(symA, :sparse) isa ST.SparseTensor
    @test ST.SparseTensor(symA) isa ST.SparseTensor
    @test_throws r"Not supported kind=.*"i ST.Tensor(symA, :abcd) 

end

_eval23_tp(θ, x) = [
    (θ[1] + θ[2] * x[1] + θ[3] * x[2] + θ[4] * x[3] +
     θ[5] * x[1]^2 / 2 + θ[6] * x[1] * x[2] +
     θ[7] * x[1] * x[3] + θ[8] * x[2]^2 / 2 +
     θ[9] * x[2] * x[3] + θ[10] * x[3]^2 / 2)
]

_eval23_tp_dx1(θ, x) = [
    θ[2] + θ[5] * x[1] + θ[6] * x[2] + θ[7] * x[3]
    θ[3] + θ[6] * x[1] + θ[8] * x[2] + θ[9] * x[3]
    θ[4] + θ[7] * x[1] + θ[9] * x[2] + θ[10] * x[3]
]

_eval23_tp_dx2(θ, x) = [
    θ[5] θ[6] θ[7]
    θ[6] θ[8] θ[9]
    θ[7] θ[9] θ[10]
]

_eval23_tp_dθ(θ, x) = [1 x[1] x[2] x[3] x[1]^2 / 2 x[1] * x[2] x[1] * x[3] x[2]^2 / 2 x[2] * x[3] x[3]^2 / 2]

_eval23_tp_dx1dθ(θ, x) = [
    0 1 0 0 x[1] x[2] x[3] 0 0 0
    0 0 1 0 0 x[1] 0 x[2] x[3] 0
    0 0 0 1 0 0 x[1] 0 x[2] x[3]
]

_eval23_tp_dx2dθ(θ, x) = [
    0 0 0 0 1 0 0 0 0 0
    0 0 0 0 0 1 0 0 0 0
    0 0 0 0 0 0 1 0 0 0
    0 0 0 0 0 0 0 1 0 0
    0 0 0 0 0 0 0 0 1 0
    0 0 0 0 0 0 0 0 0 1
]

@testset "tp_func" begin
    # quadratic polynomial of 3 variables
    deg = 2
    nvars = 3
    y = ST.TaylorPolyFunc(deg, nvars)
    @test y.derivs isa ST.DerivsContainer
    @test length(y.derivs) == deg + 1
    for i = 0:deg
        @test iszero(y.derivs[i])
    end
    θ = ST.gettheta(y)
    for reps = 1:10
        y.x̄[:] = randn(3)
        θ[:] = 0.3 * randn(size(θ))
        ST.settheta!(y, θ)
        for reps = 1:10
            x = randn(3)
            @test y(x)[:] ≈ _eval23_tp(θ, x - y.x̄)
            @test y(Val(1), x)[:] ≈ _eval23_tp_dx1(θ, x - y.x̄)
            @test y(Val(2), x)[:, :] ≈ _eval23_tp_dx2(θ, x - y.x̄)
            @test ST.d_dtheta(y, x) ≈ _eval23_tp_dθ(θ, x - y.x̄)
            @test ST.d_dtheta(Val(1), y, x) ≈ _eval23_tp_dx1dθ(θ, x - y.x̄)
            @test ST.d_dtheta(Val(2), y, x) ≈ _eval23_tp_dx2dθ(θ, x - y.x̄)
        end
    end

    hod = nothing
    @test (hod = ST.eval_hod(y, y.x̄); true)
    @test hod isa ST.DerivsContainer
    @test length(hod) == 1+deg
    for i = 0:deg
        @test hod[i] isa ST.SymmetricTensor
        @test hod[i].data ≈ y.derivs[i].data
    end

end


@testset "tp_dtheta" begin
    # compare the computed derivatives with respect to θ with
    # ones computed by first difference
    dθ = 1e-6
    for deg = 0:5, nvars = 2:4
        fun = ST.TaylorPolyFunc(deg, nvars)
        θ = ST.gettheta(fun)
        xx = zeros(nvars)
        for _ = 1:10
            θ[:] = 0.3 * randn(size(θ))
            xx[:] = randn(nvars)
            for d = 0:deg
                ST.settheta!(fun, θ)
                a = fun(Val(d), xx).data
                a1 = zeros(length(a), length(θ))
                for i in axes(a1, 2)
                    tmp = θ[i]
                    θ[i] += dθ
                    ST.settheta!(fun, θ)
                    a1[:, i] = fun(Val(d), xx).data
                    θ[i] = tmp
                end
                d1 = (a1 .- a) ./ dθ
                d2 = ST.d_dtheta(Val(d), fun, xx)
                # @info "$deg  $nvars  $d   $(maximum(abs, d1-d2))"
                @test d1 ≈ d2 rtol=1e-8
            end
        end
    end
end


@testset "tp_dx" begin
    # compare the computed derivatives with respect to x with
    # ones computed by first difference
    dx = 1e-7
    for deg = 0:3, nvars = 2:10
        fun = ST.TaylorPolyFunc(deg, nvars)
        θ = ST.gettheta(fun)
        xx = zeros(nvars)
        for _ = 1:1
            θ[:] = 0.3 * randn(size(θ))
            xx[:] = randn(nvars)
            for d = 0:deg-1
                ST.settheta!(fun, θ)
                a = Vector(fun(Val(d), xx)[:])
                a1 = zeros(length(a), nvars)
                for i in axes(a1, 2)
                    tmp = xx[i]
                    xx[i] += dx
                    a1[:, i] = fun(Val(d), xx)[:]
                    xx[i] = tmp
                end
                d1 = (a1 .- a) ./ dx
                d2 = reshape(fun(Val(d+1), xx)[:], :, nvars)
                # @info "$deg  $nvars  $d   $(norm(d1-d2)/norm(d1))"
                @test d1 ≈ d2 rtol=1e-7 atol=1e-7
            end
        end
    end
end

nothing
