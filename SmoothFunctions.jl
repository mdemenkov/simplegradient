module SmoothFunctions

using LinearAlgebra, HDF5

export SmoothFunction, Gradient, Quadratic, ArmijoParams, GradientMethod, save, load_quadratic

const MaxIter=10000
const DispIter=100
const MinGrad=1e-4
const GradStep=1e-4

abstract type SmoothFunction end

struct ArmijoParams
       s::Float64
       β::Float64
       σ::Float64
end

mutable struct Gradient
# Structure to avoid reallocating gradient vector on each iteration
        x # current point
        x_predictor # next possible point
        g  # For storing gradient
        x_plus_dx  #
        x_minus_dx # For computing gradient numerically
        d # Direction of descent 
        function Gradient(n::Integer) # Constructor
            new(zeros(n),zeros(n),zeros(n),zeros(n),zeros(n),zeros(n))
        end
end    

function inf_norm(x::Vector{Float64})
# Infinity norm
         max_val=0.0
         for i=1:length(x)
             val=abs(x[i])
             if max_val<val max_val=val end
         end
         return max_val
end

function SetDescentDirection(G::Gradient)
# Store the antigradient    
         for i=1:length(G.g)
             G.d[i]=-G.g[i]
         end
end

struct Quadratic<:SmoothFunction
    Q::Matrix{Float64}
    function Quadratic(Q) # Constructor
        isposdef(Q) ? new(Q) : error("Matrix is not positive definite")
    end
end

function save(F::Quadratic,fname)
    h5open(fname, "w") do file
        write(file, "Q", F.Q)
    end
end

function load_quadratic(fname)
# Create new quadratic function from a previously stored file
    Q=h5read(fname,"Q")
    F=Quadratic(Q)
    return F
end

function (F::Quadratic)(x::Vector{Float64})
# return function value s.t. if F=Quadratic(Q), then F(x)=x'Qx
    return (x'*F.Q*x)[1] # A trick to return scalar value
end

function ∇(F::Quadratic,G::Gradient)
# Gradient by formula for quadratic functions
    G.g=2.0*F.Q*G.x 
    return G.g
end

function Δ(F::SmoothFunction,G::Gradient) 
# Gradient computed numerically for any smooth function
         for i=1:length(G.x) 
             G.x_plus_dx[i]=G.x[i]
             G.x_minus_dx[i]=G.x[i] 
         end
         for i=1:length(G.x)
             G.x_plus_dx[i]+=GradStep
             G.x_minus_dx[i]-=GradStep
             G.g[i]=(F(G.x_plus_dx)-F(G.x_minus_dx))/(2.0*GradStep)
             G.x_plus_dx[i]=G.x[i]
             G.x_minus_dx[i]=G.x[i] 
         end
    return G.g
end

function ArmijoRule(par::ArmijoParams,F::SmoothFunction,G::Gradient)
# from D.P. Bertsekas, "Constrained optimization and Lagrange multiplier methods"
# d is descent direction s.t. <d,∇F(x)> <0
            fxk=F(G.x)
            rhs=dot(G.g,G.d)*par.s*par.σ
            βm=1
            for i=1:length(G.x) G.x_predictor[i]=G.x[i]+G.d[i]*par.s end
            while (F(G.x_predictor)-fxk)>βm*rhs
                  βm=βm*par.β
                  for i=1:length(G.x) G.x_predictor[i]=G.x[i]+βm*G.d[i]*par.s end
            end
            for i=1:length(G.x) G.x[i]=G.x_predictor[i] end
end

function GradientMethod(F::SmoothFunction,x0::Vector{Float64},par::ArmijoParams;analytic_gradient=false,verbose=false)
# with Armijo rule 
    
    G=Gradient(length(x0))
    for i=1:length(x0) G.x[i]=x0[i] end
    iter=0
    norm_is_small=false
    if analytic_gradient
         grad=∇
         println("Computing with analytic gradient")
    else 
         grad=Δ
         println("Computing with numerically estimated gradient")
    end
    iter_count=DispIter
    for k=1:MaxIter
        iter=k
        norm_is_small=inf_norm(grad(F,G))≤MinGrad 
        if norm_is_small break end
        SetDescentDirection(G)
        ArmijoRule(par,F,G)
        if verbose
        # display some progress
            if k>iter_count
                 iter_count+=DispIter
            elseif k==iter_count
                fval=F(G.x)
                gradnorm=inf_norm(G.g)
                printstyled("Iteration $(k), function value=$(fval), gradient norm=$(gradnorm)\n"; color = :green)
            end            
        end
    end

    if norm_is_small
           println("Exiting because gradient is too small")
    else
        println("Maximum iteration number reached") 
    end
    println("Iteration=",iter)
    println("Function value=",F(G.x))
    return G.x
end

function __init__()

end

end