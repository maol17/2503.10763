using LinearAlgebra
using Plots
using Plots.PlotMeasures
using DifferentialEquations
using LaTeXStrings

#---------------------------
function random_init(N::Int)
    h = randn(N,N) + 1im*randn(N,N)
    h = h + h'
    v, U = eigen(h)
    return U*diagm((sign.(v).+1)./2)*U'
end

#random damping matrix for TRS=σz (system size 2N)
function random_H(N::Int)
    g = zeros(ComplexF64,2N,2N)
    for i in 1:2N
        for j in 1:2N
            if mod(abs(i-j),2) == 1
                if i>j
                    g[i,j] = 1im
                else
                    g[i,j] = -1im
                end
            end
        end
    end

    h1 = randn(2N,2N)
    h1 = h1 + h1'

    return g.*h1
end

#TRS
function random_M_trs(N::Int)
    g = ones(ComplexF64,2N,2N)
    for i in 1:N
        for j in 1:2N
            g[2i,j] = 1im
        end
    end

    d1 = g.*randn(2N,2N)
    d2 = g.*randn(2N,2N)

    ml = d1*d1' 
    mg = d2*d2' 
    return ml, mg
end

#PHS
function random_M_phs(N::Int)
    g = diagm(ones(2N))
    for i in 1:N
        g[2i,2i] = -1
    end

    ml = randn(2N,2N)
    mg = g*ml
    return ml*ml', mg*mg'
end

#fidelity of TRS
function fd_trs(c::Matrix, N::Int)
    u = diagm(ones(2N))
    for i in 1:N
        u[2i,2i] = -1
    end
    ct = Matrix(transpose(c))
    return sum(abs.(u*ct*u-c))
end

#phs
function fd_phs(c::Matrix, N::Int)
    u = diagm(ones(2N))
    id = copy(u)
    for i in 1:N
        u[2i,2i] = -1
    end
    ct = Matrix(transpose(c))
    return sum(abs.(u*ct*u+c-id))
end

#evolution
f(u, p, t) = p[1]*u+u*(p[1])'+2*p[2]

function evolution1_diff(N::Int, tspan::Array)
    H = random_H(N)
    Ml, Mg = random_M_trs(N)
    X = 1im*Matrix(transpose(H))-Matrix(transpose(Ml))-Mg
    C0 = random_init(2N)
    t = (0.0,tspan[end])
    prob = ODEProblem(f, C0, t, [X, Mg])
    sol = solve(prob,reltol=1e-12,abstol=1e-12)
    
    fs = Array{Float64}(undef, length(tspan))
    for i in 1:length(tspan)
        fs[i] = fd_trs(sol(tspan[i]),N)
    end
    return fs
end

function evolution2_diff(N::Int, tspan::Array)
    H = random_H(N)
    Ml, Mg = random_M_phs(N)
    X = 1im*Matrix(transpose(H))-Matrix(transpose(Ml))-Mg
    C0 = random_init(2N)
    t = (0.0,tspan[end])
    prob = ODEProblem(f, C0, t, [X, Mg])
    sol = solve(prob,reltol=1e-12,abstol=1e-12)
    
    fs = Array{Float64}(undef, length(tspan))
    for i in 1:length(tspan)
        fs[i] = fd_phs(sol(tspan[i]),N)
    end
    return fs
end

function statistics1(N::Int, M::Int, tspan::Array)
    l = length(tspan)
    fa = zeros(l)
    fmax = zeros(l)
    fmin = ones(l)
    for i in 1:M
        fs = evolution1_diff(N,tspan)./(2N*2N)
        fa .= fa .+ fs
        for i in 1:l
            if fs[i]>fmax[i]
                fmax[i] = fs[i]
            end
            if fs[i]<fmin[i]
                fmin[i] = fs[i]
            end
        end
    end
    fa = fa ./ M
    return fa, fmax, fmin
end

function statistics2(N::Int, M::Int, tspan::Array)
    l = length(tspan)
    fa = zeros(l)
    fmax = zeros(l)
    fmin = ones(l)
    for i in 1:M
        fs = evolution2_diff(N,tspan)./(2N*2N)
        fa .= fa .+ fs
        for i in 1:l
            if fs[i]>fmax[i]
                fmax[i] = fs[i]
            end
            if fs[i]<fmin[i]
                fmin[i] = fs[i]
            end
        end
    end
    fa = fa ./ M
    return fa, fmax, fmin
end



#-------------------------
N = 50
M = 100
tspan = collect(0:0.001:0.05)

fa1, fmax1, fmin1 = statistics1(N,M,tspan)

p1=plot(tspan,fa1, ribbon=(fa1.-fmin1, fmax1.-fa1)
,xlabel=L"t",framestyle=:box,label="",labelfontsize=14,tickfontsize=10,tickfont="Times",dpi=400,ylim=(-0.001,0.058),lw=0.1)

fa2, fmax2, fmin2 = statistics2(N,M,tspan)

p2=plot(tspan,fa2, ribbon=((fa2.-fmin2, fmax2.-fa2)),xlabel=L"t"
,framestyle=:box,label="",labelfontsize=14,tickfontsize=10,tickfont="Times",dpi=400,ylim=(-0.001,0.058),lw=0.1)


p = plot(p1,p2,layout=(1,2),dpi=400,size=(600,250),leftmargin=0px,rightmargin=-1px,topmargin=-3px,bottommargin=8px,labelfontsize=17,tickfontsize=9)
savefig(p,"dynamics.pdf")

