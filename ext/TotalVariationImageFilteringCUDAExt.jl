module TotalVariationImageFilteringCUDAExt

using CUDA
using TotalVariationImageFiltering

import TotalVariationImageFiltering:
    AbstractPrimalConstraint,
    AbstractBoundaryCondition,
    AbstractDataFidelity,
    AbstractTVMode,
    AnisotropicTV,
    IsotropicTV,
    L2Fidelity,
    Neumann,
    Periodic,
    PDHGConfig,
    PDHGState,
    PoissonFidelity,
    NoConstraint,
    ROFConfig,
    SolverStats,
    divergence!,
    gradient!,
    _group_sum_axis!,
    project_dual_ball!,
    solve!,
    solve_batch!

const DEFAULT_THREADS = 256

@inline function _launch_config(len::Int)
    return DEFAULT_THREADS, cld(len, DEFAULT_THREADS)
end

function _gradient_dim_kernel!(gd, u, scale, stride_d::Int, n_d::Int, len::Int)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if index <= len
        i = Int(index)
        coord_d = ((i - 1) ÷ stride_d) % n_d + 1
        @inbounds if coord_d < n_d
            gd[i] = scale * (u[i+stride_d] - u[i])
        else
            gd[i] = zero(scale)
        end
    end
    return nothing
end

function _divergence_dim_kernel!(out, pd, scale, stride_d::Int, n_d::Int, len::Int)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if index <= len
        i = Int(index)
        coord_d = ((i - 1) ÷ stride_d) % n_d + 1
        acc = zero(scale)
        @inbounds if coord_d < n_d
            acc += scale * pd[i]
        end
        @inbounds if coord_d > 1
            acc -= scale * pd[i-stride_d]
        end
        @inbounds out[i] += acc
    end
    return nothing
end

function _gradient_periodic_dim_kernel!(gd, u, scale, stride_d::Int, n_d::Int, len::Int)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if index <= len
        i = Int(index)
        coord_d = ((i - 1) ÷ stride_d) % n_d + 1
        @inbounds if n_d == 1
            gd[i] = zero(scale)
        elseif coord_d < n_d
            gd[i] = scale * (u[i+stride_d] - u[i])
        else
            gd[i] = scale * (u[i-(n_d-1)*stride_d] - u[i])
        end
    end
    return nothing
end

function _divergence_periodic_dim_kernel!(out, pd, scale, stride_d::Int, n_d::Int, len::Int)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if index <= len
        i = Int(index)
        coord_d = ((i - 1) ÷ stride_d) % n_d + 1
        @inbounds if n_d > 1
            previous = coord_d > 1 ? pd[i-stride_d] : pd[i+(n_d-1)*stride_d]
            out[i] += scale * (pd[i] - previous)
        end
    end
    return nothing
end

function _group_sum_axis_kernel!(
    out,
    input,
    stride_d::Int,
    n_d::Int,
    first_offset::Int,
    last_offset::Int,
    periodic::Bool,
    len::Int,
)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if index <= len
        i = Int(index)
        coord_d = ((i - 1) ÷ stride_d) % n_d + 1
        acc = zero(eltype(input))
        @inbounds for offset = first_offset:last_offset
            source_coord = coord_d + offset
            if periodic
                source_coord = mod(source_coord - 1, n_d) + 1
                acc += input[i+(source_coord-coord_d)*stride_d]
            elseif 1 <= source_coord <= n_d
                acc += input[i+offset*stride_d]
            end
        end
        @inbounds out[i] = acc
    end
    return nothing
end

function _project_isotropic_kernel!(p, radius, radius2, len::Int)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if index <= len
        i = Int(index)
        nrm2 = zero(radius)
        @inbounds for d = 1:length(p)
            v = p[d][i]
            nrm2 = muladd(v, v, nrm2)
        end

        @inbounds if nrm2 > radius2
            inv_nrm = radius / sqrt(nrm2)
            for d = 1:length(p)
                p[d][i] *= inv_nrm
            end
        end
    end
    return nothing
end

function _project_anisotropic_kernel!(p, lower, upper, len::Int)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if index <= len
        i = Int(index)
        @inbounds for d = 1:length(p)
            p[d][i] = clamp(p[d][i], lower, upper)
        end
    end
    return nothing
end

function _dual_update_if_active_kernel!(
    pd,
    grad_pd,
    tau,
    done_mask,
    batch_stride::Int,
    batch_size::Int,
    len::Int,
)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if index <= len
        i = Int(index)
        batch_idx = ((i - 1) ÷ batch_stride) % batch_size + 1
        @inbounds if done_mask[batch_idx] == UInt8(0)
            pd[i] = pd[i] + tau * grad_pd[i]
        end
    end
    return nothing
end

function _dual_update_kernel!(pd, grad_pd, sigma, len::Int)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if index <= len
        i = Int(index)
        @inbounds pd[i] = pd[i] + sigma * grad_pd[i]
    end
    return nothing
end

function _pdhg_primal_tmp_kernel!(primal_tmp, u, divp, tau, len::Int)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if index <= len
        i = Int(index)
        @inbounds primal_tmp[i] = u[i] + tau * divp[i]
    end
    return nothing
end

function _pdhg_primal_tmp_if_active_kernel!(
    primal_tmp,
    u,
    divp,
    tau,
    done_mask,
    batch_stride::Int,
    batch_size::Int,
    len::Int,
)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if index <= len
        i = Int(index)
        batch_idx = ((i - 1) ÷ batch_stride) % batch_size + 1
        @inbounds if done_mask[batch_idx] == UInt8(0)
            primal_tmp[i] = u[i] + tau * divp[i]
        end
    end
    return nothing
end

function _pdhg_prox_l2_kernel!(u, primal_tmp, f, tau, len::Int)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if index <= len
        i = Int(index)
        denom = one(tau) + tau
        @inbounds u[i] = (primal_tmp[i] + tau * f[i]) / denom
    end
    return nothing
end

function _pdhg_prox_l2_if_active_kernel!(
    u,
    primal_tmp,
    f,
    tau,
    done_mask,
    batch_stride::Int,
    batch_size::Int,
    len::Int,
)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if index <= len
        i = Int(index)
        batch_idx = ((i - 1) ÷ batch_stride) % batch_size + 1
        @inbounds if done_mask[batch_idx] == UInt8(0)
            denom = one(tau) + tau
            u[i] = (primal_tmp[i] + tau * f[i]) / denom
        end
    end
    return nothing
end

function _pdhg_prox_poisson_kernel!(u, primal_tmp, f, tau, len::Int)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if index <= len
        i = Int(index)
        four_tau = (one(tau) + one(tau) + one(tau) + one(tau)) * tau
        two = one(tau) + one(tau)
        @inbounds begin
            shifted = primal_tmp[i] - tau
            candidate = (shifted + sqrt(shifted * shifted + four_tau * f[i])) / two
            u[i] = ifelse(candidate < zero(tau), zero(tau), candidate)
        end
    end
    return nothing
end

function _pdhg_prox_poisson_if_active_kernel!(
    u,
    primal_tmp,
    f,
    tau,
    done_mask,
    batch_stride::Int,
    batch_size::Int,
    len::Int,
)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if index <= len
        i = Int(index)
        batch_idx = ((i - 1) ÷ batch_stride) % batch_size + 1
        @inbounds if done_mask[batch_idx] == UInt8(0)
            four_tau = (one(tau) + one(tau) + one(tau) + one(tau)) * tau
            two = one(tau) + one(tau)
            shifted = primal_tmp[i] - tau
            candidate = (shifted + sqrt(shifted * shifted + four_tau * f[i])) / two
            u[i] = ifelse(candidate < zero(tau), zero(tau), candidate)
        end
    end
    return nothing
end

function _pdhg_ubar_kernel!(u_bar, u, u_prev, theta, len::Int)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if index <= len
        i = Int(index)
        @inbounds u_bar[i] = u[i] + theta * (u[i] - u_prev[i])
    end
    return nothing
end

function _pdhg_ubar_if_active_kernel!(
    u_bar,
    u,
    u_prev,
    theta,
    done_mask,
    batch_stride::Int,
    batch_size::Int,
    len::Int,
)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if index <= len
        i = Int(index)
        batch_idx = ((i - 1) ÷ batch_stride) % batch_size + 1
        @inbounds if done_mask[batch_idx] == UInt8(0)
            u_bar[i] = u[i] + theta * (u[i] - u_prev[i])
        end
    end
    return nothing
end

function _gradient_spatial!(
    g::NTuple{M,AT},
    u::CUDA.CuArray{T,N},
    inv_spacing::NTuple{M,S},
) where {T<:AbstractFloat,N,M,S<:Real,AT<:CUDA.CuArray{T,N}}
    len = length(u)
    len == 0 && return g

    threads, blocks = _launch_config(len)
    strides = ntuple(d -> stride(u, d), Val(M))

    @inbounds for d = 1:M
        CUDA.@cuda threads = threads blocks = blocks _gradient_dim_kernel!(
            g[d],
            u,
            T(inv_spacing[d]),
            strides[d],
            size(u, d),
            len,
        )
    end

    return g
end

function _gradient_spatial!(
    g::NTuple{M,AT},
    u::CUDA.CuArray{T,N},
    inv_spacing::NTuple{M,S},
    ::Periodic,
) where {T<:AbstractFloat,N,M,S<:Real,AT<:CUDA.CuArray{T,N}}
    len = length(u)
    len == 0 && return g

    threads, blocks = _launch_config(len)
    strides = ntuple(d -> stride(u, d), Val(M))
    @inbounds for d = 1:M
        CUDA.@cuda threads = threads blocks = blocks _gradient_periodic_dim_kernel!(
            g[d],
            u,
            T(inv_spacing[d]),
            strides[d],
            size(u, d),
            len,
        )
    end
    return g
end

function _gradient_spatial!(
    g::NTuple{M,AT},
    u::CUDA.CuArray{T,N},
    inv_spacing::NTuple{M,S},
    ::Neumann,
) where {T<:AbstractFloat,N,M,S<:Real,AT<:CUDA.CuArray{T,N}}
    return _gradient_spatial!(g, u, inv_spacing)
end

function _divergence_spatial!(
    out::CUDA.CuArray{T,N},
    p::NTuple{M,AT},
    inv_spacing::NTuple{M,S},
) where {T<:AbstractFloat,N,M,S<:Real,AT<:CUDA.CuArray{T,N}}
    fill!(out, zero(T))
    len = length(out)
    len == 0 && return out

    threads, blocks = _launch_config(len)
    strides = ntuple(d -> stride(out, d), Val(M))

    @inbounds for d = 1:M
        CUDA.@cuda threads = threads blocks = blocks _divergence_dim_kernel!(
            out,
            p[d],
            T(inv_spacing[d]),
            strides[d],
            size(out, d),
            len,
        )
    end

    return out
end

function _divergence_spatial!(
    out::CUDA.CuArray{T,N},
    p::NTuple{M,AT},
    inv_spacing::NTuple{M,S},
    ::Periodic,
) where {T<:AbstractFloat,N,M,S<:Real,AT<:CUDA.CuArray{T,N}}
    fill!(out, zero(T))
    len = length(out)
    len == 0 && return out

    threads, blocks = _launch_config(len)
    strides = ntuple(d -> stride(out, d), Val(M))
    @inbounds for d = 1:M
        CUDA.@cuda threads = threads blocks = blocks _divergence_periodic_dim_kernel!(
            out,
            p[d],
            T(inv_spacing[d]),
            strides[d],
            size(out, d),
            len,
        )
    end
    return out
end

function _divergence_spatial!(
    out::CUDA.CuArray{T,N},
    p::NTuple{M,AT},
    inv_spacing::NTuple{M,S},
    ::Neumann,
) where {T<:AbstractFloat,N,M,S<:Real,AT<:CUDA.CuArray{T,N}}
    return _divergence_spatial!(out, p, inv_spacing)
end

function _pdhg_dual_update!(
    p::NTuple{M,AT},
    grad_u_bar::NTuple{M,AT},
    sigma::T,
) where {T<:AbstractFloat,M,N,AT<:CUDA.CuArray{T,N}}
    len = length(p[1])
    len == 0 && return nothing
    threads, blocks = _launch_config(len)
    @inbounds for d = 1:M
        CUDA.@cuda threads = threads blocks = blocks _dual_update_kernel!(
            p[d],
            grad_u_bar[d],
            sigma,
            len,
        )
    end
    return nothing
end

function _pdhg_primal_and_overrelax!(
    state::PDHGState{T,N},
    f::CUDA.CuArray{T,N},
    tau::T,
    theta::T,
    data_fidelity::AbstractDataFidelity,
    lower::T,
    upper::T,
) where {T<:AbstractFloat,N}
    len = length(state.u)
    len == 0 && return nothing
    threads, blocks = _launch_config(len)

    CUDA.@cuda threads = threads blocks = blocks _pdhg_primal_tmp_kernel!(
        state.primal_tmp,
        state.u,
        state.divp,
        tau,
        len,
    )

    if data_fidelity isa L2Fidelity
        CUDA.@cuda threads = threads blocks = blocks _pdhg_prox_l2_kernel!(
            state.u,
            state.primal_tmp,
            f,
            tau,
            len,
        )
    elseif data_fidelity isa PoissonFidelity
        CUDA.@cuda threads = threads blocks = blocks _pdhg_prox_poisson_kernel!(
            state.u,
            state.primal_tmp,
            f,
            tau,
            len,
        )
    else
        throw(
            ArgumentError(
                "PDHG currently supports only L2Fidelity and PoissonFidelity, got $(typeof(data_fidelity))",
            ),
        )
    end

    TotalVariationImageFiltering._project_interval!(state.u, lower, upper)

    CUDA.@cuda threads = threads blocks = blocks _pdhg_ubar_kernel!(
        state.u_bar,
        state.u,
        state.u_prev,
        theta,
        len,
    )
    return nothing
end

function _pdhg_relative_residual_cuda!(
    state::PDHGState{T,N},
    boundary::AbstractBoundaryCondition,
    tau::T,
    sigma::T,
    inv_spacing::NTuple{N,T},
) where {T<:AbstractFloat,N}
    @inbounds for d = 1:N
        @. state.grad_u_bar[d] = state.p_prev[d] - state.p[d]
    end

    divergence!(state.primal_tmp, state.grad_u_bar, boundary, inv_spacing)
    @. state.divp = state.u_prev - state.u
    @. state.primal_tmp = state.divp / tau + state.primal_tmp
    primal_res_norm = T(sqrt(sum(abs2, state.primal_tmp)))

    gradient!(state.grad_u_bar, state.divp, boundary, inv_spacing)
    dual_res_norm2 = zero(T)
    @inbounds for d = 1:N
        ad = state.grad_u_bar[d]
        @. ad = (state.p_prev[d] - state.p[d]) / sigma - ad
        dual_res_norm2 += sum(abs2, ad)
    end

    dual_res_norm = T(sqrt(dual_res_norm2))
    nscale = sqrt(T(length(state.u)))
    return max(primal_res_norm, dual_res_norm) / max(nscale, one(T))
end

function gradient!(
    g::NTuple{N,AT},
    u::CUDA.CuArray{T,N},
    ::Neumann,
    inv_spacing::NTuple{N,S},
) where {T<:AbstractFloat,N,S<:Real,AT<:CUDA.CuArray{T,N}}
    return _gradient_spatial!(g, u, inv_spacing)
end

function divergence!(
    out::CUDA.CuArray{T,N},
    p::NTuple{N,AT},
    ::Neumann,
    inv_spacing::NTuple{N,S},
) where {T<:AbstractFloat,N,S<:Real,AT<:CUDA.CuArray{T,N}}
    return _divergence_spatial!(out, p, inv_spacing)
end

function gradient!(
    g::NTuple{N,AT},
    u::CUDA.CuArray{T,N},
    boundary::Periodic,
    inv_spacing::NTuple{N,S},
) where {T<:AbstractFloat,N,S<:Real,AT<:CUDA.CuArray{T,N}}
    return _gradient_spatial!(g, u, inv_spacing, boundary)
end

function divergence!(
    out::CUDA.CuArray{T,N},
    p::NTuple{N,AT},
    boundary::Periodic,
    inv_spacing::NTuple{N,S},
) where {T<:AbstractFloat,N,S<:Real,AT<:CUDA.CuArray{T,N}}
    return _divergence_spatial!(out, p, inv_spacing, boundary)
end

function _group_sum_axis_cuda!(
    out::CUDA.CuArray{T,N},
    input::CUDA.CuArray{T,N},
    d::Int,
    first_offset::Int,
    last_offset::Int,
    periodic::Bool,
) where {T,N}
    len = length(input)
    len == 0 && return out
    threads, blocks = _launch_config(len)
    CUDA.@cuda threads = threads blocks = blocks _group_sum_axis_kernel!(
        out,
        input,
        stride(input, d),
        size(input, d),
        first_offset,
        last_offset,
        periodic,
        len,
    )
    return out
end

function _group_sum_axis!(
    out::CUDA.CuArray{T,N},
    input::CUDA.CuArray{T,N},
    d::Int,
    first_offset::Int,
    last_offset::Int,
    ::Neumann,
) where {T,N}
    return _group_sum_axis_cuda!(out, input, d, first_offset, last_offset, false)
end

function _group_sum_axis!(
    out::CUDA.CuArray{T,N},
    input::CUDA.CuArray{T,N},
    d::Int,
    first_offset::Int,
    last_offset::Int,
    ::Periodic,
) where {T,N}
    return _group_sum_axis_cuda!(out, input, d, first_offset, last_offset, true)
end

function project_dual_ball!(
    p::NTuple{N,AT},
    radius::T,
    ::IsotropicTV,
) where {N,T<:AbstractFloat,AT<:CUDA.CuArray{T}}
    len = length(p[1])
    len == 0 && return p
    threads, blocks = _launch_config(len)
    radius2 = radius * radius
    CUDA.@cuda threads = threads blocks = blocks _project_isotropic_kernel!(
        p,
        radius,
        radius2,
        len,
    )
    return p
end

function project_dual_ball!(
    p::NTuple{N,AT},
    radius::T,
    ::AnisotropicTV,
) where {N,T<:AbstractFloat,AT<:CUDA.CuArray{T}}
    len = length(p[1])
    len == 0 && return p
    threads, blocks = _launch_config(len)
    lower = -radius
    upper = radius
    CUDA.@cuda threads = threads blocks = blocks _project_anisotropic_kernel!(
        p,
        lower,
        upper,
        len,
    )
    return p
end

struct ROFBatchState{T<:AbstractFloat,N,M,A<:CUDA.CuArray{T,N},G<:NTuple{M,A}}
    u::A
    u_prev::A
    divp::A
    g::A
    p::G
    grad_g::G
end

function ROFBatchState(reference::CUDA.CuArray{T,N}, ::Val{M}) where {T<:AbstractFloat,N,M}
    u = similar(reference)
    u_prev = similar(reference)
    divp = similar(reference)
    g = similar(reference)
    p = ntuple(_ -> similar(reference), Val(M))
    grad_g = ntuple(_ -> similar(reference), Val(M))

    fill!(u, zero(T))
    fill!(u_prev, zero(T))
    fill!(divp, zero(T))
    fill!(g, zero(T))
    @inbounds for d = 1:M
        fill!(p[d], zero(T))
        fill!(grad_g[d], zero(T))
    end

    return ROFBatchState{T,N,M,typeof(u),typeof(p)}(u, u_prev, divp, g, p, grad_g)
end

function _rof_batch_dual_step!(
    state::ROFBatchState{T,N,M},
    f_batch::CUDA.CuArray{T,N},
    tau::T,
    lambda::T,
    inv_spacing::NTuple{M,T},
    tv_mode::AbstractTVMode,
    boundary::AbstractBoundaryCondition,
    done_mask::CUDA.CuArray{UInt8,1},
    has_done::Bool,
    batch_stride::Int,
    batch_size::Int,
) where {T<:AbstractFloat,N,M}
    inv_lambda = inv(lambda)

    _divergence_spatial!(state.divp, state.p, inv_spacing, boundary)
    @. state.g = state.divp - inv_lambda * f_batch

    _gradient_spatial!(state.grad_g, state.g, inv_spacing, boundary)
    if has_done
        len = length(state.p[1])
        len > 0 && batch_size > 0 && batch_stride > 0 || return nothing
        threads, blocks = _launch_config(len)
        @inbounds for d = 1:M
            CUDA.@cuda threads = threads blocks = blocks _dual_update_if_active_kernel!(
                state.p[d],
                state.grad_g[d],
                tau,
                done_mask,
                batch_stride,
                batch_size,
                len,
            )
        end
    else
        @inbounds for d = 1:M
            @. state.p[d] = state.p[d] + tau * state.grad_g[d]
        end
    end

    project_dual_ball!(state.p, one(T), tv_mode)

    _divergence_spatial!(state.divp, state.p, inv_spacing, boundary)
    @. state.u = f_batch - lambda * state.divp
    return nothing
end

function _zero_batch_stats(
    ::Type{T},
    batch_count::Int,
    return_per_item_stats::Bool,
) where {T<:AbstractFloat}
    summary_stats = SolverStats{T}(0, true, zero(T))
    if return_per_item_stats
        per_item_stats = Vector{SolverStats{T}}(undef, batch_count)
        @inbounds for b = 1:batch_count
            per_item_stats[b] = SolverStats{T}(0, true, zero(T))
        end
        return summary_stats, per_item_stats
    end
    return summary_stats
end

function _batch_stats_result(
    ::Type{T},
    iterations_per_slice::Vector{Int},
    rel_changes::Vector{T},
    done_host::Vector{UInt8},
    converged_all::Bool,
    return_per_item_stats::Bool,
) where {T<:AbstractFloat}
    summary_stats =
        SolverStats{T}(maximum(iterations_per_slice), converged_all, maximum(rel_changes))
    if return_per_item_stats
        per_item_stats = Vector{SolverStats{T}}(undef, length(iterations_per_slice))
        @inbounds for b = 1:length(iterations_per_slice)
            per_item_stats[b] = SolverStats{T}(
                iterations_per_slice[b],
                done_host[b] == UInt8(1),
                rel_changes[b],
            )
        end
        return summary_stats, per_item_stats
    end
    return summary_stats
end

function solve_batch!(
    u_batch::CUDA.CuArray{T,N},
    f_batch::CUDA.CuArray{T,N},
    config::ROFConfig;
    lambda::Real,
    spacing = nothing,
    data_fidelity::AbstractDataFidelity = L2Fidelity(),
    tv_mode::AbstractTVMode = IsotropicTV(),
    boundary::AbstractBoundaryCondition = Neumann(),
    constraint::AbstractPrimalConstraint = NoConstraint(),
    state = nothing,
    return_per_item_stats::Bool = false,
) where {T<:AbstractFloat,N}
    N >= 2 ||
        throw(ArgumentError("f_batch must have at least 2 dimensions (spatial..., batch)"))
    size(u_batch) == size(f_batch) ||
        throw(ArgumentError("u_batch and f_batch must have matching sizes"))
    batch_count = size(f_batch, N)
    if batch_count == 0
        copyto!(u_batch, f_batch)
        return _zero_batch_stats(T, 0, return_per_item_stats)
    end

    TotalVariationImageFiltering._validate(config)
    data_fidelity isa L2Fidelity ||
        throw(ArgumentError("ROF currently supports only L2Fidelity"))
    constraint isa NoConstraint || throw(
        ArgumentError(
            "ROF currently supports only unconstrained problems; set constraint = NoConstraint() or use PDHGConfig",
        ),
    )
    (boundary isa Neumann || boundary isa Periodic) ||
        throw(ArgumentError("ROF batch CUDA supports only Neumann and Periodic boundaries"))

    lambda_t = T(lambda)
    lambda_t >= zero(T) || throw(ArgumentError("lambda must be non-negative"))
    if lambda_t == zero(T)
        copyto!(u_batch, f_batch)
        return _zero_batch_stats(T, batch_count, return_per_item_stats)
    end

    spatial_ndims = N - 1
    spacing_t =
        TotalVariationImageFiltering._normalize_spacing(T, Val(spatial_ndims), spacing)
    inv_spacing = ntuple(d -> inv(spacing_t[d]), Val(spatial_ndims))
    shape_spatial = ntuple(d -> size(f_batch, d), Val(spatial_ndims))

    tau_t = T(config.tau)
    tau_upper = TotalVariationImageFiltering._tau_upper_bound(inv_spacing, shape_spatial)
    tau_t < tau_upper || throw(
        ArgumentError(
            "tau must be < $(tau_upper) for this grid spacing and shape; got $(tau_t)",
        ),
    )

    local_state = if state === nothing
        ROFBatchState(f_batch, Val(spatial_ndims))
    elseif state isa ROFBatchState
        size(state.u) == size(f_batch) ||
            throw(ArgumentError("state buffers must match f_batch size"))
        length(state.p) == spatial_ndims ||
            throw(ArgumentError("state dual buffers must match spatial dimensions"))
        state
    else
        throw(ArgumentError("state must be `nothing` or a compatible ROFBatchState object"))
    end

    copyto!(local_state.u, u_batch)
    done_host = zeros(UInt8, batch_count)
    done_device = CUDA.zeros(UInt8, batch_count)
    rel_changes = fill(T(Inf), batch_count)
    iterations_per_slice = fill(config.maxiter, batch_count)
    active_count = batch_count
    has_done = false
    batch_stride = stride(f_batch, N)

    for k = 1:config.maxiter
        copyto!(local_state.u_prev, local_state.u)
        _rof_batch_dual_step!(
            local_state,
            f_batch,
            tau_t,
            lambda_t,
            inv_spacing,
            tv_mode,
            boundary,
            done_device,
            has_done,
            batch_stride,
            batch_count,
        )

        if (k % config.check_every == 0) || (k == config.maxiter)
            mask_changed = false
            @views for b = 1:batch_count
                done_host[b] == UInt8(1) && continue
                rel = TotalVariationImageFiltering._relative_change(
                    selectdim(local_state.u_prev, N, b),
                    selectdim(local_state.u, N, b),
                )
                rel_changes[b] = rel
                if rel <= T(config.tol)
                    done_host[b] = UInt8(1)
                    iterations_per_slice[b] = k
                    active_count -= 1
                    mask_changed = true
                end
            end

            if active_count == 0
                copyto!(u_batch, local_state.u)
                return _batch_stats_result(
                    T,
                    iterations_per_slice,
                    rel_changes,
                    done_host,
                    true,
                    return_per_item_stats,
                )
            end

            if mask_changed
                has_done = true
                copyto!(done_device, done_host)
            end
        end
    end

    copyto!(u_batch, local_state.u)
    return _batch_stats_result(
        T,
        iterations_per_slice,
        rel_changes,
        done_host,
        false,
        return_per_item_stats,
    )
end

function solve!(
    u::CUDA.CuArray{T,N},
    problem::TotalVariationImageFiltering.TVProblem{T,N,AF,DF,TV,BC,PC},
    config::PDHGConfig;
    state::Union{Nothing,PDHGState{T,N}} = nothing,
) where {
    T<:AbstractFloat,
    N,
    AF<:CUDA.CuArray{T,N},
    DF<:AbstractDataFidelity,
    TV<:AbstractTVMode,
    BC<:AbstractBoundaryCondition,
    PC<:AbstractPrimalConstraint,
}
    TotalVariationImageFiltering._validate(config)
    TotalVariationImageFiltering._validate_pdhg_data_fidelity(problem)
    TotalVariationImageFiltering._validate_pdhg_constraint(problem)
    problem.tv_mode isa TotalVariationImageFiltering.GroupSparseTV &&
        throw(ArgumentError("PDHG does not support GroupSparseTV; use GSTVConfig"))
    (problem.boundary isa Neumann || problem.boundary isa Periodic) ||
        throw(ArgumentError("PDHG CUDA supports only Neumann and Periodic boundaries"))
    size(u) == size(problem.f) ||
        throw(ArgumentError("You must have the same size as problem.f"))
    problem.lambda >= zero(T) ||
        throw(ArgumentError("lambda must be non-negative, got $(problem.lambda)"))

    primal_lower, primal_upper = TotalVariationImageFiltering._pdhg_primal_bounds(problem)

    local_state = state === nothing ? nothing : state
    local_state === nothing ||
        TotalVariationImageFiltering._validate_state_shape(local_state, size(u))

    if problem.lambda == zero(T)
        TotalVariationImageFiltering._pdhg_data_minimizer!(
            u,
            problem.f,
            problem.data_fidelity,
            primal_lower,
            primal_upper,
        )
        return SolverStats{T}(0, true, zero(T))
    end

    local_state = local_state === nothing ? PDHGState(problem.f) : local_state

    tau_t = T(config.tau)
    sigma_t = T(config.sigma)
    theta_t = T(config.theta)
    inv_spacing = ntuple(d -> inv(problem.spacing[d]), Val(N))

    op_norm_sq = TotalVariationImageFiltering._pdhg_operator_norm_sq_upper_bound(
        inv_spacing,
        size(problem.f),
    )
    if op_norm_sq > zero(T)
        tau_t * sigma_t * op_norm_sq < one(T) || throw(
            ArgumentError(
                "tau * sigma * ||grad||^2 must be < 1; with this spacing/shape, use tau*sigma < $(inv(op_norm_sq)), got $(tau_t * sigma_t)",
            ),
        )
    end

    copyto!(local_state.u, u)
    copyto!(local_state.u_bar, u)

    rel_change = T(Inf)
    converged = false
    iterations = config.maxiter

    for k = 1:config.maxiter
        copyto!(local_state.u_prev, local_state.u)
        @inbounds for d = 1:N
            copyto!(local_state.p_prev[d], local_state.p[d])
        end

        _gradient_spatial!(
            local_state.grad_u_bar,
            local_state.u_bar,
            inv_spacing,
            problem.boundary,
        )
        _pdhg_dual_update!(local_state.p, local_state.grad_u_bar, sigma_t)
        project_dual_ball!(local_state.p, problem.lambda, problem.tv_mode)

        _divergence_spatial!(local_state.divp, local_state.p, inv_spacing, problem.boundary)
        _pdhg_primal_and_overrelax!(
            local_state,
            problem.f,
            tau_t,
            theta_t,
            problem.data_fidelity,
            primal_lower,
            primal_upper,
        )

        if (k % config.check_every == 0) || (k == config.maxiter)
            primal_rel_change = TotalVariationImageFiltering._relative_change(
                local_state.u_prev,
                local_state.u,
            )
            pdhg_residual = _pdhg_relative_residual_cuda!(
                local_state,
                problem.boundary,
                tau_t,
                sigma_t,
                inv_spacing,
            )
            rel_change = max(primal_rel_change, pdhg_residual)
            if rel_change <= T(config.tol)
                converged = true
                iterations = k
                break
            end
        end

        iterations = k
    end

    copyto!(u, local_state.u)
    return SolverStats{T}(iterations, converged, rel_change)
end

struct PDHGBatchState{T<:AbstractFloat,N,M,A<:CUDA.CuArray{T,N},G<:NTuple{M,A}}
    u::A
    u_prev::A
    u_bar::A
    divp::A
    primal_tmp::A
    p::G
    p_prev::G
    grad_u_bar::G
end

function PDHGBatchState(reference::CUDA.CuArray{T,N}, ::Val{M}) where {T<:AbstractFloat,N,M}
    u = similar(reference)
    u_prev = similar(reference)
    u_bar = similar(reference)
    divp = similar(reference)
    primal_tmp = similar(reference)
    p = ntuple(_ -> similar(reference), Val(M))
    p_prev = ntuple(_ -> similar(reference), Val(M))
    grad_u_bar = ntuple(_ -> similar(reference), Val(M))

    fill!(u, zero(T))
    fill!(u_prev, zero(T))
    fill!(u_bar, zero(T))
    fill!(divp, zero(T))
    fill!(primal_tmp, zero(T))
    @inbounds for d = 1:M
        fill!(p[d], zero(T))
        fill!(p_prev[d], zero(T))
        fill!(grad_u_bar[d], zero(T))
    end

    return PDHGBatchState{T,N,M,typeof(u),typeof(p)}(
        u,
        u_prev,
        u_bar,
        divp,
        primal_tmp,
        p,
        p_prev,
        grad_u_bar,
    )
end

function _batch_pdhg_primal_bounds(
    f_batch::CUDA.CuArray{T},
    data_fidelity::AbstractDataFidelity,
    constraint::AbstractPrimalConstraint,
) where {T<:AbstractFloat}
    lower, upper = TotalVariationImageFiltering._constraint_bounds(T, constraint)
    if data_fidelity isa PoissonFidelity
        upper >= zero(T) || throw(
            ArgumentError(
                "PoissonFidelity with constraint [$lower, $upper] is infeasible; upper bound must be >= 0",
            ),
        )
        if (upper == zero(T)) && any(f_batch .> zero(T))
            throw(
                ArgumentError(
                    "PoissonFidelity with upper bound 0 is infeasible when observation data contains positive values",
                ),
            )
        end
        lower = max(lower, zero(T))
    end
    return lower, upper
end

function _pdhg_batch_step!(
    state::PDHGBatchState{T,N,M},
    f_batch::CUDA.CuArray{T,N},
    tau::T,
    sigma::T,
    theta::T,
    lambda::T,
    lower::T,
    upper::T,
    inv_spacing::NTuple{M,T},
    data_fidelity::AbstractDataFidelity,
    tv_mode::AbstractTVMode,
    boundary::AbstractBoundaryCondition,
    done_mask::CUDA.CuArray{UInt8,1},
    has_done::Bool,
    batch_stride::Int,
    batch_size::Int,
) where {T<:AbstractFloat,N,M}
    @inbounds for d = 1:M
        copyto!(state.p_prev[d], state.p[d])
    end

    _gradient_spatial!(state.grad_u_bar, state.u_bar, inv_spacing, boundary)
    if has_done
        len = length(state.p[1])
        len > 0 && batch_size > 0 && batch_stride > 0 || return nothing
        threads, blocks = _launch_config(len)
        @inbounds for d = 1:M
            CUDA.@cuda threads = threads blocks = blocks _dual_update_if_active_kernel!(
                state.p[d],
                state.grad_u_bar[d],
                sigma,
                done_mask,
                batch_stride,
                batch_size,
                len,
            )
        end
    else
        _pdhg_dual_update!(state.p, state.grad_u_bar, sigma)
    end

    project_dual_ball!(state.p, lambda, tv_mode)
    _divergence_spatial!(state.divp, state.p, inv_spacing, boundary)

    len = length(state.u)
    len == 0 && return nothing
    threads, blocks = _launch_config(len)

    if has_done
        CUDA.@cuda threads = threads blocks = blocks _pdhg_primal_tmp_if_active_kernel!(
            state.primal_tmp,
            state.u,
            state.divp,
            tau,
            done_mask,
            batch_stride,
            batch_size,
            len,
        )

        if data_fidelity isa L2Fidelity
            CUDA.@cuda threads = threads blocks = blocks _pdhg_prox_l2_if_active_kernel!(
                state.u,
                state.primal_tmp,
                f_batch,
                tau,
                done_mask,
                batch_stride,
                batch_size,
                len,
            )
        elseif data_fidelity isa PoissonFidelity
            CUDA.@cuda threads = threads blocks = blocks _pdhg_prox_poisson_if_active_kernel!(
                state.u,
                state.primal_tmp,
                f_batch,
                tau,
                done_mask,
                batch_stride,
                batch_size,
                len,
            )
        else
            throw(
                ArgumentError(
                    "PDHG currently supports only L2Fidelity and PoissonFidelity, got $(typeof(data_fidelity))",
                ),
            )
        end

        TotalVariationImageFiltering._project_interval!(state.u, lower, upper)

        CUDA.@cuda threads = threads blocks = blocks _pdhg_ubar_if_active_kernel!(
            state.u_bar,
            state.u,
            state.u_prev,
            theta,
            done_mask,
            batch_stride,
            batch_size,
            len,
        )
    else
        CUDA.@cuda threads = threads blocks = blocks _pdhg_primal_tmp_kernel!(
            state.primal_tmp,
            state.u,
            state.divp,
            tau,
            len,
        )

        if data_fidelity isa L2Fidelity
            CUDA.@cuda threads = threads blocks = blocks _pdhg_prox_l2_kernel!(
                state.u,
                state.primal_tmp,
                f_batch,
                tau,
                len,
            )
        elseif data_fidelity isa PoissonFidelity
            CUDA.@cuda threads = threads blocks = blocks _pdhg_prox_poisson_kernel!(
                state.u,
                state.primal_tmp,
                f_batch,
                tau,
                len,
            )
        else
            throw(
                ArgumentError(
                    "PDHG currently supports only L2Fidelity and PoissonFidelity, got $(typeof(data_fidelity))",
                ),
            )
        end

        TotalVariationImageFiltering._project_interval!(state.u, lower, upper)

        CUDA.@cuda threads = threads blocks = blocks _pdhg_ubar_kernel!(
            state.u_bar,
            state.u,
            state.u_prev,
            theta,
            len,
        )
    end
    return nothing
end

function _pdhg_batch_slice_state(
    state::PDHGBatchState{T,N,M},
    b::Int,
) where {T<:AbstractFloat,N,M}
    @views begin
        u = selectdim(state.u, N, b)
        u_prev = selectdim(state.u_prev, N, b)
        u_bar = selectdim(state.u_bar, N, b)
        divp = selectdim(state.divp, N, b)
        primal_tmp = selectdim(state.primal_tmp, N, b)
        p = ntuple(d -> selectdim(state.p[d], N, b), Val(M))
        p_prev = ntuple(d -> selectdim(state.p_prev[d], N, b), Val(M))
        grad_u_bar = ntuple(d -> selectdim(state.grad_u_bar[d], N, b), Val(M))
        return PDHGState(u, u_prev, u_bar, divp, primal_tmp, p, p_prev, grad_u_bar)
    end
end

function solve_batch!(
    u_batch::CUDA.CuArray{T,N},
    f_batch::CUDA.CuArray{T,N},
    config::PDHGConfig;
    lambda::Real,
    spacing = nothing,
    data_fidelity::AbstractDataFidelity = L2Fidelity(),
    tv_mode::AbstractTVMode = IsotropicTV(),
    boundary::AbstractBoundaryCondition = Neumann(),
    constraint::AbstractPrimalConstraint = NoConstraint(),
    state = nothing,
    return_per_item_stats::Bool = false,
) where {T<:AbstractFloat,N}
    N >= 2 ||
        throw(ArgumentError("f_batch must have at least 2 dimensions (spatial..., batch)"))
    size(u_batch) == size(f_batch) ||
        throw(ArgumentError("u_batch and f_batch must have matching sizes"))
    batch_count = size(f_batch, N)
    if batch_count == 0
        copyto!(u_batch, f_batch)
        return _zero_batch_stats(T, 0, return_per_item_stats)
    end

    TotalVariationImageFiltering._validate(config)
    tv_mode isa TotalVariationImageFiltering.GroupSparseTV &&
        throw(ArgumentError("PDHG does not support GroupSparseTV; use GSTVConfig"))
    (boundary isa Neumann || boundary isa Periodic) || throw(
        ArgumentError("PDHG batch CUDA supports only Neumann and Periodic boundaries"),
    )
    if data_fidelity isa PoissonFidelity
        TotalVariationImageFiltering._validate_poisson_data(f_batch)
    elseif !(data_fidelity isa L2Fidelity)
        throw(
            ArgumentError(
                "PDHG currently supports only L2Fidelity and PoissonFidelity, got $(typeof(data_fidelity))",
            ),
        )
    end

    lambda_t = T(lambda)
    lambda_t >= zero(T) || throw(ArgumentError("lambda must be non-negative"))
    primal_lower, primal_upper =
        _batch_pdhg_primal_bounds(f_batch, data_fidelity, constraint)
    if lambda_t == zero(T)
        TotalVariationImageFiltering._pdhg_data_minimizer!(
            u_batch,
            f_batch,
            data_fidelity,
            primal_lower,
            primal_upper,
        )
        return _zero_batch_stats(T, batch_count, return_per_item_stats)
    end

    spatial_ndims = N - 1
    spacing_t =
        TotalVariationImageFiltering._normalize_spacing(T, Val(spatial_ndims), spacing)
    inv_spacing = ntuple(d -> inv(spacing_t[d]), Val(spatial_ndims))
    shape_spatial = ntuple(d -> size(f_batch, d), Val(spatial_ndims))

    tau_t = T(config.tau)
    sigma_t = T(config.sigma)
    theta_t = T(config.theta)
    op_norm_sq = TotalVariationImageFiltering._pdhg_operator_norm_sq_upper_bound(
        inv_spacing,
        shape_spatial,
    )
    if op_norm_sq > zero(T)
        tau_t * sigma_t * op_norm_sq < one(T) || throw(
            ArgumentError(
                "tau * sigma * ||grad||^2 must be < 1; with this spacing/shape, use tau*sigma < $(inv(op_norm_sq)), got $(tau_t * sigma_t)",
            ),
        )
    end

    local_state = if state === nothing
        PDHGBatchState(f_batch, Val(spatial_ndims))
    elseif state isa PDHGBatchState
        shape = size(f_batch)
        size(state.u) == shape ||
            throw(ArgumentError("state.u size must match f_batch size"))
        size(state.u_prev) == shape ||
            throw(ArgumentError("state.u_prev size must match f_batch size"))
        size(state.u_bar) == shape ||
            throw(ArgumentError("state.u_bar size must match f_batch size"))
        size(state.divp) == shape ||
            throw(ArgumentError("state.divp size must match f_batch size"))
        size(state.primal_tmp) == shape ||
            throw(ArgumentError("state.primal_tmp size must match f_batch size"))
        length(state.p) == spatial_ndims ||
            throw(ArgumentError("state.p buffers must match spatial dimensions"))
        length(state.p_prev) == spatial_ndims ||
            throw(ArgumentError("state.p_prev buffers must match spatial dimensions"))
        length(state.grad_u_bar) == spatial_ndims || throw(
            ArgumentError("state.grad_u_bar buffers must match spatial dimensions"),
        )
        @inbounds for d = 1:spatial_ndims
            size(state.p[d]) == shape ||
                throw(ArgumentError("state.p[$d] size must match f_batch size"))
            size(state.p_prev[d]) == shape ||
                throw(ArgumentError("state.p_prev[$d] size must match f_batch size"))
            size(state.grad_u_bar[d]) == shape || throw(
                ArgumentError("state.grad_u_bar[$d] size must match f_batch size"),
            )
        end
        state
    else
        throw(ArgumentError("state must be `nothing` or a compatible PDHGBatchState object"))
    end

    copyto!(local_state.u, u_batch)
    copyto!(local_state.u_bar, u_batch)
    done_host = zeros(UInt8, batch_count)
    done_device = CUDA.zeros(UInt8, batch_count)
    rel_changes = fill(T(Inf), batch_count)
    iterations_per_slice = fill(config.maxiter, batch_count)
    active_count = batch_count
    has_done = false
    batch_stride = stride(f_batch, N)

    for k = 1:config.maxiter
        copyto!(local_state.u_prev, local_state.u)
        _pdhg_batch_step!(
            local_state,
            f_batch,
            tau_t,
            sigma_t,
            theta_t,
            lambda_t,
            primal_lower,
            primal_upper,
            inv_spacing,
            data_fidelity,
            tv_mode,
            boundary,
            done_device,
            has_done,
            batch_stride,
            batch_count,
        )

        if (k % config.check_every == 0) || (k == config.maxiter)
            mask_changed = false
            @views for b = 1:batch_count
                done_host[b] == UInt8(1) && continue
                slice_state = _pdhg_batch_slice_state(local_state, b)
                primal_rel_change = TotalVariationImageFiltering._relative_change(
                    slice_state.u_prev,
                    slice_state.u,
                )
                pdhg_residual = _pdhg_relative_residual_cuda!(
                    slice_state,
                    boundary,
                    tau_t,
                    sigma_t,
                    inv_spacing,
                )
                rel = max(primal_rel_change, pdhg_residual)
                rel_changes[b] = rel
                if rel <= T(config.tol)
                    done_host[b] = UInt8(1)
                    iterations_per_slice[b] = k
                    active_count -= 1
                    mask_changed = true
                end
            end

            if active_count == 0
                copyto!(u_batch, local_state.u)
                return _batch_stats_result(
                    T,
                    iterations_per_slice,
                    rel_changes,
                    done_host,
                    true,
                    return_per_item_stats,
                )
            end

            if mask_changed
                has_done = true
                copyto!(done_device, done_host)
            end
        end
    end

    copyto!(u_batch, local_state.u)
    return _batch_stats_result(
        T,
        iterations_per_slice,
        rel_changes,
        done_host,
        false,
        return_per_item_stats,
    )
end

end # module
