using LinearAlgebra

function solve_system(A::Matrix{Float64}, b::Vector{Float64})
    """
    Solve system of linear equations Ax = b using LU decomposition
    """
    n = size(A, 1)
    if n != size(A, 2) || n != length(b)
        throw(DimensionMismatch("Matrix and vector dimensions must match"))
    end
    
    # LU decomposition
    lu_fact = lu(A)
    
    # Solve system
    x = lu_fact \ b
    
    # Calculate residual
    residual = norm(A * x - b)
    
    return x, residual
end

# Test case
A = [2.0 1.0; 1.0 3.0]
b = [5.0, 6.0]
x, res = solve_system(A, b)
println("Solution: ", x)
println("Residual: ", res)