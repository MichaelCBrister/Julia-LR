#
# 
# Linear Regression
#
# Packages
using CSV, GLM, Plots, TypedTables

# Manual

# Import data
data = CSV.File("C:\\Users\\micha\\Desktop\\Julia\\LRv1\\housingdata.csv")

X = data.size

y = round.(Int, data.price / 1000)

t = Table(X = X, y = y)


# Plot data
gr(size =(600, 600))

p_scatter = scatter(X, y,
    xlims = (0, 5000),
    ylims = (0, 800),
    xlabel = "Size (sqft)",
    ylabel = "Price (in thousands of dollars)",
    title = "Housing Prices in Portland",
    legend = false,
    color = :red
)


# Linear Regression model using GLM Package
ols = lm(@formula(y ~ X), t)

# Add Regression Line to Plot
plot!(X, predict(ols), color = :blue, linewidth = 3)


# Make Predictions
newX = Table(X = [1250])

predict(ols, newX)


# Machine Learning Approach

epochs = 0

# Plot data
gr(size =(600, 600))

p_scatter = scatter(X, y,
    xlims = (0, 5000),
    ylims = (0, 800),
    xlabel = "Size (sqft)",
    ylabel = "Price (in thousands of dollars)",
    title = "Housing Prices in Portland (epochs = $epochs",
    legend = false, 
    color = :red
)


# Initializing parameters
theta_0 = 0.0   # y-intercept

theta_1 = 0.0   # slope


# Linear Regression model
h(x) = theta_0 .+ theta_1 * x


# Add line to plot
plot!(X, h(X), color = :green, linewidth = 3)


# Create Cost Function
m = length(X)

y_hat = h(X)

function cost(X, y)
    (1 / (2 * m)) * sum((y_hat - y).^2)  
end

J = cost(X, y)


# Pushing cost value to vector
J_history = []

push!(J_history, J)


# Batch Gradient Descent Algo
function  pd_theta_0(X, y)
    (1 / m) * sum(y_hat - y)
end

function pd_theta_1(X, y)
    (1 / m) * sum((y_hat - y) .* X)
end


# Set learning rates for quicker solution
alpha_0 = 0.09

alpha_1 = 0.00000008

# Calculate partial derivatives
theta_0_temp = pd_theta_0(X, y)

theta_1_temp = pd_theta_1(X, y)

# Adjust parameters
theta_0 -= alpha_0 * theta_0_temp

theta_1 -= alpha_1 * theta_1_temp


# Recalculate Cost function
y_hat = h(X)

J = cost(X, y)

push!(J_history, J)

# Replot 
epochs += 1

plot!(X, y_hat, color = :blue, alpha = 0.5,
    title = "Housing Price in Portland (epochs = $epochs)"
)

# Compare to the manual Linear Regression model created above
plot!(X, predict(ols), color = :green, linewidth = 2)

# Plot Learning Curve of Machine model
gr(size = (600, 600))

p_line = plot(0:epochs, J_history,
    xlabel = "Epochs",
    ylabel = "Cost",
    title = "Learning Curve",
    legend = false,
    color = :blue,
    linewidth = 3
