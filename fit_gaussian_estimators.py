from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "simple_white"


def test_univariate_gaussian():

    # Question 1 - Draw samples and print fitted model
    expectation, variance, m = 10, 1, 1000
    X = np.random.normal(expectation, variance, m)
    fittedUG = UnivariateGaussian().fit(X)
    print((fittedUG.mu_, fittedUG.var_))

    # Question 2 - Empirically showing sample mean is consistent
    mu_hat = []
    for n in range(10, 1010, 10):
        mu_hat.append(np.abs(fittedUG.mu_ - UnivariateGaussian().fit(X[:n]).mu_))
    go.Figure([go.Scatter(x=list(range(10, 1010, 10)), y=mu_hat, mode='markers+lines', name=r'$\widehat\mu$')],
              layout=go.Layout(title=r"$\text{Estimation of Expectation As Function Of Number Of Samples}$",
                               xaxis_title="$m\\text{ - number of samples}$",
                               yaxis_title="r$\hat\mu$")).write_image("./IML-EX1-Q2.png")

    # Question 3 - Plotting Empirical PDF of fitted model
    sampleValue_pdfs = np.c_[X, fittedUG.pdf(X)]
    index_arr_sample_value = sampleValue_pdfs[:, 0].argsort()
    sorted_sample_value = sampleValue_pdfs[index_arr_sample_value]
    go.Figure([go.Scatter(x=sorted_sample_value[:, 0], y=sorted_sample_value[:, 1],
                          mode='markers',
                          marker=dict(color="black"))],
              layout=go.Layout(title=r"$\text{Ordered by sample values}$",
                               xaxis_title="$\\text{Sample value}$",
                               yaxis_title="$\\text{Pdf value}$")).write_image("./IML-EX1-Q3.png")


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    cov_matrix = [[1, 0.2, 0, 0.5], [0.2, 2, 0, 0], [0, 0, 1, 0], [0.5, 0, 0, 1]]
    mu = [0, 0, 4, 0]
    X = np.random.multivariate_normal(mu, cov_matrix, size=1000)
    fitted_mgu = MultivariateGaussian().fit(X)
    print(fitted_mgu.mu_)
    print(fitted_mgu.cov_)

    # Question 5 - Likelihood evaluation
    f_space = np.linspace(-10, 10, 200)
    likelihood_values = np.zeros((200, 200))
    for i, f1 in enumerate(f_space):
        for j, f3 in enumerate(f_space):
            likelihood_values[i, j] = MultivariateGaussian.log_likelihood(np.array([f1, 0, f3, 0]), cov_matrix, X)

    go.Figure(go.Heatmap(x=f_space, y=f_space, z=likelihood_values),
              layout=dict(template="simple_white",
                          title="Log-Likelihood of Multivatiate Gaussian As Function of Expectation of Feautures 1,3",
                          xaxis_title=r"$\mu_3$",
                          yaxis_title=r"$\mu_1$")) \
        .write_image("./IML-EX1-Q5.png")

    # Question 6 - Maximum likelihood
    best_fit = np.where(likelihood_values == np.amax(likelihood_values))
    f1 = list(np.round(f_space[best_fit[0]],3))[0]
    f3 = list(np.round(f_space[best_fit[1]],3))[0]
    print(f"The model who achieved the maximum log-likelihood value is {f1=}, {f3=}")

if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
