#Distributions

import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
import plotly.express as px
import matplotlib.pyplot as plt
import streamlit as st
import plotly.figure_factory as ff
import scipy


st.title('Probability Distributionsss') #TODO: add icon and header animation here
st.header('Concept Description')

info_desc = """
A probability distribution is a mathematical function that describes the probability of different possible values of a variable. [📜 ref](https://www.scribbr.com/statistics/probability-distributions/)
\nA probability distribution function (PDF) is a mathematical function that describes the likelihood of various outcomes in a
probabilistic or random experiment. It specifies the probability associated with each possible outcome of a random variable.
In other words, it tells you how the probability is distributed across different values of the random variable.
"""

markdown_desc = """
We have two general types of PDF:
1. **Continuous Probability Distributions**  
    We name its function PDF (Probability Distribution Function)  
2. **Discrete Probability Distributions**  
    We name its function PMF (Probability Mass Function)  
    
> **PMF** is used for **discrete random variables** and gives the probabilities associated with **specific values**, 
while **PDF** is used for **continuous random variables** and provides the probability density at **various points within a continuous range**. 
"""
app_desc = """
- You can plot and modify popular distributions that exist in **Discrete** and **Continuous** categories.  
- You can do [**Kernel Density Estimation**](https://docs.scipy.org/doc/scipy/tutorial/stats.html#kernel-density-estimation) with this App.  
"""



continuous_dist = ['Normal Distribution (Gaussian Distribution)', 'Uniform Distribution','Exponential Distribution',
                   'Beta Distribution']
discrete_dist = ['Bernoulli Distribution', 'Binomial Distribution', 'Poisson Distribution', 'Geometric Distribution']

@st.cache_data(ttl=3, show_spinner='Fetching Data From API...')
def pdf_plot (dist:str, dist_type:str):
    fig, ax = plt.subplots()
    if dist_type == 'Continuous':
        if dist == "Normal Distribution (Gaussian Distribution)":
            # Define the parameters of the normal distribution
            mean = 0
            std_dev = 1
            # Generate random data from a normal distribution
            normal_dist = np.random.normal(mean, std_dev, 1000)
            hist_data = [normal_dist]
            group_labels = ['Normal Distribution']
            ax = ff.create_distplot(hist_data, group_labels, colors=['blue'], bin_size=0.1)
            ax.update_layout(title_text='Distplot with Normal Distribution (μ=0 and σ=1)' , xaxis_title='X values', yaxis_title='PDF')
        elif dist == "Uniform Distribution":
            uniform_dist = np.random.uniform(size=1000)
            hist_data = [uniform_dist]
            group_labels = ['Uniform Distribution']
            ax = ff.create_distplot(hist_data, group_labels, colors=['blue'], bin_size=0.01)
            ax.update_layout(title_text='Distplot Uniform Distribution (Low boundary=0, High boundary=1)' , xaxis_title='X values', yaxis_title='PDF')
        elif dist == "Exponential Distribution":
            exponential_dist = np.random.exponential(size=1000)
            hist_data = [exponential_dist]
            group_labels = ['Exponential Distribution']
            ax = ff.create_distplot(hist_data, group_labels, colors=['blue'], bin_size=0.1)
            ax.update_layout(title_text='Distplot Exponential Distribution (λ=1)' , xaxis_title='X values', yaxis_title='PDF')
        elif dist == "Beta Distribution":
            a=int(np.random.rand()*20+1)
            b=int(np.random.rand()*20+1)
            beta_dist = np.random.beta(size=1000, a=a, b=b)
            hist_data = [beta_dist]
            group_labels = ['Beta Distribution']
            ax = ff.create_distplot(hist_data, group_labels, colors=['blue'], bin_size=0.01)
            ax.update_layout(title_text=f'Distplot Exponential Distribution (A={a}, B={b})' , xaxis_title='X values', yaxis_title='PDF')
    else:
        if dist == "Bernoulli Distribution":
            p = np.random.rand()
            # creating a numpy array for x-axis
            x = np.arange(0, 2, 1)
            # poisson distribution data for y-axis
            y = scipy.stats.bernoulli.pmf(x, p=p)
            # For the visualization of the bar plot of Bernoulli's distribution
            ax = px.bar(x=x, y=y)
            ax.update_layout(title_text=f'Distplot Bernoulli Distribution (p={p})' , xaxis_title='X values', yaxis_title='PMF')
        elif dist == "Binomial Distribution":
            n = int(np.random.rand()*100+1)
            p = np.random.rand()
            # creating a numpy array for x-axis
            x = np.arange(0, 100, 0.5)
            # poisson distribution data for y-axis
            y = scipy.stats.binom.pmf(x, n=n, p=p)
            ax = px.bar(x=x, y=y)
            ax.update_layout(title_text=f'Distplot Binomial Distribution (n={n}, p={p})' , xaxis_title='X values', yaxis_title='PMF')
        elif dist == "Poisson Distribution":
            lam = int(np.random.rand()*200+30)
            # creating a numpy array for x-axis
            x = np.arange(0, 100, 0.5)
            # poisson distribution data for y-axis
            y = scipy.stats.poisson.pmf(x, mu=lam)
            ax = px.bar(x=x, y=y)
            ax.update_layout(title_text=f'Distplot Poisson Distribution (λ={lam})' , xaxis_title='X values', yaxis_title='PMF')
        elif dist == "Geometric Distribution":
            p = np.random.rand()
            # creating a numpy array for x-axis
            x = np.arange(0, 100, 0.5)
            # poisson distribution data for y-axis
            y = scipy.stats.geom.pmf(x, p=p)
            ax = px.bar(x=x, y=y)
            ax.update_layout(title_text=f'Distplot Geometric Distribution (p={p})' , xaxis_title='X values', yaxis_title='PMF')
    return ax

@st.cache_data(show_spinner='Fetching Data From API... **(It may takes a few seconds)**')
def kd_estimation(x, min, max):
    bandwidths = 2 ** np.linspace(min, 1, max)
    kf = KFold()
    kfcv = kf.get_n_splits(x)
    grid = GridSearchCV(KernelDensity(kernel='gaussian'),
                        {'bandwidth': bandwidths},
                        cv=kfcv)
    grid.fit(np.expand_dims(x, axis=1))
    # instantiate and fit the KDE model
    kdf_params, kde = grid.best_estimator_.get_params(), grid.best_estimator_
    return(kde, kdf_params)
    

st.info(info_desc)
st.markdown(markdown_desc)

# App Description
st.header('App Description')
st.write(app_desc)
plot_container = st.empty()
plot_container.info("You can see your plot here")

check_dist = st.sidebar.radio('Select type of distribution', options=['Continuous', 'Discrete', 'Unknown'])
pdf = None
pmf = None
if check_dist == 'Continuous':
    pdf = st.sidebar.selectbox('Select distribution', continuous_dist)
elif check_dist == 'Discrete':
    pmf = st.sidebar.selectbox('Select distribution', discrete_dist)
elif check_dist == 'Unknown':
    # upload personal dataset
    upload_files = st.sidebar.file_uploader(label='Upload your own data', help='You can upload your data with unknown distribution', type=['csv'])
    if upload_files is not None:
        df = pd.read_csv(upload_files)
        columns = df.columns
        select_column = st.sidebar.selectbox('Choose your Column', options=columns)
        # check columns type
        try:
            df = df[select_column].astype(np.int64)
            if df.shape[0] > 1000:
                df = df[0:1000].sort_values()
        except:
            st.error('This column is not number!!', icon='🚨')
            st.stop()

plot_btn = st.sidebar.button('Plot PDF/PMF')
if plot_btn and check_dist != 'Unknown':
    if pdf:
        ax = pdf_plot(dist=pdf, dist_type=check_dist)
        plot_container.empty()
        plot_container.plotly_chart(ax)
    else:
        ax = pdf_plot(dist=pmf, dist_type=check_dist)
        plot_container.empty()
        plot_container.plotly_chart(ax)
elif plot_btn and upload_files is not None:
    fig, ax = plt.subplots()
    plot_container.empty()
    kde, kde_params = kd_estimation(x=df, min=df.min(), max=df.max())
    st.write(kde_params)
    # score_samples returns the log of the probability density
    logprob = kde.score_samples(np.expand_dims(df, axis=1))
    res1 = ax.fill_between(df, np.exp(logprob), alpha=0.5)
    res2 = ax.plot(df, np.full_like(df, -0.01), '|k', markeredgewidth=1)
    res3 = ax.plot(df, np.exp(logprob), drawstyle="default", color='red')
    ax.set_title('PDF of Personal data')
    st.pyplot(fig)
# business usages
st.header('Business Usage')
with st.expander("**Risk Management**"):
    st.write("""
            - **Insurance:** Assessing and setting premiums based on the likelihood of events.
            - **Financial Risk:** Modeling asset prices and returns for risk management in finance.""")
    
with st.expander('Supply Chain Management'):
    st.write("""
             - Modeling uncertainty in factors like lead times, demand, and supply disruptions for optimizing supply chain performance.""")
    
with st.expander("**Quality Control**"):
    st.write("""
        - Statistical quality control using PDFs to set standards, assess process capability, and maintain product quality.
    """)

with st.expander("**Marketing and Sales**"):
    st.write("""
        - Forecasting customer preferences, purchase behavior, and market response for resource allocation and marketing strategy.
    """)

with st.expander("**Project Management**"):
    st.write("""
        - Modeling project completion times and costs to assess the likelihood of meeting deadlines and budgets.
    """)

with st.expander("**Customer Relationship Management (CRM)**"):
    st.write("""
        - Modeling customer lifetime value, churn rates, and behavior for segmentation and personalized marketing.
    """)

with st.expander("**Human Resources**"):
    st.write("""
        - Workforce planning and talent management using PDFs to model employee performance and recruitment metrics.
    """)

with st.expander("**Operations and Manufacturing**"):
    st.write("""
        - Modeling process parameters, such as production cycle times and defect rates, for optimization and quality control.
    """)

with st.expander("**Market Research**"):
    st.write("""
        - Modeling survey responses and customer preferences to inform decisions about product development and marketing strategies.
    """)

with st.expander("**Supply and Demand Forecasting**"):
    st.write("""
        - Modeling the distribution of demand to optimize production, inventory levels, and distribution strategies.
    """)
    
