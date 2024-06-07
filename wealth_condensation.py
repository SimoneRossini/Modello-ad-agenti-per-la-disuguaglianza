"""(c) Stefano Zapperi 2023


Python module to study simple agent based models of wealth condensation. 

The model proposed by JPB&MM (Bouchaud, J. P., & Mézard, M. (2000). Wealth condensation in a simple model of economy. Physica A: Statistical Mechanics and its Applications, 282(3-4), 536-545. https://www.sciencedirect.com/science/article/pii/S0378437100002053 ) in the variant proposed by JPB that includes taxes and the governement (Bouchaud, J. P. (2015). On growth-optimal tax rates and the issue of wealth inequalities. Journal of Statistical Mechanics: Theory and Experiment, 2015(11), P11011. https://iopscience.iop.org/article/10.1088/1742-5468/2015/11/P11011/pdf)

In the continuum version of the model, we consider a random interaction network schematized by an adjaciency matrix $$A^0_{ij}=A^0_{ji}$$.
The random network is constructed assigning a probality $$p=c/N$$ for a link to be present.

he equations of the model are the following for individuals:

$$ dW_i/dt = \eta_i(t) W_i + J_0\sum_j A^0_{ij}(W_j -W_i) - \phi W_i + f/N V$$

where parameters are: 
N number of agents
$$W_i$$ (w[i]) wealth of individual i
$$\phi$$ (phi) tax rate
f tax redistribution rate
$$J_{ij}=J_0$$ is a constant
V state wealth
$$\eta_i(t)$$ (eta[i]) Gaussian random noise with mean m and variance s.

The equation for the state is

dV.dt =  sigma  xi(t)  V + phi W + (mu-f)V

where:
- W is the total wealth
- xi is the noise (with mean 0 and variance 1)
- (sigma) is the real variance and  (mu) is the mean

To solve the model is convenient to express it in terms of an 
(N+1)-dimensional system of equations for the vector 
$$X = (W_0, ...., W_{N-1}, V)$$
which obeys

$$dX_i = \sum_j A_{ij} X_j dt + B_i X_i d\psi_i$$

where:
$$A_{ij} = A^0_{ij}J_0 + \delta_{ij} (m-J_0\sum_k A^0_{ki} -\phi)  \mbox{ for } i,j=0,N-1$$   
$$A_{iN}=f/N \mbox{ for } i<N$$ 
$$A_{Ni}=\phi  \mbox{ for } i<N$$
$$A_{NN}=(\mu-f)$$
$$B_i = s \mbox{ for } i<N$$
$$B_i=\sigma \mbox{ for } i=N$$
$$d\psi_i$$ are independent Wiener processes (with variance 1 and zero mean)


"""
import numpy as np
import random
import sdeint
import scipy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm
import networkx as nx



def sun_graph_rand(n_core,n_branch):
    G=nx.random_regular_graph(n_core-1,n_core)
    
    for i in range(n_branch):
        j=random.randint(0,n_core-1)
        G.add_node(n_core+i)
        G.add_edge(j,n_core+i)
    
    return G

def branch_graph(n_core,n_branch):
    
    G=nx.random_regular_graph(n_core-1,n_core)
      
    
    for i in range(n_branch):
        
        G.add_node(n_core+i)
        G.add_edge(i,n_core+i)
       
            
    

    return G

def probability_distribution(wealth_share,num):
    wealth=np.log10(wealth_share)
    wealth=wealth.reshape(-1)
    wealth=np.sort(wealth)
    
    
    intervals=np.linspace(wealth[0], wealth[-1], num)
    dw=intervals[1]-intervals[0]
    wealth_av=np.zeros(len(intervals)-1)
    
    for i in range(len(intervals)-1):
        wealth_av[i]=(np.power(10,intervals[i])+np.power(10,intervals[i+1]))/2
        
    p=np.zeros(len(intervals)-1)
    count=np.zeros(len(intervals)-1)
    P=np.zeros(len(intervals)-1)
    
    for i in range(len(intervals)-1):
        for j in range(len(wealth)):
            if intervals[i]<=wealth[j]<intervals[i+1]:
                   count[i]=count[i]+1
                    
        p[i]=(count[i]/dw)*(1/wealth_av[i])
        
        P[i]=count[i]/len(wealth)
    
    return wealth_av,p,P

def cumulative_probability(p):
    
    p_c=np.zeros(len(p))
    p_c=np.cumsum(p)
    
    for i in range(len(p)-1):
        p_c[i+1]=p_c[i]-p[i+1]
    
    
   
        
    return p_c/p_c[0]
        

class simulate:
    """ Class contains functions needed to simulate
    the model """""
    
    def adjaciency_matrix(c,N):
        """ The function returns the adjecency matrix
        A0[i,j] of the ineraction network where each
        link is present with probability p=c/N
        Dependencies:
        -numpy as np
        """
        p=c/N
        ### construct the adjacency matrix
        A0=np.zeros((N, N))
        for i in range(N):
            for j in range(i+1,N):
                if(np.random.rand()<p):
                    A0[i,j]=1
                    A0[j,i]=1
        return A0

    def interaction_matrix(N,mu,sigmat,m,phi,f,J0,A0,sigma,s,c):
        """The function returns the interaction 
        matrix A and the noise matrix B for the BM model. 
        The matrix A is defined as:
        A[i,j] = A0[i,j](1-\delta_{ij})JJ[i,j] + \delta_{ij} (m-\sum_k J[k,i] -phi) 
        for i,j=0,N-1   
        A[i,N]=f/N for i<N 
        A[N,i]=phi for i<N
        A[N,N]=mu-f
        
        A0[i,j] is the adjecency matrix
        of the ineraction network 
    
        The matrix B describes the noise variance.

        Dependencies:
        - numpy as np

        Returns the interaction matrix A and the noise variance vector B
        """

        #    TT=np.random.randn(N+1)*sigmat    
        ## Construct the interaction matrix
        # Create an empty N+1xN+1 matrix
        A = np.zeros((N+1, N+1))
        # Fill the matrix with the values for i,j < N
        for i in range(N):
            for j in range(N):
                if A0[i,j]==1:
                    A[i,j] = (J0/c)
                
        # fill in the diagonal
        for i in range(N):
            sumj=0.
            for j in range(N):
                sumj=sumj+A0[j,i]
                
            A[i,i]=m-sumj*(J0/c)-phi
        
        for i in range(N):
            ## fill in last rows/colums
            A[i,N]=f/N
            A[N,i]=phi
            A[N,N]=mu-f
    
        ## define matrix B
        B0=np.sqrt(2.)*s*np.ones(N+1)
        B0[N]=np.sqrt(2.)*sigma
        B=np.diag(B0)
    
        return A,B
    
    
    def interaction_matrix_sun(N,mu,sigmat,m,phi,f,J0,A0,sigma,s,c):
        """The function returns the interaction 
        matrix A and the noise matrix B for the BM model. 
        The matrix A is defined as:
        A[i,j] = A0[i,j](1-\delta_{ij})JJ[i,j] + \delta_{ij} (m-\sum_k J[k,i] -phi) 
        for i,j=0,N-1   
        A[i,N]=f/N for i<N 
        A[N,i]=phi for i<N
        A[N,N]=mu-f
        
        A0[i,j] is the adjecency matrix
        of the ineraction network 
    
        The matrix B describes the noise variance.

        Dependencies:
        - numpy as np

        Returns the interaction matrix A and the noise variance vector B
        """

        #    TT=np.random.randn(N+1)*sigmat    
        ## Construct the interaction matrix
        # Create an empty N+1xN+1 matrix
        
        A = np.zeros((N+1, N+1))
        # Fill the matrix with the values for i,j < N
        for i in range(N):
            for j in range(N):
                if A0[i,j]==1:
                    A[i,j] = (J0/c[j])
       
        # fill in the diagonal
        for i in range(N):
            sumj=0.
            for j in range(N):
                sumj=sumj+A0[j,i]
               
            A[i,i]=m-sumj*(J0/c[i])-phi
        
        for i in range(N):
            ## fill in last rows/colums
            A[i,N]=f/N
            A[N,i]=phi
            A[N,N]=mu-f
    
        ## define matrix B
        B0=np.sqrt(2.)*s*np.ones(N+1)
        B0[N]=np.sqrt(2.)*sigma
        B=np.diag(B0)
    
        return A,B
     
    def integrate_sde(x0,A,B,t_tot,dt):
        """The function integrates a N-dimensional stochastic of the type 
        dx = Axdt + Bx dw using Stratonovich interpretation
        
        Dependencies:
        - numpy as np
        - sdeint

        Returns: the time dependent solution x
        """
        tspan = np.linspace(0.0, t_tot, int(t_tot/dt))
        def f(x, t):
            return A.dot(x)
        def G(x, t):
            y=np.diag(x)
            return B.dot(y)
        result = sdeint.stratHeun(f, G, x0, tspan)
        return result

class analyze:
    """ This class contains the functions needed to analyze the data resulting from
    the simulations. """
    
    def shorrocks_index(x1,x2,N,q=10):
        """Computes the Shorrocks mobility index with q categories
        for vectors of lengths N
        x1[i] is the vector of initial categories i=0,...q-1  
        x2[j] is the vector of final categories j=0,...q-1

        Returns: Shorrocks index
        """
        trace=(x1==x2).sum()/N
        
        s_q=(q-q*trace)/(q-1)
        return s_q,trace

    def shorrocks_matrix(x1,x2,N,q):
        """Computes the Shorrocks probabilty matrix
        describing mobilities with q categories
        for vectors of lengths N
        x1[i] =0,...,q-1 is the vector of initial categories i=0,...N-1  
        x2[j]= 0,...,q-1 is the vector of final categories j=0,...N-1

        Dependencies:
        -numpy as np

        Returns: Shorrocks mobility matrix
        """
        m_sh=np.zeros((q,q))
        norm_sh=np.zeros(q)
        for i in range(N):
            m_sh[x1[i],x2[i]]=m_sh[x1[i],x2[i]]+1
            norm_sh[x1[i]]=norm_sh[x1[i]]+1

        for iq in range(q):
            if norm_sh[iq]!=0:
                m_sh[iq,:]=m_sh[iq,:]/norm_sh[iq]

        return m_sh

    def persistence_top(x1,x2,N,q=10):
        """Computes the persistence in the top category with q categories
        for vectors of lengths N
        x1[i] is the vector of initial categories i=1,...,q 
        x2[j] is the vector of final categories j=1,...,q
        Returns: Persistence in the top category
        """
        p_top=((x1==x2)&(x1==q-1)).sum()
        norm=(x1==q-1).sum()
        
        p_top=p_top/norm
        
        
        return p_top
    
    def persistence_5060(x1,x2,N,q=10):
        """Computes the persistence in the top category with q categories
        for vectors of lengths N
        x1[i] is the vector of initial categories i=1,...,q 
        x2[j] is the vector of final categories j=1,...,q
        Returns: Persistence in the top category
        """
        p_mid=((x1==x2)&(x1==5)).sum()
        norm=(x1==5).sum()
        
        p_mid=p_mid/norm
        
        
        return p_mid

    def persistence_bottom(x1,x2,N,q=10):
        """Computes the persistence in the top category with q categories
        for vectors of lengths N
        x1[i] is the vector of initial categories i=1,...,q 
        x2[j] is the vector of final categories j=1,...,q
        Returns: Probability to remain in the bottom category
        """
        p_bottom=((x1==x2)&(x1==0)).sum()
        norm=(x1==0).sum()
        
        p_bottom=p_bottom/norm
            
        return p_bottom
    
    def persistence_top_series(ranks,N,q=10,n_step=1000):
        """Computes the persistence at the top with q categories
           for time series.
       
           Input: an array (ranks) displying the rank of
           of N agents for n_step timestep
       
           Returns: Persistence in the top q-ile vs time
        """    
        ranks_q=np.int32(q*ranks/N)
        x1=ranks_q[0]
        p_top=np.zeros(n_step)
        for i in range(n_step):
            x2=ranks_q[i]
            p_top[i]=analyze.persistence_top(x1,x2,N,q)
            
        return p_top
    
    def persistence_top_series_core(ranks,N,position_core,q=10,n_step=1000):
        """Computes the persistence at the top with q categories
           for time series.
       
           Input: an array (ranks) displying the rank of
           of N agents for n_step timestep
       
           Returns: Persistence in the top q-ile vs time
        """    
        ranks_q=np.int32(q*ranks/N)
        ranks_q=ranks_q[:,position_core.astype(int)]
        x1=ranks_q[0]
        p_top=np.zeros(n_step)
        
        for i in range(n_step):
            x2=ranks_q[i]
            p_top[i]=analyze.persistence_top(x1,x2,N,q)
            
        return p_top
    
    def persistence_5060_series(ranks,N,q=10,n_step=1000):
        """Computes the persistence at the top with q categories
           for time series.
       
           Input: an array (ranks) displying the rank of
           of N agents for n_step timestep
       
           Returns: Persistence in the top q-ile vs time
        """    
        ranks_q=np.int32(q*ranks/N)
        x1=ranks_q[0]
        p_mid=np.zeros(n_step)
        for i in range(n_step):
            x2=ranks_q[i]
            p_mid[i]=analyze.persistence_5060(x1,x2,N,q)
            
        return p_mid

    
    
    def persistence_bottom_series(ranks,N,q=10,n_step=1000):
        """Computes the persistence at the bottom with q categories
           for time series.
       
           Input: an array (ranks) displying the rank of
           of N agents for n_step timestep
       
           Returns: Persistence in the bottom q-ile vs time
        """    
        ranks_q=np.int32(q*ranks/N)
        x1=ranks_q[0]
        p_bottom=np.zeros(n_step)
        for i in range(n_step):
            x2=ranks_q[i]
            p_bottom[i]=analyze.persistence_bottom(x1,x2,N,q)
            
        return p_bottom

    def shorrocks_series(ranks,N,q=10,n_step=1000):
        """Computes the Shorrocks mobility index with q categories
           for time series.
       
           Input: an array (ranks) displying the rank of
           of N agents for n_step timestep
       
           Returns: Shorrocks index time series
        """    
        ranks_q=np.int32(q*ranks/N)
        x1=ranks_q[0]
        s_index=np.zeros(n_step)
        for i in range(n_step):
            x2=ranks_q[i]
            
            s_index[i]=analyze.shorrocks_index(x1,x2,N,q)[0]
            
        return s_index
    
    def network_structure(A0):
        """ return the network associated 
        to the adjeciency matrix A0

        Dependencies:
        -pandas as pd
        -networkx as nx
        """   
        df_J=pd.DataFrame(data=A0)
        # Transform it in a links data frame (3 columns only):
        links = df_J.stack().reset_index()
        links.columns = ['var1', 'var2','value']
        links_filtered=links.loc[ (links['value'] > 0.8)]
        # Build graph
        G=nx.from_pandas_edgelist(links_filtered, 'var1', 'var2')
        return G


    def plot_network(G,node_size=100,node_color="orange", with_labels=True):
        """ Plot the network G 
        input parameters:/notebooks/
        node_size: number of nodes
        color of node: node_color
        with labels: True/False
    
        Dependencies:
        -matplotlib.pyplot as plt
        -networkx as nx
        """
        fig = plt.figure(figsize=(8, 8))
        nx.draw(G, with_labels=True, 
                node_color=node_color, 
                node_size=node_size, 
                edge_color='black', 
                linewidths=1, font_size=15)

    def inv_part_ratio(wealth_share,N):
        """ Compute inverse participation ratio of 
            wealth share time series

            Dependencies:
            -numpy as np
        """
        w_sq=wealth_share*wealth_share
        w2=np.sum(w_sq, axis=1)
        Y2=np.mean(w2)
        return Y2
    
    def bottom50_share(wealth_share,N):
        """ Compute fraction of wealth of bottom 50% 
         Input is a wealth share time series    
         Returns: average of the fractions,
        """
        
        j=int(5*N/10)
        ww=np.sort(wealth_share)
        bottom50=ww[:,:j].sum(axis=1).mean()
        return bottom50
    
    def top10_share(wealth_share,N):
        """ Compute fraction of wealth of top 10%
            
         Input is a wealth share time series    
         Returns: average of the fractions.
        """
        k=int(9*N/10)
        ww=np.sort(wealth_share)
        top10=ww[:,k:].sum(axis=1).mean()
        return top10
    
    def bottom10_share(wealth_share,N):
        """ Compute fraction of wealth of bottom 10%
            
         Input is a wealth share time series    
         Returns: average of the fractions.
        """
        k=int(N/10)
        ww=np.sort(wealth_share)
        bottom10=ww[:,:k].sum(axis=1).mean()
        return bottom10
    

    
    def gini(array):
        """Calculate the Gini coefficient of a numpy array."""
        # based on bottom eq: http://www.statsdirect.com/help/content/image/stat0206_wmf.gif
        # from: http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
        array = array.flatten() #all values are treated equally, arrays must be 1d
        if np.amin(array) < 0:
            array -= np.amin(array) #values cannot be negative
        array += 0.0000001 #values cannot be 0
        array = np.sort(array) #values must be sorted
        index = np.arange(1,array.shape[0]+1) #index per array element
        n = array.shape[0]#number of array elements
        return ((np.sum((2 * index - n  - 1) * array)) / (n * np.sum(array))) #Gini coefficient
    
    def gini_series(x,n_step=1000):
        """ Compute Gini index for series of arrays"""
        g_index=np.zeros(n_step)
        for i in range(n_step):
            x1=x[i]
            g_index[i]=analyze.gini(x1)
            
        return g_index
    
    def M_perfect_mix(q,Q):
    
        M=np.zeros(Q)
    
        for i in range(Q):
        
            qs=q[i]*q[i]
        
            qQ=q[i]*(Q-1)
            Q3=(Q*Q)/3
            Q2=Q/2
        
            M[i]=(qs-qQ+Q3-Q2+(1/6)) ** 0.5
        
        return M