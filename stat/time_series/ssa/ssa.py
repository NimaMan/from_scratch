import numpy as np
import pandas as pd


class SSA:
    '''Singular Spectrum Analysis
    Parameters:
    time_series: Pandas DataFrame object or Numpy array containing the time series
    embedding_dimension: Embedding (window) dimension of the time series
    suspected_frequency: Suspected frequency of the time series
    verbose: Print summary of the embedding
    return_df: Return the embedded time series as a DataFrame object
    '''

    def __init__(self, time_series_data, embedding_dimension=None, suspected_frequency=None, verbose=False, return_df=False):    
        self.ts = time_series_data
        self.ts_N = self.ts.shape[0]
        self.embedding_dimension = self.window_size = self.L = embedding_dimension
        self.K = self.ts_N - self.embedding_dimension + 1
        self.suspected_frequency = suspected_frequency
        self.verbose = verbose
        self.return_df = return_df
        self._compute_trajectory_matrix()
        self._compute_svd()
        self._compute_trajectory_matrix_rank()
        self._compute_elementary_matrices()
        
    def _compute_trajectory_matrix(self):
        '''Embed the time series'''
        if self.embedding_dimension is None:
            self.embedding_dimension = self.window_size = self.ts_N // 2

        self.trajectory_matrix = np.zeros((self.embedding_dimension, self.ts_N - self.embedding_dimension + 1))
        for i in range(self.embedding_dimension):
            self.trajectory_matrix[i] = self.ts[i:self.window_size + i]

        if self.verbose:
            print('Embedding dimension: {}'.format(self.embedding_dimension))
            print('Trajectory matrix shape: {}'.format(self.trajectory_matrix.shape))

    def _compute_svd(self):
        '''Compute the SVD of the trajectory matrix'''
        self.U, self.S, self.VT = np.linalg.svd(self.trajectory_matrix, full_matrices=False)
        self.V = self.VT.T
        if self.verbose:
            print('U matrix shape: {}'.format(self.U.shape))
            print('S matrix shape: {}'.format(self.S.shape))
            print('V matrix shape: {}'.format(self.V.shape))
    
    def _compute_trajectory_matrix_rank(self):
        '''Compute the rank of the trajectory matrix'''
        self.trajectory_matrix_rank = self.d = np.linalg.matrix_rank(self.trajectory_matrix)

        if self.verbose:
            print('Trajectory matrix rank: {}'.format(self.trajectory_matrix_rank))

    def _compute_elementary_matrices(self):
        '''Compute the elementary matrices of the SSA
        The elementary matrices are the matrices that are used to reconstruct the time series
        They are the product of the U and V matrices and singular values
        '''
        self.elementary_matrices = {}
        for i in range(self.trajectory_matrix_rank):
            self.elementary_matrices[i] = self.S[i] * np.outer(self.U[:, i], self.VT[i])
    
    def _elementary_matrix_to_time_series(self, elementary_matrix):
        """Averages the anti-diagonals of the given elementary matrix, X_i, and returns a time series.
        reference: https://www.kaggle.com/code/jdarcy/introducing-ssa-for-time-series-decomposition#4.-A-Python-Class-for-SSA"""

        # Reverse the column ordering of X_i
        X_rev = elementary_matrix[::-1]
        # Full credit to Mark Tolonen at https://stackoverflow.com/a/6313414 for this one:
        return np.array([X_rev.diagonal(i).mean() for i in range(-elementary_matrix.shape[0]+1, elementary_matrix.shape[1])])

    def _compute_component_time_series_from_elementary_matrices(self, components=None):
        '''Compute the SSA time series'''
        if components is None:
            components = np.arange(self.trajectory_matrix_rank)
        self.elements_time_series = {}
        for i in components:
            self.elements_time_series[i] = self._elementary_matrix_to_time_series(self.elementary_matrices[i])

    def compute_time_series_reconstruction(self, n_components=None):
        '''Compute the SSA reconstruction'''
        if n_components is None:
            n_components = self.trajectory_matrix_rank
            
        self.time_series_reconstruction = np.zeros(self.ts_N)
        for i in range(n_components):
            self.time_series_reconstruction += self.elements_time_series[i]

    def _compute_component_contributions(self):
        '''Compute the SSA component contributions'''
        sigma_squared =  self.S**2
        self.component_contributions = np.cumsum(sigma_squared) / np.sum(sigma_squared)
        
    def calc_wcorr(self):
        """
        Calculates the w-correlation matrix for the time series.
        """
             
        # Calculate the weights
        w = np.array(list(np.arange(self.L)+1) + [self.L]*(self.K-self.L-1) + list(np.arange(self.L)+1)[::-1])
        
        def w_inner(F_i, F_j):
            return w.dot(F_i*F_j)
        
        # Calculated weighted norms, ||F_i||_w, then invert.
        F_wnorms = np.array([w_inner(self.elements_time_series[i], self.elements_time_series[i]) for i in range(self.trajectory_matrix_rank)])
        F_wnorms = F_wnorms**-0.5
        
        # Calculate Wcorr.
        self.Wcorr = np.identity(self.d)
        for i in range(self.d):
            for j in range(i+1,self.d):
                self.Wcorr[i,j] = abs(w_inner(self.elements_time_series[i], self.elements_time_series[j]) * F_wnorms[i] * F_wnorms[j])
                self.Wcorr[j,i] = self.Wcorr[i,j]
    
    def _compute_orthonormal_base(self, num_components=None):
        '''Compute the orthonormal base of the SSA'''
        if num_components is None:
            num_components = self.trajectory_matrix_rank
        
        self.orthonormal_base = {i:self.U[:, i].reshape(-1, 1) for i in range(num_components)}
    
    def calculate_R(self,):
        """Calculate the R matrix of the SSA for forcasting"""
        R = np.zeros(self.orthonormal_base[0].shape)[0:-1]
        verticality_coefficient = 0
        for Pi in self.orthonormal_base.values():
            pi = Pi[-1].item()
            verticality_coefficient += pi**2
            R += pi*Pi[:-1]
        R /= (1-verticality_coefficient)
        
        return R
    
    def _compute_R_forecast(self, N_ahead):
        """Calculate the R matrix of the SSA for forcasting"""
        R = self.calculate_R()
        F_pred = self.ts.copy()
        for n in range(N_ahead):
            next_val = 0
            for i in range(len(R)):
                next_val += R[i]*F_pred[-self.window_size+1+i]
            F_pred = np.append(F_pred, next_val)
        
        return F_pred

    def forecast_ahead(self, N_ahead, forecast_method='R_forecasting'):
        """Forecast ahead N_ahead time steps"""
        if forecast_method == 'R_forecasting':
            return self._compute_R_forecast(N_ahead)
        else:
            raise ValueError('Forecast method not implemented')

    def plot_ssa_components(self, n_components=3):
        '''Plot the first n SSA components'''
        import matplotlib.pyplot as plt
        plt.figure(figsize=(15, 5))
        for i in range(n_components):
            plt.plot(self.ssa_components[i], label='Component {}'.format(i + 1))
        plt.legend()
        plt.show()

    def plot_ssa_reconstruction(self, n_components=3):
        '''Plot the reconstruction of the first n SSA components'''
        import matplotlib.pyplot as plt
        plt.figure(figsize=(15, 5))
        for i in range(n_components):
            plt.plot(self.ssa_components[i].cumsum(), label='Component {}'.format(i + 1))
        plt.legend()
        plt.show()
    


