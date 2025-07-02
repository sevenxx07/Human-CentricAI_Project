import os
import pandas as pd
import numpy as np

class Matrix_Factorization():

    def __init__(self, matrix):
        self.matrix = matrix

    def factorize(self, K, steps=100, alpha=0.01, lambd=0.1): 
        nr_users, nr_items = self.matrix.shape 
        print(f"Number of users:", nr_users)
        print(f"Number of items:", nr_items)

        # Initializing latent matrices
        # self.U = np.random.normal(scale=1./K, size = (nr_users,K))
        # self.V = np.random.normal(scale=1./K, size = (nr_items,K))

        self.U = np.random.rand(nr_users,K)
        self.V = np.random.rand(nr_items, K)

        #print("Initial latent matrix U:", U)
        #print("Initial latent matrix V:", V)

        # Get all ratings that are not NaN
        self.pairs = list(zip(*np.where(~np.isnan(R.values))))

        for step in range(self.steps):
            print("Step:", step)
            for i,j in self.pairs: 
                prediction = np.dot(self.U[i,:], self.V[j,:])
                error = self.matrix.iloc[i,j] - prediction

                #Using the gradient descent formula to update the latent matrices 
                self.U[i,:] += alpha * (error * self.V[j,:] - self.lambd*self.U[i,:])
                self.V[j,:] += alpha * (error * self.U[i,:] - self.lambd*self.V[j,:])

        return self.U, self.V
    
    def evaluate_loss(self):
        total_loss = np.inf
        for i,j in self.pairs: 
            prediction = np.dot(self.U[i,:], self.V[j,:])
            error = R.iloc[i,j] - prediction
            total_loss += error**2
        total_loss += self.lambd * (np.linalg.norm(self.U)**2 + np.linalg.norm(self.V)**2)
        if self.step % 10 == 0: 
            print(f"Step: {self.step}, Loss: {total_loss:.4f}")
    
    def split_data(self, ratio):
        nr_users, nr_items = self.matrix.shape 
        split_index = int(nr_users * ratio)
        matrix_train = self.matrix[:split_index]
        matrix_validation = self.matrix[split_index:]
        return matrix_train, matrix_validation 
    
    def cross_val(self, alpha_list: list, lambd_list: list, K_list: list, ratio):
        nr_users, nr_items = self.matrix.shape 
        split_index = int(nr_users * ratio)
        matrix_train = self.matrix[:split_index]
        matrix_validation = self.matrix[split_index:]
    
        for a in alpha_list: 
            for l in lambd_list:
                for k in K_list:
                    matrix_train.factorize(K=k, alpha=a, lambd=l)


# Getting the R matrix 
current_dir = os.path.dirname(__file__)
data_path = os.path.join(current_dir, 'R_matrix.csv')
PD = pd.read_csv(data_path, index_col=0)
alphas = [0.001, 0.005, 0.01, 0.05, 0.1]
lambdas = [0.01, 0.1, 1]
Ks = [11, 21, 51, 101]

R = Matrix_Factorization(PD)
#R.cross_val(alphas, lambdas, Ks, 0.8 )



alpha = [0.001, 0.005, 0.01, 0.05, 0.1]
lambdas = [0.01, 0.1, 1]
K = [11, 21, 51, 101]

# for a in alpha: 
#     for l in lambdas:
#         for k in K: 
#             print(matrix_factorization(R, K, steps = 101))
