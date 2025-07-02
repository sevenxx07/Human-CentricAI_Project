import os
import pandas as pd
import numpy as np

class Matrix_Factorization():

    def __init__(self, matrix):
        self.matrix = matrix

    def factorize(self, K, steps=10, alpha=0.01, lambd=0.1): 
        self.alpha = alpha
        self.K = K
        self.lambd = lambd
        self.steps = steps 
        nr_users, nr_items = self.matrix.shape 
        
        print(f"Number of users:", nr_users)
        print(f"Number of items:", nr_items)

        # Latent matrices
        self.U = np.random.rand(nr_users,K)
        self.V = np.random.rand(nr_items, K)

        self.pairs = list(zip(*np.where(~np.isnan(self.matrix.values))))

        for step in range(self.steps):
            print("Step:", step)
            for i,j in self.pairs: 
                prediction = np.dot(self.U[i,:], self.V[j,:])
                error = self.matrix.iloc[i,j] - prediction

                #Using the gradient descent formula to update the latent matrices 
                self.U[i,:] += alpha * (error * self.V[j,:] - self.lambd*self.U[i,:])
                self.V[j,:] += alpha * (error * self.U[i,:] - self.lambd*self.V[j,:])

        return self.U, self.V
    
    def evaluate_loss(self, validation_matrix):
        total_loss = 0
        validation_pairs = list(zip(*np.where(~np.isnan(validation_matrix.values))))
        for i,j in validation_pairs: 
            prediction = np.dot(self.U[i,:], self.V[j,:])
            error = validation_matrix.iloc[i,j] - prediction
            total_loss += error**2
        total_loss += self.lambd * (np.linalg.norm(self.U)**2 + np.linalg.norm(self.V)**2)
        if self.steps % 10 == 0: 
            print(f"Step: {self.steps}, Loss: {total_loss:.4f}")
        return total_loss
    
    def cross_val(self, alpha_list: list, lambd_list: list, K_list: list, ratio=0.8):
        nr_users = self.matrix.shape[0]
        split_index = int(nr_users * ratio)
        train_matrix = self.matrix.iloc[:split_index,:]
        validation_matrix = self.matrix.iloc[split_index:,:]

        best_loss = float('inf')
        best_parameters = {}
        for a in alpha_list: 
            for l in lambd_list:
                for k in K_list:
                    model = Matrix_Factorization(train_matrix)
                    model.factorize(K=k, alpha=a, lambd=l)
                    loss = model.evaluate_loss(validation_matrix)
                    print(f"Alpha={a}, Lambda = {l}, k={k}, loss= {loss:.4f}")
                    if loss < best_loss:
                        best_loss = loss
                        best_parameters = {"alpha": a,"lambda": l, "K": k}
        print("\n Best Parameters:", best_parameters)
        return best_parameters



# Getting the R matrix 
current_dir = os.path.dirname(__file__)
data_path = os.path.join(current_dir, 'R_matrix.csv')

R_matrix = pd.read_csv(data_path, index_col=0)

alphas = [0.001, 0.005, 0.01]
lambdas = [0.01, 0.1, 1.0]
Ks = [11, 21, 51]


mat_fac_model = Matrix_Factorization(R_matrix)
best_params = mat_fac_model.cross_val(alphas, lambdas, Ks, ratio=0.8)

