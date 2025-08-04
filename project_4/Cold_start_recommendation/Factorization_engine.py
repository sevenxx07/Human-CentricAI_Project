import os
import pandas as pd
import numpy as np
import math

class Matrix_Factorization():

    def __init__(self, matrix):
        self.matrix = matrix
    
    # Matrix factorization usign stochastic gradient descent     
    def factorize(self, K, steps=10, alpha=0.005, lambd=0.01):
        self.alpha = alpha # Learning rate
        self.K = K # Nr of latent features
        self.lambd = lambd # Regularization term
        self.steps = steps # Nr of training iterations
        nr_users, nr_items = self.matrix.shape 
        
        print(f"Number of users:", nr_users)
        print(f"Number of items:", nr_items)

        # Latent matrices initialization 
        self.U = np.random.rand(nr_users,K)
        self.V = np.random.rand(nr_items, K)
       
        # Extracting all known ratings 
        self.pairs = list(zip(*np.where(~np.isnan(self.matrix.values))))

        # Computing the prediction error for all known ratings. 
        for step in range(self.steps):
            print("Step:", step)
            for i,j in self.pairs: 
                prediction = np.dot(self.U[i,:], self.V[j,:])
                error = self.matrix.iloc[i,j] - prediction

                # Using gradient descent to update the latent matrices 
                self.U[i,:] += alpha * (error * self.V[j,:] - self.lambd*self.U[i,:])
                self.V[j,:] += alpha * (error * self.U[i,:] - self.lambd*self.V[j,:])  

        return self.U, self.V
    
    # Evaluating the RMSE on the validation set
    def evaluate_loss(self, validation_matrix):
        total_loss = 0
        validation_pairs = list(zip(*np.where(~np.isnan(validation_matrix.values))))
        for i,j in validation_pairs: 
            prediction = np.dot(self.U[i,:], self.V[j,:])
            if prediction > 1: 
                print("Prediction value exceeeds 1") 
            if math.isnan(validation_matrix.iloc[i,j]) == False: 
                error = validation_matrix.iloc[i,j] - prediction
                if error > 1: 
                    print("ERROR TO BIG")
            total_loss += np.sqrt(error**2)
        total_loss_norm = total_loss / len(validation_pairs)
        if self.steps % 10 == 0: 
            print(f"Step: {self.steps}, Loss: {total_loss_norm:.4f}")
        return total_loss_norm

    # Hyperparameter tuning using grid search

    # def cross_val(self, alpha_list: list, lambd_list: list, K_list: list, ratio=0.8):
    #     nr_users = self.matrix.shape[0]
    #     split_index = int(nr_users * ratio)
    #     train_matrix = self.matrix.iloc[:split_index,:]
    #     validation_matrix = self.matrix.iloc[split_index:,:]

    #     best_loss = float('inf')
    #     best_parameters = {}
    #     for a in alpha_list: 
    #         for l in lambd_list:
    #             for k in K_list:
    #                 model = Matrix_Factorization(train_matrix)
    #                 model.factorize(K=k, alpha=a, lambd=l)
    #                 loss = model.evaluate_loss(validation_matrix)
    #                 print(f"Alpha={a}, Lambda = {l}, k={k}, loss= {loss:.4f}")
    #                 if loss < best_loss:
    #                     best_loss = loss
    #                     best_parameters = {"alpha": a,"lambda": l, "K": k}
    #     print("\n Best Parameters:", best_parameters)
    #     return best_parameters

# Loading the R-matrix and running the matrix factorization with K=11 (best value found from cross-validation) 
def get_R_U_V():
    # Getting the R matrix
    current_dir = os.path.dirname(__file__)
    data_path = os.path.join(current_dir, 'R_matrix.csv')

    R_matrix = pd.read_csv(data_path, index_col=0)

    mat_fac_model = Matrix_Factorization(R_matrix)
    U, V = mat_fac_model.factorize(11)
    print(U)
    return mat_fac_model, R_matrix, U, V


# Runs the grid search 
# def find_best_params(mat_fac_model):
#     alphas = [0.001, 0.005, 0.01]
#     lambdas = [0.01, 0.1, 1.0]
#     Ks = [11, 21, 51]
#     best_params = mat_fac_model.cross_val(alphas, lambdas, Ks, ratio=0.8)
#     #Best Parameters: {'alpha': 0.005, 'lambda': 0.01, 'K': 11}

if __name__ == "__main__":
    model, R, U, V = get_R_U_V()