import os
import pandas as pd
import numpy as np

# Getting the R matrix 
current_dir = os.path.dirname(__file__)
data_path = os.path.join(current_dir, 'R_matrix.csv')
R = pd.read_csv(data_path, index_col=0)


# Matrix factorization to learn U, V
def matrix_factorization(R, K, steps=10, alpha=0.01, lambd=0.1 ):
    R = R.copy()
    nr_users, nr_items = R.shape 
    print(f"Number of users:", nr_users)
    print(f"Number of items:", nr_items)

    # Initializing latent matrices
    U = np.random.normal(scale=1./K, size = (nr_users,K))
    V = np.random.normal(scale=1./K, size = (nr_items,K))
    #print("Initial latent matrix U:", U)
    #print("Initial latent matrix V:", V)

    # Get all ratings that are not NaN
    pairs = list(zip(*np.where(~np.isnan(R.values))))

    for step in range(steps):
        print("Step:", step)
        for i,j in pairs: 
            prediction = np.dot(U[i,:], V[j,:])
            error = R.iloc[i,j] - prediction

            #Using the gradient descent formula to update the latent matrices 
            U[i,:] += alpha * (error * V[j,:] - lambd*U[i,:])
            V[j,:] += alpha * (error * U[i,:] - lambd*V[j,:])
        
        total_loss = 0
        for i,j in pairs: 
            prediction = np.dot(U[i,:], V[j,:])
            error = R.iloc[i,j] - prediction
            total_loss += error**2
        total_loss += lambd * (np.linalg.norm(U)**2 + np.linalg.norm(V)**2)
        if step % 10 == 0: 
            print(f"Step: {step}, Loss: {total_loss:.4f}")


    return U, V

K = 20
print(matrix_factorization(R, K))