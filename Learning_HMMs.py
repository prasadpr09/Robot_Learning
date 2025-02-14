# import numpy as np

# class HMM():

#     def __init__(self, Observations, Transition, Emission, Initial_distribution):
#         self.Observations = Observations
#         self.Transition = Transition
#         self.Emission = Emission
#         self.Initial_distribution = Initial_distribution

#     def forward(self):

#         # M would be according to the sequence of observasion 
#         # element wise dot product with that particular column of observation
#         alphas = []
#         alpha_0 = self.Initial_distribution * self.Observations[ :,obs_list[0]]
#         alphas.append(alpha_0)
#         prev_alpha = alpha_0

#         for i in range(1,20):

#             alpha = self.Observations[ :,obs_matrix[i]] * np.matmul(prev_alpha, self.Transition)

#             prev_alpha = alpha
#             alphas.append(alpha)
#             # print(alpha)
        
#         alphas = np.array(alphas)

#         # print("alphas", alphas)
#         # print(alphas.shape) # 20,2

#         return alphas

#     def backward(self):


#         beta_20 = [1,1]
#         prev_beta = beta_20
#         final_betas = []
#         final_betas.append(beta_20)

#         for i in range(19,0,-1):
#             beta_ = self.Observations[:,obs_matrix[i]] * prev_beta
#             prev_beta = beta_
#             betas = np.matmul(self.Transition , beta_)
#             final_betas.append(betas)

#         final_betas= np.array(final_betas)

#         betas_reversed  = final_betas[::-1, :]
#         # print(np.round(final_betas,3))
#         # print("betas", betas_reversed) # 20,2

#         return betas_reversed


#     # def backward(self, Observations, obs_list, Transition):
        
#     #     beta_T = np.array([1, 1])  # β_T = 1 for both states
#     #     final_betas = np.zeros((20, 2))  # Store β values for each time step
#     #     final_betas[20 - 1] = beta_T  # Last row = [1,1]

#     #     # Iterate backward from T-1 to 0
#     #     for t in range(20 - 2, -1, -1):
#     #         beta_next = final_betas[t + 1]  # β_{t+1}
#     #         obs_prob = Observations[:, obs_list[t + 1]]  # M(x', y_{t+1})
            
#     #         # Compute β_t using β_{t+1}, Transition, and Observation Matrix
#     #         final_betas[t] = np.dot(Transition, beta_next * obs_prob)

#     #     print("betas:\n", final_betas)  # Shape (20,2)
#     #     return final_betas


#     def gamma_comp(self, alphas, beta):

#         # normalisation = np.sum(alphas) 
#         gamma = np.zeros((20,2))

#         for i in range(0,20):
#             # element wise row multiplication 
#             gamma[i, :] = alphas[i,:] * beta[i, :]
#             gamma[i, :] /= np.sum(gamma[i, :])  # Normalize per time step
        
#         # print("Gamma:\n", np.round(gamma, 10))
#         return gamma


#     def xi_comp(self, alpha, beta, gamma):

#         # xi = np.zeros((20,4,4))
#         # for t in range(0,20):
#         #     for i in range(4):
#         #         xi[t,1,1] = 
#         #         xi[t,1,2] =  
#         #         xi[t,2,1] = 
#         #         xi[t,2,2] = 


#         T = self.Transition  # Transition matrix (2x2)
#         M = self.Emission  # Emission matrix (2x3)
#         obs_list = self.Observations  # Observation sequence
#         num_states = T.shape[0]  # Number of states (2)
#         num_time_steps = alpha.shape[0] - 1  # 19 transitions

#         xi = np.zeros((num_time_steps, num_states, num_states))

#         for k in range(num_time_steps):
#             denom = 0  # Denominator for normalization
#             for x in range(num_states):
#                 for x_next in range(num_states):
#                     xi[k, x, x_next] = (alpha[k, x] * T[x, x_next] * M[x_next, obs_list[k + 1]] * beta[k + 1, x_next])
#                     denom += xi[k, x, x_next]  # Sum for normalization

#             if denom > 0:
#                 xi[k] /= denom  # Normalize each transition probability matrix

#         print("Xi values:\n", np.round(xi, 10))

#         print(xi.shape)
#         return xi


#     def update(self, alpha, beta, gamma, xi):

#         new_init_state = ...
#         T_prime = ...
#         M_prime = ...



#         return T_prime, M_prime, new_init_state



#     def trajectory_probability(self, alpha, beta, T_prime, M_prime, new_init_state):

#         P_original = ...
#         P_prime = ...

#         return P_original, P_prime
    

# if __name__ == '__main__':
#     obs_matrix = np.array([[0.4, 0.1, 0.5], [0.1, 0.5, 0.4]])
#     trans_matrix = np.array([[0.5, 0.5], [0.5, 0.5]])
#     obs_list = np.array([2, 0, 0, 2, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 2, 0, 0, 1])
#     init_dist = np.array((0.5, 0.5))
#     hmm = HMM(obs_list, trans_matrix, obs_matrix, init_dist)
#     # alphas = hmm.forward()
#     # print(alphas)

#     # Additionally, the output for alpha and beta should be a shape (20,2) which you will use to create a table in your report. 


import numpy as np

class HMM():
    def __init__(self, Observations, Transition, Emission, Initial_distribution):
        self.Observations = Observations
        self.Transition = Transition
        self.Emission = Emission
        self.Initial_distribution = Initial_distribution



    def forward(self):
        alphas = []
        alpha_0 = self.Initial_distribution * self.Emission[:, self.Observations[0]]
        alphas.append(alpha_0)
        prev_alpha = alpha_0

        for i in range(1, 20):
            alpha = self.Emission[:, self.Observations[i]] * np.matmul(prev_alpha, self.Transition)
            prev_alpha = alpha
            alphas.append(alpha)

        alphas = np.array(alphas)
        return alphas




    def backward(self):
        beta_20 = np.array([1, 1])  # Ensure it's a NumPy array
        prev_beta = beta_20
        final_betas = []
        final_betas.append(beta_20)

        for i in range(19, 0, -1):
            beta_ = self.Emission[:, self.Observations[i]] * prev_beta
            prev_beta = np.matmul(self.Transition, beta_)
            final_betas.append(prev_beta)

        final_betas = np.array(final_betas)
        betas_reversed = final_betas[::-1, :]
        return betas_reversed




    def gamma_comp(self, alphas, beta):
        gamma = np.zeros((20, 2))
        for i in range(0, 20):
            gamma[i, :] = alphas[i, :] * beta[i, :]
            gamma[i, :] /= np.sum(gamma[i, :])  # Normalize per time step
        return gamma



    def xi_comp(self, alpha, beta, gamma):
        T = self.Transition  
        M = self.Emission  
        obs_list = self.Observations  
        num_states = T.shape[0]  
        num_time_steps = alpha.shape[0] - 1  

        xi = np.zeros((num_time_steps, num_states, num_states))

        for k in range(num_time_steps):
            denom = 0  
            for x in range(num_states):
                for x_next in range(num_states):
                    xi[k, x, x_next] = (alpha[k, x] * T[x, x_next] * M[x_next, obs_list[k + 1]] * beta[k + 1, x_next])
                    denom += xi[k, x, x_next]  

            if denom > 0:
                xi[k] /= denom  

        return xi




    def update(self, alpha, beta, gamma, xi):
        num_states = self.Transition.shape[0]
        num_observations = self.Emission.shape[1]
        T_prime = np.zeros_like(self.Transition)
        M_prime = np.zeros_like(self.Emission)

        # Update initial state distribution
        new_init_state = gamma[0]

        # Update transition matrix
        for x in range(num_states):
            denom = np.sum(gamma[:-1, x])
            if denom > 0:
                for x_next in range(num_states):
                    T_prime[x, x_next] = np.sum(xi[:, x, x_next]) / denom # adding all i,0,0 and i,0,1 and etc..

        # Update emission matrix
        for x in range(num_states):
            denom = np.sum(gamma[:, x])
            if denom > 0:
                for y in range(num_observations):
                    M_prime[x, y] = np.sum(gamma[:, x] * (self.Observations == y)) / denom

        return T_prime, M_prime, new_init_state


    def trajectory_probability(self, alpha, beta, T_prime, M_prime, new_init_state):
        # Compute the probability of the observations under the original model
        P_original = np.sum(alpha[-1])  # P(Y1, ..., Yt | λ) using forward algorithm
        
        # Compute alphas for the updated model
        updated_hmm = HMM(self.Observations, T_prime, M_prime, new_init_state)
        alpha_prime = updated_hmm.forward()
        
        # Compute the probability of the observations under the updated model
        P_prime = np.sum(alpha_prime[-1])  # P(Y1, ..., Yt | λ′)
        
        return P_original, P_prime




if __name__ == '__main__':
    obs_matrix = np.array([[0.4, 0.1, 0.5], [0.1, 0.5, 0.4]])
    trans_matrix = np.array([[0.5, 0.5], [0.5, 0.5]])
    obs_list = np.array([2, 0, 0, 2, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 2, 0, 0, 1])
    init_dist = np.array([0.5, 0.5])
    
    hmm = HMM(obs_list, trans_matrix, obs_matrix, init_dist)

    alphas = hmm.forward()
    print(alphas)

    beta = hmm.backward()
    print(beta)

    gamma = hmm.gamma_comp(alphas, beta)
    print(gamma)

    xi = hmm.xi_comp(alphas,beta,gamma)
    print(xi)

    T_prime, M_prime, new_init_state = hmm.update(alphas,beta,gamma, xi)
    print(np.round(T_prime,13),np.round(M_prime,13), new_init_state)


    P_orignal, P_prime = hmm.trajectory_probability(alphas,beta, T_prime, M_prime, new_init_state)
    print(np.round(P_orignal,13),np.round(P_prime,13))