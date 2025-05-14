import numpy as np
import pandas as pd
import networkx as nx
from collections import defaultdict

from tools import evaluate_model

class GlotMF:
    def __init__(self, num_users, num_items, K=20, alpha=0.4, lr=0.005, lambda_reg=0.01, lambda_B=0.1, lambda_E=0.1):
        self.num_users = num_users
        self.num_items = num_items
        self.K = K
        self.alpha = alpha
        self.lr = lr
        self.lambda_reg = lambda_reg
        self.lambda_B = lambda_B
        self.lambda_E = lambda_E

        self.U = np.random.normal(0, 0.1, (num_users, K))
        self.V = np.random.normal(0, 0.1, (num_items, K))

        self.user_bias = np.zeros(num_users)
        self.item_bias = np.zeros(num_items)
        self.global_mean = 0

    def train(self, ratings_df, reputation_weights, user_id_to_idx, item_id_to_idx, trust_neighbors, epochs=10):

        self.global_mean = ratings_df['label'].mean()

        reputation_weights = reputation_weights / (np.sum(reputation_weights) + 1e-8)

        
        for epoch in range(epochs):
            for row in ratings_df.itertuples(index=False):
                u = user_id_to_idx[row.user_id]
                i = item_id_to_idx[row.item_id]
                r_ui = row.label

                pred = self.predict_rating(u, i)
                err = r_ui - pred

                U_u_old = self.U[u].copy()
                V_i_old = self.V[i].copy()

                self.U[u] += self.lr * (err * V_i_old - self.lambda_reg * self.U[u])
                self.V[i] += self.lr * (err * U_u_old - self.lambda_reg * self.V[i])
                self.user_bias[u] += self.lr * (err - self.lambda_reg * self.user_bias[u])
                self.item_bias[i] += self.lr * (err - self.lambda_reg * self.item_bias[i])

            # Social regularization with reputation weights
            for u in range(self.num_users):
                neighbors = trust_neighbors.get(u, [])
                if not neighbors:
                    continue

                weights = np.array([reputation_weights[v] for v in neighbors])
                weights_sum = weights.sum()
                if weights_sum == 0:
                    continue
                weights /= weights_sum  # Normalize

                for idx, v in enumerate(neighbors):
                    w = weights[idx]
                    self.U[u] += self.lr * self.lambda_B * w * (self.U[v] - self.U[u])

            self.lr = self.lr * 0.95

            epoch_mae, epoch_rmse, epoch_mse, epoch_r2 = evaluate_model(self, ratings_df, user_id_to_idx, item_id_to_idx)
            print(f"Epoch {epoch+1} MAE: {epoch_mae:.4f} RMSE: {epoch_rmse:.4f}, MSE {epoch_mse}, R2: {epoch_r2}")



    def predict_rating(self, u, i, Rmax=5):
        pred = self.global_mean + self.user_bias[u] + self.item_bias[i] + np.dot(self.U[u], self.V[i])
        return np.clip(pred, 0, Rmax)


class GlotMFPreprocessor:
    def __init__(self, ratings_path, trust_path):
        self.ratings_path = ratings_path
        self.trust_path = trust_path

    def load_data(self):
        self.ratings_df = pd.read_csv(self.ratings_path)
        self.trust_df = pd.read_csv(self.trust_path)

        # Normalize ratings between 0 and 1
        # self.rating_min = self.ratings_df['label'].min()
        # self.rating_max = self.ratings_df['label'].max()
        # self.ratings_df['label_normalized'] = (self.ratings_df['label'] - self.rating_min) / (self.rating_max - self.rating_min)

        self.user_ids = sorted(set(self.ratings_df['user_id']) | 
                               set(self.trust_df['user_id_trustor']) |
                               set(self.trust_df['user_id_trustee']))
        self.item_ids = sorted(self.ratings_df['item_id'].unique())

        self.user_id_to_idx = {uid: idx for idx, uid in enumerate(self.user_ids)}
        self.item_id_to_idx = {iid: idx for idx, iid in enumerate(self.item_ids)}

        self.num_users = len(self.user_ids)
        self.num_items = len(self.item_ids)

        print(f"# Users: {self.num_users}, # Items: {self.num_items}")

        return self.ratings_df, self.trust_df

    def compute_pagerank(self):
        G = nx.DiGraph()
        for _, row in self.trust_df.iterrows():
            src = self.user_id_to_idx[row['user_id_trustor']]
            tgt = self.user_id_to_idx[row['user_id_trustee']]
            weight = row['trust_value']
            G.add_edge(src, tgt, weight=weight)

        pagerank_scores = nx.pagerank(G, weight='weight')
        reputation = np.array([pagerank_scores.get(i, 0.0) for i in range(self.num_users)])

        rep_scaled = 1.0 / (1 + np.log(reputation + 1e-6))
        return rep_scaled
    
    def build_trust_neighbors(self):
        trust_neighbors = defaultdict(list)
        for row in self.trust_df.itertuples(index=False):
            u = self.user_id_to_idx[row.user_id_trustor]
            v = self.user_id_to_idx[row.user_id_trustee]
            trust_neighbors[u].append(v)
        return trust_neighbors

