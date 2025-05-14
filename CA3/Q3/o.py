import pandas as pd
import os
import itertools
import json
from sklearn.model_selection import train_test_split
import networkx as nx
from collections import defaultdict
import numpy as np
from core import GlotMFPreprocessor, GlotMF
from tools import evaluate_model, generate_submission

RATING_DATA_PATH = 'content/q3_dataset/train_data_movie_rate.csv'
TRUST_DATA_PATH = 'content/q3_dataset/train_data_movie_trust.csv'
TEST_DATA_PATH = 'content/q3_dataset/test_data.csv'
MODEL_CONFIG_PATH = 'content/q3_dataset/config.json'
SUBMISSION_PATH = "content/q3_dataset/submission.csv"
epoch_count = 10

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

        best_rmse = float('inf')
        patience = 3
        no_improve_count = 0
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

            epoch_mae, epoch_rmse = evaluate_model(self, ratings_df, user_id_to_idx, item_id_to_idx)
            print(f"Epoch {epoch+1} MAE: {epoch_mae:.4f} RMSE: {epoch_rmse:.4f}")



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
    
    def build_trust_neighbors(self, ):
        trust_neighbors = defaultdict(list)
        for row in self.trust_df.itertuples(index=False):
            u = self.user_id_to_idx[row.user_id_trustor]
            v = self.user_id_to_idx[row.user_id_trustee]
            trust_neighbors[u].append(v)
        return trust_neighbors





def evaluate_model(glotmf, ratings_df, user_id_to_idx, item_id_to_idx, Rmax=5, rating_min=1, rating_max=5):
    preds, truths = [], []

    for row in ratings_df.itertuples(index=False):
        u = user_id_to_idx[row.user_id]
        i = item_id_to_idx[row.item_id]
        preds.append(glotmf.predict_rating(u, i, Rmax))
        truths.append(row.label)

    preds = np.array(preds)
    truths = np.array(truths)

    mae = np.mean(np.abs(preds - truths))
    rmse = np.sqrt(np.mean((preds - truths) ** 2))
    return mae, rmse



def safe_predict(model, u_id, i_id, u_map, i_map, Rmax=1.0):
    u = u_map.get(u_id, None)
    i = i_map.get(i_id, None)
    if u is None or i is None:
        return model.global_mean  # fallback to mean prediction
    return model.predict_rating(u, i, Rmax)

def generate_submission(model, test_data, user_id_to_idx, item_id_to_idx, path, Rmax=5, rating_min=1.0, rating_max=5.0):
    predictions = test_data.apply(
    lambda row: safe_predict(
        model,
        row.user_id,
        row.item_id,
        user_id_to_idx,
        item_id_to_idx,
        Rmax=1.0
    ),
    axis=1
)

    # Denormalize back to real scale [1, 5]
    predictions = predictions * (rating_max - rating_min) + rating_min

    submission_df = pd.DataFrame({
        'id': test_data['id'],
        'label': predictions.clip(lower=rating_min, upper=rating_max)
    })

    submission_df.to_csv(path, index=False)
    print(f"✅ Submission saved to '{path}'")




def main():
    prep = GlotMFPreprocessor("content/q3_dataset/train_data_movie_rate.csv", "content/q3_dataset/train_data_movie_trust.csv")
    ratings_df, _ = prep.load_data()
    reputation_weights = prep.compute_pagerank()
    trust_neighbors = prep.build_trust_neighbors()
    trust_neighbors = {u: vs for u, vs in trust_neighbors.items() if len(vs) >= 2}

    
    if os.path.exists(MODEL_CONFIG_PATH):
        with open(MODEL_CONFIG_PATH, 'r') as f:
            best_config = json.load(f)
    else:
        K_values = [10, 20]
        alpha_values = [0.4, 0.6, 0.8]
        lr_values = [0.005, 0.01]
        lambda_reg_values = [0.001, 0.01, 0.1]
        lambda_B_values = [0.01, 0.05, 0.1]

        best_mae = float('inf')
        best_config = None

        for K, alpha, lr, lambda_reg, lambda_B in itertools.product(K_values, alpha_values, lr_values, lambda_reg_values, lambda_B_values):
            print(f"\n🔍 Testing K={K}, alpha={alpha}, lr={lr}")
            model = GlotMF(num_users=prep.num_users, num_items=prep.num_items,
                            K=K, alpha=alpha, lr=lr,
                            lambda_reg=lambda_reg, lambda_B=lambda_B, lambda_E=0.1)

            model.train(ratings_df, reputation_weights,
                        prep.user_id_to_idx, prep.item_id_to_idx,
                        trust_neighbors, epochs=epoch_count)

            mae, rmse = evaluate_model(model, ratings_df, prep.user_id_to_idx, prep.item_id_to_idx)
            print(f"✅ MAE: {mae:.4f}, RMSE: {rmse:.4f}")

            if mae < best_mae:
                best_mae = mae
                best_config = {'K': K, 'alpha': alpha, 'lr': lr, 'lreg': lambda_reg, 'lb': lambda_B}
            
        with open(MODEL_CONFIG_PATH, 'w') as f:
            json.dump(best_config, f, indent=4)
   
        print(f"\n🏆 Best config: K={best_config['K']}, alpha={best_config['alpha']}, lr={best_config['lr']} — MAE={best_mae:.4f}")

    model = GlotMF(num_users=prep.num_users, num_items=prep.num_items,
                   K=best_config['K'], alpha=best_config['alpha'], lr=0.01,
                   lambda_reg=best_config['lreg'], lambda_B=best_config['lb'], lambda_E=0.1)
    
    
    # model.train(ratings_df, reputation_weights,
    #             prep.user_id_to_idx, prep.item_id_to_idx,
    #             trust_neighbors, epochs=epoch_count)

    # mae, rmse = evaluate_model(model, ratings_df, prep.user_id_to_idx, prep.item_id_to_idx)
    # print(f"Final MAE: {mae:.4f}, RMSE: {rmse:.4f}")
    

    train_df, val_df = train_test_split(ratings_df, test_size=0.2, random_state=42)

    model.train(train_df, reputation_weights,
                prep.user_id_to_idx, prep.item_id_to_idx,
                trust_neighbors, epochs=epoch_count)

    mae_train, rmse_train = evaluate_model(model, train_df, prep.user_id_to_idx, prep.item_id_to_idx)
    mae_val, rmse_val = evaluate_model(model, val_df, prep.user_id_to_idx, prep.item_id_to_idx)
    print(f"Train MAE: {mae_train:.4f}, RMSE: {rmse_train:.4f}")
    print(f"Validation MAE: {mae_val:.4f}, RMSE: {rmse_val:.4f}")

    test_data = pd.read_csv("content/q3_dataset/test_data.csv")
    generate_submission(model, test_data, prep.user_id_to_idx, prep.item_id_to_idx, SUBMISSION_PATH)


if __name__ == "__main__":
    main()