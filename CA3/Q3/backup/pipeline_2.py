import pandas as pd
from core import GlotMFPreprocessor, GlotMF
import numpy as np
import os
from tools import evaluate_model, compute_topk_similarity_matrix, generate_submission
from sklearn.model_selection import train_test_split
import pandas as pd
import os
from collections import defaultdict
import itertools
import json


RATING_DATA_PATH = 'content/q3_dataset/train_data_movie_rate.csv'
TRUST_DATA_PATH = 'content/q3_dataset/train_data_movie_trust.csv'
TEST_DATA_PATH = 'content/q3_dataset/test_data.csv'
S_B_PATH = 'content/q3_dataset/S_B.npy'
S_E_PATH = 'content/q3_dataset/S_E.npy'
MODEL_CONFIG_PATH = 'content/q3_dataset/config.json'
epoch_count = 10


def build_trust_neighbors(trust_df, user_id_to_idx):
    trust_neighbors = defaultdict(list)
    for row in trust_df.itertuples(index=False):
        u = user_id_to_idx[row.user_id_trustor]
        v = user_id_to_idx[row.user_id_trustee]
        trust_neighbors[u].append(v)
    return trust_neighbors

def main():
    prep = GlotMFPreprocessor("content/q3_dataset/train_data_movie_rate.csv", "content/q3_dataset/train_data_movie_trust.csv")
    ratings_df, trust_df = prep.load_data()
    T = prep.build_trust_matrix()
    reputation_weights = prep.compute_pagerank()
    trust_neighbors = build_trust_neighbors(trust_df, prep.user_id_to_idx)
    trust_neighbors = {u: vs for u, vs in trust_neighbors.items() if len(vs) >= 2}


    if os.path.exists("content/q3_dataset/S_B.npy") and os.path.exists("content/q3_dataset/S_E.npy"):
        print("🔁 Loading cached similarity matrices...")
        S_B = np.load("content/q3_dataset/S_B.npy")
        S_E = np.load("content/q3_dataset/S_E.npy")
    else:
        print("⚙️ Computing similarity matrices from scratch...")
        S_B = compute_topk_similarity_matrix(T, direction='out', top_k=20)
        S_E = compute_topk_similarity_matrix(T, direction='in', top_k=20)
        np.save("content/q3_dataset/S_B.npy", S_B)
        np.save("content/q3_dataset/S_E.npy", S_E)

    
    if os.path.exists(MODEL_CONFIG_PATH):
        with open(MODEL_CONFIG_PATH, 'r') as f:
            best_config = json.load(f)
    else:
        K_values = [10, 20]
        alpha_values = [0.4, 0.6, 0.8]
        lr_values = [0.005, 0.01]

        best_mae = float('inf')
        best_config = None

        for K, alpha, lr in itertools.product(K_values, alpha_values, lr_values):
            print(f"\n🔍 Testing K={K}, alpha={alpha}, lr={lr}")
            model = GlotMF(num_users=prep.num_users, num_items=prep.num_items,
                            K=K, alpha=alpha, lr=lr,
                            lambda_reg=0.001, lambda_B=0.1, lambda_E=0.1)

            model.train(ratings_df, reputation_weights,
                        prep.user_id_to_idx, prep.item_id_to_idx,
                        trust_neighbors, S_B, S_E, epochs=epoch_count)

            mae, rmse = evaluate_model(model, ratings_df, prep.user_id_to_idx, prep.item_id_to_idx)
            print(f"✅ MAE: {mae:.4f}, RMSE: {rmse:.4f}")

            if mae < best_mae:
                best_mae = mae
                best_config = {'K': K, 'alpha': alpha, 'lr': lr}
            
        with open(MODEL_CONFIG_PATH, 'w') as f:
            json.dump(best_config, f, indent=4)
   
        print(f"\n🏆 Best config: K={best_config['K']}, alpha={best_config['alpha']}, lr={best_config['lr']} — MAE={best_mae:.4f}")

    model = GlotMF(num_users=prep.num_users, num_items=prep.num_items,
                   K=best_config['K'], alpha=best_config['alpha'], lr=best_config['lr'],
                   lambda_reg=0.001, lambda_B=0.1, lambda_E=0.1)
    
    
    # model.train(ratings_df, reputation_weights,
    #             prep.user_id_to_idx, prep.item_id_to_idx,
    #             trust_neighbors, S_B, S_E, epochs=epoch_count)

    # mae, rmse = evaluate_model(model, ratings_df, prep.user_id_to_idx, prep.item_id_to_idx)
    # print(f"Final MAE: {mae:.4f}, RMSE: {rmse:.4f}")
    

    train_df, val_df = train_test_split(ratings_df, test_size=0.2, random_state=42)

    model.train(train_df, reputation_weights,
                prep.user_id_to_idx, prep.item_id_to_idx,
                trust_neighbors, S_B, S_E, epochs=epoch_count)

    mae_train, rmse_train = evaluate_model(model, train_df, prep.user_id_to_idx, prep.item_id_to_idx)
    mae_val, rmse_val = evaluate_model(model, val_df, prep.user_id_to_idx, prep.item_id_to_idx)
    print(f"Train MAE: {mae_train:.4f}, RMSE: {rmse_train:.4f}")
    print(f"Validation MAE: {mae_val:.4f}, RMSE: {rmse_val:.4f}")

    test_data = pd.read_csv("content/q3_dataset/test_data.csv")
    generate_submission(model, test_data, prep.user_id_to_idx, prep.item_id_to_idx)


if __name__ == "__main__":
    main()