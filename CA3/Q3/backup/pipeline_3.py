import pandas as pd
import os
import itertools
import json
from sklearn.model_selection import train_test_split

from core import GlotMFPreprocessor, GlotMF
from tools import evaluate_model, generate_submission

RATING_DATA_PATH = 'content/q3_dataset/train_data_movie_rate.csv'
TRUST_DATA_PATH = 'content/q3_dataset/train_data_movie_trust.csv'
TEST_DATA_PATH = 'content/q3_dataset/test_data.csv'
MODEL_CONFIG_PATH = 'content/q3_dataset/config.json'
SUBMISSION_PATH = "content/q3_dataset/submission.csv"
epoch_count = 10


def main():
    prep = GlotMFPreprocessor("content/q3_dataset/train_data_movie_rate.csv", "content/q3_dataset/train_data_movie_trust.csv")
    ratings_df, _ = prep.load_data()
    rating_min = prep.rating_min
    rating_max = prep.rating_max
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

        best_mae = float('inf')
        best_config = None

        for K, alpha, lr in itertools.product(K_values, alpha_values, lr_values):
            print(f"\n🔍 Testing K={K}, alpha={alpha}, lr={lr}")
            model = GlotMF(num_users=prep.num_users, num_items=prep.num_items,
                            K=K, alpha=alpha, lr=lr,
                            lambda_reg=0.001, lambda_B=0.1, lambda_E=0.1)

            model.train(ratings_df, reputation_weights,
                        prep.user_id_to_idx, prep.item_id_to_idx,
                        trust_neighbors, epochs=10)

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


    model.train(ratings_df, reputation_weights,
                prep.user_id_to_idx, prep.item_id_to_idx,
                trust_neighbors, epochs=epoch_count)

    mae, rmse = evaluate_model(model, ratings_df, prep.user_id_to_idx, prep.item_id_to_idx)
    print(f"Final MAE: {mae:.4f}, RMSE: {rmse:.4f}")

    # train_df, val_df = train_test_split(ratings_df, test_size=0.2, random_state=42)

    # model.train(train_df, reputation_weights,
    #         prep.user_id_to_idx, prep.item_id_to_idx,
    #         trust_neighbors, epochs=epoch_count)

    # mae_train, rmse_train = evaluate_model(model, train_df, prep.user_id_to_idx, prep.item_id_to_idx, rating_min=rating_min, rating_max=rating_max)
    # mae_val, rmse_val = evaluate_model(model, val_df, prep.user_id_to_idx, prep.item_id_to_idx, rating_min=rating_min, rating_max=rating_max)

    # print(f"Train MAE: {mae_train:.4f}, RMSE: {rmse_train:.4f}")
    # print(f"Validation MAE: {mae_val:.4f}, RMSE: {rmse_val:.4f}")

    test_data = pd.read_csv("content/q3_dataset/test_data.csv")
    print("Sample predictions (normalized):", model.predict_rating(0, 0, Rmax=1.0))

    generate_submission(model, test_data, prep.user_id_to_idx, prep.item_id_to_idx, SUBMISSION_PATH, Rmax=1.0, rating_min=rating_min, rating_max=rating_max)



if __name__ == "__main__":
    main()