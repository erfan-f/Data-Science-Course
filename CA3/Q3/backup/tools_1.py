import numpy as np
import pandas as pd

def evaluate_model(glotmf, ratings_df, user_id_to_idx, item_id_to_idx, Rmax=5):
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


def generate_submission(model, test_data, user_id_to_idx, item_id_to_idx, path, Rmax=5):
    predictions = test_data.apply(
        lambda row: model.predict_rating(
            user_id_to_idx[row.user_id], 
            item_id_to_idx[row.item_id], 
            Rmax=Rmax
        ),
        axis=1
    )

    submission_df = pd.DataFrame({
        'id': test_data['id'],
        'label': predictions
    })

    submission_df['label'] = submission_df['label'].clip(lower=0, upper=Rmax)

    submission_df.to_csv(path, index=False)
    print(f"✅ Submission saved to '{path}'")


# def generate_submission(model, test_data, user_id_to_idx, item_id_to_idx, path, Rmax=5, rating_min=1.0, rating_max=5.0):
#     predictions = test_data.apply(
#         lambda row: model.predict_rating(
#             user_id_to_idx.get(row.user_id, -1), 
#             item_id_to_idx.get(row.item_id, -1), 
#             Rmax=1.0  # Still predicting in normalized [0,1]
#         ),
#         axis=1
#     )

#     # Denormalize back to real scale [1, 5]
#     predictions = predictions * (rating_max - rating_min) + rating_min

#     submission_df = pd.DataFrame({
#         'id': test_data['id'],
#         'label': predictions.clip(lower=rating_min, upper=rating_max)
#     })

#     submission_df.to_csv(path, index=False)
#     print(f"✅ Submission saved to '{path}'")
