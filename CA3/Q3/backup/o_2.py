import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

class DataLoader:
    def __init__(self):
        self.ratings_df = None
        self.trust_df = None
        
    def load_data(self, ratings_path, trust_path):
        """Load data from CSV files"""
        # Load ratings data - skip header row if present
        try:
            self.ratings_df = pd.read_csv(ratings_path, 
                                        names=['user_id', 'item_id', 'label'],
                                        header=0)
        except:
            self.ratings_df = pd.read_csv(ratings_path, 
                                        names=['user_id', 'item_id', 'label'])
        
        # Load trust data
        try:
            self.trust_df = pd.read_csv(trust_path, 
                                    names=['user_id_trustor', 'user_id_trustee', 'trust_value'],
                                    header=0)
        except:
            self.trust_df = pd.read_csv(trust_path, 
                                    names=['user_id_trustor', 'user_id_trustee', 'trust_value'])
        
        
        # Split ratings into train/test if no test set provided
        self.ratings_df, self.test_df = train_test_split(
            self.ratings_df, test_size=0.2, random_state=42)
        
        # Normalize ratings to [0,1] range
        self._normalize_ratings()
        
        return self.ratings_df, self.trust_df, self.test_df
    
    def _normalize_ratings(self):
        """Normalize ratings to [0,1] range using logistic function"""
        # First ensure the label column is numeric
        self.ratings_df['label'] = pd.to_numeric(self.ratings_df['label'], errors='coerce')
        
        # Drop any rows where label couldn't be converted to numeric
        self.ratings_df = self.ratings_df.dropna(subset=['label'])
        
        max_rating = self.ratings_df['label'].max()
        self.ratings_df['label'] = self.ratings_df['label'].apply(
            lambda x: 1 / (1 + np.exp(-x/max_rating)))
        
        if self.test_df is not None:
            # Similarly handle test data
            self.test_df['label'] = pd.to_numeric(self.test_df['label'], errors='coerce')
            self.test_df = self.test_df.dropna(subset=['label'])
            self.test_df['label'] = self.test_df['label'].apply(
                lambda x: 1 / (1 + np.exp(-x/max_rating)))
        
    def get_data_stats(self):
        """Print basic statistics about the datasets"""
        print("=== Dataset Statistics ===")
        print(f"Number of ratings: {len(self.ratings_df)}")
        print(f"Number of trust relationships: {len(self.trust_df)}")
        if self.test_df is not None:
            print(f"Number of test ratings: {len(self.test_df)}")
        
        print("\nRating distribution:")
        print(self.ratings_df['label'].describe())
        
        print("\nTrust value distribution:")
        print(self.trust_df['trust_value'].describe())


class GlotMF:
    def __init__(self, num_factors=10, alpha=0.6, lambda_reg=0.001, 
                 lambda_b=0.1, lambda_e=0.1, learning_rate=0.01, max_iter=100):
        """
        Initialize GlotMF model
        
        Parameters:
        - num_factors: Number of latent factors (K in paper)
        - alpha: Weight for trustor vs trustee contributions (0.6 as optimal in paper)
        - lambda_reg: Regularization parameter
        - lambda_b: Trustor implicit interaction weight
        - lambda_e: Trustee implicit interaction weight
        - learning_rate: Learning rate for SGD
        - max_iter: Maximum iterations for training
        """
        self.num_factors = num_factors
        self.alpha = alpha
        self.lambda_reg = lambda_reg
        self.lambda_b = lambda_b
        self.lambda_e = lambda_e
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        
        # Will be initialized during fit
        self.user_ids = None
        self.item_ids = None
        self.num_users = None
        self.num_items = None
        self.user_to_idx = None
        self.item_to_idx = None
        self.R = None
        self.T = None
        self.W = None
        self.S_b = None
        self.S_e = None
        self.B = None
        self.E = None
        self.V = None
        self.omega = None
        self.psi = None
        self.loss_history = []
        
    def logistic(self, x):
        """Logistic function to bound predictions between 0-1"""
        return 1.0 / (1.0 + np.exp(-x))
    
    def calculate_reputation(self, trust_df):
        """Calculate user reputation using PageRank"""
        print("Calculating user reputation with PageRank...")
        G = nx.DiGraph()
        
        # Add edges with trust values as weights
        for _, row in tqdm(trust_df.iterrows(), total=len(trust_df)):
            G.add_edge(row['user_id_trustor'], row['user_id_trustee'], 
                      weight=row['trust_value'])
        
        # Calculate PageRank
        pr = nx.pagerank(G, alpha=0.85)
        
        # Convert to reputation weights as in paper: w_i = 1/(1+log(r_i))
        max_rank = max(pr.values())
        reputation = {u: 1.0 / (1.0 + np.log(v/max_rank)) for u, v in pr.items()}
        
        return reputation
    
    def calculate_similarities(self, trust_matrix):
        """Calculate trustor and trustee similarities"""
        print("Calculating trustor and trustee similarities...")
        
        # Trustor similarity (out-links) - cosine similarity of outgoing trust
        S_b = cosine_similarity(trust_matrix)
        
        # Trustee similarity (in-links) - cosine similarity of incoming trust
        S_e = cosine_similarity(trust_matrix.T)
        
        return S_b, S_e
    
    def fit(self, ratings_df, trust_df, test_df=None, verbose=True):
        """
        Train the GlotMF model with validation loss tracking.
        
        Parameters:
        - ratings_df: DataFrame with columns ['user_id', 'item_id', 'label']
        - trust_df: DataFrame with columns ['user_id_trustor', 'user_id_trustee', 'trust_value']
        - test_df: DataFrame with test data (for validation purposes)
        - verbose: Whether to print training progress
        """
        # Preprocess data
        self._prepare_data(ratings_df, trust_df)
        
        # Initialize validation loss history
        validation_losses = []
        
        # Training loop
        print(f"Training GlotMF model for {self.max_iter} iterations...")
        for iteration in tqdm(range(self.max_iter)):
            self._run_iteration(iteration)
            
            # Calculate and store training loss
            if iteration % 10 == 0 or iteration == self.max_iter - 1:
                train_loss = self._calculate_loss()
                self.loss_history.append(train_loss)
                
                # If test_df is provided, calculate validation loss
                if test_df is not None:
                    mae, rmse = self.evaluate(test_df)
                    validation_losses.append(rmse)
                    if verbose and iteration % 10 == 0:
                        print(f"Iteration {iteration}, Train Loss: {train_loss:.4f}, RMSE: {rmse:.4f}")
                else:
                    if verbose and iteration % 10 == 0:
                        print(f"Iteration {iteration}, Train Loss: {train_loss:.4f}")
        
        # Plot loss curves for both training and validation
        if test_df is not None:
            plt.figure(figsize=(10, 6))
            plt.plot(np.arange(len(self.loss_history)) * 10, self.loss_history, label='Train Loss')
            plt.plot(np.arange(len(validation_losses)) * 10, validation_losses, label='Validation RMSE', linestyle='--')
            plt.title('Training vs Validation Loss')
            plt.xlabel('Iteration')
            plt.ylabel('Loss / RMSE')
            plt.legend()
            plt.grid(True)
            plt.show()
        
        # After training, verify with some sample predictions
        if verbose:
            print("\nSample predictions verification:")
            for _ in range(5):
                u = np.random.choice(self.num_users)
                v = np.random.choice(self.num_items)
                if self.R[u, v] > 0:  # Only show rated items for verification
                    pred = self.predict_rating(self.user_ids[u], self.item_ids[v])
                    print(f"User {self.user_ids[u]}, Item {self.item_ids[v]}: "
                        f"True={self.R[u, v]*5:.2f}, Pred={pred:.2f}")

        
    def _prepare_data(self, ratings_df, trust_df):
        """Prepare data matrices and mappings"""
        print("Preparing data matrices...")
        
        # Get unique users and items
        self.user_ids = np.unique(np.concatenate([
            ratings_df['user_id'].unique(),
            trust_df['user_id_trustor'].unique(),
            trust_df['user_id_trustee'].unique()
        ]))
        self.item_ids = ratings_df['item_id'].unique()
        self.num_users = len(self.user_ids)
        self.num_items = len(self.item_ids)
        
        # Create mappings from IDs to indices
        self.user_to_idx = {u: i for i, u in enumerate(self.user_ids)}
        self.item_to_idx = {v: i for i, v in enumerate(self.item_ids)}
        
        # Create rating matrix R
        self.R = np.zeros((self.num_users, self.num_items))
        for _, row in ratings_df.iterrows():
            u = self.user_to_idx[row['user_id']]
            v = self.item_to_idx[row['item_id']]
            self.R[u, v] = row['label']
        
        # Create trust matrix T
        self.T = np.zeros((self.num_users, self.num_users))
        for _, row in trust_df.iterrows():
            i = self.user_to_idx.get(row['user_id_trustor'], -1)
            k = self.user_to_idx.get(row['user_id_trustee'], -1)
            if i != -1 and k != -1:
                self.T[i, k] = row['trust_value']
        
        # Calculate reputation weights W
        self.reputation = self.calculate_reputation(trust_df)
        self.W = np.array([self.reputation.get(u, 0.5) for u in self.user_ids])  # Default 0.5 if no reputation
        
        # Calculate similarity matrices S_b (trustor) and S_e (trustee)
        self.S_b, self.S_e = self.calculate_similarities(self.T)
        
        # Initialize latent factors with small random values
        self.B = np.random.normal(scale=0.1, size=(self.num_users, self.num_factors))
        self.E = np.random.normal(scale=0.1, size=(self.num_users, self.num_factors))
        self.V = np.random.normal(scale=0.1, size=(self.num_items, self.num_factors))
        
        # Find observed ratings and trust relationships
        self.omega = np.where(self.R != 0)  # (user_indices, item_indices)
        self.psi = np.where(self.T != 0)    # (trustor_indices, trustee_indices)
    
    def _run_iteration(self, iteration):
        """Run one iteration of SGD updates"""
        # Shuffle training samples
        rating_indices = np.arange(len(self.omega[0]))
        np.random.shuffle(rating_indices)
        trust_indices = np.arange(len(self.psi[0]))
        np.random.shuffle(trust_indices)
        
        # Update ratings
        for idx in rating_indices:
            u = self.omega[0][idx]
            v = self.omega[1][idx]
            r_uv = self.R[u, v]
            w_u = self.W[u]
            
            # Compute prediction
            pred = self.alpha * np.dot(self.B[u], self.V[v]) + \
                   (1 - self.alpha) * np.dot(self.E[u], self.V[v])
            pred = self.logistic(pred)
            
            # Compute error
            error = w_u * (pred - r_uv) * pred * (1 - pred)
            
            # Update factors
            grad_B = error * self.V[v] + self.lambda_reg * self.B[u]
            grad_E = error * self.V[v] + self.lambda_reg * self.E[u]
            grad_V = error * (self.alpha * self.B[u] + (1 - self.alpha) * self.E[u]) + \
                     self.lambda_reg * self.V[v]
            
            self.B[u] -= self.learning_rate * grad_B
            self.E[u] -= self.learning_rate * grad_E
            self.V[v] -= self.learning_rate * grad_V
        
        # # Update trust relationships
        # for idx in trust_indices:
        #     i = self.psi[0][idx]
        #     k = self.psi[1][idx]
        #     t_ik = self.T[i, k]
            
        #     # Compute prediction
        #     pred_t = np.dot(self.B[i], self.E[k])
        #     pred_t = self.logistic(pred_t)
            
        #     # Compute error
        #     error_t = (pred_t - t_ik) * pred_t * (1 - pred_t)
            
        #     # Update factors
        #     grad_B_t = error_t * self.E[k] + self.lambda_reg * self.B[i]
        #     grad_E_t = error_t * self.B[i] + self.lambda_reg * self.E[k]
            
        #     self.B[i] -= self.learning_rate * grad_B_t
        #     self.E[k] -= self.learning_rate * grad_E_t
        
        # # Update for implicit trustor relationships
        # for i in range(self.num_users):
        #     for k in range(self.num_users):
        #         if self.S_b[i, k] > 0:
        #             grad_B_imp = self.lambda_b * self.S_b[i, k] * (self.B[i] - self.B[k])
        #             self.B[i] -= self.learning_rate * grad_B_imp
        
        # # Update for implicit trustee relationships
        # for i in range(self.num_users):
        #     for k in range(self.num_users):
        #         if self.S_e[i, k] > 0:
        #             grad_E_imp = self.lambda_e * self.S_e[i, k] * (self.E[i] - self.E[k])
        #             self.E[i] -= self.learning_rate * grad_E_imp
    
    def _calculate_loss(self):
        """Calculate total loss function"""
        loss = 0
        
        # Rating prediction loss
        for u, v in zip(*self.omega):
            pred = self.alpha * np.dot(self.B[u], self.V[v]) + \
                   (1 - self.alpha) * np.dot(self.E[u], self.V[v])
            pred = self.logistic(pred)
            loss += 0.5 * self.W[u] * (pred - self.R[u, v])**2
        
        # Trust prediction loss
        for i, k in zip(*self.psi):
            pred_t = np.dot(self.B[i], self.E[k])
            pred_t = self.logistic(pred_t)
            loss += 0.5 * (pred_t - self.T[i, k])**2
        
        # Implicit trustor relationships
        for i in range(self.num_users):
            for k in range(self.num_users):
                if self.S_b[i, k] > 0:
                    loss += 0.5 * self.lambda_b * self.S_b[i, k] * np.linalg.norm(self.B[i] - self.B[k])**2
        
        # Implicit trustee relationships
        for i in range(self.num_users):
            for k in range(self.num_users):
                if self.S_e[i, k] > 0:
                    loss += 0.5 * self.lambda_e * self.S_e[i, k] * np.linalg.norm(self.E[i] - self.E[k])**2
        
        # Regularization terms
        loss += 0.5 * self.lambda_reg * (np.linalg.norm(self.B, 'fro')**2 + 
                                        np.linalg.norm(self.E, 'fro')**2 + 
                                        np.linalg.norm(self.V, 'fro')**2)
        
        return loss
    
    def predict(self, user_id, item_id):
        """Predict rating for a user-item pair"""
        if user_id not in self.user_to_idx or item_id not in self.item_to_idx:
            return 0.5  # Default prediction for unknown user/item
        
        u = self.user_to_idx[user_id]
        v = self.item_to_idx[item_id]
        
        pred = self.alpha * np.dot(self.B[u], self.V[v]) + \
               (1 - self.alpha) * np.dot(self.E[u], self.V[v])
        return self.logistic(pred)
    
    
    def recommend(self, user_id, n=10):
        """Generate top-n recommendations for a user"""
        if user_id not in self.user_to_idx:
            return []
        
        u = self.user_to_idx[user_id]
        scores = []
        
        for v in range(self.num_items):
            if self.R[u, v] == 0:  # Only recommend unrated items
                pred = self.alpha * np.dot(self.B[u], self.V[v]) + \
                       (1 - self.alpha) * np.dot(self.E[u], self.V[v])
                scores.append((self.item_ids[v], self.logistic(pred)))
        
        # Sort by predicted score
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:n]
    
    def evaluate(self, test_df):
        """Evaluate model on test set with MAE and RMSE"""
        errors = []
        
        for _, row in test_df.iterrows():
            true_rating = row['label']
            pred_rating = self.predict(row['user_id'], row['item_id'])
            errors.append(true_rating - pred_rating)
        
        mae = np.mean(np.abs(errors))
        rmse = np.sqrt(np.mean(np.square(errors)))
        
        return mae, rmse
    
    def plot_loss_history(self):
        """Plot training loss history"""
        plt.figure(figsize=(10, 6))
        plt.plot(np.arange(len(self.loss_history)) * 10, self.loss_history)
        plt.title('Training Loss History')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.show()


def main():
    # Initialize data loader
    loader = DataLoader()
    
    # Load data - ensure these are your actual file paths
    ratings_df, trust_df, test_df = loader.load_data(
        'content/q3_dataset/train_data_movie_rate.csv',
        'content/q3_dataset/train_data_movie_trust.csv'
    )
    
    # Inspect data
    print("\nTraining data sample:")
    print(ratings_df.head())
    print("\nTrust data sample:")
    print(trust_df.head())
    
    # Initialize and train model
    model = GlotMF(
        num_factors=20,
        alpha=0.4,
        lambda_reg=0.01,
        lambda_b=0.1,
        lambda_e=0.1,
        learning_rate=0.005,
        max_iter=10
    )
    model.fit(ratings_df, trust_df)
    
    # Evaluate the model on the test set
    mae, rmse = model.evaluate(test_df)
    print(f"\nEvaluation Results:")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

    # ----------- Prediction and Submission for Kaggle ------------
    # Denormalization function (inverse of logistic normalization)
    def unnormalize_rating(r):
        return 1 + 4 * r  # rescales [0, 1] back to [1, 5]

    # Predict for test set
    submission = []
    test_data = pd.read_csv('content/q3_dataset/test_data.csv')
    for _, row in test_data.iterrows():
        user_id = int(row['user_id'])
        item_id = int(row['item_id'])
        row_id = int(row['id'])

        pred_norm = model.predict(user_id, item_id)  # model gives normalized rating
        pred_real = unnormalize_rating(pred_norm)    # back to [1,5]

        # Optional: clip to rating range to avoid overflow due to sigmoid inverse
        pred_real = min(5.0, max(1.0, pred_real))

        submission.append({
            'id': row_id,
            'label': pred_real
        })

    # Create DataFrame
    submission_df = pd.DataFrame(submission)

    # Save to CSV (no index)
    submission_df.to_csv('content/q3_dataset/submission.csv', index=False)
    print("Submission file saved as 'submission.csv'")


    

if __name__ == "__main__":
    main()