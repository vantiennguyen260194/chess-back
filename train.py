from app.chess_env import ChessEnv
from app.q_learning import QLearningChess

def train_model():
    # Initialize the environment and AI
    env = ChessEnv()
    ai = QLearningChess(env, model_path="models/q_table.pkl")
    
    # Train the model for a specified number of episodes
    ai.train(episodes=1000)
    print("Training completed and model saved.")

if __name__ == "__main__":
    train_model()
