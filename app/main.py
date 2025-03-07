from fastapi import FastAPI
import chess
from pydantic import BaseModel

from app.chess_env import ChessEnv
from app.q_learning import QLearningChess

app = FastAPI()


class ChessMoveRequest(BaseModel):
    move: str

env = ChessEnv()
ai = QLearningChess(env)


@app.post("/move")
async def make_move(request: ChessMoveRequest):
    move = request.move
    chess_move = chess.Move.from_uci(move)
    state, reward, done = env.step(chess_move)
    return {"board": state, "reward": reward, "game_over": done}

@app.get("/board")
async def get_board():
    state = env.reset()
    return {"board": state}
