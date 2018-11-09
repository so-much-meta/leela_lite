#!/usr/bin/python3
from lcztools import load_network, LeelaBoard
import search
import chess
import chess.pgn
import sys
import time


if len(sys.argv) != 3:
    print("Usage: python3 leela_lite.py <weights file or network server ID> <nodes>")
    print(len(sys.argv))
    exit(1)
else:
    weights = sys.argv[1]
    nodes = int(sys.argv[2])

network_id = None
try:
    # If the parameter is an integer, assume it's a network server ID
    network_id = int(weights)
    weights = None
except:
    pass

def load_leela_network():
    global net, nn
    if network_id is not None:
        net = load_network(backend='net_client', network_id=network_id, policy_softmax_temp=2.2)
    else:
        net = load_network(backend='pytorch_cuda', filename=weights, policy_softmax_temp=2.2)
    nn = search.NeuralNet(net=net, lru_size=min(5000, nodes))
    

load_leela_network()

SELFPLAY = True

board = LeelaBoard()
while True:
    if not SELFPLAY:
        print(board.unicode())
        print("Enter move: ", end='')
        sys.stdout.flush()
        line = sys.stdin.readline()
        line = line.rstrip()
        board.push_uci(line)
    print(board.unicode())
    print("thinking...")
    start = time.time()
    root_q, node = search.UCT_search(board, nodes, net=nn,
                                     params=search.uct.UCTParams(c_puct=3.4, c_fpu=1.2))
    elapsed = time.time() - start
    print("best: ", node.move)
    print("Time: {:.3f} nps".format(nodes/elapsed))
    print(nn.evaluate.cache_info())
    board.push_uci(node.move)
    if board.pc_board.is_game_over() or board.is_draw():
        result = board.pc_board.result(claim_draw=True)
        print("Game over... result is {}".format(result))
        print(board.unicode())
        print()
        pgn_game = chess.pgn.Game.from_board(board.pc_board) 
        pgn_game.headers['Result'] = result
        print(pgn_game)
        break
    
