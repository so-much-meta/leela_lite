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




# net = load_network(backend='pytorch_cuda', filename=weights, policy_softmax_temp=2.2)
# net = load_network(backend='pytorch_cuda', filename=weights, policy_softmax_temp=1.0)
network_id = None
try:
    network_id = int(weights)
except:
    pass
if network_id is not None:
    net = load_network(backend='net_client', network_id=network_id)
else:
    net = load_network(backend='pytorch_cuda', filename=weights)
nn = uct.NeuralNet(net=net)
#policy, value = net.evaluate(board)
#print(policy)
#print(value)
#print(uct.softmax(policy.values()))

SELFPLAY = True

import numpy as np

with open('../lczero_tools/scripts/opening/data/bookfish_opening_seqs_8.txt') as f:
    probs, moves = zip(*(line.split(None, 1) for line in f.readlines()))
    probs = np.array([float(p) for p in probs])
    probs = probs/sum(probs)
    moves = [m.split() for m in moves]
    items = list(zip(moves, probs))
indices = np.random.choice(len(probs), 100, replace=False, p=probs)
sampled_items = [items[idx] for idx in indices]
sampled_moves, sampled_probs = zip(*sampled_items)

reg_score = ''
alt_score = ''

with open('results.txt', 'w') as f:
    for seq_idx, seq in enumerate(sampled_moves, 1):
        for alt_color in (chess.WHITE, chess.BLACK):
            board = LeelaBoard()
            for uci in seq:
                print(board)
                print("Forced move: {}".format(uci))
                board.push_uci(uci)            
            
            while True:
                if not SELFPLAY:
                    print(board)
                    print("Enter move: ", end='')
                    sys.stdout.flush()
                    line = sys.stdin.readline()
                    line = line.rstrip()
                    board.push_uci(line)
                print(board)
                print("thinking...")
                start = time.time()
                if alt_color == board.turn:
                    print("ALT...")
                    best, node = uct.UCT_search(board, nodes, net=nn, C=3.4, alt=True)
                else:
                    print("REG...")
                    best, node = uct.UCT_search(board, nodes, net=nn, C=3.4, alt=False)
                elapsed = time.time() - start
                print("best: ", best)
                print("Time: {:.3f} nps".format(nodes/elapsed))
                print(nn.evaluate.cache_info())
                board.push_uci(best)
                if board.pc_board.is_game_over() or board.is_draw():
                    result = board.pc_board.result(claim_draw=True)
                    print("Game over... result is {}".format(result))
                    print(board)
                    pgn_game = chess.pgn.Game.from_board(board.pc_board)
                    pgn_game.headers['Result'] = result
                    pgn_game.headers['White'] = 'ALT' if alt_color == chess.WHITE else 'REG'
                    pgn_game.headers['Black'] = 'ALT' if alt_color == chess.BLACK else 'REG' 
                    print(pgn_game)
                    print(pgn_game, file=f)
                    print(file=f)
                    if result == '1-0':
                        winner = chess.WHITE
                    elif result == '0-1':
                        winner = chess.BLACK
                    else:
                        winner = None
                    if winner is None:
                        alt_score += '='
                        reg_score += '='
                    elif winner == alt_color:
                        alt_score += '+'
                        reg_score += '-'
                    else:
                        alt_score += '-'
                        reg_score += '+'
                    print('ALT SCORE: {}'.format(alt_score), file=f)
                    print('REG SCORE: {}'.format(reg_score), file=f)
                    print(file=f)
                    print(file=f)
                    f.flush()
                    break
