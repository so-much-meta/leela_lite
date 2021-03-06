import numpy as np
import math
import lcztools
from lcztools import LeelaBoard
import chess
from collections import OrderedDict
import functools

def apply_temp(policy, temp):
    """Apply a temperature to an already softmaxed policy"""
    values = np.fromiter(policy.values(), dtype=np.float)
    values = values**(1/temp)
    values = values/values.sum()
    for i, k in list(enumerate(policy)):
        policy[k] = values[i]

class UCTNode():
    def __init__(self, board=None, parent=None, move=None, prior=0, is_root=False, alt=False):
        if parent:
            self.alt = parent.alt
        else:
            self.alt = alt
        self.is_root = is_root
        self.board = board
        self.move = move
        self.is_expanded = False
        self.parent = parent  # Optional[UCTNode]
        self.children = OrderedDict()  # Dict[move, UCTNode]
        self.prior = prior  # float
        self.total_value = 0  # float
        self.number_visits = 0  # int
        
    def Q(self):  # returns float
        if not self.number_visits:
            return 0 # FPU reduction, parent value like lc0???
        else:
            return self.total_value / self.number_visits

    def U(self):  # returns float
        return (math.sqrt(self.parent.number_visits)
                * self.prior / (1 + self.number_visits))
#         if self.parent.is_root and self.alt:
#             scaled_parent_visits = (self.parent.number_visits)**0.75 
#             return (scaled_parent_visits * self.prior / (1 + self.number_visits))            
#         else:
#             return (math.sqrt(self.parent.number_visits)
#                     * self.prior / (1 + self.number_visits))

    def best_child(self, C):
        return max(self.children.values(),
                   key=lambda node: node.Q() + C*node.U())

    def select_leaf(self, C):
        current = self
        while current.is_expanded and current.children:
            current = current.best_child(C)
        if not current.board:
            current.board = current.parent.board.copy()
            current.board.push_uci(current.move)
        return current

    def expand(self, child_priors):
        self.is_expanded = True
        for move, prior in child_priors.items():
            self.add_child(move, prior)

    def add_child(self, move, prior):
        self.children[move] = UCTNode(parent=self, move=move, prior=prior)
    
    def backup(self, value_estimate: float):
        current = self
        # Child nodes are multiplied by -1 because we want max(-opponent eval)
        turnfactor = -1
        while True:            
            current.number_visits += 1
            current.total_value += (value_estimate *
                                    turnfactor)
            if current.parent:
                current = current.parent
                turnfactor *= -1
            else:
                break
        current.number_visits += 1

    def dump(self, move, C):
        print("---")
        print("move: ", move)
        print("total value: ", self.total_value)
        print("visits: ", self.number_visits)
        print("prior: ", self.prior)
        print("Q: ", self.Q())
        print("U: ", self.U())
        print("BestMove: ", self.Q() + C * self.U())
        #print("math.sqrt({}) * {} / (1 + {}))".format(self.parent.number_visits,
        #      self.prior, self.number_visits))
        print("---")

def UCT_search(board, num_reads, net=None, C=1.0, alt=False):
    assert(net != None)
    root = UCTNode(board, prior=1, is_root=True, alt=alt)
    for _ in range(num_reads):
        leaf = root.select_leaf(C)
        child_priors, value_estimate = net.evaluate(leaf.board)
        if alt and child_priors:
            apply_temp(child_priors, 2.2)
        leaf.expand(child_priors)
        leaf.backup(value_estimate)

    #for m, node in sorted(root.children.items(),
    #                      key=lambda item: (item[1].number_visits, item[1].Q())):
    #    node.dump(m, C)
    # Need to make this negative as we're actually trying to maximize children Q values
    print("Root Q = {}".format(-root.Q()))
    return max(root.children.items(),
               key=lambda item: (item[1].number_visits, item[1].Q()))


class NeuralNet:

    def __init__(self, net=None, lru_size=5000):
        super().__init__()
        assert(net != None)
        self.net = net
        self.evaluate = functools.lru_cache(maxsize=lru_size)(self.evaluate)

    def evaluate(self, board):
        result = None
        
        if board.pc_board.is_game_over():
            result = board.pc_board.result()
        elif board.is_draw():
            # board.is_draw checks for threefold or fifty move rule
            # Don't use python-chess method, because threefold checks if next move can
            # be threefold as well            
            result = "1/2-1/2"
        
            
        if result:
            if result == "1/2-1/2":
                return dict(), 0.0
            else:
                # Always return -1.0 when checkmated
                return dict(), -1.0
            
        policy, value = self.net.evaluate(board)
        
        value2 = (2.0*value)-1.0
        #print("value: ", value)
        #print("value2: ", value2)
        #sm = temp_softmax(policy.values(), sm=2.2)
        #for i, k in enumerate(policy):
        #    policy[k] = sm[i]
        return policy, value2



#num_reads = 10000
#import time
#tick = time.time()
#UCT_search(GameState(), num_reads)
#tock = time.time()
#print("Took %s sec to run %s times" % (tock - tick, num_reads))
#import resource
#print("Consumed %sB memory" % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
