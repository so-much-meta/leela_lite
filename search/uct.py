import numpy as np
import math
import lcztools
from lcztools import LeelaBoard
import chess
from collections import OrderedDict

class UCTNode():
    def __init__(self, board=None, parent=None, move=None, prior=0, is_root=False, root_remaining_playouts=None):
        self.board = board
        self.move = move
        self.is_expanded = False
        self.parent = parent  # Optional[UCTNode]
        self.children = OrderedDict()  # Dict[move, UCTNode]
        self.prior = prior  # float
        self.total_value = 0  # float
        self.number_visits = 0  # int
        self.is_root = is_root
        self.root_remaining_playouts = root_remaining_playouts
        self.root_max_n = 0  # Highest visits among children
        self.root_best_child = None
        
    def Q(self):  # returns float
        if not self.number_visits:
            return -self.parent.total_value / self.parent.number_visits # - 1.2*math.sqrt(self.prior) # FPU reduction, parent value like lc0???
        else:
            return self.total_value / self.number_visits
    
    def U(self):  # returns float
        if self.parent.number_visits == 1:
            return self.prior / (1 + self.number_visits)
        else:
            return math.sqrt(self.parent.number_visits - 1) * self.prior / (1 + self.number_visits)
        
    def best_child(self, C):
        if self.is_root:
            avail_children = [node for node in self.children.values() if self.root_remaining_playouts>=self.root_max_n - node.number_visits]
            # If there's only one child to search, signal an early exit
            if len(avail_children)==1:
                # We don't really need this, but it matches what lc0 does... still does one extra node expansion for best node
                return avail_children[0], True
        else:
            avail_children = self.children.values()
        return max(avail_children,
                   key=lambda node: node.Q() + C*node.U()), False

    def select_leaf(self, C):
        current = self
        early_exit = False
        while current.is_expanded and current.children:
            # This is way more cumbersome than it neesd to be...
            # The idea is only root can trigger an early exit, but even if there's
            # an early exit, we still expand the best node if that's the one that's chosen
            # Probably can simplify, but this is LC0's behavior
            if current.is_root:
                current, early_exit = current.best_child(C)
                if early_exit:
                    # we don't really need to check this, but it's what LC0 does
                    if not current==self.root_best_child:
                        return None, True                
            else:
                current, _ = current.best_child(C)
        if not current.board:
            current.board = current.parent.board.copy()
            current.board.push_uci(current.move)
        return current, early_exit

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
            if not current.parent:
                break
            if current.parent.is_root:
                root = current.parent
                
                if root.root_best_child is None:
                    root.root_best_child = current
                    root.root_max_n = current.number_visits
                else:
                    if root.root_max_n == current.number_visits:
                        if root.root_best_child != current:
                            if root.root_best_child.Q() < current.Q():
                                root.root_best_child = current
                    elif root.root_max_n < current.number_visits:
                        root.root_max_n = current.number_visits
                        root.root_best_child = current
                    
            current = current.parent
            turnfactor *= -1
        # We're at root, so reduce remianing playouts
        current.root_remaining_playouts -= 1
        # current.number_visits += 1

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
    root = UCTNode(board, is_root=True, root_remaining_playouts=num_reads)
    for _ in range(num_reads):
        leaf, early_exit = root.select_leaf(C)
        if not leaf:
            # Early exit when root does not search best child
            break
        child_priors, value_estimate = net.evaluate(leaf.board)
        leaf.expand(child_priors)
        leaf.backup(value_estimate)
        if early_exit:
            # Early exit when root does search best child
            break
        

    #for m, node in sorted(root.children.items(),
    #                      key=lambda item: (item[1].number_visits, item[1].Q())):
    #    node.dump(m, C)
    return -root.Q(), max(root.children.items(),
               key=lambda item: (item[1].number_visits, item[1].Q()))

