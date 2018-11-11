import numpy as np
import math
import lcztools
from lcztools import LeelaBoard
import chess
from collections import OrderedDict, namedtuple

UCTParams = namedtuple('UCTParams', 'c_puct c_fpu')

class UCTNode:
    def __init__(self, parent, move, prior, board=None):
        self.board = board
        self.parent = parent  # Optional[UCTNode]
        self.move = move  # uci move
        self.prior = prior  # float
        self.children = [] # List of UCTNodes
        self.total_value = 0  # float -- start as network 
        self.number_visits = 0  # int
        # visited_policy is for calculating FPU reduction. It's the sum of policy values
        # of visited children
        self.visited_policy = 0
        
    def get_board(self):
        if not self.board:
            self.board = self.parent.get_board().copy()
            self.board.push_uci(self.move)
        return self.board        
            
    def Q(self, child, params):  # returns float
        c_fpu = params.c_fpu
        if not child.number_visits:
            return self.total_value / self.number_visits - c_fpu * math.sqrt(self.visited_policy) # FPU reduction, parent value like lc0???
        else:
            return -child.total_value / child.number_visits
    
    def U(self, child, params):  # returns float
        c_puct = params.c_puct
        if self.number_visits == 1:
            return c_puct * child.prior / (1 + child.number_visits)
        else:
            return c_puct * math.sqrt(self.number_visits - 1) * child.prior / (1 + child.number_visits)
        
    def uct_select(self, avail_children, params):
        return max(avail_children, key=lambda node: self.Q(node, params) + self.U(node, params))

    def expand_and_backup(self, priors, value, params):
        """Given policy priors and a value, expand this node and backup values"""        
        for move, prior in priors.items():
            self.add_child(move, prior)
        self.total_value = value
        self.number_visits = 1
        child, current = self, self.parent
        turnfactor = -1
        while current:
            current.add_visit(child, value * turnfactor, params)
            child, current = current, current.parent
            turnfactor *= -1
            
    def select_leaf(self, params):
        """Recurse through tree to determine node to expand given cpuct
        
        In the case of root, ignore children that are not able to be max"""
        current = self
        while current and current.number_visits and current.children:
            current = current.child_to_visit(params)
            # None may be returned in case of early exit
            if not current:
                return None
        return current            
            
    def add_visit(self, from_child, value, params):
        """Add a visit to this node
        
        Default implementation for non-root-child nodes, which have to handle things
        differently for early exiting"""
        self.total_value += value
        self.number_visits += 1
        if from_child.number_visits == 1:
            self.visited_policy += from_child.prior          
    
    def child_to_visit(self, params):
        """Get child with highest Q + U given cpuct
        
        In the case of root node, ignore children that are not able to be max"""
        return self.uct_select(self.children, params)
    
    
    def add_child(self, move, prior):
        """Add a child to the tree
        
        UCTRootNode ==> Add UCTInteriorNode
        UCTInteriorNode ==> Add UCTInteriorNode"""
        self.children.append(UCTNode(self, move, prior))
    
    def debug_string(self, params):
        c_puct, c_fpu = params.c_puct, params.c_fpu
        move = self.move if self.move else 'root'
        n = self.number_visits
        p = self.prior*100
        q = self.parent.Q(self, params) if self.parent else self.Q(child=None, params=params)
        u = self.parent.U(self, params) if self.parent else 0
        q_u = q+u
        result = "{:<5} N = {:6d}       P = {:7.3f}%   Q = {:8.5f}   U = {:8.5f}   Q+U = {:8.5f}".format(
                    move, n, p, q, u, q_u
                    )
        return result
    
    def debug_children_string(self, params):
        
        children = sorted(self.children, key=lambda node: (node.number_visits, self.Q(node, params)+self.U(node, params)))
        return '\n'.join(child.debug_string(params) for child in children)
    
    def dump(self, params):
        print(self.debug_string(params))
        print('----------------------------------------------------------------------------------')
        print(self.debug_children_string(params))
    

class UCTRootNode(UCTNode):
    def __init__(self, board, max_nodes_to_search):
        super().__init__(parent=None, move=None, prior=1, board=board)
        # num_visits is the number of total visits this node will search
        self.max_nodes_to_search = max_nodes_to_search
        self.max_child_visits = 0
        # The "best child" is the child with the highest visit count,
        # - in case of a tie, select highest Q 
        self.best_child = None
        # Signal if we shouldn't search anymore because there is no way to beat the best
        # child
        self.early_stop = False
        
    def Q(self, child, params):
        # If child is none, just get the average value at root
        if child is None:
            return self.total_value / self.number_visits
        else:
            return super().Q(child, params)
   
            
    def child_to_visit(self, params):
        """Get child with best Q + U given cpuct
        
        In the case of root node, ignore children that are not able to be max"""
        if self.early_stop:
            return None
        
        remaining_visits = self.max_nodes_to_search - self.number_visits
        avail_children = [node for node in self.children if
                            node.number_visits + remaining_visits >= self.max_child_visits]
        
        # If there's only one child to search, and it's not the best child, then exit
        # Note: It would be fine to return None regardless of whether the child is best
        #       so an NN eval isn't triggered but this is LC0's behavior
        if len(avail_children)==1:
            child_to_visit = avail_children[0]
            
            # Early stop will only happen next time we get to this function
            self.early_stop = True
            
            if self.best_child is None:
                self.best_child = child_to_visit
            elif not child_to_visit == self.best_child:
                return None
        else:
            child_to_visit = self.uct_select(avail_children, params)
        return child_to_visit
        
    def add_visit(self, from_child, value, params):
        """Add a visit to this node"""
        super().add_visit(from_child, value, params)
        
        if from_child.number_visits > self.max_child_visits:
            self.max_child_visits = from_child.number_visits
            self.best_child = from_child
        elif from_child.number_visits == self.max_child_visits:
            if self.Q(from_child, params) > self.Q(self.best_child, params):
                self.best_child = from_child      




def UCT_search(board, num_reads, net, params):
    if not isinstance(params, UCTParams):
        params = UCTParams(*params)
    root = UCTRootNode(board, max_nodes_to_search = num_reads)
    for _ in range(num_reads):
        leaf = root.select_leaf(params)
        if not leaf:
            # Early exit when root does not search best child
            break
        child_priors, value_estimate = net.evaluate(leaf.get_board())
        leaf.expand_and_backup(child_priors, value_estimate, params)
    
    return root.Q(child=None, params=params), root.best_child

