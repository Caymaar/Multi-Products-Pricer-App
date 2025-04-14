class Node:
    def __init__(self, S):
        self.S = S                   # Spot price
        self.child_up = None         # Child up
        self.child_mid = None        # Child mid
        self.child_down = None       # Child down
        self.bro_up = None           # Brother up
        self.bro_down = None         # Brother down
        self.proba = None            # Proba of reaching the node from the root
        self.NFV = None              # Net Future Value of the Node, based on the discounted average of its children's NFVs
        self.parent = None           # Only trunc Nodes will have a parent (for backward pricing)
        self.barrier_triggered = False # for options

    def Triplet(self, mid, up, down):
        self.child_mid = mid
        self.child_up = up
        self.child_down = down
        mid.bro_up = self.child_up
        mid.bro_down = self.child_down

    # Adds two connexions compared to Triplet(). Used when a div is paid, due to recombination of the Tree
    def TripletDiv(self, mid, up, down):
        self.child_mid = mid
        self.child_up = up
        self.child_down = down
        mid.bro_up = self.child_up
        mid.bro_down = self.child_down
        up.bro_down = self.child_mid
        down.bro_up = self.child_mid

    # Function for pruning, linking only one child for the parent
    # We suppose we call it only when we reach the last Node of the parents' column when we use the function BuildColumn
    def Singleton(self, mid):
        self.child_mid = mid