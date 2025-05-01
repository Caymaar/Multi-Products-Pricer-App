import numpy as np
from pricers.node import Node
from pricers.pricing_model import Engine
from collections import defaultdict
from option.option import OptionPortfolio

class TreePortfolio:
    def __init__(self, market, option_ptf: OptionPortfolio, pricing_date, n_steps):
        """
        :param market: Instance du marché
        :param option_ptf: Instance de OptionPortfolio
        :param pricing_date: Date de valorisation
        :param n_steps: Nombre d'étapes de l'arbre
        """
        self.market = market
        self.options = option_ptf
        self.pricing_date = pricing_date
        self.n_steps = n_steps
        self.trees = {}  # Dictionnaire {(T, dt): (TreeModel,options)}
        self._alpha = np.array([])  # liste des alphas pour chaque arbre de chaque option
        self._build_trees()

    def _build_trees(self):
        """
        Construit un arbre par groupe d'options ayant même maturité et même dt.
        """
        option_groups = defaultdict(list)

        for option in self.options.assets:
            T = self.market.dcc.year_fraction(
                start_date=self.pricing_date, end_date=option.T
            )
            dt = T / self.n_steps
            key = (round(T, 6), round(dt, 6))

            option_groups[key].append(option)  # Dictionnaire {(T, dt): [options]}

        for key, options in option_groups.items():

            # Instancier un arbre pour ce groupe, avec la première option
            tree = TreeModel(
                market=self.market,
                option=OptionPortfolio([options[0]], [self.options.weights[0]]),  # Un dummy pour construire l'arbre
                pricing_date=self.pricing_date,
                n_steps=self.n_steps
            )
            # On stocke (l'arbre, les options associées)
            self.trees[key] = (tree, options)

            # Pour chaque option, associer le _alpha du tree à l'option
            for _ in options:
                self._alpha = np.append(self._alpha, tree.alpha)

    def recreate_model(self, **kwargs) -> "TreePortfolio":
        """
        Recrée une nouvelle instance du TreePortfolio en reprenant
        tous les paramètres actuels, sauf ceux passés en kwargs.
        """

        base_params = {
            "market": self.market.copy(),
            "option_ptf": OptionPortfolio(self.options.assets, self.options.weights),
            "pricing_date": self.pricing_date,
            "n_steps": self.n_steps,
        }

        # Surcharge avec ce que l'utilisateur fournit
        base_params.update(kwargs)

        # Création de la nouvelle instance
        return TreePortfolio(**base_params)


    def price(self, **kwargs):
        """
        Renvoi un vecteur ou float de prix de tous les groupes d'options.
        """
        prices = np.array([])
        for (tree, options) in self.trees.values():
            for option in options:
                idx = self.options.assets.index(option) # Retrouver l'indice de l'option et son poids
                weight = self.options.weights[idx]
                tree.tree_price = None # Set le prix de l'option a None pour le recalcul
                tree.options = OptionPortfolio([option], [weight])
                prices = np.append(prices,tree.price(**kwargs))
        return prices[-1] if len(prices)==1 else prices

    def aggregated_price(self):
        """
        Calcule la somme des prix de tous les groupes d'options.
        """
        total_price = 0
        for (tree, options) in self.trees.values():
            for option in options:
                tree.assets = option
                tree.tree_price = None  # Reset le prix pour recalculer
                total_price += tree.price()
        return total_price


class TreeModel(Engine):
    def __init__(self, market, option, pricing_date, n_steps, THRESHOLD=1e-7):
        super().__init__(market, option, pricing_date, n_steps=n_steps)  # Appel du constructeur parent

        # Ajustement pour un pricing d'un portefeuille d'option unique et non vectoriel par rapport à Monte Carlo
        self.dt = self.dt[-1]
        self.df = self.market.discount_factor
        self.t_div = self.t_div[-1] if self.t_div is not None else None
        self.times = np.array([i * self.dt for i in range(self.n_steps + 1)])

        self.THRESHOLD = THRESHOLD
        self.alpha = np.exp(self.market.sigma * np.sqrt(3 * self.dt))
        self.proba_down = (np.exp(self.market.sigma ** 2 * self.dt) - 1) \
                          / ((1 - self.alpha) * (self.alpha ** (-2) - 1))
        self.proba_up = self.proba_down / self.alpha
        self.proba_mid = 1 - self.proba_down - self.proba_up
        self.root = Node(self.market.S0)
        self.root.proba = 1
        self.tree_price = None
        self.build_tree()

    def is_div_date(self, step):
        return step < self.t_div <= (step + 1) if self.t_div is not None else False

    def forward(self, parent, step):
        t_j = float(self.times[step])
        t_jp1 = float(self.times[step + 1])
        # Discount factor entre t_j et t_{j+1}
        df_step = self.market.discount_factor(t_jp1) / self.market.discount_factor(t_j)
        S_next = parent.S / df_step
        if self.is_div_date(step) and self.market.div_type=="discrete":
            return S_next - self.market.dividend
        return S_next

    def get_proba(self, div_node, step):
        t_j = float(self.times[step])
        t_jp1 = float(self.times[step + 1])
        # fact de croissance = 1/df_step
        df_step = self.market.discount_factor(t_jp1) / self.market.discount_factor(t_j)

        average = self.forward(div_node, step)
        var = div_node.S ** 2 * (np.exp(self.market.sigma ** 2 * self.dt) - 1) * (1/df_step**2)

        proba_down = (div_node.child_mid.S ** (-2) * (var + average ** 2) - 1 - (self.alpha + 1) *
                      (div_node.child_mid.S ** (-1) * average - 1)) / ((1 - self.alpha) * (self.alpha ** (-2) - 1))
        proba_up = (div_node.child_mid.S ** (-1) * average - 1 - (self.alpha ** (-1) - 1) *
                    proba_down) / (self.alpha - 1)
        proba_mid = 1 - proba_down - proba_up
        return proba_up, proba_mid, proba_down

    def build_up(self, trunc_parent, trunc_mid, trunc_up, trunc_down):
        # Set parameters for the "while loop"
        mid = trunc_mid
        up = trunc_up
        down = trunc_down
        parent = trunc_parent

        # Create triplets while parents have brothers up
        while parent.bro_up is not None:
            parent = parent.bro_up
            # Test if parents is last upper parent of the column AND if its proba < THRESHOLD (for pruning)
            if (parent.bro_up is None) and (parent.proba < self.THRESHOLD):
                mid = up
                mid.proba += parent.proba * 1
                parent.Singleton(mid)
            # Else, no pruning
            else:
                down = mid
                mid = up
                up = Node(mid.S * self.alpha)
                down.proba += parent.proba * self.proba_down
                mid.proba += parent.proba * self.proba_mid
                up.proba = parent.proba * self.proba_up
                parent.Triplet(mid, up, down)

    def build_up_div(self, trunc_parent, trunc_mid, trunc_up, trunc_down, step):
        # Similar to BuildUp, used for Dividend dates. We suppose no pruning on this column
        mid = trunc_mid
        up = trunc_up
        down = trunc_down
        parent = trunc_parent

        while parent.bro_up is not None:
            parent = parent.bro_up
            if self.forward(parent, step) < (up.S + mid.S) / 2:
                pass

            elif self.forward(parent, step) > (up.S * self.alpha + up.S) / 2:
                down = up
                mid = Node(down.S * self.alpha)
                up = Node(mid.S * self.alpha)

            else:
                down = mid
                mid = up
                up = Node(mid.S * self.alpha)

            parent.TripletDiv(mid, up, down)
            proba_up, proba_mid, proba_down = self.get_proba(parent, step)
            try:
                up.proba += parent.proba * proba_up
            except:
                up.proba = parent.proba * proba_up
            try:
                mid.proba += parent.proba * proba_mid
            except:
                mid.proba = parent.proba * proba_mid
            down.proba += parent.proba * proba_down

    def build_down(self, trunc_parent, trunc_mid, trunc_up, trunc_down):
        # Reset parameters for the next "while loop"
        mid = trunc_mid
        up = trunc_up
        down = trunc_down
        parent = trunc_parent

        # Create triplets while parents have brothers down
        while parent.bro_down is not None:
            parent = parent.bro_down
            # Test if parents is last lower parent of the column AND if its proba < THRESHOLD (for pruning)
            if (parent.bro_down is None) and (parent.proba < self.THRESHOLD):
                mid = down
                mid.proba += parent.proba * 1
                parent.Singleton(mid)
            # Else, no pruning
            else:
                up = mid
                mid = down
                down = Node(mid.S / self.alpha)
                up.proba += parent.proba * self.proba_up
                mid.proba += parent.proba * self.proba_mid
                down.proba = parent.proba * self.proba_down
                parent.Triplet(mid, up, down)

    def build_down_div(self, trunc_parent, trunc_mid, trunc_up, trunc_down, step):
        # Similar to BuildDown, used for Dividend dates. We suppose no pruning on this column
        mid = trunc_mid
        up = trunc_up
        down = trunc_down
        parent = trunc_parent

        while parent.bro_down is not None:
            parent = parent.bro_down
            if self.forward(parent, step) > (mid.S + down.S) / 2:
                pass

            elif self.forward(parent, step) < (down.S / self.alpha + down.S) / 2:
                up = down
                mid = Node(up.S / self.alpha)
                down = Node(mid.S / self.alpha)

            else:
                up = mid
                mid = down
                down = Node(mid.S / self.alpha)

            parent.TripletDiv(mid, up, down)
            proba_up, proba_mid, proba_down = self.get_proba(parent, step)
            up.proba += parent.proba * proba_up
            try:
                mid.proba += parent.proba * proba_mid
            except:
                mid.proba = parent.proba * proba_mid
            try:
                down.proba += parent.proba * proba_down
            except:
                down.proba = parent.proba * proba_down

    def build_column(self, trunc_parent, step):
        trunc_mid = Node(self.forward(trunc_parent, step))
        trunc_mid.parent = trunc_parent
        trunc_up = Node(trunc_mid.S * self.alpha)
        trunc_down = Node(trunc_mid.S / self.alpha)
        trunc_mid.proba = trunc_parent.proba * self.proba_mid
        trunc_up.proba = trunc_parent.proba * self.proba_up
        trunc_down.proba = trunc_parent.proba * self.proba_down
        trunc_parent.Triplet(trunc_mid, trunc_up, trunc_down)

        if self.is_div_date(step):
            # Build the upper Nodes of the column
            self.build_up_div(trunc_parent, trunc_mid, trunc_up, trunc_down, step)
            # Build the upper Nodes of the column
            self.build_down_div(trunc_parent, trunc_mid, trunc_up, trunc_down, step)
        else:
            # Build the upper Nodes of the column
            self.build_up(trunc_parent, trunc_mid, trunc_up, trunc_down)
            # Build the upper Nodes of the column
            self.build_down(trunc_parent, trunc_mid, trunc_up, trunc_down)

    def build_tree(self):
        root = self.root
        root_up = Node(self.root.S * self.alpha)
        root_down = Node(self.root.S / self.alpha)
        self.root.bro_up = root_up
        self.root.bro_down = root_down
        root_up.proba = 1/6
        root_down.proba = 1/6
        trunc_parent = root
        for step in range(self.n_steps):
            self.build_column(trunc_parent, step)
            trunc_parent = trunc_parent.child_mid
        return root

    def get_trunc_node(self, step): # Get the t-th Node on the trunc
        if step > self.n_steps or step < 0:
            raise ValueError("0 <= step <= n_steps est obligatoire")
        trunc_node = self.root
        for _ in range(step):
            trunc_node = trunc_node.child_mid
        return trunc_node

    def get_node(self, step, height):
        trunc_node = self.get_trunc_node(step)
        current_node = trunc_node
        if height >= 0:
            for _ in range(height):
                current_node = current_node.bro_up
        else:
            for _ in range(-height):
                current_node = current_node.bro_down
        return current_node

    # For pricing backward (we suppose that children's NFV is already computed)
    def average_child_value_no_div(self, current_node):
        try:
            average = current_node.child_up.NFV * self.proba_up + current_node.child_mid.NFV * \
                      self.proba_mid + current_node.child_down.NFV * self.proba_down
        except:
            average = current_node.child_mid.NFV * 1

        return average

    def average_child_value_div(self, current_node, step):

        proba_up, proba_mid, proba_down = self.get_proba(current_node, step)

        average = current_node.child_up.NFV * proba_up + current_node.child_mid.NFV * \
                  proba_mid + current_node.child_down.NFV * proba_down
        return average

    def average_child_value(self, current_node, step):
        if self.is_div_date(step):
            average = self.average_child_value_div(current_node, step)
        else:
            average = self.average_child_value_no_div(current_node)

        # According to exec_type (European or American)
        if self.options.assets[-1].exercise == "european":
            return average
        elif self.options.assets[-1].exercise == "american":
            return max(average, self.options.assets[-1].intrinsic_value(current_node.S))
        else:
            raise ValueError("Execution type wrongly specified. Please only use 'European' or 'American'")

    def price(self, **kwargs): # Backward pricing

        global trunc_node

        if self.tree_price is None:
            trunc_node = self.get_trunc_node(self.n_steps)
            trunc_node.NFV = self.options.assets[-1].intrinsic_value(trunc_node.S)

            up_node, down_node = trunc_node, trunc_node
            while up_node.bro_up is not None:
                up_node = up_node.bro_up
                up_node.NFV = self.options.assets[-1].intrinsic_value(up_node.S)
            while down_node.bro_down is not None:
                down_node = down_node.bro_down
                down_node.NFV = self.options.assets[-1].intrinsic_value(down_node.S)

            step = self.n_steps
            while step > 0:
                j = step - 1
                t_j = j * self.dt
                t_jp1 = step * self.dt
                # discount factor entre t_{j+1} et t_j
                df_step = self.df(t_jp1) / self.df(t_j)

                # remonter au noeud parent (noeud central de la colonne j)
                trunc_node = trunc_node.parent
                # actualisation de ce noeud
                trunc_node.NFV = df_step * self.average_child_value(trunc_node, j)

                # idem pour tous les frères up et down
                up, down = trunc_node, trunc_node
                while up.bro_up:
                    up = up.bro_up
                    up.NFV = df_step * self.average_child_value(up, j)
                while down.bro_down:
                    down = down.bro_down
                    down.NFV = df_step * self.average_child_value(down, j)

                # on passe au pas suivant
                step = j

        if kwargs.get("up"):
            return self.root.bro_up.NFV
        elif kwargs.get("down"):
            return self.root.bro_down.NFV
        else:
            self.tree_price = self.root.NFV
            return self.root.NFV

    def gap(self):
        return (3 * self.market.S0 * (np.exp(self.market.sigma ** 2 * self.dt) - 1) * np.exp(
            2 * self.market.zero_rate(self.dt) * self.dt)) \
            / (8 * np.sqrt(2 * np.pi) * np.sqrt(np.exp(self.market.sigma ** 2 * self.T) - 1))

    def proba_check(self):
        L = []
        root = self.root
        L.append(root.proba)

        while root.child_mid is not None:
            root = root.child_mid
            trunc = root
            if root.proba < 0:
                raise ValueError("Proba négative")

            proba = root.proba
            while root.bro_up is not None:
                root = root.bro_up
                if root.proba < 0:
                    raise ValueError("Proba négative")
                proba += root.proba

            root = trunc
            while root.bro_down is not None:
                root = root.bro_down
                if root.proba < 0:
                    raise ValueError("Proba négative")
                proba += root.proba

            root = trunc
            L.append(round(proba, 12))

        print("Check OK")
        return L