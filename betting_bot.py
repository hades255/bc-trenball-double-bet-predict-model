class BettingBot:
    def __init__(self, history, gcases=[], rcases=[], max_bet=16, last_count=40):
        if isinstance(history, str):
            self.history = list(history.strip())
        else:
            self.history = history

        self.gcases = gcases
        self.rcases = rcases

        self.max_bet = max_bet
        self.current_strategy = "D'Alembert"
        self.last_bet = 1
        self.rounds_since_switch = 0
        self.labouchere_sequence = [1, 2, 3, 4]
        self.fibonacci_history = []
        self.last_count = last_count

    def initialize(self):
        self.current_strategy = "D'Alembert"
        self.last_bet = 1
        self.rounds_since_switch = 0
        self.labouchere_sequence = [1, 2, 3, 4]
        self.fibonacci_history = []

    def predict_next_from_gcases(self, history):
        if isinstance(history, list):
            history = "".join(history)
        for pattern in self.gcases:
            if history.endswith(pattern):
                return "g"
        for pattern in self.rcases:
            if history.endswith(pattern):
                return "r"
        return ""

    def predict_next(self):
        if len(self.history) < 2:
            return "g"
        predicted_g = self.predict_next_from_gcases(self.history)
        if predicted_g:
            return predicted_g
        last = self.history[-1]
        g_after = sum(
            1
            for i in range(len(self.history) - 1)
            if self.history[i] == last and self.history[i + 1] == "g"
        )
        r_after = sum(
            1
            for i in range(len(self.history) - 1)
            if self.history[i] == last and self.history[i + 1] == "r"
        )
        return "g" if g_after > r_after else "r"

    def update_strategy(self):
        periods = {"D'Alembert": 15, "Fibonacci": 25, "Labouchere": 40}
        current_period = periods[self.current_strategy]
        if len(self.history) < current_period:
            return

        recent = self.history[-current_period:]
        red_ratio = recent.count("r") / len(recent)
        green_ratio = recent.count("g") / len(recent)

        new_strategy = self.current_strategy
        if red_ratio >= 0.75:
            new_strategy = "Fibonacci"
        elif green_ratio >= 0.70:
            new_strategy = "Labouchere"
        else:
            new_strategy = "D'Alembert"

        if new_strategy != self.current_strategy and self.rounds_since_switch >= 10:
            self.current_strategy = new_strategy
            self.rounds_since_switch = 0
        else:
            self.rounds_since_switch += 1

    def get_next_bet(self, last_result):
        if self.current_strategy == "D'Alembert":
            self.last_bet = (
                max(1, self.last_bet + 1)
                if last_result == "r"
                else max(1, self.last_bet - 1)
            )
        elif self.current_strategy == "Fibonacci":
            self.fibonacci_history.append(last_result)
            fib = [1, 1]
            while len(fib) <= len(self.fibonacci_history):
                fib.append(fib[-1] + fib[-2])
            self.last_bet = fib[len(self.fibonacci_history)]
        elif self.current_strategy == "Labouchere":
            if last_result == "g":
                if len(self.labouchere_sequence) >= 2:
                    self.labouchere_sequence = self.labouchere_sequence[1:-1]
                elif self.labouchere_sequence:
                    self.labouchere_sequence = []
            else:
                self.labouchere_sequence.append(self.last_bet)
            if len(self.labouchere_sequence) >= 2:
                self.last_bet = (
                    self.labouchere_sequence[0] + self.labouchere_sequence[-1]
                )
            elif self.labouchere_sequence:
                self.last_bet = self.labouchere_sequence[0]
            else:
                self.last_bet = 1

        self.last_bet = min(self.last_bet, self.max_bet)
        return self.last_bet

    def process_round(self, current_result):
        self.history.append(current_result)
        self.history = self.history[-self.last_count :]
        self.update_strategy()
        prediction = self.predict_next()
        next_bet = self.get_next_bet(current_result)
        return {
            "predicted_next": prediction,
            "strategy": self.current_strategy,
            "next_bet": next_bet,
        }
