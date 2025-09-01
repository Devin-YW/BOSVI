class EarlyStopping:
    def __init__(self, tol=1e-4, patience=10):
        self.tol = tol
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, validation_loss):
        if self.best_score is None:
            self.best_score = validation_loss
        elif validation_loss > self.best_score - self.tol:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = validation_loss
            self.counter = 0