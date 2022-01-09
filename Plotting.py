import matplotlib.pyplot as plt

plt.ion()

class Plot():
    def __init__(self):
        self.figure, self.axis = plt.subplots(nrows=2, ncols=1, figsize=(10, 8), dpi=100)
        self.figure.suptitle('Loss and Points Plots')
        self.axis[1].set(xlabel='Episode', ylabel='Loss')
        self.axis[0].set(xlabel='Game', ylabel='Points')

    def plot_points(self, point_values):
        self.axis[0].plot(point_values, 'tab:orange')
        print("Game", len(point_values), '\n', "Points:", point_values[-1])

    def plot_loss(self, loss):
        self.axis[1].plot(loss, 'tab:orange')

    def plot_graphs(self):
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()