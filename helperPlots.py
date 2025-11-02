import matplotlib.pyplot as plt

def plot_two_curves(x, y1, y2, label1="Curve 1", label2="Curve 2", title="Two Curves Plot", xlabel = "X-axis", ylabel = "Y-axis"):
    if not (len(x) == len(y1) == len(y2)):
        print("x, y1, and y2 must have the same length")
        return
    
    plt.figure(figsize=(8, 5))
    plt.plot(x, y1, label=label1, linewidth=2)
    plt.plot(x, y2, label=label2, linewidth=2, linestyle='--')

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
