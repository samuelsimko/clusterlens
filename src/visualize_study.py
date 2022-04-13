import pickle
from optuna.visualization import plot_contour
from optuna.visualization import plot_edf
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_parallel_coordinate
from optuna.visualization import plot_param_importances
from optuna.visualization import plot_slice


study = pickle.load(open("study_dump", "rb"))

plt = plot_optimization_history(study)
plt.show()
plt = plot_parallel_coordinate(study)
plt.show()
plt = plot_contour(study)
plt.show()
plt = plot_param_importances(study)
plt.show()
plt = plot_slice(study)
plt.show()
plt = plot_edf(study)
plt.show()
