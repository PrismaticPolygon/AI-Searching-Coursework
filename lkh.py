from main import load_file
# An optimal tour is a minimum 1-tree where every node has degree 2.
# If a minimum 1-tree is a tour, then the tour is optimal.

filename = "AISearchtestcase.txt"
size, distance_matrix = load_file(filename)

# Hyperparameters
runs = 10  # The total number of runs
max_trials = size  # The maximum number of trials in each run
optimum = 0  # Known optimal tour length. A run will be terminated as soon as a tour length less than or equal to optimum is achieved. Default is DBL_MAX.
max_candidates = 5  # The maximum number of candidates to be associated with each node
ascent_candidates = 50  # The number of candidate edges to be associated with each node during the ascent
excess = 1 / size  # The maximum value allowed for any candidate edge is set to excess * the absolute value of the lower bound of a solution tour (determined by the ascent)
initial_period = size / 2 if size / 2 > 100 else 100  # The length of the first period in the ascent
initial_step_size = 1  # The initial step size used in the ascent

def ascent():

    # Determines a lower bound on the optimal tour length using subgradient optimisation
    # Also transforms the original problem into a problem in which ~values reflect the likelihood of edges being optimal

    return 0

def createCandidateSet():

    return {}

def generateCandidates(maxCandidates, maxAlpha):

    return {}

def findTour():

    return []

# Approximately 4000 lines of code. Motherfucking WHAT