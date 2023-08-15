import math 
from scipy import stats
nDCGs_VSM = [0.6177777777777778, 0.6978065687666908, 0.7409226109912592, 0.7648736399691464, 0.796610302736133, 0.8220513296721008, 0.8476639246694752, 0.8590923758159547, 0.8574319748291522, 0.8571627832056755]
nDCGs_LSA = [0.6622222222222223, 0.7545824743342774, 0.799841018018684, 0.8016578152601361, 0.8327005550372368, 0.8532139344843057, 0.8587783356435169, 0.8847203360673979, 0.8885370746890162, 0.896203921958]

mean_VSM = sum(nDCGs_VSM) / len(nDCGs_VSM)
mean_LSA = sum(nDCGs_LSA) / len(nDCGs_LSA)

print("Mean nDCG of VSM:", mean_VSM)
print("Mean nDCG of LSA:", mean_LSA)

def calc_std_dev(data, mean):
    n = len(data)
    squared_diffs = [(val - mean) ** 2 for val in data]
    variance = sum(squared_diffs) / (n - 1)
    std_dev = variance ** 0.5
    return std_dev

std_dev_VSM = calc_std_dev(nDCGs_VSM, mean_VSM)
std_dev_LSA = calc_std_dev(nDCGs_LSA, mean_LSA)

print("Standard deviation of nDCG for VSM:", std_dev_VSM)
print("Standard deviation of nDCG for LSA:", std_dev_LSA)

n1 = len(nDCGs_VSM)
n2 = len(nDCGs_LSA)

s1 = math.sqrt(sum([(x - mean_VSM) ** 2 for x in nDCGs_VSM]) / (n1 - 1))
s2 = math.sqrt(sum([(x - mean_LSA) ** 2 for x in nDCGs_LSA]) / (n2 - 1))

s_pool = math.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2))

t = (mean_LSA - mean_VSM) / (s_pool * math.sqrt(1/n1 + 1/n2))


print("t-value:", t)

df = n1 + n2 - 2
print("Degrees of freedom (df):", df)

p_value = stats.t.sf(abs(t), df) 
print("p-value:", p_value)