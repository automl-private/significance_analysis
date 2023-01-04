import numpy  as np
import pandas as pd
from pymer4.models import Lmer
import scipy.stats as stats 

data=pd.read_pickle(r'C:\Users\Amega\OneDrive\Desktop\Git\bachelorproject_online\significance_analysis\22_12_23_ExperimentPlatform\concatData.pkl')
metric="mean"
system_id="algorithm"
input_id="benchmark"
bin_id="budget"
bin_labels=["short","mediums","mediuml","long"]
bin_dividers=[0.3,0.5,0.6,1]

def GLRT(mod1, mod2):
    chi_square = 2 * abs(mod1.logLike - mod2.logLike)
    delta_params = abs(len(mod1.coefs) - len(mod2.coefs)) 
    return {"chi_square" : chi_square, "df": delta_params, "p" : 1 - stats.chi2.cdf(chi_square, df=delta_params)}


def checkSignificance(data:pd.DataFrame,metric:str,system_id:str,input_id:str):

    #System-identifier: system_id
    #Input-Identifier: input_id
    #Two models, "different"-Model assumes significant difference between performance of groups, divided by system-identifier 
    #Formula has form: "metric ~ system_id + (1 | input_id)"
    differentMeans_model = Lmer(formula = metric+" ~ "+system_id+" + (1 | "+input_id+")", data = data)
    #factors specifies names of system_identifier, i.e. Baseline, or Algorithm1
    differentMeans_model.fit(factors = {system_id : list(data[system_id].unique())}, REML = False, summarize = False)

    #"Common"-Model assumes no significant difference, which is why the system-identifier is not included
    commonMean_model = Lmer(formula = metric+" ~ (1 | "+input_id+")", data = data)
    commonMean_model.fit(REML = False, summarize = False)

    #Signficant p-value shows, that different-Model fits data sign. better, i.e.
    #There is signficant difference in system-identifier
    print(GLRT(differentMeans_model, commonMean_model))

    #Post hoc divides the "different"-Model into its three systems
    post_hoc_results = differentMeans_model.post_hoc(marginal_vars = [system_id])
    #[0] shows group-means, i.e. performance of the single system-groups
    print(post_hoc_results[0]) #cell (group) means
    #[1] shows the pairwise comparisons, i.e. improvements over each other, with p-value
    print(post_hoc_results[1]) #contrasts (group differences)


def checkSignificanceBinned(data:pd.DataFrame,metric:str,system_id:str,input_id:str,bin_id:str,bin_labels:list[str],bin_dividers:list[float]):
    if not(0 in bin_dividers):
        bin_dividers.append(0)
    if not(1 in bin_dividers):
        bin_dividers.append(1)
    bin_dividers.sort()
    if len(bin_labels)!=(len(bin_dividers)-1):
        try:
            raise KeyboardInterrupt
        finally:
            print('Dividiers do not fit divider-labels')
    
    bins=[]
    for div in range(len(bin_dividers)):
        bins.append((np.min(data[bin_id])+int((np.max(data[bin_id])-np.min(data[bin_id]))*bin_dividers[div])))
    #Bin the data into classes according to bin_dividers
    data = data.assign(bin_class = lambda x: pd.cut(x[bin_id], bins=bins, labels=bin_labels, include_lowest=True))
    #New model "expanded": Divides into system AND bin-classes (Term system:bin_class allows for Cartesian Product, i.e. different Mean for each system and bin-class)
    model_expanded = Lmer(metric+" ~ "+system_id+" + bin_class + "+system_id+":bin_class + (1 | "+input_id+")", data = data)
    model_expanded.fit(factors = {system_id : list(data[system_id].unique()), "bin_class" : list(data["bin_class"].unique())}, REML = False, summarize=False)
    #Second model "nointeraction" lacks system:src-Term to hypothesise no interaction, i.e. no difference when changing bin-class
    model_nointeraction = Lmer(metric+" ~ "+system_id+" + bin_class + (1 | "+input_id+")", data = data)
    model_nointeraction.fit(factors = {system_id : list(data[system_id].unique()), "bin_class" : list(data["bin_class"].unique())}, REML = False, summarize=False)


    #If it's significant, look at if different systems perform better at different bin-classes
    print(GLRT(model_expanded, model_nointeraction)) # test interaction

    post_hoc_results = model_expanded.post_hoc(marginal_vars = system_id, grouping_vars = "bin_class")
    #Means of each combination
    print(post_hoc_results[0])
    #Comparisons for each combination
    for bin_class in range(len(bin_labels)):
        print(post_hoc_results[1].query("bin_class == '"+bin_labels[bin_class]+"'"))

checkSignificanceBinned(data,metric,system_id,input_id,bin_id,bin_labels,bin_dividers)