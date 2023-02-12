import numpy  as np
import pandas as pd
from pymer4.models import Lmer # just import the linear mixed models class 
import scipy.stats as stats 
import pyreadr

print("Starting")
eval_data=pyreadr.read_r(r"C:\Users\Amega\OneDrive\Desktop\Git\bachelorproject_online\significance_analysis\22_12_07_Significance_Kreutzer\data_ter.rds")[None]
eval_data = eval_data.astype({"sentence_id" : 'category', "system" : 'category'})
print(eval_data)

#System-identifier: system
#Input-Identifier: sentence_id
#Two models, "different"-Model assumes significant difference between performance of groups, divided by system-identifier 
#Formula has form: "evaluation_metric ~ system_identifier + (1 | input_identifier)"
differentMeans_model = Lmer(formula = "ter ~ system + (1 | sentence_id)", data = eval_data)
#When using GLRT, set REML to false
#factors specifies names of system_identifier, i.e. Baseline, or Algorithm1
differentMeans_model.fit(factors = {"system" : list(eval_data["system"].unique())}, REML = False, summarize = False)


#"Common"-Model assumes no significant difference, which is why the system-identifier is not included
commonMean_model = Lmer(formula = "ter ~ (1 | sentence_id)", data = eval_data)
commonMean_model.fit(REML = False, summarize = False)

def GLRT(mod1, mod2):
    
    chi_square = 2 * abs(mod1.logLike - mod2.logLike)
    delta_params = abs(len(mod1.coefs) - len(mod2.coefs)) 
    
    return {"chi_square" : chi_square, "df": delta_params, "p" : 1 - stats.chi2.cdf(chi_square, df=delta_params)}

#Signficant p-value shows, that different-Model fits data sign. better, i.e.
#There is signficant difference in system-identifier
print(GLRT(differentMeans_model, commonMean_model))

#Post hoc divides the "different"-Model into its three systems
post_hoc_results = differentMeans_model.post_hoc(marginal_vars = ["system"])
#[0] shows group-means, i.e. performance of the single system-groups
print(post_hoc_results[0]) #cell (group) means
#[1] shows the pairwise comparisons, i.e. improvements over each other, with p-value
print(post_hoc_results[1]) #contrasts (group differences)

#Bin the data into shot, normal and long sentences
eval_data = eval_data.assign(src_length_class = lambda x: pd.cut(x.src_length, bins=[np.min(x.src_length), 15, 55, np.max(x.src_length)], labels=["short", "typical", "very long"], include_lowest=True))
#New model "expanded": Divides into system AND length-classes (Term system:src_length_class allows for Cartesian Product, i.e. different Mean for each system and sentence-length)
model_expanded = Lmer("ter ~ system + src_length_class  + system:src_length_class + (1 | sentence_id)", data = eval_data)
model_expanded.fit(factors = {"system" : ["Baseline", "Marking", "PostEdit"], "src_length_class" : ["short", "typical", "very long"]}, REML = False, summarize=False)
#Second model "nointeraction" lacks system:src-Term to hypothesise no interaction, i.e. no difference when changing sentence length
model_nointeraction = Lmer("ter ~ system + src_length_class + (1 | sentence_id)", data = eval_data)
model_nointeraction.fit(factors = {"system" : ["Baseline", "Marking", "PostEdit"], "src_length_class" : ["short", "typical", "very long"]}, REML = False, summarize=False)

#Its significant, so looking at different systems perform better at different sentence-lengths
print(GLRT(model_expanded, model_nointeraction)) # test interaction

post_hoc_results = model_expanded.post_hoc(marginal_vars = "system", grouping_vars = "src_length_class")
#Means of each combination
print(post_hoc_results[0])
#Comparisons for each combination
#Shows, that only for long sentences, the performance edge of PostEdit over Marking is significant
#For all lengths, the two corrections are significant over the baseline
#Contrary to the analysis without grouping by sentence lengths, which did not show sign. performance difference between the two correction-systems
print(post_hoc_results[1].query("src_length_class == 'short'"))
print(post_hoc_results[1].query("src_length_class == 'typical'"))
print(post_hoc_results[1].query("src_length_class == 'very long'"))
print("Done")