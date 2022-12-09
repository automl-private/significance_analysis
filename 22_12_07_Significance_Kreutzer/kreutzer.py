import numpy  as np
import pandas as pd
from pymer4.models import Lmer # just import the linear mixed models class 
import scipy.stats as stats 
import pyreadr

print("Starting")
eval_data=pyreadr.read_r(r"C:\Users\Amega\OneDrive\Desktop\Git\bachelorproject\22_12_08_NLP\data_ter.rds")[None]
eval_data = eval_data.astype({"sentence_id" : 'category', "system" : 'category'})
print(eval_data)

differentMeans_model = Lmer(formula = "ter ~ system + (1 | sentence_id)", data = eval_data)
differentMeans_model.fit(factors = {"system" : ["Baseline", "Marking", "PostEdit"]}, REML = False, summarize = False)
commonMean_model = Lmer(formula = "ter ~ (1 | sentence_id)", data = eval_data)
commonMean_model.fit(REML = False, summarize = False)

def GLRT(mod1, mod2):
    
    chi_square = 2 * abs(mod1.logLike - mod2.logLike)
    delta_params = abs(len(mod1.coefs) - len(mod2.coefs)) 
    
    return {"chi_square" : chi_square, "df": delta_params, "p" : 1 - stats.chi2.cdf(chi_square, df=delta_params)}

print(GLRT(differentMeans_model, commonMean_model))

post_hoc_results = differentMeans_model.post_hoc(marginal_vars = ["system"])
print(post_hoc_results[0]) #cell (group) means
print(post_hoc_results[1]) #contrasts (group differences)

eval_data = eval_data.assign(src_length_class = lambda x: pd.cut(x.src_length, bins=[np.min(x.src_length), 15, 55, np.max(x.src_length)], labels=["short", "typical", "very long"], include_lowest=True))
model_expanded = Lmer("ter ~ system + src_length_class  + system:src_length_class + (1 | sentence_id)", data = eval_data)
model_expanded.fit(factors = {"system" : ["Baseline", "Marking", "PostEdit"], "src_length_class" : ["short", "typical", "very long"]}, REML = False, summarize=False)
model_nointeraction = Lmer("ter ~ system + src_length_class + (1 | sentence_id)", data = eval_data)
model_nointeraction.fit(factors = {"system" : ["Baseline", "Marking", "PostEdit"], "src_length_class" : ["short", "typical", "very long"]}, REML = False, summarize=False)

print(GLRT(model_expanded, model_nointeraction)) # test interaction
post_hoc_results = model_expanded.post_hoc(marginal_vars = "system", grouping_vars = "src_length_class")
print(post_hoc_results[0])
print(post_hoc_results[1].query("src_length_class == 'short'"))
print(post_hoc_results[1].query("src_length_class == 'typical'"))
print(post_hoc_results[1].query("src_length_class == 'very long'"))
print("Done")