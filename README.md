# BARP-Bias-Aware-Ranker-from-Pairwise-comparisons

Repository containing the code for the paper Bias-Aware Ranking from Pairwise comparisons. 

The python file opt_fair.py contains the class 'Pairwise_with_rev', which contains methods for the computation of the negative log-likelihood (objective) and the gradients w.r.t the scores (gradient_scores) and the evaluators (gradient_revs). 
The same class also contains methods for CrowdBT and FactorBT. In the same file also an implementation of RankCentrality.

Implementation of Bradley Terry is installed from the following repository: https://github.com/lucasmaystre/choix
And FAIR from here: https://github.com/fair-search/fairsearch-fair-python (note: it runs only on pythn 3.7 or before, but can be easily fixed for later versions)

Datasets are publically available here: https://github.com/Toloka/IMDB-WIKI-SbS and here https://github.com/cc-jalvarez/counterfactual-situation-testing/blob/master/data/clean_LawSchool.csv




references: 
CrowdBT: Chen, Xi, et al. "Pairwise ranking aggregation in a crowdsourced setting." Proceedings of the sixth ACM international conference on Web search and data mining. 2013.

FactorBT: Bugakova, Nadezhda, et al. "Aggregation of pairwise comparisons with reduction of biases." arXiv preprint arXiv:1906.03711 (2019).

RankCentrality: Negahban, Sahand, Sewoong Oh, and Devavrat Shah. "Iterative ranking from pair-wise comparisons." Advances in neural information processing systems 25 (2012).