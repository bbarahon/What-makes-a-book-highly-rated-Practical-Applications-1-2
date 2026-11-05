\# Evaluation of the kNN Model with the Optimized Dataset



The dataset consists of 52,478 rows, 26 predictors, and the target class (`rating\_class`). The original row order and data structure are preserved. Genres are not expanded, nor are series added. The selected set of attributes eliminates direct informational redundancy in distance metrics. Target rating leakage cannot be inferred from attribute names alone.



\## Configuration and Results



Evaluation conducted via 10-fold stratified cross-validation, seed 42, Weka 3.8.6 IBk, uniform weights, default Euclidean distance, and default linear search, sweeping $k$ from 1 to 51:



\- \*\*Optimal Configuration:\*\* $k = 42$

\- \*\*Peak Performance:\*\* 65.093944% accuracy (34,160 correct predictions out of 52,478).



Accuracy is calculated as $100 \\times \\text{sum of out-of-fold correct predictions} / 52,478$. The plot does not show standard deviations.



\## Conclusion



Tuning the hyperparameter to $k = 42$ achieves optimal model performance within the evaluated search space. Because this evaluation was used for hyperparameter selection, it does not serve as an independent final performance estimate. A held-out test set or nested validation scheme is required for independent confirmation.



The evaluated file is globally normalized, imputed, and discretized; for maximum methodological rigor, these transformations should be fitted strictly within the training fold of each cross-validation step starting from raw data.



\## Reproduction



Requires JDK 17, `weka.jar` 3.8.6, and `bounce.jar` 0.18. Maven coordinates: `nz.ac.waikato.cms.weka:weka-stable:3.8.6` and `nz.ac.waikato.cms.weka.thirdparty:bounce:0.18`.



```bash

javac -cp weka.jar KnnSweep.java

java -Xmx3g -cp weka.jar:bounce.jar:. KnnSweep dataset\_model.arff fold\_accuracy.csv

python -m pip install pandas matplotlib

python compare.py

