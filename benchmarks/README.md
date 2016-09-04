Imbalanced dataset for benchmarking
===================================

Purposes
--------

A benchmark of the different methods proposed in the `imbalanced-learn` toolbox is provided. The methods are tested in conjunction of different `scikit-learn` classifiers. ROC analysis as well as computation time analysis are performed.

Datasets
--------

The different algorithms of the `imbalanced-learn` toolbox are evaluated on a set of common dataset, which are more or less balanced. These benchmark have been proposed in [1]. The following section presents the main characteristics of this benchmark.

### Characteristics

|ID   |Name          |Repository & Target          |Ratio|# samples| # features |
|:---:|:------------:|-----------------------------|:---:|:-------:|:----------:|
|1    |Ecoli         |UCI, target: imU             |8.6:1|336      |7           |
|2    |Optical Digits|UCI, target: 8               |9.1:1|5,620    |64          |
|3    |SatImage      |UCI, target: 4               |9.3:1|6,435    |36          |
|4    |Pen Digits    |UCI, target: 5               |9.4:1|10,992   |16          |
|5    |Abalone       |UCI, target: 7               |9.7:1|4,177    |8           |
|6    |Sick Euthyroid|UCI, target: sick euthyroid  |9.8:1|3,163    |25          |
|7    |Spectrometer  |UCI, target: >=44            |11:1 |531      |93          |
|8    |Car_Eval_34   |UCI, target: good, v good    |12:1 |1,728    |6           |
|9    |ISOLET        |UCI, target: A, B            |12:1 |7,797    |617         |
|10   |US Crime      |UCI, target: >0.65           |12:1 |1,994    |122         |
|11   |Yeast_ML8     |LIBSVM, target: 8            |13:1 |2,417    |103         |
|12   |Scene         |LIBSVM, target: >one label   |13:1 |2,407    |294         |
|13   |Libras Move   |UCI, target: 1               |14:1 |360      |90          |
|14   |Thyroid Sick  |UCI, target: sick            |15:1 |3,772    |28          |
|15   |Coil_2000     |KDD, CoIL, target: minority  |16:1 |9,822    |85          |
|16   |Arrhythmia    |UCI, target: 06              |17:1 |452      |279         |
|17   |Solar Flare M0|UCI, target: M->0            |19:1 |1,389    |10          |
|18   |OIL           |UCI, target: minority        |22:1 |937      |49          |
|19   |Car_Eval_4    |UCI, target: vgood           |26:1 |1,728    |6           |
|20   |Wine Quality  |UCI, wine, target: <=4       |26:1 |4,898    |11          |
|21   |Letter Img    |UCI, target: Z               |26:1 |20,000   |16          |
|22   |Yeast _ME2    |UCI, target: ME2             |28:1 |1,484    |8           |
|23   |Webpage       |LIBSVM, w7a, target: minority|33:1 |49,749   |300         |
|24   |Ozone Level   |UCI, ozone, data             |34:1 |2,536    |72          |
|25   |Mammography   |UCI, target: minority        |42:1 |11,183   |6           |
|26   |Protein homo. |KDD CUP 2004, minority       |111:1|145,751  |74          |
|27   |Abalone_19    |UCI, target: 19              |130:1|4,177    |8           |

### References

[1] Ding, Zejin, "Diversified Ensemble Classifiers for H
ighly Imbalanced Data Learning and their Application in Bioinformatics." Dissertation, Georgia State University, (2011).

[2] Blake, Catherine, and Christopher J. Merz. "UCI Repository of machine learning databases." (1998).

[3] Chang, Chih-Chung, and Chih-Jen Lin. "LIBSVM: a library for support vector machines." ACM Transactions on Intelligent Systems and Technology (TIST) 2.3 (2011): 27.

[4] Caruana, Rich, Thorsten Joachims, and Lars Backstrom. "KDD-Cup 2004: results and analysis." ACM SIGKDD Explorations Newsletter 6.2 (2004): 95-108.
