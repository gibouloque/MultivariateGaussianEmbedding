# Multivariate Gaussian Embedding


## Content

This repository hosts the code and parameters necessary to replicate the results in our publication  ”Multivariate Side-Informed Gaussian Embedding Minimizing Statistical Detectability”.

It contains three main notebooks which show how to use the script to :

- Develop the BOSS dataset in the same way as in the paper using the Linear and the BOSS pipeline
- Compute the matrix reprensenting the processing pipeline
- Compute the covariance matrix for the 4Lat-MGE embedding scheme
- Simulate embedding using the 4Lat-MGE embedding scheme

We also provide most parameters used in the paper in the *data/* folder :

- Heteroscedastic parameters of all camera/ISO in BOSSBase *BOSS_noise_params_robust.npy*
- Saturating value all camera/ISO in BOSSBase *Camera_BOSS_max_val.npy*
- Pipeline matrices (available in the *filters/* folder of each dataset)

## Dependencies
- numpy
- scipy
- rawpy
- jpegio (use the patched version in this repository and compile from source)


# Acknowledgements

The images provided in this repository come from the BOSSBase datasets available at Pr Fridrich's website : http://agents.fel.cvut.cz/stegodata/ -- see [1] for more information.

When using all or part of this work, please credit us by citing  the following publication  *Q. Giboulot, P. Bas and R. Cogranne ”Multivariate Side-Informed Gaussian Embedding Minimizing Statistical Detectability”*

[1] Bas, P., Filler, T. and Pevný, T., 2011, May. ” Break our steganographic system”: the ins and outs of organizing BOSS. In International workshop on information hiding (pp. 59-70). Springer, Berlin, Heidelberg


