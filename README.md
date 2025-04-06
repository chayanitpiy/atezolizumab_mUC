# atezolizumab_mUC

Validation analyses

This code was used to develop the best-performing CART-OMC model to predict atezolizumab response using gene expression profiles from 298 atezolizumab-treated mUC patients in the IMvigor210 discovery dataset.
This model was then validated on 22 mUC patients treated with atezolizumab from the Synder et al. validation dataset.

This is the command line used for validation analyses:
python -u GEX_validation_analyses.py -d GEX_discovery.csv -v GEX_validation.csv

Arguments:

-u --> Python code

-d --> discovery dataset: The tabulated pharmaco-omics dataset, where columns specify patients, their corresponding drug responses (CR, PR, SD, and PD), and gene expression features.

-v, --validation dataset: The tabulated pharmaco-omics dataset, where columns specify patients, their corresponding drug responses (CR, PR, SD, and PD), and gene expression features.
