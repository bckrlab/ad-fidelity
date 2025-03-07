<!-- These are examples of badges you might want to add to your README:
     please update the URLs accordingly

[![Built Status](https://api.cirrus-ci.com/github/<USER>/ad-fidelity.svg?branch=main)](https://cirrus-ci.com/github/<USER>/ad-fidelity)
[![ReadTheDocs](https://readthedocs.org/projects/ad-fidelity/badge/?version=latest)](https://ad-fidelity.readthedocs.io/en/stable/)
[![Coveralls](https://img.shields.io/coveralls/github/<USER>/ad-fidelity/main.svg)](https://coveralls.io/r/<USER>/ad-fidelity)
[![PyPI-Server](https://img.shields.io/pypi/v/ad-fidelity.svg)](https://pypi.org/project/ad-fidelity/)
[![Conda-Forge](https://img.shields.io/conda/vn/conda-forge/ad-fidelity.svg)](https://anaconda.org/conda-forge/ad-fidelity)
[![Monthly Downloads](https://pepy.tech/badge/ad-fidelity/month)](https://pepy.tech/project/ad-fidelity)
[![Twitter](https://img.shields.io/twitter/url/http/shields.io.svg?style=social&label=Twitter)](https://twitter.com/ad-fidelity)
-->

[![Static Badge](https://img.shields.io/badge/doi-10%2Fn87w-blue?style=flat&logo=doi)](https://doi.org/10.1007/978-3-658-47422-5_18)
![Static Badge](https://img.shields.io/badge/Python-3.12-blue?style=flat&logo=python)
[![Project generated with PyScaffold](https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold)](https://pyscaffold.org/)

# ad-fidelity

> Evaluating Explanations of Convolutional Neural Networks for Alzheimer’s Disease Classification

## Setup


```shell
# clone repository
git clone
cd ad-fidelity

# optional: using conda
conda create -n ad-fidelity python=3.12
conda activate ad-fidelity

# install whole package
pip install -e .

# or install requirements only
pip install -r requirements.txt
```

## Data

This project used MRI images from the ADNI database:

> Data used in this project were obtained from the Alzheimer’s Disease Neuroimaging Initiative (ADNI) database ([adni.loni.usc.edu](adni.loni.usc.edu)).
As such, the investigators within the ADNI contributed to the design and implementation of ADNI and/or provided data but did not participate in analysis or writing of this report.  A complete listing of ADNI investigators can be found at:
>
> http://adni.loni.usc.edu/wp-content/uploads/how_to_apply/ADNI_Acknowledgement_List.pdf

You can apply for data access at:
https://adni.loni.usc.edu/data-samples/adni-data/#AccessData

Copy the `*.nii.gz` data files into:

```sh
# AD subjects
mv AD data/adni/

# CN subjects
mv CN data/CN/
```

## Publications

- Presentation at the [German Converence on Medical Image Computing](https://www.bvm-conf.org/) (BVM) 2025 
     - You can find the slides [here](docs/presentations/2025-03-09_bvm/2025-bvm.pdf)
- Hiller, Bader, Singh, Kirste, Becker, Dyrba: [*Evaluating the Fidelity of Explanations for Convolutional Neural Networks in Alzheimer’s Disease Detection*](https://doi.org/10.1007/978-3-658-47422-5_18) (2025)

## Acknowledgements

Funding was provided by the BMBF (01IS22077) and DFG (DY151/2-1). Data was provided by the Alzheimer’s Disease Neuroimaging Initiative (ADNI).

<!--
acknowledge ADNI according to their data use agreement:
https://adni.loni.usc.edu/wp-content/themes/adni_2023/documents/ADNI_Data_Use_Agreement.pdf
-->

> Data collection and sharing for this project was funded by the Alzheimer's Disease Neuroimaging Initiative (ADNI) (National Institutes of Health Grant U01 AG024904) and DOD ADNI (Department of Defense award number W81XWH-12-2-0012).
> ADNI is funded by the National Institute on Aging, the National Institute of Biomedical Imaging and Bioengineering, and through generous contributions from the following:
> AbbVie, Alzheimer’s Association; Alzheimer’s Drug Discovery Foundation; Araclon Biotech; BioClinica, Inc.; Biogen; Bristol-Myers Squibb Company; CereSpir, Inc.; Cogstate; Eisai Inc.; Elan Pharmaceuticals, Inc.; Eli Lilly and Company; EuroImmun; F. Hoffmann-La Roche Ltd and its affiliated company Genentech, Inc.; Fujirebio; GE Healthcare; IXICO Ltd.; Janssen Alzheimer Immunotherapy Research & Development, LLC.; Johnson & Johnson Pharmaceutical Research & Development LLC.; Lumosity; Lundbeck; Merck & Co., Inc.; Meso Scale Diagnostics, LLC.; NeuroRx Research; Neurotrack Technologies; Novartis Pharmaceuticals Corporation; Pfizer Inc.; Piramal Imaging; Servier; Takeda Pharmaceutical Company; and Transition Therapeutics.
> The Canadian Institutes of Health Research is providing funds to support ADNI clinical sites in Canada.
> Private sector contributions are facilitated by the Foundation for the National Institutes of Health (www.fnih.org).
> The grantee organization is the Northern California Institute for Research and Education, and the study is coordinated by the Alzheimer’s Therapeutic Research Institute at the University of Southern California.
> ADNI data are disseminated by the Laboratory for Neuro Imaging at the University of Southern California.

## License

This project is licenced under the MIT licence.

<!-- pyscaffold-notes -->

## Note

This project has been set up using PyScaffold 4.5. For details and usage
information on PyScaffold see https://pyscaffold.org/.