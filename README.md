<p align=\"center\">\n  <img width=\"250\" src=\"https://github.com/LookOutGap/LookOutGap-NLPAssistant/raw/main/images/nlplogo.png\">\n</p>\n\n# __LookOutGap-NLPAssistant__: Natural Language Processing Tool\n\n<p>\n <a href=\"https://github.com/LookOutGap/LookOutGap-NLPAssistant/actions\">\n   <img src=\"https://github.com/LookOutGap/LookOutGap-NLPAssistant/workflows/Build,%20Test,%20and%20Package/badge.svg\" alt=\"\"/>\n </a>\n <a href=\"https://codecov.io/gh/LookOutGap/LookOutGap-NLPAssistant\">\n   <img src=\"https://codecov.io/gh/LookOutGap/LookOutGap-NLPAssistant/branch/main/graph/badge.svg?token=hFbF8ID1Na\" alt=\"\"/>\n </a>\n</p>\n\nLookOutGap-NLPAssistant is a Python library that combines existing libraries for common Natural Language Processing tasks. It combines several existing open-source libraries such as pandas, spaCy, and scikit-learn to make a pipeline ready to process text data. There are many user-defined parameters depending on your type of project such as the ability to choose stemming or lemmatization. Or, you might want to define explicitly what to substitute with NaN text fields. Overall, it is a way to get you started in your NLP task, no matter what you need.\n\nA tutorial on how to use this package can be found [here](tutorial.ipynb).\n\n## Installation Instructions\n\n   - Using pip with Python version 3.7 or higher:\n        \```shell\n        pip install LookOutGap-NLPAssistant\n        \```\n   - For more information on installing packages using pip, click [here](https://pip.pypa.io/en/stable/reference/pip_install/).\n\n## Contributing \n- To help develop this package, you'll need to install a conda virtual \nenvironment defined by our dev_environment.yml file using the below command.\n\n  \```shell\n  conda env create -f dev_environment.yml\n  \```\n  - Then, just activate the environment when attempting to develop or run tests \n  using the below command.\n\n    \```shell\n    conda activate nlp_env\n    \```\n\n  - When you're all done developing or testing, just d