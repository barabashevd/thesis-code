# Comparing Methods for Detecting Concept Drift in Data Streams

Main repository for thesis code.

This is repurposed code from [BODMAS](https://github.com/whyisyoung/BODMAS) to more suit tests for my thesis. 

## Files

To run this code, some external files are necessary.

- multiple_models/bodmas.npz: BODMAS dataset's feature vectors
- multiple_models/bodmas_metadata: BODMAS dataset's labels
- multiple_models/ember.npz: EMBER Dataset for pretraining model
- multiple_models/ember_metadata.csv: EMBER Dataset's labels

## Installation

The environment was tested on a Python 3.9, the requirements can be installed by running:

```shell
pip install requirements.txt
python setup.py install
```

## Running scripts

### Training on new data

**Description:** This code replicates the findings in Figure 2.6. This code was taken from the original BODMAS article.

```shell
python bodmas_main.py --training-set bodmas --diversity no --setting-name bluehex_diversity_no --classifier gbdt --testing-time 2019-10,2020- 09 --retrain 1 --seed 0 --quiet 0
```
- training-set <dataset>: Specifies the dataset to use for training. For example, bodmas or ember.
- diversity <option>: Enables or disables diversity settings. Set to no to disable diversity. Not tested in this thesis.
- setting-name <name>: Configures the name for the training setting, used for saving logs.
- classifier <type>: Defines the classifier to be used. Supported for GBDT.
- testing-time <start,end>: Defines the time period over which testing should be conducted, separated by comma. For instance, 2019-10,2020-09 represents testing from October 2019 to September 2020.
- retrain <flag>: Whetever to load an existing model or train (and rewrite) a new one.
- seed <number>: Sets the seed for random number generation.
- quiet <mode>: Controls the verbosity of the output. Set to 0 for normal output, 1 for verbouse.

### Incremental Retraining

**Description:** This command runs the concept drift detection using the Ember dataset with specified settings. This code was taken from the original BODMAS article.

```shell
python concpet_drift_ember.py --setting-name ember_drift_random_improved --classifier gbdt --month-interval 1 --testing-time 2019-10,2020- 09 --ember-ratio 1 --seed 1 --sample-ratio 0.01 --retrain 0 --quiet 0
```

- setting-name <name>: Configures the name for the training setting, used for saving logs.
- classifier <type>: Defines the classifier to be used. Supported for GBDT.
- month-interval <number>: specifies how many month use for sampling
- testing-time <start,end>: Defines the time period over which testing should be conducted, separated by comma. For instance, 2019-10,2020-09 represents testing from October 2019 to September 2020.
- ember-ratio <float>: what size of the EMBER dataset to use for accumulative retraining.
- sample-ratio <float>: how many samples to add to accumalive retrainig
- retrain <flag>: Whetever to load an existing model or train (and rewrite) a new one.
- seed <number>: Sets the seed for random number generation.
- quiet <mode>: Controls the verbosity of the output. Set to 0 for normal output, 1 for verbouse.

### D3

**Description:** This command runs the concept drift detection using the D3
approach (results of Figure 2.10). Source: [Unsupervised Concept Drift Detection with a Discriminative Classifier](https://dl.acm.org/doi/abs/10.1145/3357384.3358144)

```shell
python d3.py --w 2500 --rho 0.1 --tau 0.99
```

- w <int>: sets sliding window size
- rho <float>: sets percentage of new data
- tau <float>: AUC treshold


### MD3

**Description:** This command runs the MD3 concept drift detection method
with the BODMAS dataset. A baseline model trained on EMBER needs to
replicate Figure 2.11.

```shell
concept_drift_md3.py --setting-name md3 --classifier gbdt --month- interval 1 --testing-time 2019-10,2020-09 --ember-ratio 1.0 --seed 1 --retrain 1 --quiet 0
```

Parameters are the same as in previous  examples.


### Improved detection

**Description:** he command runs the improved drift detection based on
MD3 and D3. This will result in Figure 2.12.

```shell
concept_drift_improved.py --setting-name ember_drift_random_improved --classifier gbdt --month-interval 1 --testing-time 2019-10,2020-09 --ember-ratio 1.0 --seed 1 --sample-ratio 0.01 --retrain 1 --quiet 0
```

Parameters are the same as in previous examples.
