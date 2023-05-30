# Speech Data Processor

Speech Data Processor (SDP) is a toolkit to make it easy to:
1. write code to process a new dataset, minimizing the amount of boilerplate code required.
2. share the steps for processing a speech dataset. Sharing processing steps can be as easy as sharing a YAML file.

SDP's philosophy is to represent processing operations as 'processor' classes. Many common processing operations are provided, and it is easy to add your own. In some cases, all you will need to do to process a new dataset is simply to write a YAML file containing the parameters needed to process your dataset.

SDP is specifically intended for the use case when you have an existing dataset with the audio & text pairs already specified in some form, and you wish to create a JSON manifest suitable for use with NeMo. SDP allows for intermediate cleaning and filtering steps which involve amending the 'ground truth' `"text"` (or some other field, specified using `"text_key"`) or dropping utterances which are deemed to be too inaccurate for training on.


![SDP overview](https://github.com/NVIDIA/NeMo/releases/download/v1.17.0/sdp_overview_diagram.png)

## Quick intro to Speech Data Processor

* The steps to process a dataset are specified by a YAML config file.
* The YAML config file contains a list of processor classes & the args to pass into the constructor.
* Each processor class inputs an existing manifest (except for classes which create an 'initial' manifest from some external transcript file)  & outputs a modified version of the manifest. It may change other files in the process, e.g. resample audio.
* To process a manifest, you need to list the chain of processors you wish to use.
* If a processor is not included, you can make your own.


## YAML config file layout
A simplified version of an SDP file can be:

```yaml
processors:

  # use existing classes for popular datasets or make your own class
  - _target_: sdp.processors.CreateInitialManifestMLS
    output_manifest_file: ...
    raw_data_dir: ...
    ...

  # use existing classes for common operations or write your own
  - _target_: sdp.processors.SubRegex
    regex_params_list:
      # specify the parameters needed for your usecase
      - {"pattern": " mr ", "repl": " mister "}
      - {"pattern": " misteak ", "repl": " mistake "}

  - _target_: sdp.processors.DropNonAlphabet
    alphabet: " abcdefghijklmnopqrstuvwxyz"
    output_manifest_file: ...
    ...
```
## Existing processor classes
In addition to those mentioned in the example config file, many more classes are already included in Speech Data Processor, for example:
* `sdp.processors.ASRInference` will run inference on the manifest using a specified `pretrained_model`.
* `sdp.processors.DropHighWER` will compute WER between `text` and `pred_text` of each utterance and remove the utterance if WER is greater than the specified `wer_threshold`.
* `sdp.processors.DropHighLowCharrate` will compute the character rate in the utterance using `text` and `duration`, and drop the utterance if it is outside the bounds of the specified `high_charrate_threshold` and `low_charrate_threshold`. Carefully chosen thresholds will allow us to drop utterances with incorrect ground truth `text`.

## Processor test cases
You can add test cases to verify you have specified your desired changes correctly and to help document why your are making these changes.

For example:
```yaml
processors:
  ...
  - _target_: sdp.processors.DropIfRegexMatch
  regex_patterns:
  - '(\D ){5,20}' # looks for between 4 and 19 characters surrounded by space

  test_cases:
    - {input: {text: "some s p a c e d out letters"}, output: null}
    - {input: {text: "normal words only"}, output: {text: "normal words only"}}
    - {input: {text: "three a b c spaced out letters"}, output: {text: "three a b c spaced out letters"}}
    - {input: {text: "four a b c d spaced out letters"}, output: null}
  ...
```

## Installing requirements

SDP is officially supported for Python 3.8, but might work for other versions.

To install all required dependencies run `pip install -r requirements.txt` and (optionally) `pip install -r tests/requirements.txt` if you want to run tests.

Some SDP processors depend on the NeMo toolkit (ASR, NLP parts) and NeMo Text Processing.
Please follow [NeMo installation instructions](https://github.com/NVIDIA/NeMo#installation) and [NeMo Text Processing installation instructions](https://github.com/NVIDIA/NeMo-text-processing#installation).


## Running SDP
After installing the requirements for SDP, to use it, you will need to download it using e.g. `git clone https://github.com/NVIDIA/NeMo-speech-data-processor.git`.

### Run config in repository
To run processing for a dataset that has a config and all processor code inside the SDP directory structure, run commands like below:

```bash
python <SDP_ROOT>/main.py \
    --config-path="dataset_configs/spanish_pc/mcv12/" \
    --config-name="config.yaml" \
    data_split="test" \
    workspace_dir="<dir where processed data will be saved, and where initial data tar file is already located>"
```

### Run own config
To run processing for a dataset with a config that is not inside the SDP directory structure, but all processors are within the SDP directory, run commands like:

```bash
python <SDP_ROOT>/main.py \
    --config-path="<path to config: either absolute path or relative path *from SDP_ROOT directory*>" \
    --config-name="<config file name>.yaml" \
    ... # any other parameters
```

### Run own config and use own SDP processors
To run processing for a dataset with a config that is not inside the SDP directory structure, and at least one processor is outside the SDP directory structure run:

```bash
PYTHONPATH=<path to dir containing your custom processor either directly or in subdirectory> python <SDP_ROOT>/main.py \
    --config-path="<path to config: either absolute path or relative path *from SDP_ROOT directory*>" \
    --config-name="<config file name>.yaml" \
    ... # any other parameters
```
Furthermore, when you add your own SDP processors in the YAML config file, you will need to define the `_target_` correctly by making sure it describes the relative path to the processor class from the `PYTHONPATH` you defined.

For example, if a new custom processor `MyProcessor` is in `/my/files/a/b/c/my_processor.py`, you can use combinations such as: `PYTHONPATH="/my/files/a/b/c/"` & `_target_: my_processor.MyProcessor`, or `PYTHONPATH="/my/files/a/b/"` & `_target_: c.my_processor.MyProcessor`.

If you add new processors within `<SDP_ROOT>`, to use them when you call SDP, you will not need to specify a `PYTHONPATH` variable, but you will need to make sure that the `_target_` inside the YAML config file is correct. If a new custom processor is `MyProcessor` inside `<SDP_ROOT>/dir/my_processor.py`, `_target_` will need to be `dir.my_processor.MyProcessor`.

## Additional documentation
More information about SDP can be found in the [NeMo docs](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/tools/speech_data_processor.html).
