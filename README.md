# Speech Data Processor

Speech Data Processor (SDP) is a toolkit to make it easy to:
1. Write code to process a new dataset, minimizing the amount of boilerplate code required.
2. Share the steps for processing a speech dataset.

SDP's philosophy is to represent processing operations as 'processor' classes, which take in a path to a NeMo-style
data manifest as input (or a path to the raw data directory if you do not have a NeMo-style manifest to start with),
apply some processing to it, and then save the output manifest file.

You specify which processors you want to run using a YAML config file. Many common processing operations are provided,
and it is easy to add your own.

![SDP overview](https://github.com/NVIDIA/NeMo/releases/download/v1.17.0/sdp_overview_diagram.png)

To learn more about SDP, have a look at our [documentation](https://nvidia.github.io/NeMo-speech-data-processor/).

## Installation

SDP is officially supported for Python 3.9, but might work for other versions.

To install all required dependencies run `pip install -r requirements/main.txt`. You will need to install
additional requirements if you want to [run tests](tests/README.md) or [build documentation](docs/README.md).

Some SDP processors depend on the NeMo toolkit (ASR, NLP parts) and NeMo Text Processing.
Please follow [NeMo installation instructions](https://github.com/NVIDIA/NeMo#installation)
and [NeMo Text Processing installation instructions](https://github.com/NVIDIA/NeMo-text-processing#installation)
if you need to use such processors.

## Contributing
We welcome community contributions! Please refer to the [CONTRIBUTING.md](CONTRIBUTING.md) for the process.