# TTS Processing Pipeline 

## Scope of this pipeline
This pipeline runs a list of Speech Data Processors (SDP) to process raw audios for TTS training and saves all the processing information in a manifest. This pipeline contains this following processors:
  1. [CreateInitialManifestYTC](../datasets/ytc/create_initial_manifest.py#L22): Creates initial manifest by resampling audio to 16kHz mono WAV format
  2. [PyAnnoteDiarizationAndOverlapDetection](./pyannote.py#L55): Runs speaker diarization and overlap detection using pyannote to get speaker information of the audio
  3. [SplitLongAudio](./split.py#L23): Splits long audio into segments of max duration specified
  4. [NeMoASRAligner](./nemo_asr_align.py#L23): Runs ASR model on above split segments to get transcripts and word level timestamps
  5. [JoinSplitAudioMetadata](./split.py#L148): Joins split audio metadata back together
  6. [MergeAlignmentDiarization](./merge_alignment_diarization.py#L19): Merges alignment and diarization information to assign word level timestamps and transcript to each diarization segment 
  7. [InverseTextNormalizationProcessor](./text.py#L22): Performs inverse text normalization
  8. [TorchSquimObjectiveQualityMetricsProcessor](./metrics.py#L30): Calculates audio quality metrics such as pesq, squim and stoi on each diarization segment using TorchSQUIM
  9. [BandwidthEstimationProcessor](./metrics.py#L126): Estimates original bandwidth of the audio signal on each diarization segment 
  10. [PrepareTTSSegmentsProcessor](./prepare_tts_segments.py#L21): Combines adjacent segments from the same speaker to create new TTS segments, ensuring each segment contains a complete utterance.

## Prerequisite
### Building the docker image
A docker image is required to run the TTS processors.

```bash
docker build -t <image-tag-name> -f docker/Dockerfile.tts_sdp .
```


## Input / Output of the pipeline
**Input manifest**:  
    A manifest json file that contains `audio_filepath` and `audio_item_id`, where the `audio_filepath` contains the audio filepath and `audio_item_id` is a unique identifier for each audio.  
    For example: {"audio_filepath": "path/to/original/audio/file.wav", "audio_item_id": "some_unique_id"}

**Output manifest**:  
    A information dense manifest that contains single speaker speech segments, word timestamps for every word, transcripts, audio metrics, etc. Here is an example output of one audio:


| Metadata Key | Data type | Information | Example |
| :-------- | :-------- | :-------- | :-------- |
|audio_filepath|str|where the audio is|/workspace/audios/example.wav|
|audio_item_id|str|unqiue identifier of this audio|9c3870cd|
|duration|float|audio duration in number of seconds|200|
|segments|list[dict]|single speaker segments for TTS training, which have speaker_id, timestamps and text segment of max 40 seconds, every single word's timestamp and estimated audio metrics including PESQ, STOI, SISDR and bandwidth |{"speaker": "9c3870cd_SPEAKER_1", "start": 132.3684210526316, "end": 134.9320882852292, "text": "the work is above and beyond the line of duty.", "words": [{"word": "the", "start": 132306125, "end": 132.546125}, {"word": "work", "start": 132.466125, "end": 132.546125}, {"word": "is", "start": 132.786125, "end": 132.866125}, {"word": "above", "start": 133.026125, "end": 133.426125}, {"word": "and", "start": 133.426125, "end": 133.506125}, {"word": "beyond", "start": 133586125, "end": 133.986125}, {"word": "the", "start": 133.986125, "end": 134.066125}, {"word": "line", "start": 134.14612499999998, "end": 134.306125}, {"word": "of", "start": 134.386125, "end": 134.546125}, {"word": "duty.", "start": 134.546125, "end": 135.106125}], "text_ITN": "the work is above and beyond the line of duty.", "metrics": {"pesq_squim": 2.023, "stoi_squim": 0.948, "sisdr_squim": 15.051, "bandwidth": 7125}}|
|alignment|list[dict]|Word level timestamps for entire video|{start, end, word}|
|overlap_segments|list[dict]|Segments that have overlap speech|{start, end, word}|
|text|str|Full text of entire video|Text with PNC


### Running the pipeline
After building the Docker image, please run the processors inside a Docker container, preferably with GPU available to enable fast inference.

Run the following command to run the entire processing pipeline:
```bash
HYDRA_FULL_ERROR=1 python main.py --config-path="/src/NeMo-speech-data-processor/nemo-sdp-tts/dataset_configs/tts/ytc" --config-name config.yaml workspace_dir=/work/ data_split="train" nemo_path=$NEMO_PATH hf_token=$HF_TOKEN +input_manifest_file={your_input_manifest_file} final_manifest={your_final_manifest_file}
```
