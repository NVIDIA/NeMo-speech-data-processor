
1. sdp.processors.huggingface.speech_recognition.ASRWhisper fails to read `input_manifest_file`

    ```markdown
        [SDP I 2024-01-20 20:43:28 run_processors:150] => Running processor "<sdp.processors.huggingface.speech_recognition.ASRWhisper object at 0x000002663AD644F0>"
        
        Error executing job with overrides: []
        
        Traceback (most recent call last):
          File "main.py", line 34, in <module>
            main()
          File "nemo_sdp_venv\lib\site-packages\hydra\main.py", line 94, in decorated_main
            _run_hydra(
          File "nemo_sdp_venv\lib\site-packages\hydra\_internal\utils.py", line 394, in _run_hydra
            _run_app(
          File "nemo_sdp_venv\lib\site-packages\hydra\_internal\utils.py", line 457, in _run_app
            run_and_report(
          File "nemo_sdp_venv\lib\site-packages\hydra\_internal\utils.py", line 223, in run_and_report
            raise ex
          File "nemo_sdp_venv\lib\site-packages\hydra\_internal\utils.py", line 220, in run_and_report
            return func()
          File "nemo_sdp_venv\lib\site-packages\hydra\_internal\utils.py", line 458, in <lambda>
            lambda: hydra.run(
          File "nemo_sdp_venv\lib\site-packages\hydra\_internal\hydra.py", line 132, in run
            _ = ret.return_value
          File "nemo_sdp_venv\lib\site-packages\hydra\core\utils.py", line 260, in return_value
            raise self._return_value
          File "nemo_sdp_venv\lib\site-packages\hydra\core\utils.py", line 186, in run_job
            ret.return_value = task_function(task_cfg)
          File "main.py", line 24, in main
            run_processors(cfg)
          File "run_processors.py", line 151, in run_processors
            processor.process()
          File "processors\huggingface\speech_recognition.py", line 56, in process
            json_list = load_manifest(Path(self.input_manifest_file))
          File "utils\common.py", line 31, in load_manifest
            for line in f:
          File "Python310\lib\encodings\cp1252.py", line 23, in decode
            return codecs.charmap_decode(input, self.errors, decoding_table)[0]
        UnicodeDecodeError: 'charmap' codec can't decode byte 0x81 in position 241: character maps to <undefined>
    ```

    Need to catch the UnicodeDecodeError and load like:
     - Generator Option - `sdp.processors.base_processor.BaseParallelProcessor.read_manifest()`
     - Direct Loading (example) - `sdp.processors.modify_manifest.common.SortManifest.SortManifest`
       ```python
        with open(self.input_manifest_file, "rt", encoding="utf8") as fin:
          dataset_entries = [json.loads(line) for line in fin.readlines()]
        ```

2. to add

