from sdp.processors.base_processor import BaseProcessor
from sdp.utils.chunk_processing import ChunkProcessingPipeline


class GroupProcessors(BaseProcessor):
    def __init__(
        self,
        output_manifest_file: str,
        input_manifest_file: str | None = None,
        chunksize: int = 500,
        **processors_cfg,
    ):
        super().__init__(
            output_manifest_file=output_manifest_file,
            input_manifest_file=input_manifest_file,
        )

        self.initial_manifest_file = input_manifest_file
        self.chunksize = chunksize
        self.processors_cfg = processors_cfg["processors"]

    def test(self):
        pass

    def process(self):
        chunked_pipeline = ChunkProcessingPipeline(
            initial_manifest_file=self.initial_manifest_file,
            chunksize=self.chunksize,
            processors_cfgs=self.processors_cfg,
        )

        chunked_pipeline.run()
