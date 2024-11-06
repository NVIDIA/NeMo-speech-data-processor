import json
import tempfile
from pathlib import Path
from sdp.utils import BootstrapProcessor

def _write_manifest(manifest_path: Path, entries):
    with manifest_path.open("w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")

def test_bootstrap_processor():
    manifest1_data = [
        {"audio_filepath": "path1.wav", "duration": 3.744, "text": "Նա նաև լավ էր գրում մանկական ոտանավորներ։", 
         "pred_text": "Նա նաև լավ էր գրում մանկական ոտանավորներ։", "wer": 0.142857, "tokens": 7, 
         "ins_rate": 0.0, "del_rate": 0.0, "sub_rate": 0.142857},
        {"audio_filepath": "path2.wav", "duration": 5.76, "text": "Ամենամեծ ջանքերը պահանջեց աղյուսների և կղմինդրների արտադրությունը։", 
         "pred_text": "Ամենամեծ ջանքերը պահանջեց աղյուսների և կաղնիտների արտադրությունը։", "wer": 0.285714, "tokens": 7, 
         "ins_rate": 0.0, "del_rate": 0.0, "sub_rate": 0.285714},
        {"audio_filepath": "path3.wav", "duration": 6.984, "text": "Եթե մոտակայքում չկան մեղվափեթակներ, ապա բերքատվությունը կնվազի մոտ երեք անգամ։", 
         "pred_text": "Եթե մոտակայքում չկան մեղվափեթակներ, ապա բերքատվությունը կնվագի մոտ երեք անգամ։", "wer": 0.1, "tokens": 10, 
         "ins_rate": 0.0, "del_rate": 0.0, "sub_rate": 0.1},
        {"audio_filepath": "path4.wav", "duration": 4.104, "text": "Դպրոցը հիմնականում պահվել է եկեղեցու եկամուտներով։", 
         "pred_text": "Դպրոցը հիմնականում պահվել է եկեղեցու եկամուտներով։", "wer": 0.0, "tokens": 6, 
         "ins_rate": 0.0, "del_rate": 0.0, "sub_rate": 0.0}
    ]
    
    manifest2_data = [
        {"audio_filepath": "path1.wav", "duration": 3.744, "text": "Նա նաև լավ էր գրում մանկական ոտանավորներ։", 
         "pred_text": "Նա նաև լավ էր գրում մանկական ոտանավորներ։", "wer": 0.142857, "tokens": 7, 
         "ins_rate": 0.0, "del_rate": 0.0, "sub_rate": 0.142857},
        {"audio_filepath": "path2.wav", "duration": 5.76, "text": "Ամենամեծ ջանքերը պահանջեց աղյուսների և կղմինդրների արտադրությունը։", 
         "pred_text": "Ամենամեծ ջանքերը պահանջեց աղյուսների և կղմիտների արտադրությունը։", "wer": 0.285714, "tokens": 7, 
         "ins_rate": 0.0, "del_rate": 0.0, "sub_rate": 0.285714},
        {"audio_filepath": "path3.wav", "duration": 6.984, "text": "Եթե մոտակայքում չկան մեղվափեթակներ, ապա բերքատվությունը կնվազի մոտ երեք անգամ։", 
         "pred_text": "Եթե մոտակայքում չկան մեղվափետներ, ապա բերքատվությունը կնվացի մոտ երեք անգամ։", "wer": 0.2, "tokens": 10, 
         "ins_rate": 0.0, "del_rate": 0.0, "sub_rate": 0.2},
        {"audio_filepath": "path4.wav", "duration": 4.104, "text": "Դպրոցը հիմնականում պահվել է եկեղեցու եկամուտներով։", 
         "pred_text": "Դպրոցը հիմնականում պահվել է եկեղեցու եկամուտներով։", "wer": 0.0, "tokens": 6, 
         "ins_rate": 0.0, "del_rate": 0.0, "sub_rate": 0.0}
    ]

    # Expected output for comparison
    expected_output = {
        "individual_results": {
            "manifest1.json": {
                "mean_wer": 5.358,
                "ci_lower": 0.5625,
                "ci_upper": 10.992625
            },
            "manifest2.json": {
                "mean_wer": 9.0725,
                "ci_lower": 5.0,
                "ci_upper": 15.234875
            }
        },
        "pairwise_comparisons": [
            {
                "file_1": "manifest1.json",
                "file_2": "manifest2.json",
                "delta_wer_mean": -1.75,
                "ci_lower": -5.0,
                "ci_upper": 0.0,
                "poi": 0.0
            }
        ]
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create temporary paths
        manifest1_path = Path(tmpdir) / "manifest1.json"
        manifest2_path = Path(tmpdir) / "manifest2.json"
        output_path = Path(tmpdir) / "output_manifest.json"

        # Write manifest data to temporary files
        _write_manifest(manifest1_path, manifest1_data)
        _write_manifest(manifest2_path, manifest2_data)

        # Run BootstrapProcessor with test parameters
        processor = BootstrapProcessor(
            bootstrap_manifest_files=[str(manifest1_path), str(manifest2_path)],
            raw_data_dir=str(tmpdir),
            output_file=str(output_path),
            num_bootstraps=10,
            bootstrap_sample_ratio=1.0,
            calculate_pairwise=True,
            metric_type='wer',
            text_key='text',
            pred_text_key='pred_text',
            ci_lower=2.5,
            ci_upper=97.5,
            random_state=42,
            output_manifest_file=None # A placeholder to skip BaseProcessor failing
        )

        processor.process()

        # Load and compare the processor output
        with open(output_path, "r") as f:
            output = json.load(f)
        
        assert output == expected_output, f"Expected {expected_output}, but got {output}"
