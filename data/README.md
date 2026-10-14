# data/

| Path | In git? | What it is |
|---|---|---|
| `splits.json` | yes | The 5 date-grouped cross-validation folds (image stem -> fold). The notebook re-creates it and checks that it is identical. |
| `fsc26_final_submission.csv` | yes | The file we submitted to Kaggle (public leaderboard PQ 0.40, 1,102 masks). The notebook compares its own output with this file. |
| `MAGFiLO_1.0_Kaggle_2026/` | no | The competition data. **You download it** (step 2 of the main README). |
| `release/` | no | Created by the notebook: the 10 trained models (`weights/`) and the blended out-of-fold maps (`oof_blend/`), downloaded from the GitHub release `v1.0` (asset `fsc26_release_v1.zip`, 514 MB). |
| `output/` | no | Created by the notebook: `submission.csv`. |

## Competition data

The data are the MAGFiLO Kaggle-2026 files, licensed CC BY-NC 4.0. We do not redistribute them.
Accept the competition rules on Kaggle, then:

```bash
pip install kaggle                       # needs your Kaggle API token
kaggle competitions download -c filament-segmentation-2026 -p data
unzip data/filament-segmentation-2026.zip -d data
```

Expected layout:

```
data/MAGFiLO_1.0_Kaggle_2026/
  train/MAGFiLO_1.0_Annotations_kaggle2026_train.json     # COCO annotations, 1,154 readings of 707 images
  train/train_images/<stem>.jpeg                          # 707 GONG H-alpha frames, 2048 x 2048
  test/test_images/<stem>.jpeg                            # 180 test frames (no annotations)
```

To keep the data elsewhere, set `MAGFILO_ROOT=/path/to/MAGFiLO_1.0_Kaggle_2026`.
