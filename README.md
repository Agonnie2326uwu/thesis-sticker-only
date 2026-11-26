# YOLO26n vs YOLO11n: PSU Vehicle Sticker Detection Benchmark

Thesis project (Palawan State University, BS Computer Science). Two object
detection models, YOLO26n and YOLO11n, are trained the same way on 1,400 photos
of PSU student vehicle stickers and compared in three tests:

| Test | What it checks | Script | Computer |
|---|---|---|---|
| 1 Natural test set | Accuracy (mAP, precision, recall, confidence) and speed (FPS) on 210 unseen photos | `evaluate_test.py` | RTX 2050 laptop |
| 2 Field test | Up to what distance (6-10 m) each model still finds the sticker | `evaluate_field.py` | RTX 2050 laptop |
| 3 Synthetic stressors | At what level of motion blur, occlusion, low light and glare each model fails | `evaluate_synthetic.py` | RTX 2050 laptop |

Everything can be run and shown from one window: **`python dashboard.py`**.

## Files

| File | Purpose |
|---|---|
| `dashboard.py` | The main window: runs each test and shows the results, charts and final comparison; exports a PDF report |
| `app.py` | Live camera demo, and the **Field test** tab used to capture field photos at the gate |
| `evaluate_test.py` | Test 1 |
| `evaluate_field.py` | Test 2 (after the field photos are labelled) |
| `evaluate_synthetic.py` | Test 3 |
| `make_gallery.py` | Photos page: draws both models' boxes on the test photos (viewing only, changes no result) |
| `train.py` | Training script (the final models are already trained; not needed to run the tests) |
| `runs/detect/carl-andre-de-castro/runs/*/weights/best.pt` | The two final trained models |
| `research_outputs/` | Results written by the tests |

Photos and datasets are **not** in this repository (they show faces and
license plates). The dataset is downloaded automatically from the Ultralytics
Platform when a test runs.

## Setup (Windows, once per computer)

1. Install **Python 3.12 or newer** and **Git**. Clone this repository and open
   the folder in VS Code.
2. Open a terminal (Terminal -> New Terminal) and create the environment:
   ```
   python -m venv venv
   venv\Scripts\activate
   ```
   Install PyTorch with NVIDIA GPU support: copy the install command for your
   system from https://pytorch.org/get-started/locally/ (choose Pip, Python,
   and the CUDA version that matches your GPU driver).
3. Install the rest:
   ```
   pip install -r requirements.txt
   ```
4. Log in to the Ultralytics Platform on this computer with an account that
   has access to the `carl-andre-de-castro` datasets (the same login used for
   training), so the tests can download the dataset.
5. Check that the GPU is visible:
   ```
   python -c "import torch; print(torch.cuda.get_device_name(0))"
   ```

## Running the study

Activate the environment (`venv\Scripts\activate`), then:

```
python dashboard.py
```

1. **Test 1** page -> **Run Test 1** (about 10-20 minutes).
2. **Test 3** page -> **Quick trial (5 photos)** to check it works, then
   **Run Test 3** (about 45-60 minutes).
3. **Field test** (at the PSU gate, with this laptop):
   - Run `python app.py`, open the **Field test** tab, start the camera,
     choose the distance (6-10 m) and press **Space** for each photo of a
     parked vehicle with a sticker, in daylight; about 30 photos per
     distance.
   - Blur faces and plate numbers (keep the file names), then upload
     `research_outputs/field_test/images/` to the Ultralytics Platform
     as the dataset `student-vehicle-sticker-field` (do not rename files) and
     draw a box around every sticker.
   - In the dashboard, **Test 2** page -> **Run Test 2**.
   - Run Test 3 again, so distance is included in the final ranking.
4. **Final Comparison** page shows the full scorecard. **Export PDF report**
   saves everything to `research_outputs/benchmark_report.pdf`.

**Start over (archive results)** in the dashboard moves old results into
`research_outputs/_archive/` (nothing is deleted) so the tests can be run
again from Test 1.

## Notes

- The camera app prefers an external USB camera over the laptop's built-in
  camera. If the wrong camera opens, choose it in the dashboard's **Camera**
  dropdown or run `python app.py --camera N` with the camera index.
- Each script checks the GPU: all three tests expect the RTX 2050, matching
  the thesis. Use `--allow-other-gpu` on the command line only if the thesis
  text is changed to match.
- Speed depends on the laptop's power state. Measure with it plugged in and on
  its highest performance setting, with nothing else using the GPU. Two
  sessions on the same RTX 2050 differed by about 30%, so report a run made
  under these conditions.
- Keep this repository **private**: the ethics section limits access to the
  research team and adviser.