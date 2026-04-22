# Model Submission

This submission contains the model package and a lightweight smoke test script for local verification.

## Contributors

This work was jointly completed by **Huaguan Chen** (<huaguanchen@ruc.edu.cn>), a **Ph.D. student** at Renmin University of China, and **Ning Lin** (<ninglin00@outlook.com>), a **Master’s student** at Renmin University of China.

## Directory structure

```text
.
├── models/
│   ├── __init__.py
│   └── aero_chrono_mixer/
│       ├── __init__.py
│       ├── model.py
│       └── state_dict.pt
├── README.md
└── smoke_test_submission.py
```

## Contents

- `models/aero_chrono_mixer/model.py`: model definition and inference-related code.
- `models/aero_chrono_mixer/state_dict.pt`: pretrained model weights.
- `smoke_test_submission.py`: a lightweight smoke test script for local verification before submission or evaluation.
- `models/__init__.py` and `models/aero_chrono_mixer/__init__.py`: package markers for import.

## Checkpoint loading

The model package already includes the weight-loading logic in `models/aero_chrono_mixer/model.py`, so no extra code is required to load the pretrained weights again. Please refer to **lines 639-643** of `model.py` for the specific implementation.

## Smoke test

A smoke test script is provided as `smoke_test_submission.py` for basic local sanity checking. It can be used to verify that the submitted package can be imported and initialized correctly in the expected submission layout.

## Notes

- Please keep the relative path between `model.py` and `state_dict.pt` unchanged.
- Please keep the package structure unchanged.
