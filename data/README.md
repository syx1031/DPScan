# Dataset

## KNOWN dataset

See known.xlsx

* 1st column, AppID —— Apple Store App ID
* 2nd column, BundleID —— Apple Store App Bundle name
* 3rd-11th columns —— Presence of corresponding ads dark patterns in the app (manually annotated).
  * Value: 1: presented; Blank: Absent.

## UNKNOWN dataset

See unknown.xlsx

* 1st column, appId: Apple Store App ID
* 2nd column, BundleID: Apple Store App Bundle name

## RAG Repository

A subset of exploration traces was extracted from AdsDP and annotated. see **db/**. Structure of this repository is as follows:

```
db/
├── test_1/                   # RAG Repository
    └── {appid}-{timestamp}/  # One sequence per folder
        ├── 0.png             # UI 1
        ├── 1.png             # UI 2
        ... ...				  # UI N
        ├── instance.json     # Annotation file, including types of dark patterns, semantic categories of UIs and elements, and auditor's interaction, et al.
        ├── 0/                # One UI per folder
        │   ├── element_0.png # Element 0 in UI 0
        │   ├── element_1.png # Element 1 in UI 0
        │   ... ...
        └── 1/				  # Elements in UI 1
        	... ...
```

Due to size constraints, we only provide a subset here. All traces will be publicly released soon.