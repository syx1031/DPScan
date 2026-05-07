# DPScan

A system for knowledge-augmented LLM-driven exploration and detection of dark patterns in mobile ads

## Structure

```
DPScan/
├── app.py                    # Main program for DPScan
├── offline_dp_detection.py   # Main program for running Detection Module
├── download_history.py       # IPA download script
├── purchase_server.py        # iTunes authentication service
├── requirements.txt          # Python dependencies
├── *.bat                     # Batch scripts
├── *_cls.py                  # Several Functional modules
├── zxtouch/                  # ZXTouch library for clicking action
├── lib/                      # Front-end library files
├── resources_for_run/
│	├── app_list.csv          # Apps that need to be downloaded
│   └── keys.py               # API key configuration
├── cached/                   # IPA cache directory (generated at runtime)
│	└── results               # Output directory of DPScan (generated at runtime)
├── db/                       # RAG repository
│	├── normal_elements       # Elements irrelevant to dark pattern
│	└── test_1                # RAG respository recording UI, UI elements, and auditors' interactions
└── output/                   # Output directory for RAG (generated at runtime)
```

## Files you may concern:

### Core components

| File            | Method                                   | Description in Paper                                         |
| --------------- | ---------------------------------------- | ------------------------------------------------------------ |
| `db/test_1`     | /                                        | RAG repository, in which UI, elements, and UI interactions are organized in Exploration Traces **(Sec IV.C.1)** |
| `LLMRAG.py`     | query_ui                                 | UI retrieval **(Sec IV.C.2.1)**                              |
| `LLMRAG.py`     | ui_dp_match                              | UI similarity calculation **(Sec IV.C.2.1)**                 |
| `LLMRAG.py`     | query_element                            | Element retrieval **(Sec IV.C.2.2)**                         |
| `LLMRAG.py`     | query_next_action                        | UI Interaction retrieval **(Sec IV.C.2.3)**                  |
| `LLMRAG.py`     | prune_viewhierarchy                      | Tree pruning **(Sec IV.C.2.3)**                              |
| `gpt_cls.py`    | GeminiClient.semantic_analyze_action     | Check planning and action formulation **(Sec IV.D)**         |
| `app_window.py` | sync_ui_screenshot                       | UI acquisition from iOS instrumentation agent **(Sec IV.E.1)** |
| `ui_cls.py`     | tree_supply                              | Visual-structural alignment **(Sec IV.E.1)**                 |
| `gpt_cls.py`    | GeminiClient.semantic_analyze_screenshot | UI semantic Interpretation **(Sec IV.E.2)**                  |
| `gpt_cls.py`    | GeminiClient.detect_dp                   | Dark pattern identification **(Sec IV.E.3)**                 |

### RAG Repository Structure

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

### Python Files

| File                      | Function                                                     |
| ------------------------- | ------------------------------------------------------------ |
| `app.py`                  | Main program for evaluating the complete DPScan workflow     |
| `offline_dp_detection.py` | Main program for evaluating only the Detection Module        |
| `benchmarking.py`         | Program for visualizing results                              |
| `LLMRAG.py`               | RAG-assisted module (similar UI, element, and UI interaction retrieval) |
| `ui_cls.py`               | UI process (crop et al.) module                              |
| `gpt_cls.py`              | Data structure definitions                                   |
| `action_cls.py`           | Action execution modules                                     |
| `paddle_cls.py`           | PaddleOCR text recognition                                   |
| `frida_cls.py`            | Frida connection module                                      |
| `appium_cls.py`           | Appium connection module                                     |
| `ssh_cls.py`              | SSH connection module                                        |
| `zxt_cls.py`              | ZXTouch module for clicking                                  |

### Batch Scripts

| File           | Function                                         |
| -------------- | ------------------------------------------------ |
| `tiproxy.bat`  | Start tidevice port forwarding for SSH/Frida/WDA |
| `xcuitest.bat` | Start WebDriverAgent (XCUITest) in iPhone        |
| `appium.bat`   | Start Appium Server                              |

### Task Configuration Files

| File                  | Function                              |
| --------------------- | ------------------------------------- |
| `BATCH_TASKS.txt`     | List of apps to be tested (Bundle ID) |
| `BATCH_TASKS.fin.txt` | List of completed apps                |

## Running

### Running Mode

#### 1. Exploration-and-Detection (`app.py`)

Connects to an iOS device to install apps, explore UIs, and detect dark patterns in real time.

**Workflow**:

1. Connect to a **jailbroken** iOS device
2. Install the target app (from an IPA file in `cached/` or the App Store)
3. Run `app.py`
4. Save results to `cached/results`
5. Visualize your results by running `benchmarking.py`

#### 2. Only Detection (`offline_dp_detection.py`)

Evaluates the **accuracy of the Module Detection** on **exploration traces** extracted from **AdsDP**.

**Workflow**:

1. Prepare some exploration traces in `db/offline`
2. Run `offline_dp_detection.py`
3. Save results to `cached/evaluations/`

### Prerequisites for running

**Notice**: If you **do not want** to **test apps** on an iPhone and only wish to **visualize the system’s outputs**, or run the system in **"Only Detection" mode** on traces, please see **(2) Configuration for Only Detection Mode or Result Visualization** below, which provides a minimal runnable version of the system setup.

We prepared sample data to help you test certain functionalities.

1. IPA files in `cached/`, run `app.py` to test it.
2. One recording of exploring UIs and identifying dark patterns in `cached/results`, you can visualize this by running `benchmarking.py`
3. Exploration traces in `db/offline` , use these to run `offline_dp_detection.py`
4. Three traces in `db/test_1` and four elements irrelevant to dark patterns in `db/normal_elements`, which form a small RAG repository that can be loaded by `LLMRAG.py`

#### 1. Configuration for Exploration-and-Detection (see also [ChameleoScan](https://github.com/ChameleoScan/framework))

##### (1) Hardware:

1. A **Jailbroken** **iPhone** device with **iOS version 14.x** (not necessarily)
2. A **macOS** PC to build **WebAgentDriver**
3. A preferably **Windows** PC to **run the framework**

##### (2) iPhone device setup:

1. **Install** **Cydia** and **AppSync** to install decrypted IPA files. (Or **login** your **Apple ID** instead.)
2. **Install** **OpenSSH**, **Frida**, **AFC2**, and **ZXTouch** tweaks (formerly **IOS13-SimulateTouch**).
3. **Build and deploy** **WebAgentDriver** (as **Appium XCUITest Driver**).
4. **Connect** it to the **Host** **PC**.

##### (3) Host PC setup (Windows as an example):

1. **Install** **Python 3.11+**, **Node.js 22+**.
2. **Install** **iTunes** or **Apple Mobile Device Service**, and make sure your iPhone device is recognizable.
3. Set up a Python **Virtual Environment** (preferably) and install `requirements.txt`.
4. Install UIDE's requirement in `LayoutCoder_develop/requirements.txt`
5. Set up **Appium** by installing **dependencies** in `package.json`.
6. Refer to [issue1](https://github.com/alibaba/tidevice/issues/277#issuecomment-1757065698) and [issue2](https://github.com/alibaba/tidevice/issues/377) to **fix tidevice** XCUITest startup issues.
7. **Replace** the default bundle id `com.1xample.WebDriverAgentRunner.xctrunner` in `xcuitest.bat` into your manually deployed one.
8. **Replace** `DEVICE_VERSION`, `DEVICE_UDID` (maybe include `STATUS_BAR_PIXELS` indicating the status bar height) in `app.py` into the correct value of your device. If the connection still cannot be established, consider checking and modifying `REMOTE_IP` in `app.py`. By default, the iPhone address should be set to `'127.0.0.1'`.

##### (4) Configure your Gemini keys:

​	Add your **keys** to the `GeminiKey` list in `resources_for_run/keys.py`

##### (5) IPA Download (`download_history.py`):

Before testing an app, you need to **download it** from iTunes. For convenience, we also provide **a pre-downloaded example IPA** in `cached/`, which can be used to **run `app.py` directly**.

1. Install **IPATool**

   Download **the executable** for your platform from [ipatool Releases](https://github.com/majd/ipatool/releases) and add it to your system PATH environment variable.
    (For example, if downloaded to `"D:\ipatool\bin\ipatool-2.3.0-windows-amd64.exe"`, add `D:\ipatool\bin` to the environment variables). Then rename `ipatool-2.3.0-windows-amd64.exe` to `ipatool.exe`. Verify installation:

   ```
   ipatool --version
   ```

2. Install and patch **iTunes** (See also [IPATool-py](https://github.com/NyaMisty/ipatool-py?tab=readme-ov-file#download-old-version))

   Install the [Windows version of iTunes](https://secure-appldnld.apple.com/itunes12/091-87819-20180912-69177170-B085-11E8-B6AB-C1D03409AD2A6/iTunes64Setup.exe), patch iTunes using `actions-iTunes-header/workflow_helper/iTunesInstall/patch_itunes.py`, and ensure that the iTunes process can be attached by Frida.

3. Log in to IPATool

​	Log in to ipatool using your Apple ID (2FA verification is required for the first login; check for the verification code):

```
ipatool auth login -e "your_email" -p "your_password" --keychain-passphrase "123456"
```

​	Verify login status:

```
ipatool auth info --keychain-passphrase "123456"
```

4. Log in to iTunes

   1. Open the iTunes application

   2. Go to Account → Sign In

   3. Log in using the same Apple ID

5. Start `purchase_server.py`

​	This service hooks into iTunes via Frida to obtain authentication information and listens on port 9100:

```
python actions_iTunes_header_master/workflow_helper/iTunesDownload/get_header.py
```

​	Keep this terminal window running.

6. Run the download script

​	Open a new terminal window and run:

```
python download_history.py
```

​	The following configurations can be modified in `download_history.py`:

​	1. passphrase`: ipatool keychain passphrase (default `"123456"`), make sure same as the passphrase in **Log in to IPATool**

​	2. `cached`: directory for saving IPA files, by default `cached/`

​	3. `read_appid_from_csv()`: the CSV file path for download application list

**Some Common Issues**

| Issue                                     | Solution                                                     |
| ----------------------------------------- | ------------------------------------------------------------ |
| `FileNotFoundError: ipatool`              | Ensure ipatool is added to PATH                              |
| `aes.KeyUnwrap(): integrity check failed` | Re-run `ipatool auth login`                                  |
| `MZCommerce.ASN.MissingPasswordToken`     | Re-login to Apple ID in iTunes                               |
| `2FA code is required`                    | Add the `--auth-code` parameter and provide the verification code |

#### 2. Configuration for Only Detection Mode or Results Visualization

##### (1) Hardware:

​	A preferably **Windows** PC to **run the framework**

##### (2) PC setup (Windows as an example):

1. **Install** **Python 3.11+**, **Node.js 22+**.
2. Set up a Python **Virtual Environment** (preferably) and install `requirements.txt`.
3. Install UIDE's requirement in `LayoutCoder_develop/requirements.txt`

##### (3) Configure your Gemini keys:

​	Add your **API keys** to the `GeminiKey` list in `resources_for_run/keys.py`

### Running the framework

#### 1. Exploration-and-Detection Mode

1. Put your **IPA file** into the `cached/` directory.
2. Put **Bundle IDs** into `BATCH_TASKS.txt`.
3. **Run** `appium.bat`, `tiproxy.bat`, `xcuitest.bat`.
4. **Run** `app.py` to start exploration and detection.
5. **Run** `benchmarking.py` to view your running results at any time.

#### 2. Only Detection Mode

1. Put your **exploration traces** into `db/offline`.
2. **Run** `offline_dp_detection.py` to start detection.

