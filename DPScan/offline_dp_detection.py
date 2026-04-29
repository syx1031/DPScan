STATUS_BAR_PIXELS = 80

import base64
import os
import json
import time
from pathlib import Path

from gpt_cls import DarkPatternRecording, UIRecording, ActionRecording, SemanticUserActionStep, UserActionSummary, UserSubTask, UserSubTaskTypeExploreApp, DPList
from app_window import App
import app_window as running_window

### Logging configuration
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='[%(asctime)s] %(levelname)s [%(name)s:%(lineno)s] %(message)s',
    handlers=[
        logging.FileHandler('offline.log', mode='a', encoding='utf-8'),  # Logs to file
        logging.StreamHandler()  # Logs to console
    ]
)
logger = logging.getLogger(__name__)

from gpt_cls import SemanticUserAction

from ui_cls import AppiumUI

from action_cls import BaseAction

app, root = None, None

from LLMRAG import LLMRAG
rag_db = LLMRAG(database_dir="db/test_1")
# rag_db = None


def app_window():
    global app, root
    root, app = App.create_window(((lambda: None, lambda: None), lambda: None, lambda: None, AppiumUI, BaseAction), rag_db)
    app.running = True
    app.is_updating = True

def fill_ui_data(uirecording: UIRecording):
    app.is_fetching_ui = True
    try:
        app.uist = app.org_uist = app.uist_before_supply = uist = {'elements': LLMRAG.rebuild_viewhierarchy(uirecording), 'screenscale': 2}
        app.org_vid_map = app.vid_map = app.vid_map_before_supply = vid_map = running_window.base_ui.get_vid_map(uist['elements'])
        base64_screenshot = uirecording.screenshot

        app.status_label.config(text="Status: Analyzing UI")

        analyzed = running_window.base_ui.analyze_ui(uist['elements'], base64_screenshot)
        semantic_analyzed = running_window.base_ui.semantic_analyze_ui(uist['elements'], base64_screenshot, rag=None)
        rag_analyzed = app.rag_fine_grain(semantic_analyzed, base64_screenshot, uist, vid_map)

        app.ui_graph.add_ui(uist['elements'], base64_screenshot, analyzed, semantic_analyzed, description='[fetch_ui_data] UI synchronizing successfully.')

        # Update & Cache the screenshot image data
        app.last_screenshot = base64.b64decode(base64_screenshot)

        app.analyzed_ui = analyzed
        app.semantic_analyzed = semantic_analyzed
        app.rag_analyze = rag_analyzed

    except Exception as e:
        logger.exception(f"Error loading UI screenshot")
        app.status_label.config(text="Status: Error")

    app.is_fetching_ui = False


def fill_action_data(action: ActionRecording):
    action_description = f"Action Type: {action.type}"
    if action.location:
        action_description += f"\n Target Element Location: {action.location}"
    if action.text:
        action_description += f"\n Text in Target Element: {action.text}"
    if action.description:
        action_description += f"\n Other Description: {action.description if action.description else 'Nothing'}"

    try:
        app.status_label.config(text="Status: Analyzing Action")

        app.actions = SemanticUserAction(
            actions=[
                SemanticUserActionStep(reasoning="", target_element_id=-1, reason_of_element="",
                                       completeness_check="",
                                       action=action.convert_gui_to_chameleoscan())
                ],
            summary=UserActionSummary(ui_summary="",
                                      action_summary=action_description),
            past_action_summary="",
            recent_repeat_action="",
            recent_repeat_count=0,
            page_semantic="",
            subtask_candidate=[],
            # random_exploration=False,
            subtask=UserSubTask(sub_task=UserSubTaskTypeExploreApp(task="explore_app")),
            reasoning="",
        )

    except Exception as e:
        logger.exception(f"Error filling action data")
        app.status_label.config(text="Status: Error")

    app.is_applying_action = True

    app.ui_graph.add_action(app.actions, description="[apply_action] Apply actions.")
    app.action_history.append(app.actions.summary.ui_summary + (
            ' ' if not app.analyzed_ui or not app.analyzed_ui.feedback_message or (
                        app.analyzed_ui.ads and app.analyzed_ui.ads.is_ad_topmost) or app.analyzed_ui.feedback_message in app.actions.summary.ui_summary else ' The UI shows feedback/status/error message: \'' + app.analyzed_ui.feedback_message + '\'. ') + app.actions.summary.action_summary)
    app._actions_history.append(app.actions)

    app.is_applying_action = False


def dp_detection():
    ui_action_history = app.history_record()
    ui_sequence = []

    # start_index = max(0, len(ui_action_history) - 5)
    for ui_index, ui_or_action in enumerate(ui_action_history): # start=start_index
        if isinstance(ui_or_action, app.UiGraph.GraphNode):
            semantic_analyzed = ui_or_action.semantic_analyzed[ui_or_action.global_local_map[ui_index]]
            rag_analyzed = ui_or_action.rag_analyze[ui_or_action.global_local_map[ui_index]]
            analysis_ids = set([e.id for e in semantic_analyzed.semantic_element] + [e.id for e in
                                                                                     rag_analyzed.rag_elements.element_groups] if rag_analyzed and rag_analyzed.rag_elements else [])
            analyze = ""
            for elem_id in analysis_ids:
                elem_sem = next((eg for eg in semantic_analyzed.semantic_element if eg.id == elem_id), None)
                elem_rag = next((eg for eg in rag_analyzed.rag_elements.element_groups if eg.id == elem_id), None)
                if elem_rag:
                    analyze += f"Semantic Type: {elem_rag.type}, Reasoning: {elem_rag.evidence} \n"
                elif elem_sem:
                    analyze += f"Semantic Type: {elem_sem.type}, Reasoning: {elem_sem.evidence} \n"

            ui_sequence.append({
                "type": "ui",
                "image": ui_or_action.screenshot[ui_or_action.global_local_map[ui_index]],
                "analyze": analyze,
            })
        else:
            ui_sequence.append({
                "type": "action",
                "description": ui_or_action.actions.summary.action_summary,
            })

    dplist = running_window.base_ui.dp_detection(ui_sequence)
    app.dp_result = dplist
    app.dp_label.config(text=f"DP Detection: Done")
    app.dp_thread = None


def collect_result(dprecord: DarkPatternRecording):
    def bbox_to_absolute_frame(bbox: tuple[int, int, int, int]) -> dict:
        x1, y1, x2, y2 = bbox
        return {'x': x1, 'y': y1, 'width': x2 - x1, 'height': y2 - y1}

    ui_action_history = app.history_record()

    element_classify = []

    idx = 0
    for groundtruth_ui, detect_ui in zip(dprecord.ui_action_sequence, ui_action_history):
        if isinstance(groundtruth_ui, UIRecording):
            analyzed_ui = detect_ui.analyzed[detect_ui.global_local_map[idx]]
            semantic_analyzed = detect_ui.semantic_analyzed[detect_ui.global_local_map[idx]]
            ui_tree = detect_ui.ui_tree[detect_ui.global_local_map[idx]]

            id_element_map = {}
            for element in groundtruth_ui.uielements:
                bbox = element.bbox
                absolute_frame = bbox_to_absolute_frame(bbox)
                view = running_window.base_ui.find_element_by_frame(ui_tree, absolute_frame)
                target_id = view.get('id', -1)
                assert target_id >= 0

                id_element_map[target_id] = {'gt_element_type': element.ui_type}

            for semantic_element in semantic_analyzed.semantic_element:
                element_type = semantic_element.type
                element_id = semantic_element.id
                if element_id in id_element_map:
                    id_element_map[element_id]['element_type'] = element_type
                else:
                    id_element_map[element_id] = {'element_type': element_type}

            id_element_map[0] = {'gt_element_type': groundtruth_ui.ui_type, 'element_type': semantic_analyzed.semantic_interface.type}

            element_classify.extend([item for item in id_element_map.values()])

        idx += 1

    dp_classify = {'gt_dp_type': dprecord.dark_patterns, 'dp_type': [dp.dptype for dp in app.dp_result.dplist]}
    if not dp_classify['dp_type']:
        dp_classify['dp_type'] = ["No Dark Pattern"]

    return element_classify, dp_classify

if __name__ == '__main__':
    SAVE_DIR_ROOTS = ["cached/evaluations/offline"]
    ROOT_DIRS_LIST = ['db/offline']
    # SAVE_DIR_ROOTS = ['cached/evaluations/cxm_db_comparison']
    # ROOT_DIRS_LIST = ['db/cxm_db_comparison']
    root_dirs = [Path(root_dir) for root_dir in ROOT_DIRS_LIST]
    CHECKPOINT = "checkpoint.json"

    recordings = []

    elements = []
    dps = []

    def step_callback(base64_screenshot: str, org_uist: dict, uist: dict, analyzed_ui: object,
                      semantic_analyzed: object, actions: object, dp_result: object,
                      action_history: list[str],
                      _actions_history: list[object], skip_steps_by_preprocess: int,
                      ui_graph: object, similar_elements: dict,
                      similar_ui: list):
        global step_time, step_count
        logger.info(f"Step {step_count} for {APP_IDENT} at {time.time()}")
        with open(f'{SAVE_DIR}/step_{step_count}.json', 'w', encoding='utf-8') as f:
            import json
            json.dump({
                'time': time.time() - step_time,
                'task': '',
                'base64_screenshot': base64_screenshot,
                'org_uist': org_uist,
                'uist': uist,
                'analyzed_ui': analyzed_ui.model_dump(mode='json') if analyzed_ui else None,
                'semantic_analyzed': semantic_analyzed.model_dump(
                    mode='json') if semantic_analyzed else None,
                'actions': actions.model_dump(mode='json') if actions else None,
                'dp_result': dp_result.model_dump(mode='json') if dp_result else None,
                'action_history': action_history if action_history else [],
                'ui_graph': ui_graph.model_dump(mode='json'),
                'similar_elements': similar_elements,
                'similar_ui': similar_ui,
            }, f, indent=0)
        step_time = time.time()

    for SAVE_DIR_ROOT, root_dir in zip(SAVE_DIR_ROOTS, root_dirs):
        done_list = {}
        if os.path.exists(os.path.join(SAVE_DIR_ROOT, CHECKPOINT)):
            with open(os.path.join(SAVE_DIR_ROOT, CHECKPOINT), 'r') as f:
                done_list = json.load(f)

        subdirs = [p for p in root_dir.iterdir() if p.is_dir()]
        for inst_idx, subdir in enumerate(subdirs):
            print(f'Processing {str(inst_idx)}/{str(len(subdirs) - 1)}: {subdir.name}')

            try:

                if subdir.is_dir():
                    instance_path = subdir / "instance_revision.json"
                    if not instance_path.exists():
                        instance_path = subdir / "instance.json"
                    if instance_path.exists():
                        parts = subdir.name.split("-")
                        APP_IDENT = parts[0]
                        TIMESTAMP = parts[-1] if len(parts) == 2 else parts[2]

                        rec = json.loads(instance_path.read_text(encoding="utf-8"))
                        for i, dp in enumerate(rec['dark_patterns']):
                            if dp == "AdClosureFailure":
                                rec['dark_patterns'][i] = "Ad Closure Failure"
                            elif dp == "RewardBasedAds":
                                rec['dark_patterns'][i] = "Reward-Based Ads"
                            elif dp == "UnexpectedFullScreenAds":
                                rec['dark_patterns'][i] = "Unexpected Full-Screen Ads"
                            elif dp == "PaidAdRemoval":
                                rec['dark_patterns'][i] = "Paid Ads Removal"
                            elif dp == "AutoRedirectAds":
                                rec['dark_patterns'][i] = "Auto-Redirect Ads"
                            elif dp == "AppResumptionAds":
                                rec['dark_patterns'][i] = "App Resumption Ads"
                            elif dp == "GestureInducedAdRedirection":
                                rec['dark_patterns'][i] = "Gesture-Induced Ad Redirection"
                            elif dp == "MultipleCloseButtons":
                                rec['dark_patterns'][i] = "Multiple Close Buttons"
                            elif dp == "AdWithoutExitOption":
                                rec['dark_patterns'][i] = "Ad without Exit Option"
                            elif dp == "BiasDrivenAd":
                                rec['dark_patterns'][i] = "Bias-Driven Ads"
                            elif dp == "DisguisedAd":
                                rec['dark_patterns'][i] = "Disguised Ads"
                            elif dp == "NoDarkpattern":
                                rec['dark_patterns'][i] = "No Dark Pattern"

                        dprecord = DarkPatternRecording.model_validate(rec)
                        recordings.append(dprecord)


                        load_history = False
                        IDENTIFIER = APP_IDENT + '-' + TIMESTAMP
                        if IDENTIFIER in done_list:
                            load_history = True
                            SAVE_DIR = done_list[IDENTIFIER]

                            if SAVE_DIR == "ignore":
                                continue
                        else:
                            SAVE_DIR = os.path.join(SAVE_DIR_ROOT, str(int(time.time())) + '-' + APP_IDENT + '-' + TIMESTAMP)
                            os.makedirs(SAVE_DIR, exist_ok=True)

                        app_window()

                        if not load_history:
                            step_count = 0
                            for ui_or_action in dprecord.ui_action_sequence:
                                if isinstance(ui_or_action, UIRecording):
                                    fill_ui_data(ui_or_action)
                                else:
                                    fill_action_data(ui_or_action)

                                    step_count += 1
                                    step_time = time.time()
                                    step_callback(base64.b64encode(app.last_screenshot).decode('utf-8'), app.org_uist, app.uist,
                                                  app.analyzed_ui, app.semantic_analyzed, app.actions, None,
                                                  app.action_history, app._actions_history, app.skip_steps_by_preprocess,
                                                  app.ui_graph, app.similar_elements, app.similar_ui)

                            dp_detection()

                            step_count += 1
                            step_time = time.time()
                            step_callback(base64.b64encode(app.last_screenshot).decode('utf-8'), app.org_uist, app.uist,
                                          app.analyzed_ui, app.semantic_analyzed, None, app.dp_result,
                                          app.action_history, app._actions_history, app.skip_steps_by_preprocess,
                                          app.ui_graph, app.similar_elements, app.similar_ui)

                            done_list[IDENTIFIER] = SAVE_DIR
                            with open(os.path.join(SAVE_DIR_ROOT, CHECKPOINT), 'w') as f:
                                json.dump(done_list, f, indent=4)
                        else:
                            step_count = int((len(dprecord.ui_action_sequence) + 1) / 2)
                            with open(f'{SAVE_DIR}/step_{step_count}.json', 'r', encoding='utf-8') as f:
                                data = json.load(f)
                                app.ui_graph = running_window.App.UiGraph.model_validate(data['ui_graph'])
                                app.dp_result = DPList.model_validate(data['dp_result']) if data['dp_result'] else DPList(dplist=[])

                        element, dp = collect_result(dprecord)
                        elements.extend(element)
                        dps.append(dp)

                        print(f"Results of {APP_IDENT} - {TIMESTAMP} in directory {SAVE_DIR}")

                        gt_set = set(dp['gt_dp_type'])
                        dt_set = set(dp['dp_type'])

                        missing = gt_set - dt_set
                        for item in missing:
                            print(f"FN: '{item}'")

                        extra = dt_set - gt_set
                        for item in extra:
                            print(f"FP: '{item}'")

            except Exception as e:
                print(f'Error when processing the instance {subdir.name}: {e}')

    from collections import defaultdict
    import numpy as np

    NONE = "__NONE__"

    y_true = []
    y_pred = []

    for item in elements:
        y_true.append(item.get('gt_element_type', NONE))
        y_pred.append(item.get('element_type', NONE))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    classes = sorted(set(y_true) | set(y_pred))

    # overall accuracy
    overall_accuracy = (y_true == y_pred).mean()

    # per-class
    metrics = {}

    for c in classes:
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))

        p = tp / (tp + fp) if tp + fp > 0 else 0.0
        r = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * p * r / (p + r) if p + r > 0 else 0.0

        metrics[c] = {
            "precision": p,
            "recall": r,
            "f1": f1,
            "support": int(tp + fn)
        }

    print("Element Accuracy:", overall_accuracy)
    print(metrics)


    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)

    subset_correct = 0
    jaccard_sum = 0
    n = len(dps)

    all_labels = set()

    for item in dps:
        gt = set(item.get('gt_dp_type', ["No Dark Pattern"]))
        pred = set(item.get('dp_type', ["No Dark Pattern"]))

        all_labels |= gt | pred

        # subset accuracy
        if gt == pred:
            subset_correct += 1

        # jaccard
        if gt or pred:
            jaccard_sum += len(gt & pred) / len(gt | pred)

        for label in gt & pred:
            tp[label] += 1
        for label in pred - gt:
            fp[label] += 1
        for label in gt - pred:
            fn[label] += 1

    # ---------- Overall metrics ----------
    TP = sum(tp.values())
    FP = sum(fp.values())
    FN = sum(fn.values())

    micro_precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    micro_recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) \
        if (micro_precision + micro_recall) > 0 else 0.0

    subset_accuracy = subset_correct / n
    jaccard_accuracy = jaccard_sum / n

    # ---------- Per-class metrics ----------
    dp_per_class = {}

    for label in all_labels:
        p = tp[label] / (tp[label] + fp[label]) if (tp[label] + fp[label]) > 0 else 0.0
        r = tp[label] / (tp[label] + fn[label]) if (tp[label] + fn[label]) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

        dp_per_class[label] = {
            'precision': p,
            'recall': r,
            'f1': f,
            'support': tp[label] + fn[label]
        }

    # ---------- Macro metrics (DP) ----------
    macro_dp_precision = np.mean(
        [v['precision'] for v in dp_per_class.values()]
    )
    macro_dp_recall = np.mean(
        [v['recall'] for v in dp_per_class.values()]
    )
    macro_dp_f1 = np.mean(
        [v['f1'] for v in dp_per_class.values()]
    )

    print("DP Macro Precision:", macro_dp_precision)
    print("DP Macro Recall:", macro_dp_recall)
    print("DP Macro F1:", macro_dp_f1)

    print("DP Micro Precision:", micro_precision)
    print("DP Micro Recall:", micro_recall)
    print("DP Micro F1:", micro_f1)
    print("DP Subset Accuracy:", subset_accuracy)
    print("DP Jaccard Accuracy:", jaccard_accuracy)
    print("DP Per-Class Metrics:")
    from pprint import pprint
    pprint(dp_per_class)
