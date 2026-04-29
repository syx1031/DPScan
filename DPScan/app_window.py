import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import io
import threading
import time
import queue
import json
import base64
import copy
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from pyvis.network import Network
import webbrowser

# from selenium.webdriver.support.expected_conditions import element_selection_state_to_be

from gpt_cls import UserActionTypeClick, UserActionTypeHover, UserActionTypeSwitch, \
    UserActionTypeRestart, UserActionTypeIdle, UIView, SemanticView, SemanticUserAction, UserActionSummary, \
    UserSubTask, SemanticUserActionStep, TaskCandidate, UserSubTaskTypeSwitch, \
    UserSubTaskTypeExploreApp, UserSubTaskTypeRemoveAd, UserSubTaskTypeNonFullScreenAd, UserSubTaskTypeObserveCloseFullscreenAd, \
    UIElementRecording, UIRecording, SemanticUIElementType, ActionRecording, RAGView
# from RAG_assist_action import RAG

import logging
logger = logging.getLogger(__name__)

from utils import *

MAX_UI_DIFF_COUNT = 3
# MAX_NO_ACTION_COUNT = 3
MAX_NO_ACTION_COUNT = 10
# MAX_STEPS_COUNT = 12
MAX_STEPS_COUNT = 100

result_callback, step_callback = None, None
get_screenshot, mask_detection, base_ui, base_action = None, None, None, None

class App:
    class UiGraph(BaseModel):
        class GraphNode(BaseModel):
            id: int = Field(...)
            ui_tree: List[dict] = Field(default_factory=list)
            screenshot: List[str] = Field(default_factory=list)
            analyzed: List[UIView|None] = Field(default_factory=list)
            semantic_analyzed: List[SemanticView|None] = Field(default_factory=list)
            rag_analyze: Optional[List[RAGView|None]] = Field(default_factory=list)
            last_visit: float = Field(default=0.0)
            start_time: float = Field(default_factory=time.time)
            visit_times: List[float] = Field(default_factory=list)
            start_edge: List[object] = Field(default_factory=list)
            end_edge: List[object] = Field(default_factory=list)
            # tasks: List[Tuple[UserSubTask, TaskStatus]] = Field(default_factory=list)

            global_local_map: Dict[int, int] = Field(default_factory=dict)
            local_global_map: Dict[int, int] = Field(default_factory=dict)

            # Pydantic v2 安全初始化逻辑
            def model_post_init(self, __context=None):
                # 初始化 last_visit 和 visit_times
                # self.last_visit = time.time() - self.start_time
                # self.visit_times = [self.last_visit]
                pass

            def visit(self, global_step, elements, screenshot, analyzed, semantic_analyzed, rag_analyze):
                """记录一次访问"""
                self.global_local_map[global_step] = len(self.visit_times)
                self.local_global_map[len(self.visit_times)] = global_step

                self.last_visit = time.time()
                self.visit_times.append(self.last_visit - self.start_time)
                self.ui_tree.append(elements)
                self.screenshot.append(screenshot)
                self.analyzed.append(analyzed)
                self.semantic_analyzed.append(semantic_analyzed)
                self.rag_analyze.append(rag_analyze)

            def get_semantic_analyzed(self, index):
                if index >= 0:
                    return self.semantic_analyzed[self.global_local_map[index]]
                elif index == -1:
                    return self.semantic_analyzed[-1]
                raise Exception("Available Index should be >= -1.")


        class GraphEdge(BaseModel):
            id: int = Field(...)
            from_node: int = Field(...)
            to_node: int = Field(...)
            actions: Optional[SemanticUserAction] = Field(default=None)
            description: str = Field(default="")
            # establish_time: float = Field(default=0.0)
            start_time: float = Field(default_factory=time.time)  # 用于初始化计算 establish_time

            global_step: int = Field(default=-1)

            def model_post_init(self, __context=None):
                # 根据 start_time 计算建立时间
                # self.establish_time = time.time() - self.start_time
                pass

        last_actions: SemanticUserAction = Field(default=None)
        last_node: GraphNode = Field(default=None)
        max_node_id: int = Field(default=0)
        max_edge_id: int = Field(default=0)
        nodes: List[GraphNode] = Field(default_factory=list)
        edges: List[GraphEdge] = Field(default_factory=list)
        start_time: float = Field(default_factory=time.time)
        history_elements: List[GraphNode|GraphEdge] = Field(default_factory=list)

        def extract_action_and_result(self, current_node: GraphNode) -> list[str]:
            if current_node.start_edge:
                desc = ["Previously, the system had explored this interface and performed certain actions on it."]
            else:
                return []

            for e_index, edge in enumerate(current_node.start_edge):
                desc.append(f"Exploration No. {e_index}:")

                current_local_no = current_node.global_local_map[edge.global_step - 1]
                # current_ui_tree = current_node.ui_tree[current_local_no]
                # current_vid_map = base_ui.get_vid_map(current_ui_tree)
                current_semantic_analyzed = current_node.semantic_analyzed[current_local_no]

                desc.append("You previously performed the following actions on this interface:" if edge.actions else
                            "You previously didn't perform any action on this interface.")
                for act_index, action in enumerate(edge.actions):
                    match action:
                        case UserActionTypeClick(type="click", element_id=element_id, times=times):
                            semantic_analyzed_element = next((e for e in current_semantic_analyzed.elements if e.id == element_id), None)
                            # element_in_tree = current_vid_map[element_id]
                            desc.append(f"{act_index}. Action Type: Click, Target element location: {semantic_analyzed_element.location}, Target element description: {semantic_analyzed_element.description}")
                        case UserActionTypeSwitch(type="switch", target=target):
                            if target == "Test App":
                                desc.append(f"{act_index}. Action Type: Switch Interface. Bring the app from background to foreground by Appium.")
                            elif target == "Home Screen":
                                desc.append(f"{act_index}. Action Type: Switch Interface. Return to the home screen.")
                        case UserActionTypeHover(type="hover", element_id=element_id):
                            semantic_analyzed_element = next((e for e in current_semantic_analyzed.elements if e.id == element_id), None)
                            desc.append(f"{act_index}. Action Type: Swipe - A finger taps on the top-left corner of the UI element and then slides to the bottom-right corner before releasing. Target element location: {semantic_analyzed_element.location}, Target element description: {semantic_analyzed_element.description}")
                        case UserActionTypeRestart(type="restart"):
                            desc.append(f"{act_index}. Action Type: Restart - Close, kill and reopen the app process.")
                        case UserActionTypeIdle(type="idle"):
                            desc.append(f"{act_index}. Action Type: Idle - Do nothing on the interface, and wait several seconds.")

                target_node = next((n for n in self.nodes if edge.to_node == n.id), None)
                local_no = target_node.global_local_map[edge.global_step + 1]
                target_semantic_analyzed = target_node.semantic_analyzed[local_no]
                target_screenshot = target_node.screenshot[local_no]
                desc.append(f"Then, you observed the following interface appearing:")
                desc.append(f"{target_semantic_analyzed.description}" if target_semantic_analyzed else "")
                desc.append("data:image/png;base64," + target_screenshot)

            return desc

        # @staticmethod
        # def append_task(task_status, node):
        #     node.tasks.append(task_status)

        def get_history(self):
            return copy.deepcopy(self.history_elements)

        def add_action(self, action, description=""):
            self.last_actions = action

        def add_ui(self, uielements, base64_screenshot, analyzed, semantic_analyzed, description="", rag_analyze=None):
            global base_ui

            for node in self.nodes:
                for ui_tree in node.ui_tree:
                    if base_ui.is_ui_equal(uielements, ui_tree):
                        edge_new = self._add_edge(len(self.history_elements), self.last_node, node, self.last_actions, description)
                        self.history_elements.append(edge_new)
                        node.visit(len(self.history_elements), uielements, base64_screenshot, analyzed, semantic_analyzed, rag_analyze)
                        self.history_elements.append(node)
                        self.last_node = node
                        self.last_actions = SemanticUserAction(
                            actions=[SemanticUserActionStep(reasoning="", target_element_id=-1, reason_of_element="",
                                                            completeness_check="",
                                                            action=UserActionTypeIdle(type="idle"))
                                     ],
                            summary=UserActionSummary(ui_summary="",
                                                      action_summary="[apply_action] This is a default action, indicating that the system did not successfully perform the intended action due to changes in the UI or other reasons."),
                            past_action_summary="",
                            recent_repeat_action="",
                            recent_repeat_count=0,
                            subtask_candidate=[],
                            # random_exploration=False,
                            page_semantic="",
                            subtask=UserSubTask(sub_task=UserSubTaskTypeExploreApp(task="explore_app")),
                            reasoning="",
                        )
                        return

            node_new = self._add_node()
            if self.last_node:
                edge_new = self._add_edge(len(self.history_elements), self.last_node, node_new, self.last_actions, description)
                self.history_elements.append(edge_new)
            node_new.visit(len(self.history_elements), uielements, base64_screenshot, analyzed, semantic_analyzed, rag_analyze)
            self.history_elements.append(node_new)
            self.last_node = node_new
            self.last_actions = SemanticUserAction(
                actions=[SemanticUserActionStep(reasoning="", target_element_id=-1, reason_of_element="",
                                                completeness_check="",
                                                action=UserActionTypeIdle(type="idle"))
                         ],
                summary=UserActionSummary(ui_summary="",
                                          action_summary="[apply_action] This is a default action, indicating that the system did not successfully perform the intended action due to changes in the UI or other reasons."),
                past_action_summary="",
                recent_repeat_action="",
                recent_repeat_count=0,
                subtask_candidate=[],
                # random_exploration=False,
                page_semantic="",
                subtask=UserSubTask(sub_task=UserSubTaskTypeExploreApp(task="explore_app")),
                reasoning="",
            )
            return

        def _add_node(self):
            node_now = self.max_node_id
            new_node = self.GraphNode(start_time=self.start_time, id=node_now)
            self.nodes.append(new_node)
            self.max_node_id += 1
            return new_node

        def _add_edge(self, global_step, from_node, to_node, actions, description):
            edge_now = self.max_edge_id
            new_edge = self.GraphEdge(start_time=self.start_time, id=edge_now, from_node=from_node.id, to_node=to_node.id, actions=actions, description=description, global_step=global_step)
            self.edges.append(new_edge)
            from_node.start_edge.append(new_edge)
            to_node.end_edge.append(new_edge)
            self.max_edge_id += 1
            return new_edge


    @classmethod
    def create_window(cls, bind_vars, rag_db=None):
        """Bind functions to class variables"""
        global get_screenshot, mask_detection, base_ui, base_action, result_callback, step_callback
        get_screenshot, mask_detection, base_ui, base_action = bind_vars[1:]
        result_callback, step_callback = bind_vars[0]
        root = tk.Tk()
        app = cls(root, rag_db)
        return root, app

    def __init__(self, root, rag_db=None):
        self.root = root
        self.root.title("UI Analyzer")
        self.root.geometry("1600x900")

        # Add flag to control thread execution
        self.running = True

        # Control variables
        self.last_restart = 1
        self.ui_diff_count = 0
        self.no_action_count = 0

        # Is the mask detection triggered?
        self.is_triggered = False
        # Is the task completed?
        self.is_completed = False
        
        # Bind window close event
        self.root.wm_protocol("WM_DELETE_WINDOW", self.on_closing)
        self._close_immediately = False
        self.schedule_close_check()

        # Configure grid weights
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=2)
        self.root.grid_columnconfigure(2, weight=1)

        # Create PanedWindow containers
        self.h_paned = ttk.PanedWindow(root, orient=tk.HORIZONTAL)
        self.h_paned.grid(row=0, column=0, columnspan=3, sticky="nsew", padx=5, pady=5)

        # Left frame for screenshot
        self.left_frame = ttk.Frame(self.h_paned, padding=5)
        self.left_frame.grid_rowconfigure(0, weight=1)
        self.left_frame.grid_columnconfigure(0, weight=1)
        self.left_frame.config(width=400)
        self.left_frame.grid_propagate(False)

        # Middle and right container
        self.right_container = ttk.PanedWindow(self.h_paned, orient=tk.HORIZONTAL)

        # Middle frame for tree
        self.middle_frame = ttk.PanedWindow(self.right_container, orient=tk.VERTICAL)
        self.middle_frame.grid_rowconfigure(0, weight=3)
        self.middle_frame.grid_rowconfigure(1, weight=1)
        self.middle_frame.grid_columnconfigure(0, weight=1)
        self.middle_frame.config(width=800)
        self.middle_frame.grid_propagate(False)

        # Right frame for details
        self.right_frame = ttk.PanedWindow(self.right_container, orient=tk.VERTICAL)
        self.right_frame.grid_rowconfigure(0, weight=3)
        self.right_frame.grid_rowconfigure(1, weight=1)
        self.right_frame.grid_columnconfigure(0, weight=1)
        self.right_frame.config(width=400)
        self.right_frame.grid_propagate(False)

        # Add frames to PanedWindows
        self.h_paned.add(self.left_frame, weight=1)
        self.h_paned.add(self.right_container, weight=2)
        
        self.right_container.add(self.middle_frame, weight=2)
        self.right_container.add(self.right_frame, weight=1)

        # Canvas for screenshot
        self.canvas = tk.Canvas(self.left_frame, bg="white", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas.bind("<Button-1>", self.on_canvas_click)

        # Add right-click bindings
        self.canvas.bind("<Button-3>", self.show_canvas_menu)
        
        # Create context menus
        self.canvas_menu = tk.Menu(root, tearoff=0)
        self.element_menu = tk.Menu(root, tearoff=0)
        
        # Initialize last right-click position
        self.last_right_click_pos = None
        self.last_right_click_element = None

        # Tree with scrollbar
        tree_frame = ttk.Frame(self.middle_frame)
        tree_frame.grid_columnconfigure(0, weight=1)
        tree_frame.grid_rowconfigure(0, weight=1)
        
        # Configure tree style
        style = ttk.Style()
        style.configure("Treeview", indent=20)  # Increase indentation
        
        self.tree = ttk.Treeview(tree_frame, selectmode="browse", style="Treeview",
                                show="tree")  # Only show tree column, remove headings
        
        # Configure the tree column to expand but allow horizontal scrolling when needed
        self.tree.column("#0", stretch=True, minwidth=2000)  # Allow column to stretch but maintain minimum width
        
        # Vertical scrollbar
        vsb = ttk.Scrollbar(tree_frame, orient="vertical", command=self.tree.yview)
        # Horizontal scrollbar
        hsb = ttk.Scrollbar(tree_frame, orient="horizontal", command=self.tree.xview)
        
        # Configure both scrollbars
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        
        # Grid layout with both scrollbars
        self.tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        
        # Configure tree_frame grid weights to allow scrollbar to show
        tree_frame.grid_rowconfigure(1, weight=0)  # Don't let horizontal scrollbar stretch
        tree_frame.grid_columnconfigure(1, weight=0)  # Don't let vertical scrollbar stretch
        
        self.tree.bind("<<TreeviewSelect>>", self.on_tree_select)
        self.tree.bind("<Button-3>", self.show_tree_menu)

        # Middle bottom text - wrap in a frame
        middle_bottom_frame = ttk.Frame(self.middle_frame)
        middle_bottom_frame.config(height=200)
        middle_bottom_frame.grid_propagate(False)
        self.step_desc = tk.Text(middle_bottom_frame, wrap=tk.WORD)
        vsb_desc = ttk.Scrollbar(middle_bottom_frame, orient="vertical", command=self.step_desc.yview)
        self.step_desc.configure(yscrollcommand=vsb_desc.set)
        self.step_desc.grid(row=0, column=0, sticky="nsew")
        vsb_desc.grid(row=0, column=1, sticky="ns")
        
        # Configure middle_bottom_frame grid
        middle_bottom_frame.grid_columnconfigure(0, weight=1)
        middle_bottom_frame.grid_rowconfigure(0, weight=1)

        # Add to middle paned window
        self.middle_frame.add(tree_frame, weight=3)
        self.middle_frame.add(middle_bottom_frame, weight=1)

        # Right details text - wrap in a frame
        right_top_frame = ttk.Frame(self.right_frame)
        right_top_frame.grid(row=0, column=0, sticky="nsew")
        # Configure right_top_frame grid
        right_top_frame.rowconfigure(0, weight=1)  # preview_frame (1)
        right_top_frame.rowconfigure(1, weight=3)  # details_text (3)
        right_top_frame.columnconfigure(0, weight=1)

        # ============================
        # 1) Preview Frame（带独立滚动条）
        # ============================
        preview_container = ttk.Frame(right_top_frame)
        preview_container.grid(row=0, column=0, sticky="nsew")

        preview_canvas = tk.Canvas(preview_container)
        preview_vsb = ttk.Scrollbar(preview_container, orient="vertical", command=preview_canvas.yview)
        preview_hsb = ttk.Scrollbar(preview_container, orient="horizontal", command=preview_canvas.xview)
        preview_canvas.configure(yscrollcommand=preview_vsb.set, xscrollcommand=preview_hsb.set)

        preview_canvas.grid(row=0, column=0, sticky="nsew")
        preview_vsb.grid(row=0, column=1, sticky="ns")
        preview_hsb.grid(row=1, column=0, sticky="ew")

        preview_container.rowconfigure(0, weight=1)
        preview_container.columnconfigure(0, weight=1)

        self.preview_frame = ttk.Frame(preview_canvas)
        preview_canvas.create_window((0, 0), window=self.preview_frame, anchor="nw")

        # 绑定更新 scrollregion
        def update_preview_scrollregion(event):
            preview_canvas.configure(scrollregion=preview_canvas.bbox("all"))

        self.preview_frame.bind("<Configure>", update_preview_scrollregion)
        self._preview_image = []

        # ============================
        # 2) Details Text（带独立滚动条）
        # ============================
        details_container = ttk.Frame(right_top_frame)
        details_container.grid(row=1, column=0, sticky="nsew")

        self.details_text = tk.Text(details_container, wrap=tk.WORD)
        details_vsb = ttk.Scrollbar(details_container, orient="vertical", command=self.details_text.yview)
        details_hsb = ttk.Scrollbar(details_container, orient="horizontal", command=self.details_text.xview)

        self.details_text.configure(yscrollcommand=details_vsb.set, xscrollcommand=details_hsb.set)

        self.details_text.grid(row=0, column=0, sticky="nsew")
        details_vsb.grid(row=0, column=1, sticky="ns")
        details_hsb.grid(row=1, column=0, sticky="ew")

        details_container.rowconfigure(0, weight=1)
        details_container.columnconfigure(0, weight=1)

        # # Right middle canvas - show RAG similar element
        # right_middle_frame = ttk.Frame(self.right_frame)
        # self.semantic_ui_desc = tk.Text(right_middle_frame, wrap=tk.WORD)
        # vsb_middle_elem = ttk.Scrollbar(right_middle_frame, orient="vertical", command=self.semantic_ui_desc.yview)
        # self.semantic_ui_desc.configure(yscrollcommand=vsb_middle_elem.set)
        # self.semantic_ui_desc.grid(row=0, column=0, sticky="nsew")
        # vsb_middle_elem.grid(row=0, column=1, sticky="ns")

        # right_middle_frame.grid_columnconfigure(0, weight=1)
        # right_middle_frame.grid_rowconfigure(0, weight=1)

        # Right bottom text - wrap in a frame
        right_bottom_frame = ttk.Frame(self.right_frame)
        self.ui_desc = tk.Text(right_bottom_frame, height=10, wrap=tk.WORD)
        vsb_elem = ttk.Scrollbar(right_bottom_frame, orient="vertical", command=self.ui_desc.yview)
        self.ui_desc.configure(yscrollcommand=vsb_elem.set)
        self.ui_desc.grid(row=0, column=0, sticky="nsew")
        vsb_elem.grid(row=0, column=1, sticky="ns")

        # Configure right_bottom_frame grid
        right_bottom_frame.grid_columnconfigure(0, weight=1)
        right_bottom_frame.grid_rowconfigure(0, weight=1)

        # Add to right paned window
        self.right_frame.add(right_top_frame, weight=3)
        # self.right_frame.add(right_middle_frame, weight=1)
        self.right_frame.add(right_bottom_frame, weight=1)

        # Control buttons frame - update grid position
        control_frame = ttk.Frame(root)
        control_frame.grid(row=1, column=0, columnspan=3, pady=5, sticky="ew")
        control_frame.grid_columnconfigure(2, weight=1)  # Make middle column expandable

        # Left-aligned controls
        left_controls = ttk.Frame(control_frame)
        left_controls.grid(row=0, column=0, sticky="w")

        self.auto_refresh = tk.BooleanVar(value=True)
        ttk.Checkbutton(left_controls, text="Auto Refresh", 
                       variable=self.auto_refresh).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(left_controls, text="Manual Refresh",
                  command=self.trigger_refresh).pack(side=tk.LEFT, padx=5)

        self.refresh_label = ttk.Label(left_controls, text="Refreshed 0 seconds ago")
        self.refresh_label.pack(side=tk.LEFT, padx=5)

        # Add a separator
        ttk.Separator(left_controls, orient="vertical").pack(side=tk.LEFT, padx=5)

        self.auto_apply = tk.BooleanVar(value=True)
        ttk.Checkbutton(left_controls, text="Auto Apply",
                       variable=self.auto_apply).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(left_controls, text="Inspect Actions",
                  command=self.trigger_action).pack(side=tk.LEFT, padx=5)

        ttk.Button(left_controls, text="Show UI Graph",
                  command=self.show_graph).pack(side=tk.LEFT, padx=5)

        # Right-aligned status
        # self.mask_label = ttk.Label(control_frame, text="Mask Detection: Pending")
        # self.mask_label.grid(row=0, column=2, sticky="e", padx=20)
        self.dp_label = ttk.Label(control_frame, text="DP Detection: Pending")
        self.dp_label.grid(row=0, column=2, sticky="e", padx=20)

        self.status_label = ttk.Label(control_frame, text="Status: Idle")
        self.status_label.grid(row=0, column=3, sticky="e", padx=20)

        self.mask_result = None
        self.mask_thread = None

        self.dp_result = None
        self.dp_thread = None

        self.action_dialog = None
        self.is_fetching_ui = False
        self.is_applying_action = False

        # Initialize variables
        self.update_interval = 100  # milliseconds
        self.last_refresh_time = time.time()
        self.scale_factor = 1.0
        self.photo = None
        self.last_screenshot = None  # cached screenshot image data 
        self.org_uist = None         # original ui elements (not being post-processed)
        self.uist_before_supply = None # original ui elements (not being supplied by visual cue)
        self.uist = None             # ui elements
        self.org_vid_map = None      # original ui elements (not being post-processed)
        self.vid_map_before_supply = None # original ui elements (not being supplied by visual cue)
        self.vid_map = None          # synchronized with current ui elements
        self.analyzed_ui = None      # analyzed ui elements
        self.actions = None          # actions analyzed for current UI
        self.action_history = []     # history of actions
        self._actions_history = []   # history of analyzed actions
        self.task = 'None'           # task description
        self.semantic_analyzed = None# semantic analyzed result
        self.ui_graph = self.UiGraph()    # UI change graph

        self.skip_steps_by_preprocess = 0 # steps skipped by sync_ui_screenshot preprocessor
        self.ui_change_history = []       # ui history if any change is detected
        self.graph_shown = None

        # subtask
        self.subtask_candidate = []
        self.current_subtask = None
        self.random_exploration = False
        self.remain_random_steps = 0
        self.keywords = None

        self.refresh_subtask()

        # RAG
        if rag_db:
            self.rag_db = rag_db
        else:
            self.rag_db = None
        self.similar_elements = {}
        self.similar_ui = []
        self.rag_analyze = None
        self.cached_rag_analyzed = None

        # Queue for communication between threads
        self.data_queue = queue.Queue()
        self.action_queue = queue.Queue()
        self.is_updating = False

        # Start UI updates in separate thread
        self.check_update()

        # Handle window resize
        self.root.bind("<Configure>", self.on_window_resize)
        self.is_resize_done = True
        
        # Start checking queue periodically
        self.check_queue()
        
        # Start refresh time updater
        self.update_refresh_time()

        # Add instance variables for highlighting
        self.highlight_rect = None
        self.current_element_id = None

        # Add cache-related variables
        self.enable_ui_cache = False  # SETTINGS: Toggle for UI caching
        self.cached_uist = None
        self.cached_analyzed = None
        self.cached_supply_uist = None
        self.cached_semantic_analyzed = None

    def refresh_subtask(self):
        self.subtask_candidate = [
            TaskCandidate(sub_task=UserSubTaskTypeExploreApp(task="explore_app")),
            TaskCandidate(sub_task=UserSubTaskTypeSwitch(task="switch_app")),
            TaskCandidate(sub_task=UserSubTaskTypeRemoveAd(task="remove_ad")),
            TaskCandidate(sub_task=UserSubTaskTypeNonFullScreenAd(task="Non-Fullscreen_Ad")),
            TaskCandidate(sub_task=UserSubTaskTypeObserveCloseFullscreenAd(task="Observe_Close_Fullscreen_Ad")),
        ]
        self.current_subtask = self.subtask_candidate[0]
        self.random_exploration = False
        self.remain_random_steps = 0
        self.keywords = None
        return

    def show_graph(self):
        # if self.graph_shown:
        #     if tmp_file := self.graph_shown.tmpfile:
        #         import os
        #         os.remove(tmp_file.name)

        if self.is_fetching_ui or self.is_applying_action:
            logger.info(f'UI Graph not available now, because actions are being applied, or ui is being fetched.')
            return
        if self.graph_shown:
            del self.graph_shown
        self.graph_shown = self.UIGraphShow(root=self.root, ui_graph=self.ui_graph)
        return

    def history_record(self):
        return self.ui_graph.get_history()

    def set_task(self, task):
        """Set the task description"""
        self.task = task

    # def mark_task_status(self, task_list, task_status):
    #     for task in task_list:
    #         if task in self.subtask_candidate:
    #             self.subtask_candidate.remove(task)
    #             self.subtask_ended.append((task, task_status))
    #             self.ui_graph.append_task((task, task_status), self.ui_graph.last_node)
    
    def close_immediately(self):
        """Close the window immediately by setting flag"""
        self._close_immediately = True
    
    def schedule_close_check(self):
        """Schedule close check"""
        if self._close_immediately:
            self.on_closing()
        else:
            self.root.after(300, self.schedule_close_check)
    
    def assert_running(self):
        """Assert that the window is still running"""
        if not self.running:
            raise Exception("Window is not running")

    def call_blocking(self, func, *args, **kwargs):
        """Call a function in a blocking manner and return its result"""
        result = None
        
        def wrapper():
            nonlocal result
            result = func(*args, **kwargs)
            
        thread = threading.Thread(target=wrapper, daemon=True)
        thread.start()
        
        while True:
            self.root.update()
            if not thread.is_alive() or not self.running:
                break
            time.sleep(0.05)
                
        return result

    def check_update(self):
        """(Loop Schedule) Check if update is needed"""
        if self.running:
            if len(self.action_history) > 3 and all(line.strip().startswith('[*]') for line in base_action._get_action_history(self.action_history).split('\n')[1:][-12:]):
                logger.warning("App continues to restart for more than 3 times, terminating...")
                result_callback(False, True)
                if not self._close_immediately:
                    self.on_closing()
                return
            if not self.is_updating and self.auto_refresh.get():
                # self.mask_result = self.mask_thread = None
                # self.mask_label.config(text="Mask Detection: Pending", foreground="black")
                self.dp_result = self.dp_thread = None
                self.dp_label.config(text="DP Detection: Pending", foreground="black")
                self.root.after(0, self.trigger_refresh)
            self.root.after(self.update_interval, self.check_update)

    def trigger_refresh(self):
        """Start the UI refreshing process"""
        if self.running and not self.is_updating:
            self.is_updating = True
            threading.Thread(target=self.fetch_ui_data, daemon=True).start()
    
    def trigger_action(self):
        """Start the action analysis process"""
        if self.running and not self.is_fetching_ui and not self.action_dialog:
            threading.Thread(target=self.fetch_action, daemon=True).start()
    
    # def trigger_mask(self, final_ui, base64_screenshot):
    #     """Start the mask detection process"""
    #     if self.running and not self.mask_thread:
    #         self.mask_thread = threading.Thread(target=self.fetch_mask, args=(final_ui, base64_screenshot), daemon=True)
    #         self.mask_thread.start()

    def trigger_dp(self):
        """Start the dark pattern detection process"""
        if self.running and not self.dp_thread:
            self.dp_thread = threading.Thread(target=self.fetch_dp, daemon=True)
            self.dp_thread.start()

    def check_queue(self):
        """(Loop Schedule) Check for completed updates"""
        data = None
        try:
            while True:
                data = self.data_queue.get_nowait()
                if not data == True:
                    self.status_label.config(text="Status: Restarting app")
                    self.action_history.append("The app may have crashed and is not responding. Restart the app.")
                    self._actions_history.append(None)
                    self.refresh_subtask()
                    self.ui_graph.add_action(SemanticUserAction(
                        actions=[SemanticUserActionStep(reasoning="", target_element_id=-1, reason_of_element="",
                                                        completeness_check="",
                                                        action=UserActionTypeRestart(type="restart"))
                                 ],
                        summary=UserActionSummary(ui_summary="",
                                                  action_summary="The app may have crashed during getting refreshing result. Restart the app."),
                        past_action_summary="",
                        recent_repeat_action="",
                        recent_repeat_count=0,
                        subtask_candidate=[],
                        # random_exploration=False,
                        page_semantic="",
                        subtask=UserSubTask(sub_task=UserSubTaskTypeExploreApp(task="explore_app")),
                        reasoning="",
                    ))
                    self.call_blocking(base_action.restart_app) # the app may crash
                    self.is_updating = False
                    self.trigger_refresh()
                    continue
                self.update_display()
        except queue.Empty:
            pass
        if data:
            # Trigger action analysis
            self.trigger_action()
            # Mask detection is not needed in DP detection
            # # Trigger mask detection if ads are not detected
            # if self.analyzed_ui.ads and self.analyzed_ui.ads.is_ad_topmost and self.analyzed_ui.ads.ad_close_button_id > 0:
            #     self.mask_label.config(text="Mask Detection: Skip", foreground="black")
            # else:
            #     self.mask_label.config(text="Mask Detection: Waiting", foreground="black")
            #     self.trigger_mask(self.uist['elements'], base64.b64encode(self.last_screenshot).decode('utf-8'))

            self.dp_label.config(text="DP Detection: Waiting", foreground="black")
            self.trigger_dp()

        try:
            while True:
                action = self.action_queue.get_nowait()
                if not action == True:
                    self.status_label.config(text="Status: Restarting app")
                    self.action_history.append("The app may have crashed and is not responding. Restart the app.")
                    self._actions_history.append(None)
                    self.refresh_subtask()
                    self.ui_graph.add_action(SemanticUserAction(
                        actions=[SemanticUserActionStep(reasoning="", target_element_id=-1, reason_of_element="",
                                                        completeness_check="",
                                                        action=UserActionTypeRestart(type="restart"))
                                 ],
                        summary=UserActionSummary(ui_summary="",
                                                  action_summary="The app may have crashed during getting refreshing result. Restart the app."),
                        past_action_summary="",
                        recent_repeat_action="",
                        recent_repeat_count=0,
                        subtask_candidate=[],
                        # random_exploration=False,
                        page_semantic="",
                        subtask=UserSubTask(sub_task=UserSubTaskTypeExploreApp(task="explore_app")),
                        reasoning="",
                    ))
                    self.call_blocking(base_action.restart_app) # the app may crash
                    self.is_updating = False
                    self.trigger_refresh()
                    continue
                self.update_display()
                self.status_label.config(text="Status: Waiting for user confirmation")
                self.create_action_dialog()
        except queue.Empty:
            pass

        # Schedule next check
        if self.running:
            self.root.after(100, self.check_queue)
    
    def create_action_dialog(self):
        """Create a dialog to confirm the action"""
        # Create dialog window
        self.action_dialog = tk.Toplevel(self.root)
        self.action_dialog.title("Confirm Action")
        self.action_dialog.geometry("600x500")  # Initial size
        
        # Restore previous position if it exists
        if hasattr(self, 'dialog_geometry'):
            self.action_dialog.geometry(self.dialog_geometry)
            
        self.action_dialog.transient(self.root)
        self.action_dialog.rowconfigure(0, weight=1)  # Make dialog resizable
        self.action_dialog.columnconfigure(0, weight=1)
        
        # Create main frame with padding that fills dialog
        main_frame = ttk.Frame(self.action_dialog, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")
        main_frame.rowconfigure(1, weight=1)  # History section
        main_frame.rowconfigure(3, weight=1)  # Action section
        main_frame.columnconfigure(0, weight=1)
        
        # Create PanedWindow to allow resizing between history and action sections
        paned = ttk.PanedWindow(main_frame, orient=tk.VERTICAL)
        paned.grid(row=1, column=0, rowspan=3, sticky="nsew")
        
        # History section
        history_frame = ttk.Frame(paned)
        history_frame.rowconfigure(1, weight=1)
        history_frame.columnconfigure(0, weight=1)
        
        history_label = ttk.Label(history_frame, text="Action History:")
        history_label.grid(row=0, column=0, sticky="w")
        
        # Add scrollbar for history text
        history_text_frame = ttk.Frame(history_frame)
        history_text_frame.grid(row=1, column=0, sticky="nsew")
        history_text_frame.rowconfigure(0, weight=1)
        history_text_frame.columnconfigure(0, weight=1)
        
        history_scroll = ttk.Scrollbar(history_text_frame)
        history_scroll.grid(row=0, column=1, sticky="ns")
        
        history_text = tk.Text(history_text_frame, wrap=tk.WORD, yscrollcommand=history_scroll.set, height=10)
        history_text.grid(row=0, column=0, sticky="nsew")
        history_scroll.config(command=history_text.yview)
        
        history_text.insert(tk.END, base_action._get_action_history(self.action_history))
        history_text.config(state=tk.DISABLED)
        
        # Action section  
        action_frame = ttk.Frame(paned)
        action_frame.rowconfigure(1, weight=1)
        action_frame.columnconfigure(0, weight=1)
        
        action_label = ttk.Label(action_frame, text="Proposed Action:")
        action_label.grid(row=0, column=0, sticky="w")
        
        # Add scrollbar for action text
        action_text_frame = ttk.Frame(action_frame)
        action_text_frame.grid(row=1, column=0, sticky="nsew")
        action_text_frame.rowconfigure(0, weight=1)
        action_text_frame.columnconfigure(0, weight=1)
        
        action_scroll = ttk.Scrollbar(action_text_frame)
        action_scroll.grid(row=0, column=1, sticky="ns")
        
        action_text = tk.Text(action_text_frame, wrap=tk.WORD, yscrollcommand=action_scroll.set, height=10)
        action_text.grid(row=0, column=0, sticky="nsew")
        action_scroll.config(command=action_text.yview)
        
        action_json = json.dumps(self.actions.model_dump(), indent=2, ensure_ascii=False)
        action_text.insert(tk.END, action_json)
        action_text.config(state=tk.DISABLED)
        
        # Add frames to paned window with equal weights
        paned.add(history_frame, weight=1)
        paned.add(action_frame, weight=1)
        
        # Button frame
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=4, column=0, sticky="ew", pady=(10,0))
        button_frame.columnconfigure((0,4), weight=1)  # Equal weight to outer columns for centering
        
        # Save geometry when window moves
        def save_geometry(event):
            self.dialog_geometry = self.action_dialog.geometry()
        self.action_dialog.bind('<Configure>', save_geometry)
        
        # Bind window close button to discard action
        self.action_dialog.protocol("WM_DELETE_WINDOW", lambda: discard_btn.invoke())

        def destroy_dialog():
            self.action_dialog.destroy()
            self.action_dialog = None

        def set_idle():
            # if (m := self.mask_thread) and m.is_alive() and self.running:
            if (m := self.dp_thread) and m.is_alive() and self.running:
                # self.status_label.config(text="Status: Pending Mask Detection")
                self.status_label.config(text="Status: Pending DP Detection")
                self.call_blocking(m.join)
            self.is_updating = False
            if self.running:
                self.status_label.config(text="Status: Idle")
        
        def move_forward():
            step_callback(self.task, base64.b64encode(self.last_screenshot).decode('utf-8'), self.org_uist, self.uist, self.analyzed_ui, self.semantic_analyzed, self.actions, self.dp_result, self.action_history, self._actions_history, self.skip_steps_by_preprocess, self.ui_change_history, self.ui_graph, self.subtask_candidate, self.similar_elements, self.similar_ui, self.rag_analyze)

            # Mask Detection doesn't end the task in ads dark patterns detection!
            # if self.is_triggered or self.is_completed:
            if self.is_completed:
                # result_callback(self.is_triggered, self.is_completed, False)
                result_callback(self.is_completed, False)
                if not self._close_immediately:
                    self.on_closing() # Close the window

            self.rag_analyze = None
        
        # Action buttons in center columns
        apply_btn = ttk.Button(button_frame, text="Apply", 
                             command=lambda: [destroy_dialog(), self.call_blocking(self.apply_action), self.update_display(), set_idle(), move_forward()])
        apply_btn.grid(row=0, column=1, padx=5)
        
        retry_btn = ttk.Button(button_frame, text="Retry",
                             command=lambda: [destroy_dialog(), self.trigger_action()])
        retry_btn.grid(row=0, column=2, padx=5)
        
        discard_btn = ttk.Button(button_frame, text="Discard",
                                command=lambda: [destroy_dialog(), set_idle()])
        discard_btn.grid(row=0, column=3, padx=5)

        if self.auto_apply.get():
            apply_btn.invoke()

    class UIGraphShow:
        @staticmethod
        def _decode_base64(base64_screenshot):
            image_bytes = base64.b64decode(base64_screenshot)
            from io import BytesIO
            image_data = BytesIO(image_bytes)
            return image_data.read()

        def __init__(self, root, ui_graph):
            # self.screenshot = None
            # self.last_screenshot = None
            self.root = root
            self.ui_graph = copy.deepcopy(ui_graph)
            self.tmpfile = None
            # self.net_window = None
            # self.html = None

            if not self.ui_graph.last_node:
                logger.warning("UI Graph unavailable.")
                return

            self.build_graph()

        def build_graph(self):
            net = Network(directed=True, notebook=False)
            for node in self.ui_graph.nodes:
                net.add_node(str(node.id), screenshot=node.screenshot[-1], title=json.dumps(list(node.global_local_map.keys())), property=json.dumps(node.model_dump(mode="json", exclude={"screenshot"}), ensure_ascii=False, indent=2))
                # , title=json.dumps(node.model_dump(mode="json", exclude={"screenshot"}), ensure_ascii=False, indent=2)
            for edge in self.ui_graph.edges:
                net.add_edge(str(edge.from_node), str(edge.to_node), title=json.dumps([each_edge.global_step for each_edge in self.ui_graph.edges if each_edge.from_node == edge.from_node and each_edge.to_node == edge.to_node]), property=json.dumps(edge.model_dump(mode="json"), ensure_ascii=False, indent=2))
                # , title=json.dumps(edge.model_dump(mode="json"), ensure_ascii=False, indent=2)

            # 注入点击事件 → 导航触发（tkinterweb 可以捕捉）
            net.set_options("""
            {
              "interaction": {"hover": true},
              "nodes": {"shape": "dot", "size": 20}
            }
            """)
            # net.html = net.generate_html()
            # net.html = net.html.replace(
            #     "network.on(\"click\",",
            #     """
            #     network.on("click", function(params) {
            #         if(params.nodes.length > 0) {
            #             var nodeId = params.nodes[0];
            #             window.location.href = "pyvis://node/" + nodeId;
            #         } else if(params.edges.length > 0) {
            #             var edgeId = params.edges[0];
            #             window.location.href = "pyvis://edge/" + edgeId;
            #         }
            #     });
            #     network.on("click", """
            # )

            # 得到 pyvis 自带的 html
            html = net.generate_html()
            html = html.replace(
                "</style>",
                '''
                    /* 公共浮动面板样式 */
                    .panel {
                        position: fixed; /* 浮动在窗口上方，不影响文档流 */
                        top: 0;
                        height: 100vh; /* 高度等于视口高度 */
                        background-color: rgba(255, 255, 255, 0.9); /* 半透明背景 */
                        border: 1px solid #ccc;
                        padding: 10px;
                        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                        z-index: 9999; /* 保证在最上层 */
                        overflow-y: auto; /* 内容可滚动 */
                    }
                    
                    /* 左侧浮动面板 */
                    #left-panel {
                        left: 0;
                        width: 400px;
                        display: flex;
                        flex-direction: column;
                        align-items: center;
                        justify-content: flex-start;
                    }
                    
                    /* 右侧浮动面板 */
                    #right-panel {
                        right: 0;
                        width: 33.3333%; /* 占窗口三分之一 */
                        display: flex;
                        flex-direction: column;
                        align-items: flex-start;
                        justify-content: flex-start;
                    }
                    
                    /* 左侧面板图片自适应 */
                    #left-panel img {
                        max-width: 100%;
                        height: auto;
                        display: block;
                        margin-bottom: 10px;
                    }
                    pre {{
                        white-space: pre-wrap;
                        word-break: break-all;
                        margin: 0; /* 移除 pre 默认外边距 */
                        font-size: 0.9em;
                    }}
                    h3 {{
                        margin-top: 0;
                        color: #333;
                    }}
                </style>'''
            )
            html = html.replace(
                '''<script type="text/javascript">''',
                '''
                <div id="left-panel" class="panel">
                    <h3>Screenshot</h3>
                    <div id="screenshot"></div>
                </div>
                <div id="right-panel" class="panel">
                    <h3>Node Properties</h3>
                    <pre id="title">点击节点后显示属性</pre>
                </div>
                <script type="text/javascript">'''
            )
            html = html.replace(
                "drawGraph();",
                '''drawGraph();
                network.on("click", function(params) {{
                    var leftPanel = document.getElementById("left-panel");
                    var rightPanel = document.getElementById("right-panel");
                    var screenshotDiv = document.getElementById("screenshot");
                    var titlePre = document.getElementById("title");

                    if (params.nodes.length > 0) {{
                        var nodeId = params.nodes[0];
                        var node = nodes.get(nodeId); // 从 pyvis 的 nodes DataSet 中获取节点数据

                        // 处理截图
                        if (node.screenshot) {{ // 检查是否存在 screenshot 属性
                            screenshotDiv.innerHTML = '<img src="data:image/png;base64,' + node.screenshot + '"/>';
                            leftPanel.style.display = 'flex'; // 显示截图面板
                        }} else {{
                            screenshotDiv.innerHTML = ''; // 清空截图区域
                            leftPanel.style.display = 'none'; // 隐藏截图面板
                        }}

                        // 处理节点属性 (title)
                        titlePre.textContent = node.property;
                        rightPanel.style.display = 'flex'; // 显示属性面板
                        
                    }} else if (params.edges.length > 0) {{
                    
                        var edgeId = params.edges[0];
                        var edge = edges.get(edgeId); // 注意 pyvis 也提供 edges DataSet
                        
                        screenshotDiv.innerHTML = ''; // 边没有截图
                        leftPanel.style.display = 'none';
                        titlePre.textContent = edge.property; // 显示边的 property
                        rightPanel.style.display = 'flex';
                        
                    }} else {{
                        // 如果点击空白区域，隐藏所有面板
                        leftPanel.style.display = 'none';
                        rightPanel.style.display = 'none';
                    }}
                }});

                // 确保页面加载时面板是隐藏的
                document.addEventListener('DOMContentLoaded', function() {{
                    document.getElementById("left-panel").style.display = 'none';
                    document.getElementById("right-panel").style.display = 'none';
                }});
                '''
            )

            import tempfile
            self.tmpfile = tempfile.NamedTemporaryFile(delete=False, dir='./tmp_resource', prefix=str(time.time()), suffix=".html")
            with open(self.tmpfile.name, "w", encoding="utf-8") as f:
                f.write(html)
            webbrowser.open(self.tmpfile.name)

            # import re
            # # 提取出 <script> 里的初始化代码
            # match = re.search(r"<script type=\"text/javascript\">(.*?)</script>", html, re.S)
            # if not match:
            #     raise RuntimeError("无法从 pyvis HTML 中提取脚本")
            # network_js = match.group(1)

            # custom_html_template = f"""
            # <html>
            # <head>
            #     <meta charset="utf-8"/>
            #     <title>UI Graph Viewer</title>
            #     <style>
            #         body {{
            #             margin: 0;
            #             overflow: hidden; /* 阻止滚动条 */
            #             font-family: Arial, sans-serif;
            #             display: flex; /* 让 body 作为flex容器 */
            #             height: 100vh; /* 占据整个视口高度 */
            #             width: 100vw; /* 占据整个视口宽度 */
            #         }}
            #         #mynetwork {{
            #             //flex-grow: 1; /* 占据剩余空间 */
            #             //position: relative; /* 确保其子元素（如果有）相对其定位 */
            #             width: 100%;
            #             height: 100%;
            #         }}
            #         .panel {{
            #             //position: absolute;
            #             position: fixed;
            #             background-color: rgba(255, 255, 255, 0.9); /* 半透明背景 */
            #             border: 1px solid #ccc;
            #             padding: 10px;
            #             box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            #             z-index: 10; /* 确保在图的上方 */
            #             display: none; /* 初始隐藏 */
            #             max-height: calc(100vh - 40px); /* 留一些边距 */
            #             overflow-y: auto; /* 允许内容滚动 */
            #             border-radius: 8px; /* 圆角 */
            #         }}
            #         #left-panel {{
            #             top: 20px;
            #             left: 20px;
            #             width: 300px; /* 截图面板宽度 */
            #             display: flex; /* 用于居中图片 */
            #             flex-direction: column;
            #             align-items: center;
            #             justify-content: center;
            #         }}
            #         #right-panel {{
            #             top: 20px;
            #             right: 20px;
            #             width: 300px; /* 属性面板宽度 */
            #         }}
            #         #left-panel img {{
            #             max-width: 100%;
            #             height: auto;
            #             display: block; /* 移除图片底部额外的空间 */
            #             border-radius: 4px;
            #         }}
            #         pre {{
            #             white-space: pre-wrap;
            #             word-break: break-all;
            #             margin: 0; /* 移除 pre 默认外边距 */
            #             font-size: 0.9em;
            #         }}
            #         h3 {{
            #             margin-top: 0;
            #             color: #333;
            #         }}
            #     </style>
            # </head>
            # <body>
            #     <div id="mynetwork"></div>
            #
            #     <div id="left-panel" class="panel">
            #         <h3>Screenshot</h3>
            #         <div id="screenshot"></div>
            #     </div>
            #     <div id="right-panel" class="panel">
            #         <h3>Node Properties</h3>
            #         <pre id="title">点击节点后显示属性</pre>
            #     </div>
            #
            #     <script type="text/javascript">
            #         // 注入 pyvis 生成的网络图 JS 代码
            #         {network_js}
            #
            #         network.on("click", function(params) {{
            #             var leftPanel = document.getElementById("left-panel");
            #             var rightPanel = document.getElementById("right-panel");
            #             var screenshotDiv = document.getElementById("screenshot");
            #             var titlePre = document.getElementById("title");
            #
            #             if (params.nodes.length > 0) {{
            #                 var nodeId = params.nodes[0];
            #                 var node = nodes.get(nodeId); // 从 pyvis 的 nodes DataSet 中获取节点数据
            #
            #                 // 处理截图
            #                 if (node.screenshot) {{ // 检查是否存在 custom_screenshot 属性
            #                     screenshotDiv.innerHTML = '<img src="data:image/png;base64,' + node.screenshot + '"/>';
            #                     //leftPanel.style.display = 'flex'; // 显示截图面板
            #                     leftPanel.style.display = 'block'; // 显示截图面板
            #                 }} else {{
            #                     screenshotDiv.innerHTML = ''; // 清空截图区域
            #                     leftPanel.style.display = 'none'; // 隐藏截图面板
            #                 }}
            #
            #                 // 处理节点属性 (title)
            #                 titlePre.textContent = node.title;
            #                 rightPanel.style.display = 'block'; // 显示属性面板
            #
            #             }} else {{
            #                 // 如果点击空白区域，隐藏所有面板
            #                 leftPanel.style.display = 'none';
            #                 rightPanel.style.display = 'none';
            #             }}
            #         }});
            #
            #         // 确保页面加载时面板是隐藏的
            #         document.addEventListener('DOMContentLoaded', function() {{
            #             document.getElementById("left-panel").style.display = 'none';
            #             document.getElementById("right-panel").style.display = 'none';
            #         }});
            #     </script>
            # </body>
            # </html>
            # """

            # import tempfile
            # tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
            # with open(tmpfile.name, "w", encoding="utf-8") as f:
            #     f.write(net.html)
            # net.show(tmpfile.name, notebook=False)

            # import tempfile
            # with tempfile.NamedTemporaryFile(delete=False, suffix=".html", mode="w", encoding="utf-8") as tmp_file:
            #     tmp_file.write(custom_html_template)
            #     html_file_path = tmp_file.name
            #
            # net.show(html_file_path, notebook=False)

            # 加载 HTML
            # self.html.load_file(tmpfile.name)

            # import os
            # # 将 tmpfile.name 转成 file:// URL
            # file_url = 'file:///' + os.path.abspath(tmpfile.name).replace('\\', '/')
            #
            # # 在 graph_frame 内嵌入 WebView2
            # webview2 = WebView2(graph_frame, width=graph_frame.winfo_width(),
            #                          height=graph_frame.winfo_height())
            # webview2.pack(fill='both', expand=True)
            # webview2.load_url(file_url)

        # def handle_click(self, url):
        #     if url.startswith("pyvis://node/"):
        #         node_id = url.split("/")[-1]
        #         node = next((n for n in self.ui_graph.nodes if str(n.id) == node_id), None)
        #         if node:
        #             # 更新右下文本
        #             self.update_attr_text(json.dumps(node.model_dump(mode="json"), exclude={"screenshot"}, ensure_ascii=False, indent=2))
        #             # 更新截图
        #             if getattr(node, "screenshot", None):
        #                 self.update_screenshot(node.screenshot)
        #
        #     elif url.startswith("pyvis://edge/"):
        #         edge_id = url.split("/")[-1]
        #         edge = next((e for e in self.ui_graph.edges if f"{e.from_node}-{e.to_node}" == edge_id), None)
        #         if edge:
        #             # 更新右下文本
        #             self.update_attr_text(json.dumps(edge.model_dump(mode="json"), ensure_ascii=False, indent=2))
        #             # 边没有截图，就不更新截图
        #
        # def destroy(self):
        #     """释放资源并销毁窗口"""
        #     if self.net_window is not None:
        #         try:
        #             self.net_window.destroy()
        #         except Exception:
        #             pass
        #         self.net_window = None
        #
        #     # 清理图像引用，帮助 GC
        #     self.screenshot = None
        #     self.last_screenshot = None
        #     self.html = None
        #     self.attr_text = None

    # class UIGraphShow:
    #     @staticmethod
    #     def _decode_base64(base64_screenshot):
    #         """Decode base64 screenshot to bytes"""
    #         return base64.b64decode(base64_screenshot)
    #
    #     def __init__(self, ui_graph):
    #         self.ui_graph = copy.deepcopy(ui_graph)
    #         if not self.ui_graph.last_node:
    #             print("UI Graph unavailable.")
    #             return
    #
    #         # 启动独立线程运行 webview
    #         # threading.Thread(target=self._start_webview, daemon=True).start()
    #
    #     def start_webview(self):
    #         self._start_webview()
    #
    #     def _start_webview(self):
    #         """生成 HTML 并用 pywebview 弹出独立窗口"""
    #         # 生成 pyvis HTML
    #         net = Network(directed=True, notebook=False)
    #         for node in self.ui_graph.nodes:
    #             net.add_node(str(node.id),
    #                          title=json.dumps(node.model_dump(mode="json", exclude={"screenshot"}), ensure_ascii=False,
    #                                           indent=2),
    #                          screenshot=node.screenshot)
    #         for edge in self.ui_graph.edges:
    #             net.add_edge(str(edge.from_node), str(edge.to_node),
    #                          title=json.dumps(edge.model_dump(mode="json"), ensure_ascii=False, indent=2))
    #
    #         net.set_options("""
    #         {
    #           "interaction": {"hover": true},
    #           "nodes": {"shape": "dot", "size": 20}
    #         }
    #         """)
    #
    #         # 注入点击事件，触发 pyvis:// 协议
    #         net.html = net.generate_html()
    #         net.html = net.html.replace(
    #             'network.on("click",',
    #             """
    #             network.on("click", function(params) {
    #                 if(params.nodes.length > 0) {
    #                     var nodeId = params.nodes[0];
    #                     window.pywebview.api.node_click(nodeId);
    #                 } else if(params.edges.length > 0) {
    #                     var edgeId = params.edges[0];
    #                     window.pywebview.api.edge_click(edgeId);
    #                 }
    #             });
    #             network.on("click", """
    #         )
    #
    #         # 生成完整 HTML 包含左截图、右上图、右下属性
    #         html_content = self._wrap_full_html(net.html, self.ui_graph.last_node.screenshot,
    #                                             json.dumps(self.ui_graph.last_node.model_dump(mode="json",
    #                                                                                           exclude={"screenshot"}),
    #                                                        ensure_ascii=False, indent=2))
    #
    #         import tempfile
    #         # 写入临时文件
    #         tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
    #         with open(tmpfile.name, "w", encoding="utf-8") as f:
    #             f.write(html_content)
    #
    #         import os
    #         # 创建 webview 窗口
    #         file_url = 'file:///' + os.path.abspath(tmpfile.name).replace('\\', '/')
    #         api = self  # 当前实例作为 pywebview API
    #         webview.create_window("UI Graph Viewer", file_url, width=1200, height=900, resizable=True, js_api=api)
    #         webview.start(debug=True)
    #
    #     def _wrap_full_html(self, pyvis_html, base64_screenshot, attr_text):
    #         """
    #         Wrap pyvis HTML into a full page with left screenshot and bottom attributes
    #         Layout: left screenshot | right top pyvis | right bottom attributes
    #         """
    #         # 将 Base64 字符串放入 img 标签
    #         img_src = f"data:image/png;base64,{base64_screenshot}"
    #         html = f"""
    #         <html>
    #         <head>
    #         <style>
    #             body {{ margin:0; font-family: Arial; height:100vh; }}
    #             .container {{ display: flex; height:100%; }}
    #             .left {{ width:30%; padding:5px; box-sizing:border-box; }}
    #             .right {{ flex:1; display:flex; flex-direction:column; }}
    #             .graph {{ flex:1; border:1px solid #ccc; }}
    #             .attr {{ flex:1; border-top:1px solid #ccc; overflow:auto; padding:5px; white-space:pre-wrap; background:#f7f7f7; }}
    #         </style>
    #         </head>
    #         <body>
    #         <div class="container">
    #             <div class="left">
    #                 <img id="screenshot" src="{img_src}" style="max-width:100%; max-height:100%;" />
    #             </div>
    #             <div class="right">
    #                 <div class="graph">
    #                     {pyvis_html}
    #                 </div>
    #                 <div class="attr" id="attr_text">{attr_text}</div>
    #             </div>
    #         </div>
    #         </body>
    #         </html>
    #         """
    #         return html
    #
    #     # pywebview JS API
    #     def node_click(self, node_id):
    #         """Called from JS when node is clicked"""
    #         node = next((n for n in self.ui_graph.nodes if str(n.id) == node_id), None)
    #         if node:
    #             # 更新属性区
    #             attr_text = json.dumps(node.model_dump(mode="json", exclude={"screenshot"}), ensure_ascii=False,
    #                                    indent=2)
    #             js_code = f"document.getElementById('attr_text').textContent = `{attr_text}`;"
    #             # 更新截图
    #             if getattr(node, "screenshot", None):
    #                 js_img = f"document.getElementById('screenshot').src = 'data:image/png;base64,{node.screenshot}';"
    #                 js_code += js_img
    #             webview.evaluate_js(js_code)
    #
    #     def edge_click(self, edge_id):
    #         """Called from JS when edge is clicked"""
    #         edge = next((e for e in self.ui_graph.edges if f"{e.from_node}-{e.to_node}" == edge_id), None)
    #         if edge:
    #             attr_text = json.dumps(edge.model_dump(mode="json"), ensure_ascii=False, indent=2)
    #             js_code = f"document.getElementById('attr_text').textContent = `{attr_text}`;"
    #             webview.evaluate_js(js_code)
    
    @time_consumed
    def sync_ui_screenshot(self, first_apply=True):
        """Fetch newest UI & Screenshot data"""

        # Should try to hide keyboard before getting screenshot
        base_action.hide_keyboard()

        # Should try to bring app to foreground before getting screenshot
        base_action.bring_to_foreground()

        sleep(rand(0.5, 1))

        self.assert_running()

        label = self.status_label.cget("text")

        self.status_label.config(text="Status: Getting UI & Screenshot")

        # Function to get UI in background
        def get_ui():
            try:
                nonlocal uist, org_vid_map, suc1
                uist, org_vid_map = base_ui.getui()
                self.org_vid_map = copy.deepcopy(org_vid_map)
                suc1 = True
            except Exception as e:
                logger.exception("Error getting UI")

        # Function to get screenshot in background
        def get_ss():
            try:
                nonlocal suc2
                get_screenshot()
                suc2 = True
            except Exception as e:
                logger.exception("Error getting screenshot")

        # Start both threads
        uist = org_vid_map = None
        suc1, suc2 = False, False
        ui_thread = threading.Thread(target=get_ui, daemon=True)
        ss_thread = threading.Thread(target=get_ss, daemon=True)
        ui_thread.start()
        ss_thread.start()

        # Check which thread finishes first and update status accordingly
        while ui_thread.is_alive() or ss_thread.is_alive():
            self.assert_running()
            if not ui_thread.is_alive() and ss_thread.is_alive():
                self.status_label.config(text="Status: Getting Screenshot")
            elif not ss_thread.is_alive() and ui_thread.is_alive():
                self.status_label.config(text="Status: Getting UI")
            time.sleep(0.1)
        
        if not suc1 or not suc2:
            raise Exception("Failed to get UI or screenshot")

        with open("screenshot.png", "rb") as f:
            screenshot = f.read()
        base64_screenshot = base64.b64encode(screenshot).decode('utf-8')

        # Check if ad can be closed immediately
        # if first_apply and any(text in ''.join(base_ui.get_text_list(uist['elements'])) for text in ['跳过', '轻滑试试', '扭动手机', '查看详情', '立即下载', '点击下载', '赶快下载', '一键下载', '第三方应用', '点击按钮', '摇一摇', '免费领取', '跳转详情页', '反馈']):
        #     if (cb := base_ui.element_selector(uist['elements'], 'name', 'closeButton_id', strict=True)) is not None \
        #         or (cb := base_ui.element_selector(uist['elements'], 'name', 'endcard_close', strict=True)) is not None \
        #         or (cb := base_ui.element_selector(uist['elements'], 'name', 'bu endCardClose', strict=True)) is not None \
        #         or (cb := base_ui.element_selector(uist['elements'], 'name', 'CSJSlpashSkipButton', strict=True)) is not None \
        #         or (cb := base_ui.element_selector(uist['elements'], 'name', 'close.png', strict=False)) is not None \
        #             or (cb := base_ui.element_selector(uist['elements'], 'value', '跳过', strict=False)) is not None:
        #         frame = base_ui.get_frame(cb)
        #         cb |= {'subviews': []}
        #         if len(cb.get('name', '')) < 20 and len(cb.get('value', '')) < 20:
        #             logger.info(f"Immediately closing advertisement (selector {cb}) with frame: {frame}")
        #             base_action.tap(frame['x'] + frame['width'] / 2, frame['y'] + frame['height'] / 2)
        #             self.skip_steps_by_preprocess += 1
        #             sleep(rand(0.5, 1))
        #             return self.sync_ui_screenshot(False)
        
        # Check if app store page can be closed immediately
        # if base_ui.element_selector(uist['elements'], 'name', 'AXStoreCollectionView', strict=True) is not None:
        #     if (cb := base_ui.element_selector(uist['elements'], 'name', 'ProductPageExtension.ProductView', strict=True)) is not None \
        #             and (cb := base_ui.element_selector(cb, 'name', '完成', strict=True)) is not None:
        #         frame = base_ui.get_frame(cb)
        #         cb |= {'subviews': []}
        #         logger.info(f"Immediately closing app store page (selector {cb}) with frame: {frame}")
        #         base_action.tap(frame['x'] + frame['width'] / 2, frame['y'] + frame['height'] / 2)
        #         self.skip_steps_by_preprocess += 1
        #         sleep(rand(0.5, 1))
        #         return self.sync_ui_screenshot(False)
        
        # Check if rating prompt can be closed immediately
        if base_ui.element_selector(uist['elements'], 'name', '轻点星形以在App Store中评分。', strict=True) is not None:
            if (cb := base_ui.element_selector(uist['elements'], 'name', '以后', strict=True)) is not None:
                frame = base_ui.get_frame(cb)
                cb |= {'subviews': []}
                logger.info(f"Immediately closing rating prompt (selector {cb}) with frame: {frame}")
                final_ui, _ = base_ui.post_process_ui(uist['elements'], base64_screenshot)
                self.ui_graph.add_ui(final_ui, base64_screenshot, None, None, description="Start to immediately close rating prompt.")
                base_action.tap(frame['x'] + frame['width'] / 2, frame['y'] + frame['height'] / 2)
                self.ui_graph.add_action(SemanticUserAction(
                    actions=[SemanticUserActionStep(reasoning="", target_element_id=-1, reason_of_element="",
                                                    completeness_check="",
                                                    action=UserActionTypeClick(type="click", element_id=-1, times=1))
                             ],
                    summary=UserActionSummary(ui_summary="",
                                              action_summary="Immediately closing rating prompt by clicking '以后' button."),
                    past_action_summary="",
                    recent_repeat_action="",
                    recent_repeat_count=0,
                    page_semantic="",
                    subtask_candidate=[],
                    # random_exploration=False,
                    subtask=UserSubTask(sub_task=UserSubTaskTypeExploreApp(task="explore_app")),
                    reasoning="",
                ))
                self.skip_steps_by_preprocess += 1
                sleep(rand(0.5, 1))
                return self.sync_ui_screenshot(False)
        
        # Check if sharing prompt can be closed immediately
        if (cb := base_ui.element_selector(uist['elements'], 'name', 'ActivityListView', strict=True)) is not None:
            if (cb := base_ui.element_selector(cb, 'name', 'UIActivityContentView', strict=True)) is not None:
                if (cb := base_ui.element_selector(cb, 'name', '关闭', strict=True)) is not None \
                        or (cb := base_ui.element_selector(cb, 'name', 'Close', strict=True)) is not None:
                    frame = base_ui.get_frame(cb)
                    cb |= {'subviews': []}
                    logger.info(f"Immediately closing sharing prompt (selector {cb}) with frame: {frame}")
                    final_ui, _ = base_ui.post_process_ui(uist['elements'], base64_screenshot)
                    self.ui_graph.add_ui(final_ui, base64_screenshot, None, None, description="Start to immediately close sharing prompt.")
                    base_action.tap(frame['x'] + frame['width'] / 2, frame['y'] + frame['height'] / 2)
                    self.ui_graph.add_action(SemanticUserAction(
                        actions=[SemanticUserActionStep(reasoning="", target_element_id=-1, reason_of_element="",
                                                        completeness_check="",
                                                        action=UserActionTypeClick(type="click", element_id=-1, times=1))
                                 ],
                        summary=UserActionSummary(ui_summary="",
                                                  action_summary="Immediately closing rating prompt by clicking '关闭' or 'close' button."),
                        past_action_summary="",
                        recent_repeat_action="",
                        recent_repeat_count=0,
                        page_semantic="",
                        subtask_candidate=[],
                        # random_exploration=False,
                        subtask=UserSubTask(sub_task=UserSubTaskTypeExploreApp(task="explore_app")),
                        reasoning="",
                    ))
                    self.skip_steps_by_preprocess += 1
                    sleep(rand(0.5, 1))
                    return self.sync_ui_screenshot(False)
        
        # Check if system alert can be handled immediately
        if base_ui.element_selector(uist['elements'], 'type', 'XCUIElementTypeStatusBar', strict=True) is not None \
                and (alert := base_ui.element_selector(uist['elements'], 'type', 'XCUIElementTypeAlert', strict=True)) is not None:
            # status bar is present, possibly in SpringBoard
            logger.info("System alert detected, handling immediately")
            for name in ['允许', 'Allow', '好', 'OK', '不允许', 'Don\'t Allow', '取消', 'Cancel', '关闭', 'Close']:
                if (cb := base_ui.element_selector(alert, 'name', name, strict=True)) is not None:
                    frame = base_ui.get_frame(cb)
                    final_ui, _ = base_ui.post_process_ui(uist['elements'], base64_screenshot)
                    self.ui_graph.add_ui(final_ui, base64_screenshot, None, None, description=f"Start to immediately close alert.")
                    base_action.tap(frame['x'] + frame['width'] / 2, frame['y'] + frame['height'] / 2)
                    self.ui_graph.add_action(SemanticUserAction(
                        actions=[SemanticUserActionStep(reasoning="", target_element_id=-1, reason_of_element="",
                                                        completeness_check="",
                                                        action=UserActionTypeClick(type="click", element_id=-1,
                                                                                   times=1))
                                 ],
                        summary=UserActionSummary(ui_summary="",
                                                  action_summary=f"Immediately closing s system alert by clicking button '{name}'."),
                        past_action_summary="",
                        recent_repeat_action="",
                        recent_repeat_count=0,
                        page_semantic="",
                        subtask_candidate=[],
                        # random_exploration=False,
                        subtask=UserSubTask(sub_task=UserSubTaskTypeExploreApp(task="explore_app")),
                        reasoning="",
                    ))
                    self.skip_steps_by_preprocess += 1
                    sleep(rand(0.5, 1))
                    return self.sync_ui_screenshot()
        
        # Check if Game Center can be closed immediately
        if (cb := base_ui.element_selector(uist['elements'], 'name', 'GKSignInView', strict=True)) is not None:
            if (cb := base_ui.element_selector(cb, 'name', '取消', strict=True)) is not None:
                frame = base_ui.get_frame(cb)
                cb |= {'subviews': []}
                logger.info(f"Immediately closing game center (selector {cb}) with frame: {frame}")
                final_ui, _ = base_ui.post_process_ui(uist['elements'], base64_screenshot)
                self.ui_graph.add_ui(final_ui, base64_screenshot, None, None, description="Start to immediately close game center.")
                base_action.tap(frame['x'] + frame['width'] / 2, frame['y'] + frame['height'] / 2)
                self.ui_graph.add_action(SemanticUserAction(
                    actions=[SemanticUserActionStep(reasoning="", target_element_id=-1, reason_of_element="",
                                                    completeness_check="",
                                                    action=UserActionTypeClick(type="click", element_id=-1,
                                                                               times=1))
                             ],
                    summary=UserActionSummary(ui_summary="",
                                              action_summary=f"Immediately closing game center by clicking button '取消'."),
                    past_action_summary="",
                    recent_repeat_action="",
                    recent_repeat_count=0,
                    page_semantic="",
                    subtask_candidate=[],
                    # random_exploration=False,
                    subtask=UserSubTask(sub_task=UserSubTaskTypeExploreApp(task="explore_app")),
                    reasoning="",
                ))
                self.skip_steps_by_preprocess += 1
                sleep(rand(0.5, 1))
                return self.sync_ui_screenshot(False)

        self.assert_running()
        self.status_label.config(text="Status: Post-processing UI")

        final_ui, vid_map = base_ui.post_process_ui(uist['elements'], base64_screenshot)

        self.assert_running()
        self.status_label.config(text=label)
        return uist, org_vid_map, base64_screenshot, final_ui, vid_map

    @time_consumed
    def rag_fine_grain(self, semantic_analyzed, base64_screenshot, uist, vid_map):
        rag_analyze_result = None
        if self.rag_db:
            semantic_like_ids = [elem.id for elem in semantic_analyzed.semantic_element if elem.id in vid_map]
            other_elements_ids = base_ui.get_semantic_like(uist['elements'])
            for elem_id in other_elements_ids:
                if len(semantic_like_ids) < 60 and elem_id not in semantic_like_ids:
                    semantic_like_ids.append(elem_id)
            semantic_like_ids.sort()
            semantic_like_els = [vid_map.get(eid) for eid in semantic_like_ids]
            semantic_like_ss = base_ui.crop_elements_from_screenshots(semantic_like_els, base64_screenshot)
            semantic_like_texts = base_ui.get_text_from_elements(semantic_like_els)

            ui_query_results = self.rag_db.query_ui(UIRecording(
                screenshot=base64_screenshot,
                view_hierarchy=uist['elements']
            ))
            element_query_results = self.rag_db.query_element(semantic_like_ss, semantic_like_texts)

            similarity = {}
            for els, similar_elements in zip(semantic_like_els, element_query_results):
                if similar_elements:
                    similarity[els['id']] = [{
                        'screenshot': s_elm.screenshot,
                        'ui_type': s_elm.ui_type
                    } for s_elm in similar_elements]
            similarity[0] = [{
                'screenshot': s_ui.screenshot,
                'ui_type': s_ui.ui_type
            } for s_ui in ui_query_results]
            self.similar_elements = similarity

            element_groups = []
            for els, ss, similar_elements in zip(semantic_like_els, semantic_like_ss, element_query_results):
                element_group = {
                    'target_id': els['id'],
                    'element_screenshot': ss,
                    'similar_elements': []
                }
                for s_elm in similar_elements:
                    element_group['similar_elements'].append({
                        'element_screenshot': s_elm.screenshot,
                        'description': 'Semantic type of similar element above: ' + s_elm.ui_type,
                    })
                element_groups.append(element_group)

            similar_uis = []
            for s_ui in ui_query_results:
                similar_uis.append({
                    'image': s_ui.screenshot,
                    'description': 'Semantic type of similar UI above: ' + s_ui.ui_type
                })

            rag_analyze = RAGView(rag_ui=None, rag_elements=None)
            group_size = 60
            for st_idx in range(0, len(element_groups), group_size):
                element_batch = element_groups[st_idx:st_idx + group_size]
                batch_rag_analyze = base_ui.rag_analyze(base64_screenshot, similar_ui=similar_uis, element_groups=element_batch)
                if st_idx == 0:
                    rag_analyze.rag_ui = batch_rag_analyze.rag_ui
                    rag_analyze.rag_elements = batch_rag_analyze.rag_elements
                else:
                    rag_analyze.rag_elements.element_groups.extend(batch_rag_analyze.rag_elements.element_groups)
            rag_analyze_result = rag_analyze
        return rag_analyze_result

    @time_consumed
    def fetch_ui_data(self, force_sync=True):
        """Fetch new UI & Screenshot data in background"""
        self.is_fetching_ui = True
        try:
            self.skip_steps_by_preprocess = 0
            uist, self.org_vid_map, base64_screenshot, final_ui, vid_map = self.sync_ui_screenshot()

            # supply_uist = copy.deepcopy(uist)
            # vid_map_before = copy.deepcopy(vid_map)
            # supply_ui, vid_map = base_ui.tree_supply(supply_uist['elements'], base64_screenshot)
            # supply_uist['elements'] = supply_ui

            self.org_uist = copy.deepcopy(uist)

            # Replace with current uielements
            uist['elements'] = final_ui

            self.assert_running()
            self.status_label.config(text="Status: Analyzing UI")

            # if self.random_exploration and self.remain_random_steps > 0:
            if False:
                logger.debug(f"Random exploration, remaining steps: {self.remain_random_steps}")
                supply_uist = copy.deepcopy(uist)
                vid_map_before = copy.deepcopy(vid_map)
                supply_ui, vid_map = base_ui.tree_supply(supply_uist['elements'], base64_screenshot)
                supply_uist['elements'] = supply_ui

                # Disable cache to avoid this information being used by semantic_analyze

                self.ui_graph.add_ui(supply_uist['elements'], base64_screenshot, None, None,
                                     description='[fetch_ui_data] UI synchronizing successfully. Random exploration without LLM analysis.')

                # Update & Cache the screenshot image data
                self.last_screenshot = base64.b64decode(base64_screenshot)

                self.uist_before_supply = uist
                self.vid_map_before_supply = vid_map_before
                self.vid_map = vid_map
                self.uist = supply_uist
                self.analyzed_ui = None
                self.semantic_analyzed = None

            else:
                # if self.rag_db:
                #     # RAG Assistance
                #     semantic_like_ids = base_ui.get_semantic_like(uist['elements'])
                #     semantic_like_els = [vid_map.get(eid) for eid in semantic_like_ids]
                #     semantic_like_ss = base_ui.crop_elements_from_screenshots(semantic_like_els, base64_screenshot)
                #     semantic_like_texts = base_ui.get_text_from_elements(semantic_like_els)
                #
                #     similarity = {}
                #
                #     # masks = self.rag_db.classify_element(semantic_like_ss)
                #     #
                #     # for els, mask in zip(semantic_like_els, masks):
                #     #     if not mask:
                #     #         similarity[els['id']] = "Not related to Dark Pattern"
                #     #
                #     # semantic_like_ids = [x for x, m in zip(semantic_like_ids, masks) if m]
                #     # semantic_like_els = [x for x, m in zip(semantic_like_els, masks) if m]
                #     # semantic_like_ss = [x for x, m in zip(semantic_like_ss, masks) if m]
                #     # semantic_like_texts = [x for x, m in zip(semantic_like_texts, masks) if m]
                #
                #     query_results = self.rag_db.query_element(semantic_like_ss, semantic_like_texts)
                #
                #     assistance_text_image = []
                #
                #     for els, similar_elements in zip(semantic_like_els, query_results):
                #         if similar_elements:
                #             assistance_text_image.append({
                #                 'description': f"The following images are similar to element {els['id']}"
                #             })
                #             assistance_text_image.extend({
                #                 'image': s_elm.screenshot,
                #                 'description': f"Type: {s_elm.ui_type}"
                #             } for s_elm in similar_elements)
                #
                #             similarity[els['id']] = [{
                #                 'screenshot': s_elm.screenshot,
                #                 'ui_type': s_elm.ui_type
                #             } for s_elm in similar_elements]
                #
                #     ui_query_results = self.rag_db.query_ui(UIRecording(
                #         screenshot=base64_screenshot,
                #         view_hierarchy=uist['elements']
                #     ))
                #
                #     assistance_text_image.append({
                #         'description': f"The following images are similar to the app's screenshot, please refer to these images to decide the app screenshot's type."
                #     })
                #     assistance_text_image.extend({
                #         'image': s_ui.screenshot,
                #         'description': f"Type: {s_ui.ui_type}"
                #     } for s_ui in ui_query_results)
                #
                #     similarity[0] = [{
                #         'screenshot': s_ui.screenshot,
                #         'ui_type': s_ui.ui_type
                #     } for s_ui in ui_query_results]
                #
                #     self.similar_elements = similarity
                #
                # else:
                #     assistance_text_image = []

                rag_analyzed = None
                # Check if UI has changed
                if self.enable_ui_cache and self.cached_uist:

                    ui_changed = not base_ui.is_ui_equal(
                        uist['elements'],
                        self.cached_uist['elements']
                    )

                    if not ui_changed and self.cached_analyzed:
                        logger.debug("UI unchanged, using cached analysis")
                        analyzed = self.cached_analyzed
                        # AdsDP: If there are too many important elements without ids, supply elements from visual cue (screenshot) by LayerOutCoder
                        supply_uist = self.cached_supply_uist
                        semantic_analyzed = self.cached_semantic_analyzed
                        rag_analyzed = self.cached_rag_analyzed
                        vid_map_before = copy.deepcopy(vid_map)
                    else:
                        # analyzed = base_ui.analyze_ui(final_ui, base64_screenshot)
                        supply_uist = copy.deepcopy(uist)
                        vid_map_before = copy.deepcopy(vid_map)
                        # if analyzed.count_elements_wo_id >= 1:
                        # if not analyzed.viewhierarchy_match:
                        supply_ui, vid_map = base_ui.tree_supply(supply_uist['elements'], base64_screenshot)
                        supply_uist['elements'] = supply_ui
                        analyzed = base_ui.analyze_ui(supply_uist['elements'], base64_screenshot)
                        semantic_analyzed = base_ui.semantic_analyze_ui(supply_uist['elements'], base64_screenshot, rag=None)
                        rag_analyzed = self.rag_fine_grain(semantic_analyzed, base64_screenshot, supply_uist, vid_map)

                        # Update cache
                        self.cached_uist = uist
                        self.cached_analyzed = analyzed
                        self.cached_supply_uist = supply_uist
                        self.cached_semantic_analyzed = semantic_analyzed
                        self.cached_rag_analyzed = rag_analyzed
                else:
                    # No cache or caching disabled
                    # analyzed = base_ui.analyze_ui(final_ui, base64_screenshot)
                    supply_uist = copy.deepcopy(uist)
                    vid_map_before = copy.deepcopy(vid_map)
                    # if analyzed.count_elements_wo_id >= 1:
                    # if not analyzed.viewhierarchy_match:
                    supply_ui, vid_map = base_ui.tree_supply(supply_uist['elements'], base64_screenshot)
                    supply_uist['elements'] = supply_ui
                    analyzed = base_ui.analyze_ui(supply_uist['elements'], base64_screenshot)
                    semantic_analyzed = base_ui.semantic_analyze_ui(supply_uist['elements'], base64_screenshot, rag=None)
                    rag_analyzed = self.rag_fine_grain(semantic_analyzed, base64_screenshot, supply_uist, vid_map)

                    # Update cache
                    self.cached_uist = uist
                    self.cached_analyzed = analyzed
                    self.cached_supply_uist = supply_uist
                    self.cached_semantic_analyzed = semantic_analyzed
                    self.cached_rag_analyzed = rag_analyzed

                # If the UI match rate is less than 0.5, re-synchronize
                if force_sync and analyzed.match_rate < 0.5:
                    logger.info(f"UI match rate {analyzed.match_rate} too low, re-synchronizing")
                    self.ui_graph.add_ui(supply_uist['elements'], base64_screenshot, analyzed, semantic_analyzed, description="[fetch_ui_data] UI match rate too low, re-synchronizing.", rag_analyze=rag_analyzed)
                    return self.fetch_ui_data(force_sync=False)

                # AdsDP added: Append UI Element detected by Gemini to UI Tree. But the bounding_box is unmatched with actual ui element...
                # append_ui, vid_map = base_ui.tree_append_element(uist['elements'], base64_screenshot, semantic_analyzed)
                # uist['elements'] = append_ui

                self.ui_graph.add_ui(supply_uist['elements'], base64_screenshot, analyzed, semantic_analyzed, description='[fetch_ui_data] UI synchronizing successfully.', rag_analyze=rag_analyzed)

                # if "login" in semantic_analyzed and semantic_analyzed["login"]:
                #     self.mark_task_status([UserSubTaskTypeRegister, UserSubTaskTypeLogin], TaskStatus.Complete)

                # Update & Cache the screenshot image data
                self.last_screenshot = base64.b64decode(base64_screenshot)

                self.uist_before_supply = uist
                self.vid_map_before_supply = vid_map_before
                self.vid_map = vid_map
                self.uist = supply_uist
                self.analyzed_ui = analyzed
                self.semantic_analyzed = semantic_analyzed
                self.rag_analyze = rag_analyzed

            self.data_queue.put(True)
            
        except Exception as e:
            logger.exception(f"Error fetching data")
            self.assert_running()
            self.data_queue.put(None)
            self.status_label.config(text="Status: Error")

        self.assert_running()
        self.is_fetching_ui = False
    
    @time_consumed
    def fetch_action(self):
        """Fetch new action data in background"""
        try:
            self.assert_running()
            self.status_label.config(text="Status: Analyzing Action")

            # if self.random_exploration and self.remain_random_steps > 0:
            if False:
                logger.debug("Random exploration without action analysis, select random clickable element instead.")
                uielements = self.uist['elements']
                relement = base_ui.get_random_clickable_element(uielements, self.keywords)
                self.actions = SemanticUserAction(
                    actions=[SemanticUserActionStep(reasoning="", target_element_id=relement["id"], reason_of_element="random select",
                                                    completeness_check="",
                                                    action=UserActionTypeClick(type="click", element_id=relement["id"], times=1))
                             ],
                    summary=UserActionSummary(ui_summary="",
                                              action_summary=f"Randomly select element (frame: {base_ui.get_frame(relement)}) to click."),
                    past_action_summary="",
                    recent_repeat_action="",
                    recent_repeat_count=0,
                    page_semantic="",
                    subtask_candidate=[],
                    # random_exploration=False,
                    subtask=UserSubTask(sub_task=UserSubTaskTypeExploreApp(task="explore_app")),
                    reasoning="",
                )

            else:
                similarity = []
                if self.rag_db:
                    # RAG Assistance
                    ui_action_history = self.history_record()
                    # ui_sequence = []
                    # start_index = max(0, len(ui_action_history) - 3)
                    # for ui_index, ui_or_action in enumerate(ui_action_history[-3:], start=start_index):
                    #     if isinstance(ui_or_action, self.UiGraph.GraphNode):
                    #         semantic_elements = []
                    #         if semantic_analyzed := ui_or_action.semantic_analyzed[ui_or_action.global_local_map[ui_index]]:
                    #             old_screenshot = ui_or_action.screenshot[ui_or_action.global_local_map[ui_index]]
                    #             old_uielements = ui_or_action.ui_tree[ui_or_action.global_local_map[ui_index]]
                    #             old_vid_map = base_ui.get_vid_map(old_uielements)
                    #
                    #             for sem_elm in semantic_analyzed.semantic_element:
                    #                 semantic_elements.append(UIElementRecording(
                    #                     ui_type=SemanticUIElementType(sem_elm.type),
                    #                     screenshot=base_ui.crop_elements_from_screenshots(old_vid_map[sem_elm.id], old_screenshot)
                    #                 ))
                    #
                    #         ui_sequence.append(UIRecording(
                    #             screenshot=ui_or_action.screenshot[ui_or_action.global_local_map[ui_index]],
                    #             uielements=semantic_elements,
                    #         ))
                    #     else:
                    #         ui_sequence.append(ActionRecording(
                    #             description=ui_or_action.actions.summary.action_summary
                    #         ))
                    # all_seqs = self.rag_db.query_action_seq(ui_sequence)

                    idx = len(ui_action_history) - 1
                    last_history = ui_action_history[idx]
                    all_seqs = self.rag_db.query_next_action(UIRecording(
                            screenshot=last_history.screenshot[last_history.global_local_map[idx]],
                            view_hierarchy=last_history.ui_tree[last_history.global_local_map[idx]]
                        ),
                        last_history.semantic_analyzed[last_history.global_local_map[idx]],
                        last_history.rag_analyze[last_history.global_local_map[idx]]
                    )

                    # assistance_text_image = []
                    if all_seqs:
                        for seq_idx, seq in enumerate(all_seqs):
                            # assistance_text_image.append({
                            #     'description': "Below is an UI screen similar to the current one, along with the action the user performed on it."
                            # })
                            # assistance_text_image.extend({
                            #     'image': ui_or_action.screenshot,
                            # } if isinstance(ui_or_action, UIRecording) else {
                            #     'description': 'action_type: ' + ui_or_action.type + 'target_element_location: ' + ui_or_action.location + \
                            #                    'text_in_target_element: ' + ui_or_action.text + 'target_element_description: ' + ui_or_action.description + '\n' # ActionRecording
                            # }
                            # for ui_or_action in seq)

                            for ui_or_action in seq:
                                if isinstance(ui_or_action, UIRecording):
                                    similarity.append({
                                        'screenshot': ui_or_action.screenshot,
                                        'ui_type': ui_or_action.ui_type,
                                    })
                                else:
                                    # similarity[-1]['action_description'] = 'action_type: ' + ui_or_action.type + (' target_element_location: ' + ui_or_action.location) if ui_or_action.location else '' + \
                                    #     (' text_in_target_element: ' + ui_or_action.text) if ui_or_action.text else '' + (' target_element_description: ' + ui_or_action.description) if ui_or_action.description else '' + '\n'
                                    desc = 'action_type: ' + ui_or_action.type
                                    if ui_or_action.location:
                                        desc += ' target_element_location: ' + ui_or_action.location
                                    if ui_or_action.text:
                                        desc += ' text_in_target_element: ' + ui_or_action.text
                                    if ui_or_action.description:
                                        desc += ' target_element_description: ' + ui_or_action.description
                                    desc += '\n'

                                    similarity[-1]['action_description'] = desc

                # else:
                    # assistance_text_image = []

                self.similar_ui = similarity

                raction = self.actions.recent_repeat_action if self.actions else ''
                rcount = self.actions.recent_repeat_count if self.actions else 0
                # self.actions = base_action.analyze_action(self.analyzed_ui, self.task, self.action_history, recent_repeat_action=raction, recent_repeat_count=rcount)
                self.actions = base_action.semantic_analyze_action(self.analyzed_ui, self.semantic_analyzed, self.task, self.action_history, self.ui_graph.extract_action_and_result(self.ui_graph.last_node), self.current_subtask, self.subtask_candidate, recent_repeat_action=raction, recent_repeat_count=rcount, rag=similarity)
                new_task_candidate = []
                for task_index, single_candidate in enumerate(self.actions.subtask_candidate):
                    subtask = single_candidate.sub_task
                    if task_index == 0:
                        if type(subtask) is type(self.subtask_candidate[0].sub_task):
                            candidate = TaskCandidate(sub_task=subtask, explored_step=self.subtask_candidate[0].explored_step+1,
                                                      steps_since_last_highest_priority=0, previously_completed=self.subtask_candidate[0].previously_completed,
                                                      previously_failed=self.subtask_candidate[0].previously_failed)
                        else:
                            candidate = TaskCandidate(sub_task=subtask, explored_step=0, steps_since_last_highest_priority=0,
                                                      previously_completed=False, previously_failed=False)
                        self.current_subtask = candidate
                    else:
                        if type(subtask) is type(self.subtask_candidate[0].sub_task):
                            candidate = TaskCandidate(sub_task=subtask, explored_step=0, steps_since_last_highest_priority=0,
                                                      previously_completed=True if self.actions.completed_or_failed=="completed" else False,
                                                      previously_failed=True if self.actions.completed_or_failed=="failed" else False)
                        else:
                            old_task = next((x for x in self.subtask_candidate if type(subtask) is type(x.sub_task)), None)
                            candidate = TaskCandidate(sub_task=subtask, explored_step=0,
                                                      steps_since_last_highest_priority=old_task.steps_since_last_highest_priority+1,
                                                      previously_completed=old_task.previously_completed,
                                                      previously_failed=old_task.previously_failed)
                    new_task_candidate.append(candidate)
                self.subtask_candidate = new_task_candidate

                # self.random_exploration = self.actions.random_exploration
                # self.remain_random_steps = self.actions.max_random_actions
                # self.keywords = self.actions.keywords

                # if self.random_exploration and self.remain_random_steps > 0:
                if False:
                    logger.debug("Random exploration without action analysis, select random clickable element instead.")
                    uielements = self.uist['elements']
                    relement = base_ui.get_random_clickable_element(uielements, self.keywords)
                    self.actions = SemanticUserAction(
                        actions=[SemanticUserActionStep(reasoning="", target_element_id=relement["id"],
                                                        reason_of_element="random select",
                                                        completeness_check="",
                                                        action=UserActionTypeClick(type="click", element_id=relement["id"], times=1))
                                 ],
                        summary=UserActionSummary(ui_summary="",
                                                  action_summary=f"Randomly select this element (frame: {base_ui.get_frame(relement)}) to click."),
                        past_action_summary="",
                        recent_repeat_action="",
                        recent_repeat_count=0,
                        page_semantic="",
                        subtask_candidate=[],
                        # random_exploration=False,
                        subtask=UserSubTask(sub_task=UserSubTaskTypeExploreApp(task="explore_app")),
                        reasoning="",
                    )

            self.action_queue.put(True)
        except Exception as e:
            logger.exception(f"Error fetching action")
            self.assert_running()
            self.action_queue.put(None)
            self.status_label.config(text="Status: Error")
    
    @time_consumed
    def apply_action(self):
        """Apply the action to the UI"""
        self.is_applying_action = True
        logger.debug(f"Before applying: ui_diff_count={self.ui_diff_count}, no_action_count={self.no_action_count}, last_restart={self.last_restart}, current_step={len(self.action_history)}")
        self.assert_running()

        try:
            self.status_label.config(text="Status: Checking UI")
            uist, org_vid_map, base64_screenshot, final_ui, vid_map = self.sync_ui_screenshot(False)
            if self.ui_diff_count < MAX_UI_DIFF_COUNT and not base_ui.is_ui_equal(final_ui, self.uist_before_supply['elements']):
                logger.info("UI has changed, actions not being applied.")
                self.assert_running()
                self.status_label.config(text="Status: Action(s) Not Applied")
                if self.analyzed_ui and self.analyzed_ui.feedback_message:
                    self.action_history.append('The UI shows feedback/status/error message: \'' + self.analyzed_ui.feedback_message + '\'')
                self.ui_diff_count += 1
                self.ui_change_history.append([final_ui, base64_screenshot])
                return
            if self.ui_diff_count >= MAX_UI_DIFF_COUNT:
                logger.info("Force applying action(s) since UI has changed too much.")
            self.ui_diff_count = 0
            self.ui_change_history = []
            self.status_label.config(text="Status: Applying Action(s)")
            try:
                # effective_actions = base_action.apply_actions(self.actions, self.vid_map, base_ui)
                effective_actions = base_action.apply_semantic_actions(self.actions, self.vid_map, base_ui)
            except Exception as e:
                logger.exception(f"Error applying action")
                effective_actions = 0
            if effective_actions is True:
                # Task completed!
                self.assert_running()
                self.status_label.config(text="Status: Task Completed")
                self.is_completed = True
                return
            if effective_actions == 0:
                self.no_action_count += 1
            else:
                self.no_action_count = 0
                self.ui_graph.add_action(self.actions, description="[apply_action] Apply actions.")
                # if isinstance(task := self.actions.subtask.sub_task, UserSubTaskTypeItemPurchase):
                #     self.cart.extend(task.items)
                # if self.random_exploration and self.remain_random_steps > 0:
                if False:
                    self.remain_random_steps -= 1
                    if self.remain_random_steps == 0:
                        self.random_exploration = False
                        self.keywords = None
            restart = False
            if self.no_action_count > MAX_NO_ACTION_COUNT:
                logger.info(f"Restart the app after {MAX_NO_ACTION_COUNT} times of no actions.")
                self.action_history.append('Restart the app since no actions are possible within the current UI.')
                restart = True
            if self.last_restart + MAX_STEPS_COUNT < len(self.action_history):
                logger.info(f"Restart the app after maximum steps count {MAX_STEPS_COUNT} reached.")
                self.action_history.append(f'Restart the app after maximum steps in an action sequence reached.')
                restart = True
            if restart:
                self._actions_history.append(None)
                self.no_action_count = 0
                self.last_restart = len(self.action_history)
                self.refresh_subtask()
                self.ui_graph.add_action(SemanticUserAction(
                    actions=[SemanticUserActionStep(reasoning="", target_element_id=-1, reason_of_element="",
                                                    completeness_check="",
                                                    action=UserActionTypeRestart(type="restart"))
                             ],
                    summary=UserActionSummary(ui_summary="",
                                              action_summary="[apply_action] System has decided to restart the app."),
                    past_action_summary="",
                    recent_repeat_action="",
                    recent_repeat_count=0,
                    page_semantic="",
                    subtask_candidate=[],
                    # random_exploration=False,
                    subtask=UserSubTask(sub_task=UserSubTaskTypeExploreApp(task="explore_app")),
                    reasoning="",
                ))
                base_action.restart_app()
                return
            self.action_history.append(self.actions.summary.ui_summary + (' ' if not self.analyzed_ui or not self.analyzed_ui.feedback_message or (self.analyzed_ui.ads and self.analyzed_ui.ads.is_ad_topmost) or self.analyzed_ui.feedback_message in self.actions.summary.ui_summary  else ' The UI shows feedback/status/error message: \'' + self.analyzed_ui.feedback_message + '\'. ') + self.actions.summary.action_summary)
            self._actions_history.append(self.actions)
        except Exception as e:
            logger.exception(f"Error applying action")
            self.assert_running()
            self.status_label.config(text="Status: Error")
        finally:
            self.is_applying_action = False

    @time_consumed
    def fetch_dp(self):
        '''Fetch new dp detection data in background'''
        self.dp_result = self.dp_detection()
        self.dp_label.config(text=f"DP Detection: Done")
        if not self.is_fetching_ui:
            self.assert_running()
            self.root.after(100, self.update_tree) # refresh ui_desc
        self.dp_thread = None
        return

    def dp_detection(self):
        # if self.random_exploration and self.remain_random_steps > 0:
        #     return DPList()

        ui_action_history = self.history_record()
        ui_sequence = []

        start_index = max(0, len(ui_action_history) - 5)
        for ui_index, ui_or_action in enumerate(ui_action_history[-5:], start=start_index): # visit the last 5 elements or actions
            if isinstance(ui_or_action, self.UiGraph.GraphNode):
                # if semantic_analyzed := ui_or_action.semantic_analyzed[ui_or_action.global_local_map[ui_index]]:
                #     ui_sequence.append({
                #         "type": "ui",
                #         "image": ui_or_action.screenshot[ui_or_action.global_local_map[ui_index]],
                #         "semantic_analyze": semantic_analyzed,
                #     })
                # else:
                #     ui_sequence.append({
                #         "type": "ui",
                #         "image": ui_or_action.screenshot[ui_or_action.global_local_map[ui_index]],
                #     })
                semantic_analyzed = ui_or_action.semantic_analyzed[ui_or_action.global_local_map[ui_index]]
                rag_analyzed = ui_or_action.rag_analyze[ui_or_action.global_local_map[ui_index]]
                analysis_ids = set([e.id for e in semantic_analyzed.semantic_element] + [e.id for e in rag_analyzed.rag_elements.element_groups] if rag_analyzed and rag_analyzed.rag_elements else [])
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

        dplist = base_ui.dp_detection(ui_sequence)
        return dplist

        # ui_action_history = self.history_record()
        # last_ui = ui_action_history[-1]
        # assert isinstance(last_ui, self.UiGraph.GraphNode) and last_ui.semantic_analyzed
        #
        # def find_previous_effective_element(index):
        #     nonlocal ui_action_history
        #
        #     for ui_index, ui in enumerate(ui_action_history[index-1::-1]):
        #         if isinstance(ui, self.UiGraph.GraphEdge):
        #             if any(isinstance(action, UserActionTypeRestart) for action in ui.actions):
        #                 return None, None
        #             continue
        #         else:
        #             if not ui.semantic_analyzed:
        #                 continue
        #             else:
        #                 return ui_index, ui
        #
        #     return None, None
        #
        # def find_previous_effective_edge(index):
        #     nonlocal ui_action_history
        #
        #     for ui_index, ui in enumerate(ui_action_history[index-1::-1]):
        #         if isinstance(ui, self.UiGraph.GraphEdge):
        #             if any(isinstance(action, UserActionTypeRestart) for action in ui.actions):
        #                 return None, None
        #             else:
        #                 return ui_index, ui
        #         else:
        #             if not ui.semantic_analyzed:
        #                 continue
        #             else:
        #                 return None, None
        #
        #     return None, None

        # dp_list_instance = self.DPList()
        # DarkPatternList = dp_list_instance.dplist
        # element = last_ui.get_semantic_analyzed(len(ui_action_history)-1).semantic_interface
        # if element.type == "Rating Interface":
        #     # Pop up Ad, Rate, Upgrade
        #     second_edge_index, second_edge_ui = find_previous_effective_edge(-1)
        #     if second_edge_ui and second_edge_ui.actions and isinstance(second_edge_ui.actions[-1], UserActionTypeClick):
        #         second_action = second_edge_ui.actions[-1]
        #         third_index, third_ui = find_previous_effective_element(second_edge_index)
        #         if third_ui and third_ui.get_semantic_analyzed(third_index).semantic_interface.status == CurrentStatus.IN_APP:
        #             if not any(
        #                     semantic_element.type == "Voluntary Rating Trigger Element" and semantic_element.id == second_action.element_id
        #                     for semantic_element in third_ui.get_semantic_analyzed(third_index).all_semantic_element):
        #                 DarkPatternList.append(self.DarkPatternBase(
        #                     dptype="Pop-up Ad, Rate, Upgrade",
        #                     indices=[third_index, second_edge_index, len(ui_action_history) - 1],
        #                     manifestation="Button-trigger, Rate"
        #                 ))
        #     if second_edge_ui and (not second_edge_ui.actions or all(
        #             isinstance(action, UserActionTypeIdle) for action in second_edge_ui.actions)):
        #         DarkPatternList.append(self.DarkPatternBase(
        #             dptype="Pop-up Ad, Rate, Upgrade",
        #             indices=[second_edge_index, len(ui_action_history) - 1],
        #             manifestation="Unprompted, Rate"
        #         ))
        #
        # if element.type == "Purchase Interface":
        #     # Pop up Ad, Rate, Upgrade
        #     second_edge_index, second_edge_ui = find_previous_effective_edge(-1)
        #     if second_edge_ui and second_edge_ui.actions and isinstance(second_edge_ui.actions[-1],
        #                                                                 UserActionTypeClick):
        #         second_action = second_edge_ui.actions[-1]
        #         third_index, third_ui = find_previous_effective_element(second_edge_index)
        #         if third_ui and third_ui.get_semantic_analyzed(
        #                 third_index).semantic_interface.status == CurrentStatus.IN_APP:
        #             if not any(
        #                     semantic_element.type == "Ad Removal Element" and semantic_element.id == second_action.element_id
        #                     for semantic_element in third_ui.get_semantic_analyzed(third_index).all_semantic_element):
        #                 DarkPatternList.append(self.DarkPatternBase(
        #                     dptype="Pop-up Ad, Rate, Upgrade",
        #                     indices=[third_index, second_edge_index, len(ui_action_history) - 1],
        #                     manifestation="Button-trigger, Upgrade"
        #                 ))
        #     if second_edge_ui and (not second_edge_ui.actions or all(
        #             isinstance(action, UserActionTypeIdle) for action in second_edge_ui.actions)):
        #         DarkPatternList.append(self.DarkPatternBase(
        #             dptype="Pop-up Ad, Rate, Upgrade",
        #             indices=[second_edge_index, len(ui_action_history) - 1],
        #             manifestation="Unprompted, Upgrade"
        #         ))
        #
        #     # Intermediate Currency
        #     if any(element.type == "Intermediate Currency" for element in last_ui.semantic_analyzed.all_semantic_element):
        #         DarkPatternList.append(self.DarkPatternBase(
        #             dptype="Intermediate Currency",
        #             indices=[len(ui_action_history) - 1],
        #         ))
        #
        # if element.type == "Full-Screen Ad Interface":
        #     # Pop up Ad, Rate, Upgrade
        #     # App Resumption Ads
        #     second_index, second_ui = find_previous_effective_element(-1)
        #     if second_ui and second_ui.get_semantic_analyzed(second_index).semantic_interface.status == CurrentStatus.OUT_OF_APP:
        #         third_index, third_ui = find_previous_effective_element(second_index)
        #         if third_ui and third_ui.get_semantic_analyzed(third_index).semantic_interface.status == CurrentStatus.IN_APP:
        #             DarkPatternList.append(self.DarkPatternBase(
        #                 # dptype="App Resumption Ads",
        #                 # indices = [third_index, second_index, len(ui_action_history) - 1],
        #                 dptype="Pop-up Ad, Rate, Upgrade",
        #                 indices=[third_index, second_index, len(ui_action_history) - 1],
        #                 manifestation="Resumption-trigger, Ad",
        #             ))
        #
        #     # Pop-up Ad, Rate, Upgrade
        #     # Unexpected Full-Screen Ads
        #     second_edge_index, second_edge_ui = find_previous_effective_edge(-1)
        #     if second_edge_ui and second_edge_ui.actions and isinstance(second_edge_ui.actions[-1], UserActionTypeClick):
        #         second_action = second_edge_ui.actions[-1]
        #         third_index, third_ui = find_previous_effective_element(second_edge_index)
        #         if third_ui and third_ui.get_semantic_analyzed(third_index).semantic_interface.status == CurrentStatus.IN_APP:
        #             if not any(semantic_element.type == "Voluntary Ad Trigger Element" and semantic_element.id == second_action.element_id for semantic_element in third_ui.get_semantic_analyzed(third_index).all_semantic_element):
        #                 DarkPatternList.append(self.DarkPatternBase(
        #                     # dptype="Unexpected Full-Screen Ads",
        #                     # indices = [third_index, second_edge_index, len(ui_action_history) - 1],
        #                     # manifestation="Button-trigger"
        #                     dptype="Pop-up Ad, Rate, Upgrade",
        #                     indices = [third_index, second_edge_index, len(ui_action_history) - 1],
        #                     manifestation="Button-trigger, Ad",
        #                 ))
        #     if second_edge_ui and (not second_edge_ui.actions or all(isinstance(action, UserActionTypeIdle) for action in second_edge_ui.actions)):
        #         DarkPatternList.append(self.DarkPatternBase(
        #             # dptype="Unexpected Full-Screen Ads",
        #             # indices = [second_edge_index, len(ui_action_history) - 1],
        #             # manifestation="Unprompted"
        #             dptype="Pop-up Ad, Rate, Upgrade",
        #             indices=[second_edge_index, len(ui_action_history) - 1],
        #             manifestation="Unprompted, Ad",
        #         ))
        #     # Ad without Exit Option
        #     if not any(semantic_element.type == "Close Button" for semantic_element in last_ui.get_semantic_analyzed(-1).all_semantic_element):
        #         DarkPatternList.append(self.DarkPatternBase(
        #             dptype="Ad without Exit Option",
        #             indices = [len(ui_action_history) - 1],
        #         ))
        #     # Ad Closure Failure
        #     second_edge_index, second_edge_ui = find_previous_effective_edge(-1)
        #     if second_edge_ui and second_edge_ui.actions and isinstance(second_edge_ui.actions[-1], UserActionTypeClick):
        #         third_index, third_ui = find_previous_effective_element(second_edge_index)
        #         if third_ui and third_ui.get_semantic_analyzed(third_index).semantic_interface.status == CurrentStatus.IN_AD:
        #             if any(semantic_element.type == "Close Button" and semantic_element.id == second_edge_ui.actions[-1].element_id for semantic_element in third_ui.get_semantic_analyzed(third_index).all_semantic_element):
        #                 DarkPatternList.append(self.DarkPatternBase(
        #                     dptype="Ad Closure Failure",
        #                     indices=[third_index, second_edge_index, len(ui_action_history) - 1]
        #                 ))
        #         elif ad_element := next((e for e in third_ui.get_semantic_analyzed(third_index).semantic_element if e.type == "Non-Full-Screen Ad"), None):
        #             if any(semantic_element.type == "Close Button" and semantic_element.id == second_edge_ui.actions[-1].element_id and base_ui.is_in(self.uist['elements'], semantic_element, ad_element) for semantic_element in third_ui.get_semantic_analyzed(third_index).all_semantic_element):
        #                 DarkPatternList.append(self.DarkPatternBase(
        #                     dptype="Ad Closure Failure",
        #                     indices=[third_index, second_edge_index, len(ui_action_history) - 1],
        #                 ))
        #     # Multiple Close Buttons
        #     if sum(1 for semantic_element in last_ui.get_semantic_analyzed(-1).all_semantic_element if semantic_element.type == "Close Button") > 1:
        #         DarkPatternList.append(self.DarkPatternBase(
        #             dptype="Multiple Close Buttons",
        #             indices = [len(ui_action_history) - 1],
        #         ))
        #     # Bias-Driven Ads
        #     if any(semantic_element.type == "Asymmetric Element Pairs" for semantic_element in last_ui.get_semantic_analyzed(-1).all_semantic_element):
        #         DarkPatternList.append(self.DarkPatternBase(
        #             dptype="Bias-Driven Ads",
        #             indices=[len(ui_action_history) - 1],
        #         ))
        #
        #     # Disguised Ads
        #     if any(semantic_element.type == "Disguised Element" for semantic_element in last_ui.get_semantic_analyzed(-1).all_semantic_element):
        #         DarkPatternList.append(self.DarkPatternBase(
        #             dptype="Disguised Ads",
        #             indices=[len(ui_action_history) - 1],
        #         ))
        #
        # # Auto-Redirect Ads
        # if element.type == "Landing Page":
        #     second_edge_index, second_edge_ui = find_previous_effective_edge(-1)
        #     if second_edge_ui and (not second_edge_ui.actions or all(isinstance(action, UserActionTypeIdle) for action in second_edge_ui.actions)):
        #         third_index, third_ui = find_previous_effective_element(second_edge_index)
        #         if third_ui and third_ui.get_semantic_analyzed(third_index).semantic_interface.status == CurrentStatus.IN_AD or any(semantic_element.type == "Non-Full-Screen Ad" for semantic_element in third_ui.get_semantic_analyzed(third_index).all_semantic_element):
        #             DarkPatternList.append(self.DarkPatternBase(
        #                 dptype="Auto-Redirect Ads",
        #                 indices=[third_index, second_edge_index, len(ui_action_history) - 1],
        #             ))
        #
        # for element_ind, element in enumerate(last_ui.get_semantic_analyzed(-1).all_semantic_element):
        #     # Paid Ad Removal
        #     if element.type == "Ad Removal Element":
        #         DarkPatternList.append(self.DarkPatternBase(
        #             dptype="Paid Ad Removal",
        #             indices = [len(ui_action_history) - 1],
        #         ))
        #     # Ad without Exit Option
        #     if element.type == "Non-Full-Screen Ad":
        #         if not any(semantic_element.type == "Close Button" and base_ui.is_in(self.uist['elements'], semantic_element, element) for semantic_element in last_ui.get_semantic_analyzed(-1).all_semantic_element):
        #             DarkPatternList.append(self.DarkPatternBase(
        #                 dptype="Ad without Exit Option",
        #                 indices = [len(ui_action_history) - 1],
        #             ))
        #     # Reward-Based Ads
        #     if element.type == "Voluntary Ad Trigger Element":
        #         for another_ind, another_element in enumerate(last_ui.get_semantic_analyzed(-1).all_semantic_element):
        #             if another_element.type == "Reward element" and base_ui.is_neighbor(self.uist['elements'], element, another_element):
        #                 DarkPatternList.append(self.DarkPatternBase(
        #                     dptype="Reward-Based Ads",
        #                     indices = [len(ui_action_history) - 1],
        #                 ))
        #                 break
        #     # Ad Closure Failure
        #     if element.type == "Non-Full-Screen Ad":
        #         second_edge_index, second_edge_ui = find_previous_effective_edge(-1)
        #         if second_edge_ui and second_edge_ui.actions and isinstance(second_edge_ui.actions[-1], UserActionTypeClick):
        #             third_index, third_ui = find_previous_effective_element(second_edge_index)
        #             if third_ui and (ad_element := next((e for e in third_ui.get_semantic_analyzed(third_index).all_semantic_element if e.type == "Non-Full-Screen Ad"), None)):
        #                 if any(semantic_element.type == "Close Button" and semantic_element.id == second_edge_ui.actions[-1].element_id
        #                        and base_ui.is_in(self.uist['elements'], semantic_element, ad_element) and base_ui.is_neighbor(self.uist['elements'], ad_element, element) for
        #                        semantic_element in third_ui.get_semantic_analyzed(third_index).all_semantic_element):
        #                     DarkPatternList.append(self.DarkPatternBase(
        #                         dptype="Ad Closure Failure",
        #                         indices=[third_index, second_edge_index, len(ui_action_history) - 1],
        #                     ))
        #     # Gesture-Induced Ad Redirection
        #     if element.type == "Shake Element":
        #         if last_ui.get_semantic_analyzed(-1).semantic_interface.status == CurrentStatus.IN_AD:
        #             DarkPatternList.append(self.DarkPatternBase(
        #                 dptype="Gesture-Induced Ad Redirection",
        #                 indices = [len(ui_action_history) - 1],
        #             ))
        #
        #     # Multiple Close Buttons
        #     if element.type == "Non-Full-Screen Ad":
        #         if sum(1 for semantic_element in last_ui.get_semantic_analyzed(-1).all_semantic_element if semantic_element.type == "Close Button" and base_ui.is_in(self.uist['elements'], semantic_element, element)):
        #             DarkPatternList.append(self.DarkPatternBase(
        #                 dptype="Multiple Close Buttons",
        #                 indices = [len(ui_action_history) - 1],
        #             ))
        #
        #     # Bias-Driven Ads
        #     if element.type == "Asymmetric Element Pairs":
        #         if any(another_element.type == "Non-Full-Screen Ad" and base_ui.is_in(self.uist['elements'], element, another_element) for another_element in last_ui.get_semantic_analyzed(-1).all_semantic_element):
        #             DarkPatternList.append(self.DarkPatternBase(
        #                 dptype="Bias-Driven Ads",
        #                 indices = [len(ui_action_history) - 1],
        #             ))
        #
        #     # Disguised Ads
        #     if element.type == "Disguised Element":
        #         if any(another_element.type == "Non-Full-Screen Ad" and base_ui.is_in(self.uist['elements'], element, another_element) for another_element in last_ui.get_semantic_analyzed(-1).all_semantic_element):
        #             DarkPatternList.append(self.DarkPatternBase(
        #                 dptype="Disguised Ads",
        #                 indices = [len(ui_action_history) - 1],
        #             ))
        #
        #     # Immortal Account
        #     if element.type == "Email or Call to Cancel Element":
        #         DarkPatternList.append(self.DarkPatternBase(
        #             dptype="Immortal Account",
        #             indices=[len(ui_action_history) - 1],
        #             manifestation="Cancel by Additional Operation"
        #         ))
        #
        # # Immortal Account
        # assert isinstance(ui_action_history[0], self.UiGraph.GraphNode)
        # has_login = False
        # has_delete_button = False
        # visited = set()
        # for ui_index, ui in enumerate(ui_action_history):
        #     if isinstance(ui, self.UiGraph.GraphNode):
        #         if any(task == UserSubTaskTypeLogin and status == TaskStatus.Complete for task, status in ui.tasks):
        #             has_login = True
        #         if any(element.type == "Delete Account Element" for element in ui.semantic_analyzed.all_semantic_element):
        #             has_delete_button = True
        #         if ui.semantic_analyzed.semantic_interface.type == "Account Interface" and has_login:
        #             if ui not in visited:
        #                 visited.add(ui)
        # if not has_delete_button and len(visited) >= 3:
        #     DarkPatternList.append(self.DarkPatternBase(
        #         dptype="Immortal Account",
        #         indices = [],
        #         manifestation="Hard to Find Cancel Methods"
        #     ))
        #
        # # Disguised Ads
        # for element in last_ui.get_semantic_analyzed(-1).semantic_element:
        #     if element.type == "Non-Full-Screen Ad":
        #         if base_ui.element_align(base_ui.get_vid_map(last_ui.ui_tree[-1])[element.id], last_ui.ui_tree[-1]) >= 2:
        #             DarkPatternList.append(self.DarkPatternBase(
        #                 dptype="Disguised Ads",
        #                 indices = [len(ui_action_history) - 1],
        #             ))
        #
        # # Sneak into Basket
        # # Hidden Costs
        # if "cart" in self.analyzed_ui and self.analyzed_ui.cart:
        #     matched_a = set()
        #     matched_b = set()
        #
        #     # 第一步：完全匹配的商品
        #     for i, item_a in enumerate(self.cart):
        #         for j, item_b in enumerate(self.analyzed_ui.cart):
        #             if j in matched_b:
        #                 continue
        #             if abs(item_a.price - item_b.price) < 0.05 and item_a.piece == item_b.piece \
        #                     and (item_a.name in item_b.name or item_b.name in item_a.name):
        #                 matched_a.add(i)
        #                 matched_b.add(j)
        #                 break
        #
        #     # 第二步：分析 B 中每个 item
        #     for j, item_b in enumerate(self.analyzed_ui.cart):
        #         if j in matched_b:
        #             continue  # 已匹配过
        #         has_any_match = False  # A中是否有任何未匹配商品可以匹配B
        #         has_same_name_diff_price = False  # 是否有同名商品但总价差>0.05
        #
        #         for i, item_a in enumerate(self.cart):
        #             if i in matched_a:
        #                 continue  # 已匹配过
        #             if abs(item_a.price - item_b.price) < 0.05:
        #                 has_any_match = True
        #             if item_a.name in item_b.name or item_b.name in item_a.name:
        #                 if item_b.price - item_a.price >= 0.05:
        #                     has_same_name_diff_price = True
        #
        #         if not has_any_match:
        #             DarkPatternList.append(self.DarkPatternBase(
        #                 dptype="Sneak into Basket",
        #                 indices=[len(ui_action_history) - 1],
        #             ))
        #         if has_same_name_diff_price:
        #             DarkPatternList.append(self.DarkPatternBase(
        #                 dptype="Drip Pricing, Hidden Costs, or Partitioned Pricing",
        #                 indices=[len(ui_action_history) - 1],
        #             ))

        # # Increased Ads with Use
        # ad_count_each_run = []
        # ad_count_now = 0
        # ad_indices = []
        # ad_indices_list = []
        # assert isinstance(ui_action_history[0], self.UiGraph.GraphNode)
        # current_status = None
        # for ui_index, ui in enumerate(ui_action_history):
        #     if isinstance(ui, self.UiGraph.GraphNode):
        #         if ui.semantic_analyzed.semantic_interface.status == CurrentStatus.IN_AD:
        #             if current_status != CurrentStatus.IN_AD:
        #                 ad_count_now += 1
        #                 ad_indices.append(ui_index)
        #         current_status = ui.semantic_analyzed.semantic_interface.status
        #     else:
        #         if any(isinstance(action, UserActionTypeRestart) for action in ui.actions):
        #             ad_count_each_run.append(ad_count_now)
        #             ad_indices_list.append(ad_indices)
        #             ad_count_now = 0
        #             ad_indices = []
        #             current_status = None
        #
        # # All task complete, add the last round's ad counts.
        # ad_count_each_run.append(ad_count_now)
        # ad_indices_list.append(ad_indices)
        #
        # if len(ad_count_each_run) > 1 and ad_count_each_run[-1] > ad_count_each_run[0] + 2 and ad_count_each_run[-1] > ad_count_each_run[0] * 2:
        #     DarkPatternList.append(self.DarkPatternBase(
        #         dptype="Increased Ads with Use",
        #         indices=[ad_indices_list[0], ad_indices_list[-1]]
        #     ))

        # return dp_list_instance

    def update_display(self):
        """Update all UI elements with new data"""
        try:
            if self._close_immediately:
                raise Exception("Window is closing")
            self.update_image()
            self.update_tree()
            self.update_action()
            self.last_refresh_time = time.time()
        except Exception as e:
            logger.exception(f"Error updating display")

    def update_action(self):
        """Update the action display"""
        step_text = 'Task: \n' + self.task + '\n\n'
        step_text += 'Subtask: \n' + json.dumps([candidate.model_dump() for candidate in self.subtask_candidate], indent=0, ensure_ascii=False) + '\n\n'
        step_text += 'Action History: \n' + base_action._get_action_history(self.action_history) + '\n\n'
        if self.actions:
            step_text += 'Proposed Action: \n' + json.dumps(self.actions.model_dump(), indent=0, ensure_ascii=False) + '\n\n'
        # step_text += 'Slim UI: \n' + base_action._get_ui_slim(self.analyzed_ui) + '\n\n'
        step_text += 'Slim UI: \n' + base_action._get_ui_slim(self.semantic_analyzed) + '\n\n' if self.semantic_analyzed else ''
            
        # Update the action text display
        self.step_desc.delete('1.0', tk.END)
        self.step_desc.insert(tk.END, step_text)

    def calculate_scale_factor(self, img_width, img_height):
        """Calculate scale factor to fit image in canvas"""
        canvas_width = max(1, self.left_frame.winfo_width() - 10)  # Account for padding
        canvas_height = max(1, self.left_frame.winfo_height() - 10)
        
        width_scale = canvas_width / max(1, img_width)
        height_scale = canvas_height / max(1, img_height + 20) # Make bottom visible
        return min(width_scale, height_scale)

    def update_image(self):
        """Update the screenshot display"""
        try:
            self.is_resize_done = True

            # Use cached image data instead of reading from disk
            if not self.last_screenshot:
                logger.error("No screenshot data available")
                return
            
            # Create image from cached data
            from io import BytesIO
            img = Image.open(BytesIO(self.last_screenshot))
            
            # Ensure valid dimensions
            if img.size[0] <= 0 or img.size[1] <= 0:
                logger.exception("Invalid image dimensions")
                return
                
            # Calculate scale factor based on actual image size
            self.scale_factor = self.calculate_scale_factor(*img.size)
            
            # Scale image
            new_size = (max(1, int(img.size[0] * self.scale_factor)),
                       max(1, int(img.size[1] * self.scale_factor)))
            
            img = img.resize(new_size, Image.Resampling.LANCZOS)
            self.photo = ImageTk.PhotoImage(img)
            
            self.canvas.delete("all")
            self.canvas.create_image(5, 5, image=self.photo, anchor=tk.NW)  # Add 5px padding
            
            # Update canvas size to match image
            self.canvas.config(width=new_size[0]+10, height=new_size[1]+10)  # Add padding

            # Restore highlight if there was one
            if self.current_element_id and self.vid_map:
                self.highlight_element(self.current_element_id)
            
        except Exception as e:
            logger.exception(f"Error updating image")

    def highlight_element(self, element_id):
        """Highlight an element in the screenshot"""
        if not self.vid_map or element_id not in self.vid_map:
            return
        
        # Remove existing highlight
        if self.highlight_rect:
            self.canvas.delete(self.highlight_rect)
            self.highlight_rect = None
        
        # Get element frame
        frame = base_ui.get_frame(self.vid_map[element_id])
        
        # Scale coordinates and add padding consistently
        padding = 5  # Same padding as used in create_image
        x1 = int(frame['x'] * self.scale_factor) + padding
        y1 = int(frame['y'] * self.scale_factor) + padding
        x2 = x1 + int(frame['width'] * self.scale_factor)
        y2 = y1 + int(frame['height'] * self.scale_factor)
        
        # Create new highlight rectangle
        self.highlight_rect = self.canvas.create_rectangle(
            x1, y1, x2, y2,
            outline="red",  # Use red for better visibility
            width=2,        # Make line thicker
            dash=(5, 5)     # Use dashed line for better visibility
        )
        
        self.current_element_id = element_id

    def update_tree(self):
        """Update the UI element tree"""
        try:
            self.refresh_tree_view()
        except Exception as e:
            logger.exception(f"Error updating tree")

    def update_refresh_time(self):
        """Update the refresh time label every second"""
        if self.last_refresh_time:
            elapsed = int(time.time() - self.last_refresh_time)
            self.refresh_label.config(text=f"Refreshed {elapsed} seconds ago")
        self.root.after(1000, self.update_refresh_time)

    def on_tree_select(self, event):
        """Handle tree selection"""
        selection = self.tree.selection()
        if selection and self.vid_map:
            item = self.tree.item(selection[0])
            element_id = item['values'][0]
            
            if element_id:  # Only process if element has an ID
                # Highlight the selected element
                self.highlight_element(element_id)
                
                # Find selected element in vid_map
                element_info = self.vid_map.get(element_id)
                
                # Find if it's an analyzed element
                analyzed_element = next((e for e in self.analyzed_ui.elements if e.id == element_id), None) if self.analyzed_ui else None
                semantic_element = next((e for e in self.semantic_analyzed.elements if e.id == element_id), None) if self.semantic_analyzed else None
                rag_element = next((e for e in self.rag_analyze.rag_elements.element_groups if e.id == element_id), None) if self.rag_analyze and self.rag_analyze.rag_elements else None

                # Update details
                self.details_text.delete(1.0, tk.END)
                for widget in self.preview_frame.winfo_children():
                    widget.destroy()
                self._preview_image = []
                
                if element_info:
                    details = f"[{element_id}] {element_info.get('type', 'Unknown')}\n"
                    details += f"Position: ({base_ui.get_frame(element_info)['x']}, {base_ui.get_frame(element_info)['y']})\n"
                    details += f"Size: {base_ui.get_frame(element_info)['width']}x{base_ui.get_frame(element_info)['height']}\n"
                    if 'name' in element_info:
                        details += f"Name: {element_info['name']}\n"
                    details += "\n"
                    if analyzed_element:
                        details += f"\nAnalyzed as:\n"
                        details += f"Type: {analyzed_element.ui_type.value}\n"
                        details += f"Description: {analyzed_element.description}\n"
                        details += f"Confidence: {analyzed_element.clickability}\n"
                        #details += f"Click Action: {analyzed_element.click_action}\n"
                        details += f"Text: {analyzed_element.text}\n"
                        details += f"Location: {analyzed_element.location}\n"
                        details += "\n"
                    if semantic_element:
                        details += f"\nSemantic as:\n"
                        details += f"Type: {semantic_element.ui_type.value}\n"
                        details += f"Description: {semantic_element.description}\n"
                        details += f"Confidence: {semantic_element.clickability}\n"
                        details += f"Text: {semantic_element.text}\n"
                        details += f"Location: {semantic_element.location}\n"
                        details += f"Semantic Type: {semantic_element.type}\n"
                        details += f"Evidence: {semantic_element.evidence}\n"
                        details += "\n"
                    if rag_element:
                        details += f"\n RAG Analysis as:\n"
                        details += f"Type: {rag_element.type}\n"
                        details += f"Detailed Analysis:\n"
                        for e_idx, detailed_analysis in enumerate(rag_element.detailed_analyze):
                            details += f"Similar element {e_idx}, Comparison: {detailed_analysis.similarity_and_difference}\n"
                        details += "\n"
                    elif element_id == 1 and self.rag_analyze and self.rag_analyze.rag_ui:
                        details += f"\n RAG Analysis as:\n"
                        details += f"Type: {self.rag_analyze.rag_ui.type}\n"
                        details += f"Detailed Analysis:\n"
                        for u_idx, detailed_analysis in enumerate(self.rag_analyze.rag_ui.detailed_analyze):
                            details += f"Similar UI {u_idx}, Comparison: {detailed_analysis.similarity_and_difference}\n"
                        details += "\n"

                    details += "\n"
                    details += json.dumps(element_info | {'subviews': []}, indent=2, ensure_ascii=False, sort_keys=True)
                    details += "\n\n"
                    details += json.dumps(self.org_vid_map.get(element_info.get('_id', element_info['id']), {}) | {'subviews': []}, indent=2, ensure_ascii=False, sort_keys=True)
                    
                    self.details_text.insert(tk.END, details)

                if element_id in self.similar_elements or element_id == 1:
                    similar_elements = []
                    if element_id in self.similar_elements:
                        similar_elements = self.similar_elements.get(element_id)
                    elif element_id == 1 and 0 in self.similar_elements.keys():
                        similar_elements = self.similar_elements.get(0)

                    if isinstance(similar_elements, str):
                        ttk.Label(self.preview_frame, text=similar_elements).grid(row=0, column=0)
                    else:
                        for similar_element in similar_elements:
                            pil_img = Image.open(io.BytesIO(base64.b64decode(similar_element['screenshot'])))
                            w, h = pil_img.size
                            max_w = 300
                            if w > max_w:
                                new_h = int(h * (max_w / w))
                                pil_img = pil_img.resize((max_w, new_h), Image.LANCZOS)
                            img_tk = ImageTk.PhotoImage(pil_img)
                            self._preview_image.append(img_tk)

                            row = len(self._preview_image) - 1
                            row_frame = ttk.Frame(self.preview_frame)
                            row_frame.grid(row=row, column=0, sticky="w", pady=2)

                            ttk.Label(row_frame, image=img_tk).grid(row=0, column=0, padx=5)
                            ttk.Label(row_frame, text=similar_element['ui_type']).grid(row=0, column=1, sticky="w")

                # if self.similar_ui:
                #     for similar_ui in self.similar_ui:
                #         pil_img = Image.open(io.BytesIO(base64.b64decode(similar_ui['screenshot'])))
                #         w, h = pil_img.size
                #         max_w = 500
                #         if w > max_w:
                #             new_h = int(h * (max_w / w))
                #             pil_img = pil_img.resize((max_w, new_h), Image.LANCZOS)
                #         img_tk = ImageTk.PhotoImage(pil_img)
                #         self._preview_image.append(img_tk)
                #         row = len(self._preview_image) - 1
                #         row_frame = ttk.Frame(self.preview_frame)
                #         row_frame.grid(row=row, column=0, sticky="w", pady=2)
                #
                #         ttk.Label(row_frame, image=img_tk).grid(row=0, column=0, padx=5)
                #         ttk.Label(row_frame, text="\n".join([similar_ui['ui_type'], similar_ui['action_description']])).grid(row=0, column=1, sticky="w")


    def on_canvas_click(self, event):
        """Handle clicks on the screenshot"""
        if not self.vid_map:
            return
        
        # Get click coordinates relative to canvas
        click_x = event.x - 5  # Adjust for padding
        click_y = event.y - 5
        
        # Convert click coordinates back to original scale
        orig_x = int(click_x / self.scale_factor)
        orig_y = int(click_y / self.scale_factor)
        
        suitable_element, depth, size = self.find_suitable_element(orig_x, orig_y)
        
        if suitable_element:
            element_id = suitable_element['id']
            logger.info(f"Clicked suitable element {element_id} at depth {depth} of size {size}")
            
            # Check if this element is analyzed
            # analyzed_element = next((e for e in self.analyzed_ui.elements if e.id == element_id), None)
            
            # Highlight the element
            self.highlight_element(element_id)
            
            # Find and select the element in tree
            def find_and_select_in_tree(items):
                for item in items:
                    values = self.tree.item(item)['values']
                    if values and values[0] == element_id:
                        self.tree.selection_set(item)
                        self.tree.see(item)
                        return True
                    if find_and_select_in_tree(self.tree.get_children(item)):
                        return True
                return False
            
            find_and_select_in_tree(self.tree.get_children())

    def on_window_resize(self, event):
        """Handle window resize events"""
        # Only update if it's the root window being resized
        if event.widget == self.root and self.is_resize_done:
            self.is_resize_done = False
            # Add delay to avoid too frequent updates
            self.root.after(100, self.update_image)

    def on_closing(self):
        """Handle window closing"""
        logger.info("Shutting down application...")

        # Disable auto refresh
        self.auto_refresh.set(False)
        self.auto_apply.set(True)

        # Stop any running threads
        # if hasattr(self, 'mask_thread') and (m := self.mask_thread) and m.is_alive():
        #     self.call_blocking(m.join)
        if hasattr(self, 'dp_thread') and (m := self.dp_thread) and m.is_alive():
            self.call_blocking(m.join)

        # Wait for any pending updates
        start_time = time.time()
        while self.is_updating:
            if time.time() - start_time > 30:
                break
            time.sleep(0.1)
            self.root.update()
        
        # Force stopping
        self.running = False

        # Stop any running threads
        # if hasattr(self, 'mask_thread') and (m := self.mask_thread) and m.is_alive():
        #     self.call_blocking(m.join)
        if hasattr(self, 'dp_thread') and (m := self.dp_thread) and m.is_alive():
            self.call_blocking(m.join)

        # Wait for any pending updates
        start_time = time.time()
        while self.is_updating or self.is_fetching_ui or self.is_applying_action: # self.is_updating:
            if time.time() - start_time > 30:
                break
            time.sleep(0.1)
            self.root.update()

        # Destroy all widgets properly
        for widget in self.root.winfo_children():
            widget.destroy()

        # Cancel any pending after callbacks
        for after_id in self.root.tk.eval('after info').split():
            try:
                self.root.after_cancel(after_id)
            except Exception as e:
                logger.exception(f"Error canceling after callback {after_id}")

        if self.graph_shown:
            # if tmp_file := self.graph_shown.tmpfile:
            #     import os
            #     os.remove(tmp_file.name)
            del self.graph_shown
        
        root = self.root

        # Clear all variables
        # for attr in vars(self).keys():
        #     if not attr.startswith('__') and not callable(getattr(self, attr)):
        #         try:
        #             setattr(self, attr, None)
        #         except:
        #             pass
        
        root.quit()
        root.destroy()
        import gc; gc.collect()

    def refresh_tree_view(self):
        """Refresh tree view with full UI tree while preserving expanded state"""
        if not self.vid_map: # or not self.analyzed_ui:
            return
        
        # Store current tree state
        expanded_items = {}
        selected_items = self.tree.selection()
        
        def store_expanded_state(items):
            for item in items:
                if self.tree.item(item)['open']:
                    # Store the item's values to identify it later
                    values = self.tree.item(item)['values']
                    if values and values[0]:  # If item has an ID
                        expanded_items[values[0]] = True
                store_expanded_state(self.tree.get_children(item))
        
        store_expanded_state(self.tree.get_children())

        def get_element_label(element):
            """Get the label for an element"""
            # Check if this element is analyzed
            analyzed_element = next((e for e in self.analyzed_ui.elements if e.id == element.get('id')), None) if self.analyzed_ui.elements else None
            semantic_element = next((e for e in self.semantic_analyzed.elements if e.id == element.get('id')), None) if self.semantic_analyzed.elements else None
            rag_element = next((e for e in self.rag_analyze.rag_elements.element_groups if e.id == element.get('id')), None) if self.rag_analyze and self.rag_analyze.rag_elements else None
            
            # Prepare display text and tags
            element_id = element.get('id', '')
            text = f"[{element_id}] {element.get('type', 'Unknown')}"
            text += " " + ("+" if element.get('ocr_text', '') else "-") + " "
            if 'name' in element and element['name']:
                name = (element['name'] if isinstance(element['name'], str) else ' '.join(element['name'])).replace('\n', ' ').replace('\r', '').replace('\t', ' ')
                text += f"{name[:30]}"
            if analyzed_element:
                text += f" Anlz: ({analyzed_element.ui_type.value})"
            if semantic_element:
                text += f" Sem: ({semantic_element.type})"
            if element_id == 1 and self.semantic_analyzed:
                text += f" Sem: ({self.semantic_analyzed.semantic_interface.type})"
            if rag_element:
                text += f" Rag: ({rag_element.type})"
            if element_id == 1 and self.rag_analyze and self.rag_analyze.rag_ui:
                text += f" Rag: ({self.rag_analyze.rag_ui.type})"
            
            return text
        
        # Check if UI structure has changed by comparing labels
        current_labels = set()
        def collect_labels(items):
            for item in items:
                label = self.tree.item(item)['text']
                current_labels.add(label)
                collect_labels(self.tree.get_children(item))
        
        collect_labels(self.tree.get_children())
        
        new_labels = set()
        def collect_new_labels(element):
            # Build label in same format as tree view
            new_labels.add(get_element_label(element))
            
            for subview in element.get('subviews', []):
                collect_new_labels(subview)
        
        collect_new_labels(self.uist['elements'])
        
        ui_changed = current_labels != new_labels

        logger.debug(f"UI changed: {ui_changed}")
        
        selected_ids = []

        if ui_changed:
            # Clear selection, highlight and descriptions
            self.current_element_id = None
            if self.highlight_rect:
                self.canvas.delete(self.highlight_rect)
                self.highlight_rect = None
            self.details_text.delete(1.0, tk.END)
            for widget in self.preview_frame.winfo_children():
                widget.destroy()
            self._preview_image = []
            selected_items = []  # Don't restore selection if UI changed
            expanded_items = {}  # Don't restore expanded state if UI changed
        else:
            # Get IDs of previously selected items
            for item in selected_items:
                try:
                    values = self.tree.item(item)['values']
                    if values:
                        selected_ids.append(values[0])
                except:
                    continue
        
        # Clear and rebuild tree
        self.tree.delete(*self.tree.get_children())

        analyzed_paths = set()  # Track paths to analyzed elements
        
        def collect_analyzed_paths(element, path=[]):
            """Collect all paths to analyzed elements"""
            current_path = path + [element.get('id', '')]

            # Check if this element is analyzed
            if self.analyzed_ui and next((e for e in self.analyzed_ui.elements if e.id == element.get('id')), None):
                analyzed_paths.update(current_path)

            # Check children
            for subview in element.get('subviews', []):
                collect_analyzed_paths(subview, current_path)

        # First collect all paths to analyzed elements
        collect_analyzed_paths(self.uist['elements'])

        semantic_paths = set()

        def collect_semantic_paths(element, path=[]):
            current_path = path + [element.get('id', '')]

            # Check if this element is analyzed
            if self.semantic_analyzed and next((e for e in self.semantic_analyzed.elements if e.id == element.get('id')), None):
                semantic_paths.update(current_path)

            # Check children
            for subview in element.get('subviews', []):
                collect_semantic_paths(subview, current_path)

        collect_semantic_paths(self.uist['elements'])

        rag_paths = set()

        def collect_rag_paths(element, path=[]):
            current_path = path + [element.get('id', '')]

            # Check if this element is analyzed
            if self.rag_analyze and self.rag_analyze.rag_elements and next(
                    (e for e in self.rag_analyze.rag_elements.element_groups if e.id == element.get('id')), None):
                rag_paths.update(current_path)

            # Check children
            for subview in element.get('subviews', []):
                collect_rag_paths(subview, current_path)

        collect_rag_paths(self.uist['elements'])
        
        def insert_element(element, parent=""):
            # Check if this element is analyzed
            analyzed_element = next((e for e in self.analyzed_ui.elements
                                     if e.id == element.get('id')), None) if self.analyzed_ui.elements else None
            semantic_element = next((e for e in self.semantic_analyzed.elements
                                     if e.id == element.get('id')), None) if self.semantic_analyzed.elements else None
            rag_element = next((e for e in self.rag_analyze.rag_elements.element_groups if e.id == element.get('id')),
                               None) if self.rag_analyze and self.rag_analyze.rag_elements else None

            # Prepare display text and tags
            element_id = element.get('id', '')
            text = get_element_label(element)
            
            # tags = ('analyzed',) if analyzed_element else ()
            tags = []
            if analyzed_element:
                tags.append('analyzed')
            if semantic_element or element_id == 1 and self.semantic_analyzed.semantic_interface:
                tags.append('semantic')
            if rag_element or element_id == 1 and self.rag_analyze and self.rag_analyze.rag_elements:
                tags.append('rag')
            tags = tuple(tags)
            
            # Determine if this item should be expanded
            should_expand = (
                element_id in expanded_items or  # Was previously expanded
                element_id in analyzed_paths or  # Is in path to analyzed element
                element_id in semantic_paths or  # Is semantic analyzed element
                element_id in rag_paths          # RAG analyzed element
            )
            
            # Insert element with only text
            item_id = self.tree.insert(parent, "end", 
                                     text=text,
                                     values=(element_id,),  # Keep ID in values for lookup
                                     tags=tags,
                                     open=should_expand)
            
            # Insert children
            for subview in element.get('subviews', []):
                insert_element(subview, item_id)
            
            return item_id
        
        # Configure tag for highlighted items
        self.tree.tag_configure('analyzed', foreground='blue', font=('TkDefaultFont', 9, 'bold'))
        self.tree.tag_configure('semantic', foreground='red', font=('TkDefaultFont', 9, 'bold'))
        self.tree.tag_configure('rag', foreground='green', font=('TkDefaultFont', 9, 'bold'))
        
        # Insert full tree
        insert_element(self.uist['elements'])
        
        # Restore selection if items still exist
        def find_item_by_id(items, target_id):
            """Recursively search for item with matching ID"""
            for item in items:
                values = self.tree.item(item)['values']
                if values and values[0] == target_id:
                    return item
                # Search children
                child_result = find_item_by_id(self.tree.get_children(item), target_id)
                if child_result:
                    return child_result
            return None

        # Find and select items with matching IDs
        for element_id in selected_ids:
            found_item = find_item_by_id(self.tree.get_children(), element_id)
            if found_item:
                self.tree.selection_add(found_item)
                self.tree.see(found_item)

        # if not self.analyzed_ui:
        if False:
            desc_text = "Random exploration without LLM analysis.\n"
        else:
            # Build the description text
            desc_text = ""
            desc_text += "LLM Analysis:\n"
            desc_text += self.analyzed_ui.description
            desc_text += "\n\nFeedback message:\n"
            desc_text += self.analyzed_ui.feedback_message
            desc_text += "\n\nIs alert topmost: " + str(self.analyzed_ui.is_alert_topmost)
            if self.analyzed_ui.ads:
                desc_text += "\nIs ad topmost: " + str(self.analyzed_ui.ads.is_ad_topmost)
                desc_text += "\nIs ad closeable: " + str(self.analyzed_ui.ads.is_ad_closeable)
                desc_text += "\nAd close button id: " + str(self.analyzed_ui.ads.ad_close_button_id)

            desc_text += "\n\nLLM Semantic Analysis:\n"
            desc_text += self.semantic_analyzed.description

            # if self.mask_result:
            #     desc_text += "\n\nMask Detection: " + self.mask_result.result + ', ' + self.mask_result.reasoning
            if self.dp_result:
                desc_text += "\n\nDP result:\n"
                desc_text += json.dumps(self.dp_result.model_dump(mode="json"))

            desc_text += "\n\nUI Tree Statistics:\n"

            total_elements = self.count_elements(self.uist['elements'])
            analyzed_elements = len(self.analyzed_ui.elements)
            # analyzed_elements = len(self.semantic_analyzed.elements)
            desc_text += f"Total elements: {total_elements}\n"
            desc_text += f"Match rate: {self.analyzed_ui.match_rate}\n"
            # desc_text += f"Match rate: {self.semantic_analyzed.match_rate}\n"
            desc_text += f"Analyzed elements: {analyzed_elements}\n"
            desc_text += f"Analysis ratio: {analyzed_elements/total_elements*100:.1f}%\n\n"

            # Add screen info
            if 'screenscale' in self.uist:
                desc_text += f"Screen scale: {self.uist['screenscale']}x\n"

            # Add analyzed elements summary
            desc_text += "\nAnalyzed Elements Summary:\n"
            for element in self.analyzed_ui.elements:
                if element.id >= 0:
                    frame = base_ui.get_frame(self.vid_map[element.id])
                    desc_text += (f"[{element.id}] {element.ui_type.value} at ({frame['x']}, {frame['y']}) "
                                f"size {frame['width']}x{frame['height']}\n")
                else:
                    desc_text += f"[Element not in UI Tree before supplement] {element.ui_type.value}\n"
        
        # Update the text widget
        self.ui_desc.delete(1.0, tk.END)
        self.ui_desc.insert(tk.END, desc_text)

    def count_elements(self, element):
        """Count total number of elements in UI tree"""
        count = 1  # Count current element
        for subview in element.get('subviews', []):
            count += self.count_elements(subview)
        return count

    def deselect_element(self):
        """Deselect current element"""
        self.tree.selection_remove(*self.tree.selection())
        if self.highlight_rect:
            self.canvas.delete(self.highlight_rect)
            self.highlight_rect = None
        self.current_element_id = None
        self.details_text.delete(1.0, tk.END)
        for widget in self.preview_frame.winfo_children():
            widget.destroy()
        self._preview_image = []

    def show_canvas_menu(self, event):
        """Show context menu for canvas right-click"""
        # Get click coordinates relative to canvas
        click_x = event.x - 5  # Adjust for padding
        click_y = event.y - 5
        
        # Convert to original coordinates
        orig_x = int(click_x / self.scale_factor)
        orig_y = int(click_y / self.scale_factor)
        
        self.last_right_click_pos = (orig_x, orig_y)
        
        # Find suitable element at click position
        suitable_element, depth, size = self.find_suitable_element(orig_x, orig_y)
        
        # Clear and rebuild menu
        self.canvas_menu.delete(0, tk.END)
        
        # Always show position
        self.canvas_menu.add_command(
            label=f"Position: ({orig_x}, {orig_y})"
        )
        
        if suitable_element:
            # Show element info if found
            self.canvas_menu.add_separator()
            self.canvas_menu.add_command(
                label=f"Element: [{suitable_element['id']}] at depth {depth} of size {size}"
            )
        
        # Always show click and deselect options
        self.canvas_menu.add_separator()
        self.canvas_menu.add_command(
            label="Deselect",
            command=self.deselect_element
        )
        self.canvas_menu.add_command(
            label="Click Here",
            command=self.click_position
        )
        
        self.canvas_menu.tk_popup(event.x_root, event.y_root)

    def show_tree_menu(self, event):
        """Show context menu for tree view right-click"""
        # Get the item under cursor
        item = self.tree.identify('item', event.x, event.y)
        if not item:
            return
        
        # Get element info
        values = self.tree.item(item)['values']
        if not values or not values[0]:
            return
        
        element_id = values[0]
        element_info = self.vid_map.get(int(element_id))  # Convert id to int
        if not element_info:
            return
        
        # Store element for click operation
        self.last_right_click_element = element_info
        
        # Build menu
        self.element_menu.delete(0, tk.END)
        self.element_menu.add_command(
            label=f"Element: [{element_id}]"
        )
        self.element_menu.add_separator()
        self.element_menu.add_command(
            label="Deselect",
            command=self.deselect_element
        )
        self.element_menu.add_command(
            label="Click Center",
            command=self.click_element_center
        )

        self.element_menu.tk_popup(event.x_root, event.y_root)

    def find_suitable_element(self, x, y):
        """Find the most suitable (smallest containing) element at the given coordinates"""
        if not self.vid_map:
            return None, 0
            
        return base_ui.find_suitable_element(self.uist['elements'], x, y, 0, 0)

    def click_position(self):
        """Click the last right-clicked position using ZXTouch"""
        if self.last_right_click_pos:
            x, y = self.last_right_click_pos
            try:
                base_action.tap(x, y)
                logger.info(f"Clicked position ({x}, {y})")
            except Exception as e:
                logger.exception(f"Failed to click position")

    def click_element_center(self):
        """Click the center of the last right-clicked element using ZXTouch"""
        if self.last_right_click_element:
            frame = base_ui.get_frame(self.last_right_click_element)
            center_x = frame['x'] + frame['width'] // 2
            center_y = frame['y'] + frame['height'] // 2
            try:
                base_action.tap(center_x, center_y)
                logger.info(f"Clicked element center at ({center_x}, {center_y})")
            except Exception as e:
                logger.exception(f"Failed to click element")

