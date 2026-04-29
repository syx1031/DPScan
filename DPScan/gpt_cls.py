import copy
from typing import Dict, List, Literal, Optional, Union, ClassVar, Tuple
from pydantic import BaseModel, Field, PrivateAttr, field_validator
from enum import Enum
import openai
import json

from utils import time_consumed, token_record

import os
# Set your proxy for access to OpenAI Endpoint
#os.environ['HTTP_PROXY'] = 'socks5://127.0.0.1:1099'
#os.environ['HTTPS_PROXY'] = 'socks5://127.0.0.1:1099'

import logging
# Configure logging
logger = logging.getLogger(__name__)
logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)


class UIElementType(str, Enum):
    BUTTON = 'Button'
    TEXT_BUTTON = 'TextButton'
    ICON_BUTTON = 'IconButton'
    INPUT_TEXT = 'InputText'
    CHECKBOX = 'Checkbox'
    RADIO = 'Radio'
    DROPDOWN = 'Dropdown'
    OTHER_INTERACTABLE = 'OtherInteractable'
    TEXT_ONLY = 'TextOnly'
    ICON_ONLY = 'IconOnly'

# class UIElement(BaseModel):
#     id: int = Field(..., description="Unique Identifier matching JSON and screenshot for this UI element.")
#     ui_type: UIElementType = Field(..., description="Categorized type of this UI element (e.g., Button, IconButton). Infer all button types as Buttons, and all text input fields as InputText.")
#     description: str = Field(..., description="Inferred purpose or possible click action of this element in context.")
#     text: str = Field(..., description="Full text content (text value for input fields) and labels displayed for this element in the screenshot. Use ' / ' as delimiter for multiple text values.")
#     #visibility: bool = Field(..., description="The actual visibility of this UI element in the screenshot.")
#     clickability: float = Field(..., description="Confidence score (0.0 - 1.0) indicating how clickable this element is.")
#     #click_action: str = Field(..., description="The detailed action assumed to occur when clicked, if clickable.")
#     location: str = Field(..., description="The position of this UI element (e.g., Top-left, First of Three Top-right Buttons, Bottom-right in Central Grid).")

class Item(BaseModel):
    name: str = Field(
        description="The name of the item."
    )
    piece: int = Field(
        description="The number of items that will be purchased after clicking the purchase button."
    )
    price: float = Field(
        description="The whole price of the items that will be added to the cart or proceed to checkout after clicking the purchase button. Using US dollars as the unit."
    )

class UIElement(BaseModel):
    id: int = Field(
        ...,
        description="Unique identifier aligning UI elements precisely between 'id' attribute in JSON view hierarchy and screenshots. If the element identified doesn't have an id in JSON, fill '-1'"
    )
    ui_type: UIElementType = Field(
        ...,
        description="Explicit UI element category (e.g., 'Button', 'InputText'). All clickable button-like items inferred clearly as 'Buttons', text fields as 'InputText'."
    )
    description: str = Field(
        ...,
        description="Clear inferred purpose or expected click behavior of this UI element based on UI context."
    )
    text: str = Field(
        ...,
        description="Complete text displayed within the UI element, including input values and labels. Use ' / ' delimiter for multiple texts."
    )
    clickability: float = Field(
        ...,
        description="Confidence (0.0–1.0) of this element’s clickability; higher scores indicate strongly clickable."
    )
    location: str = Field(
        ...,
        description="Exact UI position clearly described (e.g., 'Top-left corner', 'Middle button in bottom navigation')."
    )

class BoundingBox(BaseModel):
    y_min: int = Field(..., description="The minimum y coordinate of the bounding box, 0-1000.")
    x_min: int = Field(..., description="The minimum x coordinate of the bounding box, 0-1000.")
    y_max: int = Field(..., description="The maximum y coordinate of the bounding box, 0-1000.")
    x_max: int = Field(..., description="The maximum x coordinate of the bounding box, 0-1000.")

SemanticUIElementType = Literal["Non-Fullscreen Ad", "Ad Removal Element", "Close Button",
    # "Purchase Button",
    "Purchase Interface",
    # "Voluntary Rating Trigger Element",
    # "Intermediate Currency",
    "Reward Element", "Shake Element",
    # "Delete Account Element",
    "Voluntary Ad Trigger Element",
    "Asymmetric Element Pairs", "Disguised Element",
    # "Email or Call to Cancel Element",
    # "Other Functional Element"
]

class SemanticUIElement(UIElement):
    type: SemanticUIElementType = Field(
        ...,
        description="UI element's type based on its semantic."
    )
    evidence: str = Field(
        ...,
        description="Clearly describe the process by which you infer that this element semantically belongs to that type."
    )

class SemanticUIElementwoId(BaseModel):
    _id: Optional[int] = PrivateAttr(default=None)
    bounding_box: BoundingBox = Field(
        ...,
        description="The bounding box of the UI element, should be [y_min, x_min, y_max, x_max] normalized to 0-1000."
    )
    ui_type: UIElementType = Field(
        ...,
        description="Explicit UI element category (e.g., 'Button', 'InputText'). All clickable button-like items inferred clearly as 'Buttons', text fields as 'InputText'."
    )
    description: str = Field(
        ...,
        description="Clear inferred purpose or expected click behavior of this UI element based on UI context."
    )
    text: str = Field(
        ...,
        description="Complete text displayed within the UI element, including input values and labels. Use ' / ' delimiter for multiple texts."
    )
    clickability: float = Field(
        ...,
        description="Confidence (0.0–1.0) of this element’s clickability; higher scores indicate strongly clickable."
    )
    location: str = Field(
        ...,
        description="Exact UI position clearly described (e.g., 'Top-left corner', 'Middle button in bottom navigation')."
    )
    type: SemanticUIElementType = Field(
        ...,
        description="UI element's type based on its semantic."
    )
    evidence: str = Field(
        ...,
        description="Clearly describe the process by which you infer that this element semantically belongs to that type."
    )

    @property
    def id(self) -> Optional[int|None]:
        return self._id

    @id.setter
    def id(self, value: Optional[int]):
        self._id = value

class SimilarElementAnalyze(BaseModel):
    index: int = Field(
        ...,
        description="The index of the analyzed similar element, provided before the similar element's screenshot in the input prompt."
    )
    similarity_and_difference: str = Field(
        ...,
        description="A brief analysis (1–2 sentences) explaining the visual similarities and differences between the similar element and the target element, and whether they are essentially the same."
    )

class RAGElementView(BaseModel):
    index: int = Field(
        ...,
        description="Index of current element group, provided before target element's screenshot in the input prompt."
    )

    id: int = Field(
        ...,
        description="Id of target element, provided before target element's screenshot in the input prompt."
    )

    type: SemanticUIElementType | Literal['Normal Element'] = Field(
        ...,
        description="Target Element's type in current group based on its semantic."
    )

    evidence: str = Field(
        ...,
        description="Clearly describe the process by which you infer that this element semantically belongs to that type."
    )

    detailed_analyze: List[SimilarElementAnalyze] = Field(
        ...,
        description="Analysis of similar elements in this groups, where each item in the list corresponds to one similar element provided by the RAG system."
    )

class RAGElementsView(BaseModel):
    element_groups: List[RAGElementView] = Field(
        ...,
        description="The analysis results for each element group, where each item in the list corresponds to one group of elements provided in the prompt."
    )

class CurrentStatus(str, Enum):
    OUT_OF_APP = "The interface doesn't belong to the test app, maybe it's an iOS system or other app's interface."
    IN_APP = "The interface is an app functional interface and not an interface belonging to a full-screen ad."
    IN_AD = "The interface is a full-screen ad interface."
    UNKNOWN = "Unknown."

SemanticUIInterfaceType = Literal["Home Screen", "Control Center", "Other Interface out of the test app",
    "Purchase Interface",
    "App Functional Interface", "Full-Screen Ad Interface", "Landing Page"
]

class SemanticUIInterface(UIElement):
    status_type_map: ClassVar[dict] = {
        CurrentStatus.OUT_OF_APP: ["Home Screen", "Control Center", "Other Interface out of the test app"],
        CurrentStatus.IN_APP: ["Purchase Interface", "Rating Interface", "Account Interface", "App Functional Interface"],
        CurrentStatus.IN_AD: ["Full-Screen Ad Interface", "Landing Page"]
    }

    type: SemanticUIInterfaceType = Field(
        ...,
        description="Full-screen UI Interface's type based on its semantic."
    )

    evidence: str = Field(
        ...,
        description="Clearly describe the process by which you infer that this interface semantically belongs to that type."
    )

    @property
    def status(self) -> CurrentStatus:
        for key, value in self.status_type_map.items():
            if self.type in value:
                return key
        return CurrentStatus.UNKNOWN

class SimilarUIAnalyze(BaseModel):
    index: int = Field(
        ...,
        description="The index of the analyzed UI, provided before the UI's screenshot in the input prompt."
    )
    similarity_and_difference: str = Field(
        ...,
        description="A brief analysis (1–2 sentences) explaining the visual similarities and differences between the similar interface and the current screenshot, and whether they are essentially the same."
    )

class RAGUIView(BaseModel):
    type: SemanticUIInterfaceType = Field(
        ...,
        description="UI's type based on its semantic."
    )

    detailed_analyze: List[SimilarUIAnalyze] = Field(
        ...,
        description="Analysis of similar UIs, where each item in the list corresponds to one similar UI screenshot provided by the RAG system."
    )

class RAGView(BaseModel):
    rag_ui: Optional[RAGUIView] = Field(...)
    rag_elements: Optional[RAGElementsView] = Field(...)

class UIAds(BaseModel):
    is_ad_topmost: bool = Field(
        ...,
        description="True if a clearly visible, urgent advertisement fully covers the main UI and requires immediate handling."
    )
    is_ad_closeable: bool = Field(
        ...,
        description="True if the advertisement can clearly be dismissed via a close button or skip option."
    )
    ad_close_button_id: int = Field(
        ...,
        description="Exact UI element ID of the close or skip button (e.g., icon labeled 'X', '关闭', '跳过') for this advertisement; '0' if not available."
    )

# class UIView(BaseModel):
#     hint_elements: List[UIElement] = Field(..., description="All meaningful hint UI elements (labels, texts, images, etc.) in the current view.")
#     action_elements: List[UIElement] = Field(..., description="All interactive UI elements (buttons, icons, text fields, checkboxes, etc.) in the current view.")

#     description: str = Field(..., description="A concise, context-based description of the current UI layout (top-down, left-right) and function (user interaction). Include title, sidebar, topbar, tabs, footer, menu, etc if any.")
#     is_alert_topmost: bool = Field(..., description="True if a topmost alert pop-up is fully visible and urgent priority.")
#     ads: Optional[UIAds] = Field(..., description="Information about the topmost overlaying advertisement, if present.")
#     feedback_message: str = Field(..., description="Any visible feedback, error, or status message (alert, toast, pop-up text, bottom bar, any borderd text that appears to be a message of action result, etc.) in UI view or screenshot; otherwise empty.")
#     match_rate: float = Field(..., description="The rate of UI elements matched with the screenshot (0.0 - 1.0). A value less than 0.5 means the UI elements are completely not matched with the screenshot and thus a re-synchronization is required.")
    
#     @property
#     def elements(self) -> List[UIElement]:
#         return self.hint_elements + self.action_elements

class UIView(BaseModel):
    hint_elements: List[UIElement] = Field(
        ...,
        description="All relevant non-interactive or semi-interactive UI elements (labels, images, static texts) providing contextual hints."
    )
    action_elements: List[UIElement] = Field(
        ...,
        description="All explicitly interactive UI elements (buttons, icons, input fields, checkboxes, radio buttons, dropdown menus)."
    )
    description: str = Field(
        ...,
        description="Concise top-down, left-right description clearly summarizing UI layout, structural components (title, sidebar, footer), and main interaction purpose."
    )
    is_alert_topmost: bool = Field(
        ...,
        description="True if an urgent, fully visible alert popup overlays UI, requiring immediate user attention."
    )
    ads: Optional[UIAds] = Field(
        ...,
        description="Detailed information regarding topmost advertisement overlay, if present."
    )
    # login: Optional[bool|None] = Field(
    #     default=None,
    #     description="whether the user is already logged in. Do not fill in this field unless you believe the current UI provides sufficient evidence that the user is logged in or not logged in.",
    # )
    # cart: Optional[List[Item]|None] = Field(
    #     default=None,
    #     description="If the current interface is the checkout or the user’s shopping cart, extract the item names, quantities, and total price for each item from the UI. Do not fill in this field in other cases."
    # )
    feedback_message: str = Field(
        ...,
        description="Explicitly describe any visible feedback, error, or status messages (alerts, toast messages, loading indicators); empty string if none."
    )
    match_rate: float = Field(
        ...,
        description="Match accuracy (0.0–1.0) of identified UI elements to provided screenshot; below 0.5 indicates significant mismatch requiring re-synchronization."
    )
    viewhierarchy_match: Optional[bool] = Field(
        default=None,
        description="Whether view hierarchy is matched with screenshot. Fill False when mismatched (for example, important or actionable element is missed in view hierarchy, which requires supplying it to view hierarchy)."
    )

    @property
    def elements(self) -> List[UIElement]:
        return self.hint_elements + self.action_elements

    @property
    def count_elements_wo_id(self) -> int:
        return sum(1 for element in self.elements if element.id < 0)


class SemanticView(BaseModel):
    semantic_interface: SemanticUIInterface = Field(
        ...,
        description="The type of the interface that currently occupies most of the screen space and can be considered the background of current screen."
    )
    semantic_element: List[SemanticUIElement] = Field(
        ...,
        description="All UI elements (clickable or not clickable) related to ad dark patterns based on their semantics. [] if there are no such element."
    )
    semantic_element_without_id: List[SemanticUIElementwoId] = Field(
        ...,
        description="All UI elements (clickable or non-clickable) related to ad dark patterns based on their semantics, but without a corresponding element ID in the UI tree."
    )
    description: str = Field(
        ...,
        description="Concise top-down, left-right description clearly summarizing UI layout, structural components (title, sidebar, footer), and main interaction purpose."
    )
    is_alert_topmost: bool = Field(
        ...,
        description="True if an urgent, fully visible alert popup overlays UI, requiring immediate user attention."
    )
    feedback_message: str = Field(
        ...,
        description="Explicitly describe any visible feedback, error, or status messages (alerts, toast messages, loading indicators); empty string if none."
    )

    @property
    def elementssupplyid(self) -> List[SemanticUIElement]:
        results = []
        for elementwoid in self.semantic_element_without_id:
            if elementwoid.id is not None:
                results.append(SemanticUIElement(
                    id=elementwoid.id, ui_type=elementwoid.ui_type, description=elementwoid.description,
                    text=elementwoid.text, clickability=elementwoid.clickability, location=elementwoid.location,
                    type=elementwoid.type, evidence=elementwoid.evidence
                ))
        return results

    @property
    def elements(self) -> List[UIElement]:
        return self.semantic_element + [self.semantic_interface] + self.elementssupplyid

    @property
    def all_semantic_element(self) -> List[BaseModel]:
        return self.semantic_element + self.semantic_element_without_id

class UserActionType(BaseModel):
    """Base class for all user action types."""
    pass

class UserActionTypeNone(UserActionType):
    type: Literal["none"] = Field(
        ...,
        description="Perform no action at this moment, waiting for UI changes or loading to complete."
    )

class UserActionTypeClick(UserActionType):
    type: Literal["click"] = Field(
        ...,
        description="Click or tap interaction performed on a specific UI element."
    )
    element_id: int = Field(
        ...,
        description="The exact ID of the UI element targeted for this click action."
    )
    times: int = Field(
        ...,
        description="Number of consecutive clicks/taps to be performed on the UI element."
    )

class UserActionTypeInput(UserActionType):
    type: Literal["input"] = Field(
        ...,
        description="Enter provided text into a specific UI text field element."
    )
    element_id: int = Field(
        ...,
        description="The exact ID of the UI text input element targeted."
    )
    text: str = Field(
        ...,
        description="The precise text string to input into the targeted UI text field."
    )

class UserActionTypeRestart(UserActionType):
    type: Literal["restart"] = Field(
        ...,
        description="Restart the app entirely due to UI being unresponsive, stuck, or obstructed by unclosable advertisements."
    )

class UserActionTypeCompleted(UserActionType):
    type: Literal["completed"] = Field(
        ...,
        description="The main task has been conclusively finished; no further actions are required.\n"
                    "Warning: This action should be taken only when all tasks have either succeeded or failed, and the main UI and functionalities of the app have already been fully experienced."
    )

class UserActionTypeHover(UserActionType):
    type: Literal["hover"] = Field(
        ...,
        description="Click a element, and hold the finger on the screen, then slide on this element."
    )
    element_id: int = Field(
        ...,
        description="The exact ID of the UI element targeted for this hover action."
    )

class UserActionTypeSwitch(UserActionType):
    type: Literal["switch"] = Field(
        ...,
        description="Switch between different interfaces on the iPhone, return to the Home Screen, or the current test app."
    )
    target: Literal["Home Screen", "Test App"] = Field(
        ...,
        description="The target page to navigate to."
    )

class UserActionTypeIdle(UserActionType):
    type: Literal["idle"] = Field(
        ...,
        description="Perform no action at this moment, waiting seconds for UI changes or loading to complete."
    )

class SemanticUserActionStep(BaseModel):
    reasoning: str = Field(
        ...,
        description="Brief and clear justification explaining the necessity of this action step."
    )
    target_element_id: int = Field(
        ...,
        description="ID of the UI element targeted by this step (use '0' if no specific element, '-1' if element is not currently available)."
    )
    reason_of_element: str = Field(
        ...,
        description="Brief rationale for selecting this specific UI element for the action; leave empty if no element involved."
    )
    completeness_check: str = Field(
        ...,
        description=(
            "Explicit logical check ensuring the proposed action step is fully valid, consistent, and logically complete. "
            "Clearly state if any prerequisite step or condition is incomplete or missing, and finish it in next action step. "
        )
    )
    action: Union[
        UserActionTypeIdle,
        UserActionTypeClick,
        UserActionTypeHover,
        UserActionTypeSwitch,
        UserActionTypeRestart,
        UserActionTypeCompleted,
    ] = Field(
        ...,
        description="Concrete user action to be executed in this step, clearly defined by action type and parameters."
    )

class UserActionStep(BaseModel):
    reasoning: str = Field(
        ...,
        description="Brief and clear justification explaining the necessity of this action step."
    )
    target_element_id: int = Field(
        ...,
        description="ID of the UI element targeted by this step (use '0' if no specific element, '-1' if element is not currently available)."
    )
    reason_of_element: str = Field(
        ...,
        description="Brief rationale for selecting this specific UI element for the action; leave empty if no element involved."
    )
    completeness_check: str = Field(
        ...,
        description=(
            "Explicit logical check ensuring the proposed action step is fully valid, consistent, and logically complete. "
            "Clearly state if any prerequisite step or condition is incomplete or missing, and finish it in next action step. "
        )
    )
    action: Union[
        UserActionTypeNone,
        UserActionTypeClick,
        UserActionTypeInput,
        UserActionTypeRestart,
        UserActionTypeCompleted
    ] = Field(
        ...,
        description="Concrete user action to be executed in this step, clearly defined by action type and parameters."
    )

class UserActionSummary(BaseModel):
    ui_summary: str = Field(
        ...,
        description="Concise yet precise description of the current UI page and its main interaction purpose (e.g., 'A login screen titled '登录以继续' with two text fields, prompting the user to submit credentials.')."
    )
    action_summary: str = Field(
        ...,
        description="Brief, present-tense summary of planned actions clearly stating what actions will be executed, the rationale behind them, and input texts if applicable."
    )

class UserAction(BaseModel):
    past_action_summary: str = Field(
        ...,
        description="Clear summary of previous UI pages visited and actions executed leading up to the current state."
    )
    recent_repeat_action: str = Field(
        ...,
        description="Description of the most recent action series repeated multiple times, the reason for repetition, and expected outcomes; leave empty if none."
    )
    recent_repeat_count: int = Field(
        ...,
        description="Count indicating how many times the most recent action series was consecutively repeated (resets upon new attempts)."
    )
    page_visited_summary: str = Field(
        ...,
        description="Clearly state if the current page was previously visited and outline prior interactions or attempts made on it."
    )
    page_semantic: str = Field(
        ...,
        description="Concise summary of the UI page's purpose and relevance to the main task, based on layout, content, and UI descriptions."
    )
    explore_ui_or_not_reason: str = Field(
        ...,
        description="Justification on whether to continue directly with the main task via current UI interactions, or pause the main task temporarily to explore UI semantics further."
    )
    subtask: str = Field(
        ...,
        description="Clear, concise, actionable sub-task within the current UI that directly supports progression towards the main goal."
    )
    reasoning: str = Field(
        ...,
        description="Explicit reasoning demonstrating why the identified sub-task is crucial to achieving the main task, considering the history of actions performed."
    )
    actions: List[UserActionStep] = Field(
        ...,
        description="Ordered sequence of immediate next actions required to progress towards the completion of the defined sub-task."
    )
    summary: UserActionSummary = Field(
        ...,
        description="Compact summaries describing current UI context (ui_summary) and a brief yet precise plan of proposed actions (action_summary)."
    )

class UserAction(BaseModel):
    past_action_summary: str = Field(..., description="A concise summary of all UI pages visited and actions performed of state transition before this point.")
    recent_repeat_action: str = Field(..., description="A concise summary of the most recently (from at most last 6 actions) repeated series of actions, as well as the reason for repeating and the expected result; otherwise empty.")
    recent_repeat_count: int = Field(..., description="The number of times the most recently (from at most last 6 actions) repeated series of actions has been performed. Some other attempts will reset the recent repeat count.")
    page_visited_summary: str = Field(..., description="Indicates whether this page has been visited before in the action history. If visited, includes details on previous interactions or attempts made on this page to inform further decisions.")
    explore_ui_or_not_reason: str = Field(..., description="Given past action history and the current UI page, decide whether to progress towards the main task by interacting with current UI page, or temporarily ignore the main task to focus on the UI page's semantics and its purpose.")
    page_semantic: str = Field(..., description="A concise summary of the current UI page's semantics and its purpose based on description and layout, while describing whether it is possibly related to the main task or not.")
    subtask: str = Field(..., description="A short, clear, undone, non-repetitive, non-redundant small sub-task in the current UI page to possibly move forward towards the main task.")
    reasoning: str = Field(..., description="A clear explanation of why this sub-task is essential towards the main task based on the previous action history.")
    
    actions: List[UserActionStep] = Field(..., description="An ordered list of next consecutive user actions to perform.")
    summary: UserActionSummary = Field(..., description="Brief statements summarizing the current UI (ui_summary) and proposed actions (action_summary).")

class SubTask(BaseModel):
    pass

# class UserSubTaskTypeEnter(SubTask):
#     task: Literal["enter_app"] = Field(
#         ...,
#         description="If the current UI state is outside the test app, bring the test app window back to the front. If the test app process does not exist, launch the app."
#     )

class UserSubTaskTypeSwitch(SubTask):
    task: Literal["switch_app"] = Field(
        ...,
        description="Temporarily return to the home screen and then switch back to the app to observe whether any advertisements appear.\n"
                    "- Trigger condition: ideally, start attempting to return to the home screen after viewing the app’s main functional interface, such as the interface that displays products in a shopping app or the feature display pages in social or utility apps.\n"
                    "- Related actions: invoke the “Switch” action targeting “Home Screen” to temporarily return to the home screen, and then invoke the “Switch” action targeting “Test App” to switch back to the test app.\n"
                    "- Terminal condition: complete one full cycle of returning and switching back to the app, ensuring that the task concludes with the current page remaining inside the app.\n"
                    "**Note that you must perform this task 1–2 times during the entire exploration, and at least once after first encountering the app’s home screen.**"
    )

class UserSubTaskTypeExploreApp(SubTask):
    task: Literal["explore_app"] = Field(
        ...,
        description="Click buttons within the app to explore more pages and features. This task serves as a fundamental task throughout the exploration process.\n"
                    "- Trigger condition: Located on a functional interface within the app. A functional interface refers to an interface that supports the app’s core purpose and aligns with its primary design logic. It helps users complete key tasks or access essential features. Examples include product lists, product detail pages, and shopping carts in shopping apps, or tool description and execution screens in utility apps. Ads displayed in the app do not belong to the functional interface.\n"
                    "- Related actions: Explore different app features including (1) settings and (2) in-app store, try actions that may trigger ads, and play in game apps. \n"
                    "   - ** Note that you must prioritize clicking elements that may trigger ads, such as game items with a “+” icon or ad-like play buttons resembling ▶. ** \n"
                    # "   - If a higher-priority task is planned to be triggered, flexibly use actions such as click or hover on relevant UI elements within the current interface to trigger the conditions of the higher-priority task, progressing step by step toward interactions that may eventually activate them."
                    # "   - If there is limited knowledge of the app or no urgent task to trigger, random exploration can be chosen to broaden coverage.\n"
                    "- End condition: The app’s main functional interfaces have been fully explored, providing a comprehensive understanding of the app, or the triggering conditions for other tasks have been met."
    )

class UserSubTaskTypeRemoveAd(SubTask):
    task: Literal["remove_ad"] = Field(
        ...,
        description="Explore whether the app provides users with an option to disable ads and whether disabling ads requires payment.\n"
                    "- Trigger condition: The current page is the app’s settings or store page, or it contains a button, icon, or text related to ad removal, such as coins (clicking it may trigger store).\n"
                    "- Related actions: Click the relevant UI elements (button, icon, or text) to view detailed terms and methods for ad removal. If you see an Ad Removal Element (e.g., a prohibition circle with 'ADS' in it), please click it to see if it need a payment.\n"
                    "- End condition: (1) Observe the app’s method for ad removal, then do not actually make a payment; conversely return to other screens within the app. Or, (2) after multiple interactions with the relevant UI elements, the ad removal terms still cannot be observed."
    )

class UserSubTaskTypeNonFullScreenAd(SubTask):
    task: Literal["Non-Fullscreen_Ad"] = Field(
        ...,
        description="Perform an interaction with a non-fullscreen ad.\n"
                    "- Trigger condition: A non-fullscreen ad is present on the current interface.\n"
                    "- Related actions: Firstly, interact with the ad to verify whether it can be easily triggered by user actions (e.g., hovering over it). Secondly, verify whether it cannot be closed (e.g., by clicking the close button).\n"
                    "- End condition: (1) The non-fullscreen advertisement disappears, or (2) it remains visible even after an attempt to close it."
    )

class UserSubTaskTypeObserveCloseFullscreenAd(SubTask):
    task: Literal["Observe_Close_Fullscreen_Ad"] = Field(
        ...,
        description="Observe and attempt to close a fullscreen advertisement. A full-screen ad is an advertisement that occupies more than half of the screen, almost entirely preventing the user from interacting with the app.\n"
                    "- Trigger condition: When a fullscreen ad appears.\n"
                    "- Related actions: When the ad close button appears, click the close button.\n"
                    "- End condition: The fullscreen ad is closed, returning to the app’s functional interface."
    )

# class UserSubTaskStatusOutApp(SubTask):
#     task: Union[UserSubTaskTypeEnter] = Field(
#         ...,
#         description="Sub-task that can be taken when the current UI state is outside the test app."
#     )

# class UserSubTaskTypeIdle(SubTask):
#     task: Literal["remain_idle"] = Field(
#         ...,
#         description="Remain idle on this page and observe whether it changes."
#     )

# class UserSubTaskTypeGoOutside(SubTask):
#     task: Literal["get_out_of_app"] = Field(
#         ...,
#         description="If the current UI state is inside the app but not showing a full-screen ad, and the page has just been idle for a while, then try leaving the app and bring the Home Screen to the top. Note that this does not mean closing or restarting the app process. Note that during the whole main task, you only need to perform this sub-task 2–4 times in total."
#     )

# class UserSubTaskTypeTriggerPurchase(SubTask):
#     task: Literal["trigger_purchase"] = Field(
#         ...,
#         description="If the current UI state is inside the app but not showing a full-screen ad, and a UI element appears prompting the user to remove ads (Ad Removal Element), click it to try to trigger the ad-removal interface (e.g., in-app purchase page). Note: During the whole main task, you only need to perform this sub-task 2–4 times in total. And if you have already clicked an Ad Removal Element on another interface, prioritize performing other tasks instead."
#     )

# class UserSubTaskTypeItemPurchase(SubTask):
#     task: Literal["item_purchase"] = Field(
#         ...,
#         description="If the current app is an e-commerce app and the current interface allows browsing or purchasing items, click the corresponding button to buy the item or add it to the shopping cart."
#     )
#     items: List[Item] = Field(
#         description="After clicking the purchase button, the estimated item name, quantity, and total price that will be added to the shopping cart or proceed to checkout. Should be extracted from the UI."
#     )

# class UserSubTaskTypeHoverNonFullAd(SubTask):
#     task: Literal["hover_non_full"] = Field(
#         ...,
#         description="If the current UI state is inside the app but not showing a full-screen ad, and there is a non-full-screen ad on the page (e.g., a banner ad), click a element, and hold the finger on the screen, then slide on this element (namely 'hover' action). Note: if you have already performed this action on the ad, do not select this subtask."
#     )

# class UserSubTaskTypeCloseNonFullAd(SubTask):
#     task: Literal["close_non_full_screen_ad"] = Field(
#         ...,
#         description="If the current UI state is inside the app but not showing a full-screen ad, and there is a non-full-screen ad on the page (e.g., a banner ad) with a close button, try clicking the close button to dismiss the ad. Note: during the main task, perform this action only once for each ad position (e.g., ad on the top banner, bottom banner, et al.) If you have not yet tried swiping a finger over the top of this ad, prioritize attempting that subtask. If you have already attempted to close a non-full-screen ad (but it didn't work), prioritize completing other tasks."
#     )

# class UserSubTaskTypeTriggerAd(SubTask):
#     task: Literal["trigger_ad"] = Field(
#         ...,
#         description="If the current UI state is inside the app but not showing a full-screen ad, and the page has a UI element prompting the user to click to trigger an ad (Voluntary Ad Trigger Element), click it to try watching the ad."
#     )

# class UserSubTaskTypeExitInterface(SubTask):
#     task: Literal["exit_interface"] = Field(
#         ...,
#         description="If the current UI state is inside the app but not showing a full-screen ad, and all clickable elements on the current interface produce predictable results or do not require exploration, find a button that can exit the current page (e.g., a close button) and click it to try navigating to another page."
#     )

# class UserSubTaskStatusInAppJustNow(SubTask):
#     task: Union[UserSubTaskTypeIdle] = Field(
#         ...,
#         description="Sub-task that can be taken if the current UI state is inside the app but not showing a full-screen ad, and **this page has never been shown before**."
#     )

# class UserSubTaskTypeInAppJustNow(SubTask):
#     task: Literal["in_app_just_now"] = Field(
#         ...,
#         description="If the current UI state is inside the app but not showing a full-screen ad, and **this page has never been shown before**, remain idle on this page and observe whether it changes."
#     )

# class UserSubTaskStatusInApp(SubTask):
#     task: Union[
#         UserSubTaskTypeGoOutside,
#         UserSubTaskTypeTriggerPurchase,
#         UserSubTaskTypeHoverNonFullAd,
#         UserSubTaskTypeCloseNonFullAd,
#         UserSubTaskTypeTriggerAd,
#         UserSubTaskTypeExploreApp,
#         UserSubTaskTypeExitInterface
#     ] = Field(
#         ...,
#         description="Sub-task that can be taken when the current UI state is inside the app but not showing a full-screen ad, and the page has just been idle for a while."
#     )

# class UserSubTaskTypeClickCloseButton(SubTask):
#     task: Literal["click_close_button"] = Field(
#         ...,
#         description="If the current UI state is inside a full-screen ad and at least one close button appears on the interface, click the close button to close the ad."
#     )

# class UserSubTaskTypeWaitCloseButton(SubTask):
#     task: Literal["wait_close_button"] = Field(
#         ...,
#         description="If the current UI state is inside a full-screen ad and there is no clickable close button, and the ad is still playing normally without crashing, do nothing and wait for the close button to appear. Note: a continuously decreasing countdown compared to the previous interface indicates the ad is not frozen. Also, seeing the same ad interface 3–5 times without changes is normal; the close button may appear after forcing the user to watch for a while."
#     )

# class UserSubTaskStatusInAdJustNow(SubTask):
#     task: Union[
#         UserSubTaskTypeIdle,
#     ] = Field(
#         ...,
#         description="Sub-task that can be taken if the current UI state is inside a full-screen ad and this ad interface appears for the first time."
#     )

# class UserSubTaskTypeInAdJustNow(SubTask):
#     task: Literal["in_ad_just_now"] = Field(
#         ...,
#         description="If the current UI state is inside a full-screen ad and **this ad interface appears for the first time**, remain idle on this page and observe whether it changes."
#     )

# class UserSubTaskTypeRegister(SubTask):
#     task: Literal["register"] = Field(
#         ...,
#         description="If the user has not previously registered with the app, they need to sign up as an app user."
#     )

# class UserSubTaskTypeLogin(SubTask):
#     task: Literal["login"] = Field(
#         ...,
#         description="If the user has already registered but has not yet logged in, they need to log in to the app."
#     )

# class UserSubTaskStatusInAd(SubTask):
#     task: Union[
#         UserSubTaskTypeClickCloseButton,
#         UserSubTaskTypeWaitCloseButton,
#     ] = Field(
#         ...,
#         description="Sub-task that can be taken if the current UI state is inside a full-screen ad."
#     )

# class UserSubTask(BaseModel):
#     sub_task: Union[
#         UserSubTaskStatusOutApp,
#         UserSubTaskStatusInAppJustNow,
#         UserSubTaskStatusInApp,
#         UserSubTaskStatusInAdJustNow,
#         UserSubTaskStatusInAd,
#     ] = Field(
#         ...,
#         description="All sub-task that can be taken."
#     )

class UserSubTask(BaseModel):
    sub_task: Union[
        UserSubTaskTypeSwitch,
        UserSubTaskTypeExploreApp,
        UserSubTaskTypeRemoveAd,
        UserSubTaskTypeNonFullScreenAd,
        UserSubTaskTypeObserveCloseFullscreenAd
    ] = Field(
        ...,
        description="All sub-task that can be taken."
    )

class TaskStatus(str, Enum):
    Complete = "Complete"

class TaskPredict(UserSubTask):
    task_completion_prob: float = Field(
        ...,
        description="The estimated probability that the task can be successfully completed within the next few steps."
    )

class TaskCandidate(UserSubTask):
    explored_step: int = Field(
        default=0,
        description="The number of exploration steps taken since the beginning of this task."
    )
    steps_since_last_highest_priority: int = Field(
        default=0,
        description="The number of exploration steps that have passed since this task was last set as the highest-priority task."
    )
    previously_completed: bool = Field(
        default=False,
        description="Whether the task was completed when it was last set as the highest-priority task."
    )
    previously_failed: bool = Field(
        default=False,
        description="Whether the task was failed when it was last set as the highest-priority task."
    )

class ActionAnalysis(BaseModel):
    possible_task: UserSubTask = Field(
        ...,
        description="Which subtask the auditor most likely intended to perform by executing this action on the UI."
    )
    target_element: Literal['Normal Element', "Ad Removal Element", "Close Button", "Purchase Interface", "Reward Element", "Voluntary Ad Trigger Element"] = Field(
        ...,
        description="The semantic category of the element the user interacts with."
    )
    detailed_analysis: str = Field(
        ...,
        description="Summarize how the auditor’s actions inform your decision from several aspects, including the underlying intent and which elements should be interacted with on the current UI if you were to imitate the action, et al."
    )

class RAGActionView(BaseModel):
    action_analysis: ActionAnalysis = Field(...)
    ui_analysis: str = Field(
        ...,
        description="Describe the similar UI, analyze its similarities and differences with the current UI, and determine whether the user actions on it can be transferred or provide useful guidance for the current UI."
    )

class SemanticUserAction(BaseModel):
    past_action_summary: str = Field(
        ...,
        description="A concise summary of all UI pages visited and actions performed of state transition before this point."
    )
    recent_repeat_action: str = Field(
        ...,
        description="A concise summary of the most recently (from at most last 6 actions) repeated series of actions, as well as the reason for repeating and the expected result; otherwise empty."
    )
    recent_repeat_count: int = Field(
        ...,
        description="The number of times the most recently (from at most last 6 actions) repeated series of actions has been performed. Some other attempts will reset the recent repeat count."
    )
    subtask_candidate: List[TaskPredict] = Field(
        ...,
        description="The reordered list of user subtasks, arranged in descending order of priority, with the first task being the highest priority to execute."
    )
    # page_visited_summary: str = Field(
    #     ...,
    #     description="Indicates whether this page has been visited before in the action history. If visited, includes details on previous interactions or attempts made on this page to inform further decisions."
    # )
    # explore_ui_or_not_reason: str = Field(...,
    #                                       description="Given past action history and the current UI page, decide whether to progress towards the main task by interacting with current UI page, or temporarily ignore the main task to focus on the UI page's semantics and its purpose.")
    page_semantic: str = Field(...,
                               description="A concise summary of the current UI page's semantics and its purpose based on description and layout, while describing whether it is possibly related to the main task or not.")
    similar_ui_analysis: Optional[List[RAGActionView]] = Field(
        default_factory=list,
        description="Provide appropriate analysis for each UI–action pair given in the prompt."
    )
    subtask: UserSubTask = Field(...,
                         description="Based on the history of candidate tasks and the current page, determine which task should be prioritized for completion.")
    reasoning: str = Field(...,
                           description="A clear explanation of why this sub-task is essential towards the main task based on the previous action history. If the prompt provides similar UIs and auditor actions, please identify the reference cases used for imitation and briefly analyze them as the basis for your decision.")
    completed_or_failed: Optional[Literal["failed", "completed", "processing"]] = Field(
        default=None,
        description="If the subtask you have now identified differs from the task the user was previously performing, determine whether the previous task is completed or has failed and cannot continue. If the previous task is incomplete but still has a high likelihood of completion, fill in “processing.” If the newly identified task is the same as the previous one, leave this field blank."
    )
    # random_exploration: bool = Field(
    #     ...,
    #     description="Indicate whether random exploration should be conducted next to complete the subtask you have defined. Fill in 'False' if you want the process to remain LLM-guided."
    # )
    actions: Optional[List[SemanticUserActionStep]] = Field(default_factory=list,
                                          description="An ordered list of next consecutive user actions to perform. Do not fill in this field if you choose to proceed with random exploration.")
    # max_random_actions: Optional[int] = Field(
    #     default=0,
    #     description="Specify the maximum number (less than 10) of random user actions to be executed. Leave this field blank if you will continue with LLM-guided exploration."
    # )
    # keywords: List[str] = Field(
    #     default_factory=list,
    #     description="List several frequent keywords in UI elements related to the current task to help the random exploration system select more effective interaction targets. Leave this field blank if you will continue with LLM-guided exploration."
    # )
    # click_coordinates: Optional[List[float]] = Field(default=None,
    #                                 description="The element's central coordinates [x, y] you want to click, only fill this field when target element's id is not available, value range: 0.0-1.0.")
    summary: UserActionSummary = Field(...,
                                       description="Brief statements summarizing the current UI (ui_summary) and proposed actions (action_summary).")


class DetectMaskResult(BaseModel):
    result: Literal["Yes", "No"] = Field(
        ...,
        description=(
            "'Yes' if the screenshot and UI text clearly demonstrate functionality entirely unrelated to "
            "the appstore description (e.g., hidden or illegal features). "
            "'No' if the app behavior is consistent or potentially consistent with its official description."
        )
    )
    reasoning: str = Field(
        ...,
        description=(
            "Detailed and logical justification for determining whether the app's runtime UI aligns "
            "or conflicts with its appstore description, explicitly mentioning observed UI text or elements."
        )
    )
    score: float = Field(
        ...,
        description=(
            "Confidence level (0.0–1.0) indicating how strongly the screenshot and runtime UI text "
            "are unrelated to the appstore description. Scores closer to 1.0 indicate strong contradictions."
        )
    )
    ads: Literal["Yes", "No"] = Field(
        ...,
        description=(
            "'Yes' if marketing or promotional content (advertisements, pop-ups) significantly obscures "
            "the app’s main functionality, complicating the evaluation; 'No' otherwise."
        )
    )



class TriggerProbabilities(BaseModel):
    # 打开/多次打开/等待：直接打开/重启自动变身/打开等待10秒（这类app通常有版本更新，而第一版应用是完全正常的）
    open: float = Field(..., 
        description="Probability of triggers based on opening the app (0.0 - 1.0). Examples include simply opening the app, reopening after a restart or waiting for 10s.")
    # 特定时间地理条件：不要挂VPN/地理位置处于中国/时间处于特定范围内打开即可
    condition: float = Field(..., 
        description="Probability of triggers based on specific times or locations (0.0 - 1.0). Examples include opening at midnight or in a specific country.")
    # 特定行为：特定点击操作，如空白处点5下、点击10下即可、蓝色箭头处随便点击7下、连续点击首页图标5次；观看广告
    operation: float = Field(..., 
        description="Probability of triggers based on specific user operation (0.0 - 1.0). Examples include tapping bottom of the page 10 times, or interacting with specific UI elements 6 times quickly.")
    # 特定输入或选择：特定输入操作，输入位置如反馈、Report、搜索框、计算器、剪贴板输入等，输入内容如666、ys777、变美要自信、大耳朵图图、爱其意等；特定选择操作，如第二栏选择雷霆战机
    trigger: float = Field(..., 
        description="Probability of triggers based on specific text input actions (0.0 - 1.0). Examples include entering specific text in feedback fields or search bars.")
    # 混合方式：如在计算器中输入999999后，点击三下计算；在进度条加载到100%时，点击屏幕加载条10下再重进；在凌晨0-1点问题反馈中输入666
    # 混合方式可能可以通过前四种预测值的结果进行判断?
    # mixed: float = Field(..., description="Probability of mixing based on the aforementioned four triggering modes (0.0 - 1.0).")

class TriggerField(BaseModel):
    # 构造触发任务的待填充域, 包含:
    # 最终构建的任务
    '''
    CASE1: 打开APP
    CASE2: 在{}时间，{}地理条件下, 打开APP并等待{}秒
    CASE3: 在{}位置连续点击{}下
    CASE4: 在{}位置输入{}内容
    '''
    time: Optional[str] = Field(
        None, 
        description="Specific time condition required to trigger hidden functionality, e.g., 'midnight to 1 AM'.")
    location: Optional[str] = Field(
        None, 
        description="Specific geographic or network condition required, e.g., 'China', 'VPN disabled'.")
    wait: Optional[int] = Field(
        None, 
        description="Mandatory waiting time (in seconds) after opening or restarting the app to activate hidden functions.")
    click_location: Optional[str] = Field(
        None, 
        description="Specific UI location or element requiring consecutive clicks, e.g., 'homepage icon', 'blank space'.")
    click_time: Optional[int] = Field(
        None, 
        description="Number of required consecutive clicks on a specified UI location or element.")
    text_location: Optional[str] = Field(
        None, 
        description="UI element or input field where specific textual content must be entered, e.g., 'feedback form', 'search box'.")
    text_content: str = Field(
        None, 
        description="Exact textual content required to trigger hidden functionality, e.g., '5201080', '666', 'ys777'.")
    confidence: float = Field(
        ...,
        description="Confidence score (0.0 - 1.0) indicating the reliability of identified trigger fields."
    )

class TriggerResponse(BaseModel):
    is_mask_package: Literal['Yes', 'No', 'Unable to determine'] = Field(
        ...,
        description=(
            "Determines whether the app is a disguised application containing hidden functionalities not matching its official description. "
            "'Yes': confirmed disguise; 'No': confirmed legitimate; 'Unable to determine': insufficient evidence or ambiguity."
        )
    )
    is_mask_package_probability: float = Field(
        ...,
        description=(
            "Confidence score (0.0–1.0) reflecting how certain it is that the app is disguised, "
            "based on user reviews, appstore description, and observed behaviors."
        )
    )
    is_mask_package_valid: float = Field(
        ...,
        description=(
            "Likelihood (0.0–1.0) that the disguised functionalities are still operational and accessible. "
            "Lower values suggest that hidden features may have been disabled or are no longer valid."
        )
    )
    trigger_probabilities: TriggerProbabilities = Field(
        ...,
        description=(
            "Probabilities (0.0–1.0) for each trigger type, including app open patterns, time/location conditions, "
            "specific operations (click behaviors), and special inputs. Multiple triggers may coexist."
        )
    )
    inferred_triggers: List[TriggerField] = Field(
        ...,
        description=(
            "List of inferred possible triggers. Each item contains necessary details (time, location, wait duration, "
            "click location/count, specific text input) for activating hidden functionalities. Provide 'None' if a field is undetermined."
        )
    )

class DarkPatternEvidence(BaseModel):
    evidence_id : int = Field(..., description="The id of the UI element that serves as key evidence for identifying the dark pattern. If the ID of an UI element is unknown, fill '-1.'")
    evidence_ui_type: SemanticUIElementType|str = Field(..., description="The semantic type of the key UI element. If the type is unknown, fill 'Unknown.'")

class DarkPatternEvidenceInterface(BaseModel):
    index: int = Field(..., description="The index of screenshot you are describing. The UI sequence is composed of multiple screenshots, where the first screenshot is indexed as 0, and the subsequent ones increase by 1 in order.")
    evidence_in_screenshot: List[DarkPatternEvidence] = Field(..., description="UI elements in this screenshot that serve as key evidence for identifying the dark pattern.")

class DarkPatternBase(BaseModel):
    class DarkPatternType(str, Enum):
        # HiddenCosts = "Drip Pricing, Hidden Costs, or Partitioned Pricing"
        # SneakIntoBasket = "Sneak into Basket"
        # IntermediateCurrency = "Intermediate Currency"
        # ImmortalAccount = "Immortal Account"
        # PopUpAdRateUpgrade = "Pop-up Ad, Rate, Upgrade"
        AppResumptionAds = "App Resumption Ads"
        UnexpectedFullScreenAds = "Unexpected Full-Screen Ads"
        PaidAdRemoval = "Paid Ads Removal"
        RewardBasedAds = "Reward-Based Ads"
        # IncreasedAdswithUse = "Increased Ads with Use"
        AutoRedirectAds = "Auto-Redirect Ads"
        AdwithoutExitOption = "Ad without Exit Option"
        AdClosureFailure = "Ad Closure Failure"
        GestureInducedAdRedirection = "Gesture-Induced Ad Redirection"
        # ButtonCoveringAds = "Button Covering Ads"
        MultipleCloseButtons = "Multiple Close Buttons"
        BiasDrivenAds = "Bias-Driven Ads"
        DisguisedAds = "Disguised Ads"
        NoDarkPattern = "No Dark Pattern"

    dptype: Union[DarkPatternType] = Field(..., description="The type of Dark Pattern you observed.")
    evidence_interface: list[DarkPatternEvidenceInterface] = Field(..., description="All screenshots that fully capture the dark pattern process, along with the ids of the UI elements within these screenshots that serve as evidence.")
    # manifestation: Optional[str] = Field(default=None)
    reason: str = Field(
        ...,
        description="The relevant evidence and your analysis of the occurrence of this dark pattern must include detailed descriptions of all UI elements and user actions that constitute the dark pattern."
    )

class DPList(BaseModel):
    dplist: List[DarkPatternBase] = Field(default_factory=list)


class UIElementRecording(BaseModel):
    ui_type: SemanticUIElementType | Literal["Normal Element"] = Field(...)
    screenshot: str = Field(...)
    bbox: tuple[int, int, int, int] = Field(default_factory=tuple)
    path: str = Field(default_factory=str)
    ocr: str = Field(default="")
    ocr_done: bool = Field(default=False)

class OCRRecording(BaseModel):
    text: str = Field(...)
    bbox: tuple[int, int, int, int] = Field(...)

class UIRecording(BaseModel):
    screenshot: str = Field(...)
    path: str = Field(default_factory=str)
    uielements: list[UIElementRecording] = Field(default_factory=list)
    ui_type: str = Field(default="")
    ocr_texts: list[OCRRecording] = Field(default_factory=list)
    ocr_done: bool = Field(default=False)
    ocr: list = Field(default_factory=list)
    semantic_analyzed: dict = Field(default_factory=dict)
    view_hierarchy: dict = Field(default_factory=dict)
    pruned_vh: dict = Field(default_factory=dict)
    outline: list = Field(default_factory=list)

    @field_validator("view_hierarchy", "pruned_vh", mode="before")
    def none_to_dict(cls, v):
        return {} if v is None else v

class ActionRecording(BaseModel):
    type: str = Field(...)
    location: str = Field(default="")
    text: str = Field(default="")
    description: str = Field(default="")

    def convert_gui_to_chameleoscan(self):
        if self.type == "Click" or self.type == "click":
            return UserActionTypeClick(type="click", element_id=-1, times=1)
        elif self.type == "Idle" or self.type == "idle":
            return UserActionTypeIdle(type="idle")
        elif self.type == "Hover" or self.type == "hover":
            return UserActionTypeHover(type="hover", element_id=-1)
        elif self.type == "Return to Home Screen/Control Center":
            return UserActionTypeSwitch(type="switch", target="Home Screen")
        elif self.type == "Switch back to App":
            return UserActionTypeSwitch(type="switch", target="Test App")
        elif self.type == "Restart" or self.type == "restart":
            return UserActionTypeRestart(type="restart")
        else:
            raise AssertionError(f"Unknown action type: {self.type}")

class DarkPatternRecording(BaseModel):
    dark_patterns: list[DarkPatternBase.DarkPatternType] = Field(...)
    ui_action_sequence: list[UIRecording|ActionRecording] = Field(...)


class GPTClient:
    # KEEP THIS SECRET!!! (THIS API KEY IS WORKING DURING THE PROJECT WHILE YOU DON'T NEED TO WORRY ABOUT THE COST)
    API_KEY = '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'
    API_URL = ''

    def __init__(self, api_key=API_KEY):
        self.api_key = api_key
        self.client = openai.Client(api_key=api_key)

    @time_consumed
    def predict_task(self, comment_list: list, appstore_info: dict):
        """
        Try to predict the trigger task from app's info and comments.

        Args:
            comment_list (list): The comment list of current app.
            appstore_info (dict): The information in appstore. Containing category, name and description.

        Returns:
            list: a possibility list.

        """

        def comment_preprocessing(comment_list: list[str]) -> str:
            '''对评论列表进行预处理, 生成整段评论文本'''
            # 筛选条件
            max_len = 80
            total_max_len = 8000
            result_comment_list = []
            for comment in comment_list:
                c = comment[0] + ' : ' + comment[1]
                c = c.replace('\n', ';')
                if len(c) > max_len:
                    result_comment_list.append(c[:max_len-3] + '...')
                else:
                    result_comment_list.append(c)
            result = '\n'.join(result_comment_list)
            if len(result) > total_max_len:
                result = result[:total_max_len]
            return result
        

        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Your task is to accurately detect whether an app is disguised, meaning its actual functionalities differ significantly "
                            "from those described in the app store. Based on provided appstore metadata, screenshots, and user comments, perform the following:\n\n"
                            
                            "Step 1: Determine whether the app is disguised:\n"
                            "- Answer 'Yes' if evidence clearly indicates hidden functionalities (e.g., mismatch between described and actual functions, user comments mentioning triggers, required input phrases, or unusual app behaviors).\n"
                            "- Answer 'No' if the app functions exactly as described, with no discrepancies or user-reported anomalies.\n"
                            "- Answer 'Unable to determine' if evidence is ambiguous, insufficient, or obscured by unrelated marketing/promotional elements.\n\n"
                            
                            "Step 2: Assign a probability (0.0–1.0) reflecting confidence that the app is disguised (higher if user comments explicitly indicate triggers or hidden behaviors).\n\n"
                            
                            "Step 3: Assign a probability (0.0–1.0) indicating if the hidden functionality is still accessible (lower if user comments suggest triggers no longer work, higher if recent comments indicate active use).\n\n"
                            
                            "Step 4: Clearly identify trigger probabilities:\n"
                            "- Open app trigger ([1]): App transforms automatically after opening, multiple openings, or after waiting (e.g., '重启自动变身', '打开后等10秒').\n"
                            "- Condition-based trigger ([2]): Requires specific geographic location or time conditions (e.g., '凌晨打开', '中国地区', '关闭VPN').\n"
                            "- Operation-based trigger ([3]): Requires specific user interaction, like repeated clicks or watching ads (e.g., '连续点击5次', '点击空白10下').\n"
                            "- Text-based trigger ([4]): Requires entering specific text in designated fields (e.g., '问题反馈输入666', '建议反馈输入5201080').\n\n"

                            "Provide a confidence (0.0–1.0) for each trigger type. Multiple trigger methods can coexist.\n\n"

                            "Step 5: Clearly infer possible trigger details based on evidence. Include the following if explicitly mentioned or strongly implied by user comments:\n"
                            "- Specific time (e.g., '0-1点')\n"
                            "- Specific geographic/location conditions (e.g., '中国', '关闭VPN')\n"
                            "- Waiting time after app launch (e.g., '10秒')\n"
                            "- Click locations and counts (e.g., '点击屏幕底部5次')\n"
                            "- Exact input texts and their fields (e.g., '在意见反馈处输入5201080')\n\n"

                            "Step 6: Handling Fields:\n"
                            "- If 'text_content' (specific input content) is unknown, DO NOT generate a trigger involving text input.\n"
                            "- Do NOT write explanations or descriptions inside empty fields.\n"
                            "- If the exact location of text input ('text_location') is unknown, you may possibly assume a common field (e.g., '意见反馈', '问题反馈', '文本输入框') if ever mentioned; otherwise explicitly leave unknown.\n\n"
                            
                            "Use provided comments for inference:\n"
                            "Example [Automatic transformation - wait 10s]:\n"
                            "[['打开', '打开等待'], ['提示更新闪退', '一直提示更新，无论更新与不更新都闪退，无法观看'], ['更新', '更新']]\n\n"

                            "Example [Clicking locations - click donuts 5 times]:\n"
                            "[['点击', '点击5次'], ['甜甜圈', '一直点'], ['打开', '打开']]\n\n"
                            
                            "Example [Specific text input trigger - two 5201080 or 666 possible]:\n"
                            "[['求暗号', '打不开'], ['口令', '五二0零八零 数字'], ['口令', '5⃣️2⃣️o1⃣️o8⃣️o'], ['提交完进不去啊', '口令，我爱零你零八零'], ['666', '666'], ['太良心的软件 孩子很喜欢！！！！', '太好用了 孩子很喜欢520！！！！我真的会爱上1080遍！！！！'], ['666', '666']]\n\n"

                            "Example [Specific text input trigger - two 橘子真甜 or 666 possible]:\n"
                            "[['橘子真甜', '橘子真甜'], ['橘子真甜', '橘子真甜'], ['提交', '口令'], ['666', '666'], ['橘子真甜', '橘子真甜'], ['橘子真甜', '橘子真甜']]\n\n"

                            "Example [Spatiotemporal conditions - in China with VPN off]:\n"
                            "[['无广子', '关闭 V*N'], ['定位中国', '定位中国'], ['打不开', '允许连接网络权限']]\n\n"

                            "Example text input triggers: 666, ys777, 大耳朵图图, 大马猴, 你好123, 变美要自信, 我爱中国, 999999, 5201080, 小柿子, 蒙太奇, 和气生财, 好运家, 龙年大吉, 八方来财, 鸭梨鸭梨, 继续加油, 新年快乐, 好久不见, 完美工具, c1, c7, and so on.\n"
                            
                            #"Only infer a text input trigger if multiple comments explicitly repeat the same unusual phrase or number sequence.\n"
                            "Positively infer a text input trigger if comments indicate unusually repeating phrases or number sequences, that is abrupt or similar to known examples.\n"
                            "Clearly specify any uncertainties by marking fields as 'None' if not confidently determined.\n\n"

                            "Ensure all provided examples guide your inference clearly and accurately. Do not deviate from the output schema or include unsupported assumptions."
                        ),
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    # {
                    #     "type": "text",
                    #     "text": (
                    #         "README： Disguise app represents an app whose actual function does not match the displayed function on the app store, and requires special conditions to make the APP that meets the display page 'transform'. I need to infer from the basic information and comments whether it is a disguised application and the triggering method.\n"

                    #         "-----Trigger method-----\n"

                    #         "Firstly, I will provide a series of triggering methods for disguised apps:\n"
                    #         "[1] Open/open multiple times/wait: such as opening directly, restarting to automatically transform, waiting for 10 seconds after open.\n"
                    #         "[2] Specific time and geographical conditions: such as locate in China, open within a specific time range.\n"
                    #         "[3] Specific behavior: Specific click operations, such as clicking 5 times in the blank space, continuously clicking the homepage icon 5 times, Watch advertisements.\n"
                    #         "[4] Specific input content or selection: Enter specific content in a text box or select specific content in a selection box (such as 问题反馈, 我的建议, 参数栏, 搜索框).\n"
                    #         "these methods can be mixed: for example, after entering 999999 in the calculator, click three times to calculate (mixed with [3] and [4]); Click the screen loading bar 10 times and then re-enter (mixed with [1] and [3])\n"
                                
                    #         # "领域知识补充：不同类别的应用通常倾向于采用以下触发方式：\n"
                    #         # "- 科学计算类应用：倾向于使用触发方式[4]，会要求在参数空格中填入特定内容，如要求在计算器中输入特定数字后点击计算\n"
                    #         # "- 游戏娱乐类应用：倾向于使用触发方式[1]或[3]\n"
                    #         # "- 生活工具类应用：倾向于使用触发方式[3]或[4]\n"
                    #         # "- 媒体娱乐类应用：可能通过观看广告或进行简单操作触发隐藏内容，常见于[1]和[3]\n"
                                       
                    #         # "-----可疑评论-----\n"

                    #         # "经观察，部分伪装应用的评论会与该应用的简介和功能完全不符。具体而言，这些评论可以分为以下2类：\n"
                    #         # "[1]用户询问、抱怨类：如“广告太多”“怎么变身啊”“根本进不去”“一点就闪退”“啥都没有”“提交了也进不去”“口令是啥啊”“骗子应用”\n"
                    #         # "[2]触发码类：针对[4]特定输入操作变身类，一些app的评论会包含需要输入的关键词，如“爱其意8：爱其意8”“你好123：你好123”“大马猴：大马猴”“666：666”；这类评论通常简短而重复率高，同时评论标题和内容（以：分割）有很大的可能是一样的\n"

                    #         "-----Requirement-----\n"
                    #         "Please identify the keywords that may be related to the four triggering methods based on the comment information, including: waiting time, geographic or time conditions, click location, click count, and specific input text.\n"

                    #         "-----Supplementary Information-----\n"
                    #         # "请从以下信息中判断一个应用是否为伪装应用，如果是，输出其可能的触发方式\n"
                    #         # "请注意：\n"
                    #         # " - 非伪装应用也可能因为问题多出现投诉；666、很想看等也是交际常用语；特定输入可能与评论区的手机号、网址重合\n"
                    #         # " - 如果一个应用的评论只有'进不去''怎么变啊''骗子'，该应用可能是伪装应用\n"
                    #         # "如果一个应用的评论出现了明显与该应用功能不同的内容，该应用可能是伪装应用\n"
                    #         "Please pay close attention to comments with the same title and content, usually chinese phrases with a length of 3-5 and it's not a question sentence, which might be specific input content.\n"
                    #         "Only when phrases unrelated to the application appear multiple times in comments, or when a phrase is clearly associated with words like trigger or transform, will it be considered as specific input content.\n"
                    #         "Example specific content input: 666, ys777, 大耳朵图图, 大马猴, 你好123, 变美要自信, 我爱中国, 999999, 5201080, 小柿子, 蒙太奇. Example specific content input location: 意见反馈, 文本输入框, 搜索框\n"
                    #         # " - 大量出现的评论关键词可能是触发方式[4]所需的触发码，这类评论通常简短而重复率高，同时评论标题和内容（以:分割）有很大可能是一样的\n"
                    #         # "接下来将会提供一系列包含应用名称、应用简介、所有评论的标题及内容信息\n"
                    #         # "请根据一个APP的全部评论综合严格判断是否为伪装应用，以及其触发方式。触发方式以触发域的格式进行预测，包含时间、地点、点击位置、输入内容等等字段，请根据评论填充能够预测的字段，无法预测的则返回None\n"

                    #         "-----Input-----\n"

                    #     )  
                    # },
                    {
                        "type": "text",
                        "text": "Appstore metadata:\n\n",
                    },
                    {
                        "type": "text",
                        "text": 
                            f"name: {appstore_info['name']}\n" + 
                            f"category: {appstore_info['category']}\n" + 
                            f"description: {appstore_info['description']}\n\n"
                    },
                    {
                        "type": "text",
                        "text": "Comments (provided in format 'Title: Content'):\n\n",
                    },
                    {
                        "type": "text",
                        "text": comment_preprocessing(comment_list),
                    },
                ],
            }
        ]

        logger.debug(f"User reviews: {comment_preprocessing(comment_list)}")

        # Generate the response schema from the Pydantic model
        response_format = TriggerResponse

        # Call the OpenAI API
        try:
            completion  = self.client.beta.chat.completions.parse(
                model="gpt-4o-2024-11-20",
                messages=messages,
                temperature=0.1,
                n=1,
                response_format=response_format,
            )

            # Extract the response content
            result = completion.choices[0].message.parsed

            logger.debug(
                f"Predict_Task: token consumed: "
                f"{completion.usage.prompt_tokens} + {completion.usage.completion_tokens} "
                f"= {completion.usage.total_tokens}"
            )

            token_record('predict_task', completion.usage.prompt_tokens, completion.usage.completion_tokens)

            return result
        except Exception as e:
            logger.error(f"Error during OpenAI API call: {e}")
            raise e

    @staticmethod
    def _render_trigger_response(trigger_field: TriggerField) -> str:
        '''动态生成触发任务的描述，拼接非空字段以生成完整的自然语言任务说明'''
        components = []
        if trigger_field.time:
            components.append(f"在'{trigger_field.time}'时间")
        if trigger_field.location:
            components.append(f"在'{trigger_field.location}'地理条件")
        if trigger_field.wait:
            components.append(f"打开APP并等待'{trigger_field.wait}'秒")
        if trigger_field.click_location or trigger_field.click_time:
            components.append(f"在'{getattr(trigger_field, 'click_location', None) or '{任意}'}'位置连续点击'{str(getattr(trigger_field, 'click_time', None) or 1)}'下")
        if trigger_field.text_location or trigger_field.text_content:
            components.append(f"在'{getattr(trigger_field, 'text_location', None) or '{文本输入框}'}'位置输入'{getattr(trigger_field, 'text_content', None) or '{潜在触发词}'}'内容")
        return "，".join(components) + "，完成触发；结合不同角度以及页面语义进行探索（优先特别注意 functional icons、menu、tab bars、my center、feedback、search、settings、preferences、personal、about、chat、message 等功能）" if components else ""
    
    @time_consumed
    def detect_mask(self, base64_screenshot: str, text_list: List[str], appstore_info: dict):
        """
        Check whether the screenshot and text conflict with the information in appstore.

        Args:
            base64_screenshot (str): The screenshot encoded in base64.
            text_list (List[str]): The text list in current UI.
            appstore_info (dict): The information in appstore. Containing category, name and description.

        Returns:
            str: the result.
        """
        # Prepare the messages for the chat completion
        # Construct the content as a list of content parts

        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        # "text": (
                        #     "I need to determine if an application provides pages and features that are completely unrelated to the descriptions in the application store. " 
                        #     # "Below, I will provide the basic information of an app from the app store, along with a runtime screenshot and selected UI text excerpts. " 
                        #     "Please evaluate whether the run time screenshot and text are completely unrelated to the description provided in the basic information, and output 'Yes' or 'No':\n\n"

                        #     "● Output 'Yes': \n"
                        #     "   - The screenshot and text clearly and completely contradict the appstore description, showing features and pages that are totally unrelated and possibly illegal.\n"
                        #     "   - The screenshot is actively switching apps to a Safari browser that is totally unrelated.\n"
                        #     "   - The screenshot and text shows alerts or pop-ups contain explicit instruction to activate hidden modes or additional behaviors(e.g., '开启隐藏模式', '解锁所有功能', '请按要求进行解锁', '输入享受完整功能').\n\n"
                        #     # "   - Do not consider advertisements and promotions when making this determination.\n"
                        #     # "   - Do not consider permissions and services mentioned in the page, unless these permissions being actively used in current page.\n\n"

                        #     "● Output 'No': \n"
                        #     "   - The screenshot and text potentially align with the appstore description.\n"
                        #     "   - Such discrepancy is brought by user input in a input field, which is not part of the app itself.\n"

                        #     # "● Output 'Unable to determine': \n"
                        #     "   - The presence of marketing elements (e.g., advertisements, promotions, app stores, often containing keywords such as '广告', '立即下载', '摇一摇', '前往APP', '查看详情', '第三方应用') cover and obscure the original page, making the functional evaluation unclear.\n"
                        #     # "   - Permission requests for features such as photos, videos, or location services do not independently imply hidden functionality unless evidence shows these permissions being actively used in an undocumented way.\n"
                        #     "   - The page shown is generic or neutral (e.g., '意见反馈', '个人中心', '用户协议', '菜单', '输入框', '登录'), and no direct evidence of hidden features is observed.\n"
                        #     # "   - Permissions, services or settings descriptions are present, but no evidence of direct use of those permissions beyond stated features is visible.\n"
                        #     "   - The page is a system-level alert or pop-up (app store, system settings, file dialog, sharing, etc.), which is not related to the app.\n"
                        #     "   - The page is still loading, presenting a empty, blank or logo-only page temporarily.\n"
                        # )
                        "text": (
                            "Your task is to determine if the actual runtime UI of an app (as shown in the provided screenshot and accompanying texts) "
                            "is clearly unrelated or contradictory to the official description provided on the app store.\n\n"
                            
                            "Step-by-Step Evaluation Criteria:\n\n"

                            "Step 1: Evaluate whether the screenshot and UI text directly conflict with the appstore description:\n"
                            "- Output 'Yes' only if:\n"
                            "  • The app clearly provides features or content totally unrelated to its official description.\n"
                            "  • The screenshot explicitly indicates switching or redirecting to another unrelated app or website (e.g., browser with unrelated content).\n"
                            "  • The UI explicitly mentions unlocking hidden modes or additional behaviors, such as phrases: '开启隐藏模式', '解锁所有功能', '输入暗号解锁', '输入享受完整功能'.\n\n"
                            
                            "- Output 'No' only if:\n"
                            "  • UI screens clearly still loading, incomplete, or temporarily blank (e.g., blank screens, loading spinners, logos only without content).\n"
                            "  • Generic UI pages without explicit evidence of hidden functionality (e.g., simple login screens, feedback pages, settings pages, input screens without explicit trigger instructions).\n"
                            "  • Unclear UI pages with least elements but not showing any suspicious clues about hidden functionality (e.g., an input field without any description, an image without any description).\n"
                            "  • The app’s runtime UI matches, or potentially matches, its described functionalities from the app store.\n"
                            "  • Observed discrepancies originate solely from user-entered text rather than built-in app behaviors.\n\n"

                            "- Otherwise, output 'Yes' or 'No' whichever best matches the conditions.\n\n"

                            "Step 2: Provide detailed reasoning:\n"
                            "- Explicitly reference exact UI texts or visual elements from the screenshot supporting your conclusion.\n"
                            "- Highlight critical discrepancies or confirmations, clearly connecting them to the provided appstore description.\n\n"
                            
                            "Step 3: Assign a confidence score (0.0–1.0):\n"
                            "- 0.8–1.0: Strong, explicit contradiction (e.g., hidden mode clearly mentioned).\n"
                            "- 0.5–0.7: Moderate contradiction (e.g., suspicious but not explicitly stated).\n"
                            "- Below 0.5: Weak or no contradiction.\n\n"
                            
                            "Step 4: Clearly identify if advertisements or promotions significantly obscure UI evaluation:\n"
                            "- Output 'Yes' if promotional pop-ups or ads heavily obscure the UI, making accurate evaluation difficult.\n"
                            "- Output 'No' otherwise.\n\n"
                            
                            "Examples:\n\n"
                            
                            "[Example 1 - Clear contradiction (Output: Yes)]:\n"
                            "  Appstore Description: 'An educational app for kids to learn math.'\n"
                            "  Screenshot/UI text: A pop-up, '点击此处进入隐藏模式', '请输入口令解锁所有影片'.\n"
                            "  Reasoning: 'UI explicitly mentions unlocking hidden video features unrelated to math education.'\n"
                            "  Score: 0.95\n"
                            "  Ads: No\n\n"
                            
                            "[Example 2 - No contradiction (Output: No)]:\n"
                            "  Appstore Description: 'A fitness app tracking daily steps and calories.'\n"
                            "  Screenshot/UI text: A page with lists and categories, '今日步数: 1024', '消耗热量: 320千卡'.\n"
                            "  Reasoning: 'Runtime UI fully matches described fitness tracking features.'\n"
                            "  Score: 0.1\n"
                            "  Ads: No\n\n"
                            
                            "[Example 3 - Ads obscuring evaluation (Output: No, Ads: Yes)]:\n"
                            "  Appstore Description: 'News aggregator providing global headlines.'\n"
                            "  Screenshot/UI text: An overlaying ad, '立即下载精彩游戏', '广告倒计时: 30秒'.\n"
                            "  Reasoning: 'Advertisements completely obscure underlying content; unable to assess actual functionality clearly.'\n"
                            "  Score: 0.4\n"
                            "  Ads: Yes\n\n"

                            "[Example 4 - Clear contradiction (Output: Yes)]:\n"
                            "  Appstore Description: 'A tool for calculating resistance in an electric circuit.'\n"
                            "  Screenshot/UI text: Blank page, '大鱼影视', '在线观看'.\n"
                            "  Reasoning: 'UI is generic but runtime texts uncover the hidden functionality of watching movies.'\n"
                            "  Score: 0.8\n"
                            "  Ads: No\n\n"

                            "Strictly adhere to the output schema. Make no assumptions beyond provided evidence and examples. "
                            "If uncertain, reflect that clearly in reasoning and score."
                        ),
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Appstore metadata:\n\n",
                    },
                    {
                        "type": "text",
                        "text": (
                            "Name: " + appstore_info['name'] + "\n"
                            "Category: " + appstore_info['category'] + "\n"
                            "Description: " + appstore_info['description'].replace('\n', ' ') + "\n"
                        )
                    },
                    {
                        "type": "text",
                        "text": "\n\nPart of UI runtime texts:\n",
                    },
                    {
                        "type": "text",
                        "text": '\n'.join(text_list) + "\n\n",
                    },
                    {
                        "type": "text",
                        "text": "Screenshot image for reference:\n",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_screenshot}"
                        },
                    },
                ],
            }
        ]

        # Generate the response schema from the Pydantic model
        response_format = DetectMaskResult

        # Call the OpenAI API
        try:
            completion  = self.client.beta.chat.completions.parse(
                model="gpt-4o-2024-11-20",
                messages=messages,
                temperature=0.1,
                n=1,
                response_format=response_format,
            )

            # Extract the response content
            result = completion.choices[0].message.parsed

            logger.debug(f"Detect_Mask: text_list: {text_list}")
            
            logger.debug(
                f"Detect_Mask: token consumed: "
                f"{completion.usage.prompt_tokens} + {completion.usage.completion_tokens} "
                f"= {completion.usage.total_tokens}"
            )

            token_record('detect_mask', completion.usage.prompt_tokens, completion.usage.completion_tokens)

            logger.debug(f"Detect_Mask: result: {result}")

            return result
        except Exception as e:
            logger.error(f"Error during OpenAI API call: {e}")
            raise e

    
    @staticmethod
    def _get_slim_ui(ui: UIView) -> str:
        """
        Convert UIView to a "slim" text so GPT can focus on essential data
        """
        slim_ui = []
        for element in ui.elements:
            # Decide category
            if element.ui_type == UIElementType.INPUT_TEXT:
                category = "InputText"
            elif element.ui_type == UIElementType.ICON_BUTTON:
                category = "IconButton"
            elif element.ui_type == UIElementType.CHECKBOX:
                category = "Checkbox"
            elif element.ui_type == UIElementType.RADIO:
                category = "Radio"
            elif element.ui_type == UIElementType.DROPDOWN:
                category = "Dropdown"
            elif element.ui_type in [
                UIElementType.BUTTON,
                UIElementType.TEXT_BUTTON,
                UIElementType.OTHER_INTERACTABLE
            ] or element.clickability > 0.5:
                category = "Button"
            elif element.ui_type == UIElementType.TEXT_ONLY:
                category = "Text"
            elif element.ui_type == UIElementType.ICON_ONLY:
                category = "Image"
            else:
                # Fallback logic
                category = "Text" if element.text.strip() else "Image"
            
            slim_ui.append({'id': element.id, 'category': category, 'description': element.description, 'text': element.text, 'location': element.location})
        
        slim_ui_text = json.dumps(slim_ui, indent=2, ensure_ascii=False)
        return slim_ui_text

    @staticmethod
    def _concat_slim_ui(slim_ui_text, semantic_ui: SemanticView) -> str:
        slim_ui_json = json.loads(slim_ui_text)

        for element in semantic_ui.elements:
            if item := next((x for x in slim_ui_json if x["id"] == element.id), None):
                item["description"] += "Additionally, you considered the current UI element or interface to be semantically related to ad UI dark patterns. For the specific category and reasoning process, refer to the “type” and “evidence” fields."
                item["type"] = element.type
                item["evidence"] = element.evidence
            elif element.id != -1:
                # Decide category
                if element.ui_type == UIElementType.INPUT_TEXT:
                    category = "InputText"
                elif element.ui_type == UIElementType.ICON_BUTTON:
                    category = "IconButton"
                elif element.ui_type == UIElementType.CHECKBOX:
                    category = "Checkbox"
                elif element.ui_type == UIElementType.RADIO:
                    category = "Radio"
                elif element.ui_type == UIElementType.DROPDOWN:
                    category = "Dropdown"
                elif element.ui_type in [
                    UIElementType.BUTTON,
                    UIElementType.TEXT_BUTTON,
                    UIElementType.OTHER_INTERACTABLE
                ] or element.clickability > 0.5:
                    category = "Button"
                elif element.ui_type == UIElementType.TEXT_ONLY:
                    category = "Text"
                elif element.ui_type == UIElementType.ICON_ONLY:
                    category = "Image"
                else:
                    # Fallback logic
                    category = "Text" if element.text.strip() else "Image"

                slim_ui_json.append({
                    'id': element.id,
                    'category': category,
                    'description': element.description + "Additionally, you considered the current UI element or interface to be semantically related to ad UI dark patterns. For the specific category and reasoning process, refer to the “semantic_type” and “evidence” fields.",
                    'text': element.text,
                    'location': element.location,
                    'semantic_type': element.type,
                    'evidence': element.evidence
                })

        slim_ui_text_append = json.dumps(slim_ui_json, indent=2, ensure_ascii=False)
        return slim_ui_text_append

    @staticmethod
    def _get_action_history(action_history: List[str]) -> str:
        """
        Convert action history to a text
        """
        action_history_text = "[*] 1. Launch the application from SpringBoard\n" + "\n".join([("[*] " if any(s in action.lower() for s in ['. restart ', 'restart the app']) else "") + f"{i}. {action}" for i, action in enumerate(action_history, start=2)])
        return action_history_text

    @time_consumed
    def analyze_action(self, analyzed_ui: UIView, task: str, action_history: List[str], recent_repeat_action: str = '', recent_repeat_count: int = 0) -> UserAction:
        """
        Analyzes the current UI (analyzed_ui), the user's 'task', and 'action_history', 
        then instructs GPT to propose a valid UserAction. This method is constructed 
        similarly to analyze_screenshot(), making an API call and parsing GPT output 
        as a UserAction according to a given schema.
        """

        slim_ui_text = self._get_slim_ui(analyzed_ui)
        action_history_text = self._get_action_history(action_history)
        
        # Build system and user messages
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        # "text": (
                        #     "Fully analyze the current UI state and propose next actions:\n\n"

                        #     "1. Tasks & Key Points:\n"
                        #     "- Understand the user's ultimate task, current UI JSON data (especially description field for each element) and past action history.\n"
                        #     "- Actively explore the app by clicking tabs, buttons, following page instructions if the subtask cannot be completed on the current page.\n"
                        #     "- Learn the patterns and pages from the previous action history and avoid wasting time on unnecessary, redundant and repeating actions.\n"
                        #     "- Retry whole or part of action series for at most 2 times if it did not yield the expected feedback or result.\n"
                        #     "- Never repeat identical action series in previously visited pages, unless such action potentially leads to a new state that never observed before.\n\n"

                        #     "2. Basic Principles:\n"
                        #     "- Treat the UI JSON as the primary reference.\n"
                        #     "- Match the UI element ID exactly with action target.\n"
                        #     "- Propose actions that advance the user task or has never been done before; avoid unnecessary, unrelated and repeating steps.\n"
                        #     "- If the task is somehow related to the current UI page, prioritize actions that advance the user task by interacting just with current UI page before exploring other pages.\n"
                        #     "- If no direct action is possible, attempt alternative navigation or following page's semantic instructions before issuing none action.\n"
                        #     "- If repeating same action series but no progress, temporarily ignore the main task and focus on exploring the app's original features and functionalities while understanding the page's semantic instructions, before issuing none action.\n\n"

                        #     "3. Restrictions & Conditions:\n"
                        #     "- Handle urgent and topmost alerts or advertisements first. Close or skip them at minimal effort.\n"
                        #     "- Handle urgent non-intended alerts before any advertisements.\n"
                        #     "- Grant any kind of system or tracking permissions if required by the app.\n"
                        #     "- Halt any operations that involve local file/photo/image dialog, unrelated to this app and task.\n"
                        #     "- If the UI page seems to be still loading or no action is possible after careful consideration, suggest doing nothing and wait for the next attempt.\n"
                        #     "- If the advertisements are not closeable, suggest restarting the app after careful consideration. Do not click on any advertisement details.\n"
                        #     "- If still no actions are possible after multiple attempts of waiting, or it's impossible to navigate to the other page from current UI, suggest restarting the app after careful consideration.\n"
                        #     "- Do not easily suggest restart unless stucking in current stage.\n"
                        #     "- Consider the task has been completed only based on previous feedback message and action history.\n"
                        #     "- Directly issue Input action towards TextField elements (Input action will click and then clear the text field first), instead of clicking on them to reduce the number of clicks.\n\n"

                        #     "4. Output & Reasoning:\n"
                        #     "- Follow the output JSON schema and field description.\n"
                        #     "- Summaries must remain concise and accurate.\n\n"
                        # )
                        "text": (
                            "Your task is to propose clear and actionable user interactions based on the current UI state, historical actions, and the main task provided.\n\n"

                            "Detailed Steps and Restrictions:\n\n"

                            "Step 1: Fully assess the current UI:\n"
                            "- Clearly summarize the page’s primary function and interaction points (buttons, input fields, tabs).\n"
                            "- Handle urgent UI states first (alerts, pop-ups, advertisements). Ads or alerts must be closed/skipped immediately if possible, unless they are mentioned in the main task.\n"
                            "- If topmost UI elements are non-dismissible, suggest app restart.\n\n"

                            "Step 2: Evaluate the action history:\n"
                            "- Clearly summarize previously visited pages and performed actions.\n"
                            "- Explicitly indicate repeated action sequences (more than twice), and clearly state alternatives if repetition yields no results.\n\n"

                            "Step 3: Identify a precise sub-task:\n"
                            "- Clearly define a concise, actionable sub-task advancing toward the main goal.\n"
                            "- Explicitly verify and clearly state completeness or incompleteness of all required conditions (e.g., all inputs completed, required navigational steps previously done, prerequisite UI interactions).\n"
                            "- Provide explicit reasoning why this sub-task is critical based on prior interactions and current UI context, and necessary for the logical flow toward task completion.\n\n"

                            "Step 4: Propose detailed next actions:\n"
                            "- Actions must be sequential, logically coherent, fully justified, and explicitly validated through the completeness check (including UI element IDs).\n"
                            "- Directly issue 'Input' actions instead of clicks for text entry fields to minimize interactions.\n"
                            "- If no immediate actions are feasible, propose a 'None' action clearly stating the reason (e.g., loading, waiting for UI update).\n"
                            "- Mark task completion explicitly when previous actions clearly indicate the goal is achieved.\n\n"

                            "Step 5: Summarize clearly:\n"
                            "- Provide concise UI page summary clearly describing current context and interaction purpose.\n"
                            "- Provide a precise action summary clearly stating planned actions and rationale, including exact input texts if applicable.\n\n"

                            "Strictly follow provided output schema and examples. Ensure actions are practical, relevant, and clearly explained."
                        )
                    }
                ],
            },
            # 2. System message (context injection): Task
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "The user's task to accomplish: \n",
                    },
                    {
                        "type": "text",
                        "text": task + "\n\n",
                    },
                ]
            },
            # 3. User message: Slim UI data
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Current UI summary:\n" + analyzed_ui.description + "\n\n"
                            "The UI has a visible feedback/status/error message: " + analyzed_ui.feedback_message if analyzed_ui.feedback_message else "" + "\n\n"
                            "The UI has an alert or pop-up, which should be handled before doing anything else.\n\n" if analyzed_ui.is_alert_topmost else ""
                            "The UI has an urgent topmost overlaying advertisement, which must be handled before doing anything else.\n\n" if analyzed_ui.ads and analyzed_ui.ads.is_ad_topmost and analyzed_ui.ads.is_ad_closeable and analyzed_ui.ads.ad_close_button_id > 0 else ""
                            # If not closeable, the Bot is not able to handle it.
                            "\n"
                        )
                    },
                    {
                        "type": "text",
                        "text": "Current UI view:\n"
                    },
                    {
                        "type": "text",
                        "text": slim_ui_text + "\n\n"
                    },
                    {
                        "type": "text",
                        "text": "Previous action history:\n" + action_history_text + "\n\n" + ((
                            "The recent action series '" + recent_repeat_action + "' has been repeated " + str(recent_repeat_count) + " times.\n"
                            "Do not repeat this action series again. Actively seek and propose alternatives.\n\n"
                        ) if recent_repeat_count >= 3 else "")
                    },
                    {
                        "type": "text",
                        "text": "The user's task (again) is to:\n",
                    },
                    {
                        "type": "text",
                        "text": task + "\n\n",
                    },
                    {
                        "type": "text",
                        "text": "Now start proposing the next user action(s) to move toward completing the task.",
                    },
                ],
            },
        ]

        # We want GPT to parse the final response as a UserAction
        response_format = UserAction

        # Token counting for demonstration/logging
        from tiktoken import encoding_for_model
        enc = encoding_for_model("gpt-4o")

        #text_tokens = sum(len(enc.encode(content["text"])) for msg in messages for content in msg["content"] if "text" in content)
        #logger.debug(f"Estimated request tokens: {text_tokens}")
        #logger.debug(f"Input prompt: %s", '\n'.join([content["text"] for msg in messages for content in msg["content"] if "text" in content]))

        try:
            completion = self.client.beta.chat.completions.parse(
                model="gpt-4o-2024-11-20",
                messages=messages,
                temperature=0.1,
                n=1,
                response_format=response_format,
            )

            # Parsed result is a valid UserAction object
            result = completion.choices[0].message.parsed

            logger.debug(
                f"Analyze_Action: token consumed: "
                f"{completion.usage.prompt_tokens} + {completion.usage.completion_tokens} "
                f"= {completion.usage.total_tokens}"
            )

            token_record('analyze_action', completion.usage.prompt_tokens, completion.usage.completion_tokens)

            return result

        except Exception as e:
            logger.error(f"Error during OpenAI API call for analyze_action: {e}")
            # Return a fallback "none" action if GPT fails
            raise e

    @time_consumed
    def semantic_analyze_action(self, analyzed_ui: UIView, semantic_analyzed_ui: SemanticView, task: str, action_history: List[str], recent_repeat_action: str = '', recent_repeat_count: int = 0) -> SemanticUserAction:
        """
        Analyzes the current UI (analyzed_ui), the user's 'task', and 'action_history',
        then instructs GPT to propose a valid UserAction. This method is constructed
        similarly to analyze_screenshot(), making an API call and parsing GPT output
        as a SemanticUserAction according to a given schema.
        """

        slim_ui_text = self._get_slim_ui(analyzed_ui)
        slim_ui_text = self._concat_slim_ui(slim_ui_text, semantic_analyzed_ui)
        action_history_text = self._get_action_history(action_history)

        # Build system and user messages
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Your task is to propose clear and actionable user interactions based on the current UI state, historical actions, and the main task provided.\n\n"

                            "Detailed Steps and Restrictions:\n\n"

                            "Step 1: Fully assess the current UI:\n"
                            "- Clearly summarize the page’s primary function and interaction points (buttons, input fields, tabs).\n"
                            "- Handle urgent UI states first (alerts, pop-ups). Alerts must be closed immediately if possible, unless they are mentioned in the main task.\n"
                            # "- If topmost UI elements are non-dismissible, suggest app restart.\n\n"

                            "Step 2: Evaluate the action history:\n"
                            "- Clearly summarize previously visited pages and performed actions.\n"
                            "- Explicitly indicate repeated action sequences (more than twice), and clearly state alternatives if repetition yields no results.\n\n"
                            
                            '''
                            Step 3: Evaluate the user’s current task progress and its future prospects:
                            - Carefully compare the user’s previously defined exploration subtask with the actual actions, UI change history, and the number of steps already explored in this task to determine whether the subtask is still being effectively executed.
                            - Based on the current UI, determine the status of the user’s subtask (completed, failed, or processing). 
                                - “Completed”: the UI required for the current task has been sufficiently explored. 
                                - “Failed”: the user has spent considerable effort on the task but the likelihood of completion remains low. 
                                - “Processing”: the task has not yet succeeded but has a high probability of completion within the next few steps.
                            - If the current task can be completed within the next few steps, carefully estimate the number of additional UI interactions required.
                            '''

                            "Step 4: Reorder all candidate tasks in descending order of priority:\n"
                            "- Clearly define a concise, actionable, highest-priority sub-task advancing toward the main goal.\n"
                            "- Rank all tasks in order of priority, with those closer to the top having higher priority, according to the following guidelines:\n"
                                "- Place the task that should currently be executed at the top of the list.\n"
                                    "- For apps with onboarding tutorials (especially games), the top priority is to finish them as quickly as possible by skipping or following the required actions.\n"
                                "- Emphasize exploration breadth; never repeatedly explore a single app feature or interface such as settings, ad removal, or purchases.\n"
                                    "- Specifically, except for explore tasks, tasks that have already been executed (whether failed or completed) should be assigned lower priority.\n\n"
                                # "- Tasks that are expected to be completed with fewer interactions on the current UI have higher priority.\n"
                                # "- For the task the user is currently performing, assign the lowest priority if its status is 'completed,' and a relatively low priority if it has 'failed.'\n"
                                # "- Tasks that have not been executed for a longer period of time have higher priority.\n"
                            # "- Provide explicit reasoning why the highest-priority sub-task is critical based on prior interactions and current UI context, and necessary for the logical flow toward task completion.\n\n"

                            "Step 5: Propose detailed next actions:\n"
                            # "- If you decide to perform random UI exploration and button clicks, only provide the maximum number of interactions for the random exploration. Enable random exploration in the following two situations:\n"
                            #     "- When the system has remained on the current UI for multiple rounds and performed several repeated actions without receiving any response.\n"
                            #     "- When the current task has no clear triggering method and there are no UI elements on the current interface that explicitly correspond to the task.\n"
                            # "- Otherwise, carefully design the next interaction step:\n"
                            "- Carefully design the next interaction step: \n"
                                "- Actions must be sequential, logically coherent, fully justified, and explicitly validated through the completeness check (including UI element IDs).\n"
                                "- If no immediate actions are feasible, propose an 'Idle' action clearly stating the reason (e.g., loading, waiting for UI update).\n"
                                "- Never perform ID-based actions (e.g., click, hover) on elements that do not including ID in the UI View; only click elements with valid IDs."

                            "Step 6: Summarize clearly:\n"
                            "- Provide concise UI page summary clearly describing current context and interaction purpose.\n"
                            "- Provide a precise action summary clearly stating planned actions and rationale, including exact input texts if applicable.\n\n"

                            "Strictly follow provided output schema and examples. Ensure actions are practical, relevant, and clearly explained."
                        )
                    }
                ],
            },
            # 2. System message (context injection): Task
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "The user's main task to accomplish: \n",
                    },
                    {
                        "type": "text",
                        "text": task + "\n\n",
                    },
                ]
            },
            # 3. User message: Slim UI data
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            # "Current UI status:\n" + semantic_analyzed_ui.semantic_interface.status + "\n\n"
                            "Current UI summary:\n" + analyzed_ui.description + "\n\n"
                            "The UI has a visible feedback/status/error message: " + analyzed_ui.feedback_message if analyzed_ui.feedback_message else "" + "\n\n"
                            "The UI has an alert or pop-up, which should be handled before doing anything else.\n\n" if analyzed_ui.is_alert_topmost else ""
                            "\n"
                        )
                    },
                    {
                        "type": "text",
                        "text": "Current UI view:\n"
                    },
                    {
                        "type": "text",
                        "text": slim_ui_text + "\n\n"
                    },
                    {
                        "type": "text",
                        "text": "Previous action history:\n" + action_history_text + "\n\n" + ((
                             "The recent action series '" + recent_repeat_action + "' has been repeated " + str(
                             recent_repeat_count) + " times.\n"
                             "Do not repeat this action series again. Actively seek and propose alternatives.\n\n"
                             ) if recent_repeat_count >= 3 else "")
                    },
                    {
                        "type": "text",
                        "text": "The user's task (again) is to:\n",
                    },
                    {
                        "type": "text",
                        "text": task + "\n\n",
                    },
                    {
                        "type": "text",
                        "text": "Now start proposing the next user action(s) to move toward completing the task.",
                    },
                ],
            },
        ]

        # We want GPT to parse the final response as a UserAction
        response_format = SemanticUserAction

        # Token counting for demonstration/logging
        from tiktoken import encoding_for_model
        enc = encoding_for_model("gpt-4o")

        # text_tokens = sum(len(enc.encode(content["text"])) for msg in messages for content in msg["content"] if "text" in content)
        # logger.debug(f"Estimated request tokens: {text_tokens}")
        # logger.debug(f"Input prompt: %s", '\n'.join([content["text"] for msg in messages for content in msg["content"] if "text" in content]))

        try:
            completion = self.client.beta.chat.completions.parse(
                model="gpt-4o-2024-11-20",
                messages=messages,
                temperature=0.1,
                n=1,
                response_format=response_format,
            )

            # Parsed result is a valid UserAction object
            result = completion.choices[0].message.parsed

            logger.debug(
                f"Analyze_Action: token consumed: "
                f"{completion.usage.prompt_tokens} + {completion.usage.completion_tokens} "
                f"= {completion.usage.total_tokens}"
            )

            token_record('semantic_analyze_action', completion.usage.prompt_tokens, completion.usage.completion_tokens)

            return result

        except Exception as e:
            logger.error(f"Error during OpenAI API call for analyze_action: {e}")
            # Return a fallback "none" action if GPT fails
            raise e
        
    @time_consumed
    def analyze_screenshot(self, base64_screenshot: str, ui_runtime_elements_json: dict, focus_ads: bool = False) -> tuple[UIView, SemanticView]:
        """
        Analyzes a screenshot and extracts significant and clickable UI elements.

        Args:
            base64_screenshot (str): The screenshot encoded in base64.
            ui_runtime_elements_json (dict): JSON dictionary of extracted UI runtime elements.
            focus_ads (bool): Whether there is an ad.

        Returns:
            UIView: An object containing significant and clickable UI elements with descriptions,
                    visibility, and confidence of clickability.
        """
        # Ensure ui_runtime_elements_json is a dictionary
        if isinstance(ui_runtime_elements_json, str):
            try:
                ui_runtime_elements = json.loads(ui_runtime_elements_json)
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON input: {e}")
                return None
        else:
            ui_runtime_elements = copy.deepcopy(ui_runtime_elements_json)
        
        # Remove all attributes starting with '_'
        def traverse(view):
            for k, v in view.copy().items():
                if k.startswith('_'):
                    del view[k]
                elif isinstance(v, dict):
                    traverse(v)
                elif isinstance(v, list):
                    for item in v:
                        if isinstance(item, dict):
                            traverse(item)
        traverse(ui_runtime_elements)

        from utils import _dump_xml

        # Prepare the messages for the chat completion
        # Construct the content as a list of content parts

        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        # "text": (
                        #     "Fully analyze the screenshot and its runtime UI elements:\n\n"

                        #     "1. Tasks & Key Points:\n"
                        #     "- Match each XML element's ID and bounding box with the screenshot. If the bounding box is not fully matched, discard the element.\n"
                        #     "- Suggest waiting for UI and screenshot to synchronize if their discrepancy is huge.\n"
                        #     "- Prioritize critical elements such as the close, skip, nagivate, action buttons and functional icons including search, submit, settings, tools, heart, up vote, location, back, return, etc.\n"
                        #     "- Subviews under the same UI parent are sorted in top-to-bottom, left-to-right order.\n"
                        #     "- If an advertisement is present, make sure to find its close button (typically in the corners, labelled as 'ICON_X').\n\n"

                        #     "2. Requirements & Bounds:\n"
                        #     "- Ignore iOS native status/navigation bars and keyboard.\n"
                        #     "- Mandatorily include TextFields, Buttons, functional Icons, and any meaningful StaticTexts and Images, especially those nested within clickable Cells.\n"
                        #     "- Focus on elements with ocr_text attribute extracted from independent OCR process.\n"
                        #     "- Focus on elements with icon_label attribute extracted from independent CNN image classification.\n"
                        #     "- Actively expand the already included elements to include possibly surrounding elements.\n\n"

                        #     "3. Restrictions & Conditions:\n"
                        #     "- Provide a summarized UI view describing potential user actions and the UI layout (top-down, left-right), including title, sidebar, topbar, tabs, footer, menu, etc if any.\n"
                        #     "- Identify urgent, topmost alerts (if fully visible and exclusive to the entire UI). Identify alerts as advertisements if they have promotional texts ('Download', 'Detail', etc.).\n"
                        #     "- Don't consider an alert or advertisement urgent or topmost if it only lies in the corner and does not overlay other UI elements.\n"
                        #     "- Discard any promotional texts and buttons within an advertisement pop-up. Include only the close button of the advertisement to focus on the original purpose of the UI.\n"
                        #     "- Identify any visible feedback, status, or error messages (as alert, toast, pop-up text, bottom bar, or any borderd text that appears to be a message of action result) in UI or screenshot; or whether the UI or operation is still loading.\n\n"

                        #     "4. Output Format:\n"
                        #     "- Adhere strictly to the output schema, which includes 'hint_elements' for all meaningful non-interactive (or partially interactive) UI hints, and 'action_elements' for distinctly interactive/clickable UI features.\n\n"
                        # )
                        "text": (
                            "Your task is to analyze provided runtime UI XML data and corresponding screenshot, carefully matching and classifying UI elements to clearly generate a structured UIView object.\n\n"

                            "Detailed Steps and Exact Restrictions:\n\n"

                            "Step 1: Carefully match XML element IDs and bounding boxes precisely with visible screenshot elements:\n"
                            "- Discard any XML elements that clearly do not match or are not fully visible in the screenshot.\n"
                            "- Prioritize essential functional elements clearly needed for interaction (close buttons, search icons, navigation buttons, input fields).\n"
                            "- Do not use IDs that are not present in the XML elements to describe items in the screenshot. Even if some buttons or text truly exist, they may not be recorded as XML elements." # AdsDP added: GPT provides some element ids that are not in UI Tree.
                            "- Ignore native OS-level status/navigation bars or keyboards.\n\n"

                            "Step 2: Classify UI elements explicitly into two categories:\n"
                            "- 'hint_elements': Clearly non-interactive or semi-interactive hints (labels, static texts, descriptive images).\n"
                            "- 'action_elements': Explicitly interactive elements (buttons, text fields, checkboxes, radios, dropdowns, clickable icons).\n\n"

                            "Step 3: Explicitly identify and describe advertisements and alerts:\n"
                            "- Clearly detect topmost overlay advertisements requiring immediate dismissal; identify their exact close buttons clearly labeled (e.g., 'X', '关闭', '跳过').\n"
                            "- Clearly detect any fully visible, topmost urgent alerts requiring immediate action.\n"
                            "- If an ad or alert is partially visible or non-urgent (small corner overlay), clearly mention it but mark as non-topmost.\n\n"

                            "Step 4: Clearly describe UI layout and function concisely:\n"
                            "- Provide a clear, structured top-down, left-right UI description explicitly mentioning UI components (e.g., header bar, sidebar, footer menu, content area).\n"
                            "- Explicitly note any feedback, status, or error messages clearly visible in UI; leave empty if none.\n\n"

                            "Step 5: Clearly calculate UI-to-screenshot matching accuracy:\n"
                            "- Provide a precise matching accuracy (0.0–1.0). Scores below 0.5 explicitly indicate severe mismatches, recommending re-synchronization.\n\n"

                            "Strictly follow the output schema and clearly adhere to provided examples. "
                            "Do not infer elements or UI functions beyond visible evidence."
                        )
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Below is the UI runtime elements XML:\n",
                    },
                    {
                        "type": "text",
                        #"text": json.dumps(ui_runtime_elements, indent=2, ensure_ascii=False, sort_keys=True),
                        "text": _dump_xml(ui_runtime_elements),
                        # Does some indent makes LLM better to understand the hierarchy?
                    },
                    {
                        "type": "text",
                        "text": "\n\nBelow is the screenshot image for reference:\n",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_screenshot}"
                        },
                    },
                    {
                        "type": "text",
                        "text": (
                            "\n\n"
                            "" if not focus_ads else "Advertisements seem to exist in the page. Identify any topmost alerts as advertisements if they do have a close button.\n\n"
                            "Now start matching elements by their XML ID and bounding box within the screenshot to infer meaning."
                        ),
                    }
                ],
            },
        ]

        # Generate the response schema from the Pydantic model
        response_format = UIView

        # Calculate token count for request
        from tiktoken import encoding_for_model
        enc = encoding_for_model("gpt-4o")
        
        # Count tokens in text portions
        #text_tokens = text_tokens = sum(len(enc.encode(content["text"])) for msg in messages for content in msg["content"] if "text" in content)
        
        # Add tokens for image (estimated as per OpenAI docs)
        #image_tokens = 85 + 170 * 4  # 85 tokens for image metadata + 4 tiles
        #total_tokens = text_tokens + image_tokens
        
        #logger.debug(f"Estimated request tokens: {total_tokens}")
        #logger.debug(f"Input prompt: %s", '\n'.join([content["text"] for msg in messages for content in msg["content"] if "text" in content]))

        # Call the OpenAI API
        try:
            completion = self.client.beta.chat.completions.parse(
                model="gpt-4o-2024-11-20", 
                messages=messages,
                temperature=0.1,
                n=1,
                response_format=response_format,
            )

            # Extract the response content
            result = completion.choices[0].message.parsed
            
            # Get token count from completion response
            logger.debug(f"Analyze_UI: token consumed: {completion.usage.prompt_tokens} + {completion.usage.completion_tokens} = {completion.usage.total_tokens}")
            if completion.usage.prompt_tokens >= 16000:
                logger.warning(f"Analyze_UI: Used {completion.usage.prompt_tokens} tokens for prompt >= 16k")
            
            token_record('analyze_ui', completion.usage.prompt_tokens, completion.usage.completion_tokens)

            # return result
        except Exception as e:
            logger.error(f"Error during OpenAI API call: {e}")
            raise e

        messages_semantic = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Your task is to analyze provided runtime UI XML data and corresponding screenshot, carefully matching and classifying UI elements to clearly generate a structured UIView object.\n\n"

                            "Detailed Steps and Exact Restrictions:\n\n"

                            "Step 1: Carefully match XML element IDs and bounding boxes precisely with visible screenshot elements:\n"
                            "- Discard any XML elements that clearly do not match or are not fully visible in the screenshot.\n"
                            "- Prioritize essential functional elements clearly needed for interaction (close buttons, search icons, navigation buttons, input fields).\n"
                            "- Do not use IDs that are not present in the XML elements to describe items in the screenshot. Even if some buttons or text truly exist, they may not be recorded as XML elements." # AdsDP added: GPT provides some element ids that are not in UI Tree.
                            "- Ignore native OS-level status/navigation bars or keyboards.\n\n"
                            
                            "Step 2: Carefully identify the type of the interface that currently occupies most of the screen space and can be considered the background of current screen.\n"
                            '''
                            Interface out of the test app:
                                - 'Home Screen': The interface user returns to after leaving the app. Characterized by a grid of app icons, a dock at the bottom, and a wallpaper background.
                                - 'Control Center': An overlay (usually swiped from the top-right) with system control toggles (e.g., Wi-Fi, Brightness) over a blurred background.
                                - 'Other interface out of the test app': Interfaces not rendered or displayed within the currently tested app’s process, such as iOS system interfaces (e.g., Game Center, Notification Center) or other app interfaces (e.g., browser tabs).
                            Interface in the test app but not in a full-screen ad:
                                - 'Purchase Interface': A screen, pop-up, or dedicated section within the app specifically designed to solicit a payment from the user for a service, subscription, or premium feature. This interface should include: explicit pricing (e.g., "$4.99/month", "¥50/year", "Unlock for $9.99"), list of benefits/features: (e.g., "Ad-free experience," "Access to all filters," "Unlimited cloud storage"), which is often presented as a list with checkmarks (✓), and a clear Call-to-Action (CTA) (e.g., "Subscribe Now," "Upgrade to Pro," "Start Free Trial," or "Continue.")
                                - 'App Functional Interface': Functional interfaces rendered and displayed while the app is running, which may include non-fullscreen ads (or ad banners).
                            Interface in a full-screen ad:
                                - 'Full-Screen Ad Interface': the ad interface which occupies the entire screen or the majority.
                                - 'Landing Page': A specific, interactive full-screen destination a user arrives from an in-app ad. The screen must show a Third-Party UI Frame which cover the majority of the screen. The page is displayed inside a context, such as:
                                    - A web browser (e.g., Chrome, Safari), indicated by a visible URL bar, tabs, or navigation buttons.
                                    - An app store (e.g., Apple App Store, Google Play Store), indicated by a blue close button "Done", "Get"/"Install" buttons in their standard positions, and other standard layouts.
                            '''

                            "Step 3: Carefully search for UI elements with the following semantics:\n" 
                            '''
                            - 'Ad Removal Element': A UI component (like a button, banner, or menu item) that initiates the process of getting an ad-free version of the app, usually through a payment. Look for the following text or icon as evidence:
                                Text: 'Remove Ads', 'Go Ad-Free', 'Ad-Free', 'No Ads', 'Disable Ads', et al.
                                Icon: Shield (🛡️): Symbol for 'blocking' or 'protection', or prohibition sign with a slash with text 'Ad(s)' in it.
                            - 'Close Button': Any clickable UI element that visually suggests to the user that they can dismiss, skip, or otherwise exit the current interface. Look for the following text or icon:
                                An 'X' icon (e.g., X, ✕, ⊗).
                                Buttons that use words to indicate the action. For example, buttons with text labels like 'Close,' 'Skip,' 'Skip Ad,' 'Done,' or 'Skip Video.'
                                Progression or Skip Icons: Icons that metaphorically suggest moving past the ad. For example, a 'fast-forward' icon (>> or ►►), which implies speeding through the ad. A 'next' icon (>), which implies moving on to the next content and always combines with text 'Continue to the app.'
                            - 'Purchase Button': A button that initiates a purchase when pressed.
                            - 'Reward Element': any visual component (e.g., text, icon, button) that explicitly offers, grants, or confirms a reward to the user. It can be buttons/prompts informing users to watch an ad for a benefit (e.g., "Watch Ad to Double Coins," "Get a free life").
                                Exclusions: 
                                - Standard Navigation: Buttons like "Continue," "Next Level," or "Retry" that do not offer any explicit bonus. 
                                - Standard Score Screens: Game-over or level-complete screens that only show performance metrics (score, time) without highlighting any bonus items or currency earned.
                            - 'Shake Element': any visual component in a user interface designed to instruct, prompt, or suggest that the user should physically shake their mobile device to trigger an action. It can be text ("Shake", "Shake to...", "摇一摇", "摇动", etc.) or icon (an UI element (e.g., a gift box, red envelope, button, card) that uses a shaking or wobbling animation).
                            - 'Intermediate Currency': In-game currency that does not exist in the real world but can sometimes be exchanged for real money.
                            - "Delete Account Element": Any button that initiates the account deletion process upon being clicked, or text that informs the user of the method to delete their account.
                            - 'Voluntary Ad Trigger Element': an interactive UI element (e.g., a button, icon, or text link) that, when clicked by the user, initiates the play of an ad. Looking for the following text or icon as evidence:
                                - Explicit Text, such as: "Watch Ad", "Watch Video", "Unlock by watching a video"
                                - Suggestive Iconography, including: A play symbol (▶️), A video camera icon (📹), and coin stack (🪙) or gem (💎), often with a small play symbol overlayed on it.\n
                            - 'Non-Fullscreen Ad': an ad that doesn't occupy the entire or the majority of screen.
                            - 'Asymmetric Element Pairs': A pair of elements are positioned next to each other in the UI, with a clear semantic symmetry: one benefits the user, while the other benefits the developer (e.g., “close ad” vs. “open landing page”). However, the visual design clearly emphasizes the developer-benefiting element (e.g., highlighted color or bold font). 
                                - You should treat the pair as a single UI element when outputting. If the UI Tree does not contain a container that includes both elements, create a new element and figure out the minimal bounding box that encloses the pair.
                            - 'Disguised Element': UI elements visually imitating iOS system interfaces such as notifications or alerts, but actually designed by the app or ad developer. The purpose is usually to trick the user into clicking the element.
                            - 'Email or Call to Cancel Element': A UI element that informs the user that deleting a registered account requires sending an email, a text message, or making a phone call.
                            - 'Other Functional Element': other element that belongs to current interface and can be click to produce valuable result for continuing the task.
                            '''

                            "Step 4: Explicitly identify and describe alerts:\n"
                            "- Clearly detect any fully visible, topmost urgent alerts requiring immediate action.\n"
                            "- If an alert is partially visible or non-urgent (small corner overlay), clearly mention it but mark as non-topmost.\n\n"

                            "Step 5: Clearly describe UI layout and function concisely:\n"
                            "- Provide a clear, structured top-down, left-right UI description explicitly mentioning UI components (e.g., header bar, sidebar, footer menu, content area).\n"
                            "- Explicitly note any feedback, status, or error messages clearly visible in UI; leave empty if none.\n\n"

                            "Strictly follow the output schema and clearly adhere to provided examples. "
                            "Do not infer elements or UI functions beyond visible evidence."
                        )
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Below is the UI runtime elements XML:\n",
                    },
                    {
                        "type": "text",
                        # "text": json.dumps(ui_runtime_elements, indent=2, ensure_ascii=False, sort_keys=True),
                        "text": _dump_xml(ui_runtime_elements),
                        # Does some indent makes LLM better to understand the hierarchy?
                    },
                    {
                        "type": "text",
                        "text": "\n\nBelow is the screenshot image for reference:\n",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_screenshot}"
                        },
                    },
                    {
                        "type": "text",
                        "text": (
                            "\n\n"
                            "" if not focus_ads else "Advertisements seem to exist in the page.\n\n"
                            "Now start matching elements by their XML ID and bounding box within the screenshot to infer meaning."
                        ),
                    }
                ],
            },
        ]

        # Generate the response schema from the Pydantic model
        response_format_semantic = SemanticView

        # Call the OpenAI API
        try:
            completion = self.client.beta.chat.completions.parse(
                model="gpt-4o-2024-11-20",
                messages=messages_semantic,
                temperature=0.1,
                n=1,
                response_format=response_format_semantic,
            )

            # Extract the response content
            semantic_analyze_result = completion.choices[0].message.parsed

            # Get token count from completion response
            logger.debug(
                f"Analyze_UI: token consumed: {completion.usage.prompt_tokens} + {completion.usage.completion_tokens} = {completion.usage.total_tokens}")
            if completion.usage.prompt_tokens >= 16000:
                logger.warning(f"Analyze_UI: Used {completion.usage.prompt_tokens} tokens for prompt >= 16k")

            token_record('semantic_analyze_ui', completion.usage.prompt_tokens, completion.usage.completion_tokens)

            return result, semantic_analyze_result
        except Exception as e:
            logger.error(f"Error during OpenAI API call: {e}")
            raise e
    
    @DeprecationWarning
    def ocr(self, base64_image: str):
        """
        Extracts text from an image using OCR (Optical Character Recognition).

        Args:
            base64_image (str): The image encoded in base64.

        Returns:
            str: The extracted text from the image.
        """
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "You are an advanced OCR (Optical Character Recognition) and UI analysis assistant. "
                            "Your task is to analyze App UI screenshots, identify key UI elements, and provide detailed information about each detected component. "
                            "The screenshots are divided into a visible grid system, where each grid cell is labeled with a unique index. "
                            "For each detected UI element: "
                            "1. Specify whether it is text, an icon, or a clickable element. "
                            "2. Provide the Minimum Bounding Rectangle (MBR) indices, defined by the top-left and bottom-right grid indices, to encapsulate the element within the minimal number of grid cells. "
                            "Ensure that the MBR is strictly minimal and contains only the detected element, without expanding it to include surrounding areas. "
                            "3. Include any detected text associated with the element. "
                            "4. Provide an assumed description of the element's function or role in the UI. "
                            "5. Assess the likelihood of the element being clickable and include a confidence score for this assessment. "
                            
                        )
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        },
                    },
                ],
            }
        ]

        class BoundingBox(BaseModel):
            left_top_idx: int = Field(..., description="The index of the top-left grid.")
            right_bottom_idx: int = Field(..., description="The index of the bottom-right grid.")

        class UIElement(BaseModel):
            ui_type: Literal['Button', 'TextButton', 'IconButton', 'InputText', 'Checkbox', 'Radio', 'Dropdown', 'TextOnly', 'IconOnly']
            description: str = Field(..., description="The description (assumed effect in context) of the UI element.")
            text: str = Field(..., description="Current text content of the UI element.")
            visibility: bool = Field(..., description="The actual visibility of the UI element in the screenshot.")
            confidence: float = Field(..., description="The assumed confidence of clickability of the UI element. (0.0 - 1.0)")
            bounding_box: BoundingBox = Field(..., description="The bounding box indices of the UI element.")

        class UIView(BaseModel):
            elements: List[UIElement] = Field(..., description="List of all (especially those significant and clickable) UI elements.")

        response_format = UIView

        try:
            completion = self.client.beta.chat.completions.parse(
                model="gpt-4o",
                messages=messages,
                temperature=0,
                n=1,
                response_format=response_format,
            )

            result = completion.choices[0].message.parsed

            return result
        except Exception as e:
            logger.error(f"Error during OpenAI API call: {e}")
            raise e
    

    @DeprecationWarning
    def fix_ocr(self, base64_img: str, ocr_result: dict):
        """
        Extract possibly significant or clickable elements from OCR results with absolute coordinates.

        Args:
            base64_img (str): The image encoded in base64.
            ocr_result (JSON dict): The OCR results to be fixed.

        Returns:
            dict: A dictionary containing significant and clickable UI elements with descriptions,
                  visibility, and confidence of clickability.
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Analyze the attached OCR results and screenshot to identify significant and interactable UI elements. "
                            "The OCR results contain extracted text and bounding box coordinates for each text element. "
                            "For each text element, infer and classify it as a UI element by assuming its type (e.g., Button, TextButton, IconButton, etc.) based on its label, context, or icon proximity."
                            "(Preferably assume type as Button or another clickable type.)"
                            "Include attributes such as ID, assumed type, assumed description, current text content, and confidence of clickability. "
                            "Additionally, provide a high-level description of the overall UI view and any alert or feedback messages visible in the screenshot. "
                            "Focus only on elements corresponding to the indices (IDs) in the OCR JSON and those fully or partly visible in the screenshot. "
                            "Ignore elements that are not listed in the OCR JSON or are entirely out of view."
                        )
                    },
                    {
                        "type": "text",
                        "text": "Below are the OCR results:",
                    },
                    {
                        "type": "text",
                        "text": json.dumps(ocr_result, indent=0),
                    },
                    {
                        "type": "text",
                        "text": "Below is the screenshot image for reference:",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_img}"
                        },
                    },
                ],
            }
        ]
        
        response_format = UIView

        try:
            completion  = self.client.beta.chat.completions.parse(
                model="gpt-4o",
                messages=messages,
                temperature=0.1,
                n=1,
                response_format=response_format,
            )

            result = completion.choices[0].message.parsed

            return result
        except Exception as e:
            logger.error(f"Error during OpenAI API call: {e}")
            raise e



from google import genai
from google.genai import types
import threading
import random
import time
from resources_for_run.keys import GeminiKey, UniapiPaidKey, Uniapi

MODEL_LIST = [
    "gemini-2.5-flash",
]

# MODEL_LIST = [
#     "gemini-3-flash-preview",
# ]

class EmptyResponseTextError(Exception):
    """Response.text is None or empty"""
    pass

class GeminiClient:
    if Uniapi:
        API_KEYS = {
            key: {'client': None, # genai.Client(api_key=key, http_options={'timeout': 600000}),
                  'available_model': {model: True for model in MODEL_LIST},
                  'status': True,
                  'timer': None,
                  'lock': threading.Lock()}
            for key in UniapiPaidKey
        }
    else:
        API_KEYS = {
            key: {'client': None,  # genai.Client(api_key=key, http_options={'timeout': 600000}),
                  'available_model': {model: True for model in MODEL_LIST},
                  'status': True,
                  'timer': None,
                  'lock': threading.Lock()}
            for key in GeminiKey
        }
    STATUS_LOCK = threading.Lock()

    def __init__(self, api_key=None):
        with self.STATUS_LOCK:
            if api_key:
                self.current_api_key = api_key
            else:
                self.current_api_key = random.choice([key for key, api in self.API_KEYS.items() if api['status']])

            current_client = self.API_KEYS[self.current_api_key]['client']
            if current_client:
                self.client = current_client
            else:
                if Uniapi:
                    self.client = self.API_KEYS[self.current_api_key]['client'] = genai.Client(
                        http_options=types.HttpOptions(base_url='https://api.uniapi.io/gemini'),
                        api_key=self.current_api_key
                    )
                else:
                    self.client = self.API_KEYS[self.current_api_key]['client'] = genai.Client(
                        api_key=self.current_api_key,
                        http_options={'timeout': 600000}
                    )

    def replace_client(self):
        self.__init__()
        return self

    def get_available_model(self):
        with self.STATUS_LOCK:
            for model in MODEL_LIST:
                if self.API_KEYS[self.current_api_key]["available_model"][model]:
                    return model
            return None

    def feedback(self, model):
        with self.STATUS_LOCK:
            self.API_KEYS[self.current_api_key]["available_model"][model] = False

            if all(model_status is False for model_status in self.API_KEYS[self.current_api_key]["available_model"].values()):
                self.API_KEYS[self.current_api_key]["status"] = False
                logger.info(f"API {self.current_api_key} marked as unavailable.")

    def send_request(self, model, contents, config, retry_sec=20, depth=0):
        if model not in MODEL_LIST:
            model = self.get_available_model()

        if not model:
            self.replace_client()
            return self.send_request(model, contents, config, retry_sec, depth + 1)

        try_counter = 0
        while True:
            try:
                # Send request with function declarations
                response = self.client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=config,
                )
                if response and response.text:
                    break
                else:
                    raise EmptyResponseTextError
            except Exception as e:
                logger.warning(f'Request failed: {self.current_api_key}. Detail: {e}')

                if depth >= 4:
                    raise

                elif hasattr(e, 'code') and e.code == 500: # Gemini Internal Error
                    pass
                    # if try_counter >= 2:
                    #     for part in contents.parts:
                    #         if hasattr(part, 'video_metadata') and part.video_metadata:
                    #             part.video_metadata = None
                elif hasattr(e, 'code') and e.code == 429:
                    try:
                        if any(x.get('violations', None) and x['violations'][0]['quotaId'] == "GenerateRequestsPerDayPerProjectPerModel-FreeTier" for x in e.details['error']['details']):
                            self.feedback(model)
                            model = self.get_available_model()
                            if not model:
                                self.replace_client()
                                return self.send_request(model, contents, config, retry_sec, depth + 1)
                            try_counter = 0
                        elif any(x.get('violations', None) and x['violations'][0]['quotaId'] == "GenerateContentInputTokensPerModelPerMinute-FreeTier" for x in e.details['error']['details']):
                            retry_sec = max(60, retry_sec)
                        else:
                            self.replace_client()
                            return self.send_request(model, contents, config, retry_sec, depth + 1)
                    except Exception as e_again:
                        self.replace_client()
                        return self.send_request(model, contents, config, retry_sec, depth + 1)

                elif hasattr(e, 'code') and e.code == 403:
                    self.feedback(model)
                    while model := self.get_available_model():
                        self.feedback(model)
                    logger.warning(f'Suspended Key: {self.current_api_key}')
                    self.replace_client()
                    return self.send_request(model, contents, config, retry_sec, depth + 1)

                try_counter += 1
                if try_counter >= 3:
                    self.replace_client()
                    return self.send_request(model, contents, config, retry_sec, depth + 1)
                time.sleep(retry_sec)

        return response

    def count_token(self, model, contents, retry_sec=5):
        if model not in MODEL_LIST:
            model = self.get_available_model()

        if not model:
            self.replace_client()
            return self.count_token(model, contents, retry_sec)

        try_counter = 0
        while True:
            try:
                # Send request with function declarations
                response = self.client.models.count_tokens(
                    model=model,
                    contents=contents,
                )
                if response:
                    break
            except Exception as e:
                logger.warning(f'Count tokens failed: {self.current_api_key}. Detail: {e}')

                try_counter += 1
                if try_counter >= 3:
                    return None
                time.sleep(retry_sec)

        return response

    @time_consumed
    def detect_dp(self, ui_sequence: list):
        messages_detect = {
            "role": "user",
            "parts": [
                {
                    "text": "".join((
                        "Your task is to carefully analyze the following screenshots and sequence of actions one by one, and—based on the definitions—determine whether any dark patterns appear.\n\n"
                        # "Note that only list dark patterns that appear in the final screenshot (static) or end with the final screenshot (dynamic).\n\n"

                        "Detailed Steps and Exact Restrictions:\n\n"

                        "Step 1: Carefully understand the definition of each dark pattern below. Each dark pattern includes a description, along with its main UI cues and combinations.\n"
                        '''
                        - App Resumption Ads: Users may temporarily exit the app and visit an external interface (i.e., accessing the iPhone’s Control Center, swiping up to return to the Home Screen, et al.) Upon returning to the app, they may be forced to view an ad covering the majority of the screen.
                            ** Warning: Carefully distinguish whether the ad was present before leaving the app or appeared after returning; only label it as an App Resumption Ad in the latter case.**
                            ** Moreover, the ad landing page should be treated as an in-app page; therefore, observing and then closing it does not trigger an App Resumption Ad.**
                            - Cues & Combinations: "UI1: Home Screen / Control Center -> UI2: Full-Screen Ad in App Interface"
                        - Unexpected Full-Screen Ads: A full-screen ad is displayed to the user in a way that subverts their expectations and without their clear, informed consent. **Note that** only judge this dark pattern when the evidence explicitly states the user interaction before the ad appeared.
                            ** Warning: Similar as App Resumption Ads, only label this dark pattern if you are certain the ad appears after entering the in-app interface rather than before. Specifically, if the ad page appears after the ad landing page (e.g., the App Store interface), you should label it as Ad Closure Failure but not Unexpected Full-Screen Ads because landing page also belongs to the ad itself.**
                            - Cues & Combinations 1: "UI1: App Interface without Voluntary Ad Trigger Element -> Action1: User clicks functional button in App Interface -> UI2: Full-Screen Ads in App Interface"
                            - Cues & Combinations 2: "UI1: Any content in App Interface -> Action1: No action -> UI2: Full-Screen Ads in App Interface"
                        - Paid Ad Removal: any feature or option within the app that allows the user to pay money in exchange for removing ads. **Note that only label this when there is explicit evidence of a payment requirement (e.g., a price tag). **
                            - Cues & Combinations: "UI: Ad Removal Element in Purchase Interface"
                        - Reward-Based Ads: Any user interface mechanism that offers an in-app reward (e.g., game items, currency, feature access) to entice or persuade a user into watching an ad. 
                            ** Warning: Only label this when there is explicit evidence that clicking an element will trigger an ad. Do not ASSUME or IMAGINE that a reward is at the cost of triggering an ad.**
                            - Cues & Combinations: "UI: Voluntary Ad Trigger Element and Reward Element in App Interface"
                        - Auto-Redirect Ads: An ad automatically navigates the user to a landing page (e.g., a website or an app store page, excluding payment interface) without a clear, direct, and intentional action from the user to trigger that navigation. **Note that** only consider it an automatic redirect if it occurs without any user interaction with the ad; if the user clicks anywhere (e.g., a close button), the landing page cannot be confirmed as automatic.
                            - Cues & Combinations: "UI1: Full-Screen Ad Interface -> Action1: No action -> UI2: Landing Page Interface"
                        - Ad without Exit Option: An ad does not provide a close button or delay its appearance to force users to watch the ad. **Carefully analyze each ad and its surrounding visual design to determine the number and placement of close buttons.**
                            ** Warning: You need to carefully analyze all ads one by one, and examine if they have close buttons.**
                            - Cues & Combinations: "UI: Ad (any size) without Skip or Close Button"
                        - Multiple Close Buttons: An ad displays two or more close buttons simultaneously on the current interface.
                            - Cues & Combinations 1: "UI: Two or more Close Buttons in a Full-Screen Ad Interface"
                            - Cues & Combinations 2: "UI: Two or more Close Buttons in a Non-Fullscreen Ad Element"
                        - Ad Closure Failure: The user's attempt to close an ad by clicking a designated close button does not result in the ad being dismissed as expected. **Note that** ad's feedback interface (titled with "Ads served by Google") should not be considered as an independent ad and doesn't form this dark pattern.
                            - Cues & Combinations: "UI1: Close Button in Ad (any size) -> Action1: Click the Close Button -> UI2: Ad (any size) in the same place / Landing Page Interface"
                        - Disguised Ads: The ad is designed to mimic the iOS system interface, such as system pop-ups or notifications, sometimes presenting two options like “Accept” or “Decline.” DO NOT label this dark pattern if the interface imitates designs from social media, apps, or websites rather than the iOS system.
                            ** Note that some widget may use UI design similar as those in App Store, this should not be identified as Disguised ads because these ads are made with App Store SDK.**
                            - Cues & Combinations: "UI: Disguised Element in Ad Interface"
                        - Bias-Driven Ads: The ad uses asymmetric UI design to emphasize developer-preferred options through font, size, or color. 
                            ** Note that 1. only label this when the design appears within the ad itself, not outside it.
                            2. To qualify as a bias-driven ad pair, the two elements must be positioned close to each other and share a little visual similarity (e.g., similar size or aligned side by side). Do not label combinations such as a small close button with a large INSTALL button, or a top-right close button with a bottom banner, as this dark pattern.**
                            - Cues & Combinations: "UI: Asymmetric Element Pair in Ad Interface \n"
                        '''

                        "Step 2: Carefully analyze the following UI screenshots and user action sequences which are shown in chronological order to determine whether the UI and actions match the cues or combinations of each dark pattern, or satisfy its textual definition.\n"
                        "** Warning: When analyzing multiple UIs, first determine how many ads appear across the pages. Identify consecutive ads across adjacent pages as the same ad based on the main content, and treat non-full-screen ad components within a full-screen ad (e.g., a lower banner introducing the product) as part of the same ad rather than separate ads (so that they share same close buttons). \n"
                        "Carefully determine the timing of ad appearances, as it directly affects labeling dynamic dark patterns such as Unexpected Full-Screen Ads and App Resumption Ads. \n\n**"

                    ))
                },
                {
                    "text": "\n\nBelow is the screenshot and user actions for reference:\n",
                },
                *[
                    item
                    for h in ui_sequence
                    for item in (
                        [
                            {
                                "inlineData": {
                                    "mimeType": "image/jpeg",
                                    "data": h["image"],
                                },
                            },
                            {
                                "text": "The following are your previous inferences and UI elements you identified in this screenshot, note that some elements with ids may not be mentioned in the following analyzing:\n"
                                        # + json.dumps(h["semantic_analyze"].model_dump(), indent=0, ensure_ascii=False) if "semantic_analyze" in h else ""
                                        + h["analyze"]
                            }
                        ] if h["type"] == "ui" else [
                            {
                                "text": "Before the next screenshot appeared, the user performed the following actions:" +
                                    h["description"]
                            }
                        ]
                    )
                ],
                {
                    "text": "To reiterate, your task is to carefully analyze whether dark patterns appear in these screenshots. You must identify all instances of each DP type in these screenshots and provide the reasoning for your judgments.\n"
                }
            ],
        }

        # Generate the response schema from the Pydantic model
        response_format_detect = DPList

        config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(
                include_thoughts=True,
                thinking_budget=10000,  # max thinking
            ),
            response_mime_type="application/json",
            response_schema=response_format_detect,
            top_p=0.01,
            top_k=1,
            temperature=0.0,
        )

        # Call the OpenAI API
        try:
            completion = self.send_request(
                model="auto",
                contents=messages_detect,
                config=config,
            )

            # Extract the response content
            dp_detect_result = completion.parsed

            return dp_detect_result
        except Exception as e:
            raise e

    @staticmethod
    def _get_slim_ui(ui: UIView) -> str:
        """
        Convert UIView to a "slim" text so GPT can focus on essential data
        """
        slim_ui = []
        for element in ui.elements:
            # Decide category
            if element.ui_type == UIElementType.INPUT_TEXT:
                category = "InputText"
            elif element.ui_type == UIElementType.ICON_BUTTON:
                category = "IconButton"
            elif element.ui_type == UIElementType.CHECKBOX:
                category = "Checkbox"
            elif element.ui_type == UIElementType.RADIO:
                category = "Radio"
            elif element.ui_type == UIElementType.DROPDOWN:
                category = "Dropdown"
            elif element.ui_type in [
                UIElementType.BUTTON,
                UIElementType.TEXT_BUTTON,
                UIElementType.OTHER_INTERACTABLE
            ] or element.clickability > 0.5:
                category = "Button"
            elif element.ui_type == UIElementType.TEXT_ONLY:
                category = "Text"
            elif element.ui_type == UIElementType.ICON_ONLY:
                category = "Image"
            else:
                # Fallback logic
                category = "Text" if element.text.strip() else "Image"

            slim_ui.append(
                {'id': element.id, 'category': category, 'description': element.description, 'text': element.text,
                 'location': element.location})

        slim_ui_text = json.dumps(slim_ui, indent=2, ensure_ascii=False)
        return slim_ui_text

    @staticmethod
    def _concat_slim_ui(slim_ui_text, semantic_ui: SemanticView) -> str:
        slim_ui_json = json.loads(slim_ui_text)

        for element in semantic_ui.semantic_element:
            if item := next((x for x in slim_ui_json if x["id"] == element.id), None):
                item["description"] += '''Additionally, you considered it to be semantically related to dark patterns. Refer to the "type" and "evidence" fields.'''
                item["type"] = element.type
                item["evidence"] = element.evidence
            elif element.id > 0:
                # Decide category
                if element.ui_type == UIElementType.INPUT_TEXT:
                    category = "InputText"
                elif element.ui_type == UIElementType.ICON_BUTTON:
                    category = "IconButton"
                elif element.ui_type == UIElementType.CHECKBOX:
                    category = "Checkbox"
                elif element.ui_type == UIElementType.RADIO:
                    category = "Radio"
                elif element.ui_type == UIElementType.DROPDOWN:
                    category = "Dropdown"
                elif element.ui_type in [
                    UIElementType.BUTTON,
                    UIElementType.TEXT_BUTTON,
                    UIElementType.OTHER_INTERACTABLE
                ] or element.clickability > 0.5:
                    category = "Button"
                elif element.ui_type == UIElementType.TEXT_ONLY:
                    category = "Text"
                elif element.ui_type == UIElementType.ICON_ONLY:
                    category = "Image"
                else:
                    # Fallback logic
                    category = "Text" if element.text.strip() else "Image"

                slim_ui_json.append({
                    'id': element.id, 'category': category,
                    'description': element.description + "Additionally, you considered it to be semantically related to dark patterns. Refer to the 'type' and 'evidence' fields.",
                    'text': element.text, 'location': element.location, 'type': element.type, 'evidence': element.evidence
                })

        slim_ui_text_append = json.dumps(slim_ui_json, indent=2, ensure_ascii=False)
        return slim_ui_text_append

    @staticmethod
    def _get_action_history(action_history: List[str]) -> str:
        """
        Convert action history to a text
        """
        action_history_text = "[*] 1. Launch the application from SpringBoard\n" + "\n".join(
            [("[*] " if any(s in action.lower() for s in ['. restart ', 'restart the app']) else "") + f"{i}. {action}"
             for i, action in enumerate(action_history, start=2)])
        return action_history_text

    @staticmethod
    def _get_slim_task(task_candidate: List[TaskCandidate]) -> str:
        slim_task_json = []
        for task in task_candidate:
            slim_task_json.append(task.model_dump(mode="json"))
        slim_task_text = json.dumps(slim_task_json, indent=2, ensure_ascii=False)
        return slim_task_text

    @time_consumed
    def analyze_action(self, analyzed_ui: UIView, task: str, action_history: List[str], recent_repeat_action: str = '',
                       recent_repeat_count: int = 0) -> UserAction:
        """
        Analyzes the current UI (analyzed_ui), the user's 'task', and 'action_history',
        then instructs GPT to propose a valid UserAction. This method is constructed
        similarly to analyze_screenshot(), making an API call and parsing GPT output
        as a UserAction according to a given schema.
        """

        slim_ui_text = self._get_slim_ui(analyzed_ui)
        action_history_text = self._get_action_history(action_history)

        # Build system and user messages
        messages = {
            "role": "user",
            "parts": [
                {
                    "text": "".join((
                        "Your task is to propose clear and actionable user interactions based on the current UI state, historical actions, and the main task provided.\n\n"

                        "Detailed Steps and Restrictions:\n\n"

                        "Step 1: Fully assess the current UI:\n"
                        "- Clearly summarize the page’s primary function and interaction points (buttons, input fields, tabs).\n"
                        "- Handle urgent UI states first (alerts, pop-ups, advertisements). Ads or alerts must be closed/skipped immediately if possible, unless they are mentioned in the main task.\n"
                        "- If topmost UI elements are non-dismissible, suggest app restart.\n\n"

                        "Step 2: Evaluate the action history:\n"
                        "- Clearly summarize previously visited pages and performed actions.\n"
                        "- Explicitly indicate repeated action sequences (more than twice), and clearly state alternatives if repetition yields no results.\n\n"

                        "Step 3: Identify a precise sub-task:\n"
                        "- Clearly define a concise, actionable sub-task advancing toward the main goal.\n"
                        "- Explicitly verify and clearly state completeness or incompleteness of all required conditions (e.g., all inputs completed, required navigational steps previously done, prerequisite UI interactions).\n"
                        "- Provide explicit reasoning why this sub-task is critical based on prior interactions and current UI context, and necessary for the logical flow toward task completion.\n\n"

                        "Step 4: Propose detailed next actions:\n"
                        "- Actions must be sequential, logically coherent, fully justified, and explicitly validated through the completeness check (including UI element IDs).\n"
                        "- Directly issue 'Input' actions instead of clicks for text entry fields to minimize interactions.\n"
                        "- If no immediate actions are feasible, propose a 'None' action clearly stating the reason (e.g., loading, waiting for UI update).\n"
                        "- Mark task completion explicitly when previous actions clearly indicate the goal is achieved.\n\n"

                        "Step 5: Summarize clearly:\n"
                        "- Provide concise UI page summary clearly describing current context and interaction purpose.\n"
                        "- Provide a precise action summary clearly stating planned actions and rationale, including exact input texts if applicable.\n\n"

                        "Strictly follow provided output schema and examples. Ensure actions are practical, relevant, and clearly explained."
                    ))
                },
                {
                    "text": "The user's task to accomplish: \n",
                },
                {
                    "text": task + "\n\n",
                },
                {
                    "text": "".join((
                        "Current UI summary:\n" + analyzed_ui.description + "\n\n"
                        "The UI has a visible feedback/status/error message: " + analyzed_ui.feedback_message if analyzed_ui.feedback_message else "" + "\n\n"
                        "The UI has an alert or pop-up, which should be handled before doing anything else.\n\n" if analyzed_ui.is_alert_topmost else ""
                        "The UI has an urgent topmost overlaying advertisement, which must be handled before doing anything else.\n\n" if analyzed_ui.ads and analyzed_ui.ads.is_ad_topmost and analyzed_ui.ads.is_ad_closeable and analyzed_ui.ads.ad_close_button_id > 0 else ""
                        # If not closeable, the Bot is not able to handle it.
                        "\n"
                    ))
                },
                {
                    "text": "Current UI view:\n"
                },
                {
                    "text": slim_ui_text + "\n\n"
                },
                {
                    "text": "".join((
                        "Previous action history:\n" + action_history_text + "\n\n" + ((
                        "The recent action series '" + recent_repeat_action + "' has been repeated " + str(
                        recent_repeat_count) + " times.\n"
                        "Do not repeat this action series again. Actively seek and propose alternatives.\n\n"
                        ) if recent_repeat_count >= 3 else "")
                    ))
                },
                {
                    "text": "The user's task (again) is to:\n",
                },
                {
                    "text": task + "\n\n",
                },
                {
                    "text": "Now start proposing the next user action(s) to move toward completing the task.",
                },
            ],
        }

        # We want GPT to parse the final response as a UserAction
        response_format = UserAction

        config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(
                include_thoughts=True,
                thinking_budget=24576,  # max thinking
            ),
            response_mime_type="application/json",
            response_schema=response_format,
            top_p=0.01,
            top_k=1,
            temperature=0.0,
        )

        try:
            completion = self.send_request(
                model="auto",
                contents=messages,
                config=config,
            )

            # Parsed result is a valid UserAction object
            result = completion.parsed

            logger.debug(
                f"Analyze_Action: token consumed: "
                f"{completion.usage_metadata.prompt_token_count} + {completion.usage_metadata.candidates_token_count}"
                f"= {completion.usage_metadata.total_token_count}"
            )

            token_record('analyze_action', completion.usage_metadata.prompt_token_count, completion.usage_metadata.candidates_token_count)

            return result

        except Exception as e:
            logger.error(f"Error during Gemini API call for analyze_action: {e}")
            # Return a fallback "none" action if GPT fails
            raise e

    @time_consumed
    def semantic_analyze_action(self, analyzed_ui: UIView, semantic_analyzed_ui: SemanticView, task: str,
                                action_history: List[str], ui_graph_history: List[str],
                                current_task: TaskCandidate, task_candidate: List[TaskCandidate],
                                recent_repeat_action: str = '',
                                recent_repeat_count: int = 0,
                                rag: list = None) -> SemanticUserAction:
        """
        Analyzes the current UI (analyzed_ui), the user's 'task', and 'action_history',
        then instructs GPT to propose a valid UserAction. This method is constructed
        similarly to analyze_screenshot(), making an API call and parsing GPT output
        as a SemanticUserAction according to a given schema.
        """

        slim_ui_text = self._get_slim_ui(analyzed_ui)
        # slim_ui_text = "[]"
        slim_ui_text = self._concat_slim_ui(slim_ui_text, semantic_analyzed_ui)
        action_history_text = self._get_action_history(action_history)
        slim_task_text = self._get_slim_task(task_candidate)

        # Build system and user messages
        messages = {
            "role": "user",
            "parts": [
                {
                    "text": "".join((
                        "Your task is to propose clear and actionable user interactions based on the current UI state, historical actions, and the main task provided.\n\n"

                        "Detailed Steps and Restrictions:\n\n"

                        "Step 1: Fully assess the current UI:\n"
                        "- Clearly summarize the page’s primary function and interaction points (buttons, input fields, tabs).\n"
                        "- Handle urgent UI states first (alerts, pop-ups). Alerts must be closed immediately if possible, unless they are mentioned in the main task.\n"
                        # "- If topmost UI elements are non-dismissible, suggest app restart.\n\n"

                        "Step 2: Evaluate the action history:\n"
                        "- Clearly summarize previously visited pages and performed actions.\n"
                        "- Explicitly indicate repeated action sequences (more than twice), and clearly state alternatives if repetition yields no results.\n\n"
                        
                        '''
                        Step 3: Evaluate the user’s current task progress and its future prospects:
                        - Carefully compare the user’s previously defined exploration subtask with the actual actions, UI change history, and the number of steps already explored in this task to determine whether the subtask is still being effectively executed.
                        - Based on the current UI, determine the status of the user’s subtask (completed, failed, or processing). 
                            - “Completed”: the UI required for the current task has been sufficiently explored. 
                            - “Failed”: the user has spent considerable effort on the task but the likelihood of completion remains low. 
                            - “Processing”: the task has not yet succeeded but has a high probability of completion within the next few steps.
                        - If the current task can be completed within the next few steps, carefully estimate the number of additional UI interactions required.
                        '''

                        "Step 4: Reorder all candidate tasks in descending order of priority:\n"
                        "- Clearly define a concise, actionable, highest-priority sub-task advancing toward the main goal.\n"
                        "- Note that when deciding the prior task and action, you must referencing and imitating actions taken by human auditors on similar interfaces. \n"
                        "- After decide the highest-priority sub-task, rank all tasks in order of priority, with those closer to the top having higher priority, according to the following guidelines:\n"
                            "- Place the task that should currently be executed at the top of the list.\n"
                            "- Emphasize exploration breadth; never repeatedly explore a single app feature or interface such as settings, ad removal, or purchases.\n"
                                "- Specifically, except for general explore tasks, tasks that have already been executed (whether failed or completed) should be assigned lower priority.\n\n"

                        "Step 5: Propose detailed next actions:\n"
                        "- Carefully design the next interaction step: \n"
                            "- Actions must be sequential, logically coherent, fully justified, and explicitly validated through the completeness check (including UI element IDs).\n"
                            "- Prioritize clicking elements that may trigger ads, such as game items with a “+” icon or ad-like play buttons resembling ▶.\n"
                            "- If no immediate actions are feasible, propose an 'Idle' action clearly stating the reason (e.g., loading, waiting for UI update).\n"
                            "- Never perform ID-based actions (e.g., click, hover) on elements that do not including ID in the UI View; only click elements with valid IDs.\n\n"

                        "Step 6: Summarize clearly:\n"
                        "- Provide concise UI page summary clearly describing current context and interaction purpose.\n"
                        "- Provide a precise action summary clearly stating planned actions and rationale, including exact input texts if applicable.\n\n"

                        "Strictly follow provided output schema and examples. Ensure actions are practical, relevant, and clearly explained.\n"
                    ))
                },
                {
                    "text": "The user's main task to accomplish: \n",
                },
                {
                    "text": task + "\n\n",
                },
                {
                    "text": f"The subtask currently set as the highest priority is: {current_task.sub_task.__class__.__name__}, and {current_task.explored_step} steps of UI exploration have been conducted based on this task so far.\n\n"
                },
                {
                    "text": "The definitions and current execution statuses of all candidate tasks. Use this information to reorder their priorities accordingly:\n",
                },
                {
                    "text": slim_task_text + "\n\n",
                },
                {
                    "text": "Current UI view:\n"
                },
                {
                    "text": slim_ui_text + "\n\n"
                },
                {
                    "text": "".join((
                        "Previous action history:\n" + action_history_text + "\n\n" + ((
                        "The recent action series '" + recent_repeat_action + "' has been repeated " + str(recent_repeat_count) + " times.\n"
                        "Do not repeat this action series again. Actively seek and propose alternatives.\n\n"
                        ) if recent_repeat_count >= 3 else "")
                    ))
                },
                # *[
                #     {
                #         "inlineData": {
                #             "mimeType": "image/png",
                #             "data": h.split(",", 1)[1]
                #         }
                #     } if h.startswith("data:image/png;base64,") else
                #     {
                #         "text": h
                #     }
                #     for h in ui_graph_history
                # ],
                # {
                #     "text": "The user's task (again) is to:\n",
                # },
                # {
                #     "text": task + "\n\n",
                # },
                {
                    "text": "Now start proposing the next sub-task and action(s) to move toward completing the task.",
                },
            ]
        }

        if rag:
            messages["parts"].extend([
                {
                    "text": "\n\n"
                            "The following screenshots and texts reflect auditors' decision when exploring similar UI screens. These screenshots below are visually similar to your current screenshot or contains similar UI elements.\n"
                            "For each upcoming screenshot–action pair, you must explicitly analyze the similarities and differences between the current UI and those screenshots. For screenshots that share strong similarities with the current UI, also explain how the auditor’s actions inform your decision—e.g., which element to prioritize when multiple interaction options are available.\n"
                            "Finally, you need to identify which screenshots are most relevant, summarize all kinds of dark-pattern-related elements (e.g., Ad Removal Element) that the auditor interacted with in those screenshots, map them to actionable elements in the current UI, and decide which one to prioritize. Note that if the auditor’s task conflicts with your own priority, follow the auditor’s task unless you have already performed it before. \n"
                            "For your convenience, we summarize all kinds of elements along with their definition here: \n"
                            '''
                                - 'Ad Removal Element': A UI component (like a button, banner, or menu item) that initiates the process of getting an ad-free version of the app, usually through a payment. For example:
                                    Text: 'Remove Ads', 'Go Ad-Free', 'Ad-Free', 'No Ads', 'Disable Ads', et al.
                                    Icon: Shield (🛡️): Symbol for 'blocking' or 'protection', or prohibition sign with a slash with text 'Ad(s)' in it.
                                - 'Close Button': Any clickable UI element that visually suggests to the user that they can dismiss, skip, or otherwise exit the current interface. For example:
                                    An 'X' icon (e.g., X, ✕, ⊗).
                                    Buttons that use words to indicate the action. For example, buttons with text labels like 'Close,' 'Skip,' 'Skip Ad,' 'Done,' or 'Skip Video.'
                                    Progression or Skip Icons: Icons that metaphorically suggest moving past the ad. For example, a 'fast-forward' icon (>> or ►►), which implies speeding through the ad. A 'next' icon (>), which implies moving on to the next content and always combines with text 'Continue to the app.'
                                - 'Purchase Interface': A screen, pop-up, or dedicated section within the app specifically designed to solicit a payment from the user for a service, subscription, or premium feature. This interface should include: explicit pricing (e.g., "$4.99/month", "¥50/year", "Unlock for $9.99"), list of benefits/features: (e.g., "Ad-free experience," "Access to all filters," "Unlimited cloud storage"), which is often presented as a list with checkmarks (✓), and a clear Call-to-Action (CTA) (e.g., "Subscribe Now," "Upgrade to Pro," "Start Free Trial," or "Continue.")
                                - 'Reward Element': any visual component (e.g., text, icon, button) that explicitly offers, grants, or confirms a reward to the user. It can be buttons/prompts informing users to watch an ad for a benefit (e.g., "Watch Ad to Double Coins," "Get a free life").
                                    Exclusions: 
                                        - Standard Navigation: Buttons like "Continue," "Next Level," or "Retry" that do not offer any explicit bonus. 
                                        - Standard Score Screens: Game-over or level-complete screens that only show performance metrics (score, time) without highlighting any bonus items or currency earned.
                                - 'Voluntary Ad Trigger Element': an interactive UI element (e.g., a button, icon, or text link) that, when clicked by the user, initiates the play of an ad. Looking for the following text or icon as evidence:
                                    - Explicit Text, such as: "Watch Ad", "Watch Video", "Unlock by watching a video"
                                    - Suggestive Iconography, including: A play symbol (▶️), A video camera icon (📹), and coin stack (🪙) or gem (💎), often with a small play symbol overlayed on it.\n
                                - 'Normal Element': Other elements do not belong to above categories.
                            '''
                },
                *[
                    item
                    for index, h in enumerate(rag)
                    for item in (
                        [
                            {
                                "text": f"Similar UI-action pair {index}\n"
                            },
                            {
                                "inlineData": {
                                    "mimeType": "image/jpeg",
                                    "data": h["screenshot"],
                                }
                            },
                            {
                                "text": f"Action performed on UI above: {h['action_description']}"
                            }
                        ]
                    )
                ]
            ])

        # We want GPT to parse the final response as a UserAction
        response_format = SemanticUserAction

        config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(
                include_thoughts=True,
                thinking_budget=24576,  # max thinking
            ),
            response_mime_type="application/json",
            response_schema=response_format,
            top_p=0.01,
            top_k=1,
            temperature=0.0,
        )

        try:
            completion = self.send_request(
                model="auto",
                contents=messages,
                config=config,
            )

            # Parsed result is a valid UserAction object
            result = completion.parsed

            logger.debug(
                f"Analyze_Action: token consumed: "
                f"{completion.usage_metadata.prompt_token_count} + {completion.usage_metadata.candidates_token_count}"
                f"= {completion.usage_metadata.total_token_count}"
            )

            token_record('semantic_analyze_action', completion.usage_metadata.prompt_token_count, completion.usage_metadata.candidates_token_count)

            return result

        except Exception as e:
            logger.error(f"Error during Gemini API call for analyze_action: {e}")
            # Return a fallback "none" action if GPT fails
            raise e

    @time_consumed
    def analyze_screenshot(self, base64_screenshot: str, ui_runtime_elements_json: dict, focus_ads: bool = False, retry: int = 0) -> UIView:
        """
        Analyzes a screenshot and extracts significant and clickable UI elements.

        Args:
            base64_screenshot (str): The screenshot encoded in base64.
            ui_runtime_elements_json (dict): JSON dictionary of extracted UI runtime elements.
            focus_ads: If there is an ad.

        Returns:
            UIView: An object containing significant and clickable UI elements with descriptions,
                    visibility, and confidence of clickability.
        """
        # Ensure ui_runtime_elements_json is a dictionary
        if isinstance(ui_runtime_elements_json, str):
            try:
                ui_runtime_elements = json.loads(ui_runtime_elements_json)
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON input: {e}")
                return None
        else:
            ui_runtime_elements = copy.deepcopy(ui_runtime_elements_json)

        # Remove all attributes starting with '_'
        def traverse(view):
            for k, v in view.copy().items():
                if k.startswith('_'):
                    del view[k]
                elif isinstance(v, dict):
                    traverse(v)
                elif isinstance(v, list):
                    for item in v:
                        if isinstance(item, dict):
                            traverse(item)

        traverse(ui_runtime_elements)

        from utils import _dump_xml

        # Prepare the messages for the chat completion
        # Construct the content as a list of content parts

        messages = {
            "role": "user",
            "parts": [
                {
                    "text": "".join((
                        "Your task is to analyze provided runtime UI XML data and corresponding screenshot, carefully matching and classifying UI elements to clearly generate a structured UIView object.\n\n"
    
                        "Detailed Steps and Exact Restrictions:\n\n"
    
                        "Step 1: Carefully match XML element IDs and bounding boxes precisely with visible screenshot elements:\n"
                        "- Discard any XML elements that clearly do not match or are not fully visible in the screenshot.\n"
                        "- Prioritize essential functional elements clearly needed for interaction (close buttons, search icons, navigation buttons, input fields).\n"
                        "- Ignore native OS-level status/navigation bars or keyboards.\n\n"
    
                        "Step 2: Classify UI elements explicitly into two categories:\n"
                        "- 'hint_elements': Clearly non-interactive or semi-interactive hints (labels, static texts, descriptive images).\n"
                        "- 'action_elements': Explicitly interactive elements (buttons, text fields, checkboxes, radios, dropdowns, clickable icons).\n"
                        # added: For all elements without id, its id should be filled as '-1'.
                        # "Remember that element's type (e.g., StaticText, Button) is not always reliable, so if an XML element exists whose text content exactly matches a detected actionable UI element but has an non-actionable type (StaticText), you MUST reuse the XML element id to describe this element.\n"
                        # "For instance, if you think a 'no' static text with id '5' is clickable, just label an actionable element with id '5', but not '-1'.\n\n"
                        "If you identify any sufficiently important element (such as one that is large, attention-grabbing, or interactive) but no corresponding element exists in the JSON UI Tree, you should still describe the element and set its id attribute to -1.\n\n"
    
                        "Step 3: Explicitly identify and describe advertisements and alerts:\n"
                        "- Clearly detect topmost overlay advertisements requiring immediate dismissal; identify their exact close buttons clearly labeled (e.g., 'X', '关闭', '跳过').\n"
                        "- Clearly detect any fully visible, topmost urgent alerts requiring immediate action.\n"
                        "- If an ad or alert is partially visible or non-urgent (small corner overlay), clearly mention it but mark as non-topmost.\n\n"
    
                        "Step 4: Clearly describe UI layout and function concisely:\n"
                        "- Provide a clear, structured top-down, left-right UI description explicitly mentioning UI components (e.g., header bar, sidebar, footer menu, content area).\n"
                        "- Explicitly note any feedback, status, or error messages clearly visible in UI; leave empty if none.\n\n"
                        "- If the current UI interface is related to the user account, you need to determine whether the user is already logged in."
    
                        "Step 5: Clearly calculate UI-to-screenshot matching accuracy:\n"
                        "- Provide a precise matching accuracy (0.0–1.0). Scores below 0.5 explicitly indicate severe mismatches, recommending re-synchronization.\n\n"
    
                        "Strictly follow the output schema and clearly adhere to provided examples. "
                        "Do not infer elements or UI functions beyond visible evidence."
                    )),
                },
                {
                    "text": "Below is the UI runtime elements XML:\n",
                },
                {
                    "text": _dump_xml(ui_runtime_elements),
                },
                {
                    "text": "\n\nBelow is the screenshot image for reference:\n",
                },
                {
                    "inlineData": {
                        "mimeType": "image/png",
                        "data": base64_screenshot,
                    }
                },
                {
                    "text": "".join((
                        "\n\n"
                        "" if not focus_ads else "Advertisements seem to exist in the page. Identify any topmost alerts as advertisements if they do have a close button.\n\n"
                        "Now start matching elements by their XML ID and bounding box within the screenshot to infer meaning."
                    )),
                }
            ],
        }

        # Generate the response schema from the Pydantic model
        response_format = UIView

        config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(
                include_thoughts=False,
                thinking_budget=0,  # max thinking
            ),
            # max_output_tokens=50000,
            http_options={'timeout': 600000},
            response_mime_type="application/json",
            response_schema=response_format,
            top_p=0.01,
            top_k=1,
            temperature=0.0,
        )

        # Call the OpenAI API
        try:
            completion = self.send_request(
                model="auto",
                contents=messages,
                config=config,
            )

            # Extract the response content
            result = completion.parsed
            if not result and retry <= 1:
                logger.warning("Got empty result after parsing, recall 'analyze_screenshot'...")
                return self.analyze_screenshot(base64_screenshot, ui_runtime_elements_json, focus_ads, retry + 1)

            # Get token count from completion response
            logger.debug(
                f"Analyze_UI: token consumed:"
                f"{completion.usage_metadata.prompt_token_count} + {completion.usage_metadata.candidates_token_count}"
                f"= {completion.usage_metadata.total_token_count}"
            )
            if completion.usage_metadata.prompt_token_count >= 16000:
                logger.warning(f"Analyze_UI: Used {completion.usage_metadata.prompt_token_count} tokens for prompt >= 16k")

            token_record('analyze_ui', completion.usage_metadata.prompt_token_count, completion.usage_metadata.candidates_token_count)

            return result
        except Exception as e:
            logger.error(f"Error during Gemini API call: {e}")
            raise e

        # import base64
        # # 如果有 data URI 前缀，去掉
        # if base64_screenshot.startswith("data:image"):
        #     base64_str = base64_screenshot.split(",")[1]
        # else:
        #     base64_str = base64_screenshot
        # image_bytes = base64.b64decode(base64_str)
        #
        # from io import BytesIO
        # from PIL import Image
        # image = Image.open(BytesIO(image_bytes))


    def semantic_analyze_screenshot(self, base64_screenshot: str, ui_runtime_elements_json: dict, focus_ads: bool = False, rag: list = []) -> SemanticView:
        """
        Analyzes a screenshot and extracts dark pattern-related elements.

        Args:
            base64_screenshot (str): The screenshot encoded in base64.
            ui_runtime_elements_json (dict): JSON dictionary of extracted UI runtime elements.
            focus_ads: If there is an ad.

        Returns:
            UIView: An object containing significant and clickable UI elements with descriptions,
                    visibility, and confidence of clickability.
        """
        # Ensure ui_runtime_elements_json is a dictionary
        if isinstance(ui_runtime_elements_json, str):
            try:
                ui_runtime_elements = json.loads(ui_runtime_elements_json)
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON input: {e}")
                return None
        else:
            ui_runtime_elements = copy.deepcopy(ui_runtime_elements_json)

        # Remove all attributes starting with '_'
        def traverse(view):
            for k, v in view.copy().items():
                if k.startswith('_'):
                    del view[k]
                elif isinstance(v, dict):
                    traverse(v)
                elif isinstance(v, list):
                    for item in v:
                        if isinstance(item, dict):
                            traverse(item)

        traverse(ui_runtime_elements)

        from utils import _dump_xml

        messages_semantic = {
            "role": "user",
            "parts": [
                {
                    "text": "".join((
                        "Your task is to analyze provided runtime UI XML data and corresponding screenshot, carefully matching and classifying UI elements to clearly generate a structured UIView object.\n\n"

                        "Detailed Steps and Exact Restrictions:\n\n"

                        "Step 1: Carefully match XML element IDs and bounding boxes precisely with visible screenshot elements:\n"
                        "- Discard any XML elements that clearly do not match or are not fully visible in the screenshot.\n"
                        "- Prioritize essential functional elements clearly needed for interaction (close buttons, search icons, navigation buttons, input fields).\n"
                        "- Ignore native OS-level status/navigation bars or keyboards.\n\n"

                        "Step 2: Carefully identify the type of the interface that currently occupies most of the screen space or at the topmost and can be considered the background of current screen.\n"
                        "When determining the interface type of the current screen, if the interface prevents the user from accessing all or most of the UI elements beneath it (even if those elements remain visible), it should be considered the current screen’s interface type.\n"
                        "For example, when an iOS notification pop-up appears, the user cannot continue interacting with the app and must tap “OK” to proceed. Although the pop-up occupies only a small portion of the screen, it is still regarded as the topmost interface.\n"
                        '''
                        Specifically, firstly pay close attention to **whether the current page is a full-screen ad** by following steps:
                            carefully checking the screenshot four corners for:
                                (1) AdChoices ▶,
                                (2) Ad info icon (“i” in a circle), 
                                (3) ad's countdown timer (a number representing seconds left of this ad), 
                                (4) a speaker icon, which is used for adjusting ad's volume,
                                (5) a video playback progress bar, typically a thin yellow rectangular bar located at the bottom of the image.
                        Then decide the current interface's type by following definition:
                        Interface out of the test app:
                            - 'Home Screen': The interface user returns to after leaving the app. Characterized by a grid of app icons, a dock at the bottom, and a wallpaper background.
                            - 'Control Center': An overlay (usually swiped from the top-right) with system control toggles (e.g., Wi-Fi, Brightness) over a blurred background.
                            - 'Other interface out of the test app': Interfaces not rendered or displayed within the currently tested app’s process, such as iOS system interfaces (e.g., Game Center, Notification Center) or other app interfaces (e.g., browser tabs).
                        Interface in the test app but not in a full-screen ad:
                            - 'Purchase Interface': A screen, pop-up, or dedicated section within the app specifically designed to solicit a payment from the user for a service, subscription, or premium feature. This interface should include: explicit pricing (e.g., "$4.99/month", "¥50/year", "Unlock for $9.99"), list of benefits/features: (e.g., "Ad-free experience," "Access to all filters," "Unlimited cloud storage"), which is often presented as a list with checkmarks (✓), and a clear Call-to-Action (CTA) (e.g., "Subscribe Now," "Upgrade to Pro," "Start Free Trial," or "Continue.")
                            - 'App Functional Interface': Functional interfaces rendered and displayed while the app is running, which may include non-fullscreen ads (or ad banners).
                        Interface in a full-screen ad:
                            - 'Full-Screen Ad Interface': the ad interface which occupies the entire screen or the majority.
                            - 'Landing Page': A specific, interactive full-screen destination a user arrives from an in-app ad. The screen must show a Third-Party UI Frame which cover the majority of the screen. The page is displayed inside a context, such as:
                                - A web browser (e.g., Chrome, Safari), indicated by a visible URL bar, tabs, or navigation buttons.
                                - An app store (e.g., Apple App Store, Google Play Store), indicated by a blue close button "Done", "Get"/"Install" buttons in their standard positions, and other standard layouts.\n\n
                        '''

                        "Step 3: Carefully search for UI elements with the following semantics:\n"
                        '''
                        - 'Ad Removal Element': A UI component (like a button, banner, or menu item) that initiates the process of getting an ad-free version of the app, usually through a payment. Look for the following text or icon as evidence:
                            Text: 'Remove Ads', 'Go Ad-Free', 'Ad-Free', 'No Ads', 'Disable Ads', et al.
                            Icon: Shield (🛡️): Symbol for 'blocking' or 'protection', or prohibition sign with a slash with text 'Ad(s)' in it.
                        - 'Close Button': Any clickable UI element that visually suggests to the user that they can dismiss, skip, or otherwise exit the current interface. Look for the following text or icon:
                            An 'X' icon (e.g., X, ✕, ⊗).
                            Buttons that use words to indicate the action. For example, buttons with text labels like 'Close,' 'Skip,' 'Skip Ad,' 'Done,' or 'Skip Video.'
                            Progression or Skip Icons: Icons that metaphorically suggest moving past the ad. For example, a 'fast-forward' icon (>> or ►►), which implies speeding through the ad. A 'next' icon (>), which implies moving on to the next content and always combines with text 'Continue to the app.'
                        - 'Purchase Interface': A screen, pop-up, or dedicated section within the app specifically designed to solicit a payment from the user for a service, subscription, or premium feature. This interface should include: explicit pricing (e.g., "$4.99/month", "¥50/year", "Unlock for $9.99"), list of benefits/features: (e.g., "Ad-free experience," "Access to all filters," "Unlimited cloud storage"), which is often presented as a list with checkmarks (✓), and a clear Call-to-Action (CTA) (e.g., "Subscribe Now," "Upgrade to Pro," "Start Free Trial," or "Continue.")
                        - 'Reward Element': any visual component (e.g., text, icon, button) that explicitly offers, grants, or confirms a reward to the user. It can be buttons/prompts informing users to watch an ad for a benefit (e.g., "Watch Ad to Double Coins," "Get a free life").
                            Exclusions: 
                            - Standard Navigation: Buttons like "Continue," "Next Level," or "Retry" that do not offer any explicit bonus. 
                            - Standard Score Screens: Game-over or level-complete screens that only show performance metrics (score, time) without highlighting any bonus items or currency earned.
                        - 'Shake Element': any visual component in a user interface designed to instruct, prompt, or suggest that the user should physically shake their mobile device to trigger an action. It can be text ("Shake", "Shake to...", "摇一摇", "摇动", etc.) or icon (an UI element (e.g., a gift box, red envelope, button, card) that uses a shaking or wobbling animation).
                        - 'Voluntary Ad Trigger Element': an interactive UI element (e.g., a button, icon, or text link) that, when clicked by the user, initiates the play of an ad. Looking for the following text or icon as evidence:
                            - Explicit Text, such as: "Watch Ad", "Watch Video", "Unlock by watching a video"
                            - Suggestive Iconography, including: A play symbol (▶️), A video camera icon (📹), and coin stack (🪙) or gem (💎), often with a small play symbol overlayed on it.\n
                        - 'Non-Fullscreen Ad': an ad that doesn't occupy the entire or the majority of screen.
                        - 'Asymmetric Element Pairs': A pair of elements are positioned next to each other in the UI, with a clear semantic symmetry: one benefits the user, while the other benefits the developer (e.g., “close ad” vs. “open landing page”). However, the visual design clearly emphasizes the developer-benefiting element (e.g., highlighted color or bold font). 
                            - You should treat the pair as a single UI element when outputting. If the UI Tree does not contain a container that includes both elements, create a new element and figure out the minimal bounding box that encloses the pair.
                        - 'Disguised Element': UI elements visually imitating iOS system interfaces such as notifications or alerts, but actually designed by the app or ad developer. The purpose is usually to trick the user into clicking the element.
                        '''

                        "Step 4: Explicitly identify and describe alerts:\n"
                        "- Clearly detect any fully visible, topmost urgent alerts requiring immediate action.\n"
                        "- If an alert is partially visible or non-urgent (small corner overlay), clearly mention it but mark as non-topmost.\n\n"

                        "Step 5: Clearly describe UI layout and function concisely:\n"
                        "- Provide a clear, structured top-down, left-right UI description explicitly mentioning UI components (e.g., header bar, sidebar, footer menu, content area).\n"
                        "- Explicitly note any feedback, status, or error messages clearly visible in UI; leave empty if none.\n\n"

                        "Strictly follow the output schema and clearly adhere to provided examples.\n"
                        "An element can belong to multiple types at the same time. For example, an icon can both invite the user to watch an ad (Ad Trigger Element) and display the ad reward (Reward Element).\n"
                        "In this case, list all the semantic types of that element as separate list items in the element list."
                    ))
                },
                {
                    "text": "Below is the UI runtime elements XML:\n",
                },
                {
                    "text": _dump_xml(ui_runtime_elements),
                },
                {
                    "text": "\n\nBelow is the screenshot image for reference:\n",
                },
                {
                    "inlineData": {
                        "mimeType": "image/png",
                        "data": base64_screenshot,
                    },
                },
                {
                    "text": "".join((
                        "\n\n"
                        "" if not focus_ads else "Advertisements seem to exist in the page.\n\n"
                        "Now start matching elements by their XML ID and bounding box within the screenshot to infer meaning."
                    )),
                },
            ],
        }

        original_messages_semantic = copy.deepcopy(messages_semantic)

        if rag:
            messages_semantic["parts"].extend([
                {
                    "text": "\n\n"
                            "Below are several images of UI elements similar to those in the current screenshot. Please refer to the annotations of these images to help identify and classify the UI elements in the screenshot.\n"
                },
                *[
                    item
                    for h in rag
                    for item in (
                        [
                            {
                                "inlineData": {
                                    "mimeType": "image/jpeg",
                                    "data": h["image"],
                                }
                            },
                            {
                                "text": h["description"]
                            }
                        ] if ("image" in h and "description" in h) else [
                            {
                                "text": h["description"]
                            }
                        ]
                    )
                ]
            ])

        if not Uniapi:
            count_token = self.count_token(model="auto", contents=messages_semantic)
            if count_token and count_token.total_tokens >= 25000:
                messages_semantic = original_messages_semantic
            if count_token:
                logger.info(f"Successfully count prompt tokens: {count_token.total_tokens}")

        # Generate the response schema from the Pydantic model
        response_format_semantic = SemanticView

        config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(
                include_thoughts=True,
                thinking_budget=10000,  # max thinking
            ),
            http_options={'timeout': 600000},
            # max_output_tokens=50000,
            response_mime_type="application/json",
            response_schema=response_format_semantic,
            top_p=0.01,
            top_k=1,
            temperature=0.0,
        )

        # Call the Gemini API
        try:
            completion = self.send_request(
                model="auto",
                contents=messages_semantic,
                config=config,
            )

            # Extract the response content
            semantic_analyze_result = completion.parsed

            # Get token count from completion response
            logger.debug(
                f"Analyze_UI: token consumed:"
                f"{completion.usage_metadata.prompt_token_count} + {completion.usage_metadata.candidates_token_count}"
                f"= {completion.usage_metadata.total_token_count}"
            )
            if completion.usage_metadata.prompt_token_count >= 16000:
                logger.warning(f"Semantic_Analyze_UI: Used {completion.usage_metadata.prompt_token_count} tokens for prompt >= 16k")

            token_record('semantic_analyze_ui', completion.usage_metadata.prompt_token_count, completion.usage_metadata.candidates_token_count)

            return semantic_analyze_result
        except Exception as e:
            logger.error(f"Error during Gemini API call: {e}")
            raise e

        # if semantic_analyze_result.semantic_element_without_id:
        #     messages_relocate = {
        #         "role": "user",
        #         "parts": [
        #             {
        #                 "text": "".join((
        #                     "In the UI screenshot, detect the corresponding UI element and bounding box for each description in the list below. The box should be [y_min, x_min, y_max, x_max] normalized to 0-1000.\n"
        #                 ))
        #             },
        #             {
        #                 "text": "Below is the UI elements' description:\n",
        #             },
        #             {
        #                 "text": json.dumps([{
        #                     'id': order,
        #                     "element_location": item.location,
        #                     "description": item.description,
        #                 } for order, item in enumerate(semantic_analyze_result.semantic_element_without_id)], indent=2, ensure_ascii=False),
        #             },
        #             {
        #                 "text": "\n\nBelow is the screenshot image for reference:\n",
        #             },
        #             {
        #                 "inlineData": {
        #                     "mimeType": "image/png",
        #                     "data": base64_screenshot,
        #                 },
        #             },
        #             {
        #                 "text": "Now start identify these elements and determining their bounding box within the screenshot."
        #             }
        #         ],
        #     }
        #
        #     class PartSemanticElement(BaseModel):
        #         id: int = Field(
        #             ...,
        #             description="The ID of element, ensure the output id is aligned with 'id' field in the input prompt."
        #         )
        #         bounding_box: BoundingBox = Field(
        #             ...,
        #             description="The bounding box of UI element. The box should be [y_min, x_min, y_max, x_max] normalized to 0-1000."
        #         )
        #
        #     class SemanticElementsResponse(BaseModel):
        #         elements: List[PartSemanticElement]
        #
        #     # Generate the response schema from the Pydantic model
        #     response_format_semantic_part = SemanticElementsResponse
        #
        #     config = types.GenerateContentConfig(
        #         thinking_config=types.ThinkingConfig(
        #             include_thoughts=False,
        #             thinking_budget=0,  # max thinking
        #         ),
        #         response_mime_type="application/json",
        #         response_schema=response_format_semantic_part,
        #         top_p=0.01,
        #         top_k=1,
        #         temperature=0.0,
        #     )
        #
        #     # Call the OpenAI API
        #     try:
        #         completion = self.send_request(
        #             model="auto",
        #             contents=messages_relocate,
        #             config=config,
        #         )
        #
        #         # Extract the response content
        #         semantic_analyze_relocate = completion.parsed
        #
        #         # Get token count from completion response
        #         logger.debug(
        #             f"Relocate_UI_Element: token consumed:"
        #             f"{completion.usage_metadata.prompt_token_count} + {completion.usage_metadata.candidates_token_count}"
        #             f"= {completion.usage_metadata.total_token_count}"
        #         )
        #         if completion.usage_metadata.prompt_token_count >= 16000:
        #             logger.warning(
        #                 f"Relocate_UI: Used {completion.usage_metadata.prompt_token_count} tokens for prompt >= 16k")
        #
        #         token_record('relocate_ui_element', completion.usage_metadata.prompt_token_count,
        #                      completion.usage_metadata.candidates_token_count)
        #
        #         for element in semantic_analyze_relocate.elements:
        #             semantic_analyze_result.semantic_element_without_id[element.id].bounding_box = element.bounding_box
        #     except Exception as e:
        #         logger.error(f"Error during Gemini API call: {e}")
        #         raise e

    def rag_analyze(self, base64_screenshot: str, similar_ui: list = [], element_groups: list = []) -> RAGView:
        ui_analyze_result = None
        # if similar_ui:
        messages_rag_ui = {
            "role": "user",
            "parts": [
                {
                    "text": "".join((
                        "Your task is to analyze the provided UI and visually compare it with similar UIs from the RAG system to accurately determine the current UI’s semantic category.\n\n"
                        "Detailed Steps and Exact Restrictions:\n\n"

                        "Carefully identify the type of the interface that currently occupies most of the screen space or at the topmost and can be considered the background of current screen.\n"
                        "When determining the interface type of the current screen, if the interface prevents the user from accessing all or most of the UI elements beneath it (even if those elements remain visible), it should be considered the current screen’s interface type.\n"
                        "For example, when an iOS notification pop-up appears, the user cannot continue interacting with the app and must tap “OK” to proceed. Although the pop-up occupies only a small portion of the screen, it is still regarded as the topmost interface.\n"
                        '''
                        Specifically, firstly pay close attention to **whether the current page is a full-screen ad** by following steps:
                            carefully checking the screenshot four corners for:
                                (1) AdChoices ▶,
                                (2) Ad info icon (“i” in a circle), 
                                (3) ad's countdown timer (a number representing seconds left of this ad), 
                                (4) a speaker icon, which is used for adjusting ad's volume,
                                (5) a video playback progress bar, typically a thin yellow rectangular bar located at the bottom of the image.
                        Then select the most appropriate semantic category for the current UI from the following types. Note that the RAG system provides reference pages and their semantic categories based on layout similarity. Briefly analyze the similarities and differences between the current UI and the retrieved pages. If any retrieved page is semantically the same as the current one, please adopt its semantic category as the final decision.
                        Interface out of the test app:
                            - 'Home Screen': The interface user returns to after leaving the app. Characterized by a grid of app icons, a dock at the bottom, and a wallpaper background.
                            - 'Control Center': An overlay (usually swiped from the top-right) with system control toggles (e.g., Wi-Fi, Brightness) over a blurred background.
                            - 'Other interface out of the test app': Interfaces not rendered or displayed within the currently tested app’s process, such as iOS system interfaces (e.g., Game Center, Notification Center) or other app interfaces (e.g., browser tabs).
                        Interface in the test app but not in a full-screen ad:
                            - 'Purchase Interface': A screen, pop-up, or dedicated section within the app specifically designed to solicit a payment from the user for a service, subscription, or premium feature. This interface should include: explicit pricing (e.g., "$4.99/month", "¥50/year", "Unlock for $9.99"), list of benefits/features: (e.g., "Ad-free experience," "Access to all filters," "Unlimited cloud storage"), which is often presented as a list with checkmarks (✓), and a clear Call-to-Action (CTA) (e.g., "Subscribe Now," "Upgrade to Pro," "Start Free Trial," or "Continue.")
                            - 'App Functional Interface': Functional interfaces rendered and displayed while the app is running, which may include non-fullscreen ads (or ad banners).
                        Interface in a full-screen ad:
                            - 'Full-Screen Ad Interface': the ad interface which occupies the entire screen or the majority.
                            - 'Landing Page': A specific, interactive full-screen destination a user arrives from an in-app ad. The screen must show a Third-Party UI Frame which cover the majority of the screen. The page is displayed inside a context, such as:
                                - A web browser (e.g., Chrome, Safari), indicated by a visible URL bar, tabs, or navigation buttons.
                                - An app store (e.g., Apple App Store, Google Play Store), indicated by a blue close button "Done", "Get"/"Install" buttons in their standard positions, and other standard layouts.\n\n
                        '''
                    ))
                },
                {
                    "text": "\n\nBelow is the screenshot image for reference:\n",
                },
                {
                    "inlineData": {
                        "mimeType": "image/png",
                        "data": base64_screenshot,
                    },
                },
            ],
        }

        messages_rag_ui["parts"].extend([
            {
                "text": "\n\n"
                        "Below are several UIs similar to current screenshot. Please refer to the annotations of these images to help identify and classify current screenshot.\n"
            },
            *[
                item
                for index, h in enumerate(similar_ui)
                for item in (
                    [
                        {
                            "text": f"Similar UI {index}\n"
                        },
                        {
                            "inlineData": {
                                "mimeType": "image/jpeg",
                                "data": h["image"],
                            }
                        },
                        {
                            "text": '\n' + h["description"] + '\n\n'
                        }
                    ]
                )
            ]
        ])

        # Generate the response schema from the Pydantic model
        response_format_semantic = RAGUIView

        config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(
                include_thoughts=True,
                thinking_budget=10000,  # max thinking
            ),
            http_options={'timeout': 600000},
            response_mime_type="application/json",
            response_schema=response_format_semantic,
            top_p=0.01,
            top_k=1,
            temperature=0.0,
        )

        # Call the Gemini API
        try:
            completion = self.send_request(
                model="auto",
                contents=messages_rag_ui,
                config=config,
            )

            # Extract the response content
            ui_analyze_result = completion.parsed

            # Get token count from completion response
            logger.debug(
                f"RAG_Analyze_UI: token consumed:"
                f"{completion.usage_metadata.prompt_token_count} + {completion.usage_metadata.candidates_token_count}"
                f"= {completion.usage_metadata.total_token_count}"
            )
            if completion.usage_metadata.prompt_token_count >= 16000:
                logger.warning(
                    f"RAG_Analyze_UI: Used {completion.usage_metadata.prompt_token_count} tokens for prompt >= 16k")

            token_record('rag_analyze_ui', completion.usage_metadata.prompt_token_count,
                         completion.usage_metadata.candidates_token_count)
        except Exception as e:
            logger.error(f"Error during Gemini API call: {e}")
            raise e


        element_analyze_result = None
        # if element_groups:
        messages_rag_elements = {
            "role": "user",
            "parts": [
                {
                    "text": "".join((
                        "Your task is to analyze the provided UI element groups and visually compare target element with similar cases from the RAG system to accurately determine the target element’s semantic category.\n\n"
                        "Detailed Steps and Exact Restrictions:\n\n"

                        "Carefully analyze UI elements and decide their semantics:\n"
                        '''
                        We provide multiple element groups below. Each group includes the current element’s ID, its screenshot, and UI elements recommended by the RAG system based on visual similarity along with their semantic categories. For each group, carefully analyze the visual features of the current element, briefly compare its visual similarities and differences with the retrieved elements, and determine the semantic category of the current element.
                        - 'Ad Removal Element': A UI component (like a button, banner, or menu item) that initiates the process of getting an ad-free version of the app, usually through a payment. Look for the following text or icon as evidence:
                            Text: 'Remove Ads', 'Go Ad-Free', 'Ad-Free', 'No Ads', 'Disable Ads', et al.
                            Icon: Shield (🛡️): Symbol for 'blocking' or 'protection', or prohibition sign with a slash with text 'Ad(s)' in it.
                        - 'Close Button': Any clickable UI element that visually suggests to the user that they can dismiss, skip, or otherwise exit the current interface. Look for the following text or icon:
                            An 'X' icon (e.g., X, ✕, ⊗).
                            Buttons that use words to indicate the action. For example, buttons with text labels like 'Close,' 'Skip,' 'Skip Ad,' 'Done,' or 'Skip Video.'
                            Progression or Skip Icons: Icons that metaphorically suggest moving past the ad. For example, a 'fast-forward' icon (>> or ►►), which implies speeding through the ad. A 'next' icon (>), which implies moving on to the next content and always combines with text 'Continue to the app.'
                        - 'Purchase Interface': A screen, pop-up, or dedicated section within the app specifically designed to solicit a payment from the user for a service, subscription, or premium feature. This interface should include: explicit pricing (e.g., "$4.99/month", "¥50/year", "Unlock for $9.99"), list of benefits/features: (e.g., "Ad-free experience," "Access to all filters," "Unlimited cloud storage"), which is often presented as a list with checkmarks (✓), and a clear Call-to-Action (CTA) (e.g., "Subscribe Now," "Upgrade to Pro," "Start Free Trial," or "Continue.")
                        - 'Reward Element': Any visual design that displays in-game items (e.g., coins, health) that the user already owns or will receive—for example, a status bar showing current coins or coins shown next to an ad button indicating the reward.
                            There are also some exclusions: 
                            - Standard Navigation: Buttons like "Continue," "Next Level," or "Retry" that do not offer any explicit bonus. 
                            - Standard Score Screens: Game-over or level-complete screens that only show performance metrics (score, time) without highlighting any bonus items or currency earned.
                        - 'Shake Element': any visual component in a user interface designed to instruct, prompt, or suggest that the user should physically shake their mobile device to trigger an action. It can be text ("Shake", "Shake to...", "摇一摇", "摇动", etc.) or icon (an UI element (e.g., a gift box, red envelope, button, card) that uses a shaking or wobbling animation).
                        - 'Voluntary Ad Trigger Element': an interactive UI element (e.g., a button, icon, or text link) that, when clicked by the user, initiates the play of an ad. Looking for the following text or icon as evidence:
                            - Explicit Text, such as: "Watch Ad", "Watch Video", "Unlock by watching a video"
                            - Suggestive Iconography, including: A play symbol (▶️), A video camera icon (📹), and coin stack (🪙) or gem (💎), often with a small play symbol overlayed on it.\n
                        - 'Non-Fullscreen Ad': an ad that doesn't occupy the entire or the majority of screen.
                        - 'Asymmetric Element Pairs': A pair of elements are positioned next to each other in the UI, with a clear semantic symmetry: one benefits the user, while the other benefits the developer (e.g., “close ad” vs. “open landing page”). However, the visual design clearly emphasizes the developer-benefiting element (e.g., highlighted color or bold font). 
                            - You should treat the pair as a single UI element when outputting. If the UI Tree does not contain a container that includes both elements, create a new element and figure out the minimal bounding box that encloses the pair.
                        - 'Disguised Element': UI elements visually imitating iOS system interfaces such as notifications or alerts, but actually designed by the app or ad developer. The purpose is usually to trick the user into clicking the element.
                        - 'Normal Element': A UI element that does not belong to any of the above semantic categories and has no distinctive characteristics.\n\n
                        '''
                    ))
                },
            ],
        }

        messages_rag_elements["parts"].extend([
            {
                "text": "\n\n"
                        "Below are several groups of elements. Please refer to the annotations of similar elements to help identify and classify target element in each group.\n"
            },
            *[
                item
                for index, h in enumerate(element_groups)
                for item in (
                    [
                        {
                            "text": f"Group {index}\n"
                        },
                        {
                            "text": f"Target Element id: {h['target_id']}\n"
                        },
                        {
                            "text": f"Target Element screenshot:\n"
                        },
                        {
                            "inlineData": {
                                "mimeType": "image/jpeg",
                                "data": h["element_screenshot"],
                            }
                        },
                        *[
                            elem_item
                            for element_index, e in enumerate(h['similar_elements'])
                            for elem_item in (
                                [
                                    {
                                        "text": f"\nSimilar Element {element_index} to Target Element {h['target_id']}\n"
                                    },
                                    {
                                        "inlineData": {
                                            "mimeType": "image/jpeg",
                                            "data": e["element_screenshot"],
                                        }
                                    },
                                    {
                                        "text": '\n' + e['description'] + '\n',
                                    }
                                ]
                            )
                        ]
                    ]
                )
            ]
        ])

        # Generate the response schema from the Pydantic model
        response_format_semantic = RAGElementsView

        config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(
                include_thoughts=True,
                thinking_budget=10000,  # max thinking
            ),
            http_options={'timeout': 600000},
            response_mime_type="application/json",
            response_schema=response_format_semantic,
            top_p=0.01,
            top_k=1,
            temperature=0.0,
        )

        # Call the Gemini API
        try:
            completion = self.send_request(
                model="auto",
                contents=messages_rag_elements,
                config=config,
            )

            # Extract the response content
            element_analyze_result = completion.parsed

            # Get token count from completion response
            logger.debug(
                f"RAG_Analyze_Elements: token consumed:"
                f"{completion.usage_metadata.prompt_token_count} + {completion.usage_metadata.candidates_token_count}"
                f"= {completion.usage_metadata.total_token_count}"
            )
            if completion.usage_metadata.prompt_token_count >= 16000:
                logger.warning(
                    f"RAG_Analyze_Elements: Used {completion.usage_metadata.prompt_token_count} tokens for prompt >= 16k")

            token_record('rag_analyze_elements', completion.usage_metadata.prompt_token_count,
                         completion.usage_metadata.candidates_token_count)
        except Exception as e:
            logger.error(f"Error during Gemini API call: {e}")
            raise e

        return RAGView(
            rag_ui=ui_analyze_result,
            rag_elements=element_analyze_result
        )
