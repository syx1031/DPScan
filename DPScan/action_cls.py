from abc import ABC, abstractmethod
from typing import List

from gpt_cls import GPTClient, GeminiClient, UIView, UserAction, SemanticUserAction, UserActionTypeClick, UserActionTypeCompleted, UserActionTypeInput, UserActionTypeNone, UserActionTypeRestart, UserActionTypeSwitch, UserActionTypeHover, UserActionTypeIdle, SemanticView
from ui_cls import BaseUI

from utils import sleep, rand

import logging
logger = logging.getLogger(__name__)

class BaseAction(ABC):
    @abstractmethod
    def __init__(self, vars):
        pass

    @classmethod
    @abstractmethod
    def create_action(cls, vars):
        pass

    def replace_action(self, vars):
        self.__init__(vars)
        return self

    @staticmethod
    def analyze_action(*args, **kwargs) -> UserAction:
        # Use GPT-4o to analyze the UIView and provide a response
        # gpt = GPTClient()
        # actions = gpt.analyze_action(*args, **kwargs)
        gemini = GeminiClient()
        actions = gemini.analyze_action(*args, **kwargs)
        return actions

    @staticmethod
    def semantic_analyze_action(*args, **kwargs) -> SemanticUserAction:
        # gpt = GPTClient()
        # semantic_actions = gpt.semantic_analyze_action(*args, **kwargs)
        gemini = GeminiClient()
        semantic_actions = gemini.semantic_analyze_action(*args, **kwargs)
        return semantic_actions
    
    @staticmethod
    def _get_ui_slim(semantic_ui: SemanticView) -> str:
        # return GPTClient._get_slim_ui(ui)
        # return GeminiClient._get_slim_ui(ui)
        return GeminiClient._concat_slim_ui("[]", semantic_ui)
    
    @staticmethod
    def _get_action_history(action_history: List[str]) -> str:
        # return GPTClient._get_action_history(action_history)
        return GeminiClient._get_action_history(action_history)

    @abstractmethod
    def apply_actions(self, actions: UserAction, vid_map: dict, base_ui: BaseUI) -> int|bool:
        pass

class iOSAction(BaseAction):
    def __init__(self, vars):
        self.restart_app, self.zc, self.appium_ui, self.bring_to_foreground, self.bring_to_background, self.gbscript, self.screensize = vars

    @classmethod
    def create_action(cls, vars):
        return cls(vars)

    def tap(self, x, y):
        self.zc.tap(x, y)

    def hover(self, start, end):
        self.appium_ui.appium.hover(start, end)
    
    def hide_keyboard(self):
        self.bring_to_foreground() # Bring to foreground before calling gbscript (otherwise stuck)
        self.zc.hide_keyboard() # This zxtouch solution also works
        self.gbscript.exports_sync.hidekeyboard() # This frida solution also works
        sleep(rand(0.5, 1)) # make sure the keyboard is hidden
    
    def input_text(self, text: str):
        self.zc.input_text(text)

    def apply_actions(self, actions: UserAction, vid_map: dict, base_ui: BaseUI) -> int|bool:
        effective_actions = 0
        for action in actions.actions:
            match action.action:
                case UserActionTypeClick(type="click", element_id=element_id, times=times):
                    assert action.target_element_id == element_id, f"Target element ID mismatch: {action.target_element_id} != {element_id}"
                    logger.debug(f"Applying action: Click action on element_id={element_id}, {times} time(s).")
                    center = base_ui.get_center(vid_map[element_id])
                    for _ in range(times):
                        self.tap(center['x'], center['y'])
                        sleep(rand(0.1, 0.2))
                    self.hide_keyboard()
                    effective_actions += 1
                case UserActionTypeInput(type="input", element_id=element_id, text=text):
                    assert action.target_element_id == element_id, f"Target element ID mismatch: {action.target_element_id} != {element_id}"
                    logger.debug(f"Applying action: Input action on element_id={element_id} with text='{text}'.")
                    center = base_ui.get_center(vid_map[element_id])
                    self.tap(center['x'], center['y'])
                    sleep(rand(0.5, 1))
                    self.input_text('\b' * 20 + text + '\n')
                    sleep(rand(0.5, 1))
                    self.hide_keyboard()
                    effective_actions += 1
                case UserActionTypeRestart(type="restart"):
                    logger.debug("Applying action: Restart the app.")
                    sleep(2)
                    self.restart_app()
                    effective_actions += 1
                case UserActionTypeCompleted(type="completed"):
                    logger.info("Applying action: The task is already accomplished.")
                    return True
                case UserActionTypeNone(type="none"):
                    logger.debug("Applying action: Skip.")
                case _:
                    raise NotImplementedError(f"Unknown action type: {action}")
        return effective_actions

    def apply_semantic_actions(self, actions: SemanticUserAction, vid_map: dict, base_ui: BaseUI) -> int|bool:
        # coordinates = actions.click_coordinates

        effective_actions = 0
        for action in actions.actions:
            match action.action:
                case UserActionTypeClick(type="click", element_id=element_id, times=times):
                    assert action.target_element_id == element_id, f"Target element ID mismatch: {action.target_element_id} != {element_id}"
                    logger.debug(f"Applying action: Click action on element_id={element_id}, {times} time(s).")
                    if element_id > 0:
                        center = base_ui.get_center(vid_map[element_id])
                    # elif coordinates:
                    #     center = {'x': int(coordinates[0] * self.screensize[0]), 'y': int(coordinates[1] * self.screensize[1])}
                    else:
                        raise NotImplementedError(f"No available coordinates for clicking: id={element_id}")
                    for _ in range(times):
                        self.tap(center['x'], center['y'])
                        sleep(rand(0.1, 0.2))
                    self.hide_keyboard()
                    effective_actions += 1
                case UserActionTypeSwitch(type="switch", target=target):
                    logger.debug(f"Applying action: Switch to interface {target}")
                    if target == "Test App":
                        self.bring_to_foreground()
                    elif target == "Home Screen":
                        self.bring_to_background()
                    effective_actions += 1
                case UserActionTypeHover(type="hover", element_id=element_id):
                    logger.debug(f"Applying action: Hover action on element_id={element_id}.")
                    top_left, right_bottom = base_ui.get_corners(vid_map[element_id])
                    self.hover(top_left, right_bottom)
                    self.hide_keyboard()
                    effective_actions += 1
                # case UserActionTypeInput(type="input", element_id=element_id, text=text):
                #     assert action.target_element_id == element_id, f"Target element ID mismatch: {action.target_element_id} != {element_id}"
                #     logger.debug(f"Applying action: Input action on element_id={element_id} with text='{text}'.")
                #     center = base_ui.get_center(vid_map[element_id])
                #     self.tap(center['x'], center['y'])
                #     sleep(rand(0.5, 1))
                #     self.input_text('\b' * 20 + text + '\n')
                #     sleep(rand(0.5, 1))
                #     self.hide_keyboard()
                #     effective_actions += 1
                case UserActionTypeRestart(type="restart"):
                    logger.debug("Applying action: Restart the app.")
                    sleep(2)
                    self.restart_app()
                    effective_actions += 1
                case UserActionTypeCompleted(type="completed"):
                    logger.info("Applying action: The task is already accomplished.")
                    return True
                case UserActionTypeIdle(type="idle"):
                    logger.debug("Applying action: Idle.")
                    # sleep(3)
                    effective_actions += 1
                # case UserActionTypeNone(type="none"):
                #     logger.debug("Applying action: Skip.")
                case _:
                    raise NotImplementedError(f"Unknown action type: {action}")
        return effective_actions

