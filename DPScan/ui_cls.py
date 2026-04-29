from abc import ABC, abstractmethod
import json
import cv2
import base64
import numpy as np
from typing import Callable, List

from deprecation import deprecated

from gpt_cls import GPTClient, GeminiClient, UIElementType, UIView, SemanticView, UIElement, SemanticUIElement, DPList, RAGView

from utils import time_consumed

class BaseUI(ABC):
    @abstractmethod
    def __init__(self, vars):
        pass

    @classmethod
    @abstractmethod
    def create_ui(cls, vars):
        pass

    def replace_ui(self, vars):
        self.__init__(vars)
        return self

    @abstractmethod
    def getui(self):
        pass

    @staticmethod
    @abstractmethod
    def post_process_ui(uielements: dict, base64_screenshot: str) -> tuple[dict, dict]:
        pass

    @staticmethod
    def dp_detection(ui_sequence: list) -> DPList:
        gemini = GeminiClient()
        detection = gemini.detect_dp(ui_sequence)
        return detection

    @staticmethod
    def rag_analyze(*args, **kwargs) -> RAGView:
        gemini = GeminiClient()
        rag_result = gemini.rag_analyze(*args, **kwargs)
        return rag_result

    @staticmethod
    def analyze_ui(uielements: dict, base64_screenshot: str) -> UIView:
        # Use GPT-4o to analyze the UI elements and provide a response
        # gpt = GPTClient()
        gemini = GeminiClient()

        focus_ads = False

        if BaseUI.element_selector(uielements, 'name', '广告', strict=True) is not None \
            or BaseUI.element_selector(uielements, 'name', '打开或下载第三方应用', strict=True) is not None\
                or BaseUI.element_selector(uielements, 'ocr_text', '广告', strict=True) is not None:
            focus_ads = True
        
        # analyzed, semantic_analyzed = gpt.analyze_screenshot(base64_screenshot, uielements, focus_ads)
        analyzed = gemini.analyze_screenshot(base64_screenshot, uielements, focus_ads)

        if analyzed.ads and analyzed.ads.is_ad_closeable and analyzed.ads.ad_close_button_id > 0:
            # Add the close button if not present (which is strange but frequently happens)
            if not any(element.id == analyzed.ads.ad_close_button_id for element in analyzed.elements):
                analyzed.action_elements.append(UIElement(id=analyzed.ads.ad_close_button_id, ui_type='IconButton', description='Close button for the'+(' urgent topmost' if analyzed.ads.is_ad_topmost else '')+' advertisement', text='Close', clickability=1.0, location=''))

            # if not any(element.id == analyzed.ads.ad_close_button_id for element in semantic_analyzed.semantic_element):
            #     semantic_analyzed.semantic_element.append(SemanticUIElement(id=analyzed.ads.ad_close_button_id, ui_type='IconButton', description='Close button for the'+(' topmost' if analyzed.ads.is_ad_topmost else '')+' advertisement', text='Close', clickability=1.0, location='', type='Close Button', evidence=''))

        def align_buttons_and_text_inputs(elements: list[UIElement]):
            # Make sure buttons and text inputs are aligned
            vid_map = BaseUI.get_vid_map(uielements)
            for element in elements:
                if element.id in vid_map:
                    if any(k in vid_map[element.id].get('type', '') for k in ['TextField', 'TextView']) and element.ui_type != UIElementType.INPUT_TEXT:
                        element.ui_type = UIElementType.INPUT_TEXT
                    elif 'Button' in vid_map[element.id].get('type', '') and element.ui_type not in [UIElementType.BUTTON, UIElementType.TEXT_BUTTON, UIElementType.ICON_BUTTON]:
                        element.ui_type = UIElementType.OTHER_INTERACTABLE

        align_buttons_and_text_inputs(analyzed.elements)
        # align_buttons_and_text_inputs(semantic_analyzed.elements)

        return analyzed

    @staticmethod
    def semantic_analyze_ui(uielements: dict, base64_screenshot: str, rag: list = []) -> SemanticView:
        # Use GPT-4o to analyze the UI elements and provide a response
        # gpt = GPTClient()
        gemini = GeminiClient()

        focus_ads = False

        if BaseUI.element_selector(uielements, 'name', '广告', strict=True) is not None \
                or BaseUI.element_selector(uielements, 'name', '打开或下载第三方应用', strict=True) is not None \
                or BaseUI.element_selector(uielements, 'ocr_text', '广告', strict=True) is not None:
            focus_ads = True

        semantic_analyzed = gemini.semantic_analyze_screenshot(base64_screenshot, uielements, focus_ads, rag=rag)

        def align_buttons_and_text_inputs(elements: list[UIElement]):
            # Make sure buttons and text inputs are aligned
            vid_map = BaseUI.get_vid_map(uielements)
            for element in elements:
                if element.id in vid_map:
                    if any(k in vid_map[element.id].get('type', '') for k in
                           ['TextField', 'TextView']) and element.ui_type != UIElementType.INPUT_TEXT:
                        element.ui_type = UIElementType.INPUT_TEXT
                    elif 'Button' in vid_map[element.id].get('type', '') and element.ui_type not in [
                        UIElementType.BUTTON, UIElementType.TEXT_BUTTON, UIElementType.ICON_BUTTON]:
                        element.ui_type = UIElementType.OTHER_INTERACTABLE

        align_buttons_and_text_inputs(semantic_analyzed.elements)

        return semantic_analyzed
    
    @staticmethod
    def get_frame(element: dict) -> dict:
        if 'absolute_frame' in element:
            return element['absolute_frame']
        elif 'frame_box' in element:
            return {'x': element['frame_box']['top_left']['x'], 'y': element['frame_box']['top_left']['y'], 'width': element['frame_box']['right_bottom']['x'] - element['frame_box']['top_left']['x'], 'height': element['frame_box']['right_bottom']['y'] - element['frame_box']['top_left']['y']}
        else:
            return element['_absolute_frame']
        
    @staticmethod
    def get_center(element: dict) -> dict:
        frame = BaseUI.get_frame(element)
        return {'x': frame['x'] + frame['width'] // 2, 'y': frame['y'] + frame['height'] // 2}

    @staticmethod
    def get_corners(element: dict) -> tuple[dict, dict]:
        frame = BaseUI.get_frame(element)
        top_left = {'x': frame['x'], 'y': frame['y']}
        bottom_right = {'x': frame['x'] + frame['width'], 'y': frame['y'] + frame['height']}
        return top_left, bottom_right

    @staticmethod
    def get_random_clickable_element(uielements: dict, keywords: list = None) -> dict:
        element_candidates = []
        premium_candidates = []

        def traverse(element):
            nonlocal element_candidates, premium_candidates

            def get_text_of(view):
                # 以 name/value/label/text 为主，再追加 ocr_text（若存在）
                candidates = []
                for k in ('name', 'value', 'label', 'text'):
                    if k in view and view[k] is not None:
                        v = view[k]
                        # traversex 已把 list -> " / "，但以防仍为 list
                        if isinstance(v, list):
                            v = ' / '.join(str(i) for i in v)
                        candidates.append(str(v).strip())
                # 主文本（优先第一个非空）
                main_text = next((c for c in candidates if c), '')
                # 附加 OCR（若存在）
                ocr = view.get('ocr_text', '')
                if isinstance(ocr, list):
                    ocr = ' / '.join(str(i) for i in ocr)
                ocr = str(ocr).strip()
                if main_text and ocr:
                    return main_text + ' ' + ocr
                return main_text or ocr or ''
            all_text = get_text_of(element)

            # icon = element.get('icon_label', '')

            frame = BaseUI.get_frame(element)

            if 'Button' in element.get('type', '') and frame["height"] < 900:
                element_candidates.append(element)
                if keywords and any(keyword in all_text for keyword in keywords):
                    premium_candidates.append(element)

            for subview in element.get('subviews', []):
                traverse(subview)

        traverse(uielements)

        import random
        return random.choice(element_candidates) if premium_candidates else random.choice(element_candidates) if element_candidates else None

    @staticmethod
    def find_element_by_frame(uielements: dict, target_frame):
        suitable_element = None

        def check_element(element):
            nonlocal target_frame, suitable_element

            frame = BaseUI.get_frame(element)
            # Check if rectangle overlaps with element bounds
            if target_frame['x'] == frame['x'] and target_frame['y'] == frame['y'] and target_frame['width'] == frame['width'] and target_frame['height'] == frame['height']:
                suitable_element = element

            # Recursively check subviews
            for subview in element.get('subviews', []):
                check_element(subview)

        # Start recursive search from root element
        root_element = uielements
        check_element(root_element)

        return suitable_element
    
    @staticmethod
    def find_suitable_element(uielements: dict, x: int, y: int, width: int, height: int, tolerance: float = 0.01, need_visible=False) -> tuple[dict, int, int]:
        """Find the most suitable (smallest containing) element at the given rectangle"""

        if not uielements:
            return None, 0, 0
            
        suitable_element = None
        max_depth = -1
        min_area = float('inf')
        
        def check_element(element, depth=0):
            nonlocal suitable_element, max_depth, min_area, need_visible

            if need_visible and not element.get('visible', False):
                return
            
            frame = BaseUI.get_frame(element)
            # Check if rectangle overlaps with element bounds
            if (frame['x'] - frame['width'] * tolerance <= x and x + width <= frame['x'] + frame['width'] + frame['width'] * tolerance and
                frame['y'] - frame['height'] * tolerance <= y and y + height <= frame['y'] + frame['height'] + frame['height'] * tolerance):
                    
                area = frame['width'] * frame['height']
                # Update if this contains smaller area
                if area < min_area or (area == min_area and depth > max_depth):
                    suitable_element = element
                    min_area = area
                    max_depth = depth
                    
            # Recursively check subviews
            for subview in element.get('subviews', []):
                check_element(subview, depth + 1)
                
        # Start recursive search from root element
        root_element = uielements
        check_element(root_element)
        
        return suitable_element, max_depth, min_area

    @staticmethod
    def element_template_matching(base64_screenshot: str):
        # 1. 读取图片
        def base64_to_cv2(base64_str):
            # 1. 如果字符串包含 "data:image/jpeg;base64," 等前缀，需要去掉
            if "," in base64_str:
                base64_str = base64_str.split(",")[1]

            # 2. 将 base64 字符串解码为字节流
            img_data = base64.b64decode(base64_str)

            # 3. 将字节流转换为 numpy 数组 (uint8 类型)
            nparr = np.frombuffer(img_data, np.uint8)

            # 4. 使用 cv2.imdecode 将其转换为 OpenCV 图像格式
            img_output = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            return img_output

        img = base64_to_cv2(base64_screenshot)

        # ================================
        # 准备 "X" 模板
        # ================================
        # 创建一个 14x14 的黑底图像
        template_size = 14
        x_template = np.zeros((template_size, template_size), dtype=np.uint8)
        # 画两条白色的对角线 (线宽设为 2 或 3，增加一点容错率)
        # 左上到右下
        cv2.line(x_template, (2, 2), (11, 11), 255, 2)
        # 右上到左下
        cv2.line(x_template, (11, 2), (2, 11), 255, 2)

        original = img.copy()

        # ================================
        # 核心步骤 1：图像预处理，转灰度图
        # ================================
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)

        # ================================
        # 核心步骤 2：边缘检测与连接
        # ================================
        canny_output = cv2.Canny(filtered, 30, 150)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed_edges = cv2.morphologyEx(canny_output, cv2.MORPH_CLOSE, kernel)

        # ================================
        # 核心步骤 3：轮廓提取与筛选
        # ================================
        contours, _ = cv2.findContours(closed_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        detected_elements = []

        image_h, image_w = img.shape[:2]
        total_pixels = image_h * image_w

        for cnt in contours:
            # --- 基础筛选，去掉太小太大的区域 ---
            area = cv2.contourArea(cnt)
            if area > (total_pixels * 0.5):
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / h

            # --- 过滤掉太小的区域 ---
            if w < 30 and h < 30:
                continue

            # --- 计算轮廓的凸包，方便确认轮廓是凹多边形还是凸多边形 ---
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            if hull_area == 0: continue

            solidity = float(area) / hull_area # 轮廓面积占凸包面积的比例，太小的一般是凹多边形
            extent = float(area) / (w * h) # 轮廓面积占 bounding box 面积的比例

            # # ========================================================
            # # ★ 实时查看当前遍历到的轮廓区域 ★
            # # ========================================================
            # # 1. 截取当前轮廓对应的原图区域
            # roi_focus = img[y:y + h, x:x + w].copy()
            #
            # # 2. (可选) 如果图片太小，放大一点方便观察
            # if w < 100 or h < 100:
            #     roi_focus = cv2.resize(roi_focus, (200, 200), interpolation=cv2.INTER_NEAREST)
            #
            # # 3. 在窗口中显示，并附带一点信息
            # cv2.putText(roi_focus, f"W:{w} H:{h}", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            # cv2.imshow("Scanning Contours (Press Any Key)", roi_focus)
            #
            # # 4. 等待按键，如果想自动播放可以把 0 改成 100 (毫秒)
            # key = cv2.waitKey(0)

            shape_type = None
            color = (0, 255, 0)

            # ========================================================
            # 预计算：X 形状模板匹配，查找是否有 X 形状的关闭按钮
            # ========================================================
            match_score = 0

            # 前置条件：长宽比接近正方形，且不能太大了
            if 0.7 < aspect_ratio < 1.3 and float(w) < 100 and float(h) < 100:
                try:
                    # # 1. [可视化步骤1] 获取原始 ROI (用于对比)
                    # roi_gray = gray[y:y + h, x:x + w]

                    # 2. 创建 Mask 并填充轮廓
                    mask = np.zeros((h, w), dtype=np.uint8)
                    offset_cnt = cnt - [x, y]
                    # thickness=-1 表示填充内部，变成实心图
                    cv2.drawContours(mask, [offset_cnt], -1, 255, thickness=-1)

                    # 3. Resize 到 14x14 (归一化)
                    target_mini = cv2.resize(mask, (template_size, template_size), interpolation=cv2.INTER_AREA)

                    # 4. 模板匹配
                    res = cv2.matchTemplate(target_mini, x_template, cv2.TM_CCOEFF_NORMED)
                    match_score = res[0][0]

                    # # ====================================================
                    # # ★★★ 生成调试图表 (核心修改) ★★★
                    # # ====================================================
                    # # 只展示那些分数稍微有点像的 (>0.2)，避免全是噪点弹窗
                    # if match_score > 0.2:
                    #     display_size = 100  # 为了人眼能看清，把 14x14 放大到 100x100 显示
                    #
                    #     # A. 原始区域 (灰度)
                    #     vis_roi = cv2.resize(roi_gray, (display_size, display_size))
                    #     vis_roi = cv2.cvtColor(vis_roi, cv2.COLOR_GRAY2BGR)
                    #     cv2.putText(vis_roi, "Original", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                    #
                    #     # B. 填充后的 Mask (二值)
                    #     vis_mask = cv2.resize(mask, (display_size, display_size))
                    #     vis_mask = cv2.cvtColor(vis_mask, cv2.COLOR_GRAY2BGR)
                    #     cv2.putText(vis_mask, "Filled", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                    #
                    #     # C. 归一化后的输入 (14x14 -> 放大)
                    #     vis_input = cv2.resize(target_mini, (display_size, display_size),
                    #                            interpolation=cv2.INTER_NEAREST)
                    #     vis_input = cv2.cvtColor(vis_input, cv2.COLOR_GRAY2BGR)
                    #     cv2.putText(vis_input, "Input(14px)", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                    #
                    #     # D. 标准模板 (14x14 -> 放大)
                    #     vis_template = cv2.resize(x_template, (display_size, display_size),
                    #                               interpolation=cv2.INTER_NEAREST)
                    #     vis_template = cv2.cvtColor(vis_template, cv2.COLOR_GRAY2BGR)
                    #     cv2.putText(vis_template, "Template", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                    #
                    #     # 拼接成一条长图
                    #     debug_strip = np.hstack([vis_roi, vis_mask, vis_input, vis_template])
                    #
                    #     # 在图上写上最终分数
                    #     cv2.putText(debug_strip, f"Score: {match_score:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    #                 (0, 0, 255), 2)
                    #
                    #     # 弹窗显示 (按任意键继续检测下一个)
                    #     cv2.imshow("X-Detection Pipeline (Press Key)", debug_strip)
                    #     cv2.waitKey(0)  # 暂停，等待看清楚

                except Exception as e:
                    pass

            # ========================================================
            # 分支 A: 识别三角形（可能是 Voluntary Ad Trigger Element）
            # ========================================================
            try:
                # 求最小外接三角形，能够和自己的外接三角形差不多重合的就应该是三角形
                tri_area, triangle_cnt = cv2.minEnclosingTriangle(cnt)
                triangle_match_ratio = area / tri_area
            except:
                triangle_match_ratio = 0

            # 是三角形
            if solidity > 0.6 and triangle_match_ratio > 0.75:
                shape_type = "Triangle"
                color = (0, 255, 255)  # 黄色

            # ========================================================
            # 分支 B: 识别 "X" 形 (使用模板匹配分数)
            # ========================================================
            # 0.45 是一个经验阈值，因为 resize 可能会丢失细节，
            # 而且手绘风格的 X 不会和标准 X 完美重合，通常 > 0.4 就很像了
            elif match_score > 0.6:
                shape_type = "X-Shape"
                color = (0, 0, 255)
                # 调试：打印分数看看
                # print(f"Found X with score: {match_score:.2f}")

            # ========================================================
            # 分支 C: 识别常规按钮，即矩形或圆角矩形等
            # ========================================================
            # 比较饱满并且跟自己的外接矩形即 bbox 差不多重合
            elif solidity > 0.85 and extent > 0.6:
                shape_type = "Rectangle"
                color = (0, 255, 0)  # 绿色

            # --- 绘制结果 ---
            if shape_type:
                detected_elements.append((shape_type, x, y, w, h))
                # cv2.rectangle(original, (x, y), (x + w, y + h), color, 2)
                # cv2.putText(original, shape_type, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # # 显示结果
        # print(f"检测到 {len(detected_elements)} 个元素")
        #
        # debug_view = np.hstack((cv2.cvtColor(closed_edges, cv2.COLOR_GRAY2BGR), original))
        # scale_percent = 50
        # width = int(debug_view.shape[1] * scale_percent / 100)
        # height = int(debug_view.shape[0] * scale_percent / 100)
        # resized = cv2.resize(debug_view, (width, height), interpolation=cv2.INTER_AREA)
        #
        # cv2.imshow("Processing (Left) vs Classification (Right)", resized)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        return detected_elements

    @staticmethod
    def is_in(uielements: dict, el1: object, el2: object) -> bool:
        el1_in_el2 = False
        in_subtree, el1_frame, el2_frame = False, None, None
        def check_element(element, depth=0):
            nonlocal in_subtree, el1_frame, el2_frame, el1_in_el2
            if element['id'] == el1.id:
                el1_frame = BaseUI.get_frame(element)
            elif element['id'] == el2.id:
                el2_frame = BaseUI.get_frame(element)

            if element['id'] == el2.id:
                in_subtree = True
            elif element['id'] == el1.id and in_subtree:
                el1_in_el2 = True

            # Recursively check subviews
            for subview in element.get('subviews', []):
                check_element(subview, depth + 1)

            if element['id'] == el2.id:
                in_subtree = False

        # Start recursive search from root element
        root_element = uielements
        check_element(root_element)

        if el1_in_el2:
            return True

        if not el1_frame or not el2_frame:
            return False

        if el2_frame['x'] <= el1_frame['x'] and el2_frame['x'] + el2_frame['width'] >= el1_frame['x'] + el1_frame['width'] and \
            el2_frame['y'] <= el1_frame['y'] and el2_frame['y'] + el2_frame['height'] >= el1_frame['y'] + el1_frame['height']:
            return True

        return False

    @staticmethod
    def is_in_element(el1: dict, el2: dict):
        el1_frame = BaseUI.get_frame(el1)
        el2_frame = BaseUI.get_frame(el2)

        return el2_frame['x'] <= el1_frame['x'] and el2_frame['x'] + el2_frame['width'] >= el1_frame['x'] + el1_frame['width'] and \
            el2_frame['y'] <= el1_frame['y'] and el2_frame['y'] + el2_frame['height'] >= el1_frame['y'] + el1_frame['height']

    @staticmethod
    def is_align(el1: dict, el2: dict, tolerance=0.05) -> bool:
        el1_frame = BaseUI.get_frame(el1)
        el2_frame = BaseUI.get_frame(el2)

        # 宽高平均值用于相对差异计算
        avg_width = (el1_frame['width'] + el2_frame['width']) / 2
        avg_height = (el1_frame['height'] + el2_frame['height']) / 2

        # 水平对齐条件：
        # x 起点差距 < 宽度5%，且高度差距 < 高度5%
        horizontal_aligned = (
                abs(el1_frame['x'] - el2_frame['x']) <= tolerance * avg_width and
                abs(el1_frame['width'] - el2_frame['width']) <= tolerance * avg_width and
                abs(el1_frame['height'] - el2_frame['height']) <= tolerance * avg_height
        )

        # 垂直对齐条件：
        # y 起点差距 < 高度5%，且宽度差距 < 宽度5%
        vertical_aligned = (
                abs(el1_frame['y'] - el2_frame['y']) <= tolerance * avg_height and
                abs(el1_frame['height'] - el2_frame['height']) <= tolerance * avg_height and
                abs(el1_frame['width'] - el2_frame['width']) <= tolerance * avg_width
        )

        return horizontal_aligned or vertical_aligned

    @staticmethod
    def is_neighbor(uielements: dict, el1: object, el2: object) -> bool:
        el1_frame, el2_frame = None, None

        def check_element(element, depth=0):
            nonlocal el1_frame, el2_frame
            if element['id'] == el1.id:
                el1_frame = BaseUI.get_frame(element)
            elif element['id'] == el2.id:
                el2_frame = BaseUI.get_frame(element)

            # Recursively check subviews
            for subview in element.get('subviews', []):
                check_element(subview, depth + 1)

        # Start recursive search from root element
        root_element = uielements
        check_element(root_element)

        if not el1_frame or not el2_frame:
            return False

        def expand_bbox(bbox, scale=1.1):
            cx = bbox["x"] + bbox["width"] / 2
            cy = bbox["y"] + bbox["height"] / 2
            new_w = bbox["width"] * scale
            new_h = bbox["height"] * scale
            return {
                "x": cx - new_w / 2,
                "y": cy - new_h / 2,
                "width": new_w,
                "height": new_h
            }

        b1 = expand_bbox(el1_frame, scale=1.1)
        b2 = expand_bbox(el2_frame, scale=1.1)

        # 计算是否有交集
        x_overlap = not (b1["x"] + b1["width"] <= b2["x"] or b2["x"] + b2["width"] <= b1["x"])
        y_overlap = not (b1["y"] + b1["height"] <= b2["y"] or b2["y"] + b2["height"] <= b1["y"])

        return x_overlap and y_overlap

    @staticmethod
    def almost_overlap(el1: dict, el2:dict, tolerance=0.7):
        """
        判断两个元素是否几乎完全重叠
        tolerance: 重叠面积占较小元素面积的比例，默认 0.7
        """
        # 计算交集矩形坐标
        x_left = max(el1['x'], el2['x'])
        y_top = max(el1['y'], el2['y'])
        x_right = min(el1['x'] + el1['width'], el2['x'] + el2['width'])
        y_bottom = min(el1['y'] + el1['height'], el2['y'] + el2['height'])

        # 如果没有交集
        if x_right <= x_left or y_bottom <= y_top:
            return False

        # 交集面积
        inter_area = (x_right - x_left) * (y_bottom - y_top)

        # 两个元素面积
        area1 = el1['width'] * el1['height']
        area2 = el2['width'] * el2['height']

        # 判断交集面积是否占较小元素面积的比例 >= tolerance
        return inter_area / area1 >= tolerance and inter_area / area2 >= tolerance

    @staticmethod
    def get_vid_map(uielements: dict) -> dict:
        vid_map = {}
        def traverse(view):
            vid_map[view['id']] = view
            for subview in view.get('subviews', []):
                traverse(subview)
        traverse(uielements)
        return vid_map
    
    @staticmethod
    def is_ui_equal(uielements1: dict, uielements2: dict) -> bool:
        import copy
        uielements1 = copy.deepcopy(uielements1)
        uielements2 = copy.deepcopy(uielements2)

        def extract_and_remove_uied_nodes(node):
            """
            从树状结构中删除所有 type == 'UIED' 的节点，
            并将这些节点的子节点提升到父节点的 subviews 中。
            """

            uied_nodes = []

            if 'subviews' not in node or not node['subviews']:
                return []

            new_subviews = []
            for child in node['subviews']:
                # 递归处理子节点
                uied_nodes.extend(extract_and_remove_uied_nodes(child))
                if 'type' in child and child.get('type') == 'ElementByUIED':
                    # 将UIED节点的子节点上浮到父节点
                    new_subviews.extend(child.get('subviews', []))

                    uied_nodes.append(child)
                else:
                    new_subviews.append(child)

            node['subviews'] = new_subviews

            return uied_nodes

        uied1 = extract_and_remove_uied_nodes(uielements1)
        uied2 = extract_and_remove_uied_nodes(uielements2)

        def check_view(view1, view2):
            if any(k not in view1 or k not in view2 or view1[k] != view2[k] for k in ['id', 'type']):
                return False
            if len(view1['subviews']) != len(view2['subviews']):
                return False
            view1['subviews'].sort(key=lambda x: x['id'])
            view2['subviews'].sort(key=lambda x: x['id'])
            for subview1, subview2 in zip(view1['subviews'], view2['subviews']):
                if not check_view(subview1, subview2):
                    return False
            return True
        
        text_list1 = BaseUI.get_text_list(uielements1)
        text_list2 = BaseUI.get_text_list(uielements2)
        try:
            jaccard_similarity = len(set(text_list1) & set(text_list2)) / len(set(text_list1) | set(text_list2))
        except ZeroDivisionError:
            jaccard_similarity = 0.0
        diff_len = len(set(text_list1) - set(text_list2))

        from shapely.geometry import Polygon
        from shapely.ops import unary_union

        def boxes_to_polygons(boxes):
            """将 bounding_box 列表转为 shapely Polygon 列表"""
            polys = []
            for box in boxes:
                x, y, w, h = box["x"], box["y"], box["width"], box["height"]
                polys.append(Polygon([
                    (x, y),
                    (x + w, y),
                    (x + w, y + h),
                    (x, y + h)
                ]))
            return polys

        def compute_uied_similarity(boxes1, boxes2, threshold=0.8):
            """计算两棵树 UIED 区域的相似度"""
            if not boxes1 or not boxes2:
                return False

            polys1 = boxes_to_polygons(boxes1)
            polys2 = boxes_to_polygons(boxes2)

            union1 = unary_union(polys1)
            union2 = unary_union(polys2)
            intersection = union1.intersection(union2)

            area1 = union1.area
            area2 = union2.area
            inter_area = intersection.area

            if area1 == 0 or area2 == 0:
                return False

            ratio1 = inter_area / area1
            ratio2 = inter_area / area2
            similar = ratio1 >= threshold and ratio2 >= threshold

            return similar

        boxes1 = [BaseUI.get_frame(uied_element) for uied_element in uied1]
        boxes2 = [BaseUI.get_frame(uied_element) for uied_element in uied2]
        uied_similarity = compute_uied_similarity(boxes1, boxes2)

        return check_view(uielements1, uielements2) or (diff_len <= 4 and jaccard_similarity >= 0.8) or (len(boxes1) > 2 and len(boxes2) > 2 and uied_similarity)

    @staticmethod
    def element_align(target_element: dict, uielements: dict, minimal_alignment=2) -> int:
        def traverse_nodes(view):
            """递归生成树中所有节点"""
            yield view
            for subview in view.get('subviews', []):
                yield from traverse_nodes(subview)

        align_elements_count = 0

        # 收集 target_element 子树中所有节点的引用
        target_nodes = set(traverse_nodes(target_element))

        # 遍历 target_element 子树的每个节点
        for t_node in target_nodes:
            align_elements = 0
            # 遍历整棵树的所有节点
            for node in traverse_nodes(uielements):
                # 如果 node 不在 target 子树内
                if node not in target_nodes:
                    if BaseUI.is_align(t_node, node):
                        align_elements += 1
            if align_elements >= minimal_alignment:
                align_elements_count += 1

        return align_elements_count

    @staticmethod
    def get_text_list(uielements: dict) -> List[str]:
        text_list = set()
        def traverse(view):
            if not (view['type'] and view['type'] == "Application"):
                for key in ['text', 'name', 'value', 'label']:
                    if key in view:
                        val = view[key] if isinstance(view[key], str) else '/'.join(view[key])
                        text_list.add(view.get('type', '') + ': ' + key + '="' + val + '"')
            for subview in view.get('subviews', []):
                traverse(subview)
        traverse(uielements)
        return list(text_list)
    
    @staticmethod
    def element_selector(uielements: dict, name: str, value: str, strict: bool = False) -> dict|None:
        if (strict and uielements.get(name, '') == value) or (not strict and value.lower() in uielements.get(name, '').strip().lower()):
            return uielements
        for subview in uielements.get('subviews', []):
            if (sub := BaseUI.element_selector(subview, name, value, strict)) is not None:
                return sub
        return None
    
    @staticmethod
    def element_selector_custom(uielements: dict, func: Callable[[dict], bool]) -> dict|None:
        if func(uielements):
            return uielements
        for subview in uielements.get('subviews', []):
            if (sub := BaseUI.element_selector_custom(subview, func)) is not None:
                return sub
        return None
    


class AppiumUI(BaseUI):
    def __init__(self, vars):
        (self.gbscript, self.appium) = vars

    @classmethod
    def create_ui(cls, vars):
        """Create the UI object"""
        return cls(vars)
    
    @staticmethod
    def getui_raw(uist):
        """Get the UI elements by existing content"""
        uielements = uist['elements']
        screen_scale = uist['screenscale']

        VID = 0
        def getvid():
            nonlocal VID
            VID += 1
            return VID
        VID_MAP = {}
        
        def process_view(view):
            view['subviews'] = view.get('subviews', [])
            for key in list(view.keys()):
                if key in ['subviews']:
                    continue
                if key.startswith('@'):
                    view[key[1:]] = view.pop(key)
                else:
                    childs = []
                    def flap_list(l):
                        if isinstance(l, list):
                            for i in l:
                                flap_list(i)
                        else:
                            childs.append(l)
                    flap_list(view.pop(key))
                    view['subviews'].extend(childs)

            view['id'] = getvid()
            VID_MAP[view['id']] = view
            x = view.get('x', 0)
            y = view.get('y', 0)
            width = view.get('width', 0)
            height = view.get('height', 0)

            # Apply screen scale
            x *= screen_scale
            y *= screen_scale
            width *= screen_scale
            height *= screen_scale

            view['absolute_frame'] = {'x': x, 'y': y, 'width': width, 'height': height}

            # Calculate the absolute center point
            center_x = x + width / 2
            center_y = y + height / 2
            view['absolute_center'] = {'x': center_x, 'y': center_y}

            for key in ['name', 'value', 'label', 'text']:
                if key in view:
                    view[key] = str(view[key]) # Why are they possibly not a string?

            # Process subviews recursively
            for subview in view.get('subviews', []):
                process_view(subview)

        # Process the UI elements
        process_view(uielements)

        return uist, VID_MAP

    @time_consumed
    def getui(self):
        uielements = self.appium.get_page_source()
        screen_scale = 2 # self.gbscript.exports_sync.getscale() # TODO: is this stuck if app is not foreground?
        
        import json, xmljson
        from lxml.etree import fromstring
        xml = fromstring(uielements.encode('utf-8'))
        uielements = json.loads(json.dumps(xmljson.badgerfish.data(xml)))

        try:
            uielements = uielements['AppiumAUT']['XCUIElementTypeApplication']
        except:
            ...
        
        assert uielements['@type'] == 'XCUIElementTypeApplication', "Illegal Appium UI page source"
        
        uist = {'elements': uielements, 'screenscale': screen_scale}

        return self.getui_raw(uist)

    @staticmethod
    def get_semantic_like(uielements):
        allowed_types = {
            'Button', 'Image', 'StaticText', 'TextField', 'TextView', 'Switch', 'Element', 'OCRElement', 'ElementByUIED', 'ElementMarkedByHuman'
            # 如需可扩展：'Slider','Picker','Cell', ...
        }

        result_ids: List[int] = []

        def traverse(view: dict):
            # view 里 type 可能是字符串或列表（见post_process代码）
            vtype = view.get('type', '')
            # 统一处理：如果为列表则变为 set
            if isinstance(vtype, list):
                typeset = set(vtype)
            elif isinstance(vtype, str):
                typeset = {vtype}
            else:
                typeset = set()

            # 若有交集并且有 id 字段，则记录 id
            if typeset & allowed_types:
                # 尽量保证 id 为 int
                if 'id' in view:
                    frame = BaseUI.get_frame(view)
                    # 限制UI元素的大小，排除掉大小超过正常范围的元素
                    if 300 > frame['width'] > 25 or 600 > frame['height'] > 25:
                        if 0 <= frame['x'] <= 828 and 0 <= frame['y'] <= 1792 and 828 >= frame['width'] >= 0 and 0 <= frame['height'] <= 1792:
                            if 'visible' in view and view['visible']:
                                result_ids.append(int(view['id']))

            for sub in view.get('subviews', []):
                traverse(sub)

        traverse(uielements)

        return result_ids

    @staticmethod
    def get_text_from_elements(elements):
        texts = []
        for el in elements:
            if text := el.get('text', ''):
                texts.append(text)
            elif text := el.get('ocr_text', ''):
                texts.append(text)
            elif text := el.get('name', ''):
                texts.append(text)
            elif text := el.get('value', ''):
                texts.append(text)
            elif text := el.get('label', ''):
                texts.append(text)
            else:
                texts.append('')
        return texts

    @staticmethod
    def crop_elements_from_screenshots(elements, base64_screenshot: str) -> List[str]|str:
        """
        elements: list of element dicts (包含 id/type/... 等)，按照输入顺序逐个处理
        base64_screenshot: 原始整张截图的 base64 字符串（PNG/JPEG 等）
        返回：与 elements 长度相同顺序的 list[str]，每个为裁剪区域的 PNG base64 字符串。
        若某元素无法裁剪（越界、零尺寸等），对应位置返回空字符串 ''（也可以改为 None）。
        """
        single = False

        if isinstance(elements, str):
            single = True
            elements = [elements]

        # decode base64 -> PIL.Image
        import base64
        from PIL import Image
        import io
        img_bytes = base64.b64decode(base64_screenshot)
        image = Image.open(io.BytesIO(img_bytes)).convert('RGBA')  # 保留透明通道（如有）

        img_w, img_h = image.size
        outputs: List[str]|str = []

        for el in elements:
            # 获取 bounding box
            frame = AppiumUI.get_frame(el)

            x = int(round(frame.get('x', 0)))
            y = int(round(frame.get('y', 0)))
            w = int(round(frame.get('width', 0)))
            h = int(round(frame.get('height', 0)))

            # 忽略无效尺寸
            if w <= 0 or h <= 0:
                outputs.append('')
                continue

            # 计算与图片的交集（防止越界）
            left = max(0, x)
            upper = max(0, y)
            right = min(img_w, x + w)
            lower = min(img_h, y + h)

            # 若交集为空，则跳过
            if right <= left or lower <= upper:
                outputs.append('')
                continue

            # 裁剪并转换为 PNG base64
            cropped = image.crop((left, upper, right, lower))
            # 可选：如果只需 RGB，可以 convert('RGB')
            buf = io.BytesIO()
            cropped.save(buf, format='PNG')
            cropped_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            outputs.append(cropped_b64)

        if single:
            outputs = outputs[0]

        return outputs

    @staticmethod
    @time_consumed
    def post_process_ui(uielements, base64_screenshot):
        icon_like = []

        max_id = 0
        def extract_views(view, parent):
            subviews = []

            nonlocal max_id
            max_id = max(max_id, view['id'])

            FIELDS = ['id', 'type', 'name', 'absolute_frame', 'enabled', 'visible', 'accessible', 'index']
            myview = {k: view[k] for k in FIELDS if k in view}

            # for subview in view.get('subviews', []):
            #     if (subview := extract_views(subview, myview)) is not None:
            #         subviews.extend(subview) # Don't do any optimization (for benchmarking)
            
            myview['subviews'] = subviews

            #return [myview] # Don't do any optimization (for benchmarking)

            if myview.get('name', '') == 'splash_ad':
                myview['name'] = '' # This confuses LLM

            # Keep value only if different
            if 'value' in view and ('name' not in view or view['name'] != view['value']):
                myview['value'] = view['value']
            
            if abs(myview['absolute_frame']['width'] - myview['absolute_frame']['height']) < max(20, 0.2 * min(myview['absolute_frame']['width'], myview['absolute_frame']['height'])):
                if myview['type'] == 'XCUIElementTypeOther' or (myview['type'] == 'XCUIElementTypeStaticText' and not myview.get('value', '').strip() and not myview.get('name', '').strip()):
                    myview['type'] = 'XCUIElementTypeIcon'
                icon_like.append(myview)

            # Naming convention
            myview['frame_box'] = myview['absolute_frame']
            myview['_absolute_frame'] = myview['absolute_frame']
            #del myview['absolute_frame']
            x, y, width, height = myview['frame_box']['x'], myview['frame_box']['y'], myview['frame_box']['width'], myview['frame_box']['height']
            myview['frame_box'] = {'top_left': {'x': x, 'y': y}, 'right_bottom': {'x': x + width, 'y': y + height}, 'width': width, 'height': height}
            del myview['frame_box']

            for subview in view.get('subviews', []):
                if (subview := extract_views(subview, myview)) is not None:
                    subviews.extend(subview)

            # Suppress scroll bars
            if myview['type'] == 'XCUIElementTypeOther' and any(myview.get('name', '').startswith(prefix) for prefix in ['Vertical scroll bar, ', 'Horizontal scroll bar, ', '水平滚动条, ', '垂直滚动条, ']):
                return None
            
            # Suppress empty window
            if myview['type'] == 'XCUIElementTypeWindow' and len(subviews) == 0:
                return None

            # Remove overflow (TODO: not considering scrollable)
            if myview['_absolute_frame']['x'] > BaseUI.get_frame(uielements)['width'] or myview['_absolute_frame']['y'] > BaseUI.get_frame(uielements)['height']:
                return None

            empty_view = {'subviews': [], 'type': '', 'id': 0, 'index': 0, 'accessible': True, 'enabled': True, 'visible': True}

            # Suppress useless views
            if myview['type'] in ['XCUIElementTypeOther', 'XCUIElementTypeStaticText'] and parent and json.dumps(myview | empty_view, sort_keys=True, indent=0) == json.dumps(parent | empty_view, sort_keys=True, indent=0):
                return subviews
            
            empty_view = {'subviews': [], 'id': 0, 'index': 0, 'accessible': True, 'enabled': True, 'visible': True}

            # Suppress duplicate views
            if parent and json.dumps(myview | empty_view, sort_keys=True, indent=0) == json.dumps(parent | empty_view, sort_keys=True, indent=0):
                return subviews

            # Merge single child
            if myview['type'] == 'XCUIElementTypeOther' and len(subviews) == 1 and len(subviews[0].get('subviews', [])) == 0:
                return [myview | subviews[0]]
            
            # Suppress empty size
            if myview['absolute_frame']['width'] == 0 or myview['absolute_frame']['height'] == 0:
                return subviews

            return [myview]

        merged = extract_views(uielements, None)[0]

        #return merged, BaseUI.get_vid_map(merged) # Don't do any optimization (for benchmarking)

        # Suppress XCUIElementType and sort subviews by x-axis (left-to-right)
        def traverse(view):
            if view['type'].startswith('XCUIElementType'):
                view['type'] = view['type'][len('XCUIElementType'):]
            # Split some grids for sorting
            view['subviews'].sort(key=lambda x: (BaseUI.get_frame(x)['y'] // 100, BaseUI.get_frame(x)['x'] // 100, BaseUI.get_frame(x)['y'] // 10, BaseUI.get_frame(x)['x'] // 10, BaseUI.get_frame(x)['y'], BaseUI.get_frame(x)['x']))

            for subview in view.get('subviews', []):
                traverse(subview)

            # Remove subviews with same frame, keeping the first occurrence
            seen_frames = set([(0, 0, 0, 0)])
            filtered_subviews = []
            for subview in view['subviews']:
                frame_tuple = tuple(BaseUI.get_frame(subview).values())
                if frame_tuple not in seen_frames:
                    seen_frames.add(frame_tuple)
                    filtered_subviews.append(subview)
                else:
                    filtered_subviews.extend(subview['subviews'])
            view['subviews'] = filtered_subviews

            # Such merge doesn't make sense (ABANDONED)
            # # Find the maximum area of absolute frame
            # max_frame = None
            # max_area = 0
            # for subview in view['subviews']:
            #     frame = BaseUI.get_frame(subview)
            #     area = frame['width'] * frame['height']
            #     if max_frame is None or area > max_area:
            #         max_frame = frame
            #         max_area = area
            # # If all frames is within the max_frame
            # for subview in view['subviews']:
            #     frame = BaseUI.get_frame(subview)
            #     if frame['x'] < max_frame['x'] or frame['y'] < max_frame['y'] or frame['x'] + frame['width'] > max_frame['x'] + max_frame['width'] or frame['y'] + frame['height'] > max_frame['y'] + max_frame['height']:
            #         break
            # else:
            #     view['_mark_for_merge'] = False
        traverse(merged)

        # Suppress single child
        def traverse1(view):
            for subview in view.get('subviews', []):
                traverse1(subview)
            if len(view['subviews']) == 1 or view.get('_mark_for_merge', False):
                view.pop('_mark_for_merge', None)
                subviews = []
                for view1 in view['subviews']:
                    subviews.extend(view1['subviews'])
                    for key in ['type', 'name', 'value', 'label', 'text']:
                        if key in view1:
                            if key not in view:
                                view[key] = view1[key]
                            else:
                                if isinstance(view[key], str):
                                    view[key] = [view[key]]
                                if isinstance(view1[key], str):
                                    view1[key] = [view1[key]]
                                view[key] = view[key] + [v1k for v1k in view1[key] if v1k not in view[key]]
                    # Suppress duplicated type
                    if 'Other' in view['type'] and len(view['type']) > 1:
                        view['type'] = [type for type in view['type'] if type != 'Other']
                    if 'StaticText' in view['type'] and len(view['type']) > 1:
                        view['type'] = [type for type in view['type'] if type != 'StaticText']
                    if len(view['type']) == 1:
                        view['type'] = view['type'][0]
                view['subviews'] = subviews
        traverse1(merged)

        # Reordering helps LLM to learn the pattern
        max_id = 0
        VID = 0
        def traversex(view):
            nonlocal VID, max_id
            VID += 1
            view['_id'] = view['id']
            view['id'] = VID
            if view['type'] == 'Other':
                view['type'] = 'Frame' # attempt to make llm more focused
            max_id = max(max_id, VID)
            for key in ['type', 'name', 'value', 'label', 'text']:
                if key in view and isinstance(view[key], list):
                    view[key] = ' / '.join(view[key])
            for subview in view.get('subviews', []):
                traversex(subview)
        traversex(merged)

        # CNN classification for icons
        from cnn_cls import EfficientNetClient
        from utils import crop_image
        for view in icon_like:
            if '_id' in view:   # not being removed
                frame = BaseUI.get_frame(view)
                if frame['x'] + frame['width'] > BaseUI.get_frame(merged)['width'] or frame['y'] + frame['height'] > BaseUI.get_frame(merged)['height']:
                    # overflow, skip
                    continue
                icon = crop_image(base64_screenshot, frame['x'], frame['y'], frame['width'], frame['height'])
                icon_label, confidence = EfficientNetClient.classify(icon)
                if confidence > 0.5:
                    view['icon_label'] = icon_label

        # OCR to find any potential text in the screenshot
        from paddle_cls import PaddleOCRClient
        ocr = PaddleOCRClient()
        ocr_result = ocr.ocr(base64_screenshot)
        all_suitable_same = None
        ocr_result = [result for result in ocr_result if result[5] > 0.8]
        for result in ocr_result:
            (x, y, width, height, text, conf) = result
            suitable_element, depth, min_area = BaseUI.find_suitable_element(merged, x, y, width, height, need_visible=True)
            if suitable_element and all_suitable_same is not False:
                all_suitable_same = suitable_element if all_suitable_same is None or suitable_element['id'] == all_suitable_same['id'] else False
            text_normalize = lambda text: text.lower().replace(' ', '').replace('\n', '').replace('\t', '').replace('\r', '').strip()
            text_compare = lambda text1, text2: text_normalize(text1 if isinstance(text1, str) else ' '.join(text1)) in text_normalize(text2 if isinstance(text2, str) else ' '.join(text2))
            label_match = lambda element, text: any(k in element and text_compare(text, element[k]) for k in ['name', 'value', 'label', 'text'])
            if suitable_element and not label_match(suitable_element, text) and \
                    not any(label_match(child, text) for child in suitable_element.get('subviews', [])):
                # Match first level subviews (why child may be a bit larger than parent?)
                suitable_element['ocr_text'] = (suitable_element.get('ocr_text', '') + ' ' + text).strip()
        
        if all_suitable_same and len(ocr_result) > 1:
            # No UI element found, manually append
            all_suitable_same.pop('ocr_text', None)
            for result in ocr_result:
                (x, y, width, height, text, conf) = result
                max_id += 1
                all_suitable_same['subviews'].append({'ocr_text': text, 'absolute_frame': {'x': x, 'y': y, 'width': width, 'height': height}, 'id': max_id, 'type': 'OCRElement', 'subviews': [], 'visible': True})

        return merged, BaseUI.get_vid_map(merged)

    @DeprecationWarning
    @staticmethod
    @time_consumed
    def tree_append_element(uielements: dict, base64_screenshot: str, semantic_analyzed: SemanticView):
        # Match (or append) UI Element identified by LLM without id to UI Tree element.
        import base64
        image_data = base64.b64decode(base64_screenshot)

        from PIL import Image
        from io import BytesIO
        image = Image.open(BytesIO(image_data))

        full_width, full_height = image.size

        max_id = 0

        def get_maxid(view: dict):
            nonlocal max_id
            max_id = max(max_id, view['id'])
            for subview in view.get('subviews', []):
                get_maxid(subview)

        get_maxid(uielements)

        for elementwoid in semantic_analyzed.semantic_element_without_id:
            x = int(elementwoid.bounding_box.x_min / 1000 * full_width)
            y = int(elementwoid.bounding_box.y_min / 1000 * full_height)
            width = int((elementwoid.bounding_box.x_max - elementwoid.bounding_box.x_min) / 1000 * full_width)
            height = int((elementwoid.bounding_box.y_max - elementwoid.bounding_box.y_min) / 1000 * full_height)
            new_rec = {'x': x, 'y': y, 'width': width, 'height': height}

            suitable_element, depth, min_area = BaseUI.find_suitable_element(uielements, x, y, width, height, tolerance=0.3)

            if suitable_element:
                suitable_element_id = 0
                if BaseUI.almost_overlap(suitable_element['absolute_frame'], new_rec):
                    # Append attribute to the suitable_element
                    # suitable_element['description'] += "Additionally, LLM previously considered the current UI element or interface to be semantically related to ad UI dark patterns."
                    # suitable_element['semantic_type'] = elementwoid.type
                    # suitable_element['evidence'] = elementwoid.evidence
                    suitable_element_id = suitable_element['id']
                else:
                    max_id += 1
                    suitable_element['subviews'].append({
                        'id': max_id, 'absolute_frame': new_rec, 'type': 'LLMElement', 'subviews': [],
                        'description': "LLM previously considered the current UI element or interface to be semantically related to ad UI dark patterns.",
                        'semantic_type': elementwoid.type, 'evidence': elementwoid.evidence,
                    })
                    suitable_element_id = max_id

                elementwoid.id = suitable_element_id

        # TBD: We can no longer reorder or modify the IDs of elements in the UI Tree, as this would cause misalignment with analyzed_ui and semantic_analyzed.

        return uielements, BaseUI.get_vid_map(uielements)

    @staticmethod
    @time_consumed
    def tree_supply(uielements: dict, base64_screenshot: str):
        import base64
        image_data = base64.b64decode(base64_screenshot)

        from PIL import Image
        from io import BytesIO
        image = Image.open(BytesIO(image_data))
        # full_width, full_height = image.size

        max_id = 0
        def get_maxid(view: dict):
            nonlocal max_id
            max_id = max(max_id, view['id'])
            for subview in view.get('subviews', []):
                get_maxid(subview)
        get_maxid(uielements)

        # Simple template matching for detecting rectangle, circle, triangle, and "X"
        matching_results = BaseUI.element_template_matching(base64_screenshot)

        for element in matching_results:
            shape, x, y, width, height = element
            new_rec = {'x': x, 'y': y, 'width': width, 'height': height}

            suitable_element, depth, min_area = BaseUI.find_suitable_element(uielements, x, y, width, height, tolerance=0.2, need_visible=True)

            if suitable_element:
                if not BaseUI.almost_overlap(suitable_element['absolute_frame'], new_rec):
                    max_id += 1
                    new_subview = {
                        'id': max_id, 'absolute_frame': new_rec, 'type': 'ElementByUIED', 'subviews': [],
                        'class': f"Template_{shape}",
                        'visible': True, 'accessible': True, 'enabled': True
                    }

                    new_subview['subviews'] = [subview for subview in suitable_element['subviews'] if BaseUI.is_in_element(subview, new_subview)]
                    suitable_element['subviews'] = [subview for subview in suitable_element['subviews'] if not BaseUI.is_in_element(subview, new_subview)]
                    suitable_element['subviews'].append(new_subview)

        # UIED
        from LayoutCoder_develop.run_single import single_process
        import tempfile
        from os.path import join, basename, splitext

        UIED_CACHED_DIRS = "./LayoutCoder_develop"
        file_name = None
        with tempfile.NamedTemporaryFile(dir=join(UIED_CACHED_DIRS, "data"), delete=False, suffix=".png") as f:
            image.save(f, format="PNG")
            file_name = f.name
        single_process(file_name, join(UIED_CACHED_DIRS, 'results', splitext(basename(file_name))[0]))
        # Temporarily use UIED's raw detection result, maybe use LayoutCoder's layout future ...
        with open(join(UIED_CACHED_DIRS, 'results', splitext(basename(file_name))[0], 'merge', splitext(basename(file_name))[0] + '.json'), 'rb') as f:
            layout_coder_detection = json.load(f)

        for compo in layout_coder_detection["compos"]:
            x = compo["position"]["column_min"]
            y = compo["position"]["row_min"]
            width = compo["width"]
            height = compo["height"]
            new_rec = {'x': x, 'y': y, 'width': width, 'height': height}

            suitable_element, depth, min_area = BaseUI.find_suitable_element(uielements, x, y, width, height, tolerance=0.2, need_visible=True)

            if suitable_element:
                # suitable_element_id = 0
                if BaseUI.almost_overlap(suitable_element['absolute_frame'], new_rec):
                    # Append attribute to the suitable_element
                    # suitable_element['description'] += "Additionally, LLM previously considered the current UI element or interface to be semantically related to ad UI dark patterns."
                    # suitable_element['semantic_type'] = elementwoid.type
                    # suitable_element['evidence'] = elementwoid.evidence
                    # suitable_element_id = suitable_element['id']
                    pass
                else:
                    if compo["class"] != "Block":
                        max_id += 1
                        new_subview = {
                            'id': max_id, 'absolute_frame': new_rec, 'type': 'ElementByUIED', 'subviews': [],
                            'class': "Component" if compo["class"] == "Compo" else "Text",
                            'visible': True, 'accessible': True, 'enabled': True
                        }
                        if compo["class"] == "Text":
                            new_subview['text'] = compo["text_content"]

                        new_subview['subviews'] = [subview for subview in suitable_element['subviews'] if BaseUI.is_in_element(subview, new_subview)]
                        suitable_element['subviews'] = [subview for subview in suitable_element['subviews'] if not BaseUI.is_in_element(subview, new_subview)]
                        suitable_element['subviews'].append(new_subview)
                        # suitable_element_id = max_id
                    else:
                        pass

                # elementwoid.id = suitable_element_id

        # TBD: We can no longer reorder or modify the IDs of elements in the UI Tree, as this would cause misalignment within analyzed_ui and semantic_analyzed.
        # Warning: Only use below code when you are sure that analyze_screenshot & semantic_analyze_screenshot will be called after tree_supply but not before this function
        def reorder_by_position(view: dict):
            if 'subviews' in view and view['subviews']:
                view['subviews'].sort(
                    key=lambda x: (
                        BaseUI.get_frame(x)['y'] // 100,
                        BaseUI.get_frame(x)['x'] // 100,
                        BaseUI.get_frame(x)['y'] // 10,
                        BaseUI.get_frame(x)['x'] // 10,
                        BaseUI.get_frame(x)['y'],
                        BaseUI.get_frame(x)['x'],
                    )
                )
                for subview in view['subviews']:
                    reorder_by_position(subview)

        reorder_by_position(uielements)

        new_id = 0

        def reassign_id(view: dict):
            nonlocal new_id
            new_id += 1
            view['id'] = new_id
            for subview in view.get('subviews', []):
                reassign_id(subview)

        reassign_id(uielements)

        return uielements, BaseUI.get_vid_map(uielements)


# Deprecated
class FridaUI(BaseUI):
    def __init__(self, vars):
        (self.gbscript, ) = vars

    @classmethod
    def create_ui(cls, vars):
        return cls(vars)

    def getui(self):
        uist = self.gbscript.exports_sync.getui()
        
        uielements = uist['elements']
        screen_scale = uist['screenscale']

        VID = 0
        def getvid():
            nonlocal VID
            VID += 1
            return VID
        VID_MAP = {}
        
        def process_view(view):
            view['id'] = getvid()
            VID_MAP[view['id']] = view
            frame = view.get('absolute_frame', {})
            x = int(frame.get('x', 0))
            y = int(frame.get('y', 0))
            width = int(frame.get('width', 0))
            height = int(frame.get('height', 0))

            # Apply screen scale
            x *= screen_scale
            y *= screen_scale
            width *= screen_scale
            height *= screen_scale

            view['absolute_frame'] = {'x': x, 'y': y, 'width': width, 'height': height}

            # Calculate the absolute center point
            center_x = x + width // 2
            center_y = y + height // 2
            view['absolute_center'] = {'x': center_x, 'y': center_y}

            # Check if the view is possibly visible and clickable
            alpha = view.get('alpha', 1.0)
            hidden = view.get('hidden', False)
            userInteractionEnabled = view.get('userInteractionEnabled', True)
            opaque = view.get('opaque', True)

            # Convert properties to appropriate types
            alpha = float(alpha) if isinstance(alpha, (int, float, str)) else 1.0
            hidden = bool(hidden) if isinstance(hidden, bool) else False
            userInteractionEnabled = bool(userInteractionEnabled) if isinstance(userInteractionEnabled, bool) else True
            opaque = bool(opaque) if isinstance(opaque, bool) else True

            # Determine visibility and clickability
            is_visible = (alpha > 0) and not hidden and opaque
            is_clickable = is_visible and userInteractionEnabled

            view['is_visible'] = is_visible
            view['is_clickable'] = is_clickable

            # Process subviews recursively
            for subview in view.get('subviews', []):
                process_view(subview)

        # Process the UI elements
        process_view(uielements)

        return uist, VID_MAP
    
    @staticmethod
    def post_process_ui(uielements, base64_screenshot):
        def extract_views(view):
            subviews = []

            FIELDS = ['id', 'class', 'text', 'absolute_frame', 'index', 'type', 'name', 'value']#, 'alpha', 'userInteractionEnabled', 'hidden', 'opaque']
            myview = {k: view[k] for k in FIELDS if k in view}

            for subview in view.get('subviews', []):
                subviews.append(extract_views(subview))

            myview['subviews'] = subviews
            
            return myview
        
        uielements = extract_views(uielements)

        return uielements, BaseUI.get_vid_map(uielements)


