from gpt_cls import DarkPatternRecording, UIRecording, ActionRecording, UIElementRecording, SemanticUIElementType, \
    SemanticView, RAGView
from collections import defaultdict
from pathlib import Path
from google import genai
import os
import time
import json
import math
import uuid
import torch
import base64
from io import BytesIO
from PIL import Image
import copy
import cv2
import numpy as np
import random
from typing import get_args
import Levenshtein
import logging

from ui_cls import BaseUI

from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
from langchain_chroma import Chroma
from langchain_core.embeddings.embeddings import Embeddings

from apted import APTED, Config

from resources_for_run.keys import GeminiKey
import random

logger = logging.getLogger(__name__)

FORCE_REMBD = False
NORMAL_ELEMENT_PATH = "db/normal_elements"


class TreeNode:
    def __init__(self, label, children=None, data=None):
        self.label = label
        self.children = children if children else []
        self.data = data if data else {}  # store arbitrary UI properties

    def __iter__(self):
        return iter(self.children)


class UICost(Config):
    def delete(self, node):
        # deletion cost
        return 1

    def insert(self, node):
        # insertion cost
        return 1

    def rename(self, n1, n2):
        """
        n1, n2 are Node objects.

        Here you can read anything from n1.data / n2.data.
        Implement your own multi-attribute similarity (type, text embedding etc).
        """
        # Example: simple similarity based on type + text

        type_cost = 0 if n1.label == n2.label else 1

        # # text similarity; just a stub here
        # text1 = n1.data.get("text", "")
        # text2 = n2.data.get("text", "")
        # text_cost = 1 - Levenshtein.jaro_winkler(text1, text2) # 可以用Levenshtein距离替代，或者sentence-transformer等超快的MiniLM

        # combined similarity
        cost = type_cost  # + text_cost

        return cost


class GeminiEmbedding(Embeddings):
    def __init__(self):
        self.img_high_similarity_threshold = 0.9
        self.img_low_similarity_threshold = 0.6
        self.client = genai.Client(
            api_key=random.choice(GeminiKey),
        )

    @DeprecationWarning
    def embed_documents(self, vectors: list[list[float]]) -> list[list[float]]:
        """Embed a list of texts (documents)."""
        pass

    @DeprecationWarning
    def embed_query(self, image: Image.Image):
        """Embed a single query string."""
        pass

    def embed_image(self, images: str | list[str]):
        ret_single = False
        if isinstance(images, str):
            ret_single = True
            images = [images]

        all_results = []
        for i in range(0, len(images), len(images)):
            batch = images[i:i + len(images)]

            messages = [
                {
                    'parts': [
                        {
                            "inlineData": {
                                "mimeType": "image/png",
                                "data": image,
                            },
                        }
                    ]
                }
            for image in batch]
            while True:
                try:
                    result = self.client.models.embed_content(
                        model='gemini-embedding-2-preview',
                        contents=messages
                    )
                    break
                except Exception as e:
                    logger.warning(f"Failed to embed screenshot: {e}, retrying...")
                    time.sleep(5)

            all_results.extend([embd.values for embd in result.embeddings])

        return all_results if not ret_single else all_results[0]

    def embed_text(self, texts: str | list[str]):
        ret_single = False
        if isinstance(texts, str):
            ret_single = True
            texts = [texts]

        messages = [
            {
                'parts': [
                    {
                        'text': text
                    }
                ]
            }
        for text in texts]
        while True:
            try:
                results = self.client.models.embed_content(
                    model='gemini-embedding-2-preview',
                    contents=messages
                )
                break
            except Exception as e:
                logger.warning(f"Failed to embed text: {e}, retrying...")
                time.sleep(5)

        return [embd.values for embd in results.embeddings] if not ret_single else results.embeddings[0].values


class LLMRAG:
    @staticmethod
    def base64_to_pil(base64_str: str) -> Image.Image:
        if base64_str.startswith("data:image"):
            base64_str = base64_str.split(",")[1]

            # 解码并转为二进制流
        image_data = base64.b64decode(base64_str)
        image = Image.open(BytesIO(image_data)).convert("RGB")  # 转为 RGB 格式
        return image

    def __init__(self, database_dir):
        self.retriever = None
        self.element_retriever = None
        self.all_element_retriever = None
        self.embedding_function = GeminiEmbedding()
        self.recordings = []
        self.normal_elements = []
        self.element_descriptions = [
            "'Ad Removal Element': Any UI element that initiates the process of getting an ad-free version of the app.",
            "'Close Button': Any clickable UI element that can dismiss, skip, or otherwise exit the current interface or some frame.",
            "'Purchase Button': Any element that initiates a purchase using currency in the real world when pressed.",
            "'Reward Element': Any element that explicitly offers, grants, or confirms a reward (e.g., coins, currency, functions of apps or games) to the user.",
            "'Voluntary Ad Trigger Element': an interactive UI element (e.g., a button, icon, or text link) that, when clicked by the user, initiates the play of an ad."
            # "'Normal Element': Other elements that do not belong to any of the following types: 1. Ad Removal Element: Starts the process of obtaining an ad-free version. 2. Close Button: Dismisses, skips, or exits the current interface. 3. Purchase Button: Initiates a real-money purchase. 4. Reward Element: Explicitly offers or grants a reward to the user.5. Voluntary Ad Trigger Element: An element that the user clicks to start an ad."
            "'Normal Element': Any element that can not be classified as an ad, and does not suggest any service about advertising."
        ]
        # descriptions_embeddings = [[] for i in range(len(self.element_descriptions))]
        descriptions_embeddings = self.embedding_function.embed_text(self.element_descriptions)
        element_types = ["Ad Removal Element", "Close Button", "Purchase Interface", "Reward Element", "Voluntary Ad Trigger Element", "Normal Element"]
        self.descriptions_embeddings = {
            element_type: description_embedding
        for element_type, description_embedding in zip(element_types, descriptions_embeddings)}

        from paddle_cls import PaddleOCRClient
        self.ocr = PaddleOCRClient()

        self.load_database(database_dir)

    def create_image_database(self, images_from, images_order, images_info, images_embedding, collection_name):
        def create_multi_vector_retriever(vectorstore, images_from, images_order, images_embedding, images_info):
            """
            Create retriever that indexes summaries, but returns raw videos or images
            """

            # Initialize the storage layer
            store = InMemoryStore()
            id_key = "doc_id"

            # Create the multi-vector retriever
            retriever = MultiVectorRetriever(
                vectorstore=vectorstore,
                docstore=store,
                id_key=id_key,
            )

            def add_documents(retriever, doc_from, doc_order, doc_embeddings, doc_contents):
                doc_ids = [str(uuid.uuid4()) for _ in doc_contents]

                retriever.vectorstore._collection.upsert(
                    ids=doc_ids,
                    embeddings=doc_embeddings,
                    documents=doc_contents,
                    metadatas=[{"doc_id": doc_ids[i], "from": doc_from[i], "order": doc_order[i]} for i in
                               range(len(doc_ids))]
                )

                # 把原始文档放入 docstore
                retriever.docstore.mset(list(zip(doc_ids, doc_contents)))

            # Add video into the database
            if images_info:
                add_documents(retriever, images_from, images_order, images_embedding, images_info)

            return retriever

        # The vectorstore to use to index the summaries
        vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=self.embedding_function,
            collection_metadata={"hnsw:space": "cosine"},
        )

        # Create retriever
        retriever_multi_vector = create_multi_vector_retriever(
            vectorstore,
            images_from,
            images_order,
            images_embedding,
            images_info,
        )

        return retriever_multi_vector

    def paddle_ocr(self, base64_screenshot: str):
        # OCR to find any potential text in the screenshot
        ocr_result = self.ocr.ocr(base64_screenshot)
        ocr_result = [result for result in ocr_result if result[5] > 0.8]
        return ocr_result

    @staticmethod
    def append_ocr_to_viewhierarchy(vh: dict, ocr_results: list):
        # Reordering helps LLM to learn the pattern
        max_id = 0

        def traversex(view):
            nonlocal max_id
            max_id = max(max_id, view['id'])
            for subview in view.get('subviews', []):
                traversex(subview)

        traversex(vh)

        for result in ocr_results:
            (x, y, width, height, text, conf) = result
            suitable_element, depth, min_area = BaseUI.find_suitable_element(vh, x, y, width, height)
            text_normalize = lambda text: text.lower().replace(' ', '').replace('\n', '').replace('\t', '').replace(
                '\r', '').strip()
            text_compare = lambda text1, text2: text_normalize(
                text1 if isinstance(text1, str) else ' '.join(text1)) in text_normalize(
                text2 if isinstance(text2, str) else ' '.join(text2))
            label_match = lambda element, text: any(
                k in element and text_compare(text, element[k]) for k in ['text', 'ocr_text'])
            if suitable_element and not label_match(suitable_element, text) and not any(
                    label_match(child, text) for child in suitable_element.get('subviews', [])):
                # Match first level subviews (why child may be a bit larger than parent?)
                suitable_element['ocr_text'] = (suitable_element.get('ocr_text', '') + ' ' + text).strip()

    @staticmethod
    def append_ocr_to_recording(uielements: list[UIElementRecording], ocr_results: list):
        for result in ocr_results:
            (x, y, width, height, text, conf) = result
            for uielement in uielements:
                bbox = uielement.bbox
                if x >= bbox[0] and y >= bbox[1] and x + width <= bbox[2] and y + height <= bbox[3]:
                    uielement.ocr = (uielement.ocr + ' ' + text).strip()
        for uielement in uielements:
            uielement.ocr_done = True

    @staticmethod
    def rebuild_viewhierarchy(ui: UIRecording):
        nodes = []

        def bbox_to_absolute_frame(bbox: tuple[int, int, int, int]) -> dict:
            x1, y1, x2, y2 = bbox
            return {'x': x1, 'y': y1, 'width': x2 - x1, 'height': y2 - y1}

        # 1. 包装 uielements，id 从 1 开始
        for idx, element in enumerate(ui.uielements, start=1):
            node = {
                'id': idx,
                'subviews': [],
                'type': 'ElementMarkedByHuman',
                'semantic_type': element.ui_type,
                'absolute_frame': bbox_to_absolute_frame(element.bbox),
                'accessible': True,
                'visible': True,
                'enabled': True,
            }
            if element.ocr_done and element.ocr:
                node['ocr_text'] = element.ocr

            nodes.append(node)

        # 2. 构建树
        # 找到每个节点的父节点，如果没有父节点就归根节点
        root_level_nodes = []
        for child in nodes:
            parent_found = False
            for potential_parent in nodes:
                if child['id'] == potential_parent['id']:
                    continue
                if BaseUI.is_in_element(child, potential_parent):
                    potential_parent['subviews'].append(child)
                    parent_found = True
                    break
            if not parent_found:
                root_level_nodes.append(child)

        img_bytes = base64.b64decode(ui.screenshot)
        image = Image.open(BytesIO(img_bytes))
        img_width, img_height = image.size

        # 3. 构建固定根节点
        # 根节点 id = 0
        root_node = {
            'id': 0,
            'subviews': root_level_nodes,
            'type': 'ElementMarkedByHuman',
            'semantic_type': ui.ui_type,
            'absolute_frame': {'x': 0, 'y': 0, 'width': img_width, 'height': img_height}
        }

        return root_node

    @staticmethod
    def rebuild_outlines(screenshot, viewhierarchy=None):
        outlines = []

        from LayoutCoder_develop.run_single import single_process
        import tempfile
        from os.path import join, basename, splitext

        # file_name = None
        # with tempfile.NamedTemporaryFile(dir="./LayoutCoder_develop/data", delete=False, suffix=".png") as f:
        #     Image.open(BytesIO(base64.b64decode(screenshot))).save(f, format="PNG")
        #     file_name = f.name
        # single_process(file_name, join('./LayoutCoder_develop/results', splitext(basename(file_name))[0]))
        # # Temporarily use UIED's raw detection result, maybe use LayoutCoder's layout future...
        # with open(join('./LayoutCoder_develop/results', splitext(basename(file_name))[0], 'merge',
        #                splitext(basename(file_name))[0] + '.json'), 'rb') as f:
        #     layout_coder_detection = json.load(f)

        UIED_CACHED_DIRS = "./LayoutCoder_develop"
        file_name = None
        with tempfile.NamedTemporaryFile(dir=join(UIED_CACHED_DIRS, "data"), delete=False, suffix=".png") as f:
            Image.open(BytesIO(base64.b64decode(screenshot))).save(f, format="PNG")
            file_name = f.name
        single_process(file_name, join(UIED_CACHED_DIRS, 'results', splitext(basename(file_name))[0]))
        # Temporarily use UIED's raw detection result, maybe use LayoutCoder's layout future ...
        with open(join(UIED_CACHED_DIRS, 'results', splitext(basename(file_name))[0], 'merge',
                       splitext(basename(file_name))[0] + '.json'), 'rb') as f:
            layout_coder_detection = json.load(f)

        for compo in layout_coder_detection["compos"]:
            x_min = compo["position"]["column_min"]
            y_min = compo["position"]["row_min"]
            x_max = x_min + compo["width"]
            y_max = y_min + compo["height"]
            bbox = [x_min, y_min, x_max, y_max]

            if x_max - x_min > 25 and y_max - y_min > 25:
                outlines.append({
                    'bbox': bbox,
                    'type': 'ElementByUIED-Compo' if compo["class"] == "Compo" else 'ElementByUIED-Text',
                })

        useful_element_type = ['WebView']

        def traverse_extract_vh(view):
            nonlocal outlines, useful_element_type

            if view.get('type', '') in useful_element_type:
                frame = view['absolute_frame']
                outlines.append({
                    'bbox': [frame['x'], frame['y'], frame['x'] + frame['width'], frame['y'] + frame['height']],
                    'type': 'ElementByUIED-Compo'
                })

            for subview in view.get('subviews', []):
                traverse_extract_vh(subview)

            return

        if viewhierarchy:
            traverse_extract_vh(viewhierarchy)

        outlines = sorted(outlines, key=lambda e: (e["bbox"][1], e["bbox"][0]))

        return outlines

    @staticmethod
    def extract_outlines(view_hierarchy):
        outlines = []
        frame_types = ['WebView', 'Frame']

        def traversex(view):
            nonlocal outlines, frame_types

            if not view.get('visible', False):
                return

            frame = view['absolute_frame']
            x1 = frame['x']
            y1 = frame['y']
            x2 = x1 + frame['width']
            y2 = y1 + frame['height']

            view_type = view.get('type', '')
            subviews = view.get('subviews', [])

            is_frame_type = view_type in frame_types
            is_button = view_type == 'Button'
            is_leaf = len(subviews) == 0
            is_small = frame['width'] <= 30 and frame['height'] <= 30

            # Step 1: 框架类型 —— 加入 outline，但继续递归
            if is_frame_type:
                outlines.append({
                    'bbox': [x1, y1, x2, y2],
                    'type': view_type
                })

                for subview in subviews:
                    traversex(subview)
                return

            # Step 2: 叶节点 或 足够小 —— 加入 outline，终止递归
            if is_leaf or is_small or is_button:
                outlines.append({
                    'bbox': [x1, y1, x2, y2],
                    'type': view_type
                })
                return

            # Step 3: 其他情况 —— 不加入 outline，继续递归
            for subview in subviews:
                traversex(subview)

        traversex(view_hierarchy)

        outlines = sorted(outlines, key=lambda e: (e["bbox"][1], e["bbox"][0]))

        return outlines

    def load_database(self, database_dir):
        embd_func = self.embedding_function.__class__.__name__

        root_dir = Path(database_dir)

        screenshots_info = []
        screenshots_embeddings = []
        screenshots_from = []
        screenshots_order = []

        all_element_info = []
        all_element_embeddings = []
        all_element_from = []
        all_element_order = []

        uielements = {}
        for element in list(get_args(SemanticUIElementType)):
            uielements[str(element)] = {'element_info': [], 'element_embedding': [], 'element_from': [], 'element_order': []}
        uielements["Normal Element"] = {'element_info': [], 'element_embedding': [], 'element_from': [], 'element_order': []}

        element_unique_id = 0
        ui_unique_id = 0
        dpidx = 0
        for subdir in root_dir.iterdir():
            # print("Processing: ", subdir.name)
            if subdir.is_dir():
                instance_path = subdir / "instance_revision.json"
                if not instance_path.exists():
                    instance_path = subdir / "instance.json"
                # instance_path = subdir / "instance.json"
                if instance_path.exists():
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
                    self.recordings.append(dprecord)
                    # print(f'Processing sequence {len(self.recordings)}')

                    order = 0
                    instance_map = {}
                    for ui_or_action in dprecord.ui_action_sequence:
                        if isinstance(ui_or_action, UIRecording):
                            if not ui_or_action.view_hierarchy and not ui_or_action.pruned_vh:
                                ui_or_action.pruned_vh = self.rebuild_viewhierarchy(ui_or_action)
                            elif ui_or_action.view_hierarchy and ui_or_action.semantic_analyzed:
                                ui_or_action.pruned_vh = self.prune_viewhierarchy(ui_or_action.view_hierarchy,
                                                                                  ui_or_action.semantic_analyzed)
                            assert ui_or_action.pruned_vh
                            vh = ui_or_action.pruned_vh

                            # ui_or_action.outline = self.rebuild_outlines(ui_or_action.screenshot)
                            if not ui_or_action.view_hierarchy and not ui_or_action.outline:
                                ui_or_action.outline = self.rebuild_outlines(ui_or_action.screenshot)
                            elif ui_or_action.view_hierarchy:
                                ui_or_action.outline = self.rebuild_outlines(ui_or_action.screenshot,
                                                                             ui_or_action.view_hierarchy)

                            if not FORCE_REMBD and os.path.exists(
                                    str(subdir / ui_or_action.path) + f'.llmrag.uiss.{embd_func}.embedding'):
                                embedding = torch.load(str(subdir / ui_or_action.path) + f'.llmrag.uiss.{embd_func}.embedding',
                                                       weights_only=False)
                            else:
                                embedding = self.embedding_function.embed_image(ui_or_action.screenshot)
                                torch.save(embedding, str(subdir / ui_or_action.path) + f'.llmrag.uiss.{embd_func}.embedding')

                            screenshots_info.append(json.dumps({"screenshot": ui_or_action.screenshot}))
                            screenshots_embeddings.append(embedding)
                            screenshots_from.append(str(subdir))
                            screenshots_order.append(order)

                            element_order = 0
                            element_map = {}
                            for element in ui_or_action.uielements:
                                if not FORCE_REMBD and os.path.exists(
                                        str(subdir / element.path) + f'.llmrag.elm.{embd_func}.image.embedding'):
                                    embedding = torch.load(
                                        str(subdir / element.path) + f'.llmrag.elm.{embd_func}.image.embedding',
                                        weights_only=False)
                                else:
                                    embedding = self.embedding_function.embed_image(element.screenshot)
                                    torch.save(embedding,
                                               str(subdir / element.path) + f'.llmrag.elm.{embd_func}.image.embedding')

                                uielements[element.ui_type]['element_info'].append(json.dumps(
                                    {'screenshot': element.screenshot, 'ui_type': element.ui_type,
                                     'unique_id': element_unique_id}))
                                uielements[element.ui_type]['element_embedding'].append(embedding)
                                uielements[element.ui_type]['element_from'].append((str(subdir), order))
                                uielements[element.ui_type]['element_order'].append(element_order)

                                all_element_info.append(json.dumps({'ui_type': element.ui_type, 'screenshot': element.screenshot, 'unique_id': element_unique_id}))
                                all_element_embeddings.append(embedding)
                                all_element_from.append((str(subdir), order))
                                all_element_order.append(element_order)

                                element_map[element_order] = {
                                    'ui_type': element.ui_type,
                                    'screenshot': element.screenshot,
                                    'similar': False,
                                }

                                element_order += 1

                                element_unique_id += 1

                            instance_map[order] = {
                                'similar': False,
                                'screenshot': ui_or_action.screenshot,
                                'elements': element_map,
                            }

                            ui_unique_id += 1

                        order += 1

                    open(str(subdir / "instance_revision.json"), "w", encoding="utf-8").write(dprecord.model_dump_json(indent=4))

                    dpidx += 1

        # pictures = 0
        for filename in os.listdir(NORMAL_ELEMENT_PATH):
            file_path = os.path.join(NORMAL_ELEMENT_PATH, filename)
            if os.path.isfile(file_path) and filename.lower().endswith((".png", ".jpg", ".jpeg")):
                # print(f'Processing {filename}, {pictures}')
                # pictures += 1

                with open(file_path, "rb") as f:
                    screenshot = base64.b64encode(f.read()).decode("utf-8")

                if not FORCE_REMBD and os.path.exists(file_path + f'.llmrag.elm.{embd_func}.image.embedding'):
                    embedding = torch.load(file_path + f'.llmrag.elm.{embd_func}.image.embedding', weights_only=False)
                else:
                    embedding = self.embedding_function.embed_image(screenshot)
                    torch.save(embedding, file_path + f'.llmrag.elm.{embd_func}.image.embedding')

                uielements["Normal Element"]['element_info'].append(json.dumps({'screenshot': screenshot, 'ui_type': "Normal Element", 'unique_id': element_unique_id}))
                uielements["Normal Element"]['element_embedding'].append(embedding)
                uielements["Normal Element"]['element_from'].append(file_path)
                uielements["Normal Element"]['element_order'].append(-1)

                all_element_info.append(json.dumps({'ui_type': "Normal Element", 'screenshot': screenshot, 'unique_id': element_unique_id}))
                all_element_embeddings.append(embedding)
                all_element_from.append(file_path)
                all_element_order.append(-1)

                element_unique_id += 1

        self.retriever = self.create_image_database(screenshots_from, screenshots_order, screenshots_info, screenshots_embeddings, "screenshots_database")

        self.element_retriever = {}
        for element in uielements.keys():
            self.element_retriever[element] = self.create_image_database(
                [json.dumps(elm) for elm in uielements[element]['element_from']], uielements[element]['element_order'],
                uielements[element]['element_info'], uielements[element]['element_embedding'],
                f"{element.replace(' ', '_')}_database")

        self.all_element_retriever = self.create_image_database([json.dumps(elm) for elm in all_element_from],
                                                                all_element_order, all_element_info,
                                                                all_element_embeddings, "all_element_database")

    def query_element(self, element_sss, element_texts, top_k=3):
        if not element_sss:
            return []

        def cosine_similarity(a, b):
            a = np.array(a)
            b = np.array(b)
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

        embedding_fn = self.embedding_function
        if isinstance(element_sss, str):
            element_sss = [element_sss]

        image_query_embeddings = embedding_fn.embed_image(element_sss)

        categories = [{} for _ in range(len(image_query_embeddings))]
        for index, image_embedding in enumerate(image_query_embeddings):
            score = {}
            for element_type, description_embedding in self.descriptions_embeddings.items():
                score[element_type] = cosine_similarity(image_embedding, description_embedding)
            categories[index] = dict(sorted(score.items(), key=lambda x: x[1], reverse=True))

        image_query_results = self.all_element_retriever.vectorstore._collection.query(
            query_embeddings=image_query_embeddings,
            n_results=top_k,
            include=['documents', 'distances', 'metadatas']
        )

        new_dir = []

        results = []
        for idx in range(len(image_query_embeddings)):
            single_results = []

            new_dir.append(os.path.join("output\\elements", str(uuid.uuid4())))
            os.makedirs(new_dir[idx])
            self.base64_to_pil(element_sss[idx]).save(new_dir[idx] + "\\original_ss.png")

            top_1_category = next(iter(categories[idx].keys()))
            top_1_query_results = self.element_retriever[top_1_category].vectorstore._collection.query(
                query_embeddings=image_query_embeddings[idx],
                n_results=top_k,
                include=['documents', 'distances', 'metadatas']
            )
            for doc, dist, meta in zip(top_1_query_results['documents'][0], top_1_query_results['distances'][0],
                                       top_1_query_results['metadatas'][0]):
                info = json.loads(doc)
                cosine_sim = 1 - dist
                if cosine_sim >= embedding_fn.img_low_similarity_threshold and len(single_results) < top_k / 3:
                    self.base64_to_pil(info['screenshot']).save(new_dir[idx] + f"\\img_sim_{str(1 - dist)}_type_{info['ui_type']}_top_1.png")
                    single_results.append(UIElementRecording(
                        ui_type=info['ui_type'],
                        screenshot=info['screenshot']
                    ))
                else:
                    break

            from itertools import islice
            top_2_category = next(islice(categories[idx].keys(), 1, None))
            top_2_query_results = self.element_retriever[top_2_category].vectorstore._collection.query(
                query_embeddings=image_query_embeddings[idx],
                n_results=top_k,
                include=['documents', 'distances', 'metadatas']
            )
            for doc, dist, meta in zip(top_2_query_results['documents'][0], top_2_query_results['distances'][0],
                                       top_2_query_results['metadatas'][0]):
                info = json.loads(doc)
                cosine_sim = 1 - dist
                if cosine_sim >= embedding_fn.img_low_similarity_threshold and len(single_results) < top_k * 2 / 3:
                    self.base64_to_pil(info['screenshot']).save(new_dir[idx] + f"\\img_sim_{str(1 - dist)}_type_{info['ui_type']}_top_2.png")
                    single_results.append(UIElementRecording(
                        ui_type=info['ui_type'],
                        screenshot=info['screenshot']
                    ))
                else:
                    break

            for doc, dist, meta in zip(image_query_results['documents'][idx], image_query_results['distances'][idx],
                                       image_query_results['metadatas'][idx]):
                info = json.loads(doc)
                cosine_sim = 1 - dist  # The distance returned by Chroma equals to '1 - cosine'.
                if cosine_sim >= embedding_fn.img_high_similarity_threshold and len(single_results) < top_k:
                    self.base64_to_pil(info['screenshot']).save(new_dir[idx] + f"\\img_sim_{str(1 - dist)}_type_{info['ui_type']}.png")
                    single_results.append(UIElementRecording(
                        ui_type=info['ui_type'],
                        screenshot=info['screenshot'],
                    ))
                else:
                    break
            results.append(single_results)

        return results

    def visualize_outline_matches(self, outline_distances, ui1, screensize, save_dir, top_k=5):
        from PIL import Image, ImageDraw, ImageFont

        def draw_bbox(draw, bbox, color, width=3):
            x1, y1, x2, y2 = bbox
            draw.rectangle([x1, y1, x2, y2], outline=color, width=width)

        def draw_text(draw, pos, text, color, font_size=30):
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
            except:
                font = ImageFont.load_default()
            draw.text(pos, text, fill=color, font=font)

        import os
        os.makedirs(save_dir, exist_ok=True)

        ui1_outline = ui1.outline.copy()

        # similarity 越大越相似 → 取前 top_k
        topk_distances = sorted(outline_distances, key=lambda x: x[0], reverse=True)[:top_k]

        for rank, (score, dpidx, uiidx) in enumerate(topk_distances, 1):
            ui2 = self.recordings[dpidx].ui_action_sequence[uiidx]

            ui2_outline = self.resize_outline(ui2.screenshot, ui2.outline, screensize)

            # 重新算一次，拿 matches
            _, matches = self.ui_dp_match(
                ui1_outline,
                ui2_outline,
                screensize
            )

            # base64 → PIL
            img_left = self.base64_to_pil(ui1.screenshot)
            img_right = self.resize_screenshot(ui2.screenshot, screensize)

            W, H = img_left.size

            # 拼接左右图
            canvas = Image.new("RGB", (W + img_right.size[0], H), (255, 255, 255))
            canvas.paste(img_left, (0, 0))
            canvas.paste(img_right, (W, 0))

            draw = ImageDraw.Draw(canvas)

            # 匹配索引集合
            matched_left = set(i for i, _, _ in matches)
            matched_right = set(j for _, j, _ in matches)

            # -------- 左图 --------
            for i, elem in enumerate(ui1_outline):
                bbox = elem["bbox"]
                if i in matched_left:
                    draw_bbox(draw, bbox, color="blue")
                else:
                    draw_bbox(draw, bbox, color="red")

            # -------- 右图 --------
            for j, elem in enumerate(ui2_outline):
                bbox = elem["bbox"]
                bbox_shifted = [
                    bbox[0] + W,
                    bbox[1],
                    bbox[2] + W,
                    bbox[3],
                ]
                if j in matched_right:
                    draw_bbox(draw, bbox_shifted, color="blue")
                else:
                    draw_bbox(draw, bbox_shifted, color="yellow")

            FONT_SIZE = 30
            MARGIN = 6  # 与 bbox 的间距

            for idx, (i, j, sim) in enumerate(matches, 1):
                bbox_l = ui1_outline[i]["bbox"]
                bbox_r = ui2_outline[j]["bbox"]

                # ---------- 左图 ----------
                x1, y1, x2, y2 = bbox_l

                # 放在 bbox 上方；如果越界则放在框内顶部
                text_y = y1 - FONT_SIZE - MARGIN
                if text_y < 0:
                    text_y = y1 + MARGIN

                # 编号（左上）
                draw_text(
                    draw,
                    (x1, text_y),
                    f"{idx}",
                    "blue",
                    font_size=FONT_SIZE
                )

                # ---------- 左图相似度（右下） ----------
                sim_text = f"{sim:.2f}"

                # 优先放在 bbox 外侧下方
                sim_y = y2 + MARGIN
                if sim_y + FONT_SIZE > H:
                    # 放不下，退回到 bbox 内侧下方
                    sim_y = y2 - FONT_SIZE - MARGIN

                draw_text(
                    draw,
                    (x2 - FONT_SIZE * 2, sim_y),
                    sim_text,
                    "blue",
                    font_size=FONT_SIZE
                )

                # ---------- 右图 ----------
                x1, y1, x2, y2 = bbox_r
                x1 += W
                x2 += W

                text_y = y1 - FONT_SIZE - MARGIN
                if text_y < 0:
                    text_y = y1 + MARGIN

                draw_text(
                    draw,
                    (x1, text_y),
                    f"{idx}",
                    "blue",
                    font_size=FONT_SIZE
                )

                # ---------- 右图相似度（右下） ----------
                sim_text = f"{sim:.2f}"

                sim_y = y2 + MARGIN
                if sim_y + FONT_SIZE > H:
                    sim_y = y2 - FONT_SIZE - MARGIN

                draw_text(
                    draw,
                    (x2 - FONT_SIZE * 2, sim_y),
                    sim_text,
                    "blue",
                    font_size=FONT_SIZE
                )

            # 保存
            out_path = os.path.join(
                save_dir,
                f"outline_rank{rank}_sim{score:.3f}.png"
            )
            canvas.save(out_path)

            pass

    @staticmethod
    def resize_outline(screenshot, outline, target_size):
        outline = copy.deepcopy(outline)

        original_size = Image.open(BytesIO(base64.b64decode(screenshot))).size
        scale = target_size[1] / original_size[1]

        for box in outline:
            box['bbox'] = [t * scale for t in box['bbox']]

        return outline

    @staticmethod
    def resize_screenshot(screenshot, target_size):
        ss = Image.open(BytesIO(base64.b64decode(screenshot)))

        w, h = ss.size
        target_h = target_size[1]

        scale = target_h / h
        new_w = int(w * scale)

        ss_resized = ss.resize((new_w, target_h), Image.LANCZOS)

        return ss_resized

    def query_ui(self, ui: UIRecording, top_k=3):
        new_dir = os.path.join("output\\ui", str(uuid.uuid4()))
        os.makedirs(new_dir)
        self.base64_to_pil(ui.screenshot).save(new_dir + "\\original_ss.png")

        screensize = Image.open(BytesIO(base64.b64decode(ui.screenshot))).size
        # ui.outline = self.rebuild_outlines(ui.screenshot, ui.view_hierarchy)
        ui.outline = self.extract_outlines(ui.view_hierarchy)

        outline_distances = []
        for dpidx, dprecord in enumerate(self.recordings):
            for uiidx, ui_or_action in enumerate(dprecord.ui_action_sequence):
                if isinstance(ui_or_action, UIRecording):
                    similarity, matches = self.ui_dp_match(ui.outline, self.resize_outline(ui_or_action.screenshot,
                                                                                           ui_or_action.outline,
                                                                                           screensize), screensize)
                    outline_distances.append((similarity, dpidx, uiidx))

        outline_distances.sort(key=lambda x: x[0], reverse=True)

        self.visualize_outline_matches(outline_distances, ui, screensize, save_dir=new_dir)

        score_dict = defaultdict(float)

        # ---------- 处理 outline_distances ----------
        rank = 1
        for similarity, dpidx, uiidx in outline_distances:
            # if similarity < 0.5 * max(6, len(ui.outline)):
            #     break
            score_dict[(dpidx, uiidx)] += 1.0 / rank
            rank += 1

        # ---------- 整理并排序 ----------
        fused = [
            (score, dpidx, uiidx)
            for (dpidx, uiidx), score in score_dict.items()
        ]
        fused.sort(key=lambda x: x[0], reverse=True)

        results = []
        for dist, case_id, ui_idx in reversed(fused[:top_k]):
            results.append(self.recordings[case_id].ui_action_sequence[ui_idx])

        return results

    @staticmethod
    def ui_dp_match(ui1, ui2, screensize):
        """
        输入：
            ui1, ui2: list[dict], 每个 dict 至少包含 'bbox'
        输出：
            total_score: 最大结构相似度
            matches: [(i, j, sim)] 匹配的索引与相似度
        """

        def _bbox_center(bbox):
            x_min, y_min, x_max, y_max = bbox
            return ((x_min + x_max) / 2.0, (y_min + y_max) / 2.0)

        def _bbox_area(bbox):
            x_min, y_min, x_max, y_max = bbox
            return max(0.0, x_max - x_min) * max(0.0, y_max - y_min)

        def element_similarity(
                e1: dict,
                e2: dict,
                screen_size: tuple[int, int],
                alpha: float = 0.25,
        ) -> float:
            """
            原文 GUIPilot 风格的 UI 元素相似度

            Parameters
            ----------
            e1, e2 : dict
                UI 元素，至少包含：
                - 'bbox': [x_min, y_min, x_max, y_max]
                - 'type': str
            screen_size : (W, H)
                UI 截图的宽高，用于归一化距离
            alpha : float
                位置相似度衰减系数

            Returns
            -------
            float
                [0, 1] 区间的相似度
            """

            # ---------- 1. 位置相似度 sp ----------
            c1 = _bbox_center(e1["bbox"])
            c2 = _bbox_center(e2["bbox"])

            dx = c1[0] - c2[0]
            dy = c1[1] - c2[1]
            dist = math.sqrt(dx * dx + dy * dy)

            W, H = screen_size
            screen_diag = math.sqrt(W * W + H * H)

            sp = math.exp(-dist / (alpha * screen_diag))

            # # ---------- 2. 尺寸相似度 ss ----------
            # a1 = _bbox_area(e1["bbox"])
            # a2 = _bbox_area(e2["bbox"])
            #
            # if a1 <= 0 or a2 <= 0:
            #     ss = 0.0
            # else:
            #     ss = math.exp(-abs(math.log(a1 / a2)))

            # ---------- 2. 长宽相似度 -----------
            w1 = e1["bbox"][2] - e1["bbox"][0]
            h1 = e1["bbox"][3] - e1["bbox"][1]
            w2 = e2["bbox"][2] - e2["bbox"][0]
            h2 = e2["bbox"][3] - e2["bbox"][1]

            w_ss = math.exp(-abs(math.log(w1 / w2)))
            h_ss = math.exp(-abs(math.log(h1 / h2)))

            # ---------- 3. 类型相似度（hard constraint） ----------
            # coef = 0.0
            # flex_elem = ["ElementByUIED", "OCRElement"]
            # if e1.get("type") == e2.get("type"):
            #     if e1.get("type") in flex_elem or e2.get("type") in flex_elem:
            #         if sp < 0.8 or ss < 0.8:
            #             coef = 0.0
            #         else:
            #             coef = 1.0
            #     else:
            #         coef = 1.0

            coef = 0.0
            # if e1.get("type") == e2.get("type"):
            if sp < 0.8 or w_ss < 0.8 or h_ss < 0.8:
                coef = 0.0
            else:
                coef = 1.0

            # ---------- 4. 总相似度（乘积） ----------
            return coef * sp * w_ss * h_ss

        m, n = len(ui1), len(ui2)

        # 相似度矩阵
        A = [[element_similarity(ui1[i], ui2[j], screensize) for j in range(n)]
             for i in range(m)]

        # DP 表
        dp = [[0.0] * (n + 1) for _ in range(m + 1)]
        trace = [[None] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                options = [
                    (dp[i - 1][j], "up"),
                    (dp[i][j - 1], "left"),
                    (dp[i - 1][j - 1] + A[i - 1][j - 1], "diag"),
                ]
                dp[i][j], trace[i][j] = max(options, key=lambda x: x[0])

        # 回溯匹配
        matches = []
        i, j = m, n
        while i > 0 and j > 0:
            if trace[i][j] == "diag":
                if A[i - 1][j - 1] > 0:
                    matches.append((i - 1, j - 1, A[i - 1][j - 1]))
                i -= 1
                j -= 1
            elif trace[i][j] == "up":
                i -= 1
            else:
                j -= 1

        matches.reverse()
        return dp[m][n], matches

    def query_next_action(self, ui: UIRecording, semantic_analyzed: SemanticView, rag_analyzed=None, top_k: int = 5):
        ui.pruned_vh = self.prune_viewhierarchy(ui, semantic_analyzed, rag_analyzed)
        # pruned_vh_size = self.get_viewhierarchy_size(ui.pruned_vh)
        query_apted_node = self.convert_to_apted_node(ui.pruned_vh)

        new_dir = os.path.join("output\\action", str(uuid.uuid4()))
        os.makedirs(new_dir)
        self.base64_to_pil(ui.screenshot).save(new_dir + "\\original_ss.png")
        open(new_dir + "\\original_ss.viewhierarchy.txt", "w").write(json.dumps(ui.pruned_vh))

        distances = []
        for dpidx, dprecord in enumerate(self.recordings):
            for uiidx, ui_or_action in enumerate(dprecord.ui_action_sequence[:-1]):
                if isinstance(ui_or_action, UIRecording):
                    ui_apted_node = self.convert_to_apted_node(ui_or_action.pruned_vh)
                    dist = APTED(query_apted_node, ui_apted_node, UICost()).compute_edit_distance()
                    distances.append((dist, dpidx, uiidx))

        distances.sort(key=lambda x: x[0])

        for dist, case_id, ui_idx in distances[:top_k]:
            self.base64_to_pil(self.recordings[case_id].ui_action_sequence[ui_idx].screenshot).save(
                new_dir + f"\\viewhierarchy_dist_{str(dist)}_case_{case_id}_ui_{ui_idx}.png")
            open(new_dir + f"\\viewhierarchy_dist_{str(dist)}_case_{case_id}_ui_{ui_idx}.viewhierarchy.txt", "w").write(
                json.dumps(self.recordings[case_id].ui_action_sequence[ui_idx].pruned_vh))
            action = self.recordings[case_id].ui_action_sequence[ui_idx + 1]
            open(new_dir + f"\\viewhierarchy_dist_{str(dist)}_case_{case_id}_ui_{ui_idx}.next_action.txt", "w").write(
                "Action type: " + action.type +
                "Target element location: " + action.location +
                "Text in element: " + action.text +
                "Other description: " + action.description
            )

        # screensize = Image.open(BytesIO(base64.b64decode(ui.screenshot))).size
        # ui.outline = self.extract_outlines(ui.view_hierarchy)
        #
        # outline_distances = []
        # for dpidx, dprecord in enumerate(self.recordings):
        #     for uiidx, ui_or_action in enumerate(dprecord.ui_action_sequence[:-1]):
        #         if isinstance(ui_or_action, UIRecording):
        #             similarity, matches = self.ui_dp_match(ui.outline, self.resize_outline(ui_or_action.screenshot,
        #                                                                                    ui_or_action.outline,
        #                                                                                    screensize), screensize)
        #             outline_distances.append((similarity, dpidx, uiidx))
        #
        # outline_distances.sort(key=lambda x: x[0], reverse=True)
        #
        # self.visualize_outline_matches(outline_distances, ui, screensize, save_dir=new_dir)

        # for similarity, case_id, ui_idx in outline_distances[:5]:
        #     self.base64_to_pil(self.recordings[case_id].ui_action_sequence[ui_idx].screenshot).save(new_dir + f"\\outline_sim_{str(similarity)}.png")
        #     open(new_dir + f"\\outline_sim_{str(similarity)}.viewhierarchy.txt", "w").write(json.dumps(self.recordings[case_id].ui_action_sequence[ui_idx].pruned_vh))

        score_dict = defaultdict(float)
        # ---------- 处理 distances ----------
        rank = 1
        for dist, dpidx, uiidx in distances:
            # if dist > pruned_vh_size:
            #     break
            score_dict[(dpidx, uiidx)] += 1.0 / rank
            rank += 1

        # # ---------- 处理 outline_distances ----------
        # rank = 1
        # for similarity, dpidx, uiidx in outline_distances:
        #     # if similarity < 0.2 * max(6, len(ui.outline)):
        #     #     break
        #     score_dict[(dpidx, uiidx)] += 1.0 / rank
        #     rank += 1

        # ---------- 整理并排序 ----------
        fused = [
            (score, dpidx, uiidx)
            for (dpidx, uiidx), score in score_dict.items()
        ]
        fused.sort(key=lambda x: x[0], reverse=True)

        results = []
        for dist, case_id, ui_idx in reversed(fused[:top_k]):
            results.append([self.recordings[case_id].ui_action_sequence[ui_idx],
                            self.recordings[case_id].ui_action_sequence[ui_idx + 1]])

        return results

    @staticmethod
    def convert_to_apted_node(ui_node: dict) -> TreeNode:
        """
        将字典格式的 UI 节点转换为 APTED 算法所需的 Node 对象。

        转换规则：
        1. Node.label = ui_node["semantic_type"]
        2. Node.data["text"] = ui_node["text"] + ui_node["ocr_text"]
        3. Node.data["type"] = ui_node["semantic_type"] (为了兼容 UICost 逻辑)
        4. 递归处理 subviews
        """

        # 1. 处理 Label (semantic_type)
        # 假如 semantic_type 为 None，给一个默认值 "Unknown" 防止报错
        label = ui_node.get("semantic_type", "Unknown")

        # 2. 处理 Data (text + ocr_text)
        # 提取 text 和 ocr_text，如果为 None 则视为空字符串
        raw_text = ui_node.get("text") or ""
        ocr_text = ui_node.get("ocr_text") or ""
        combined_text = raw_text + ocr_text
        node_data = {
            "text": combined_text,
        }

        # 3. 递归处理子节点
        children_dicts = ui_node.get("subviews", [])
        children_nodes = [LLMRAG.convert_to_apted_node(child) for child in children_dicts]

        # 4. 返回 Node 对象
        return TreeNode(label, children_nodes, node_data)

    @staticmethod
    def get_viewhierarchy_size(viewhierarchy: dict):
        def traversex(view):
            size = 1
            subviews = view.get('subviews')
            for subview in subviews:
                size += traversex(subview)

            return size

        return traversex(viewhierarchy)

    @staticmethod
    def prune_viewhierarchy(ui: UIRecording, semantic_analyzed: SemanticView, rag_analyzed: RAGView = None):
        # 1. 准备基础数据
        view_hierarchy = ui.view_hierarchy

        # 集合用于快速查找需要保留的ID
        # keep_ids = {elem.id for elem in semantic_analyzed.semantic_element}
        keep_ids = set([e.id for e in semantic_analyzed.semantic_element] + \
                       [e.id for e in rag_analyzed.rag_elements.element_groups if e.type != "Normal Element"] if rag_analyzed and rag_analyzed.rag_elements else [])
        keep_ids.add(view_hierarchy['id'])  # 确保根节点始终保留

        # 映射表：id -> semantic_type
        semantic_type_map = {elem.id: elem.type for elem in semantic_analyzed.semantic_element}
        rag_type_map = {}
        if rag_analyzed:
            rag_type_map = {elem.id: elem.type for elem in rag_analyzed.rag_elements.element_groups if elem.type != "Normal Element"}
        root_semantic_type = semantic_analyzed.semantic_interface.type

        # =============================
        # 核心递归函数：增加了 path_seen_types 参数
        # =============================
        def collect_kept_nodes(node: dict, path_seen_types: set):
            node_id = node["id"]
            original_subviews = node.get("subviews", [])

            # 1. 预先计算当前节点的语义类型（为了判断是否重复）
            current_type = (
                root_semantic_type if node_id == view_hierarchy["id"]
                else rag_type_map.get(node_id) if node_id in rag_type_map
                else semantic_type_map.get(node_id)
            )

            # 2. 判断逻辑：ID在保留列表内，且其类型在路径上未出现过
            # 注意：如果 current_type 为 None，通常建议不触发去重逻辑，或者根据业务决定
            is_duplicate_type = current_type is not None and current_type in path_seen_types
            should_keep = (node_id in keep_ids) and not is_duplicate_type

            # 3. 准备传递给子节点的类型集合
            # 如果当前节点被保留，则将它的类型加入“已见集合”传给后代
            new_path_types = path_seen_types.copy()
            if should_keep and current_type is not None:
                new_path_types.add(current_type)

            # 4. 递归处理所有子节点
            all_surviving_descendants = []
            for child in original_subviews:
                # 注意这里传递的是 new_path_types
                survivors = collect_kept_nodes(child, new_path_types)
                all_surviving_descendants.extend(survivors)

            # 5. 根据判断结果决定返回形式
            if should_keep:
                # === 情况 A: 当前节点是幸存者 ===
                new_node = {
                    "id": node_id,
                    "absolute_frame": node.get("absolute_frame"),
                    "semantic_type": current_type,
                    # 接管所有下层传上来的幸存节点
                    "subviews": all_surviving_descendants,

                    'type': node.get('type', ''),
                    'accessible': node.get('accessible', True),
                    'visible': node.get('visible', True),
                    'enabled': node.get('enabled', True),
                }
                if "text" in node:
                    new_node["text"] = node.get("text")
                if "ocr_text" in node:
                    new_node["ocr_text"] = node.get("ocr_text")

                return [new_node]
            else:
                # === 情况 B: 当前节点被剪枝（可能是ID不在列表，或者是类型在路径中重复了） ===
                # 透传子树中幸存的节点
                return all_surviving_descendants

        # # =============================
        # # 核心递归函数：返回该节点子树下所有"幸存"的节点列表
        # # =============================
        # def collect_kept_nodes(node: dict):
        #     node_id = node["id"]
        #     original_subviews = node.get("subviews", [])
        #
        #     # 1. 递归处理所有子节点
        #     # 这里汇集了当前节点所有分支下，所有经过层层筛选后留下来的节点
        #     all_surviving_descendants = []
        #     for child in original_subviews:
        #         survivors = collect_kept_nodes(child)
        #         all_surviving_descendants.extend(survivors)
        #
        #     # 2. 判断当前节点是否需要保留
        #     if node_id in keep_ids:
        #         # === 情况 A: 当前节点是幸存者 ===
        #         # 创建新节点（只保留需要的字段）
        #         new_node = {
        #             "id": node_id,
        #             "absolute_frame": node.get("absolute_frame"),
        #             "semantic_type": (
        #                 root_semantic_type if node_id == view_hierarchy["id"]
        #                 else rag_type_map.get(node_id) if node_id in rag_type_map
        #                 else semantic_type_map.get(node_id)
        #             ),
        #             # 接管所有下层传上来的幸存节点作为自己的直接子节点
        #             "subviews": all_surviving_descendants,
        #
        #             'type': node.get('type', ''),
        #             'accessible': node.get('accessible', True),
        #             'visible': node.get('visible', True),
        #             'enabled': node.get('enabled', True),
        #         }
        #         if "text" in node:
        #             new_node["text"] = node.get("text")
        #         if "ocr_text" in node:
        #             new_node["ocr_text"] = node.get("ocr_text")
        #         # 返回包含自己的列表，供父级使用
        #         return [new_node]
        #     else:
        #         # === 情况 B: 当前节点需要被剪枝 ===
        #         # 自己消失，但负责将下层幸存的节点原样向上层透传
        #         # 这样，上层的“幸存者”就会直接连接到这些节点，从而跳过当前层
        #         return all_surviving_descendants

        # =============================
        # 执行
        # =============================
        # 因为根节点肯定在 keep_ids 里，所以结果列表一定只有一个元素（即根节点）
        result_list = collect_kept_nodes(view_hierarchy, set())
        # result_list = collect_kept_nodes(view_hierarchy)

        if result_list:
            return result_list[0]
        else:
            # 理论上不会走到这里，除非 view_hierarchy 本身为空或者根节点没加进去
            return None


if __name__ == '__main__':
    # import numpy as np
    # def cosine_similarity(a, b):
    #     a = np.array(a)
    #     b = np.array(b)
    #     return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    #
    # embed_fn = GeminiEmbedding()
    # embedding1 = embed_fn.embed_image(base64.b64encode(open("output/elements/5f3d67d5-726b-452c-af77-a414ea37fc18/original_ss.png", "rb").read()).decode("utf-8"))
    # embedding2 = embed_fn.embed_image(base64.b64encode(open("dataset/train/Others/normal_2.jpg", "rb").read()).decode("utf-8"))
    # embedding3 = embed_fn.embed_image(base64.b64encode(open("dataset/train/Others/normal_3.jpg", "rb").read()).decode("utf-8"))
    #
    # print(cosine_similarity(embedding1, embedding2), cosine_similarity(embedding1, embedding3))
    # pass

    import base64
    RAG = LLMRAG('db/test_1')
    # TARGET_PATH = "cached/results/1769932616_com.outfit7.mytalkingtom_1/step_3.json"
    # TARGET_PATH = "cached/results/1770475973_com.cheese.themepack_1/step_1.json"
    # TARGET_PATH = "cached/results/1770486729_com.appcubby.mirror_1/step_2.json"
    TARGET_PATH = "cached/results/1770537478_kidultlovin.word.zen_1/step_1.json"
    # TARGET_PATH = "cached/results/1771484031_flexus.dye.hard_1/step_1.json"
    # TARGET_PATH = "cached/results/1771737931_com.dynlwpp.ios.atur129_1/step_1.json"
    # TARGET_PATH = "cached/results/1771830947_lolagre.cubesolver_1/step_1.json"
    # TARGET_PATH = "cached/results/1776047000_com.outfit7.mytalkingtom_1/step_1.json"
    with open(TARGET_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    base64_screenshot = data['base64_screenshot']
    uist = data['uist']
    semantic_analyzed = data['semantic_analyzed']
    semantic_analyzed = SemanticView.model_validate(semantic_analyzed, strict=False)
    # rag_analyzed = data["rag_analyze"]
    # rag_analyzed = RAGView.model_validate(rag_analyzed, strict=False)

    next_actions = RAG.query_next_action(UIRecording(
            screenshot=base64_screenshot,
            view_hierarchy=uist['elements']
        ),
        semantic_analyzed,
        # rag_analyzed
        None
    )

    pass

    # RAG.query_element(base64.b64encode(open("dataset/train/Voluntary Ad Trigger Element/vate_1.png", "rb").read()).decode("utf-8"), [])
    # embed_fn = GeminiEmbedding()
    #
    # text1 = '''
    #     'Ad Removal Element': A UI component (like a button, banner, or menu item) that initiates the process of getting an ad-free version of the app.
    # '''
    # embedding1 = embed_fn.embed_text(text1)
    # text2 = '''
    #     'Close Button': Any clickable UI element that visually suggests to the user that they can dismiss, skip, or otherwise exit the current interface.
    # '''
    # embedding2 = embed_fn.embed_text(text2)
    # text3 = '''
    #     'Purchase Button': A button that initiates a purchase when pressed.
    # '''
    # embedding3 = embed_fn.embed_text(text3)
    # text4 = '''
    #     'Reward Element': any visual component (e.g., text, icon, button) that explicitly offers, grants, or confirms a reward to the user. It can be buttons/prompts informing users to watch an ad for a benefit (e.g., "Watch Ad to Double Coins," "Get a free life").
    # '''
    # embedding4 = embed_fn.embed_text(text4)
    #
    # import base64
    # # positive_image_path = "dataset/train/Close Button/1e2a40a8e8344fd591e394d1f03e6195.png"
    # # positive_image_path = "dataset/train/Purchase Interface/0baca48cc62b4d70b72a1aa099b9a7e6.png"
    # # positive_image_path = "dataset/train/Reward Element/1bf444d9a06845cdb340e3a64d3527fb.png"
    # positive_image_path = "dataset/train/Reward Element/f6d65261ff91492b9bb24f3801e786bf.png"
    # img = base64.b64encode(open(positive_image_path, "rb").read()).decode("utf-8")
    # embedding_positive_img = embed_fn.embed_image(img)
    #
    # # negative_image_path = "dataset/train/Reward Element/1e403d4ba45d4404a2f23766ad2fc435.png"
    # # img = base64.b64encode(open(negative_image_path, "rb").read()).decode("utf-8")
    # # embedding_negative_img = embed_fn.embed_image(img)
    #
    # import numpy as np
    # def cosine_similarity(a, b):
    #     a = np.array(a)
    #     b = np.array(b)
    #     return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    #
    # cos1 = cosine_similarity(embedding1, embedding_positive_img)
    # cos2 = cosine_similarity(embedding2, embedding_positive_img)
    # cos3 = cosine_similarity(embedding3, embedding_positive_img)
    # cos4 = cosine_similarity(embedding4, embedding_positive_img)
    # # cos2 = cosine_similarity(embedding, embedding_negative_img)
    # # cos3 = cosine_similarity(embedding_positive_img, embedding_negative_img)
    # print(cos1, cos2, cos3, cos4)
    # pass
