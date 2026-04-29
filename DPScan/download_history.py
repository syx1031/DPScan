import subprocess
import time
import sys
import re
import json
import pandas as pd
import glob
import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='[%(asctime)s] %(levelname)s [%(name)s:%(lineno)s] %(message)s',
    handlers=[
        logging.FileHandler('download.log', mode='a', encoding='utf-8'),  # Logs to file
        logging.StreamHandler()  # Logs to console
    ]
)
logger = logging.getLogger(__name__)

passphrase = "123456"

SIMPLE_DOWNLOAD = False


def extract_year(metadata_str):
    """从 metadata 字符串中提取年份"""
    match = re.search(r"(2019|2020|2021|2022|2023|2024|2025)", metadata_str)
    return int(match.group(1)) if match else None


def uniform_sample(lst, k):
    """均匀采样 k 个元素（包含首尾）"""
    if len(lst) <= k:
        return lst

    idxs = [round(i * (len(lst) - 1) / (k - 1)) for i in range(k)]
    return [lst[i] for i in idxs]


def select_versions_for_download(appid, version_list):
    """
    version_list: 所有历史版本号（从旧到新）
    get_metadata: 外部函数，输入版本号，返回发布时间字符串
    """

    # 1️⃣ 均匀选 15 个或全部
    sampled_versions = (
        version_list
        if len(version_list) < 15
        else uniform_sample(version_list, 15)
    )

    # 2️⃣ 查询 metadata & 提取年份
    info = []
    for v in sampled_versions:
        meta = get_version_metadata(appid, v)
        if not meta:
            return []
        year = extract_year(meta['displayVersion'])
        info.append({
            "version": v,
            "all_metadata": meta,
            "metadata": meta["displayVersion"],
            "year": year
        })

    # 3️⃣ 必选：最早 & 最新
    selected = {info[0]["version"], info[-1]["version"]}

    # 4️⃣ 无年份的版本 → 均匀选 3 个
    no_year = [x for x in info[1:-1] if x["year"] is None]
    if len(no_year) > 0:
        k = min(3, len(no_year))
        idxs = [round(i * (len(no_year) - 1) / (k - 1)) for i in range(k)] if k > 1 else [0]
        for i in idxs:
            selected.add(no_year[i]["version"])

    # 5️⃣ 有年份的版本 → 2019–2025 每年最多 1 个
    with_year = {}
    for x in info:
        if x["year"] is not None and 2019 <= x["year"] <= 2025:
            with_year.setdefault(x["year"], []).append(x)

    for year in range(2019, 2026):
        if year in with_year:
            selected.add(with_year[year][0]["version"])  # 取该年的第一个

    # 6️⃣ logger 输出
    for x in info:
        if x["version"] in selected:
            logger.info(
                f"[DOWNLOAD] version={x['version']} | metadata={x['all_metadata']}"
            )

    ordered_selected = [
        v for v in sampled_versions if v in selected
    ]
    return ordered_selected[:-1]


def select_one_version_near_2021(appid, version_list):
    # 查询 metadata & 提取年份
    info = []
    for v in version_list:
        meta = get_version_metadata(appid, v)
        if not meta:
            return []
        year = extract_year(meta['displayVersion'])
        info.append({
            "version": v,
            "all_metadata": meta,
            "metadata": meta["displayVersion"],
            "year": year
        })

    # 1️⃣ 找到所有年份 ≤ 2023 的版本
    candidates = [x for x in info if x["year"] is not None and x["year"] <= 2023]

    # 2️⃣ 选择版本
    if candidates:
        # 选择最接近 2021 的
        selected_item = min(candidates, key=lambda x: abs(x["year"] - 2021))
    else:
        # 否则选择最老版本
        selected_item = info[0]

    selected_version = selected_item["version"]

    # logger 输出
    logger.info(
        f"[DOWNLOAD] version={selected_item['version']} | metadata={selected_item['all_metadata']}"
    )

    return [selected_version]


def get_version_metadata(appid, version):
    cmd = [
        "ipatool",
        "get-version-metadata", "-i", appid, "--external-version-id", version, "--non-interactive", "--format", "json", "--keychain-passphrase", passphrase,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    # bundleid = None
    try:
        version_metadata = json.loads(result.stdout)
        if version_metadata['level'] == 'error':
            logger.error(f"Failed to get metadata of {appid} - {version}. Error information: {version_metadata}")
            return None
        return version_metadata
    except Exception as e:
        logger.error(
            f"Failed to get metadata of {appid} - {version}. Error information: {result.stdout}, {result.stderr}")
    return None


def purchase_and_download(appid, version, purchase, cache_path):
    old_argv = sys.argv[:]  # 保存原参数
    original_sys_path = sys.path[:]
    sys.path.insert(0, 'ipatool_py_master')

    try:
        from ipatool_py_master import main
        sys.argv = [
            "main.py", "--json",
            "lookup", "--appId", appid, "-c", "US",
            "download", "-s", "http://127.0.0.1:9000", "-o", cache_path,
        ]
        if version:
            sys.argv.extend(["--appVerId", version])
        if purchase:
            sys.argv.append("--purchase")

        json_info = main.main()  # 调用 main()
        pass
    except Exception as e:
        logger.error(f"Failed to download the app {appid}.")
        raise e
    finally:
        sys.argv = old_argv  # 恢复原参数
        sys.path = original_sys_path


def batch_purchase_and_download(appid, versions, cache_path="./temp_cached"):
    for ver in versions:
        purchase_and_download(appid, ver, False, cache_path)
        time.sleep(5)


def download_by_id_with_history(appid, cached, ios_release_time="2021"):
    purchase_and_download(appid, None, True, cached)
    time.sleep(5)

    cmd = [
        "ipatool",
        "auth", "info", "--non-interactive", "--format", "json", "--keychain-passphrase", passphrase,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    try:
        login_info = json.loads(result.stdout)
        assert login_info['success'] == True
    except Exception as e:
        logger.error(
            "IPATool may need to login manually. Please login in cmd with your own accont by 'ipatool auth login -e xxx -p xxx' and then set passphrase as '123456'")
        logger.error(f'Error information: {result.stdout}, {result.stderr}')
        return

    cmd = [
        "ipatool",
        "list-versions", "-i", appid, "--non-interactive", "--format", "json", "--keychain-passphrase", passphrase,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    try:
        version_list = json.loads(result.stdout)
        history_version = version_list['externalVersionIdentifiers']
    except Exception as e:
        logger.error(f"Failed to get historical version information of {appid}. Error information: {result.stdout}, {result.stderr}")
        if "Unknown" in result.stdout:
            raise
        return

    logger.info(f"Got information, try to purchase and download {appid}")

    if SIMPLE_DOWNLOAD:
        selected_versions = select_one_version_near_2021(appid, history_version)
    else:
        selected_versions = select_versions_for_download(appid, history_version)
    time.sleep(5)

    batch_purchase_and_download(appid, selected_versions, cache_path=cached)

    # cmd = [
    #     "ipatool",
    #     "download", "-i", appid, "--purchase", "-o", "./temp_cached", "--non-interactive", "--format", "json", "--keychain-passphrase", passphrase,
    # ]
    # result = subprocess.run(cmd, capture_output=True, text=True)
    # try:
    #     download_info = json.loads(result.stdout)
    #     pass
    # except Exception as e:
    #     logger.error(f"Failed to download the app {appid}. Error information: {result.stdout}, {result.stderr}")


def read_appid_list(path):
    df = pd.read_excel(path)  # 替换为你的文件名

    # 取出 'appid' 列并去重
    appid_list = df['appid'].drop_duplicates().astype(str).tolist()

    return appid_list

def read_appid_from_csv(path, start_index, end_index, threshold=0):
    """
    从 CSV 文件中读取 appId 列，按原始顺序返回满足条件的 appId 列表

    :param path: csv 文件路径
    :param start_index: 起始行索引（包含，0-based）
    :param end_index: 结束行索引（不包含，None 表示到文件末尾）
    :param threshold: 只有 adCommentCount > threshold 的行才会被保留
    """
    df = pd.read_csv(path, encoding="gbk")

    # 按行号切片（保持原有顺序）
    df = df.iloc[start_index:end_index]

    # 过滤 adCommentCount
    # df = df[df["adCommentCount"] > threshold]

    # 取出 appId 列，保持顺序，不去重
    appid_list = df["appId"].astype(str).tolist()

    return appid_list


if __name__ == "__main__":
    # apps = read_appid_list("resources_for_run/ground_truth_syx_us.xlsx")
    # apps = read_appid_from_csv("resources_for_run/AdsDP_us_have_ads.csv", 31, 40)
    apps = read_appid_from_csv("resources_for_run/apps_list.csv", 0, 1)

    cached = './cached'

    downloaded_files = glob.glob(cached + "/*")
    for app in apps:
        # if not any(app in downloaded_file for downloaded_file in downloaded_files):
        #     download_by_id_with_history(app, cached)
        # else:
        #     logger.info(f"App {app} already downloaded. Skip.")
        skip = False
        from pathlib import Path
        for p in Path(cached).iterdir():
            if app in p.name:
                logger.info(f"Found cache in {cached} of app {app}, stop downloading.")
                skip = True
                break

        if skip:
            continue
            # pass

        download_by_id_with_history(app, cached)