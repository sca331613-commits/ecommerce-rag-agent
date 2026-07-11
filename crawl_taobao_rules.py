"""
crawl_taobao_rules.py — 淘宝规则爬虫
从 rule.taobao.com 和第三方站点抓取规则文档，输出 Markdown 到 data/taobao_rules/
"""
import requests
import re
import time
import json
from pathlib import Path
from urllib.parse import urljoin

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}

OUT_DIR = Path(__file__).parent / "data" / "taobao_rules"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 方案 A: 直连 rule.taobao.com API
# ============================================================

def try_taobao_api():
    """尝试淘宝规则中心 JSON API"""

    # 淘宝规则频道使用的后端接口
    api_urls = [
        # 规则频道搜索接口
        "https://rulechannel.taobao.com/api/rule/search",
        # 规则分类列表
        "https://rulechannel.taobao.com/api/category/list",
    ]

    for url in api_urls:
        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
            print(f"  [{resp.status_code}] {url}")
            if resp.status_code == 200:
                print(f"  响应: {resp.text[:300]}")
        except Exception as e:
            print(f"  ✗ {url}: {e}")

# ============================================================
# 方案 B: 从第三方站点抓取规则摘要
# ============================================================

KNOWN_RULE_SOURCES = [
    # 淘宝规则频道热门规则列表页
    "https://rulechannel.taobao.com/",
    # 天猫规则页
    "https://www.tmall.com/wow/seller/act/rule-detail",
    # 淘宝平台规则总则
    "https://rule.taobao.com/detail-11000238.htm",
    # 淘宝网市场管理与违规处理规范
    "https://rule.taobao.com/detail-11000227.htm",
    # 争议处理规则
    "https://rule.taobao.com/detail-11000243.htm",
    # 七天无理由退货规范
    "https://rule.taobao.com/detail-11000246.htm",
    # 发货管理规范
    "https://rule.taobao.com/detail-11000250.htm",
    # 评价规范
    "https://rule.taobao.com/detail-11000261.htm",
    # 保证金管理规范
    "https://rule.taobao.com/detail-11000240.htm",
    # 营销活动规范
    "https://rule.taobao.com/detail-11000198.htm",
]

THIRD_PARTY_SOURCES = [
    # 第三方整理的淘宝规则
    ("https://www.mgzxzs.com/mp/3/11914.html", "2025淘宝新规汇总"),
    ("https://www.mgzxzs.com/mobile/index/show/catid/3/id/12481.html", "七天无理由退货规范变更"),
]

def fetch_page(url, title="未命名"):
    """抓取单个页面，清洗文本并保存为 Markdown"""
    print(f"\n  抓取: {title}")
    print(f"  URL: {url}")

    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
        resp.encoding = resp.apparent_encoding or 'utf-8'
        print(f"  状态: {resp.status_code}, 大小: {len(resp.text)} 字节")
    except Exception as e:
        print(f"  ✗ 请求失败: {e}")
        return None

    if resp.status_code != 200:
        print(f"  ✗ HTTP {resp.status_code}")
        return None

    html = resp.text

    # 清洗: 移除 script/style/nav/footer
    for tag in ['script', 'style', 'nav', 'footer', 'header', 'iframe']:
        html = re.sub(rf'<{tag}[^>]*>.*?</{tag}>', '', html, flags=re.DOTALL | re.IGNORECASE)

    # 提取纯文本
    text = re.sub(r'<br\s*/?>', '\n', html)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)

    # 截取有效内容（跳过前200字符的无用header）
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    if len(lines) < 5:
        print(f"  ⚠️ 内容太少，跳过")
        return None

    content = '\n\n'.join(lines)

    # 保存
    safe_name = re.sub(r'[\\/:*?"<>|]', '_', title)[:60]
    filepath = OUT_DIR / f"{safe_name}.md"
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(f"# {title}\n\n")
        f.write(f"> 来源: {url}\n\n")
        f.write(content[:50000])  # 限5万字符

    print(f"  ✅ 已保存: {filepath} ({len(content)} 字符)")
    return filepath


def main():
    print("=" * 60)
    print("  淘宝规则爬虫")
    print("=" * 60)

    # Step 1: 尝试 API
    print("\n[1] 尝试 API 接口...")
    try_taobao_api()

    # Step 2: 抓取淘宝官方规则页
    print(f"\n[2] 抓取 {len(KNOWN_RULE_SOURCES)} 个淘宝官方规则页...")
    for url in KNOWN_RULE_SOURCES:
        fetch_page(url, url.split('/')[-1].replace('.htm', ''))
        time.sleep(1.5)  # 礼貌间隔

    # Step 3: 抓取第三方整理
    print(f"\n[3] 抓取 {len(THIRD_PARTY_SOURCES)} 个第三方规则汇总...")
    for url, title in THIRD_PARTY_SOURCES:
        fetch_page(url, title)
        time.sleep(1.5)

    # 汇总
    files = list(OUT_DIR.glob("*.md"))
    print(f"\n{'=' * 60}")
    print(f"  完成！共抓取 {len(files)} 个文档")
    print(f"  输出目录: {OUT_DIR}")
    for f in files:
        size = f.stat().st_size
        print(f"    {f.name} ({size:,} 字节)")

    # 生成索引
    with open(OUT_DIR / "INDEX.md", 'w', encoding='utf-8') as f:
        f.write("# 淘宝规则知识库索引\n\n")
        for mf in sorted(files):
            if mf.name != "INDEX.md":
                with open(mf, 'r', encoding='utf-8') as m:
                    first_line = m.readline().strip('# \n')
                f.write(f"- [{first_line}]({mf.name})\n")

    print(f"  索引: {OUT_DIR / 'INDEX.md'}")


if __name__ == "__main__":
    main()
