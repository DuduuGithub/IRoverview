import json
import networkx as nx
import community  # python-louvain
import numpy as np
import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import defaultdict
import math
import os

def load_predictions(file_path):
    """加载预测结果"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['predictions']

def create_word_network(predictions, score_threshold=0.4):
    """创建词语关系网络"""
    G = nx.Graph()
    
    # 添加边和权重
    for pred in predictions:
        if float(pred['score']) > score_threshold:
            word1 = pred['original_word']
            word2 = pred['predicted_word']
            weight = float(pred['score'])
            
            # 添加边，如果边已存在则累加权重
            if G.has_edge(word1, word2):
                G[word1][word2]['weight'] += weight
            else:
                G.add_edge(word1, word2, weight=weight)
    
    return G

def detect_communities(G):
    """使用Louvain算法检测社区"""
    communities = community.best_partition(G)
    return communities

def get_community_colors(communities, n_main=5, n_shades=6):
    """为每个社区分配颜色"""
    community_ids = sorted(set(communities.values()))
    n_communities = len(community_ids)
    color_map = {}
    main_hues = [(i / n_main) for i in range(n_main)]
    
    for idx, cid in enumerate(community_ids):
        main_idx = idx % n_main
        shade_idx = idx // n_main
        v = 0.85 - 0.15 * (shade_idx % n_shades) / max(1, n_shades-1)
        s = 0.7 + 0.2 * (shade_idx % n_shades) / max(1, n_shades-1)
        rgb = matplotlib.colors.hsv_to_rgb((main_hues[main_idx], s, v))
        color = matplotlib.colors.rgb2hex(rgb)
        color_map[cid] = color
    return color_map

def darken_color(hex_color, factor=0.7):
    rgb = mcolors.to_rgb(hex_color)
    hsv = mcolors.rgb_to_hsv(rgb)
    hsv = list(hsv)
    hsv[2] *= factor  # 降低亮度
    return mcolors.to_hex(mcolors.hsv_to_rgb(hsv))

def create_word_clouds(G, communities, output_file='word_clouds.png'):
    """为每个社区生成单独词云图片，并生成HTML卡片网格展示（主词高亮）"""
    color_map = get_community_colors(communities, n_main=5, n_shades=8)
    community_words = defaultdict(dict)
    for node in G.nodes():
        community_id = communities[node]
        importance = G.degree(node, weight='weight')
        community_words[community_id][node] = importance

    # 1. 计算每个社区的主词
    community_main_word = {}
    for community_id, words in community_words.items():
        if words:
            main_word = max(words.items(), key=lambda x: x[1])[0]
            community_main_word[community_id] = main_word

    # 2. 词云主词颜色加深
    def make_color_func(main_word, main_color_dark, default_color):
        def color_func(word, *args, **kwargs):
            if word == main_word:
                return main_color_dark
            return default_color
        return color_func

    # 确保输出图片目录存在
    img_dir = 'word_cloud_communities'
    os.makedirs(img_dir, exist_ok=True)
    img_paths = {}
    for community_id, words in community_words.items():
        main_word = community_main_word.get(community_id, None)
        main_color = color_map[community_id]
        main_color_dark = darken_color(main_color, 0.7)
        wordcloud = WordCloud(
            width=400,
            height=400,
            background_color='white',
            color_func=make_color_func(main_word, main_color_dark, main_color),
            prefer_horizontal=0.9,
            min_font_size=15,
            max_font_size=120,
            relative_scaling=0.5,
            collocations=False,
            contour_width=2,
            contour_color=main_color
        ).generate_from_frequencies(words)
        img_path = os.path.join(img_dir, f'word_cloud_community_{community_id}.png')
        wordcloud.to_file(img_path)
        img_paths[community_id] = img_path

    # 3. 生成HTML卡片网格（主词高亮，悬浮弹窗含主词）
    html_output = 'word_clouds.html'
    color_palette = [
        "#e57373", "#64b5f6", "#81c784", "#ffd54f", "#ba68c8", "#4db6ac", "#ff8a65", "#7986cb", "#a1887f"
    ]
    with open(html_output, 'w', encoding='utf-8') as f:
        f.write('''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Community Word Clouds</title>
            <style>
                body { background: #fafbfc; }
                .word-cloud-container {
                    display: flex;
                    flex-wrap: wrap;
                    justify-content: center;
                    gap: 32px;
                    padding: 24px;
                }
                .word-cloud-card {
                    background: #fff;
                    border: 1.5px solid #e0e0e0;
                    border-radius: 12px;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
                    width: 310px;
                    padding: 12px 12px 8px 12px;
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    margin-bottom: 10px;
                    transition: box-shadow 0.3s, transform 0.3s;
                    position: relative;
                    z-index: 1;
                }
                .word-cloud-card:hover {
                    box-shadow: 0 8px 32px rgba(0,0,0,0.18);
                    transform: translateY(-12px) scale(1.08);
                    z-index: 10;
                }
                .word-cloud-card img {
                    width: 270px;
                    height: 270px;
                    object-fit: contain;
                    border-radius: 8px;
                    margin-bottom: 8px;
                }
                .word-cloud-card .info {
                    width: 100%;
                    text-align: left;
                }
                .word-cloud-card .info h3 {
                    margin: 0 0 4px 0;
                    font-size: 1.08em;
                    color: #222;
                }
                .word-cloud-card .info p {
                    margin: 2px 0;
                    font-size: 0.98em;
                    color: #444;
                }
                .main-word {
                    font-size: 1.25em;
                    font-weight: bold;
                    color: inherit;
                    background: #f5f5f5;
                    padding: 2px 8px;
                    border-radius: 6px;
                    margin-bottom: 4px;
                    display: inline-block;
                }
                .main-word-label {
                    font-size: 0.95em;
                    color: #222;
                    font-weight: bold;
                    margin-right: 6px;
                }
                .top-words {
                    margin-top: 6px;
                    display: flex;
                    flex-wrap: wrap;
                    gap: 8px;
                }
                .top-word {
                    display: flex;
                    align-items: baseline;
                    font-size: 1.15em;
                    font-weight: bold;
                    margin-right: 6px;
                }
                .top-word-value {
                    font-size: 0.7em;
                    color: #888;
                    margin-left: 3px;
                    font-weight: normal;
                }
                .related-words-pop {
                    display: none;
                    position: absolute;
                    left: 50%;
                    bottom: 12px;
                    transform: translateX(-50%);
                    background: rgba(255,255,255,0.98);
                    border-radius: 8px;
                    box-shadow: 0 2px 12px rgba(0,0,0,0.13);
                    padding: 12px 16px;
                    min-width: 220px;
                    z-index: 100;
                    font-size: 1em;
                    color: #333;
                    animation: fadeIn 0.3s;
                }
                .word-cloud-card:hover .related-words-pop {
                    display: block;
                }
                .main-word-pop {
                    font-size: 1.08em;
                    font-weight: bold;
                    color: inherit;
                    margin-bottom: 4px;
                }
                .main-word-pop-label {
                    color: #222;
                    font-weight: bold;
                    margin-right: 6px;
                }
                @keyframes fadeIn {
                    from { opacity: 0; transform: translateY(20px);}
                    to { opacity: 1; transform: translateY(0);}
                }
            </style>
        </head>
        <body>
            <div class="word-cloud-container">
        ''')
        for idx, (community_id, words) in enumerate(community_words.items()):
            main_word = community_main_word.get(community_id, None)
            main_color = color_map[community_id]
            main_color_dark = darken_color(main_color, 0.7)
            top_words = sorted(words.items(), key=lambda x: x[1], reverse=True)[:3]
            top_words_html = ""
            for i, (word, weight) in enumerate(top_words):
                color = color_palette[i % len(color_palette)]
                top_words_html += (
                    f'<span class="top-word" style="color:{color}">' 
                    f'{word}<span class="top-word-value">({weight:.2f})</span>'
                    f'</span>'
                )
            # 悬浮弹窗内容，含主词
            related_words_html = (
                f'<div class="main-word-pop-label">Main word: <span class="main-word-pop" style="color:{main_color_dark}">{main_word}</span></div>'
                f'<div style="margin-top:6px;"><b>Most related words:</b><br>' + ", ".join(
                    [f'<span style="color:{color_palette[i % len(color_palette)]};font-size:1.1em;font-weight:bold;">{word}</span>'
                     f'<span style="color:#888;font-size:0.9em;">({weight:.2f})</span>'
                     for i, (word, weight) in enumerate(top_words)]
                ) + '</div>'
            )
            f.write(f'''
                <div class="word-cloud-card">
                    <img src="{img_paths[community_id]}" alt="Community {community_id}">
                    <div class="info">
                        <span class="main-word-label">Main word:</span> <span class="main-word" style="color:{main_color_dark}">{main_word}</span><br>
                        <h3>Community {community_id}</h3>
                        <p>Words: {len(words)}</p>
                        <div class="top-words">{top_words_html}</div>
                    </div>
                    <div class="related-words-pop">{related_words_html}</div>
                </div>
            ''')
        f.write('''
            </div>
        </body>
        </html>
        ''')
    print(f"每个社区词云图片已保存到 {img_dir}/，HTML 网格已保存为 {html_output}")

def export_theme_thesaurus(community_words, community_main_word, output_file='theme_thesaurus.json'):
    result = []
    for comm_id, words in community_words.items():
        main_word = community_main_word[comm_id]
        max_score = max(words.values()) if words else 1.0
        word_list = []
        for word, score in words.items():
            norm_score = score / max_score if max_score else 0.0
            word_list.append({
                "word": word,
                "score": 1.0 if word == main_word else round(norm_score, 4)
            })
        # 主词排最前
        word_list = sorted(word_list, key=lambda x: (x['word'] != main_word, -x['score']))
        result.append({
            "theme": main_word,
            "words": word_list
        })
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"主题叙词表已导出到 {output_file}")

def main():
    # 加载已筛选的预测结果
    predictions = load_predictions('filtered_thesaurus_nomean(part).json')

    # 创建网络
    G = create_word_network(predictions, score_threshold=1) # 可以设置score_threshold阈值，控制过滤的窗口大小和预测词数量

    # 检测社区
    communities = detect_communities(G)

    # 统计社区信息
    print(f"网络包含 {G.number_of_nodes()} 个节点和 {G.number_of_edges()} 条边")
    print(f"检测到 {len(set(communities.values()))} 个社区")

    # 直接用全部节点和社区进行可视化
    create_word_clouds(G, communities)

    # 统计社区词和主词
    community_words = defaultdict(dict)
    for node in G.nodes():
        community_id = communities[node]
        importance = G.degree(node, weight='weight')
        community_words[community_id][node] = importance
    community_main_word = {}
    for community_id, words in community_words.items():
        if words:
            main_word = max(words.items(), key=lambda x: x[1])[0]
            community_main_word[community_id] = main_word

    # 导出主题叙词表
    export_theme_thesaurus(community_words, community_main_word)

if __name__ == "__main__":
    main() 