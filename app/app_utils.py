def inverted_index_to_text(index_str):
    '''将倒排索引字符串还原为原顺序
    
    Args:
        index_str(str):倒排索引字符串，形如   {"The": [0, 47, 57, 103, 115], "case": [1], "of": [2, 28, 49, 67, 70, 83, 125], "a": [3, 13, 68, 79, 90, 98], "79-year-old": [4], "woman": [5], "with": [6], "neurotoxin": [7, 74], "producing": [8], "Clostridium": [9], "tetani": [10, 51, 73], "identified": [11, 53], "in": [12], "lower": [14], "limb": [15], "laceration": [16], "that": [17, 117], "was": [18, 52, 63, 77, 105], "promptly": [19], "treated": [20], "is": [21], "presented;": [22], "the": [23, 71, 118], "patient": [24, 104, 119], "developed": [25], "no": [26], "symptoms": [27, 124], "tetanus.": [29], "Her": [30], "antibody": [31], "levels": [32, 39], "were": [33, 43], "measured": [34], "as": [35, 112], "0.01": [36], "U/ml": [37], "(protective": [38], ">0.01": [40], "U/ml)": [41], "and": [42, 92], "therefore": [44], "not": [45, 121], "protective.": [46], "isolate": [48], "C": [50, 72], "by": [54, 65], "16S": [55], "sequencing.": [56], "potential": [58], "to": [59, 133, 139], "produce": [60], "tetanus": [61, 129], "toxin": [62], "determined": [64], "detection": [66], "fragment": [69], "gene.": [75], "She": [76], "given": [78, 107], "week": [80], "long": [81], "course": [82], "oral": [84], "flucloxacillin,": [85], "500": [86], "mg": [87, 95], "four": [88], "times": [89, 97], "day": [91], "metronidazole,": [93], "400": [94], "three": [96], "day,": [99], "for": [100], "5": [101], "days.": [102], "subsequently": [106], "prophylactic": [108], "immunoglobulin": [109], "(500": [110], "IU)": [111], "per": [113], "guidelines.": [114], "fact": [116], "did": [120], "manifest": [122], "any": [123], "localised": [126], "or": [127], "generalised": [128], "could": [130], "be": [131], "attributed": [132], "prompt": [134], "management": [135], "when": [136], "she": [137], "presented": [138], "her": [140], "primary": [141], "care": [142], "site.": [143]}

    Returns:
        str:还原后的字符串
    '''
    index_str = index_str['abstract_inverted_index']
    if pd.isna(index_str):
        return np.nan
    try:
        index = json.loads(index_str)
        max_position = 0
        for positions in index.values():
            for pos in positions:
                if pos > max_position:
                    max_position = pos
        words = [""] * (max_position + 1)
        for word, positions in index.items():
            for pos in positions:
                words[pos] = word
        text = " ".join(words).replace("  ", " ")
        return text
    except json.JSONDecodeError:
        print("输入的字符串不是有效的 JSON 格式。")
        return None