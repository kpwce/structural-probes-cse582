# Get language tagging by iterating thru cs data and aligning it
# with English/Spanish monolingual texts.
# e.g., output ["en", "es", "es", ..]
from codeswitch.codeswitch import LanguageIdentification
def get_alignment(cs_data, en_data, es_data):
    out = [] 
    progress = [[en_data.split(), 0, "en"], [es_data.split(), 0, "es"]]
    state = 0
    for w in cs_data.split():
        if progress[state][0][progress[state][1]] != w:
            state = 1 if state == 0 else 0
            while progress[state][0][progress[state][1]] != w:
                progress[state][1] += 1
                assert progress[state][1] < len(progress[state][0]), "Uh oh, no alignment"

        out.append(progress[state][2])
        progress[state][1] += 1

    return out

def get_alignment_2(sublist, monolingual_list):
    """
    Find and return the interval in the monolingual list that best matches the given sublist.
    "Best matches" is defined as the number of matching tokens.
    
    :Parameters:
    - :sublist: The list of words to align to
    - :monolingual_list: The monolingual text we will search through to find some interval 
        that aligns with the given sublist
    """
    
    best_subarray = []
    best_cost = float("inf")
    for start in range(0, len(monolingual_list) - len(sublist) + 1):
        # check interval of start..start + len(sublist)
        # nuggets spicy vs spicy nuggets
        subarray_target = sublist.copy()
        subarray_current = monolingual_list[start:start + len(sublist)]
        cost = 0
        for token in subarray_current:
            count = subarray_target.count(token)
            if (count == 0):
                cost += 1
            else:
                subarray_target.remove(token)
        if (cost < best_cost):
            best_cost = cost
            best_subarray = subarray_current
    return best_subarray

# use black box classifier to get tree subarrays
def get_lang_subintervals(cs_data):
    lid = LanguageIdentification('spa-eng') 
    res = lid.identify(cs_data)
    lang = res[0]['entity']
    all_trees = []
    all_langs = []
    current_tree = []
    current_tree.append(res[0]['word'])

    # construct parse tree
    for i in range(1, len(res)):
        if res[i]['word'][0:2] == "##":
            # merge with prev (we make assumption that all subtokens are classified as the same language)
            current_tree[-1] = current_tree[-1] + res[i]['word'][2:]
            continue
        if res[i]['entity'] != lang:
            if (lang != "other"): # ignore things like punctuation
                all_trees.append(current_tree.copy())
                all_langs.append(lang)
            current_tree = []
            lang = res[i]['entity']
        current_tree.append(res[i]['word'])
    if (lang != "other"):
        all_trees.append(current_tree)
        all_langs.append(lang)


    print(all_trees)
    return all_trees, all_langs
