# Get language tagging by iterating thru cs data and aligning it
# with English/Spanish monolingual texts.
# e.g., output ["en", "es", "es", ..]
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
