# This is code we're using to handle some internal calculations

def calcScore(group: str, tags: list, d: dict, gd: dict=None) -> int:
    grouptags = d[group]
    score = 0
    for tag, weight in grouptags:
        if tag in tags:
            try:
                score+= weight * gd[tag] 
            except:
                score+= weight
    return score

def groupScore(ad: dict, akey: str, g: dict, gd: dict=None) -> dict:
    r = {}
    for k in g:
        r[k] = calcScore(k, ad[akey], d=g, gd=gd)
    return r

def getScores(ad: dict, g: dict, gd: dict=None) -> dict:
    r = {}
    for a in ad:
        r[a] = groupScore(ad, a, g, gd=gd)

    return r

def getScoresbyTag(ad:dict, g:dict, gd:dict, tags: list) -> list:
    r = getScores(ad, g, gd)
    fin = {}
    for a in r:
        s = 0
        for k in tags:
            s+= r[a][k]
        fin[a] = s
    final = []    
    for k,v in sorted(fin.items(), key=lambda item: item[1], reverse=True):
        final.append((k,v))

    return final
    