# -*- coding: utf-8 -*-
"""
Created on Wed Feb 18 13:59:42 2015

@author: franciscojavierarceo
"""

 
import urllib
import simplejson
 
 
BASE_URL = "https://www.googleapis.com/customsearch/v1?key=<YOUR GOOGLE API KEY>&cx=<YOUR GOOGLE SEARCH ENGINE CX>"
 
 
def __get_all_hcards_from_query(query, index=0, hcards={}):
 
    url = query
 
    if index != 0:
 
        url = url + '&start=%d' % (index)
 
    json = simplejson.loads(urllib.urlopen(url).read())
 
    if json.has_key('error'):
 
        print "Stopping at %s due to Error!" % (url)
 
        print json
 
    else:
 
        for item in json['items']:
 
            try:
 
                hcards[item['pagemap']['hcard'][0]['fn']] = item['pagemap']['hcard'][0]['title']
 
            except KeyError as e:
 
                pass
 
        if json['queries'].has_key('nextPage'):
 
            return __get_all_hcards_from_query(query, json['queries']['nextPage'][0]['startIndex'], hcards)
 
    return hcards
 
 
def get_all_employees_by_company_via_linkedin(company):
 
    queries = ['"at %s" inurl:"in"', '"at %s" inurl:"pub"']
 
    result = {}
 
    for query in queries:
 
        _query = query % company
 
        result.update(__get_all_hcards_from_query(BASE_URL + '&q=' + _query))
 
    return list(result)