# -*- coding: utf-8 -*-
"""
Created on Wed Feb 18 14:27:31 2015

@author: franciscojavierarceo
"""

#!/usr/bin/env python
 
"""

Use Twitter API to grab user information from list of organizations; 
export text file

Uses Twython module to access Twitter API

"""
 
import sys
import string
import simplejson
import pandas as pd
from twython import Twython
 
#WE WILL USE THE VARIABLES DAY, MONTH, AND YEAR FOR OUR OUTPUT FILE NAME
import datetime
now = datetime.datetime.now()
day=int(now.day)
month=int(now.month)
year=int(now.year)
 
 
#FOR OAUTH AUTHENTICATION -- NEEDED TO ACCESS THE TWITTER API
t = Twython(app_key='GotdByfuWw6wLT3OYEaNFbF9c', #REPLACE 'APP_KEY' WITH YOUR APP KEY, ETC., IN THE NEXT 4 LINES
    app_secret='91iEbNlzNYUUjbybW0I8DvECxbffQTEiDZ52FpdEzFzzJWt5C7',
    oauth_token='868966136-IA3MAwfujlzd2TEZbLEgJENlN826TOjAtWIT0sVK',
    oauth_token_secret='P6m4Hyd7dgAezY5J4VwEzmi8lC8uDIpi5T2FrNx8jxFyA')

tmpdf= pd.read_csv(r"/Users/franciscojavierarceo/ProjectWork/Misc/TwitterUsers.txt",
                         header=-1,sep='\t',names=['indx','TwitterName','ignore'])
#ACCESS THE LOOKUP_USER METHOD OF THE TWITTER API -- GRAB INFO ON UP TO 100 IDS WITH EACH API CALL
#THE VARIABLE USERS IS A JSON FILE WITH DATA ON THE 32 TWITTER USERS LISTED ABOVE
userlist = tmpdf['TwitterName']

Rawdata = []
for i in userlist:
    try:
        out = t.lookup_user(screen_name=i)
    except Exception:
        out = "Shit"
    Rawdata.append(out)

# Extracting the fields first
x = Rawdata[0][0]
fields = []
for i in x:
    fields.append(i)

answers = []
for i in fields:
    answers.append(x[i])

df = pd.DataFrame(answers,index=fields)
df = df.T

for i in xrange(1,18):
    print i,Rawdata[i][0]['screen_name']
    

for i in xrange(1,len(Rawdata)):
    x = Rawdata[i][0]    
    newanswers = []
    for j in x:
        if type(j)==bool:
            j = str(j)
        x[j]
        df[j].append()

    tmpdf = pd.DataFrame(newanswers,index=fields)
    tmpdf = tmpdf.T
    df = df.append(tmpdf)
    newanswers = None
    print 'Iteration'+str(i)

print j
print x[j]
pd.DataFrame(x[j])



#NAME OUR OUTPUT FILE - %i WILL BE REPLACED BY CURRENT MONTH, DAY, AND YEAR
outfn = "twitter_user_data_%i.%i.%i.txt" % (now.month, now.day, now.year)
 
#NAMES FOR HEADER ROW IN OUTPUT FILE
fields = "id screen_name name created_at url followers_count friends_count statuses_count \
    favourites_count listed_count \
    contributors_enabled description protected location lang expanded_url".split()
 
#INITIALIZE OUTPUT FILE AND WRITE HEADER ROW   
outfp = open(outfn, "w")
outfp.write(string.join(fields, "\t") + "\n")  # header
 
#THE VARIABLE 'USERS' CONTAINS INFORMATION OF THE 32 TWITTER USER IDS LISTED ABOVE
#THIS BLOCK WILL LOOP OVER EACH OF THESE IDS, CREATE VARIABLES, AND OUTPUT TO FILE
for entry in users:
    #CREATE EMPTY DICTIONARY
    r = {}
    for f in fields:
        r[f] = ""
    #ASSIGN VALUE OF 'ID' FIELD IN JSON TO 'ID' FIELD IN OUR DICTIONARY
    r['id'] = entry['id']
    #SAME WITH 'SCREEN_NAME' HERE, AND FOR REST OF THE VARIABLES
    r['screen_name'] = entry['screen_name']
    r['name'] = entry['name']
    r['created_at'] = entry['created_at']
    r['url'] = entry['url']
    r['followers_count'] = entry['followers_count']
    r['friends_count'] = entry['friends_count']
    r['statuses_count'] = entry['statuses_count']
    r['favourites_count'] = entry['favourites_count']
    r['listed_count'] = entry['listed_count']
    r['contributors_enabled'] = entry['contributors_enabled']
    r['description'] = entry['description']
    r['protected'] = entry['protected']
    r['location'] = entry['location']
    r['lang'] = entry['lang']
    #NOT EVERY ID WILL HAVE A 'URL' KEY, SO CHECK FOR ITS EXISTENCE WITH IF CLAUSE
    if 'url' in entry['entities']:
        r['expanded_url'] = entry['entities']['url']['urls'][0]['expanded_url']
    else:
        r['expanded_url'] = ''
    print r
    #CREATE EMPTY LIST
    lst = []
    #ADD DATA FOR EACH VARIABLE
    for f in fields:
        lst.append(unicode(r[f]).replace("\/", "/"))
    #WRITE ROW WITH DATA IN LIST
    outfp.write(string.join(lst, "\t").encode("utf-8") + "\n")
 
print r