# -*- coding: utf-8 -*-
"""
Created on Wed Feb 18 14:57:48 2015

@author: franciscojavierarceo
"""

#!/usr/bin/env python

# This version is based off Greg's original script. Wayne added annotations for Python beginners 

"""
This script is provided by Dr.Gregory D. Saxton (http://www.buffalo.edu/content/cas/communication/faculty/saxton.third.html)

The annotation to the script is by Weiai Xu (Wayne), contributor for The Curiosity Bits Blog (http://curiositybits.com/) 

For further distribution of the script, please contact The Curiosity Bits Blog (http://curiositybits.com/) 
    
    
"""

#import necessary Python libraries. To run the code, you need to install urllib,simplejson,sqlite3,twython.
# use pip install to install the libraries, for more information on installing libraries, check out our tutorials 

import sys
import urllib
import string
import simplejson
import sqlite3

import time
import datetime
from pprint import pprint

import sqlalchemy
from sqlalchemy.orm import mapper, sessionmaker
from sqlalchemy import Column, Integer, String, ForeignKey, Text, DateTime, Unicode, Float # importing Unicode is important! If not, you likely encounter data type error.
from sqlalchemy.ext.declarative import declarative_base
from types import *

from datetime import datetime, date, time

from twython import Twython

t = Twython(app_key='GotdByfuWw6wLT3OYEaNFbF9c', #REPLACE 'APP_KEY' WITH YOUR APP KEY, ETC., IN THE NEXT 4 LINES
    app_secret='91iEbNlzNYUUjbybW0I8DvECxbffQTEiDZ52FpdEzFzzJWt5C7',
    oauth_token='868966136-IA3MAwfujlzd2TEZbLEgJENlN826TOjAtWIT0sVK',
    oauth_token_secret='P6m4Hyd7dgAezY5J4VwEzmi8lC8uDIpi5T2FrNx8jxFyA')

Base = declarative_base()

class TWEET(Base):
    __tablename__ = 'typhoon' # This is table name, you can change to yours
    rowid = Column(Integer, primary_key=True)  
    query = Column(String)
    user_type = Column (String)
    #tweet_id = Column(Integer) 
    tweet_id = Column(String) 
    #to_user_id = Column(String)
    inserted_date = Column(DateTime)
    truncated = Column(String)
    language = Column(String)
    possibly_sensitive = Column(String)  ### NEW 
    coordinates = Column(String)					
    retweeted_status = Column(String)
    withheld_in_countries = Column(String)
    withheld_scope = Column(String)
    created_at_text = Column(String)   				
    created_at = Column(DateTime)
    month = Column(String)
    year = Column(String)
    content = Column(Text)
    from_user_screen_name = Column(String)			
    from_user_id = Column(String)   
    from_user_followers_count = Column(Integer)  	
    from_user_friends_count = Column(Integer)  		
    from_user_listed_count = Column(Integer)  		
    from_user_favourites_count = Column(Integer)	
    from_user_statuses_count = Column(Integer)  
    from_user_description = Column(String)  		
    from_user_location = Column(String)  			
    from_user_created_at = Column(String)  			
    retweet_count = Column(Integer)
    favorite_count = Column(Integer)				
    entities_urls = Column(Unicode(255))
    entities_urls_count = Column(Integer)        
    entities_hashtags = Column(Unicode(255))
    entities_hashtags_count = Column(Integer)    
    entities_mentions = Column(Unicode(255))    
    entities_mentions_count = Column(Integer)  
    in_reply_to_screen_name = Column(String)  
    in_reply_to_status_id = Column(String)  
    source = Column(String)
    entities_expanded_urls = Column(Text) 
    json_output = Column(String)
    entities_media_count = Column(Integer)
    media_expanded_url = Column(Text) 
    media_url = Column(Text) 
    media_type = Column(Text) 
    video_link = Column(Integer)
    photo_link = Column(Integer)
    twitpic = Column(Integer)
    num_characters = Column(Integer)    				
    num_words = Column(Integer)        					
    #hashtags = Column(Unicode(255))
    #direct = Column(Unicode(255))
    #initial_direct = Column(Unicode(255))
    #initial_rt = Column(Unicode(255))
    #clicks = Column(Integer)    
    #total_clicks = Column(Integer)    
    
    def __init__(self, query, user_type, tweet_id, inserted_date, truncated, language, possibly_sensitive, coordinates, 
    retweeted_status, withheld_in_countries, withheld_scope, created_at_text, created_at, month, year, content, 
    from_user_screen_name, from_user_id, from_user_followers_count, from_user_friends_count,   
    from_user_listed_count, from_user_favourites_count, from_user_statuses_count, from_user_description,   
    from_user_location, from_user_created_at, retweet_count, favorite_count, entities_urls, entities_urls_count,         
    entities_hashtags, entities_hashtags_count, entities_mentions, entities_mentions_count,   
    in_reply_to_screen_name, in_reply_to_status_id, source, entities_expanded_urls, json_output, 
    entities_media_count, media_expanded_url, media_url, media_type, video_link, photo_link, twitpic, 
    num_characters, num_words,  
    ):
    # make sure the screen names in the quoto match all names after self. and in class.
        self.query = query
        self.user_type = user_type
        self.tweet_id = tweet_id
        self.inserted_date = inserted_date
        #self.to_user_id =  to_user_id
        self.truncated = truncated
        self.language = language
        self.possibly_sensitive = possibly_sensitive
        self.coordinates = coordinates
        self.retweeted_status = retweeted_status
        self.withheld_in_countries = withheld_in_countries
        self.withheld_scope = withheld_scope
        self.created_at_text = created_at_text
        self.created_at = created_at 
        self.month = month
        self.year = year
        self.content = content
        self.from_user_screen_name = from_user_screen_name
        self.from_user_id = from_user_id       
        self.from_user_followers_count = from_user_followers_count
        self.from_user_friends_count = from_user_friends_count
        self.from_user_listed_count = from_user_listed_count
        self.from_user_favourites_count = from_user_favourites_count
        self.from_user_statuses_count = from_user_statuses_count
        self.from_user_description = from_user_description
        self.from_user_location = from_user_location
        self.from_user_created_at = from_user_created_at
        self.retweet_count = retweet_count
        self.favorite_count = favorite_count
        self.entities_urls = entities_urls
        self.entities_urls_count = entities_urls_count        
        self.entities_hashtags = entities_hashtags
        self.entities_hashtags_count = entities_hashtags_count
        self.entities_mentions = entities_mentions
        self.entities_mentions_count = entities_mentions_count     
        self.in_reply_to_screen_name = in_reply_to_screen_name
        self.in_reply_to_status_id = in_reply_to_status_id
        self.source = source
        self.entities_expanded_urls = entities_expanded_urls
        self.json_output = json_output
        self.entities_media_count = entities_media_count
        self.media_expanded_url = media_expanded_url
        self.media_url = media_url
        self.media_type = media_type
        self.video_link = video_link
        self.photo_link = photo_link
        self.twitpic = twitpic
        self.num_characters = num_characters
        self.num_words = num_words
    

    def __repr__(self):
       return "<sender, created_at('%s', '%s')>" % (self.from_user_screen_name,self.created_at)

# This is for defining a table that has a list of screen names to be mined.         
class ACCOUNT(Base):
    __tablename__ = 'accounts' # this is the table name for a list of scree names to be mined. You need to go to SQLite Database browser and create a new DB (make sure that DB's name matches the one defined in this script); within that DB, create a table, make sure the table name and field names match the ones defined here. 
    rowid = Column(Integer, primary_key=True)     
    screen_name = Column(String)  
    user_type = Column(String) # for categorizing user's attribute (e.g. organization or individual)

    def __init__(self, screen_name, user_type
    ):       
    
        self.screen_name = screen_name
        self.user_type = user_type

   
    def __repr__(self):
       return "<Company, CSR_account('%s', '%s')>" % (self.rowid, self.screen_name)


def get_data(kid):
    try:        
        
        d = t.get_user_timeline(screen_name=kid, count="1", page="1", include_entities="true", include_rts="1")  #NEW LINE        
    
    except Exception, e:
        
        print "Error reading id %s, exception: %s" % (kid, e)
        return None
    
    print len(d), 
    return d
        



def write_data(self, d, screen_name, user_type):   
        
    query = screen_name  
    
    user_type = user_type

   
    for entry in d: 			
        print user_type
        json_output = str(entry)
        
        
        tweet_id = entry['id']
        inserted_date = datetime.now()  
    
       
     
        truncated = entry['truncated']
        language = entry['lang']
        

       
        if 'possibly_sensitive' in entry:
            possibly_sensitive= entry['possibly_sensitive']
        else:
            possibly_sensitive = ''
        
        coordinates = []
        if 'coordinates' in entry and entry['coordinates'] != None:
            print entry['coordinates']['coordinates']
            
            coordinates = ', '.join(map(str, entry['coordinates']['coordinates']))	
            print coordinates
           
        else:
            coordinates = ''
            
        if 'retweeted_status' in entry:
            retweeted_status = 'THIS IS A RETWEET'
        else:
            retweeted_status = ''  
            
        if 'withheld_in_countries' in entry:
            withheld_in_countries = 'WITHHELD --> CHECK JSON'
        else:
            withheld_in_countries = ''
     
        if 'withheld_scope' in entry:
            withheld_scope = entry['withheld_scope']
        else:
            withheld_scope = ''
       

        content = entry['text']
        content = content.replace('\n','')       
        
        created_at_text = entry['created_at']    

        
        created_at = datetime.strptime(created_at_text, '%a %b %d %H:%M:%S +0000 %Y')  
        
        created_at2 = created_at.strftime('%Y-%m-%d %H:%M:%S')  

        month = created_at.strftime('%m')
        year = created_at.strftime('%Y')
        print 'month', month, 'year', year
        
        
  
        from_user_screen_name = entry['user']['screen_name']

        from_user_id = entry['user']['id'] 
        from_user_followers_count = entry['user']['followers_count']
        from_user_friends_count = entry['user']['friends_count']   
        from_user_listed_count = entry['user']['listed_count']
        from_user_favourites_count = entry['user']['favourites_count']
        print 'from_user_favourites_count-------------->', from_user_favourites_count
        from_user_statuses_count = entry['user']['statuses_count'] 
        from_user_description = entry['user']['description'] 
        from_user_location = entry['user']['location'] 
        from_user_created_at = entry['user']['created_at']
        
        retweet_count = entry['retweet_count'] 
        favorite_count = entry['favorite_count']
        
        in_reply_to_screen_name = entry['in_reply_to_screen_name']
        in_reply_to_status_id = entry['in_reply_to_status_id']



        
        entities_urls_count = len(entry['entities']['urls'])    
        entities_hashtags_count = len(entry['entities']['hashtags'])   
        entities_mentions_count = len(entry['entities']['user_mentions']) 
    
        source = entry['source']

      
      

        entities_urls, entities_expanded_urls, entities_hashtags, entities_mentions = [], [], [], []
        
              
        for link in entry['entities']['urls']:
            if 'url' in link:
                url = link['url']
                expanded_url = link['expanded_url']
                #print link['url'], link['expanded_url']
                entities_urls.append(url)
                entities_expanded_urls.append(expanded_url)
            else:
                print "No urls in entry"

        for hashtag in entry['entities']['hashtags']:
            if 'text' in hashtag:
                tag = hashtag['text']
                #print hashtag['text']
                entities_hashtags.append(tag)
            else:
                print "No hashtags in entry"

        for at in entry['entities']['user_mentions']:
            if 'screen_name' in at:
                mention = at['screen_name']
                #print at['screen_name']
                entities_mentions.append(mention)
            else:
                print "No mentions in entry"
                
        entities_mentions = string.join(entities_mentions, u", ")
        entities_hashtags = string.join(entities_hashtags, u", ")
        entities_urls = string.join(entities_urls, u", ")
        entities_expanded_urls = string.join(entities_expanded_urls, u", ")    
        
        video_link = 0
        if 'vimeo' in entities_expanded_urls or 'youtube' in entities_expanded_urls or 'youtu' in entities_expanded_urls or 'vine' in entities_expanded_urls:
            video_link = 1					#All of these videos show up in 'View Media' in tweets
            print "FOUND A VIDEO!!!"
        else:
            video_link = 0
            
        
        if 'twitpic' in entities_expanded_urls:
            twitpic = 1						#twitpic images show up in 'View Media' in tweets
            print "FOUND A TWITPIC LINK!"
        else:
            twitpic = 0
        if 'twitpic' in entities_expanded_urls or 'instagram' in entities_expanded_urls or 'instagr' in entities_expanded_urls:
            photo_link = 1					#instagram images DO NOT show up in 'View Media' in tweets
            print "FOUND A TWITPIC OR INSTAGRAM LINK!!!"
        else:
            photo_link = 0

       
        entities_urls = unicode(entities_urls)
        entities_expanded_urls = unicode(entities_expanded_urls)
        entities_hashtags = unicode(entities_hashtags)
        entities_mentions = unicode(entities_mentions)
        
        print "urls...?....", #entities_urls 
        print "user_mentions...?....", #entities_mentions
        print "hashtags...?....", #entities_hashtags


        if 'symbols' in entry['entities']:
		    print "HERE ARE THE SYMBOLS.......", #entry['entities']['symbols']
        else:
		    print "THERE AIN'T NO entry['entities']['symbols']"
		
        if 'media' in entry['entities']:
			print "HERE ARE THE MEDIA.......", 
			entities_media_count = len(entry['entities']['media'])   
			          
        else:
            entities_media_count = ''
					
					
			
        

        if 'media' in entry['entities']:
            if 'expanded_url' in entry['entities']['media'][0]:
		        media_expanded_url = entry['entities']['media'][0]['expanded_url']
            else:
                print "THERE AIN'T NO expanded_url in entry['entities']['media']"
                media_expanded_url = ''
					    
            if 'media_url' in entry['entities']['media'][0]:
		        media_url = entry['entities']['media'][0]['media_url']
            else:
		        print "THERE AIN'T NO media_url in entry['entities']['media']"
		        media_url = ''
					    
            if 'type' in entry['entities']['media'][0]:
		        media_type = entry['entities']['media'][0]['type']
            else:
		        print "THERE AIN'T NO type in entry['entities']['media']"
		        media_type = ''
        else:
		    media_type = ''
		    media_url = ''
		    media_expanded_url = ''


      
        updates = self.session.query(TWEET).filter_by(query=query, from_user_screen_name=from_user_screen_name,
                content=content).all() 
               
        if not updates:
            print "inserting, query:", query                   
    
            upd = TWEET(query, user_type, tweet_id, inserted_date, truncated, language, possibly_sensitive, 
                coordinates, retweeted_status, withheld_in_countries, withheld_scope, created_at_text, 
                created_at, month, year, content, from_user_screen_name, from_user_id, from_user_followers_count, 
                from_user_friends_count, from_user_listed_count, from_user_favourites_count, from_user_statuses_count, from_user_description,   
                from_user_location, from_user_created_at, retweet_count, favorite_count, entities_urls, entities_urls_count,         
                entities_hashtags, entities_hashtags_count, entities_mentions, entities_mentions_count,   
                in_reply_to_screen_name, in_reply_to_status_id, source, entities_expanded_urls, json_output, 
                entities_media_count, media_expanded_url, media_url, media_type, video_link, photo_link, twitpic, None, None,
                )
            self.session.add(upd)
      
               
        else:
            if len(updates) > 1:
                print "Warning: more than one update matching to_user=%s, text=%s"\
                        % (to_user, content)
            else:
                print "Not inserting, dupe.."
        
        
        

class Scrape:
    def __init__(self):    
        engine = sqlalchemy.create_engine("sqlite:///Users/franciscojavierarceo/python_beginner.sqlite", echo=False)  # enter your file path here.
        Session = sessionmaker(bind=engine)
       
        self.session = Session()  
        Base.metadata.create_all(engine)

    
    def main(self):
    
        all_ids = self.session.query(ACCOUNT).all()
        
        keys = []
        for i in all_ids[0:]: # starting from the first (1) row
        #for i in all_ids:
            screen_name = i.screen_name
            kid = screen_name    			#THIS ROW IS DIFFERENT FROM THE MENTIONS AND DMS FILES
            rowid = i.rowid
            user_type = i.user_type

            print "\rprocessing id %s/%s  --  %s" % (rowid, len(all_ids), screen_name),
            sys.stdout.flush()


            d = get_data(kid)
            if not d:
                continue	     
          
            
            if len(d)==0:    			
                print "THERE WERE NO STATUSES RETURNED........MOVING TO NEXT ID"
                continue


                      
            write_data(self, d, screen_name, user_type)   
              
            self.session.commit()


        self.session.close()



if __name__ == "__main__":
    s = Scrape()
    s.main()
