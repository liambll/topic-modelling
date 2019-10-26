#from django.shortcuts import render

# mongod.exe --dbpath D:\Training\Software\MongoDB\data
# python manage.py runserver
# http://127.0.0.1:8000/topic_modeling/index
# Create your views here.
from django.http import HttpResponse
import pandas as pd
import numpy as np
from pymongo import MongoClient
import matplotlib.pyplot as plt
from spark_celery.tasks import topicPredict

def index(request):
    topics = pd.read_csv("topics.csv").sort_values(by=["Label"], ascending=[True])
    topics.index = range(topics["Id"].count())
    output = "<h2>Index <a href='search'>Search</a></h2>"
    output += "<table border=1>"
    output += "<tr><th>No</th><th>Topic</th></tr>"
    for i in range(topics["Id"].count()):
        output += "<tr><td>" + str(i+1) + "</td><td><a href='browse?id=" + str(topics.loc[i,"Id"]) +"'>" + topics.loc[i,"Label"] + "</a></td></tr>"
    output += "</table>"
    return HttpResponse(output)

def browse(request):
    N_RECORDS = 10
    N_TOPICS = 5
    
    topicId =  request.GET.get('id')
    topics = pd.read_csv("topics.csv")
    topic_names = topics["Label"].values
    client = MongoClient("mongodb://localhost:27017")
    #client = MongoClient("mongodb://gateway.sfucloud.ca:27017")
    db = client['publications']
    collection = db['topicDistribution']
    
    output = "<h2><a href='index'>Index</a> <a href='search'>Search</a></h2>"
    output += "<h3>Topic: " + topic_names[int(topicId)] + "</h3>"
    output += "<table border=1>"
    output += "<tr><th>No</th><th>Title</th><th>Abstract</th><th>Topic Contribution</th></tr>"
    i = 0
    for document in collection.find().sort([("topic_distribution."+str(topicId), -1)]):
        if (i >= N_RECORDS):
            break
        i = i + 1
        
        # get chart
        topicDistribution = np.argsort(document["topic_distribution"])
        topic_percentage = np.flip(100*np.array(document["topic_distribution"])[topicDistribution[-N_TOPICS:]],0)
        topic_label = np.flip(topic_names[topicDistribution[-N_TOPICS:]], 0)
        fig, ax = plt.subplots(figsize=(5,3))
        y_pos = np.arange(len(topic_label))
        ax.barh(y_pos, topic_percentage, color='green', align='center')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(topic_label)
        ax.invert_yaxis()  # labels read top-to-bottom
        ax.set_xlabel('(%)')
        plt.tight_layout()
        plt.savefig("topic_modeling/static/image_"+str(i)+".png")
        
        output += "<tr><td>" + str(i) + "</td><td><a href='" + document["url"] +"' target='new'>" + document["title"] + \
            "</a></td><td>" + document["abstract"] + "</td><td><img src='/static/image_" + str(i) + ".png'/></td></tr>"
    
    output += "</table>"
    return HttpResponse(output)

def search(request):
    output = "<h2><a href='index'>Index</a> Search</h2>"
    output += "<form action='analysis' method='get' id='queryForm'>"
    output += "<input type='Submit'>"
    output += "</form>"
    output += "<textarea name='query' rows='4' cols='50' form='queryForm'></textarea>"
    
    return HttpResponse(output)

def analysis(request):
    N_RECORDS = 10
    N_TOPICS = 5
    topics = pd.read_csv("topics.csv")
    topic_names = topics["Label"].values

    query =  request.GET.get('query')
    result = topicPredict(query)
    
    output = "<h2><a href='index'>Index</a> <a href='search'>Search</a></h2>"
    output += "<b>Query: </b>" + query
    output += "<table border=1>"
    output += "<tr><th>No</th><th>Title</th><th>Abstract</th><th>Topic Contribution</th></tr>"
    i = 0
    for document in result:
        if (i >= N_RECORDS):
            break
        i = i + 1
        
        # get chart
        topicDistribution = np.argsort(document[4])
        topic_percentage = np.flip(100*np.array(document[4])[topicDistribution[-N_TOPICS:]],0)
        topic_label = np.flip(topic_names[topicDistribution[-N_TOPICS:]], 0)
        fig, ax = plt.subplots(figsize=(5,3))
        y_pos = np.arange(len(topic_label))
        ax.barh(y_pos, topic_percentage, color='green', align='center')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(topic_label)
        ax.invert_yaxis()  # labels read top-to-bottom
        ax.set_xlabel('(%)')
        plt.tight_layout()
        plt.savefig("topic_modeling/static/image_"+str(i)+".png")
        
        output += "<tr><td>" + str(i) + "</td><td><a href='" + document[3] +"' target='new'>" + document[1] + \
            "</a></td><td>" + document[2] + "</td><td><img src='/static/image_" + str(i) + ".png'/></td></tr>"
    
    output += "</table>"
    
    #output = result
    return HttpResponse(output)
    
    
    
    
    