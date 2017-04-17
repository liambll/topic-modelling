# -*- coding: utf-8 -*-
from django.conf.urls import url

from . import views

urlpatterns = [
    url(r'^index$', views.index, name='index'),
    url(r'^browse', views.browse, name='browse'),
    url(r'^search', views.search, name='search'),
    url(r'^analysis', views.analysis, name='analysis')
]