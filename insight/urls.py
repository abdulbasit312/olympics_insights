from django.urls import path
from . import views
urlpatterns=[path('',views.trial,name='trial'),
            path('query',views.get_query,name="search")]