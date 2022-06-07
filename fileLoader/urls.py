from rest_framework import routers
from django.urls import path
from django.conf.urls import url
from django.urls import path, include
from . import views

router = routers.DefaultRouter()

urlpatterns = [
    path('main/', views.index, name='index'),
    path('api-auth/', include('rest_framework.urls', namespace='rest_framework')),
    path('api/v1/', views.file_upload),
]