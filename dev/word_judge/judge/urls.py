from django.urls import path
from .views import judge

urlpatterns = [
    path('', judge,),
]