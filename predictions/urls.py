from django.urls import path
from .views import PredictView, Home

urlpatterns = [
    path('', Home.as_view(), name='home'),
    path('predict/', PredictView.as_view(), name='predict'),
]
