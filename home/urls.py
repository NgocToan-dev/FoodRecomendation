import imp
from django.urls import path

from .views import *

urlpatterns = [
    path('homePage/',HomePageView.as_view(),name='homePageView'),
    path("login/", LoginView.as_view(), name="loginView"),
    path("foodDetail/<int:pk>",FoodDetail.as_view(),name="foodDetail")
]