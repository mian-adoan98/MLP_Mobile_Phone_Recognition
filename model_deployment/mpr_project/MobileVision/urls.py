from django.urls import path
from . import views

# Create a list of url patterns
urlpatterns = [
  path("", views.index),
]