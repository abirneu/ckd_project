from django.urls import path
from . import views

urlpatterns = [
    # Define your URL patterns here
    path('', views.home, name='home'),
    path('predict/', views.predict, name='predict'),
    # path('group/', views.group, name='group'),
    path('group/', views.predict_grp, name='group'),  # Ensure this matches the form's URL
    # path('print_grp/', views.print_report, name='print_grp'),
    # path('group/', views.print_report, name='group'),
    path('report_grp/', views.print_grp, name='report_grp'),
    path('report_ckd/', views.report_ckd, name='report_ckd'),





]
