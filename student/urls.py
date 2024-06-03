from django.urls import path
from student import views

urlpatterns = [
    path('add/', views.add, name='add'),
    path('edit/<int:current_pk>', views.edit, name='edit'),
    path('dels/', views.dels, name='dels'),
    path('search/', views.search, name='search'),
    path('import_student/', views.import_student, name='import_student'),

]
