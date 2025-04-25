from django.db import models


# Create your models here.
class Student(models.Model):
    name = models.CharField(max_length=50)
    matriculation_nr = models.CharField(max_length=30)


class Project(models.Model):
    project_name = models.CharField(max_length=50)
    project_URL = models.CharField(max_length=100)
