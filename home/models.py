from django.db import models

# Create your models here.
class Person(models.Model):
    name = models.CharField(max_lenght = 50)
    matriculation_nr = models.CharField(max_lenght = 30)

class Project(models.Model):
    project_name = models.Charfield(max_lenght = 50)
    project_URL = models.CharField(max_lenght = 100)


