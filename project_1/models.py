from django.db import models


class Action(models.Model):
    action_name = models.CharField(max_length=50)
    action_URL = models.CharField(max_length=100)
