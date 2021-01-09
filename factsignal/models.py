from django.db import models

class Main(models.Model):
    result = models.CharField(max_length=100)