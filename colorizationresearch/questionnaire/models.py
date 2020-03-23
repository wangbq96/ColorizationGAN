from django.db import models
from django.utils import timezone


# Create your models here.
class Question(models.Model):
    user = models.TextField(default='')
    img1 = models.TextField(default='')
    img2 = models.TextField(default='')
    answer = models.TextField(default='')
    time = models.DateTimeField(auto_now_add=timezone.now)


class Image(models.Model):
    fake_image = models.TextField(default='')
    real_image = models.TextField(default='')
    gray_image = models.TextField(default='')
    model_name = models.TextField(default='')


class User(models.Model):
    name = models.TextField(default='')
    uuid = models.TextField(default='')
