from django.db import models


class File(models.Model):
    name = models.CharField(max_length=50, default="")
    path = models.CharField(max_length=200, default=0)

    def __str__(self):
        return self.name


class Audio(models.Model):
    duration = models.IntegerField(default=0)

