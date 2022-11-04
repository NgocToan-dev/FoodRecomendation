from django.db import models

# Create your models here.

class User(models.Model):
    userID = models.AutoField(primary_key=True)
    username = models.CharField(max_length=255)
    password = models.CharField(max_length=255)
    phone = models.CharField(max_length=255)
    address = models.CharField(max_length=255)
    role = models.IntegerField()

    def __str__(self):
        return self.username