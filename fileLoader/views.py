from django.core.files.storage import FileSystemStorage
from django.conf import settings
from django.http import HttpResponse, JsonResponse
from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.parsers import JSONParser
from rest_framework.response import Response
from django.db import models
from rest_framework import status

from fileLoader import LeNet5, File, AudioController, FileUploadController


def index(request):
    if request.method == 'POST' and request.FILES['uploaded_file']:

        net = LeNet5()
        net.load_state()

        uploaded_file = request.FILES['uploaded_file']

        file = File(uploaded_file.name, settings.MEDIA_ROOT)
        FileUploadController.save_file(file, uploaded_file)

        if not FileUploadController.check_size(file):
            return render(request, 'fileLoader/main.html', {
                'error': "file too big",
            })

        if not FileUploadController.check_extension(file):
            return render(request, 'fileLoader/main.html', {
                'error': "File should have the extension .mp3 or .wav",
            })

        audio = AudioController.toAudio(file.path)
        spectrogram = AudioController.toSpectro(audio)



        res = net.get_predicted_genre(spectrogram.get_tensor())

        return render(request, 'fileLoader/main.html', {
            'test': res,
            'file_name': file.name
        })
    return render(request, 'fileLoader/main.html')


def load(request):
    return HttpResponse(request.POST)


@api_view(["POST"])
def file_upload(request):
    uploaded_file = request.FILES['file']

    file = File(uploaded_file.name, settings.MEDIA_ROOT)
    FileUploadController.save_file(file, uploaded_file)


    if not FileUploadController.check_size(file):
        return Response({'error': "file too big"}, status=status.HTTP_400_BAD_REQUEST)

    if not FileUploadController.check_extension(file):
        return Response({"error": "File should have the extension .mp3 or .wav"}, status=status.HTTP_400_BAD_REQUEST)

    audio = AudioController.toAudio(file.path)
    spectrogram = AudioController.toSpectro(audio)

    net = LeNet5()
    net.load_state("fileLoader/params.py.py")
    res = net.get_predicted_genre(spectrogram.get_tensor())

    return Response(res, status=status.HTTP_200_OK)
