from django.shortcuts import render
from django.conf import settings

from .findSimilarPapers import get_most_similar_title
import json

def home(request):
    queryTitle, papersFound = '', []
    if request.method == 'POST':
        queryTitle = request.POST.get('queryTitle', '')
        print('Query Title: '+request.POST['queryTitle'])

        papersFound = get_most_similar_title(queryTitle)

    return render(request, "home.html", {'queryTitle':queryTitle,'papersFound':papersFound})
