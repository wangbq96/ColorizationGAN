from django.shortcuts import render
from django.shortcuts import redirect
from questionnaire.models import Question
from questionnaire.models import Image
from questionnaire.models import User
import uuid
import random


# Create your views here.
def home_page(request):
    return render(request, 'home.html')


def submit_user(request):
    if request.method == 'POST':
        name = request.POST['name']
        user_uuid = str(uuid.uuid1())
        User.objects.create(name=name, uuid=user_uuid)
        return redirect('/question/{}/1/'.format(user_uuid))


def question(request, user, q_idx):
    if request.method == 'POST':
        # 保存上个问题的结果
        # Question.objects.create(
        #     user=user,
        #     img1=request.POST['img1'],
        #     img2=request.POST['img2'],
        #     answer=request.POST['answer'],
        # )
        return redirect('/question/{}/{}/'.format(user, q_idx+1))

    else:
        if q_idx < 0 or q_idx > 20:
            # over
            return redirect('/finish/')

        img1, img2, gray_img = get_images()
        context = {
            'q_idx': q_idx,
            'user': user,
            'img1': img1,
            'img2': img2,
            'gray_img': gray_img,
        }
        return render(request, 'question.html', context)


def finish(request):
    return render(request, 'finish.html')


def get_images():
    total = int(Image.objects.count())
    result = Image.objects.get(id=random.randint(1, total))

    fake_img = result.fake_image
    real_img = result.real_image
    gray_img = result.gray_image

    if random.random() < 0.5:
        img1 = fake_img
        img2 = real_img
    else:
        img1 = real_img
        img2 = fake_img

    return img1, img2, gray_img
