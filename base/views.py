from django.shortcuts import render, HttpResponse, redirect
from PIL import Image, ImageDraw, ImageFont

from io import BytesIO
import random

import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder

from base.reg_forms import RegForm
from student import models
from utils.res import ResponseData
from django.http import JsonResponse
from django.contrib import auth
from django.db import transaction
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
from collections import Counter


def is_ajax(request):
    return request.headers.get('X-Requested-With') == 'XMLHttpRequest'

def index(request):
    return render(request, 'base/index.html', locals())


def signin(request):
    if is_ajax(request):
        if request.method == 'POST':
            res_dict = ResponseData()
            session_code = request.session.get('auth_code')
            auth_code = request.POST.get('auth_code')
            if auth_code.upper() != session_code.upper():
                res_dict.status = 4000
                res_dict.message = '验证码输入不正确!'
                return JsonResponse(res_dict.get_dict)
            username = request.POST.get('username')
            password = request.POST.get('password')
            user_obj = auth.authenticate(request, username=username, password=password)
            if user_obj:
                auth.login(request, user_obj)
                res_dict.message = '登录成功!'
            else:
                res_dict.status = 4000
                res_dict.message = '用户名或密码输入不正确!'
            return JsonResponse(res_dict.get_dict)
    return render(request, 'base/signin.html')


def signout(request):
    auth.logout(request)
    return redirect('base:index')


def register(request):
    form_obj = RegForm()
    if is_ajax(request):
        if request.method == 'POST':
            res_dict = ResponseData()
            form_obj = RegForm(request.POST)
            if form_obj.is_valid():
                clean_data = form_obj.cleaned_data
                clean_data.pop('confirm_password')
                avatar = request.FILES.get('avatar')
                if avatar:
                    clean_data['avatar'] = avatar
                username = clean_data.get('username')
                try:
                    with transaction.atomic():
                        stu_obj = models.Student.objects.create(name=username)
                        models.UserInfo.objects.create_user(**clean_data, student=stu_obj)
                except Exception as e:
                    print(e)
                    print('服务器错误---register!')
            else:
                error_data = form_obj.errors
                res_dict.status = 4000
                res_dict.message = '数据检验失败!'
                res_dict.data = error_data
            return JsonResponse(res_dict.get_dict)

    return render(request, 'base/register.html', {'form_obj': form_obj})


def get_auth_code(request, size=(450, 35), mode="RGB", bg_color=(255, 255, 255)):
    """ 生成一个图片验证码 """
    _letter_cases = "abcdefghjkmnpqrstuvwxy"  # 小写字母，去除可能干扰的i，l，o，z
    _upper_cases = _letter_cases.upper()  # 大写字母
    _numbers = ''.join(map(str, range(3, 10)))  # 数字
    chars = ''.join((_letter_cases, _upper_cases, _numbers))

    width, height = size  # 宽高
    # 创建图形
    img = Image.new(mode, size, bg_color)
    draw = ImageDraw.Draw(img)  # 创建画笔

    def get_chars():
        """生成给定长度的字符串，返回列表格式"""
        return random.sample(chars, 4)

    def create_lines():
        """绘制干扰线"""
        line_num = random.randint(*(1, 2))  # 干扰线条数

        for i in range(line_num):
            # 起始点
            begin = (random.randint(0, size[0]), random.randint(0, size[1]))
            # 结束点
            end = (random.randint(0, size[0]), random.randint(0, size[1]))
            draw.line([begin, end], fill=(0, 0, 0))

    def create_points():
        """绘制干扰点"""
        chance = min(100, max(0, int(2)))  # 大小限制在[0, 100]

        for w in range(width):
            for h in range(height):
                tmp = random.randint(0, 100)
                if tmp > 100 - chance:
                    draw.point((w, h), fill=(0, 0, 0))

    def create_code():
        """绘制验证码字符"""
        char_list = get_chars()
        code_string = ''.join(char_list)  # 每个字符前后以空格隔开

        for i in range(len(char_list)):
            code_str = char_list[i]
            font = ImageFont.truetype('media/static/font/Rondal-Semibold.ttf', size=24)
            draw.text(((i + 1) * 75, 0), code_str, "red", font=font)

        return code_string

    create_lines()
    create_points()
    code = create_code()
    print(code)

    request.session['auth_code'] = code

    io_obj = BytesIO()
    img.save(io_obj, 'PNG')
    return HttpResponse(io_obj.getvalue())


def edit_avatar(request):
    if request.method == 'POST':
        new_avatar = request.FILES.get('new_avatar')
        if new_avatar:
            request.user.avatar = new_avatar
            request.user.save()
        return redirect('base:index')


def edit_password(request):
    if request.method == 'POST':
        old_password = request.POST.get('old_password')
        new_password = request.POST.get('new_password')
        is_right = request.user.check_password(old_password)
        if is_right:
            request.user.set_password(new_password)
            request.user.save()
            return redirect('base:signin')
        return HttpResponse('原密码错误!')


def students(request):
    student_queryset = models.Student.objects.all()
    paginator = Paginator(student_queryset, 5)  # 修改每页显示为5个学生
    total_pages = paginator.num_pages  # 不需要减1了
    current_page = request.GET.get('page', 1)
    current_page = int(current_page)
    
    # 验证当前页码是否在有效范围内
    if current_page < 1 or current_page > paginator.num_pages:
        current_page = 1
    
    previous_page = current_page - 1
    next_page = current_page + 1
    
    # 简化页码范围逻辑
    page_ranges = range(max(1, current_page - 2), min(paginator.num_pages, current_page + 3))
    
    stu_list = paginator.page(current_page)
    tem_dict = {
        'student_queryset': student_queryset,
        'paginator': paginator,
        'current_page': current_page,
        'total_pages': total_pages,
        'page_ranges': page_ranges,
        'stu_list': stu_list,
    }
    return render(request, 'base/students.html', tem_dict)


def classes(request):
    class_queryset = models.Classes.objects.all()
    return render(request, 'base/classes.html', locals())


def course(request):
    course_queryset = models.Course.objects.all()
    return render(request, 'base/course.html', locals())


def teacher(request):
    teacher_queryset = models.Teacher.objects.all()
    return render(request, 'base/teacher.html', locals())

def CanLook(request):
    return render(request,'CanLook.html',locals())

def GetData(request):
    course_queryset = models.Course.objects.all()
    course_names = list(models.Course.objects.all().values_list('name', flat=True))
    class_names = list(models.Classes.objects.all().values_list('name', flat=True))
    course_ids = list(models.Student2Course.objects.all().values_list("course_id",flat=True))
    #print(course_names)
    #print(class_names)
    #热力图数据
    course_student_pairs = list(models.Student2Course.objects.all().values_list("course_id", "student_id", flat=False))
    #print(course_student_pairs)
    # 使用Counter统计每个course_id出现的次数
   # 提取课程ID和学生ID
    course_ids = [pair[0] for pair in course_student_pairs]
    student_ids = [pair[1] for pair in course_student_pairs]

    # 使用 Counter 统计每个课程的选课人数
    course_counts = Counter(course_ids)

    #折线图数据
    courseSelectTime=[]
    courseChineseName = []

    # 输出每个课程的选课人数
    for course_id, count in course_counts.items():
        print(f"课程 {course_id} 被选了 {count} 次。")
        courseChineseName.append(course_names[course_id-1])
        courseSelectTime.append(count)

    #print(courseChineseName)
    #print(courseSelectTime)


    class_student_pairs = list(models.Student.objects.all().values_list("id", "classes", flat=False))
    #print(class_student_pairs)

    result = []
    #将学号换为班级号，用于统计
    for i in course_student_pairs:
        for j in class_student_pairs:
            if(i[1]==j[0]):
                result.append((i[0],j[1]))
    
    # print(result)
    # print(course_names)
    # print(class_names)

    XData,YData,ResultMap = returnHotData(result,course_names,class_names)
    #print(XData)
    # print(YData)
    # print(ResultMap)


    course_id_counts = Counter(course_ids)
    
    # 将统计结果转换为列表或其他所需格式
    course_id_counts_list = list(course_id_counts.items())
    
    courseNameNum = []
    courseNameCount = []
    for i in course_id_counts_list:
        courseNameNum.append(i[0])
        courseNameCount.append(i[1])
    
    courseNameString = []
    for i in courseNameNum:
        courseNameString.append(course_names[i-1])
    
    # 使用列表推导式将courseNameCount和courseNameString转换成所需的格式
    formatted_data = [
    {"value": count, "name": name}
    for count, name in zip(courseNameCount, courseNameString)
    ]
    
    #print(formatted_data)
    #print(course_names)
    #resultForest =  randomForest()

    #print(resultForest)

    

    return JsonResponse({"data":courseNameCount,
                         "courseName":courseNameString,
                         #柱状图
                         
                         #饼图
                         "pieData":formatted_data,
                         #热力图
                         "XData":class_names,
                         "YData":course_names,
                         "resultMap":ResultMap,
                         #折线图
                         "courseSelectTime":courseSelectTime,
                         "courseChineseName":courseChineseName},)


def returnHotData(result,course_id,classes_id):
    # 假设course_names和class_names已经按照课程和班级的顺序排序
    course_names = course_id
    class_names = classes_id

    # 初始化一个字典来存储每个课程和班级的选课数量
    course_class_count = {}

    # 遍历result列表，统计每个课程和班级的选课数量
    for course_id, class_id in result:
        if (course_id, class_id) in course_class_count:
            course_class_count[(course_id, class_id)] += 1
        else:
            course_class_count[(course_id, class_id)] = 1

    # 构建热力图的数据
    heatmap_data = []
    for course_name in course_names:
        for class_name in class_names:
            # 假设每个班级和每个课程的组合都存在，如果没有选课则数量为0
            count = course_class_count.get((course_names.index(course_name) + 1, class_names.index(class_name) + 1), 0)
            heatmap_data.append([class_names.index(class_name), course_names.index(course_name), count])

    # 构建x轴和y轴的数据
    x_axis_data = class_id
    y_axis_data = course_id

    # 打印热力图的数据，这里仅作为示例，实际应用中你可能需要转换为JSON或其他格式
    #print("xAxis data:", x_axis_data)
    #print("yAxis data:", y_axis_data)
    #print("Heatmap data:", heatmap_data)

    return x_axis_data,y_axis_data,heatmap_data


def randomForest():
        # 假设这是从您的Django模型中获取的 course_student_pairs 数据

    course_student_pairs = list(models.Student2Course.objects.all().values_list("course_id", "student_id", flat=False))

    

        # 提取学生ID和课程ID
    # 提取课程ID和学生ID列表
    course_ids, student_ids = zip(*course_student_pairs)

    # 使用Counter来计算每个课程被选择的次数
    course_selection_counts = Counter(course_ids)

    # 找出最受欢迎的课程，这里我们选择最受欢迎的5个课程
    most_popular_courses = course_selection_counts.most_common(5)

    # 假设新学生可能会选择最受欢迎的课程
    predicted_courses = [course for course, count in most_popular_courses]

    print(f"新学生可能选择的课程ID: {predicted_courses}")
        