from django.shortcuts import render
from judge_text import judge_tools

# Create your views here.
def judge(req):
    judge_result="score:"
    text="文章を書き込んでみてね"
    result=""
    if req.method=="POST":
        text=req.POST["value"]
        judge_result = judge_tools.judgefunc(text)
        if judge_result>=0.5:
            result="この文章は誹謗中傷の可能性があります"
        else:
            result="この文章は誹謗中傷ではありません"
    params = {
        "title":"これはタイトルです",
        "text":text,
        "judge":judge_result[0],
        "result":result
    }
    return render(req, "index.html", params)
