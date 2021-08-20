class Student():
    def __init__(self,args):
        self.info = [args[0],args[1],args[2]]
        self.gpa = round(args[3],2)
    
    def student_info(self):
        print(' '.join(self.info)+' '+str(self.gpa))

    def study(self,time):
        self.gpa += 0.05*time
        self.gpa = min(round(self.gpa,2),4.0)
infos = [
    ['Aaron','Fiksel','freshman',2.4],
    ['Mia','Hong','sophomore',3.8],
    ['Irene','Yang','junior',3.1],
    ['Brandon','Shah','senior',3.5],
    ['Xu','Henderson','sophomore',2.7]
]

# 创建学生对象并存放于roster列表中
roster = []
for info in infos:
    roster.append(Student(info))

# 通过student_info函数打印每个学生信息
for student in roster:
    student.student_info()

# 每个学生学习指定时长
study_time = [15,10,4,2,13]
for student,time in zip(roster,study_time):
    student.study(time)

# 再次调用student_info，以查看学习后的gpa变化情况
for student in roster:
    student.student_info()