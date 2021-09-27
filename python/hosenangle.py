import pyrealsense2 as rs
# Import Numpy for easy array manipulation
import numpy as np
# Import OpenCV for easy image rendering
import cv2
import math
import csv
from matplotlib import pyplot as plt
from scipy.linalg import norm
def AngleCal(veca,vecb):
    angle=np.arccos(np.dot(veca,vecb)/(norm(veca)*norm(vecb)))*(180/math.pi)
    return angle
def Floor(angle):
    if np.isnan(angle):
        y=0
    else:
        y=math.floor(angle)
    return y
def conv(s):
    try:
        s=float(s)
    except ValueError:
        pass    
    return s
def R(x,y,vec):
    rotatex=np.array([[1,0,0],[0,np.cos(x),-np.sin(x)],[0,np.sin(x),np.cos(x)]])
    rotatey=np.array([[np.cos(y),0,np.sin(y)],[0,1,0],[-np.sin(y),0,np.cos(y)]])
    Ax=np.dot(rotatey,vec)
    Ay=np.dot(rotatex,Ax)
    return Ay
def R2(x,y,vec):
    rotatex=np.array([[1,0,0],[0,np.cos(x),-np.sin(x)],[0,np.sin(x),np.cos(x)]])
    rotatey=np.array([[np.cos(y),0,np.sin(y)],[0,1,0],[-np.sin(y),0,np.cos(y)]])
    Ax=np.dot(rotatex,vec)
    Ay=np.dot(rotatey,Ax)
    return Ay

def identify_outliers(ys):
    quartile_1, quartile_3 = np.percentile(ys, [25, 75])
    iqr = quartile_3 - quartile_1
    # 下限
    lower_bound = quartile_1 - (iqr * 7)
    # 上限
    upper_bound = quartile_3 + (iqr * 7)

    return np.array(ys)[((ys > quartile_3+5) | (ys < quartile_1-5))]
def identify_outlier2s(ys):
    quartile_1, quartile_3 = np.percentile(ys, [25, 75])
    iqr = quartile_3 - quartile_1
    # 下限
    lower_bound = quartile_1 - (iqr * 7)
    # 上限
    upper_bound = quartile_3 + (iqr * 7)
    #print(upper_bound,lower_bound)
    return np.array(ys)[((ys > quartile_3+5) | (ys < quartile_1-5))]
###########

A=np.zeros((3))
B=np.zeros((6,3))
flag=False
x,y,z=0,0,0
dataxlist=[]
dataylist=[]
datazlist=[]
mark_point=[]
meanB=np.zeros((10,3))
meanG=np.zeros((10,3))
meanR=np.zeros((10,3))
nx=[0,0,0]
ny=[0,0,0]
nz=[0,0,0]
Angle=30
tempx,tempy,tempz=[0,0,0],[0,0,0],[0,0,0]
anglez=0
x,y,z=0,0,0
bx,by,bz=0,0,0
ax,ay,az=0,0,0
N=0
pitch=np.zeros((N))
roll=np.zeros((N))
yaw=np.zeros((N))
csvname='Hosen1.csv'
with open(csvname,'w',newline="") as f:
     writer = csv.writer(f)
     writer.writerow(["内積","角度"])
with open('dataXYtest303.csv') as f:
    reader = csv.reader(f)
    frame=0
    count=0
    for rows in reader:
        if frame>0:
            row = [conv(s) for s in rows]
            pointB=[row[0],row[1],row[2]]
            pointG=[row[3],row[4],row[5]]
            pointR=[row[6],row[7],row[8]]
            base=row[15]
            flag3=row[16]
            frag4="False"
            meanB[frame%10]=pointB
            meanG[frame%10]=pointG
            meanR[frame%10]=pointR
            #if (frame==805):
                #print(pointR[2])
                #print(identify_outlier2s(meanR[:,2]))
                #print(meanR[:,2])
            if flag:
                p,q,r=identify_outliers(meanR[:,0]),identify_outliers(meanR[:,1]),identify_outliers(meanR[:,2])
                if (len(p)>0)|(len(q)>0)|(len(r)>0):
                    pointR=meanR[(frame-1)%10]
                    meanR[frame%10]=pointR
                p,q,r=identify_outliers(meanG[:,0]),identify_outliers(meanG[:,1]),identify_outliers(meanG[:,2])
                if (len(p)>0)|(len(q)>0)|(len(r)>0):
                    pointG=meanG[(frame-1)%10]
                    meanG[frame%10]=pointG
                p,q,r=identify_outliers(meanB[:,0]),identify_outliers(meanB[:,1]),identify_outliers(meanB[:,2])
                if (len(p)>0)|(len(q)>0)|(len(r)>0):
                    pointB=meanB[(frame-1)%10]
                    meanB[frame%10]=pointB

            
            pointG2=[np.mean(meanG[:,0]),np.mean(meanG[:,1]),np.mean(meanG[:,2])]
            pointR2=[np.mean(meanR[:,0]),np.mean(meanR[:,1]),np.mean(meanR[:,2])]
            pointB2=[np.mean(meanB[:,0]),np.mean(meanB[:,1]),np.mean(meanB[:,2])]

            vecgr=[pointG2[0]-pointR2[0],pointG2[1]-pointR2[1],pointG2[2]-pointR2[2]]
            vecbr=[pointB2[0]-pointR2[0],pointB2[1]-pointR2[1],pointB2[2]-pointR2[2]]
            
            B=np.cross(vecgr,vecbr)
            if base=='True':
                A=np.cross(vecgr,vecbr)
                flag=True
                nz=[vecbr[1],-vecbr[0],0]
                nx=[0,A[2],-A[1]]
                ny=[-A[2],0,A[0]]
                
            angle=AngleCal((A), B)
            
               
            tempx=[0,B[1],B[2]]
            tempy=[B[0],0,B[2]]
            angley = AngleCal(tempy, ny)-90
            anglex = AngleCal(tempx, nx)-90
            C=R((0),(-30*math.pi/180),A)
            C0=R((Angle*math.pi/180),(0*math.pi/180),A)
            C1=R((Angle*math.pi/180),(10*math.pi/180),A)
            C2=R((Angle*math.pi/180),(20*math.pi/180),A)
            C3=R((Angle*math.pi/180),(-30*math.pi/180),A)
            C33=R2((Angle*math.pi/180),(-30*math.pi/180),A)

            C4=R((Angle*math.pi/180),(40*math.pi/180),A)
            C5=R((Angle*math.pi/180),(50*math.pi/180),A)
            C6=R((Angle*math.pi/180),(60*math.pi/180),A)
            P=(np.dot(C,B))/(norm(C)*norm(B))
            PP=AngleCal(C,B)
            P0=(np.dot(C0,B))/(norm(C0)*norm(B))
            P02=AngleCal(C0,B)
            P1=(np.dot(C1,B))/(norm(C1)*norm(B))
            P12=AngleCal(C1,B)
            P2=(np.dot(C2,B))/(norm(C2)*norm(B))
            P22=AngleCal(C2,B)            
            P3=(np.dot(C3,B))/(norm(C3)*norm(B))
            P32=AngleCal(C3,B)            
            P33=(np.dot(C33,B))/(norm(C33)*norm(B))
            P332=AngleCal(C33,B)         
            P4=(np.dot(C4,B))/(norm(C4)*norm(B))
            P42=AngleCal(C4,B)            
            P5=(np.dot(C5,B))/(norm(C5)*norm(B))
            P52=AngleCal(C5,B)
            P6=(np.dot(C6,B))/(norm(C6)*norm(B))
            P62=AngleCal(C6,B)
            
            ypi=-angley * math.pi / 180
            xpi=-anglex * math.pi / 180
            xrotate=vecbr
            yrotate2=R(xpi,ypi,xrotate)
            if flag3=='True':
                mark_point.append(len(dataylist))
                
            #93.2000041
            if np.isnan(yrotate2[0]):
                    ztemp=[0,0,0]
            else:
                    ztemp=[yrotate2[0],yrotate2[1],0]
                    anglez=AngleCal(ztemp, nz)-90

            if flag:
                dataxlist.append((anglex))
                dataylist.append((angley))
                datazlist.append((anglez))
                count=count+1
                with open(csvname,'a',newline="") as f:
                    writer = csv.writer(f)
                    #writer.writerow([pointB[0],pointB[1],pointB[2],pointG[0],pointG[1],pointG[2],pointR[0],pointR[1],pointR[2],base])
                    #writer.writerow([vecbr[0],vecbr[1],vecbr[2],vecgr[0],vecgr[1],vecgr[2],B[0],B[1],B[2],angley,A[0],A[1],A[2]])
                    #writer.writerow([B[0],B[1],B[2],angley*100])
                    #writer.writerow([P0,P02,P1,P12,P2,P22,P3,P32,P4,P42,P5,P52,P6,P62,A[0],A[1],A[2],B[0],B[1],B[2],base,flag3,anglex,angley])
                    writer.writerow([P,PP,P0,P02,P3,P32,P33,P332,A[0],A[1],A[2],B[0],B[1],B[2],base,flag3,anglex,angley])
                    #writer.writerow([C[0],C[1],C[2],C0[0],C0[1],C0[2],C3[0],C3[1],C3[2],C33[0],C33[1],C33[2],B[0],B[1],B[2],base,flag3,anglex,angley])


                """
                with open(csvname,'a',newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([p])

                """
            
        frame=frame+1
                
x = np.linspace(0, len(dataxlist), len(dataxlist))
y = np.linspace(0, len(dataylist), len(dataylist))
z = np.linspace(0, len(datazlist), len(datazlist))
plt.figure()
plt.plot(x, dataxlist, label="test",marker="D",mfc='orange',ms=7,markevery=mark_point)
#plt.savefig("Hmeanx.png")
plt.figure()
plt.plot(y, dataylist, label="test",marker="D",mfc='orange',ms=7,markevery=mark_point)
#plt.savefig("Hmeany.png")
