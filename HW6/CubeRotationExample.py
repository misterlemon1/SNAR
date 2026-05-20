from HW6.QUAT import Quaternion



vertices={
    "A1":[1,1,1],
    "A2":[1,1,-1],
    "B1":[1,-1,1],
    "B2":[1,-1,-1],
    "C1":[-1,1,1],
    "C2":[-1,1,-1],
    "D1":[-1,-1,1],
    "D2":[-1,-1,-1]
}#кубик с центром в 0


qvert={key:Quaternion([0]+i) for key,i in vertices.items()}

center1=[0,0,0]#Центр отн. вращения
axis1=[0,0,1]#ось вращения
deg1=45
qc1=Quaternion([0]+center1)

center2=[0,0,-1]
axis2=[0,0,1]
deg2=45
qc2=Quaternion([0]+center2)

center3=[-1,-1,-1]
axis3=[0,0,1]
deg3=45
qc3=Quaternion([0]+center3)

vertices1={}
vertices2={}
vertices3={}

for key,q in qvert.items():
    vertices1[key]=((q-qc1).rotate_deg(axis1,deg1)+qc1).vector3
    vertices2[key] =((q-qc2).rotate_deg(axis2, deg2)+qc2).vector3
    vertices3[key] =((q-qc3).rotate_deg(axis3, deg3)+qc3).vector3

print(vertices1)
print(vertices2)
print(vertices3)
#Визуализируем самостоятельно