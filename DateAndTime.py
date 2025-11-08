import datetime

# for more detail can refer the date time module 

date= datetime.date(2025,1,2)
#datetime.date(year,month,date )
print(date)

 # this will print today date 
today=datetime.date.today()
print(today)

#print time 
time=datetime.time(12,30,0)
#datetime.time(hour,min,sec)
print(time)

#print current time
now=datetime.datetime.now()
print(now)
#output : 2025-04-06 23:47:29.369832 

now = now.strftime("%H:%M:%S %m /%d /%Y")
print(now)

target_datetime= datetime.datetime(2020,1,2,12,30,1)

currentDateTime= datetime.datetime.now()

if target_datetime< currentDateTime:
    print("Target date has pass")
else:
    print("target date has not passed")

#multireading 0 use to perfome multiple task concurrently (multitasking)
#Good for Input/Output bound task like reading files or fetching data from API
#threading.Thread(target=my_function)

import threading
import time

def walk_dog(first):
    time.sleep(8)
    print(f"You finish walking the {first}")

def takeOutTrash():
    time.sleep(2)
    print("You take out the trash")

def getMail():
    time.sleep(4)
    print("You get the mail")

print()
walk_dog("Joana")
takeOutTrash()
getMail()
# the follwing method will return the value by smallest time to longest time 
print()

chore1=threading.Thread(target=walk_dog,args=("Scooby",))
chore1.start()

chore2=threading.Thread(target=takeOutTrash)
chore2.start()

chore3=threading.Thread(target=getMail)
chore3.start()

print()
# if you want to print value after all the chores are finihs use .join()
chore1.join()
chore2.join()
chore3.join()
print("all chores are complete ")