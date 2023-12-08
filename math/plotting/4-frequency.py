#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

plt.hist(student_grades, bins=np.arange(0, 110, 10), edgecolor='black')
plt.xticks(np.arange(0, 110, 10))
plt.xlabel("Number of Students")
plt.ylabel("Grades")
plt.xlim(0,100)
plt.ylim(0,30)
plt.title("Project A")
plt.show()
