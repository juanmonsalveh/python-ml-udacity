# get and process input for a list of names
names =  input("Enter student names, separated by commas: ")
# get and process input for a list of the number of assignments
assignments =  input("Enter assignment counts, separated by commas: ")
# get and process input for a list of grades
grades =  input("Enter grades, separated by commas: ")

# message string to be used for each student
# HINT: use .format() with this string in your for loop
message = "Hi {},\n\nThis is a reminder that you have {} assignments left to \
submit before you can graduate. You're current grade is {} and can increase \
to {} if you submit all assignments before the due date.\n\n"

# write a for loop that iterates through each set of names, assignments, and grades to print each student's message
names = names.split(",")
assignments = assignments.split(",")
grades = grades.split(",")
data_set = zip(names, assignments, grades)
print(data_set)
for name, assignment, grade in data_set:
    potential_grade = str(int(grade) + int(assignment)*2)
    print(message.format(name,assignment,grade,potential_grade))
