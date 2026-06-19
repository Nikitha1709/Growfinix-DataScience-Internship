# ==========================================
# TASK 3: CHATGPT FOR DATA Q&A
# Growfinix Internship Project
# ==========================================

import pandas as pd
from datetime import datetime

print("=================================")
print("CHATGPT DATA Q&A PROJECT")
print("=================================")

# ----------------------------------
# LOAD DATASET
# ----------------------------------

try:
    df = pd.read_csv("sample_data.csv")
except FileNotFoundError:

    sample = {
        "Student": ["Arun", "Priya", "Karthik", "Divya", "Rahul"],
        "Math": [95, 88, 76, 92, 81],
        "Science": [90, 85, 80, 95, 78]
    }

    df = pd.DataFrame(sample)

    df.to_csv(
        "sample_data.csv",
        index=False
    )

print("\nDataset Loaded Successfully")

# ----------------------------------
# BASIC INFORMATION
# ----------------------------------

print("\nRows and Columns:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nSample Data:")
print(df.head())

# ----------------------------------
# DATASET SUMMARY
# ----------------------------------

print("\nDataset Summary:")
print(df.describe())

# ----------------------------------
# PREDEFINED QUESTIONS
# ----------------------------------

def answer_question(question):

    question = question.lower()

    if "top student" in question:

        topper = df.loc[
            df["Math"].idxmax()
        ]

        return (
            f"Top student in Math is "
            f"{topper['Student']} "
            f"with {topper['Math']} marks."
        )

    elif "average math" in question:

        return (
            f"Average Math Marks: "
            f"{df['Math'].mean():.2f}"
        )

    elif "highest science" in question:

        topper = df.loc[
            df["Science"].idxmax()
        ]

        return (
            f"Highest Science Marks: "
            f"{topper['Science']} "
            f"by {topper['Student']}"
        )

    elif "total students" in question:

        return (
            f"Total Students: "
            f"{len(df)}"
        )

    else:

        return (
            "Question not found in "
            "predefined questions."
        )

# ----------------------------------
# QUESTION LOG FILE
# ----------------------------------

log_file = "qa_log.csv"

try:
    logs = pd.read_csv(log_file)
except:
    logs = pd.DataFrame(
        columns=[
            "Timestamp",
            "Question",
            "Answer"
        ]
    )

# ----------------------------------
# USER INPUT
# ----------------------------------

print("\nExample Questions:")
print("1. Top student in Math")
print("2. Average Math")
print("3. Highest Science")
print("4. Total Students")

question = input(
    "\nEnter Your Question: "
)

answer = answer_question(
    question
)

print("\nAnswer:")
print(answer)

# ----------------------------------
# SAVE LOG
# ----------------------------------

new_entry = pd.DataFrame(
    {
        "Timestamp":
        [datetime.now()],

        "Question":
        [question],

        "Answer":
        [answer]
    }
)

logs = pd.concat(
    [logs, new_entry],
    ignore_index=True
)

logs.to_csv(
    log_file,
    index=False
)

print("\nQuestion Saved to Log File")

# ----------------------------------
# EXPORT REPORT
# ----------------------------------

report = open(
    "QA_Report.txt",
    "w"
)

report.write(
    "CHATGPT DATA Q&A REPORT\n"
)

report.write(
    "=====================\n\n"
)

report.write(
    f"Question: {question}\n"
)

report.write(
    f"Answer: {answer}\n"
)

report.close()

print(
    "\nReport Exported Successfully"
)

print(
    "\nProject Completed Successfully!"
)
