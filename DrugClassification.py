from tkinter import *
import tkinter
from tkinter import filedialog
import numpy as np
from tkinter.filedialog import askopenfilename
import pandas as pd 
from tkinter import simpledialog
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn import metrics

main = tkinter.Tk()
main.title(" Multi-class Drug Classification using Machine Learning Models") 
main.geometry("1000x650")

global filename
global x_train,y_train,x_test,y_test
global X, Y
global le
global dataset
accuracy = []
precision = []
recall = []
fscore = []
global classifier
global cnn_model

def uploadDataset():
    global filename
    global dataset
    filename = filedialog.askopenfilename(initialdir = "Dataset")
    text.delete('1.0', END)
    text.insert(END,filename+' Loaded\n')
    dataset = pd.read_csv(filename)
    text.insert(END,str(dataset.head())+"\n\n")

def preprocessDataset():
    global X, Y
    global le
    global dataset
    global x_train,y_train,x_test,y_test
    le = LabelEncoder()
    text.delete('1.0', END)
    dataset.fillna(0, inplace = True)
    print(dataset.info())
    text.insert(END,str(dataset.head())+"\n\n")
    
    # Create a count plot
    sns.set(style="darkgrid")  # Set the style of the plot
    plt.figure(figsize=(8, 6))  # Set the figure size
    # Replace 'dataset' with your actual DataFrame and 'Drug' with the column name
    ax = sns.countplot(x='Drug', data=dataset, palette="Set3")
    plt.title("Count Plot")  # Add a title to the plot
    plt.xlabel("Drug Categories")  # Add label to x-axis
    plt.ylabel("Count")  # Add label to y-axis
    # Annotate each bar with its count value
    for p in ax.patches:
        ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                textcoords='offset points')

    plt.show()  # Display the plot
    le = LabelEncoder()
    dataset['Sex'] = le.fit_transform(dataset['Sex'])
    dataset['BP'] = le.fit_transform(dataset['BP'])
    dataset['Cholesterol'] = le.fit_transform(dataset['Cholesterol'])
    dataset['Drug'] = le.fit_transform(dataset['Drug'])
    X=dataset.iloc[:,0:5].values
    y=dataset.iloc[:,-1].values
    text.insert(END,"Total records found in dataset: "+str(X.shape[0])+"\n\n")
    x_train, x_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)
    text.insert(END,"Total records found in dataset to train: "+str(x_train.shape[0])+"\n\n")
    text.insert(END,"Total records found in dataset to test: "+str(x_test.shape[0])+"\n\n")
    print(x_train)

def rocGraph(testY, predict, algorithm):
    random_probs = [0 for i in range(len(testY))]
    p_fpr, p_tpr, _ = roc_curve(testY, random_probs, pos_label=1)
    plt.plot(p_fpr, p_tpr, linestyle='--', color='orange',label="True classes")
    ns_fpr, ns_tpr, _ = roc_curve(testY, predict,pos_label=1)
    plt.plot(ns_fpr, ns_tpr, linestyle='--', label='Predicted Classes')
    plt.title(algorithm+" ROC Graph")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive rate')
    plt.show()

def analysis():
    bp = dataset['BP'].value_counts()
    plt.figure(figsize=(8, 6))  # Set the figure size
    # Create a bar plot using matplotlib
    bars = plt.bar(bp.index, bp)
    plt.title('BP Count')  # Add a title to the plot
    plt.xlabel('BP Categories')  # Add label to x-axis
    plt.ylabel('Count')  # Add label to y-axis
    # Annotate each bar with its count value
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), ha='center', va='bottom', fontsize=10, color='black')
    plt.show()  # Display the plot
    
    plt.figure(figsize=(7, 5))  # Set the figure size
    # Create the count plot using Seaborn
    sns.set(style="darkgrid")  # Set the style of the plot
    ax = sns.countplot(x=dataset['Sex'], hue=dataset['Drug'], palette="Set1")
    plt.title("Count Plot by Sex and Drug")  # Add a title to the plot
    plt.xlabel("Sex")  # Add label to x-axis
    plt.ylabel("Count")  # Add label to y-axis
    # Customize the legend
    ax.legend(title="Drug", loc="upper right")
    plt.show()  # Display the plot
    
    # Create the count plot using Seaborn
    sns.set(style="darkgrid")  # Set the style of the plot
    sns.countplot(x=dataset['BP'], hue=dataset['Drug'])
    plt.title("Count Plot by BP and Drug")  # Add a title to the plot
    plt.xlabel("BP")  # Add label to x-axis
    plt.ylabel("Count")  # Add label to y-axis
    # Customize the legend
    plt.legend(title="Drug", loc="upper right")
    plt.show()  # Display the plot
    
    plt.figure(figsize=(7, 5))  # Set the figure size
    # Create the count plot using Seaborn
    sns.set(style="darkgrid")  # Set the style of the plot
    sns.countplot(x=dataset['Cholesterol'], hue=dataset['Drug'])
    plt.title("Count Plot by Cholesterol and Drug")  # Add a title to the plot
    plt.xlabel("Cholesterol")  # Add label to x-axis
    plt.ylabel("Count")  # Add label to y-axis
    # Customize the legend
    plt.legend(title="Drug", loc="upper right")
    plt.show()  # Display the plot
    

def custom_knn_classifier():
    global x_train, y_train
    
    KNN = KNeighborsClassifier(n_neighbors=10,leaf_size=30,metric='minkowski',)  # Create an instance of KNeighborsClassifier
    #x_train_reshaped = np.array(x_train).reshape(-1, 1)
    #x_test_reshaped = np.array(x_test).reshape(-1, 1)
    KNN.fit(x_train, y_train)
    predict = KNN.predict(x_test)
    p = precision_score(y_test, predict, average='macro') * 100
    r = recall_score(y_test, predict, average='macro') * 100
    f = f1_score(y_test, predict, average='macro') * 100
    a = accuracy_score(y_test, predict) * 100
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    text.insert(END, "KNN Precision : " + str(p) + "\n")
    text.insert(END, "KNN Recall    : " + str(r) + "\n")
    text.insert(END, "KNN FMeasure  : " + str(f) + "\n")
    text.insert(END, "KNN Accuracy  : " + str(a) + "\n\n")
    rocGraph(y_test, predict, "KNN")
    # Compute confusion matrix
    cm = confusion_matrix(y_test,predict)
    # Compute classification report
    report = classification_report(y_test,predict)
    # Display confusion matrix in the Text widget
    text.insert(END, "Confusion Matrix:\n")
    text.insert(END, str(cm) + "\n\n")
    # Display classification report in the Text widget
    text.insert(END, "Classification Report:\n")
    text.insert(END, report)
   

def Randomforestclassifier():
    global x_train, y_train, x_test, y_test
    
    rf=RandomForestClassifier()
    rf.fit(x_train, y_train)
    
    predict = rf.predict(x_test)
    
    p = precision_score(y_test, predict, average='macro', zero_division=0) * 100
    r = recall_score(y_test, predict, average='macro', zero_division=0) * 100
    f = f1_score(y_test, predict, average='macro', zero_division=0) * 100
    a = accuracy_score(y_test, predict) * 100
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    # Display precision, recall, F1-score, and accuracy in the Text widget
    text.insert(END, "RF Precision: " + str(p) + "\n")
    text.insert(END, "RF Recall: " + str(r) + "\n")
    text.insert(END, "RF FMeasure: " + str(f) + "\n")
    text.insert(END, "RF Accuracy: " + str(a) + "\n\n")
    rocGraph(y_test, predict, "RFC")
    # Compute confusion matrix
    cm = confusion_matrix(y_test, predict)
    
    # Compute classification report
    report = classification_report(y_test, predict)
    
    # Display confusion matrix in the Text widget
    text.insert(END, "Confusion Matrix:\n")
    text.insert(END, str(cm) + "\n\n")
    
    # Display classification report in the Text widget
    text.insert(END, "Classification Report:\n")
    text.insert(END, report)

def Performance():
    rf=RandomForestClassifier()
    rf.fit(x_train, y_train)
    
    predict = rf.predict(x_test)
    
    cm = confusion_matrix(y_test, predict)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    report= classification_report(y_test, predict)
    print(report)
def graph():
    # Create a DataFrame
    df = pd.DataFrame([
    ['KNN', 'Precision', precision[0]],
    ['KNN', 'Recall', recall[0]],
    ['KNN', 'F1 Score', fscore[0]],
    ['KNN', 'Accuracy', accuracy[0]],
    ['rf', 'Precision', precision[-1]],
    ['rf', 'Recall', recall[-1]],
    ['rf', 'F1 Score', fscore[-1]],
    ['rf', 'Accuracy', accuracy[-1]],
    ], columns=['Parameters', 'Algorithms', 'Value'])

    # Pivot the DataFrame and plot the graph
    pivot_df = df.pivot_table(index='Parameters', columns='Algorithms', values='Value', aggfunc='first')
    pivot_df.plot(kind='bar')
    # Set graph properties
    plt.title('Classifier Performance Comparison')
    plt.ylabel('Score')
    plt.xticks(rotation=0)
    plt.tight_layout()
    # Display the graph
    plt.show()
def close():
    main.destroy()

font = ('times', 16, 'bold')
title = Label(main, text=' Multi-class Drug Classification using Machine Learning Models', justify=LEFT)
title.config(bg='lavender blush', fg='black')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=100,y=5)
title.pack()

font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Upload Dataset", command=uploadDataset)
uploadButton.place(x=200,y=100)
uploadButton.config(font=font1)

preprocessButton = Button(main, text="Preprocess Dataset", command=preprocessDataset)
preprocessButton.place(x=500,y=100)
preprocessButton.config(font=font1) 

analysisButton = Button(main, text="Data Analysis", command=analysis)
analysisButton.place(x=200,y=150)
analysisButton.config(font=font1) 

knnButton = Button(main, text="KNeighborsClassifier", command=custom_knn_classifier)
knnButton.place(x=500,y=150)
knnButton.config(font=font1)

LRButton = Button(main, text="Randomforestclassifier", command=Randomforestclassifier)
LRButton.place(x=200,y=200)
LRButton.config(font=font1)

predictButton = Button(main, text="Performance_Evaluation", command=Performance)
predictButton.place(x=500,y=200)
predictButton.config(font=font1)

graphButton = Button(main, text="Comparison Graph", command=graph)
graphButton.place(x=200,y=250)
graphButton.config(font=font1)

exitButton = Button(main, text="Exit", command=close)
exitButton.place(x=500,y=250)
exitButton.config(font=font1)

                            

font1 = ('times', 12, 'bold')
text=Text(main,height=20,width=120)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=300)
text.config(font=font1) 

main.config(bg='LightSteelBlue1')
main.mainloop()
