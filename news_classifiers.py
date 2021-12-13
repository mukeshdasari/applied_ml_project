import process_data
import select_feature
import numpy as np
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import  LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, f1_score, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

#string to test
doc_new = ['Trump is running for president in 2016']

#building classifier using naive bayes 
nb_pipeline = Pipeline([
        ('NBCV',select_feature.countV),
        ('nb_clf',MultinomialNB())])

nb_pipeline.fit(process_data.train_news['Statement'],process_data.train_news['Label'])
predicted_nb = nb_pipeline.predict(process_data.test_news['Statement'])
np.mean(predicted_nb == process_data.test_news['Label'])


#building classifier using logistic regression
logR_pipeline = Pipeline([
        ('LogRCV',select_feature.countV),
        ('LogR_clf',LogisticRegression(max_iter=10000))
        ])

logR_pipeline.fit(process_data.train_news['Statement'],process_data.train_news['Label'])
predicted_LogR = logR_pipeline.predict(process_data.test_news['Statement'])
np.mean(predicted_LogR == process_data.test_news['Label'])


#building Linear SVM classfier
svm_pipeline = Pipeline([
        ('svmCV',select_feature.countV),
        ('svm_clf',svm.LinearSVC())
        ])

svm_pipeline.fit(process_data.train_news['Statement'],process_data.train_news['Label'])
predicted_svm = svm_pipeline.predict(process_data.test_news['Statement'])
np.mean(predicted_svm == process_data.test_news['Label'])


#using SVM Stochastic Gradient Descent on hinge loss
sgd_pipeline = Pipeline([
        ('svm2CV',select_feature.countV),
        # ('svm2_clf',SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=10))
        ('svm2_clf',SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3))
        ])

sgd_pipeline.fit(process_data.train_news['Statement'],process_data.train_news['Label'])
predicted_sgd = sgd_pipeline.predict(process_data.test_news['Statement'])
np.mean(predicted_sgd == process_data.test_news['Label'])


#random forest
random_forest = Pipeline([
        ('rfCV',select_feature.countV),
        ('rf_clf',RandomForestClassifier(n_estimators=200, n_jobs=3))
        ])
    
random_forest.fit(process_data.train_news['Statement'],process_data.train_news['Label'])
predicted_rf = random_forest.predict(process_data.test_news['Statement'])
np.mean(predicted_rf == process_data.test_news['Label'])


#User defined functon for K-Fold cross validatoin
def create_confusion_matrix(classifier):
    
    k_fold = KFold(n_splits=5)
    scores = []
    confusion = np.array([[0,0],[0,0]])
    for train_ind, test_ind in k_fold.split(process_data.train_news):
        train_text = process_data.train_news.iloc[train_ind]['Statement'] 
        train_y = process_data.train_news.iloc[train_ind]['Label']
    
        test_text = process_data.train_news.iloc[test_ind]['Statement']
        test_y = process_data.train_news.iloc[test_ind]['Label']
        
        classifier.fit(train_text,train_y)
        predictions = classifier.predict(test_text)
        
        confusion += confusion_matrix(test_y,predictions)
        score = f1_score(test_y,predictions)
        scores.append(score)
    
    return (print('Total statements classified:', len(process_data.train_news)),
    print('Score:', sum(scores)/len(scores)),
    print('score length', len(scores)),
    print('Confusion matrix:'),
    print(confusion))
    
print('---------------------Bag of Words---------------------')
create_confusion_matrix(nb_pipeline)
create_confusion_matrix(logR_pipeline)
create_confusion_matrix(svm_pipeline)
create_confusion_matrix(sgd_pipeline)
create_confusion_matrix(random_forest)



#using n-grams
#naive-bayes classifier
nb_pipeline_ngram = Pipeline([
        ('nb_tfidf',select_feature.tfidf_ngram),
        ('nb_clf',MultinomialNB())])

nb_pipeline_ngram.fit(process_data.train_news['Statement'],process_data.train_news['Label'])
predicted_nb_ngram = nb_pipeline_ngram.predict(process_data.test_news['Statement'])
np.mean(predicted_nb_ngram == process_data.test_news['Label'])


#logistic regression classifier
logR_pipeline_ngram = Pipeline([
        ('LogR_tfidf',select_feature.tfidf_ngram),
        ('LogR_clf',LogisticRegression(penalty="l2",C=1))
        ])

logR_pipeline_ngram.fit(process_data.train_news['Statement'],process_data.train_news['Label'])
predicted_LogR_ngram = logR_pipeline_ngram.predict(process_data.test_news['Statement'])
np.mean(predicted_LogR_ngram == process_data.test_news['Label'])


#linear SVM classifier
svm_pipeline_ngram = Pipeline([
        ('svm_tfidf',select_feature.tfidf_ngram),
        ('svm_clf',svm.LinearSVC())
        ])

svm_pipeline_ngram.fit(process_data.train_news['Statement'],process_data.train_news['Label'])
predicted_svm_ngram = svm_pipeline_ngram.predict(process_data.test_news['Statement'])
np.mean(predicted_svm_ngram == process_data.test_news['Label'])


#sgd classifier
sgd_pipeline_ngram = Pipeline([
         ('sgd_tfidf',select_feature.tfidf_ngram),
         ('sgd_clf',SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3))
         ])

sgd_pipeline_ngram.fit(process_data.train_news['Statement'],process_data.train_news['Label'])
predicted_sgd_ngram = sgd_pipeline_ngram.predict(process_data.test_news['Statement'])
np.mean(predicted_sgd_ngram == process_data.test_news['Label'])


#random forest classifier
random_forest_ngram = Pipeline([
        ('rf_tfidf',select_feature.tfidf_ngram),
        ('rf_clf',RandomForestClassifier(n_estimators=300,n_jobs=3))
        ])
    
random_forest_ngram.fit(process_data.train_news['Statement'],process_data.train_news['Label'])
predicted_rf_ngram = random_forest_ngram.predict(process_data.test_news['Statement'])
np.mean(predicted_rf_ngram == process_data.test_news['Label'])


print('---------------------TFIDF and N-Grams---------------------')

create_confusion_matrix(nb_pipeline_ngram)
create_confusion_matrix(logR_pipeline_ngram)
create_confusion_matrix(svm_pipeline_ngram)
create_confusion_matrix(sgd_pipeline_ngram)
create_confusion_matrix(random_forest_ngram)


print(classification_report(process_data.test_news['Label'], predicted_nb_ngram))
print(classification_report(process_data.test_news['Label'], predicted_LogR_ngram))
print(classification_report(process_data.test_news['Label'], predicted_svm_ngram))
print(classification_report(process_data.test_news['Label'], predicted_sgd_ngram))
print(classification_report(process_data.test_news['Label'], predicted_rf_ngram))

process_data.test_news['Label'].shape

#grid-search parameter optimization
#random forest classifier parameters
parameters = {'rf_tfidf__ngram_range': [(1, 1), (1, 2),(1,3),(1,4),(1,5)],
               'rf_tfidf__use_idf': (True, False),
               'rf_clf__max_depth': (1,2,3,4,5,6,7,8,9,10,11,12,13,14,15)
}

gs_clf = GridSearchCV(random_forest_ngram, parameters, n_jobs=-1)
gs_clf = gs_clf.fit(process_data.train_news['Statement'][:10000],process_data.train_news['Label'][:10000])

gs_clf.best_score_
gs_clf.best_params_
gs_clf.cv_results_

#logistic regression parameters
parameters = {'LogR_tfidf__ngram_range': [(1, 1), (1, 2),(1,3),(1,4),(1,5)],
               'LogR_tfidf__use_idf': (True, False),
               'LogR_tfidf__smooth_idf': (True, False)
}

gs_clf = GridSearchCV(logR_pipeline_ngram, parameters, n_jobs=-1)
gs_clf = gs_clf.fit(process_data.train_news['Statement'][:10000],process_data.train_news['Label'][:10000])

gs_clf.best_score_
gs_clf.best_params_
gs_clf.cv_results_

#Linear SVM 
parameters = {'svm_tfidf__ngram_range': [(1, 1), (1, 2),(1,3),(1,4),(1,5)],
               'svm_tfidf__use_idf': (True, False),
               'svm_tfidf__smooth_idf': (True, False),
               'svm_clf__penalty': ('l1','l2'),
}

gs_clf = GridSearchCV(svm_pipeline_ngram, parameters, n_jobs=-1)
gs_clf = gs_clf.fit(process_data.train_news['Statement'][:10000],process_data.train_news['Label'][:10000])

#finding the model with best performing parameters
gs_clf.best_score_
gs_clf.best_params_
gs_clf.cv_results_

#running both random forest and logistic regression models again with best parameter found with GridSearch method
random_forest_final = Pipeline([
        ('rf_tfidf',TfidfVectorizer(stop_words='english',ngram_range=(1,3),use_idf=True,smooth_idf=True)),
        ('rf_clf',RandomForestClassifier(n_estimators=300,n_jobs=3,max_depth=10))
        ])
    
random_forest_final.fit(process_data.train_news['Statement'],process_data.train_news['Label'])
predicted_rf_final = random_forest_final.predict(process_data.test_news['Statement'])
np.mean(predicted_rf_final == process_data.test_news['Label'])
print(classification_report(process_data.test_news['Label'], predicted_rf_final))

logR_pipeline_final = Pipeline([
        ('LogR_tfidf',TfidfVectorizer(stop_words='english',ngram_range=(1,5),use_idf=True,smooth_idf=False)),
        ('LogR_clf',LogisticRegression(penalty="l2",C=1))
        ])

logR_pipeline_final.fit(process_data.train_news['Statement'],process_data.train_news['Label'])
predicted_LogR_final = logR_pipeline_final.predict(process_data.test_news['Statement'])
np.mean(predicted_LogR_final == process_data.test_news['Label'])
print(classification_report(process_data.test_news['Label'], predicted_LogR_final))


#saving best model to the disk
model_file = 'final_model.sav'
pickle.dump(logR_pipeline_ngram,open(model_file,'wb'))

def find_most_informative_features(model, vect, clf, text=None, n=50):
    # Extract the vectorizer and the classifier from the pipeline
    vectorizer = model.named_steps[vect]
    classifier = model.named_steps[clf]

     # Check to make sure that we can perform this computation
    if not hasattr(classifier, 'coef_'):
        raise TypeError(
            "Cannot compute most informative features on {}.".format(
                classifier.__class__.__name__
            )
        )
            
    if text is not None:
        # Compute the coefficients for the text
        tvec = model.transform([text]).toarray()
    else:
        # Otherwise simply use the coefficients
        tvec = classifier.coef_

    # Zip the feature names with the coefs and sort
    coefs = sorted(
        zip(tvec[0], vectorizer.get_feature_names()),
        reverse=True
    )
    
    # Get the top n and bottom n coef, name pairs
    topn  = zip(coefs[:n], coefs[:-(n+1):-1])

    # Create the output string to return
    output = []

    # If text, add the predicted value to the output.
    if text is not None:
        output.append("\"{}\"".format(text))
        output.append(
            "Classified as: {}".format(model.predict([text]))
        )
        output.append("")

    # Create two columns with most negative and most positive features.
    for (cp, fnp), (cn, fnn) in topn:
        output.append(
            "{:0.4f}{: >15}    {:0.4f}{: >15}".format(
                cp, fnp, cn, fnn
            )
        )
    print(output)

find_most_informative_features(logR_pipeline_ngram,vect='LogR_tfidf',clf='LogR_clf')
find_most_informative_features(nb_pipeline_ngram,vect='nb_tfidf',clf='nb_clf')
find_most_informative_features(svm_pipeline_ngram,vect='svm_tfidf',clf='svm_clf')
find_most_informative_features(sgd_pipeline_ngram,vect='sgd_tfidf',clf='sgd_clf')